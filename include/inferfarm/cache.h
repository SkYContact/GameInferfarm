// ============================================================
//  inferfarm/cache.h — 推理缓存（KataGo NNCacheTable 思想的吸收，
//  2026-09-22；出处与判决见 docs/design-judgments.md 判决13）
//
//  键 = 全部输入行的**组装字节**128 位哈希 + 权重代次 gen；
//  值 = 逐输出 (名字, n, fp32 字节) 的逐字节重放。
//  正确性前提=行独立契约的同输入行必同输出（逐位）——缓存只是把
//  "后端算过的函数值"提前取出，不引入任何新语义。
//
//  结构判决（KataGo 同款，勿凭直觉改）：
//   - 2^log2_cap 直接索引表：单槽单条目，碰撞即逐出（无链无 LRU——
//     自博弈状态访问近似均匀，简单覆写=最优性价比）；
//   - MutexPool 条锁（256 把）：get/set 锁内只做 shared_ptr 拷贝/交换，
//     重活（make_shared、memcpy）全在锁外；
//   - 出借 shared_ptr<const>：命中条目即使被并发逐出，调用方手中的
//     指针仍活着（收割侧安全读）；
//   - gen 代次：换心（RefitWeights）成功即 ++，旧代条目永不再命中
//     （无需清表——直接索引自会覆写）。
//
//  哈希非密码学：目标是结构化浮点输入下的低碰撞 + 进程内确定性
//  （跨平台/跨进程一致无要求——缓存本就是进程内资产）。
// ============================================================
#pragma once
#include "types.h"
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace inferfarm {

struct CacheKey128 {
    uint64_t a = 0, b = 0;
    bool operator==(const CacheKey128& o) const { return a == o.a && b == o.b; }
};

// 紧凑 128 位哈希（双 64 位车道 + splitmix 终局雪崩）
class CacheHasher {
public:
    void Update(const void* data, size_t n) {
        const uint8_t* p = (const uint8_t*)data;
        len_ += n;
        while (n >= 8) {
            uint64_t w;
            std::memcpy(&w, p, 8);
            Step(w);
            p += 8;
            n -= 8;
        }
        if (n) {
            uint64_t w = 0;
            std::memcpy(&w, p, n);   // 残尾：n < 8，memcpy 不越界读
            Step(w ^ (kTail ^ (uint64_t)n));
        }
    }
    CacheKey128 Finalize() const {
        uint64_t x = Mix(a_ + 0x165667B19E3779F9ull * len_);
        uint64_t y = Mix(b_ ^ 0x9E3779B97F4A7C15ull * (len_ + 0x100000001B3ull));
        CacheKey128 k;
        k.a = x;
        k.b = Mix(y ^ x);
        return k;
    }

private:
    static uint64_t Mix(uint64_t z) {   // splitmix64 终局器
        z += 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    void Step(uint64_t w) {
        a_ = (a_ ^ w) * 0x9E3779B97F4A7C15ull;
        a_ ^= a_ >> 29;
        b_ = (b_ + w) * 0xC2B2AE3D27D4EB4Full;
        b_ = (b_ << 23) | (b_ >> 41);
        b_ ^= b_ >> 25;
    }
    static constexpr uint64_t kTail = 0xAD93B7187E5CF37Bull;
    uint64_t a_ = 0x243F6A8885A308D3ull;   // 圆周率位——任意固定奇种子即可
    uint64_t b_ = 0x13198A2E03707344ull;
    size_t len_ = 0;
};

// 缓存值：逐输出重放面（与收割写入 dests 的内容逐字节等价——n 全宽，
// 含模型宽 <n 时的尾部陈旧字节：按"请求即消费"契约重放，语义不弱于收割路径）
struct CachedOut {
    std::string name;
    int n = 0;
    std::vector<float> vals;
};
struct CachedResult {
    CacheKey128 key;
    uint64_t gen = 0;
    std::vector<CachedOut> outs;
};

class InferCache {
public:
    // log2_cap=0 关（缺省）；>0 表容=2^log2_cap 条（上限 24=1600 万条）
    void Init(int log2_cap) {
        if (log2_cap <= 0) return;
        if (log2_cap > 24) log2_cap = 24;
        mask_ = (size_t)1 << log2_cap;
        mask_ -= 1;
        slots_.assign(mask_ + 1, nullptr);
        on_ = true;
    }
    bool on() const { return on_; }

    // 命中=键+代次全符。返回的 shared_ptr 出借（并发逐出不失效）。
    std::shared_ptr<const CachedResult> Lookup(const CacheKey128& k, uint64_t gen) {
        lookups_.fetch_add(1, std::memory_order_relaxed);
        if (!on_) return nullptr;
        std::shared_ptr<const CachedResult> p;
        {
            std::lock_guard<std::mutex> lk(Pool(Idx(k)));
            p = slots_[Idx(k)];
        }   // 锁内仅 shared_ptr 拷贝（重活在锁外）
        if (!p || !(p->key == k) || p->gen != gen) return nullptr;
        hits_.fetch_add(1, std::memory_order_relaxed);
        return p;
    }
    void Insert(CacheKey128 k, uint64_t gen, std::vector<CachedOut> outs) {
        if (!on_) return;
        auto e = std::make_shared<CachedResult>();   // 锁外构重物
        e->key = k;
        e->gen = gen;
        e->outs = std::move(outs);
        std::lock_guard<std::mutex> lk(Pool(Idx(k)));
        slots_[Idx(k)] = std::move(e);   // 碰撞即逐出（直接索引判决）
    }

    unsigned long long lookups() const { return lookups_.load(std::memory_order_relaxed); }
    unsigned long long hits() const { return hits_.load(std::memory_order_relaxed); }

private:
    size_t Idx(const CacheKey128& k) const { return (size_t)(k.a ^ (k.b * 0x9E3779B97F4A7C15ull)) & mask_; }
    std::mutex& Pool(size_t idx) { return pool_[idx & (kPool - 1)]; }

    static constexpr size_t kPool = 256;   // 条锁池（锁粒度与表解耦）
    std::mutex pool_[kPool];
    std::vector<std::shared_ptr<const CachedResult>> slots_;
    size_t mask_ = 0;
    bool on_ = false;
    std::atomic<unsigned long long> lookups_{0}, hits_{0};
};

} // namespace inferfarm
