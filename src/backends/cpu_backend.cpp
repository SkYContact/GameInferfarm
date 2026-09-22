// cpu_backend.cpp — 确定性 CPU 后端（玩具/稠密模型）。
//
// 用途：①无 GPU 环境验证全链（fiber/银行/census/驱动环）；②确定性门
// （bank vs inline 逐位一致的参照系）；③refit 语义试验田（RW1 直接改权重）。
//
// 模型语义（行独立·逐位确定）：
//   输出 out[j] 行宽 w_j，对每个输入 i 取行首 min(行元素数, K) 个元素：
//     out[j][k] = tanhf( Σ_i dot(x_i[0..L), W[j][k][i]) )
//   W 由 SplitMix32(weight_seed, j, k, i) 生成（同种子=同权重）。纯 f32
//   定序运算 ⇒ 同输入字节=同输出字节，与批组成无关（行独立）。
//   i64/i32 输入按元素求和后以 f32 参与（toy 语义，够用）。
//
// refit 协议（toy）：RW1 条目名 "<out>.W<in>"（f32，numel=权重长）——
// 同 blob 两次=逐位同；不同 blob=结果必变（门 A1/A2 的 CPU 版）。
#include "inferfarm/backend.h"
#include "inferfarm/refit.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace inferfarm {

namespace {

inline uint32_t SplitMix32(uint32_t x) {
    x += 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x21f0aaadu;
    x = (x ^ (x >> 15)) * 0x735a2d97u;
    return x ^ (x >> 15);
}
inline float WNext(uint32_t& s) {
    s = SplitMix32(s);
    return ((float)(s >> 8) / (float)0x00FFFFFF - 0.5f) * 2.0f;
}

struct CpuIn {
    InputMeta meta;
    void* host = nullptr;          // carve 起点（arena 内）
    size_t off = 0;                // arena 内偏移
};
struct CpuOut {
    OutputMeta meta;
    float* host = nullptr;
    size_t off = 0;
};

struct CpuSession {
    int slots = 64;
    std::vector<CpuIn> ins;
    std::vector<CpuOut> outs;
    std::vector<unsigned char> in_arena;
    std::vector<float> out_arena;
    // 权重：out j 行 k 对 in i 的向量（长 = min(in 元素数, K)）
    static const int kK = 8;       // 每输入取行首元素数上限
    std::vector<std::vector<std::vector<std::vector<float>>>> w;   // [j][k][i]
    unsigned seq = 0;
    int last_n = 0;
    bool computed = false;
};

class CpuBackend : public InferBackend {
public:
    const char* Name() const override { return "cpu"; }

    bool LoadSpec(const ModelConfig& cfg, int slots, ModelSpec& out) override {
        const CpuModelDecl& d = cfg.cpu;
        out.backend = "cpu";
        out.slots = d.slots > 0 ? d.slots : slots;
        out.ins.clear();
        for (const auto& i : d.ins) {
            InputMeta m;
            m.name = i.name;
            m.et = i.et;
            m.esize = DtypeSize(i.et);
            m.dims.push_back(out.slots);
            size_t row = 1;
            for (int64_t dd : i.row_dims) { m.dims.push_back(dd); row *= (size_t)dd; }
            m.row_bytes = row * m.esize;
            if (!m.esize || row == 0) {
                std::fprintf(stderr, "[cpu] 输入 %s 行形状非法\n", i.name.c_str());
                return false;
            }
            out.ins.push_back(std::move(m));
        }
        for (const auto& o : d.outs) {
            OutputMeta m;
            m.name = o.name;
            m.dims = {(int64_t)out.slots, (int64_t)o.width};
            m.width = o.width;
            if (o.width < 1) return false;
            out.outs.push_back(std::move(m));
        }
        if (out.ins.empty() || out.outs.empty()) {
            std::fprintf(stderr, "[cpu] 模型须至少 1 输入 1 输出\n");
            return false;
        }
        if (out.slots != slots) {
            std::fprintf(stderr, "[cpu] 声明 slots=%d ≠ 配置 slots=%d\n", out.slots, slots);
            return false;
        }
        return true;
    }

    void* CreateSession(const ModelConfig& cfg, const ModelSpec& spec, bool for_bank) override {
        (void)for_bank;   // CPU 无图会话/线程绑定问题
        CpuSession* s = new CpuSession();
        s->slots = spec.slots;
        // 输入 arena（256B 对齐 carve——与 GPU 路径同布局纪律）
        const size_t kAlign = 256;
        size_t off = 0;
        s->ins.resize(spec.ins.size());
        for (size_t i = 0; i < spec.ins.size(); i++) {
            s->ins[i].meta = spec.ins[i];
            size_t bytes = spec.ins[i].row_bytes * (size_t)spec.slots;
            off = (off + kAlign - 1) / kAlign * kAlign;
            s->ins[i].off = off;
            off += bytes;
        }
        s->in_arena.assign(off + kAlign, 0);
        for (auto& i : s->ins)
            i.host = s->in_arena.data() + i.off;
        // 输出 arena
        off = 0;
        s->outs.resize(spec.outs.size());
        for (size_t j = 0; j < spec.outs.size(); j++) {
            s->outs[j].meta = spec.outs[j];
            size_t bytes = (size_t)spec.outs[j].width * 4 * (size_t)spec.slots;
            off = (off + kAlign - 1) / kAlign * kAlign;
            s->outs[j].off = off / 4;
            off += bytes;
        }
        s->out_arena.assign(off / 4 + 64, 0.0f);
        for (auto& o : s->outs)
            o.host = s->out_arena.data() + o.off;
        BuildWeights(s, cfg.cpu.weight_seed);
        Track(s);
        return s;
    }

    bool Warmup(void* session) override {
        CpuSession* s = (CpuSession*)session;
        std::memset(s->in_arena.data(), 0, s->in_arena.size());
        unsigned seq = 0;
        for (int r = 0; r < 3; r++)
            if (!SubmitBatch(session, s->slots, seq)) return false;
        return true;
    }

    bool ProbeGraph(void* session) override {
        // 图地址烧死小实验的 CPU 版：两图案可分辨+逐位复算一致。
        CpuSession* s = (CpuSession*)session;
        // 图案 1：输入填 0.5f（首 f32 输入），跑，存输出
        bool has_f32 = false;
        for (auto& i : s->ins)
            if (i.meta.et == DTYPE_F32) {
                size_t elems = i.meta.row_bytes / 4 * (size_t)s->slots;
                float* p = (float*)i.host;
                for (size_t e = 0; e < elems; e++) p[e] = 0.5f;
                has_f32 = true;
                break;
            }
        if (!has_f32) return true;   // 无 f32 输入（纯整型行）：跳过分辨实验
        unsigned seq = 0;
        if (!SubmitBatch(session, s->slots, seq)) return false;
        std::vector<float> ref1(s->out_arena.begin(), s->out_arena.end());
        if (!SubmitBatch(session, s->slots, seq)) return false;
        std::vector<float> ref2(s->out_arena.begin(), s->out_arena.end());
        bool stable = ref1 == ref2;
        // 图案 2：改输入 → 输出必须跟着变（非烧死快照）
        for (auto& i : s->ins)
            if (i.meta.et == DTYPE_F32) {
                size_t elems = i.meta.row_bytes / 4 * (size_t)s->slots;
                float* p = (float*)i.host;
                for (size_t e = 0; e < elems; e++) p[e] = -0.25f;
                break;
            }
        if (!SubmitBatch(session, s->slots, seq)) return false;
        bool diff = s->out_arena != ref1;
        std::printf("[cpu-probe] 可分辨=%d 复算稳定=%d%s\n", (int)diff, (int)stable,
                    diff && stable ? "" : " ←FAIL");
        std::fflush(stdout);
        return diff && stable;
    }

    void DestroySession(void* session) override {
        CpuSession* s = (CpuSession*)session;
        Untrack(s);
        delete s;
    }

    void* InputRow(void* session, const char* name, int slot, size_t* row_bytes) override {
        CpuSession* s = (CpuSession*)session;
        for (auto& i : s->ins)
            if (i.meta.name == name) {
                if (row_bytes) *row_bytes = i.meta.row_bytes;
                return (char*)i.host + (size_t)slot * i.meta.row_bytes;
            }
        return nullptr;
    }

    bool SubmitBatch(void* session, int n_rows, unsigned& seq_out) override {
        CpuSession* s = (CpuSession*)session;
        if (n_rows < 1) n_rows = 1;
        if (n_rows > s->slots) n_rows = s->slots;
        s->seq++;
        s->last_n = n_rows;
        s->computed = true;
        // 行独立计算 [0,n)（尾行保持旧值——与 GPU 银行同语义）
        for (int j = 0; j < (int)s->outs.size(); j++) {
            CpuOut& o = s->outs[(size_t)j];
            for (int r = 0; r < n_rows; r++) {
                for (int k = 0; k < o.meta.width; k++) {
                    float acc = 0.0f;
                    for (size_t i = 0; i < s->ins.size(); i++) {
                        CpuIn& ci = s->ins[i];
                        const std::vector<float>& wv = s->w[(size_t)j][(size_t)k][i];
                        const unsigned char* row =
                            (const unsigned char*)ci.host + (size_t)r * ci.meta.row_bytes;
                        if (ci.meta.et == DTYPE_F32) {
                            const float* x = (const float*)row;
                            float dot = 0.0f;
                            for (size_t e = 0; e < wv.size(); e++) dot += x[e] * wv[e];
                            acc += dot;
                        } else if (ci.meta.et == DTYPE_I64) {
                            const int64_t* x = (const int64_t*)row;
                            size_t elems = ci.meta.row_bytes / 8;
                            int64_t sum = 0;
                            for (size_t e = 0; e < elems; e++) sum += x[e];
                            acc += (float)sum * wv[0];
                        } else if (ci.meta.et == DTYPE_I32) {
                            const int32_t* x = (const int32_t*)row;
                            size_t elems = ci.meta.row_bytes / 4;
                            int64_t sum = 0;
                            for (size_t e = 0; e < elems; e++) sum += x[e];
                            acc += (float)sum * wv[0];
                        } else {
                            const unsigned char* x = row;
                            size_t elems = ci.meta.row_bytes;
                            int64_t sum = 0;
                            for (size_t e = 0; e < elems; e++) sum += x[e];
                            acc += (float)sum * wv[0];
                        }
                    }
                    o.host[(size_t)r * (size_t)o.meta.width + (size_t)k] = std::tanh(acc);
                }
            }
        }
        seq_out = s->seq;
        return true;
    }

    bool CompletionReached(void* session, unsigned seq) override {
        CpuSession* s = (CpuSession*)session;
        return s->computed && s->seq >= seq;
    }
    void CompletionFence() override {}

    const float* OutputRow(void* session, const char* name, int slot) override {
        CpuSession* s = (CpuSession*)session;
        for (auto& o : s->outs)
            if (o.meta.name == name)
                return o.host + (size_t)slot * (size_t)o.meta.width;
        return nullptr;
    }
    int OutputWidth(void* session, const char* name) override {
        CpuSession* s = (CpuSession*)session;
        for (auto& o : s->outs)
            if (o.meta.name == name) return o.meta.width;
        return -1;
    }

    bool RefitWeights(const char* rw1_path) override {
        std::vector<char> blob;
        std::vector<Rw1Entry> ents;
        if (!ParseRw1(rw1_path, blob, ents)) return false;
        // 会话级权重更新（toy 协议 "<out>.W<in>"）——对所有存活会话生效：
        // 遍历本后端已建会话（登记表）
        std::lock_guard<std::mutex> lk(mx_);
        int n_set = 0, n_skip = 0;
        for (CpuSession* s : sessions_) {
            for (const Rw1Entry& e : ents) {
                // 名字解析：<out>.W<in>[.<k>]——.k 定向单输出行；缺省=全部行
                const std::string nm = e.name;
                size_t dot = nm.find(".W");
                if (dot == std::string::npos || e.dtype != 2) { n_skip++; continue; }
                std::string on = nm.substr(0, dot);
                std::string rest = nm.substr(dot + 2);
                int only_k = -1;
                std::string in = rest;
                size_t d2 = rest.rfind('.');
                if (d2 != std::string::npos) {
                    in = rest.substr(0, d2);
                    only_k = atoi(rest.c_str() + d2 + 1);
                }
                int j = -1, ii = -1;
                for (size_t x = 0; x < s->outs.size(); x++)
                    if (s->outs[x].meta.name == on) { j = (int)x; break; }
                for (size_t x = 0; x < s->ins.size(); x++)
                    if (s->ins[x].meta.name == in) { ii = (int)x; break; }
                if (j < 0 || ii < 0) { n_skip++; continue; }
                int width = s->outs[(size_t)j].meta.width;
                if (only_k >= width) { n_skip++; continue; }
                bool any = false;
                for (int k = 0; k < width; k++) {
                    if (only_k >= 0 && k != only_k) continue;
                    std::vector<float>& wv = s->w[(size_t)j][(size_t)k][(size_t)ii];
                    if (e.numel == wv.size()) {
                        memcpy(wv.data(), e.data, e.bytes);
                        any = true;
                    }
                }
                if (any) n_set++;
                else n_skip++;
            }
        }
        std::fprintf(stderr, "[cpu-refit] 换心 %d 项（跳过 %d）: %s\n", n_set, n_skip, rw1_path);
        return true;
    }

private:
    void BuildWeights(CpuSession* s, uint32_t seed) {
        s->w.assign(s->outs.size(), {});
        for (size_t j = 0; j < s->outs.size(); j++) {
            s->w[j].assign((size_t)s->outs[j].meta.width,
                           std::vector<std::vector<float>>(s->ins.size()));
            for (int k = 0; k < s->outs[j].meta.width; k++)
                for (size_t i = 0; i < s->ins.size(); i++) {
                    CpuIn& ci = s->ins[i];
                    size_t L;
                    if (ci.meta.et == DTYPE_F32)
                        L = ci.meta.row_bytes / 4 < (size_t)CpuSession::kK
                                ? ci.meta.row_bytes / 4 : (size_t)CpuSession::kK;
                    else
                        L = 1;
                    std::vector<float>& wv = s->w[j][(size_t)k][i];
                    wv.resize(L);
                    uint32_t st = seed ^ (uint32_t)(j * 7919 + (size_t)k * 104729 + i * 1299709);
                    for (size_t e = 0; e < L; e++) wv[e] = WNext(st);
                }
        }
    }
    std::mutex mx_;
    std::vector<CpuSession*> sessions_;   // refit 广播面

public:
    void Track(CpuSession* s) { std::lock_guard<std::mutex> lk(mx_); sessions_.push_back(s); }
    void Untrack(CpuSession* s) {
        std::lock_guard<std::mutex> lk(mx_);
        for (size_t i = 0; i < sessions_.size(); i++)
            if (sessions_[i] == s) { sessions_.erase(sessions_.begin() + (long)i); break; }
    }
};

} // namespace

InferBackend* CreateCpuBackend() { return new CpuBackend(); }

} // namespace inferfarm
