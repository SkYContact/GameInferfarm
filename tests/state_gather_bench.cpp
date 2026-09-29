// ============================================================
//  state_gather_bench.cpp — 状态面行拷内核微基准（DATA13 判决实验）
//
//  面向真 GPU 手动跑（GPU 协议：~/gpu_lock.sh acquire 后再执行，计时
//  数字只在锁内有效）。三件事：
//    1. 正确性：内核路径 vs 逐行 cudaMemcpyAsync 参考路径，D2H 回读
//       逐位比对（gather/scatter 两 mode；跳过行=-1 目标先铺垃圾，
//       比对即验证"留陈旧内容"语义）；rb 取 28672（16 倍数）和
//       1000（非 16 倍数=对齐兜底路径）两尺寸。
//    2. 计时：1536 行/批，逐行 MemcpyAsync vs 单次内核 launch，
//       cudaEvent 计时各 200 次取均值。
//    3. 打印一行结论。
// ============================================================
#include "backends/state_gather.h"
#include "backends/cudart_dyn.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <random>
#include <vector>

using namespace inferfarm;

namespace {

Cudart g_cu;

#define CK(expr)                                                          \
    do {                                                                  \
        /* Cudart 符号统一按 void* 返回（镜像 cudaError_t 指针面） */       \
        long _rc = (long)(intptr_t)(expr);                                \
        if (_rc != 0) {                                                   \
            std::fprintf(stderr, "[bench] CUDA 错误 %ld @ %s\n", _rc, #expr); \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

struct Bufs {
    void* pool = nullptr;     // 1024 行池（+尾保留零行已在表内用 1023 行号表达）
    void* pool_ref = nullptr; // scatter 参考路径用第二池
    void* rows = nullptr;     // gather 目标 / scatter 源（1536 行）
    void* rows_ref = nullptr;
    void* idx = nullptr;      // 设备端 int32 表
    size_t rb = 0;
    int n = 0;
};

// 建随机 idx 表：约 12% 幻影(-1)、约 8% 池末零行(1023)、其余 [0,1023)。
// 计时用（scatter 语义下重复目标=调用方非法，计时只看搬运量不受影响）。
void MakeIdx(std::vector<int>& v, std::mt19937& rng) {
    v.resize(1536);
    std::uniform_int_distribution<int> d(0, 1023), pct(0, 99);
    for (auto& x : v) {
        int p = pct(rng);
        if (p < 12) x = -1;
        else if (p < 20) x = 1023;      // 池末保留零行哨兵
        else x = d(rng);
    }
}

// 建去重 idx 表（正确性比对用）：非 -1 行号全表唯一（scatter 语义=每池行
// 至多写一次；随机表重复目标时逐行参考是 r 序 last-writer-wins，与内核
// 无序并行不可复现地比对——非法输入不进正确性面）。
void MakeIdxUnique(std::vector<int>& v, std::mt19937& rng) {
    v.assign(1536, -1);
    std::vector<int> pos(1536), ids(1024);
    for (int i = 0; i < 1536; ++i) pos[i] = i;
    for (int i = 0; i < 1024; ++i) ids[i] = i;
    std::shuffle(pos.begin(), pos.end(), rng);
    std::shuffle(ids.begin(), ids.end(), rng);
    for (int k = 0; k < 900; ++k) v[pos[k]] = ids[k];   // 900 活跃行，含 1023 零行哨兵
}
// 参考路径：与旧逐行实现逐位同（跳过行 continue→目标留原内容）
void RefPath(const Bufs& b, const std::vector<int>& idx, int mode) {
    for (int r = 0; r < b.n; ++r) {
        int i = idx[r];
        if (i < 0) continue;
        if (mode == 0)
            CK(g_cu.MemcpyAsync((char*)b.rows_ref + (size_t)r * b.rb,
                                (char*)b.pool + (size_t)i * b.rb, b.rb,
                                2 /*D2D*/, nullptr));
        else
            CK(g_cu.MemcpyAsync((char*)b.pool_ref + (size_t)i * b.rb,
                                (char*)b.rows_ref + (size_t)r * b.rb, b.rb,
                                2 /*D2D*/, nullptr));
    }
    CK(g_cu.DeviceSynchronize());
}

// 内核路径（单次 launch）
bool KernelPath(const Bufs& b, void* idx_dev, int mode, void* stream) {
    if (mode == 0)
        return StateGatherLaunch(b.rows, b.pool, (const int*)idx_dev, b.n,
                                 b.rb, stream, 0);
    return StateGatherLaunch(b.rows, b.pool, (const int*)idx_dev, b.n,
                             b.rb, stream, 1);
}

// 正确性：目标区先铺垃圾（0xAB）→ 两路径各跑一份 → D2H 逐位比对
bool Check(Bufs& b, void* idx_dev, const std::vector<int>& idx, int mode,
           size_t bytes_rows, size_t bytes_pool) {
    // 目标区铺垃圾（验证跳过行留陈旧内容），源区铺随机数据（H2D 保证
    // 两路径源一致）：gather 目标=rows、源=pool；scatter 反之
    std::mt19937 rng(1234);
    size_t bytes_src = (mode == 0) ? bytes_pool : bytes_rows;
    std::vector<char> src(bytes_src);
    for (auto& c : src) c = (char)(rng() & 0xFF);
    if (mode == 0) {
        CK(g_cu.Memset(b.rows, 0xAB, bytes_rows));
        CK(g_cu.Memset(b.rows_ref, 0xAB, bytes_rows));
        CK(g_cu.Memcpy(b.pool, src.data(), bytes_pool, 1));
        CK(g_cu.Memcpy(b.pool_ref, src.data(), bytes_pool, 1));
    } else {
        CK(g_cu.Memset(b.pool, 0xAB, bytes_pool));
        CK(g_cu.Memset(b.pool_ref, 0xAB, bytes_pool));
        CK(g_cu.Memcpy(b.rows, src.data(), bytes_rows, 1));
        CK(g_cu.Memcpy(b.rows_ref, src.data(), bytes_rows, 1));
    }
    CK(g_cu.DeviceSynchronize());

    if (!KernelPath(b, idx_dev, mode, nullptr)) return false;
    RefPath(b, idx, mode);
    CK(g_cu.DeviceSynchronize());

    std::vector<char> got, want;
    if (mode == 0) {
        got.resize(bytes_rows); want.resize(bytes_rows);
        CK(g_cu.Memcpy(got.data(), b.rows, bytes_rows, 2));
        CK(g_cu.Memcpy(want.data(), b.rows_ref, bytes_rows, 2));
    } else {
        got.resize(bytes_pool); want.resize(bytes_pool);
        CK(g_cu.Memcpy(got.data(), b.pool, bytes_pool, 2));
        CK(g_cu.Memcpy(want.data(), b.pool_ref, bytes_pool, 2));
    }
    if (std::memcmp(got.data(), want.data(), got.size()) != 0) {
        for (size_t k = 0; k < got.size(); ++k)
            if (got[k] != want[k]) {
                std::fprintf(stderr, "[bench] mode=%d rb=%zu 首个错位 @%zu "
                             "(%02x vs %02x)\n", mode, b.rb, k,
                             (unsigned char)got[k], (unsigned char)want[k]);
                break;
            }
        return false;
    }
    return true;
}

// 计时：200 批取均值（主机时钟 + StreamSynchronize 差分；Cudart 通道
// 无 EventElapsedTime，且逐行路径本身要等流——含 launch+同步的全路径
// 代价，正是批内真实开销）
double TimePath(const Bufs& b, void* idx_dev, const std::vector<int>& idx,
                int use_kernel, int iters, void* stream) {
    double us_sum = 0.0;
    using clk = std::chrono::steady_clock;
    for (int it = 0; it < iters; ++it) {
        auto t0 = clk::now();
        if (use_kernel) {
            if (!StateGatherLaunch(b.rows, b.pool, (const int*)idx_dev, b.n,
                                   b.rb, stream, 0))
                std::exit(1);
        } else {
            for (int r = 0; r < b.n; ++r) {
                int i = idx[r];
                if (i < 0) continue;
                CK(g_cu.MemcpyAsync((char*)b.rows + (size_t)r * b.rb,
                                    (char*)b.pool + (size_t)i * b.rb, b.rb,
                                    2 /*D2D*/, stream));
            }
        }
        CK(g_cu.StreamSynchronize(stream));
        auto t1 = clk::now();
        us_sum += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    return us_sum / iters;   // µs/批
}

} // namespace

int main() {
    // cudart 动态装载（与仓内后端同通道）
    if (!g_cu.Load("")) {
        std::fprintf(stderr, "[bench] cudart 装载失败\n");
        return 1;
    }
    if (g_cu.SetDevice) CK(g_cu.SetDevice(0));
    int ndev = 0;
    if (g_cu.GetDeviceCount) { CK(g_cu.GetDeviceCount(&ndev)); }
    if (ndev <= 0) {
        std::fprintf(stderr, "[bench] 无可用 CUDA 设备\n");
        return 1;
    }
    if (!StateGatherInit()) {
        std::fprintf(stderr, "[bench] StateGatherInit 失败\n");
        return 1;
    }
    if (!StateGatherReady()) {
        std::fprintf(stderr, "[bench] StateGatherReady=false\n");
        return 1;
    }

    void* stream = nullptr;
    CK(g_cu.StreamCreate(&stream, 0));
    std::mt19937 rng(2026);
    const int kIters = 200;
    bool all_ok = true;
    double t_row_28k = 0, t_k_28k = 0, t_row_1k = 0, t_k_1k = 0;

    for (size_t rb : { (size_t)28672, (size_t)1000 }) {
        Bufs b;
        b.rb = rb;
        b.n = 1536;
        size_t bytes_rows = (size_t)b.n * rb;
        size_t bytes_pool = (size_t)1024 * rb;
        CK(g_cu.Malloc(&b.pool, bytes_pool));
        CK(g_cu.Malloc(&b.pool_ref, bytes_pool));
        CK(g_cu.Malloc(&b.rows, bytes_rows));
        CK(g_cu.Malloc(&b.rows_ref, bytes_rows));
        CK(g_cu.Malloc(&b.idx, b.n * sizeof(int)));

        std::vector<int> idx, idxu;
        MakeIdx(idx, rng);
        MakeIdxUnique(idxu, rng);
        CK(g_cu.Memcpy(b.idx, idxu.data(), b.n * sizeof(int), 1));
        CK(g_cu.DeviceSynchronize());

        for (int mode = 0; mode <= 1; ++mode) {
            bool ok = Check(b, b.idx, idxu, mode, bytes_rows, bytes_pool);
            std::printf("[bench] 正确性 rb=%zu mode=%d(%s): %s\n", rb, mode,
                        mode == 0 ? "gather" : "scatter",
                        ok ? "PASS" : "FAIL");
            all_ok = all_ok && ok;
        }

        if (true) {
            double tr = TimePath(b, b.idx, idx, 0, kIters, stream);
            double tk = TimePath(b, b.idx, idx, 1, kIters, stream);
            if (rb == 28672) { t_row_28k = tr; t_k_28k = tk; }
            else             { t_row_1k = tr;  t_k_1k = tk; }
        }
        CK(g_cu.Free(b.pool)); CK(g_cu.Free(b.pool_ref));
        CK(g_cu.Free(b.rows)); CK(g_cu.Free(b.rows_ref));
        CK(g_cu.Free(b.idx));
    }

    if (true) {
        std::printf("[bench] 计时 1536 行/批, %d 批均值:\n", kIters);
        std::printf("[bench]   rb=28672: 逐行=%.1fµs  内核=%.1fµs  加速=%.1fx\n",
                    t_row_28k, t_k_28k, t_row_28k / t_k_28k);
        std::printf("[bench]   rb=1000 : 逐行=%.1fµs  内核=%.1fµs  加速=%.1fx\n",
                    t_row_1k, t_k_1k, t_row_1k / t_k_1k);
        double best = t_k_28k < t_k_1k ? t_k_28k : t_k_1k;
        std::printf("[bench] 结论: %s；内核单批 %.1fµs（逐行 %.1fµs）\n",
                    all_ok ? "正确性逐位 PASS" : "正确性 FAIL(勿用)",
                    best,
                    t_row_28k < t_row_1k ? t_row_28k : t_row_1k);
    }
    if (!all_ok) return 1;
    return 0;
}
