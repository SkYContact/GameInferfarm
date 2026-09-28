// probe_partial_d2h.cpp — gd 复合图部分行 D2H 后端级探针（框架回执工件，不入仓）。
// 两次进程跑（FARM_D2H_PARTIAL=0/1 env 决定档位），各 n_rows 档：
//   ① 固定输入跑 1 批 → 输出前 n_rows 行 FNV 哈希（跨档比对=部分拷正确性门）
//   ② 同输入连跑 300 批 → submit→完成 平均 µs（跨档比对=D2H 税测量）
// 输入填充镜像 trt_backend.cpp FillPattern（i32=0 防越界 Gather）。
#include "inferfarm/backend.h"
#include "inferfarm/backend_factory.h"
#include "inferfarm/types.h"
#include "inferfarm/types.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace inferfarm;

static unsigned long long Fnv1a(const void* p, size_t n, unsigned long long h) {
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

int main(int argc, char** argv) {
    const char* eng = argc > 1 ? argv[1] : "models/gd_pfd_comp.fb64.trt";
    const int slots = argc > 2 ? atoi(argv[2]) : 64;
    InferBackend* be = CreateTrtBackend();
    ModelConfig cfg;
    cfg.backend = "trt";
    cfg.engine_path = eng;
    ModelSpec spec;
    if (!be->LoadSpec(cfg, slots, spec)) { std::printf("LoadSpec FAIL\n"); return 1; }
    std::printf("[probe] engine=%s slots=%d ins=%zu outs=%zu out_full_MB=%.2f\n",
                eng, spec.slots, spec.ins.size(), spec.outs.size(),
                [&] { double b = 0; for (auto& o : spec.outs)
                          b += (double)o.width * 4.0 * spec.slots; return b / 1048576.0; }());
    void* s = be->CreateSession(cfg, spec, true);
    if (!s) { std::printf("CreateSession FAIL\n"); return 1; }
    if (!be->Warmup(s)) { std::printf("Warmup FAIL\n"); return 1; }

    // 确定性输入：FillPattern 镜像
    for (auto& im : spec.ins) {
        size_t rb = 0;
        for (int r = 0; r < slots; r++) {
            unsigned char* row = (unsigned char*)be->InputRow(s, im.name.c_str(), r, &rb);
            if (im.et == DTYPE_F32) {
                float* p = (float*)row; size_t n = rb / 4;
                for (size_t e = 0; e < n; e++)
                    p[e] = (float)((int)((e * 31 + (size_t)r * 997) % 2039) - 1019) / 1019.0f;
            } else if (im.et == DTYPE_I64) {
                long long* p = (long long*)row; size_t n = rb / 8;
                for (size_t e = 0; e < n; e++) p[e] = (long long)((e * 7 + (size_t)r * 13) % 14969);
            } else if (im.et == DTYPE_I32) {
                int* p = (int*)row; size_t n = rb / 4;
                for (size_t e = 0; e < n; e++) p[e] = 0;
            } else {
                for (size_t e = 0; e < rb; e++) row[e] = (unsigned char)((e + (size_t)r) & 1);
            }
        }
    }

    const int kRows[] = {8, 16, 32, 64};
    for (int n_rows : kRows) {
        // 哈希批（一次）
        unsigned long long h = 1469598103934665603ULL;
        unsigned seq = 0;
        if (!be->SubmitBatch(s, n_rows, seq)) { std::printf("submit FAIL\n"); return 1; }
        while (!be->CompletionReached(s, seq)) {}
        be->CompletionFence();
        for (auto& om : spec.outs) {
            int w = be->OutputWidth(s, om.name.c_str());
            for (int r = 0; r < n_rows; r++)
                h = Fnv1a(be->OutputRow(s, om.name.c_str(), r), (size_t)w * 4, h);
        }
        // 计时批（300 次同输入）
        const int K = 300;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < K; i++) {
            if (!be->SubmitBatch(s, n_rows, seq)) return 1;
            while (!be->CompletionReached(s, seq)) {}
            be->CompletionFence();
        }
        double us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0).count() / K;
        std::printf("[probe] n=%2d hash=%016llx cycle=%.1fus (%.1fk dec/s)\n",
                    n_rows, h, us, 1000.0 / us * n_rows);
    }
    be->DestroySession(s);
    return 0;
}
