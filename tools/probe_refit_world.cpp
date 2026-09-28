// probe_refit_world.cpp — refit vs 重烤 float 世界定谳（掼蛋演化确定性硬门）。
// 用法 A：probe <engine>            → 固定输入跑 → 全输出锚行哈希
// 用法 B：probe <engine> <rw1>      → 跑→RefitWeights→再跑 → 两个哈希+换心耗时
// 定谳法：base+wb.rw1 的"换后哈希" vs bakeB（同权重直烤）的哈希 逐位比对。
#include "inferfarm/backend.h"
#include "inferfarm/backend_factory.h"
#include "inferfarm/types.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace inferfarm;

static unsigned long long Fnv1a(const void* p, size_t n, unsigned long long h) {
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: probe <engine> [rw1]\n"); return 1; }
    InferBackend* be = CreateTrtBackend();
    ModelConfig cfg;
    cfg.backend = "trt";
    cfg.engine_path = argv[1];
    ModelSpec spec;
    if (!be->LoadSpec(cfg, 64, spec)) return 1;
    void* s = be->CreateSession(cfg, spec, true);
    if (!s || !be->Warmup(s)) return 1;

    for (auto& im : spec.ins) {   // 固定输入（镜像 FillPattern）
        for (int r = 0; r < 64; r++) {
            size_t rb = 0;
            unsigned char* row = (unsigned char*)be->InputRow(s, im.name.c_str(), r, &rb);
            if (im.et == DTYPE_F32) {
                float* p = (float*)row;
                for (size_t e = 0; e < rb / 4; e++)
                    p[e] = (float)((int)((e * 31 + (size_t)r * 997) % 2039) - 1019) / 1019.0f;
            } else if (im.et == DTYPE_I64) {
                long long* p = (long long*)row;
                for (size_t e = 0; e < rb / 8; e++) p[e] = (long long)((e * 7 + (size_t)r * 13) % 14969);
            } else if (im.et == DTYPE_I32) {
                int* p = (int*)row; for (size_t e = 0; e < rb / 4; e++) p[e] = 0;
            } else {
                for (size_t e = 0; e < rb; e++) row[e] = (unsigned char)((e + (size_t)r) & 1);
            }
        }
    }
    auto run_hash = [&]() -> unsigned long long {
        unsigned seq = 0;
        if (!be->SubmitBatch(s, 64, seq)) { std::printf("submit FAIL\n"); return 0; }
        for (int i = 0; i < 3000000 && !be->CompletionReached(s, seq); i++) {}
        be->CompletionFence();
        unsigned long long h = 1469598103934665603ULL;
        for (auto& om : spec.outs) {
            int w = be->OutputWidth(s, om.name.c_str());
            for (int r = 0; r < 8; r++)
                h = Fnv1a(be->OutputRow(s, om.name.c_str(), r), (size_t)w * 4, h);
        }
        return h;
    };
    unsigned long long h0 = run_hash();
    std::printf("[refit-probe] 换前 hash=%016llx\n", h0);
    if (argc > 2) {
        auto t0 = std::chrono::steady_clock::now();
        bool ok = be->RefitWeights(argv[2]);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        std::printf("[refit-probe] RefitWeights(%s)=%d 耗时 %.1fms\n", argv[2],
                    (int)ok, ms);
        if (!ok) return 2;
        unsigned long long h1 = run_hash();
        std::printf("[refit-probe] 换后 hash=%016llx\n", h1);
    }
    be->DestroySession(s);
    return 0;
}
