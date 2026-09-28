// probe_batch_invariance.cpp — §1 判决：TRT 引擎跨批形状逐位稳定性探针。
// 判据：同一份固定输入（全 slots 行），不同 n_rows 提交，锚行集（前 8 行）
// 输出哈希跨 n 逐位比对——红=变批改结果（掼蛋 hands 抖动根因坐实）；
// 绿=引擎跨 shape 逐位定，抖动另有根因。另跑同 shape 复跑对照（必须绿）。
// 前置认知：R5 门（gomoku fb8）绿=小图跨 shape 稳；本探针测 gd 复合图 fb64。
#include "inferfarm/backend.h"
#include "inferfarm/backend_factory.h"
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
    const int kAnchor = 8;   // 锚行数（跨 n 比对的前 8 行）
    InferBackend* be = CreateTrtBackend();
    ModelConfig cfg;
    cfg.backend = "trt";
    cfg.engine_path = eng;
    ModelSpec spec;
    if (!be->LoadSpec(cfg, slots, spec)) { std::printf("LoadSpec FAIL\n"); return 1; }
    void* s = be->CreateSession(cfg, spec, true);
    if (!s || !be->Warmup(s)) { std::printf("session FAIL\n"); return 1; }

    for (auto& im : spec.ins) {   // 固定确定性输入（镜像 FillPattern）
        for (int r = 0; r < slots; r++) {
            unsigned char* row = (unsigned char*)be->InputRow(s, im.name.c_str(), r, nullptr);
            size_t rb = 0;
            be->InputRow(s, im.name.c_str(), r, &rb);
            if (im.et == DTYPE_F32) {
                float* p = (float*)row;
                for (size_t e = 0; e < rb / 4; e++)
                    p[e] = (float)((int)((e * 31 + (size_t)r * 997) % 2039) - 1019) / 1019.0f;
            } else if (im.et == DTYPE_I64) {
                long long* p = (long long*)row;
                for (size_t e = 0; e < rb / 8; e++) p[e] = (long long)((e * 7 + (size_t)r * 13) % 14969);
            } else if (im.et == DTYPE_I32) {
                int* p = (int*)row;
                for (size_t e = 0; e < rb / 4; e++) p[e] = 0;
            } else {
                for (size_t e = 0; e < rb; e++) row[e] = (unsigned char)((e + (size_t)r) & 1);
            }
        }
    }

    auto run_anchor_hash = [&](int n_rows, unsigned long long* h_all) -> bool {
        unsigned seq = 0;
        if (!be->SubmitBatch(s, n_rows, seq)) return false;
        for (int spin = 0; spin < 2000000; spin++) {
            if (be->CompletionReached(s, seq)) break;
        }
        if (!be->CompletionReached(s, seq)) return false;
        be->CompletionFence();
        unsigned long long h = 1469598103934665603ULL;
        for (auto& om : spec.outs) {
            int w = be->OutputWidth(s, om.name.c_str());
            for (int r = 0; r < kAnchor; r++)   // 只哈希锚行（跨 n 可比）
                h = Fnv1a(be->OutputRow(s, om.name.c_str(), r), (size_t)w * 4, h);
        }
        *h_all = h;
        return true;
    };

    // 行位置敏感性（正确版）：锚内容 C=原行 0-7。基准跑：C 放槽 0-7；
    // 对照跑：C 放槽 32-39（其余槽拿原行 40+ 内容填充）。哈希各自槽位
    // 的输出比对——同内容不同位置应逐位同（行独立契约实证面）
    {
        std::vector<std::vector<std::vector<unsigned char>>> rows_store;
        for (auto& im : spec.ins) {
            std::vector<std::vector<unsigned char>> rs;
            for (int r = 0; r < slots; r++) {
                size_t rb = 0;
                unsigned char* row = (unsigned char*)be->InputRow(s, im.name.c_str(), r, &rb);
                rs.emplace_back(row, row + rb);
            }
            rows_store.push_back(std::move(rs));
        }
        auto place = [&](int base) {   // C(8 行)放 [base, base+8)，其余槽用原行 (40..) 内容
            for (size_t i = 0; i < spec.ins.size(); i++)
                for (int r = 0; r < slots; r++) {
                    size_t rb = 0;
                    unsigned char* row = (unsigned char*)be->InputRow(s, spec.ins[i].name.c_str(), r, &rb);
                    int src = (r >= base && r < base + 8) ? r - base
                                                         : 40 + (r % (slots - 40 > 0 ? slots - 40 : 1));
                    if (src >= slots) src = r;
                    std::memcpy(row, rows_store[i][(size_t)src].data(), rb);
                }
        };
        auto hash_slots = [&](int base, unsigned long long* out) {
            unsigned seq = 0;
            if (!be->SubmitBatch(s, slots, seq)) { *out = 1; return; }
            for (int spin = 0; spin < 2000000; spin++)
                if (be->CompletionReached(s, seq)) break;
            be->CompletionFence();
            unsigned long long h = 1469598103934665603ULL;
            for (auto& om : spec.outs) {
                int w = be->OutputWidth(s, om.name.c_str());
                for (int r = base; r < base + 8; r++)
                    h = Fnv1a(be->OutputRow(s, om.name.c_str(), r), (size_t)w * 4, h);
            }
            *out = h;
        };
        unsigned long long hA = 0, hB = 0;
        place(0); hash_slots(0, &hA);
        place(32); hash_slots(32, &hB);
        std::printf("[probe] 行位置：C@槽0-7=%016llx C@槽32-39=%016llx %s\n",
                    hA, hB, hA == hB ? "同（行位置无关 ✓）" : "**异（同内容换位变结果）**");
        place(0);   // 恢复基准输入供后续 n 扫描
    }
    const int kNs[] = {1, 2, 4, 8, 16, 32, 64};
    unsigned long long ref = 0;
    bool ok = true;
    std::printf("[probe] 锚行=%d 引擎=%s\n", kAnchor, eng);
    for (int n : kNs) {
        if (n > slots) break;
        unsigned long long h = 0;
        if (!run_anchor_hash(n, &h)) { std::printf("n=%d FAIL\n", n); return 1; }
        if (n == kNs[0]) ref = h;
        bool same = (h == ref);
        if (!same) ok = false;
        std::printf("[probe] n=%2d 锚行哈希=%016llx %s\n", n, h,
                    same ? "同" : "**异**");
    }
    // 同 shape 复跑对照（必须同）
    unsigned long long h2 = 0;
    run_anchor_hash(16, &h2);
    unsigned long long h16 = 0;
    for (int n : kNs) if (n == 16) h16 = h2;   // 已算
    unsigned long long h_rep = 0;
    run_anchor_hash(16, &h_rep);
    std::printf("[probe] n=16 复跑=%016llx %s\n", h_rep,
                h_rep == h16 ? "同（同 shape 确定 ✓）" : "**异（连复跑都不稳！）**");
    std::printf(ok ? "[probe] 判决：引擎跨批形状逐位稳定（变批假设被证伪）\n"
                   : "[probe] 判决：**变批改结果**——hands 抖动根因坐实"
                     "（不同 n 的浮点归约序/ tactic 差异）\n");
    be->DestroySession(s);
    return ok ? 0 : 2;
}
