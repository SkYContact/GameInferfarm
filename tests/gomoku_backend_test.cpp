// gomoku_backend_test.cpp — 真模型可选门（ORT/TRT 后端）。
//
// 门内容（工件存在才跑；否则 SKIP 退 0——CI/无 GPU 机器不拦路）：
//   R1 ort 银行 vs inline 逐位一致（含指纹）+ 复跑逐位同
//   R2 trt 同款（trt 构建才可用；未编 TRT=SKIP）
//   R3 ort vs trt 跨后端逐位一致（fp32+TF32 关的强性质；观察项，不设硬门
//      ——不同引擎逐位等价不总是成立，本仓当前工件实测成立）
//   R4 异构双设备（判决15）：ort/cuda 主卡 + ort/dml 第二卡（如 AMD 核显），
//      腿完成 + 复跑逐位同（链→组钉扎的跨厂商确定性）。环境
//      FARM_DML_DIR=onnxruntime-directml 的 capi 目录（缺席=SKIP）
//   R5 批次/位置不变性门（G13 的真后端版，2026-09-24）：同一行内容在批大小
//      n=1..满 与行位置变化下输出逐位同——GPU 归约策略随 shape 变化的直接
//      检验。ort/trt 各一（工件/后端缺席=SKIP）
//
// 工件烤制：python tools/bake_gomoku_mlp.py --slots 8 --hidden 64 \
//   --out models/gomoku_mlp.fb8.onnx --trt models/gomoku_mlp.fb8.trt
#include "../examples/gomoku/gomoku_adapter.h"
#include "inferfarm/backend.h"
#include "inferfarm/backend_factory.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace inferfarm;
using namespace inferfarm::gomoku;

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", msg); g_fail++; } \
    else std::printf("ok: %s\n", msg); \
    std::fflush(stdout); \
} while (0)

static bool FileExists(const char* p) {
    FILE* f = fopen(p, "rb");
    if (f) { fclose(f); return true; }
    return false;
}

struct R { unsigned long long fp; int games; double sec; };

static R Leg(const char* backend, const char* model, const char* engine, int banks,
              bool count_fail = true) {
    FarmConfig cfg;
    cfg.name = "gomoku-real";
    cfg.chains = 8;
    cfg.games = 32;
    cfg.seed0 = 20260922u;
    cfg.banks = banks;
    cfg.slots = 8;
    cfg.workers = 4;
    cfg.stagger_ms = 1;
    cfg.model.backend = backend;
    cfg.model.model_path = model ? model : "";
    cfg.model.engine_path = engine ? engine : "";
    Farm farm;
    if (!farm.Init(cfg)) {
        if (count_fail) g_fail++;
        return {0, 0, 0};
    }
    double sec = farm.RunLeg(MakeGomokuAdapter, nullptr);
    return {farm.tally().fingerprint, farm.tally().games_done, sec};
}

int main() {
    const char* kOnnx = "models/gomoku_mlp.fb8.onnx";
    const char* kTrt = "models/gomoku_mlp.fb8.trt";
    std::printf("=== 真模型可选门（工件缺席=SKIP）===\n");
    bool have_ort = FileExists(kOnnx);
    bool have_trt = FileExists(kTrt);
    R ort{}, ort2{}, orti{}, trt{}, trt2{}, trti{};
    if (have_ort) {
        ort = Leg("ort", kOnnx, nullptr, 2);
        ort2 = Leg("ort", kOnnx, nullptr, 2);
        orti = Leg("ort", kOnnx, nullptr, 0);
        CHECK(ort.games == 32, "R1 ort 银行腿完成（32 局）");
        CHECK(ort.fp == ort2.fp, "R1 ort 复跑逐位同");
        CHECK(ort.fp == orti.fp, "R1 ort 银行 vs inline 逐位一致（含指纹）");
        std::printf("[R1] ort 银行 %.0f 局/s / inline %.0f 局/s\n",
                    ort.games / ort.sec, orti.games / orti.sec);
    } else {
        std::printf("SKIP R1: 无 %s（tools/bake_gomoku_mlp.py 烤制）\n", kOnnx);
    }
    if (have_trt) {
        trt = Leg("trt", nullptr, kTrt, 2, /*count_fail=*/false);   // 可用性探测
        if (trt.games == 0) {
            std::printf("SKIP R2: trt 后端不可用（本构建未开 INFERFARM_WITH_TRT）\n");
        } else {
            trt2 = Leg("trt", nullptr, kTrt, 2);
            trti = Leg("trt", nullptr, kTrt, 0);
            CHECK(trt.games == 32, "R2 trt 银行腿完成（32 局）");
            CHECK(trt.fp == trt2.fp, "R2 trt 复跑逐位同");
            CHECK(trt.fp == trti.fp, "R2 trt 银行 vs inline 逐位一致（含指纹）");
            std::printf("[R2] trt 银行 %.0f 局/s / inline %.0f 局/s\n",
                        trt.games / trt.sec, trti.games / trti.sec);
            if (have_ort && ort.games == 32) {
                // 观察项（不设硬门）：fp32+TF32 关下两引擎逐位等价——当前工件实测成立
                std::printf("%s: ort vs trt 跨后端指纹 %s（%016llx vs %016llx）\n",
                            ort.fp == trt.fp ? "观察[逐位同]" : "观察[不等]",
                            ort.fp == trt.fp ? "一致" : "不等",
                            ort.fp, trt.fp);
            }
        }
    } else {
        std::printf("SKIP R2: 无 %s\n", kTrt);
    }
    if (const char* dml_dir = getenv("FARM_DML_DIR")) {
        if (have_ort) {
            auto hetero_leg = [&](uint32_t seed0) {
                FarmConfig cfg;
                cfg.name = "hetero";
                cfg.chains = 8;
                cfg.games = 32;
                cfg.seed0 = seed0;
                cfg.slots = 8;
                cfg.workers = 4;
                cfg.stagger_ms = 1;
                DeviceConfig a, b;
                a.model.backend = "ort";
                a.model.model_path = kOnnx;
                a.banks = 2;
                b.model.backend = "ort";
                b.model.ort_ep = "dml";
                b.model.device_id = 1;
                b.model.model_path = kOnnx;
                b.model.ort_dir = dml_dir;
                b.banks = 1;
                cfg.devices = {a, b};
                Farm farm;
                if (!farm.Init(cfg)) { g_fail++; return R{0, 0, 0}; }
                double sec = farm.RunLeg(MakeGomokuAdapter, nullptr);
                return R{farm.tally().fingerprint, farm.tally().games_done, sec};
            };
            R h1 = hetero_leg(20260922u);
            R h2 = hetero_leg(20260922u);
            CHECK(h1.games == 32, "R4 异构双设备腿完成（cuda+dml，32 局）");
            CHECK(h1.fp == h2.fp, "R4 异构复跑逐位同（链→组钉扎的跨厂商确定性）");
            std::printf("[R4] cuda+dml 异构 %.0f 局/s（指纹 %016llx）\n",
                        h1.games / h1.sec, h1.fp);
        } else {
            std::printf("SKIP R4: 无 %s\n", kOnnx);
        }
    } else {
        std::printf("SKIP R4: 无 FARM_DML_DIR（onnxruntime-directml 的 capi 目录）\n");
    }
    // ---------------- R5：批次/位置不变性门（G13 真后端版）----------------
    {
        auto r5 = [&](const char* tag, InferBackend* be, const ModelConfig& mcfg,
                      int hint) {
            if (!be) { std::printf("SKIP R5 %s: 后端不可用\n", tag); return; }
            ModelSpec spec;
            if (!be->LoadSpec(mcfg, hint, spec) || spec.slots < 2) {
                std::printf("SKIP R5 %s: LoadSpec 失败（工件缺席或形状不符）\n", tag);
                delete be;
                return;
            }
            void* sess = be->CreateSession(mcfg, spec, /*for_bank=*/true);
            if (!sess || !be->Warmup(sess)) {
                std::printf("SKIP R5 %s: 会话/热身不可用（运行时缺席）\n", tag);
                if (sess) be->DestroySession(sess);
                delete be;
                return;
            }
            const int S = spec.slots;
            auto run = [&](const std::vector<int>& slot_seed, int slot_read,
                           std::vector<float>& out) -> bool {
                auto fill = [&](int slot, int content_seed) -> bool {
                    for (size_t i = 0; i < spec.ins.size(); i++) {
                        if (spec.ins[i].population) continue;
                        size_t rb = 0;
                        void* row = be->InputRow(
                            sess, spec.ins[i].name.c_str(), slot, &rb);
                        if (!row) return false;
                        if (spec.ins[i].et == DTYPE_F32) {
                            uint32_t x = 0x9E3779B9u * (uint32_t)(content_seed + 1);
                            float* f = (float*)row;
                            for (size_t j = 0; j < rb / sizeof(float); j++) {
                                x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                                f[j] = (float)((int)(x & 0xFFFF) - 32768) / 32768.0f;
                            }
                        } else {
                            std::memset(row, 0, rb);
                        }
                    }
                    return true;
                };
                for (size_t r = 0; r < slot_seed.size(); r++)
                    if (!fill((int)r, slot_seed[r])) return false;
                unsigned seq = 0;
                if (!be->SubmitBatch(sess, (int)slot_seed.size(), seq)) return false;
                while (!be->CompletionReached(sess, seq)) {}
                be->CompletionFence();
                out.clear();   // 复用缓冲：先清（跨调用累积会让 biteq 假败）
                for (size_t i = 0; i < spec.outs.size(); i++) {
                    int w = be->OutputWidth(sess, spec.outs[i].name.c_str());
                    const float* src = w > 0
                        ? be->OutputRow(sess, spec.outs[i].name.c_str(), slot_read)
                        : nullptr;
                    if (!src) return false;
                    out.insert(out.end(), src, src + w);
                }
                return true;
            };
            auto biteq = [](const std::vector<float>& a, const std::vector<float>& b) {
                return a.size() == b.size()
                    && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
            };
            std::vector<std::vector<float>> ref((size_t)S);
            std::vector<int> full;
            for (int r_ = 0; r_ < S; r_++) full.push_back(r_);
            bool base = true;
            for (int r_ = 0; r_ < S; r_++) base = run(full, r_, ref[(size_t)r_]) && base;
            std::string msg;
            msg = std::string(tag) + " R5 满批基线";
            CHECK(base, msg.c_str());
            std::vector<float> got;
            bool ok = run({0}, 0, got) && biteq(got, ref[0]);
            msg = std::string(tag) + " R5a 批大小不变性：n=1 行0 逐位同";
            CHECK(ok, msg.c_str());
            ok = run({0, 1, 2}, 2, got) && biteq(got, ref[2]);
            msg = std::string(tag) + " R5b 批大小不变性：n=3 行2 逐位同";
            CHECK(ok, msg.c_str());
            ok = true;
            for (int p : {1, S / 2, S - 1}) {
                std::vector<int> pc;
                for (int i = 0; i < p; i++) pc.push_back(i + 1);
                pc.push_back(0);
                ok = run(pc, p, got) && biteq(got, ref[0]) && ok;
            }
            msg = std::string(tag) + " R5c 行位置不变性：内容0@槽{1,S/2,S-1} 逐位同";
            CHECK(ok, msg.c_str());
            be->DestroySession(sess);
            delete be;
        };
        if (have_ort) {
            ModelConfig m;
            m.backend = "ort";
            m.model_path = kOnnx;
            r5("ort", CreateOrtBackend(), m, 8);
        } else {
            std::printf("SKIP R5 ort: 无 %s\n", kOnnx);
        }
        if (InferBackend* trt_be = CreateTrtBackend()) {
            ModelConfig m;
            m.backend = "trt";
            m.engine_path = kTrt;
            r5("trt", trt_be, m, 8);
        } else {
            std::printf("SKIP R5 trt: 本构建未编 TRT\n");
        }
    }
    if (!have_ort && !have_trt) {
        std::printf("（本目录无模型工件——全部 SKIP 属正常）\n");
        return 0;
    }
    std::printf("=== 完成：%s（%d 失败）===\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
