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
//   R6 fence 桥接门（FARM_ORT_ASYNC=3，2026-09-24）：patch_fence.py 补丁模型
//      fence 模式 == sync 基线逐位（跨通道主门）+ 复跑 + 银行 vs inline
//      （工件缺席=SKIP）
//   R7 trt refit 真引擎换心（B5，2026-09-24）：refittable 引擎 + RW1 全零
//      换心必变 + 复采逐位同 + 二次 refit 幂等 + 名单外假名负路径（引擎
//      不可 refit=SKIP——旧工件需重烤）
//   R8 声明式增量 H2D（判决25，2026-09-27）：R8a backend 级——append 会话 vs
//      full 会话同行内容输出逐位（增长段/换局重铸/未声明整行兜底/满深/零深/
//      零段全路径）；R8a-V sentinel 违约门（FARM_H2D_DELTA_DEBUG=1 + 人为
//      前缀违约→OrtDeltaDebugViolations 必增，clean 批不增）；R8b farm 级——
//      包装适配器（own 面深度前缀模式+FaceDepth 申报）append 腿 == 强制
//      full 腿逐位同（增量漏传必指纹红=哨兵语义验收）+ 复跑 + inline 同
//      （工件缺席=SKIP）
//   R8c headlive 头部活跃面（判决25 扩展，2026-09-27）：newest-first 面
//      [0,depth) 每批可变/尾槽恒零——同深度内容全换（逆序移位，append 做不到）、
//      缩深 memset 重铸、未声明兜底、宿主尾槽非零哨兵必报
//
// 工件烤制：python tools/bake_gomoku_mlp.py --slots 8 --hidden 64 \
//   --out models/gomoku_mlp.fb8.onnx --trt models/gomoku_mlp.fb8.trt
// （--trt 自带 BuilderFlag.REFIT=refittable；RW1 导出器 tools/refit_mlp_rw1.py）
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

// MSVC _putenv_s 的可移植等价（POSIX=setenv；R6 切档用，2026-09-27）
#ifdef _WIN32
static void TestSetEnv(const char* k, const char* v) { _putenv_s(k, v); }
#else
static void TestSetEnv(const char* k, const char* v) { setenv(k, v, 1); }
#endif

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", msg); g_fail++; } \
    else std::printf("ok: %s\n", msg); \
    std::fflush(stdout); \
} while (0)

// ---------------- R9：③成对状态行玩具适配器 ----------------
// 池路径：S_prev 不写（框架 D2D 填充）+S_next 不申报 dest（不过主机）；
// 主机路径：影子累加（S_prev 自写+S_next 回读步进）——双腿同 seed 必逐位同
//（引擎相同、状态演化等价：主机影子=S_next 的逐位拷贝）。
class StateToyAdapter : public GameAdapter {
public:
    StateToyAdapter(int chain, bool pool) : chain_(chain), pool_(pool) {}
    void NewGame(uint64_t seed, bool we_first) override {
        seed_ = seed; (void)we_first;
        dec_ = 0; fail_ = false;
        for (int j = 0; j < 4; j++) acc_[j] = 0.0f;   // 池路径不用（框架池行清零）
        fp_ = 1469598103934665603ULL ^ seed;
    }
    bool AdvanceToDecision() override { return dec_ < 6 && !fail_; }
    void AssembleInto(SlotWriter& slot) override {
        float* x = (float*)slot.Row("x", nullptr);
        if (x)
            for (int j = 0; j < 4; j++)
                x[j] = (float)(((int)((chain_ * 31 + dec_ * 7 + j + (int)(seed_ % 13))
                                     % 17) - 8)) * 0.25f;
        if (!pool_) {   // 主机路径：影子状态写 S_prev
            float* s = (float*)slot.Row("S_prev", nullptr);
            if (s) for (int j = 0; j < 4; j++) s[j] = acc_[j];
        }
        // 池路径：S_prev 不写（契约窄化——框架 D2D 池行填充）
    }
    int CollectOutputs(OutputDest* d, int cap) override {
        if (cap < 1 || fail_) return 0;
        d[0].name = "policy"; d[0].dst = &pol_; d[0].n = 1;
        if (!pool_ && cap >= 2) {   // 主机路径：S_next 回读
            d[1].name = "S_next"; d[1].dst = nxt_; d[1].n = 4;
            return 2;
        }
        return 1;
    }
    void ApplyResult() override {
        if (!pool_) for (int j = 0; j < 4; j++) acc_[j] = nxt_[j];
        unsigned bits; std::memcpy(&bits, &pol_, 4);   // 指纹掺 policy 位
        fp_ = (fp_ ^ bits) * 1099511628211ULL;
        dec_++;
    }
    void OnInferFail() override { fail_ = true; }
    bool IsDone() override { return dec_ >= 6 || fail_; }
    int Outcome() override { return fail_ ? -1 : (pol_ > 0.0f ? 1 : 0); }
    bool WeAreFirst() override { return true; }
    long long GameFingerprint() override { return (long long)fp_; }
    ITlsFrame* TlsFrame() override { return nullptr; }
private:
    int chain_; bool pool_; uint64_t seed_ = 0; int dec_ = 0; bool fail_ = false;
    float acc_[4] = {0, 0, 0, 0}, nxt_[4] = {0, 0, 0, 0}, pol_ = 0;
    uint64_t fp_ = 0;
};
static GameAdapter* StateToyMake(int chain, void* user) {
    return new StateToyAdapter(chain, user != nullptr);
}

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

int main(int argc, char** argv) {
    std::fprintf(stderr, "[test] main 进入\n");
    std::fflush(stderr);
    // 可选后端过滤（2026-09-27）：gomoku_backend_test [ort|trt]
    // 缺省=全量（原行为）。动机=Linux 实测同进程 TRT 推理会污染其后的
    // ORT CUDA 图捕获热身（cudaErrorInvalidValue@Concat，判决 27）——
    // 两面各跑各的 ALL PASS，共存限制由文档承载。
    const char* only = (argc > 1) ? argv[1] : nullptr;
    // r9 子档（argv[2]=="r9"）：③状态池门独立跑——trt 后端一进程一引擎
    //（全局缓存），R9 的玩具引擎须避开 R2-R7 的 gomoku 引擎
    const bool r9_only = argc > 2 && !std::strcmp(argv[2], "r9");
    if (only) std::printf("=== 后端过滤：仅 %s 面 ===\n", only);
    const char* kOnnx = "models/gomoku_mlp.fb8.onnx";
    const char* kTrt = "models/gomoku_mlp.fb8.trt";
    std::printf("=== 真模型可选门（工件缺席=SKIP）===\n");
    bool have_ort = FileExists(kOnnx) && !(only && !std::strcmp(only, "trt"));
    bool have_trt = FileExists(kTrt) && !(only && !std::strcmp(only, "ort"))
                    && !r9_only;
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
        InferBackend* trt_be = have_trt ? CreateTrtBackend() : nullptr;
        if (trt_be) {
            ModelConfig m;
            m.backend = "trt";
            m.engine_path = kTrt;
            r5("trt", trt_be, m, 8);
        } else {
            std::printf("SKIP R5 trt: 本构建未编 TRT\n");
        }
    }
    // ---------------- R6：fence 桥接门（FARM_ORT_ASYNC=3，2026-09-24）----------------
    // 判决12 翻案通道的验收：patch_fence.py 打补丁的模型（数学零变化）+ fence
    // 模式腿 == sync 基线（R1 ort.fp）逐位——上次 =2 用户流方案就是挂在
    // 这类跨通道对拍上（3 跑 3 指纹）。env 每会话读取（CreateSession 处），
    // 同进程 _putenv 切档。工件缺席=SKIP。
    // ⚠ 运行面：本门相对路径取工件，须在仓根且 fb8/fb8_fence 两件在场才跑
    // （从 build/Release 等目录跑=SKIP；CI 无工件恒 SKIP——fence 语义的回归
    // 保护靠本地仓根跑，CI 绿不覆盖本门）。
    {
        const char* kFence = "models/gomoku_mlp.fb8_fence.onnx";
        if (!have_ort) {
            std::printf("SKIP R6: 无 %s\n", kOnnx);
        } else if (!FileExists(kFence)) {
            std::printf("SKIP R6: 无 %s（python tools/patch_fence.py %s "
                        "--out %s）\n", kFence, kOnnx, kFence);
        } else {
            TestSetEnv("FARM_ORT_ASYNC", "3");
            long long f0 = OrtFenceEngagedTotal();
            R f1 = Leg("ort", kFence, nullptr, 2);
            R f2 = Leg("ort", kFence, nullptr, 2);
            // 空过防线：fence 静默回落同步也会逐位同——真启用断言靠此计数
            //（每腿 2 银行 × 2 腿 = 4 会话，Warmup 烟雾通过才计）
            long long f_bank = OrtFenceEngagedTotal() - f0;
            CHECK(f_bank == 4, "R6 fence 真启用=4 会话（静默回落=此门红）");
            R fi = Leg("ort", kFence, nullptr, 0);
            TestSetEnv("FARM_ORT_ASYNC", "0");
            CHECK(OrtFenceEngagedTotal() - f0 == f_bank,
                  "R6 inline 腿不误登记（armed 票号握手）");
            CHECK(f1.games == 32, "R6 fence 银行腿完成（32 局）");
            CHECK(f1.fp == f2.fp, "R6 fence 复跑逐位同");
            CHECK(f1.fp == fi.fp, "R6 fence 银行 vs inline 逐位一致（含指纹）");
            if (ort.games == 32)
                CHECK(f1.fp == ort.fp, "R6 fence==sync 跨通道逐位一致（主门）");
            std::printf("[R6] fence 银行 %.0f 局/s / fence inline %.0f 局/s"
                        "（sync 银行 %.0f 局/s）\n",
                        f1.games / f1.sec, fi.games / f1.sec,
                        ort.games / ort.sec);
        }
    }
    // ---------------- R7：trt refit 真引擎换心（B5，2026-09-24）----------------
    // refittable 引擎（bake 端 BuilderFlag.REFIT；kREFIT_NONE=SKIP——未重烤
    // 的旧工件兼容）+ C++ 内构 RW1（名单权威=onnx initializer：fc1.weight/
    // fc1.bias/fc2.weight/fc2.bias，值全零）：
    //   ① 已知输入采 policy 行0 → ref0
    //   ② RefitWeights(rw1) → 同输入 ref1 ≠ ref0（换心必变，G5 异 blob TRT 版）
    //   ③ 复采 ref1b == ref1（逐位确定）
    //   ④ 同 blob 二次 refit → ref1c == ref1（refit 幂等）
    //   ⑤ 全名单外假名 RW1 → 引擎不动（ref2 == ref1；负路径）
    // 名单对齐实证：真名 4 项全中（名单外跳过 0）+ missing 拒绝语义由后端
    // ApplyRefitWeights 把守（仓根 tools/refit_mlp_rw1.py 同名单）。R5 同款
    // backend 级采样；前置=腿可用（trt.games != 0=后端在）。
    if (have_trt && trt.games != 0) {
        InferBackend* be = CreateTrtBackend();
        ModelConfig mc;
        mc.backend = "trt";
        mc.engine_path = kTrt;
        ModelSpec spec;
        if (!be || !be->LoadSpec(mc, 8, spec)) {
            std::printf("SKIP R7: LoadSpec 失败\n");
            delete be;
        } else {
            void* sess = be->CreateSession(mc, spec, /*for_bank=*/true);
            if (!sess || !be->Warmup(sess)) {
                std::printf("SKIP R7: 会话/热身不可用\n");
                if (sess) be->DestroySession(sess);
                delete be;
            } else {
                auto biteq7 = [](const std::vector<float>& a,
                                 const std::vector<float>& b) {
                    return a.size() == b.size()
                        && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
                };
                auto sample0 = [&]() -> std::vector<float> {
                    for (size_t i = 0; i < spec.ins.size(); i++) {
                        size_t rb = 0;
                        void* row = be->InputRow(sess, spec.ins[i].name.c_str(), 0, &rb);
                        uint32_t x = 0x1234567u * (uint32_t)(i + 7);
                        float* f = (float*)row;
                        for (size_t j = 0; j < rb / sizeof(float); j++) {
                            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                            f[j] = (float)((int)(x & 0xFFFF) - 32768) / 32768.0f;
                        }
                    }
                    unsigned seq = 0;
                    if (!be->SubmitBatch(sess, 1, seq)) return {};
                    while (!be->CompletionReached(sess, seq)) {}
                    be->CompletionFence();
                    const float* p = be->OutputRow(sess, "policy", 0);
                    return std::vector<float>(p, p + (size_t)spec.outs[0].width);
                };
                auto write_rw1 = [](const char* path, const char* const* names,
                                    size_t n_names, float fill) {
                    std::vector<char> b;
                    auto put32 = [&](uint32_t v) {
                        for (int i = 0; i < 4; i++) b.push_back((char)(v >> (8 * i)));
                    };
                    auto put16 = [&](uint16_t v) {
                        for (int i = 0; i < 2; i++) b.push_back((char)(v >> (8 * i)));
                    };
                    b.insert(b.end(), {'R', 'W', '1', '\0'});
                    put32(1);
                    put32((uint32_t)n_names);
                    static const int kDims[4][2] = {{64, 450}, {64, 1}, {225, 64}, {225, 1}};
                    for (size_t k = 0; k < n_names && k < 4; k++) {
                        std::string nm = names[k];
                        put16((uint16_t)nm.size());
                        b.insert(b.end(), nm.begin(), nm.end());
                        b.push_back((char)2);   // f32
                        int n_el = kDims[k][0] * kDims[k][1];
                        put32((uint32_t)n_el);
                        for (int e = 0; e < n_el; e++) put32(0);
                    }
                    FILE* f = fopen(path, "wb");
                    if (f) { fwrite(b.data(), 1, b.size(), f); fclose(f); }
                    return f != nullptr;
                };
                std::vector<float> ref0 = sample0();
                CHECK(!ref0.empty(), "R7 换心前基线采样");
                static const char* kNames[4] = {"fc1.weight", "fc1.bias",
                                                "fc2.weight", "fc2.bias"};
                const char* kRw1 = "gomoku_r7_refit.rw1";
                bool w1 = write_rw1(kRw1, kNames, 4, 0.0f);
                bool refit1 = w1 && be->RefitWeights(kRw1);
                std::vector<float> ref1 = refit1 ? sample0() : std::vector<float>{};
                CHECK(refit1 && !ref1.empty() && !biteq7(ref0, ref1),
                      "R7 换心必变（refittable 真引擎+名单对齐 4 项全中）");
                std::vector<float> ref1b = sample0();
                CHECK(biteq7(ref1, ref1b), "R7 换心后复采逐位同");
                bool refit2 = be->RefitWeights(kRw1);
                std::vector<float> ref1c = refit2 ? sample0() : std::vector<float>{};
                CHECK(refit2 && biteq7(ref1, ref1c), "R7 同 blob 二次 refit 幂等");
                static const char* kFake[2] = {"nope.weight", "also_fake.bias"};
                bool w3 = write_rw1(kRw1, kFake, 2, 0.0f);
                bool refit3 = w3 && be->RefitWeights(kRw1);
                std::vector<float> ref2 = refit3 ? sample0() : std::vector<float>{};
                CHECK(refit3 && biteq7(ref1, ref2),
                      "R7 名单外假名=引擎不动（负路径）");
                std::remove(kRw1);
                be->DestroySession(sess);
                delete be;
            }
        }
    }
    // ---------------- R8：声明式增量 H2D（判决25）----------------
    // 深度前缀模式（content=f(k) 与局无关）是 append-only 承诺的结构性成立形态：
    // 任意行任意局写 [0,d) 都是同一前缀 → 槽轮转/换局下增长段假设恒真。
    // 掼蛋真负载的接入语义见判决 25 接入指引（承诺不可行时就别声明=full 兜底）。
    if (have_ort) {
        // ---- R8a：backend 级（delta 会话 vs full 会话，同写同报）----
        InferBackend* be8 = CreateOrtBackend();
        ModelConfig m8a;
        m8a.backend = "ort";
        m8a.model_path = kOnnx;
        m8a.append_inputs.push_back("own");
        ModelConfig m8f = m8a;
        m8f.append_inputs.clear();   // 强制 full 对照
        ModelSpec sp8, sp8f;
        bool r8_ok = be8 && be8->LoadSpec(m8a, 8, sp8)
            && be8->LoadSpec(m8f, 8, sp8f);
        bool own_append = false;
        for (auto& i : sp8.ins)
            if (i.name == "own" && i.append) own_append = true;
        CHECK(r8_ok && own_append, "R8a append 面进 spec（append_inputs→InputMeta）");
        void* sd = r8_ok ? be8->CreateSession(m8a, sp8, true) : nullptr;
        void* sf = r8_ok ? be8->CreateSession(m8f, sp8f, true) : nullptr;
        bool warm8 = sd && sf && be8->Warmup(sd) && be8->Warmup(sf);
        CHECK(warm8, "R8a delta/full 双会话建+热身");
        if (warm8) {
            const int S = sp8.slots;
            const int D = 225;   // own 面行首维（深度上限）
            // 深度前缀模式：行内容 [0,d)=pat(k)、[d,D)=0（零基组装的显式版）
            auto fill_hist = [&](void* sess, int slot, int depth) {
                size_t rb = 0;
                float* own = (float*)be8->InputRow(sess, "own", slot, &rb);
                if (!own) return;
                for (int k = 0; k < D; k++) {
                    uint32_t x = 0x9E3779B9u * (uint32_t)(k + 1);
                    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                    own[k] = k < depth
                        ? (float)((int)(x & 0xFFFF) - 32768) / 32768.0f : 0.0f;
                }
            };
            auto fill_board = [&](void* sess, int slot, uint32_t seed) {
                size_t rb = 0;
                float* opp = (float*)be8->InputRow(sess, "opp", slot, &rb);
                if (!opp) return;
                uint32_t x = seed;
                for (size_t k = 0; k < rb / sizeof(float); k++) {
                    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                    opp[k] = (float)((int)(x & 0xFFFF) - 32768) / 32768.0f;
                }
            };
            auto snap = [&](void* sess, int slot) {
                std::vector<float> v;
                for (size_t i = 0; i < sp8.outs.size(); i++) {
                    int w = be8->OutputWidth(sess, sp8.outs[i].name.c_str());
                    const float* p = w > 0
                        ? be8->OutputRow(sess, sp8.outs[i].name.c_str(), slot)
                        : nullptr;
                    if (p) v.insert(v.end(), p, p + w);
                }
                return v;
            };
            // 一轮：两会话同写 rows[0,n)，delta 侧按 depths 申报（-1=不申报）
            // → 各自 SubmitBatch → 逐槽逐位比对
            auto round = [&](int n, const std::vector<int>& depths,
                             uint32_t seed, const char* tag) {
                for (int r = 0; r < n; r++) {
                    fill_hist(sd, r, depths[(size_t)r]);
                    fill_hist(sf, r, depths[(size_t)r]);
                    fill_board(sd, r, seed + (uint32_t)r);
                    fill_board(sf, r, seed + (uint32_t)r);
                    if (depths[(size_t)r] >= 0)
                        be8->NoteFaceDepth(sd, "own", r, depths[(size_t)r]);
                }
                unsigned q1 = 0, q2 = 0;
                bool ok = be8->SubmitBatch(sd, n, q1) && be8->SubmitBatch(sf, n, q2);
                while (ok && !be8->CompletionReached(sd, q1)) {}
                while (ok && !be8->CompletionReached(sf, q2)) {}
                be8->CompletionFence();
                bool same = ok;
                for (int r = 0; r < n && same; r++)
                    same = snap(sd, r) == snap(sf, r);
                CHECK(same, tag);
                return same;
            };
            bool r8a = true;
            r8a = round(5, {3, 5, 2, 7, 4}, 111u, "R8a-1 首批增长段（5 行 fresh depth）") && r8a;
            r8a = round(8, {8, 6, 5, 9, 12, 1, D, 0}, 222u,
                        "R8a-2 混合批（增长/换局重铸/满深/零深/fresh）") && r8a;
            // 行 6 在上轮满深（225→kFullSync）：本轮申报 d=1 = kFullSync 重置
            //（行界钳制回归面——虚增量+整行兜底组合曾在此 4GB memset 越界）
            r8a = round(8, {10, -1, 4, 5, -1, 2, 1, 3}, 333u,
                        "R8a-3 未声明行整行兜底+kFullSync 重置（逐行混批）") && r8a;
            r8a = round(8, {10, -1, 4, 5, -1, 2, 1, 3}, 333u,
                        "R8a-4 零段批（d==synced 不传）逐位同") && r8a;
            // 哨兵违约门：debug 会话（env 于建会话前设）+ 人为前缀违约——
            // [0,4) 换内容后申报 d=6（增长段只传 [4,6)）→ 影子≠宿主必报
            long long vio0 = OrtDeltaDebugViolations();
            CHECK(vio0 == 0, "R8a clean 批哨兵零违约（对照面）");
            TestSetEnv("FARM_H2D_DELTA_DEBUG", "1");
            void* sv = be8->CreateSession(m8a, sp8, true);
            bool wv = sv && be8->Warmup(sv);
            TestSetEnv("FARM_H2D_DELTA_DEBUG", "0");
            CHECK(wv, "R8a-V debug 哨兵会话建+热身");
            if (wv) {
                fill_hist(sv, 0, 4);
                fill_board(sv, 0, 444u);
                be8->NoteFaceDepth(sv, "own", 0, 4);
                unsigned q = 0;
                bool ok1 = be8->SubmitBatch(sv, 1, q);
                while (ok1 && !be8->CompletionReached(sv, q)) {}
                be8->CompletionFence();
                // 违约：换掉已同步前缀 [0,4) 的内容，再申报增长 d=6
                size_t rb = 0;
                float* own = (float*)be8->InputRow(sv, "own", 0, &rb);
                for (int k = 0; k < 6; k++) {
                    uint32_t x = 0xDEADBEEFu * (uint32_t)(k + 3);
                    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                    own[k] = (float)((int)(x & 0xFFFF) - 32768) / 32768.0f;
                }
                be8->NoteFaceDepth(sv, "own", 0, 6);
                bool ok2 = be8->SubmitBatch(sv, 1, q);
                while (ok2 && !be8->CompletionReached(sv, q)) {}
                be8->CompletionFence();
                CHECK(OrtDeltaDebugViolations() > vio0,
                      "R8a-V 前缀违约哨兵必报（计数增长；fprintf 第一现场）");
                be8->DestroySession(sv);
            }
            (void)r8a;
            be8->DestroySession(sd);
            be8->DestroySession(sf);
        } else if (sd || sf) {
            if (sd) be8->DestroySession(sd);
            if (sf) be8->DestroySession(sf);
        }
        delete be8;

        // ---- R8c：headlive 头部活跃面（判决25 扩展，2026-09-27）----
        // newest-first 面：[0,depth) 每批可任意变化（逆序移位——append 做不到
        // 的语义），[depth,slots) 宿主恒零。与 full 对照会话同写同报逐位；
        // 缩深=设备 memset 零基重铸；哨兵抓宿主尾槽非零（承诺违约）。
        {
            InferBackend* bec = CreateOrtBackend();
            ModelConfig m8h;
            m8h.backend = "ort";
            m8h.model_path = kOnnx;
            m8h.headlive_inputs.push_back("own");
            ModelConfig m8c = m8h;
            m8c.headlive_inputs.clear();   // 强制 full 对照
            ModelSpec sph, spc;
            bool r8c_ok = bec && bec->LoadSpec(m8h, 8, sph)
                && bec->LoadSpec(m8c, 8, spc);
            bool own_hl = false;
            for (auto& i : sph.ins)
                if (i.name == "own" && i.headlive && !i.append) own_hl = true;
            CHECK(r8c_ok && own_hl,
                  "R8c headlive 面进 spec（headlive_inputs→InputMeta）");
            void* sh = r8c_ok ? bec->CreateSession(m8h, sph, true) : nullptr;
            void* sc = r8c_ok ? bec->CreateSession(m8c, spc, true) : nullptr;
            bool warmc = sh && sc && bec->Warmup(sh) && bec->Warmup(sc);
            CHECK(warmc, "R8c headlive/full 双会话建+热身");
            if (warmc) {
                const int D = 225;
                auto fill_live = [&](void* sess, int slot, int depth,
                                     uint32_t seed) {
                    size_t rb = 0;
                    float* own = (float*)bec->InputRow(sess, "own", slot, &rb);
                    if (!own) return;
                    for (int k = 0; k < D; k++) {
                        uint32_t x = seed * (uint32_t)(k + 1);
                        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                        own[k] = k < depth
                            ? (float)((int)(x & 0xFFFF) - 32768) / 32768.0f
                            : 0.0f;
                    }
                };
                auto fill_board2 = [&](void* sess, int slot, uint32_t seed) {
                    size_t rb = 0;
                    float* opp = (float*)bec->InputRow(sess, "opp", slot, &rb);
                    if (!opp) return;
                    uint32_t x = seed;
                    for (size_t k = 0; k < rb / sizeof(float); k++) {
                        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                        opp[k] = (float)((int)(x & 0xFFFF) - 32768) / 32768.0f;
                    }
                };
                auto snapc = [&](void* sess, int slot) {
                    std::vector<float> v;
                    for (size_t i = 0; i < sph.outs.size(); i++) {
                        int w = bec->OutputWidth(sess, sph.outs[i].name.c_str());
                        const float* p = w > 0
                            ? bec->OutputRow(sess, sph.outs[i].name.c_str(), slot)
                            : nullptr;
                        if (p) v.insert(v.end(), p, p + w);
                    }
                    return v;
                };
                auto roundc = [&](int n, const std::vector<int>& depths,
                                  uint32_t seed, const char* tag) {
                    for (int r = 0; r < n; r++) {
                        int d = depths[(size_t)r];
                        fill_live(sh, r, d, seed + (uint32_t)r * 7u);
                        fill_live(sc, r, d, seed + (uint32_t)r * 7u);
                        fill_board2(sh, r, seed + (uint32_t)r);
                        fill_board2(sc, r, seed + (uint32_t)r);
                        if (d >= 0) bec->NoteFaceDepth(sh, "own", r, d);
                    }
                    unsigned q1 = 0, q2 = 0;
                    bool ok = bec->SubmitBatch(sh, n, q1)
                        && bec->SubmitBatch(sc, n, q2);
                    while (ok && !bec->CompletionReached(sh, q1)) {}
                    while (ok && !bec->CompletionReached(sc, q2)) {}
                    bec->CompletionFence();
                    bool same = ok;
                    for (int r = 0; r < n && same; r++)
                        same = snapc(sh, r) == snapc(sc, r);
                    CHECK(same, tag);
                    return same;
                };
                bool r8c = true;
                r8c = roundc(5, {4, 4, 4, 4, 4}, 1000u,
                             "R8c-1 同深度内容全换（newest-first 移位）逐位同") && r8c;
                r8c = roundc(5, {4, 4, 4, 4, 4}, 2000u,
                             "R8c-2 同深度再换批（内容漂移持续）逐位同") && r8c;
                r8c = roundc(6, {12, 8, 3, 9, 1, 6}, 3000u,
                             "R8c-3 缩深/增长混合（换局 memset 重铸）逐位同") && r8c;
                r8c = roundc(4, {5, -1, 2, 7}, 4000u,
                             "R8c-4 未声明行整行兜底逐位同") && r8c;
                long long vio0c = OrtDeltaDebugViolations();
                TestSetEnv("FARM_H2D_DELTA_DEBUG", "1");
                void* svh = bec->CreateSession(m8h, sph, true);
                bool wvh = svh && bec->Warmup(svh);
                TestSetEnv("FARM_H2D_DELTA_DEBUG", "0");
                CHECK(wvh, "R8c-V debug 哨兵会话建+热身");
                if (wvh) {
                    fill_live(svh, 0, 3, 5000u);
                    fill_board2(svh, 0, 5001u);
                    bec->NoteFaceDepth(svh, "own", 0, 3);
                    unsigned q = 0;
                    bool ok1 = bec->SubmitBatch(svh, 1, q);
                    while (ok1 && !bec->CompletionReached(svh, q)) {}
                    bec->CompletionFence();
                    // 违约：宿主尾槽 [3,225) 写非零再申报同深——设备/影子尾
                    // 恒零 vs 宿主≠零 → memcmp 必报（headlive 承诺的执法面）
                    size_t rb = 0;
                    float* own = (float*)bec->InputRow(svh, "own", 0, &rb);
                    own[100] = 0.5f;
                    bec->NoteFaceDepth(svh, "own", 0, 3);
                    bool ok2 = bec->SubmitBatch(svh, 1, q);
                    while (ok2 && !bec->CompletionReached(svh, q)) {}
                    bec->CompletionFence();
                    CHECK(OrtDeltaDebugViolations() > vio0c,
                          "R8c-V 宿主尾槽非零哨兵必报（承诺违约）");
                    bec->DestroySession(svh);
                }
                (void)r8c;
                bec->DestroySession(sh);
                bec->DestroySession(sc);
            } else {
                if (sh) bec->DestroySession(sh);
                if (sc) bec->DestroySession(sc);
            }
            delete bec;
        }

        // ---- R8b：farm 级（append 声明腿 == 强制 full 腿，指纹逐位）----
        struct AppendGomokuAdapter : gomoku::GomokuAdapter {
            explicit AppendGomokuAdapter(int chain) : GomokuAdapter(chain) {}
            static float HistPat(int k) {
                uint32_t x = 0x9E3779B9u * (uint32_t)(k + 1);
                x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                return (float)((int)(x & 0xFFFF) - 32768) / 32768.0f;
            }
            void AssembleInto(SlotWriter& slot) override {
                GomokuAdapter::AssembleInto(slot);   // 棋盘两平面照写
                float* own = (float*)slot.Row("own", nullptr);
                if (!own) return;
                const int depth = moves + 1 < kCells ? moves + 1 : kCells;
                for (int k = 0; k < depth; k++) own[k] = HistPat(k);
                std::memset(own + depth, 0, (size_t)(kCells - depth) * sizeof(float));
                slot.FaceDepth("own", depth);   // 声明式增量 H2D 的申报点
            }
        };
        auto append_make = [](int chain, void*) -> GameAdapter* {
            return new AppendGomokuAdapter(chain);
        };
        auto r8_leg = [&](int banks, bool append) {
            FarmConfig cfg;
            cfg.name = "gomoku-append";
            cfg.chains = 8;
            cfg.games = 32;
            cfg.seed0 = 20260922u;
            cfg.banks = banks;
            cfg.slots = 8;
            cfg.workers = 4;
            cfg.stagger_ms = 1;
            cfg.model.backend = "ort";
            cfg.model.model_path = kOnnx;
            if (append) cfg.model.append_inputs.push_back("own");
            Farm farm;
            if (!farm.Init(cfg)) { g_fail++; return R{0, 0, 0}; }
            double sec = farm.RunLeg(append_make, nullptr);
            return R{farm.tally().fingerprint, farm.tally().games_done, sec};
        };
        R a1 = r8_leg(2, true);
        R a2 = r8_leg(2, true);
        R af = r8_leg(2, false);
        R ai = r8_leg(0, true);
        CHECK(a1.games == 32 && af.games == 32, "R8b append/full 腿完成（32 局）");
        CHECK(a1.fp == af.fp,
              "R8b append==强制 full 逐位同（增量漏传必指纹红=主门）");
        CHECK(a1.fp == a2.fp, "R8b append 复跑逐位同");
        CHECK(a1.fp == ai.fp,
              "R8b append 银行 vs inline 逐位同（协议不变量=设备==宿主的端到端）");
        std::printf("[R8] append 银行 %.0f 局/s / full %.0f 局/s\n",
                    a1.games / a1.sec, af.games / af.sec);
    } else {
        std::printf("SKIP R8: 无 %s\n", kOnnx);
    }
    // ---------------- R9：③成对状态行（设备池，docs/state-residency-design.md）----------------
    // 玩具引擎 S_next=S_prev+x 跨决策累加+每链多局（换局池行清零验证）。
    // 主门：池路径（state_pairs 声明）==主机路径（影子累加）同 seed 逐位同
    //（池错/粘滞错/清零漏必指纹红）+复跑同。trt 面（工件缺席=SKIP）。
    // 独立子档跑（`trt r9`）：trt 后端一进程一引擎（全局缓存），与 R2-R7 的
    // gomoku 引擎互斥——有 gomoku 工件在场的完整面自动 SKIP R9。
    if ((!only || !std::strcmp(only, "trt")) && !have_trt) {
        const char* kToy = "models/state_toy.fb8.trt";
        if (!FileExists(kToy)) {
            std::printf("SKIP R9: 无 %s（tools/bake_state_toy.py + bake_fb8_trt.py）\n",
                        kToy);
        } else {
            auto r9_leg = [&](bool pool) -> R {
                FarmConfig cfg;
                cfg.name = "r9";
                cfg.chains = 4;
                cfg.games = 16;   // 每链 4 局=换局清零进主门
                cfg.seed0 = 20260929u;
                cfg.banks = 2;
                cfg.slots = 8;
                cfg.workers = 4;
                cfg.stagger_ms = 1;
                cfg.model.backend = "trt";
                cfg.model.engine_path = kToy;
                if (pool) cfg.model.state_pairs.push_back({"S_prev", "S_next"});
                Farm farm;
                if (!farm.Init(cfg)) { g_fail++; return R{0, 0, 0}; }
                double sec = farm.RunLeg(StateToyMake, (void*)(pool ? 1 : 0));
                return R{farm.tally().fingerprint, farm.tally().games_done, sec};
            };
            R rh = r9_leg(false);
            R rp = r9_leg(true);
            R rp2 = r9_leg(true);
            CHECK(rh.games == 16, "R9 主机路径腿完成（16 局）");
            CHECK(rp.games == 16 && rp.fp == rh.fp,
                  "R9a 池路径==主机路径逐位同（含指纹；池错/清零漏必红=主门）");
            CHECK(rp2.fp == rp.fp, "R9b 池路径复跑逐位同");
            if (rh.games == 16 && rp.games == 16)
                std::printf("[R9] 池 %.0f 局/s / 主机 %.0f 局/s（玩具小图，吞吐"
                            "非观测量）\n", rp.games / rp.sec, rh.games / rh.sec);
        }
    }
    if (!have_ort && !have_trt && !r9_only) {
        std::printf("（本目录无模型工件——全部 SKIP 属正常）\n");
        return 0;
    }
    std::printf("=== 完成：%s（%d 失败）===\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
