// farm.cpp — 编排器：种子协议/驱动环/收账/env 覆盖（YGO OppRunOneGame/
// OppCppChain/RunLegFibers 驱动语义的游戏无关化）。
#include "inferfarm/farm.h"
#include "inferfarm/backend_factory.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#endif

namespace inferfarm {

// ---------------- 后端注册面 ----------------
static InferBackend* MakeBackend(const std::string& name) {
    if (name == "cpu") return CreateCpuBackend();
    if (name == "ort") return CreateOrtBackend();
    if (name == "trt") return CreateTrtBackend();
    return nullptr;
}

// ---------------- env 覆盖 ----------------
static int EnvInt(const char* key, int def) {
    const char* e = getenv(key);
    return e && *e ? atoi(e) : def;
}
static double EnvDouble(const char* key, double def) {
    const char* e = getenv(key);
    return e && *e ? atof(e) : def;
}

bool Farm::Init(FarmConfig cfg) {
    cfg_ = cfg;
    // FARM_* env 覆盖（快速实验通道；显式 Config 为准，env 只在未显式动过时
    // 覆盖——此处取"env 优先"与 YGO 产线一致：便于不改码扫参）
    if (const char* e = getenv("FARM_FIBERS")) cfg_.fibers = atoi(e) == 1;
    if (const char* e = getenv("FARM_CENSUS")) cfg_.census = atoi(e) == 1;
    cfg_.workers = EnvInt("FARM_FIBER_WORKERS", cfg_.workers);
    cfg_.banks = EnvInt("FARM_BANKS", cfg_.banks);
    cfg_.window_floor = EnvDouble("FARM_BANK_WINDOW_FLOOR", cfg_.window_floor);
    cfg_.stagger_ms = EnvDouble("FARM_STAGGER_MS", cfg_.stagger_ms);
    cfg_.cache_log2 = EnvInt("FARM_CACHE_LOG2", cfg_.cache_log2);

#ifdef _WIN32
    timeBeginPeriod(1);   // 窗的真相：不开=定时量子 15.6ms（银行窗 ms 级全废）
    timer_armed_ = true;
#endif
    census_.on = cfg_.census;
    cache_.Init(cfg_.cache_log2);   // 推理缓存（0=关零行为差）
    // 配置校验（除零/巨分配防线）：误配 fail fast 而非崩溃
    if (cfg_.chains < 1 || cfg_.chains > 4096 || cfg_.games < 1 || cfg_.games > 100000000
        || cfg_.slots < 1 || cfg_.slots > 1024 || cfg_.banks < 0 || cfg_.banks > 32
        || cfg_.workers < 0 || cfg_.workers > 512
        || cfg_.cache_log2 < 0 || cfg_.cache_log2 > 24
        || !(cfg_.window_ms > 0) || !(cfg_.stagger_ms >= 0)
        || cfg_.max_decisions < 1) {
        std::fprintf(stderr, "[farm] 配置非法: chains=%d games=%d slots=%d banks=%d "
                     "workers=%d window=%.3f stagger=%.3f max_decisions=%lld cache_log2=%d"
                     "（界: chains[1,4096] games[1,1e8] slots[1,1024] banks[0,32] "
                     "workers[0,512] window>0 stagger>=0 cache_log2[0,24]）\n",
                     cfg_.chains, cfg_.games, cfg_.slots, cfg_.banks, cfg_.workers,
                     cfg_.window_ms, cfg_.stagger_ms, cfg_.max_decisions, cfg_.cache_log2);
        return false;
    }
    backend_ = MakeBackend(cfg_.model.backend);
    if (!backend_) {
        std::fprintf(stderr, "[farm] 未知后端: %s（cpu|ort|trt）\n", cfg_.model.backend.c_str());
        return false;
    }
    if (!backend_->LoadSpec(cfg_.model, cfg_.slots, spec_)) {
        std::fprintf(stderr, "[farm] LoadSpec 失败（backend=%s）\n", cfg_.model.backend.c_str());
        return false;
    }
    if (spec_.slots != cfg_.slots) {
        std::fprintf(stderr, "[farm] 模型批形状 dim0=%d ≠ slots=%d\n", spec_.slots, cfg_.slots);
        return false;
    }
    spec_ok_ = true;
    // init 期一次性换心：在 context/图创建**之前**（权重设备内存先落定）
    if (!cfg_.model.refit_weights.empty()
        && !RefitWeights(cfg_.model.refit_weights.c_str())) {
        std::fprintf(stderr, "[farm] init 期换心失败: %s\n",
                     cfg_.model.refit_weights.c_str());
        return false;
    }
    if (cfg_.banks > 0) {
        BankConfig bc;
        bc.banks = cfg_.banks;
        bc.slots = cfg_.slots;
        bc.window_ms = cfg_.window_ms;
        bc.window_floor = cfg_.window_floor;
        bank_obj_.Bind(*backend_, &census_);
        if (!bank_obj_.Init(bc, cfg_.model, &spec_)) {
            std::fprintf(stderr, "[farm] 银行制启动失败\n");
            return false;
        }
        bank_ = &bank_obj_;
    } else if (!inline_.Init(*backend_, cfg_.model, spec_)) {
        std::fprintf(stderr, "[farm] inline 会话启动失败\n");
        return false;
    }
    pool_.Configure(cfg_.workers, &census_);
    return true;
}

void Farm::Shutdown() {
    if (bank_) { bank_->Shutdown(); bank_ = nullptr; }
#ifdef _WIN32
    if (timer_armed_) { timeEndPeriod(1); timer_armed_ = false; }
#endif
    inline_.Shutdown();
    if (backend_) { delete backend_; backend_ = nullptr; }
    spec_ok_ = false;
}

void Farm::NoteGameDone(bool we_first, int outcome, long long dec, bool infer_fail,
                         long long fingerprint, int chain_id, int game_id) {
    std::lock_guard<std::mutex> lk(tally_mx_);
    if (we_first) { tally_.first_total++; if (outcome == 1) tally_.first_wins++; }
    else          { tally_.second_total++; if (outcome == 1) tally_.second_wins++; }
    tally_.games_done++;
    tally_.decisions += dec;
    if (infer_fail) tally_.infer_fails++;
    unsigned long long fp = (unsigned long long)fingerprint;
    // 位混淆 + 掺局身份（链/局号——完成序无关！并发下两腿完成序可不同）：
    // 防同结局成对抵消（偶数局全同 XOR=0 的教训），同时保 XOR 顺序无关性
    fp = (fp + (unsigned long long)(uint32_t)chain_id * 0xD1B54A32D192ED03ull
             + (unsigned long long)(uint32_t)game_id * 0xC2B2AE3D27D4EB4Full)
         * 0x9E3779B97F4A7C15ull;
    tally_.fingerprint ^= fp;
}

// ---------------- 驱动环 ----------------
// 组装行字节 → 缓存键。槽独占期内（Claim 后 Submit/Abandon 前）调用：本槽
// 行不可能被他人触碰（游标串行发号），读的是本决策刚组装的最终字节
// （含 Claim 清零后未写区的零——零基组装契约的一部分）。
CacheKey128 Farm::HashSlot(int bk, int sl) {
    CacheHasher h;
    for (size_t i = 0; i < spec_.ins.size(); i++) {
        size_t rb = 0;
        void* row = bank_->InputRow(bk, sl, spec_.ins[i].name.c_str(), &rb);
        if (row && rb) h.Update(row, rb);
    }
    return h.Finalize();
}

bool Farm::DriveDecision(GameAdapter* g) {
    if (bank_) {
        OutputDest dests[8];
        int nd = g->CollectOutputs(dests, 8);
        int bk = -1, sl = -1;
        if (!bank_->Claim(bk, sl)) return false;
        // 组装直写槽（GameAdapter 契约 1：此处无挂起点——drain 有界的前提）
        struct BankWriter : SlotWriter {
            BankScheduler* bank = nullptr;
            int b = 0, s = 0;
            void* Row(const char* name, size_t* row_bytes) override {
                return bank->InputRow(b, s, name, row_bytes);
            }
        } w;
        w.bank = bank_;
        w.b = bk;
        w.s = sl;
        g->AssembleInto(w);
        // 推理缓存（判决13）：键=本槽全输入行字节+权重代次。命中=Abandon
        // 弃槽（协议原生路径：作废槽+完工照减+发车跳过）+逐字节回放 dests；
        // 未命中=正常 SubmitWait，收割后从 dests 采录（n 全宽重放语义）。
        // 布局门：条目输出面（名字+n 逐位）与本次申报不符=视同未命中——
        // 同行字节不同申报面的适配器不共享条目，正确性无条件保住。
        if (cache_.on()) {
            CacheKey128 key = HashSlot(bk, sl);
            std::shared_ptr<const CachedResult> hit = cache_.Lookup(key, infer_gen_);
            if (hit && (int)hit->outs.size() == nd) {
                bool layout_ok = true;
                for (int i = 0; i < nd && layout_ok; i++)
                    if (!dests[i].name || hit->outs[(size_t)i].name != dests[i].name
                        || hit->outs[(size_t)i].n != dests[i].n)
                        layout_ok = false;
                if (layout_ok) {
                    bank_->Abandon(bk, sl);
                    for (int i = 0; i < nd; i++)
                        if (dests[i].dst && hit->outs[(size_t)i].n > 0)
                            std::memcpy(dests[i].dst, hit->outs[(size_t)i].vals.data(),
                                        sizeof(float) * (size_t)hit->outs[(size_t)i].n);
                    return true;
                }
            }
            bool ok = bank_->SubmitWait(bk, sl, dests, nd);
            if (ok) {
                std::vector<CachedOut> outs;
                outs.reserve((size_t)nd);
                for (int i = 0; i < nd; i++)
                    if (dests[i].name) {
                        CachedOut co;
                        co.name = dests[i].name;
                        co.n = dests[i].n;
                        if (dests[i].dst && co.n > 0) {
                            co.vals.resize((size_t)co.n);
                            std::memcpy(co.vals.data(), dests[i].dst,
                                        sizeof(float) * (size_t)co.n);
                        }
                        outs.push_back(std::move(co));
                    }
                cache_.Insert(key, infer_gen_, std::move(outs));
            }
            return ok;
        }
        return bank_->SubmitWait(bk, sl, dests, nd);
    }
    return inline_.Run(g);
}

void Farm::DriveGame(GameAdapter* g, uint64_t seed, bool we_first,
                      int chain_id, int game_id) {
    g->NewGame(seed, we_first);
    long long dec = 0;
    bool infer_fail = false;
    for (long long guard = 0; guard < cfg_.max_decisions; guard++) {
        if (g->IsDone()) break;
        if (!g->AdvanceToDecision()) break;
        dec++;
        if (!DriveDecision(g)) {
            g->OnInferFail();   // 判负纪律：不静默重试（会撕裂确定性）
            infer_fail = true;
            break;
        }
        g->ApplyResult();
    }
    NoteGameDone(g->WeAreFirst(), g->Outcome(), dec, infer_fail,
                 infer_fail ? -1 : g->GameFingerprint(), chain_id, game_id);
}

// ---------------- 腿 ----------------
struct FarmGameCtx {
    Farm* farm;
    AdapterFactory make;
    void* user;
    uint32_t seed0;
    int per;
    std::vector<GameAdapter*> adapters;   // 链号→适配器（腿末 delete）
};

static void FarmGameMain(int chain, int game, void* p) {
    FarmGameCtx* c = (FarmGameCtx*)p;
    // 种子协议（YGO 产线逐位同公式）：链 c 局 i 种子 = seed0 + c*per + i
    uint64_t seed = (uint64_t)(c->seed0 + (uint64_t)((uint64_t)chain * (uint64_t)c->per + (uint64_t)game));
    bool we_first = ((uint32_t)game % 2 == 0);   // 逐局交替（偶数局我方先攻）
    GameAdapter* g = c->adapters[(size_t)chain];
    c->farm->DriveGame(g, seed, we_first, chain, game);
}

static ITlsFrame* FarmFrameFactory(int chain, void* p) {
    FarmGameCtx* c = (FarmGameCtx*)p;
    return c->adapters[(size_t)chain]->TlsFrame();
}

double Farm::RunLeg(AdapterFactory make, void* user) {
    tally_ = FarmTally{};   // 腿清零（refit-jobs 形态：每作业一腿）
    const unsigned long long lu0 = cache_.lookups(), hi0 = cache_.hits();
    int per = (cfg_.games + cfg_.chains - 1) / cfg_.chains;
    FarmGameCtx ctx;
    ctx.farm = this;
    ctx.make = make;
    ctx.user = user;
    ctx.seed0 = cfg_.seed0;
    ctx.per = per;
    ctx.adapters.resize((size_t)cfg_.chains);
    for (int c = 0; c < cfg_.chains; c++) ctx.adapters[(size_t)c] = make(c, user);
    auto t0 = std::chrono::steady_clock::now();
    if (census_.on) census_.StartPrinter(t0);
    std::printf("[%s] 腿起跑: %d 链 %d 局（每链 %d），backend=%s%s，seed0=%u%s\n",
                cfg_.name.c_str(), cfg_.chains, cfg_.games, per,
                cfg_.model.backend.c_str(),
                bank_ ? "（银行制）" : "（inline）",
                cfg_.seed0, cfg_.fibers ? "，fiber 唤醒队列" : "，线程模式");
    std::fflush(stdout);
    double sec;
    if (cfg_.fibers)
        sec = pool_.RunLeg(cfg_.chains, per, FarmGameMain, FarmFrameFactory, &ctx,
                           cfg_.stagger_ms);
    else
        sec = RunLegThreads(cfg_.chains, per, FarmGameMain, FarmFrameFactory, &ctx,
                            cfg_.stagger_ms);
    // census 收尾：线程普查须在工人 join 前（RunLeg 已 join——普查退化为
    // 调度台/驱动面；YGO 产线在 join 前调，此处保接口可用性）
    if (bank_ && census_.on) census_.DumpThreads();
    if (census_.on) census_.StopPrinter();
    for (auto* a : ctx.adapters) delete a;
    // 腿汇总（[wb-bench] 同款语义）
    tally_.cache_lookups = cache_.lookups() - lu0;
    tally_.cache_hits = cache_.hits() - hi0;
    const FarmTally& t = tally_;
    int total = t.first_total + t.second_total;
    int wins = t.first_wins + t.second_wins;
    std::printf("[%s] 汇总: 先手 %d/%d, 后手 %d/%d, 综合 %d/%d (%.1f%%)，决策 %lld，"
                "推理故障局 %d\n",
                cfg_.name.c_str(), t.first_wins, t.first_total, t.second_wins,
                t.second_total, wins, total, total ? wins * 100.0 / total : 0.0,
                t.decisions, t.infer_fails);
    if (t.cache_lookups > 0)
        std::printf("[%s] 缓存: 查 %llu 命中 %llu (%.1f%%)\n", cfg_.name.c_str(),
                    t.cache_lookups, t.cache_hits,
                    t.cache_lookups ? t.cache_hits * 100.0 / t.cache_lookups : 0.0);
    std::printf("[%s] 全链结束: %d 局用时 %.2fs（%.2f 局/秒）\n",
                cfg_.name.c_str(), total, sec, sec > 0 ? total / sec : 0.0);
    std::fflush(stdout);
    return sec;
}

} // namespace inferfarm
