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

// 组间模型结构一致性（多设备契约）：同名输入同 dtype 同行宽、同名输出同宽
// （slots 已另行核对）。结构不一致=银行行协议/适配器契约撕裂，fail fast。
static bool SpecStructurallyEqual(const ModelSpec& a, const ModelSpec& b) {
    if (a.ins.size() != b.ins.size() || a.outs.size() != b.outs.size()) return false;
    for (size_t i = 0; i < a.ins.size(); i++) {
        if (a.ins[i].name != b.ins[i].name || a.ins[i].et != b.ins[i].et
            || a.ins[i].row_bytes != b.ins[i].row_bytes
            || a.ins[i].dims.size() != b.ins[i].dims.size()) return false;
        for (size_t d = 1; d < a.ins[i].dims.size(); d++)
            if (a.ins[i].dims[d] != b.ins[i].dims[d]) return false;
    }
    for (size_t j = 0; j < a.outs.size(); j++)
        if (a.outs[j].name != b.outs[j].name || a.outs[j].width != b.outs[j].width)
            return false;
    return true;
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
    // cpu 路由模式便利：pop_p>0 而未点名 population_input → 缺省 "pop"
    // （须在设备组展开拷贝 model 之前）
    if (cfg_.model.population_input.empty() && cfg_.model.backend == "cpu"
        && cfg_.model.cpu.pop_p > 0)
        cfg_.model.population_input = "pop";
    // 多权重模式（判决16）：models>0 = 一等多权重负载——链数=P×每模型局数
    // （每链 1 局全并发，喂网格满度）、链 c→模型 c%P（DriveGame 喂 SetModelId）
    if (cfg_.population.models > 0) {
        if (cfg_.population.models > 1000000
            || cfg_.population.games_each < 1 || cfg_.population.games_each > 1000000) {
            std::fprintf(stderr, "[farm] population 配置非法（models=%d games_each=%d）\n",
                         cfg_.population.models, cfg_.population.games_each);
            return false;
        }
        if (cfg_.model.population_input.empty()) {
            std::fprintf(stderr, "[farm] population 模式（多权重）须 population_input（路由图）"
                         "或 cpu 后端 pop_p>0\n");
            return false;
        }
        long long chains = (long long)cfg_.population.models
                           * cfg_.population.games_each;
        if (chains > 4096) {
            std::fprintf(stderr, "[farm] population models×games_each=%lld > 4096（链上限；"
                         "减小每模型局数或分波）\n", chains);
            return false;
        }
        cfg_.chains = (int)chains;
        cfg_.games = (int)chains;   // 每链 1 局（全并发=网格满度的前提）
    }
    // ---- 设备组展开（空=单设备老行为=cfg.model+cfg.banks）----
    std::vector<DeviceConfig> devs = cfg_.devices;
    if (devs.empty()) {
        DeviceConfig d;
        d.model = cfg_.model;
        d.banks = cfg_.banks;
        devs.push_back(d);
    }
    if (devs.size() > 8) {
        std::fprintf(stderr, "[farm] 设备组数 %zu > 8\n", devs.size());
        return false;
    }
    int total_banks = 0;
    for (auto& d : devs) {
        if (d.banks < 0 || d.banks > 32) {
            std::fprintf(stderr, "[farm] 设备组 banks=%d ∉ [0,32]\n", d.banks);
            return false;
        }
        if (d.slots < 0 || d.slots > 1024) {
            std::fprintf(stderr, "[farm] 设备组 slots=%d ∉ [0,1024]\n", d.slots);
            return false;
        }
        total_banks += d.banks;
    }
    if (total_banks > 32) {
        std::fprintf(stderr, "[farm] 跨设备组总银行数 %d > 32\n", total_banks);
        return false;
    }
    // ---- 每组建后端 + LoadSpec + 组间结构核对 ----
    double share_sum = 0;
    for (auto& d : devs) {
        if (d.share < 0) {
            std::fprintf(stderr, "[farm] 设备组 share=%.2f 非法（≥0；且不可全 0）\n",
                         d.share);
            return false;
        }
        share_sum += d.share;
    }
    if (share_sum <= 0) {
        std::fprintf(stderr, "[farm] 设备组 share 全 0——至少一组 >0\n");
        return false;
    }
    group_bes_.clear();
    group_specs_.clear();
    bool g0_deferred = false;   // 探测砍除：组 0 spec 由首个银行会话产出
    for (size_t gi = 0; gi < devs.size(); gi++) {
        InferBackend* be = MakeBackend(devs[gi].model.backend);
        if (!be) {
            std::fprintf(stderr, "[farm] 设备 %zu 未知后端: %s（cpu|ort|trt）\n",
                         gi, devs[gi].model.backend.c_str());
            for (auto* x : group_bes_) delete x;
            return false;
        }
        group_bes_.push_back(be);
        // 组形状提示：主组=cfg.slots；非主组可自带（异构小图，如核显 fb4）
        const int hint = (gi == 0 || devs[gi].slots <= 0) ? cfg_.slots
                                                          : devs[gi].slots;
        // 探测会话砍除（能力位 ProbeFreeSpec，ort 专属）：组 0 且银行制 →
        // LoadSpec 延后到首个银行会话顺带产出 spec（YGO 清单模式：探测
        // ≈0.1s/次 × 56 腿/代 ≈ 5.6s/代 纯探测税）。组 0 spec 占位
        // （ins 空=延迟标记）；组 1..N 照旧 probe，其结构对拍移到
        // InitGroups 后与组 0 实 spec 进行。
        if (gi == 0 && total_banks > 0 && be->ProbeFreeSpec()) {
            g0_deferred = true;
            ModelSpec placeholder;
            placeholder.slots = hint;
            placeholder.backend = devs[gi].model.backend;
            group_specs_.push_back(placeholder);
            continue;
        }
        ModelSpec s;
        if (!be->LoadSpec(devs[gi].model, hint, s)) {
            std::fprintf(stderr, "[farm] 设备 %zu LoadSpec 失败（backend=%s）\n",
                         gi, devs[gi].model.backend.c_str());
            for (auto* x : group_bes_) delete x;
            group_bes_.clear();
            return false;
        }
        if (s.slots != hint) {
            std::fprintf(stderr, "[farm] 设备 %zu 模型批形状 dim0=%d ≠ %d\n",
                         gi, s.slots, hint);
            for (auto* x : group_bes_) delete x;
            group_bes_.clear();
            return false;
        }
        if (gi == 0) {
            spec_ = s;
        } else if (!g0_deferred && !SpecStructurallyEqual(spec_, s)) {
            std::fprintf(stderr, "[farm] 设备 %zu 模型结构与设备 0 不一致"
                         "（输入名/行宽/dtype、输出名/宽须全同；dim0 可异"
                         "[异构批形状]）\n", gi);
            for (auto* x : group_bes_) delete x;
            group_bes_.clear();
            group_specs_.clear();
            return false;
        }
        group_specs_.push_back(s);
    }
    spec_ok_ = true;
    backend_ = group_bes_[0];
    n_dev_ = (int)devs.size();
    // population 路由校验（判决16）：须银行制（inline 单会话面未覆盖）+ 主组
    // spec 确有标记为 population 的输入（探测砍除通道下后半移到 InitGroups 后）
    if (!cfg_.model.population_input.empty()) {
        if (total_banks <= 0) {
            std::fprintf(stderr, "[farm] population 路由须银行制（banks>0）\n");
            return false;
        }
        if (!g0_deferred) {
            bool has_pop = false;
            for (auto& m : spec_.ins)
                if (m.population) { has_pop = true; break; }
            if (!has_pop) {
                std::fprintf(stderr, "[farm] population_input=\"%s\" 在模型输入中未找到"
                             "（cpu 后端=decl.pop_p>0 自动追加；ort=路由图导出）\n",
                             cfg_.model.population_input.c_str());
                return false;
            }
        }
    }
    // 链→组分配：平滑加权轮询（nginx 同款；确定性=链号函数，重跑逐位不破）。
    // 缺省全 share=1 ⇒ 两组时≡c%n_groups 老行为。share=0 ⇒ 该组不接链
    // （备用/测试位）。share 表存成员（SetLegShape 热调时重建分配）。
    {
        dev_shares_.clear();
        for (auto& d : devs) dev_shares_.push_back(d.share);
        BuildChainGroups();
    }
    // init 期一次性换心：多设备组不支持（半换心农场撕裂确定性）——fail fast
    if (!cfg_.model.refit_weights.empty()) {
        if (n_dev_ > 1) {
            std::fprintf(stderr, "[farm] init 期换心不支持多设备组（单组农场才可）\n");
            return false;
        }
        if (!RefitWeights(cfg_.model.refit_weights.c_str())) {
            std::fprintf(stderr, "[farm] init 期换心失败: %s\n",
                         cfg_.model.refit_weights.c_str());
            return false;
        }
    }
    if (total_banks > 0) {
        BankConfig bc;
        bc.banks = total_banks;
        bc.slots = cfg_.slots;
        bc.window_ms = cfg_.window_ms;
        bc.window_floor = cfg_.window_floor;
        bc.spin = EnvInt("FARM_BANK_SPIN", 0);            // 0 关/1 纯自旋
        bank_obj_.Bind(*backend_, &census_);   // 单组兼容面（primary=组 0）
        std::vector<BankGroupCfg> groups;
        for (size_t gi = 0; gi < devs.size(); gi++) {
            BankGroupCfg g;
            g.be = group_bes_[gi];
            g.model = devs[gi].model;
            g.banks = devs[gi].banks;
            g.spec = group_specs_[gi];   // 组规格（slots=组实际形状）
            g.slots = group_specs_[gi].slots;
            groups.push_back(g);
        }
        if (!bank_obj_.InitGroups(bc, groups, &spec_)) {
            std::fprintf(stderr, "[farm] 银行制启动失败\n");
            return false;
        }
        bank_ = &bank_obj_;
        if (g0_deferred) {
            // 探测砍除通道的后置校验（原 LoadSpec 期检查后移）：slots/组间
            // 结构/population——失败=显式关银行（Init 只建未跑，安全）再退
            bool bad = spec_.slots != cfg_.slots;
            for (size_t gi = 1; gi < group_specs_.size() && !bad; gi++)
                bad = !SpecStructurallyEqual(spec_, group_specs_[gi]);
            if (!bad && !cfg_.model.population_input.empty()) {
                bool has_pop = false;
                for (auto& m : spec_.ins)
                    if (m.population) { has_pop = true; break; }
                bad = !has_pop;
            }
            if (bad) {
                std::fprintf(stderr, "[farm] 探测砍除通道后置校验失败"
                             "（dim0=%d hint=%d）——关农场\n",
                             spec_.slots, cfg_.slots);
                // 真因定位（bad 三选一：slots 恒打印相同时也可能组间结构差；
                // 逐组打印首个结构差面的输入名/行宽/dims 深）
                for (size_t gi = 1; gi < group_specs_.size(); gi++) {
                    const ModelSpec& a = spec_;
                    const ModelSpec& b = group_specs_[gi];
                    if (SpecStructurallyEqual(a, b)) continue;
                    std::fprintf(stderr, "[farm]   组 %zu 结构差:\n", gi);
                    std::fprintf(stderr, "[farm]     a(组0会话) ins:");
                    for (auto& m : a.ins)
                        std::fprintf(stderr, " [%s rb=%zu d=%zu]",
                                     m.name.c_str(), m.row_bytes, m.dims.size());
                    std::fprintf(stderr, " outs:");
                    for (auto& m : a.outs)
                        std::fprintf(stderr, " [%s w=%zu]", m.name.c_str(), m.width);
                    std::fprintf(stderr, "\n[farm]     b(LoadSpec) ins:");
                    for (auto& m : b.ins)
                        std::fprintf(stderr, " [%s rb=%zu d=%zu]",
                                     m.name.c_str(), m.row_bytes, m.dims.size());
                    std::fprintf(stderr, " outs:");
                    for (auto& m : b.outs)
                        std::fprintf(stderr, " [%s w=%zu]", m.name.c_str(), m.width);
                    std::fprintf(stderr, "\n");
                }
                bank_obj_.Shutdown();
                bank_ = nullptr;
                Shutdown();
                return false;
            }
        }
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
    for (InferBackend* be : group_bes_) delete be;   // 含组 0（=backend_）
    group_bes_.clear();
    backend_ = nullptr;
    spec_ok_ = false;
}

// ---------------- 链→组分配（Init 与 SetLegShape 共用）----------------
void Farm::BuildChainGroups() {
    double share_sum = 0;
    for (double s : dev_shares_) share_sum += s;
    if (share_sum <= 0 || dev_shares_.empty()) {
        chain_grp_.assign((size_t)cfg_.chains, 0);
        return;
    }
    chain_grp_.assign((size_t)cfg_.chains, 0);
    std::vector<double> cw(dev_shares_.size(), 0.0);
    for (int c = 0; c < cfg_.chains; c++) {
        int best = -1;
        for (size_t gi = 0; gi < dev_shares_.size(); gi++) {
            cw[gi] += dev_shares_[gi];
            if (best < 0 || cw[gi] > cw[(size_t)best]) best = (int)gi;
        }
        cw[(size_t)best] -= share_sum;
        chain_grp_[(size_t)c] = best;
    }
}

// 腿形状热调（refit 清单/多形状驱动，2026-09-22 回接方需求）：腿间改
// chains/games/seed0 而**不重建银行池**（Shutdown/Init=建池+热身+探针的秒级
// 开销每作业付一次，恰是清单模式要消灭的）。前置条件=腿已返回（同
// RefitWeights 纪律）。物理面（slots/banks/devices/model/window）Init 烧死
// 不可动——会话/arena/图地址婚姻。population 模式下 chains=P×games_each
// 派生：拒绝改链（games 随链重置）。
bool Farm::SetLegShape(int chains, int games, uint32_t seed0) {
    if (!spec_ok_) return false;
    if (cfg_.population.models > 0) {
        if (chains != cfg_.chains) {
            std::fprintf(stderr, "[farm] SetLegShape: population 模式链数=P×games_each"
                         " 派生，不可改（当前 %d）\n", cfg_.chains);
            return false;
        }
    } else if (chains < 1 || chains > 4096) {
        std::fprintf(stderr, "[farm] SetLegShape: chains=%d ∉ [1,4096]\n", chains);
        return false;
    }
    if (games < 1 || games > 100000000) {
        std::fprintf(stderr, "[farm] SetLegShape: games=%d ∉ [1,1e8]\n", games);
        return false;
    }
    cfg_.chains = chains;
    cfg_.games = cfg_.population.models > 0 ? chains : games;   // population:每链 1 局
    cfg_.seed0 = seed0;
    BuildChainGroups();
    return true;
}

void Farm::NoteGameDone(bool we_first, int outcome, long long dec, bool infer_fail,
                         long long fingerprint, int chain_id, int game_id) {    std::lock_guard<std::mutex> lk(tally_mx_);
    if (we_first) { tally_.first_total++; if (outcome == 1) tally_.first_wins++; }
    else          { tally_.second_total++; if (outcome == 1) tally_.second_wins++; }
    tally_.games_done++;
    tally_.decisions += dec;
    if (infer_fail) tally_.infer_fails++;
    if (cfg_.population.models > 0) {   // 多权重按模型计数（结算语义归驱动）
        int mid = (int)((uint32_t)chain_id % (uint32_t)cfg_.population.models);
        if ((int)tally_.model_games.size() <= mid) {
            tally_.model_games.resize((size_t)mid + 1, 0);
            tally_.model_wins.resize((size_t)mid + 1, 0);
        }
        tally_.model_games[(size_t)mid]++;
        if (outcome == 1) tally_.model_wins[(size_t)mid]++;
    }
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
CacheKey128 Farm::HashSlot(int bk, int sl, int grp) {
    CacheHasher h;
    uint32_t gn = (uint32_t)grp * 0x1B873593u;   // 设备命名空间：异构组同字节
    h.Update(&gn, sizeof gn);                     // 行输出逐位可异，不共享条目
    for (size_t i = 0; i < spec_.ins.size(); i++) {
        if (spec_.ins[i].population) continue;   // pop 面不哈希（代次 gen 已管）
        size_t rb = 0;
        void* row = bank_->InputRow(bk, sl, spec_.ins[i].name.c_str(), &rb);
        if (row && rb) h.Update(row, rb);
    }
    return h.Finalize();
}

bool Farm::DriveDecision(GameAdapter* g, int grp) {
    if (bank_) {
        OutputDest dests[BankScheduler::kMaxOutputDests];
        int nd = g->CollectOutputs(dests, BankScheduler::kMaxOutputDests);
        if (nd > BankScheduler::kMaxOutputDests) {
            // 适配器违约（契约=返回条数 ≤cap）：截断防越界读，失败交判负纪律
            std::fprintf(stderr, "[farm] CollectOutputs 返回 %d > cap %d——截断"
                         "（适配器违约）\n", nd, BankScheduler::kMaxOutputDests);
            nd = BankScheduler::kMaxOutputDests;
        }
        int bk = -1, sl = -1;
        if (!bank_->Claim(bk, sl, grp)) return false;
        // 组装直写槽（GameAdapter 契约 1：此处无挂起点——drain 有界的前提；
        // ScopedNoSuspend=debug 断言把契约变成机器校验）
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
        {
            ScopedNoSuspend ns;
            g->AssembleInto(w);
        }
        // 推理缓存（判决13）：键=本槽全输入行字节+权重代次。命中=Abandon
        // 弃槽（协议原生路径：作废槽+完工照减+发车跳过）+逐字节回放 dests；
        // 未命中=正常 SubmitWait，收割后从 dests 采录（n 全宽重放语义）。
        // 布局门：条目输出面（名字+n 逐位）与本次申报不符=视同未命中——
        // 同行字节不同申报面的适配器不共享条目，正确性无条件保住。
        if (cache_.on()) {
            CacheKey128 key = HashSlot(bk, sl, grp);
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
    if (cfg_.population.models > 0)   // 链→模型映射（框架喂，适配器免工厂闭包）
        g->SetModelId((int64_t)((uint32_t)chain_id
                                % (uint32_t)cfg_.population.models));
    const int grp = chain_grp_.empty() ? 0
        : chain_grp_[(size_t)((uint32_t)chain_id % (uint32_t)chain_grp_.size())];
                                          // 链→组（加权轮询表；表长=chains）
    long long dec = 0;
    bool infer_fail = false;
    for (long long guard = 0; guard < cfg_.max_decisions; guard++) {
        if (g->IsDone()) break;
        // 契约 1 的机器校验只包 AssembleInto（Claim→Submit 在途写手窗口，
        // drain 有界的真正对象）——advance 不包：legacy monolith 形态（整局
        // 跑在 AdvanceToDecision 里，如 YGO 回接）合法地在 advance 内
        // Claim/SubmitWait=挂起发生在槽已提交之后，对 drain 无害（2026-09-24
        // YGO 腿首跑实锤误伤，此前断言覆盖面写宽了）。
        if (!g->AdvanceToDecision()) break;
        dec++;
        if (!DriveDecision(g, grp)) {
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
    if (n_dev_ > 1) {
        std::vector<DeviceConfig> devs = cfg_.devices.empty()
            ? std::vector<DeviceConfig>{} : cfg_.devices;
        if (devs.empty()) {
            DeviceConfig d;
            d.model = cfg_.model;
            d.banks = cfg_.banks;
            devs.push_back(d);
        }
        for (size_t gi = 0; gi < devs.size(); gi++) {
            int nch = 0;
            for (int c = 0; c < cfg_.chains; c++)
                if (chain_grp_[(size_t)c] == (int)gi) nch++;
            std::printf("[%s] 设备组 %zu: %s ep=%s dev=%d ×%d 银行 share=%.2f → %d 链\n",
                        cfg_.name.c_str(), gi, devs[gi].model.backend.c_str(),
                        devs[gi].model.ort_ep.c_str(), devs[gi].model.device_id,
                        devs[gi].banks, devs[gi].share, nch);
        }
    }
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
    if (cfg_.population.models > 0 && !t.model_games.empty()) {
        int bw = -1, bp_ = -1;
        for (size_t i = 0; i < t.model_games.size(); i++) {
            if (t.model_games[i] <= 0) continue;
            if (bw < 0 || t.model_wins[i] > t.model_wins[(size_t)bw]) bw = (int)i;
            if (bp_ < 0 || t.model_wins[i] < t.model_wins[(size_t)bp_]) bp_ = (int)i;
        }
        if (bw >= 0)
            std::printf("[%s] 多权重: %d 模型 × %d 局；按模型战绩（Outcome 契约计数，"
                        "语义归驱动） 最好 #%d %d/%d 最差 #%d %d/%d\n",
                        cfg_.name.c_str(), cfg_.population.models,
                        cfg_.population.games_each, bw, t.model_wins[(size_t)bw],
                        t.model_games[(size_t)bw], bp_, t.model_wins[(size_t)bp_],
                        t.model_games[(size_t)bp_]);
    }
    std::printf("[%s] 全链结束: %d 局用时 %.2fs（%.2f 局/秒）\n",
                cfg_.name.c_str(), total, sec, sec > 0 ? total / sec : 0.0);
    std::fflush(stdout);
    return sec;
}

} // namespace inferfarm
