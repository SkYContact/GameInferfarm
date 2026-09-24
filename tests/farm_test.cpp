// farm_test.cpp — 确定性门与协议冒烟（G1/G2/F 系 + census X=0 + refit 语义）。
//
// 门语义（承接 YGO 产线 G1/F1/F2）：
//  G1 逐位：银行路径 vs inline 路径，同种子 → 每局 outcome/决策数全同
//       （行独立 + 零基组装 ⇒ 设计自带性质；失败=有 bug）
//  G2 确定性：同配置连跑两次 → 全同
//  G3 fiber vs 线程模式（banks=2）：全同（TLS 帧纪律的行为级验证）
//  G4 census：X（失踪人口）恒 0、腿末 live=0、全状态归零
//  G5 refit：同 blob 两次换心 → 逐位同；不同 blob → 结果必变（A1/A2 的 CPU 版）
//  G7 推理缓存：键=组装行字节+权重代次；开=关逐位同；代次门；热缓存腿全同
//  G8a 多设备组（同构仿真）：分组=单组逐位同+重跑逐位同（真硬件=R4/gomoku --device）
//  G9 population 路由（演化，判决16）：均匀 pop=普通单模型腿逐位同；重跑同；
//     SetPopulation 换代必变；换代后缓存不串代（=新鲜无缓存农场逐位同）
//  G12 腿形状热调（回接方清单需求）：同农场 SetLegShape 续腿=新鲜农场同形状
//     逐位同；非法形状拒绝
//  G13 批次/位置不变性门（后端契约级，行独立的直接机器校验）：同一行内容
//     在批大小 n=1..满 与行位置变化下输出逐位同（G1 是游戏负载的
//     间接版；本门直接枚举。ort/trt 同协议=R5/gomoku_backend_test）
//  G14 Farm 代际重建逐位门（清单模式契约，2026-09-24 接入方提案）：同进程
//     建→跑→销毁 ×3 代同种子逐位同，代间搀异构形态腿弄脏静态；缓存开变体
//     另验（此前该契约只有 YGO --ort-jobs 侧对拍）
#include "../examples/toy/toy_adapter.h"
#include "../examples/gomoku/gomoku_adapter.h"
#include "inferfarm/backend.h"
#include "inferfarm/backend_factory.h"
#include "inferfarm/cache.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace inferfarm;
using namespace inferfarm::toy;

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s（%s:%d）\n", msg, __FILE__, __LINE__); g_fail++; } \
    else std::printf("ok: %s\n", msg); \
    std::fflush(stdout); \
} while (0)

// 一腿的逐局记录：链 c 局 i → (outcome, decisions)。收账走 tally 之外的
// 逐局探针——复用适配器自身记录（链级累计）+ tally 汇总即可对 G1；逐局
// 记录用第二个 Farm tally（先后手胜率）+ 每链 decisions。够 G1/G2/G3。
struct LegResult {
    int fw, ft, sw, st;      // 先/后手胜场与总局
    long long decisions;
    // census 腿末快照（G4）
    int live = 0, R = 0, Q = 0, W = 0;
    long long rev_n = 0;
    unsigned long long fp = 0;   // 逐局指纹 XOR
    unsigned long long clo = 0, chi = 0;   // 缓存查/命中（G7）
};

static LegResult RunOne(int banks, bool fibers, int workers, uint32_t seed0,
                         bool census = false, int cache_log2 = 0) {
    FarmConfig cfg;
    cfg.name = "test";
    cfg.chains = 6;
    cfg.games = 48;
    cfg.seed0 = seed0;
    cfg.fibers = fibers;
    cfg.workers = workers;
    cfg.banks = banks;
    cfg.slots = 8;
    cfg.window_ms = 0.2;
    cfg.stagger_ms = 1;
    cfg.census = census;
    cfg.cache_log2 = cache_log2;
    cfg.model.backend = "cpu";
    cfg.model.cpu = ToyModelDecl(cfg.slots);
    Farm farm;
    if (!farm.Init(cfg)) {
        std::printf("FATAL: farm init 失败（banks=%d fibers=%d）\n", banks, (int)fibers);
        g_fail++;
        return {0, 0, 0, 0, -1};
    }
    farm.RunLeg(MakeToyAdapter, nullptr);
    const FarmTally& t = farm.tally();
    LegResult r{t.first_wins, t.first_total, t.second_wins, t.second_total, t.decisions};
    Census* c = farm.census();
    r.live = c->live.load();
    r.R = c->state[2].load();
    r.Q = c->state[1].load();
    r.W = c->state[3].load();
    r.rev_n = c->rev_n.load();
    r.fp = t.fingerprint;
    r.clo = t.cache_lookups;
    r.chi = t.cache_hits;
    return r;
}

// RW1 blob 构造（CPU 后端 toy 协议："<out>.W<in>" f32 向量）
static std::vector<char> MakeRw1(const std::vector<std::pair<std::string, std::vector<float>>>& w) {
    std::vector<char> b;
    auto put32 = [&](uint32_t v) { for (int i = 0; i < 4; i++) b.push_back((char)(v >> (8 * i))); };
    auto put16 = [&](uint16_t v) { for (int i = 0; i < 2; i++) b.push_back((char)(v >> (8 * i))); };
    b.insert(b.end(), {'R', 'W', '1', '\0'});
    put32(1);
    put32((uint32_t)w.size());
    for (auto& e : w) {
        put16((uint16_t)e.first.size());
        b.insert(b.end(), e.first.begin(), e.first.end());
        b.push_back((char)2);   // f32
        put32((uint32_t)e.second.size());
        const char* p = (const char*)e.second.data();
        b.insert(b.end(), p, p + e.second.size() * 4);
    }
    return b;
}
static std::vector<std::pair<std::string, std::vector<float>>> ents_empty() {
    return {{"policy.Wobs", std::vector<float>(8, 0.1f)}};
}
static bool WriteFile(const char* path, const std::vector<char>& b) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fwrite(b.data(), 1, b.size(), f);
    fclose(f);
    return true;
}

int main() {
    std::printf("=== inferfarm 确定性门 ===\n");

    // G1：银行 vs inline（fiber 同参数）
    LegResult bank1 = RunOne(2, true, 4, 4242);
    LegResult inl = RunOne(0, true, 4, 4242);
    CHECK(bank1.decisions >= 0 && inl.decisions >= 0, "两腿均完成");
    CHECK(bank1.fw == inl.fw && bank1.sw == inl.sw && bank1.ft == inl.ft
          && bank1.st == inl.st && bank1.decisions == inl.decisions
          && bank1.fp == inl.fp,
          "G1 银行 vs inline 逐位一致（outcome+决策数+指纹）");

    // G2：同配置连跑两次
    LegResult bank2 = RunOne(2, true, 4, 4242);
    CHECK(bank1.fw == bank2.fw && bank1.sw == bank2.sw
          && bank1.decisions == bank2.decisions && bank1.fp == bank2.fp,
          "G2 同配置两腿全同（含指纹）");

    // G3：fiber vs 线程模式（银行同参数）——TLS 帧纪律行为级
    LegResult thr = RunOne(2, false, 6, 4242);
    CHECK(bank1.fw == thr.fw && bank1.sw == thr.sw
          && bank1.decisions == thr.decisions && bank1.fp == thr.fp,
          "G3 fiber vs 线程模式全同（帧纪律，含指纹）");

    // 不同种子必须不同（防"确定性=恒同输出"的假绿）
    LegResult alt = RunOne(2, true, 4, 777);
    CHECK(alt.decisions == bank1.decisions, "异种子腿完成（决策数=局×回合 恒定）");
    bool alt_differs = (alt.fw != bank1.fw) || (alt.sw != bank1.sw)
        || alt.fp != bank1.fp;
    CHECK(alt_differs, "异种子结果不同（种子协议有效性，含指纹）");

    // G4：census X=0 + 腿末归零（读 Farm 自己的 census 对象）
    {
        LegResult cs = RunOne(2, true, 4, 4242, /*census=*/true);
        CHECK(cs.decisions == bank1.decisions, "G4 census 开=结果逐位同（~5% 税不改行为）");
        int X = cs.live - cs.R - cs.Q - cs.W;
        CHECK(cs.live == 0 && cs.R == 0 && cs.Q == 0 && cs.W == 0 && X == 0,
              "G4 census 腿末：live=R=Q=W=0（X 恒 0 不变量）");
        CHECK(cs.rev_n > 0, "G4 复活路径有样本（唤醒队列在跑）");
    }

    // G5：refit 语义（CPU 后端）
    {
        // 逐行向量（.k 定向）：常量向量会把动作区分度抹平——必须逐 k 不同。
        // diff=全项取反（tanh 单调 ⇒ 动作排行整体倒置 ⇒ argmax 必翻）
        std::vector<std::pair<std::string, std::vector<float>>> ents;
        for (int k = 0; k < kActN; k++) {
            std::vector<float> wo(8), wm(4), wc(1, 0.3f);
            for (int e = 0; e < 8; e++) wo[(size_t)e] = 0.25f * (float)(k + 1) * ((e % 2) ? -1 : 1);
            for (int e = 0; e < 4; e++) wm[(size_t)e] = 0.2f * (float)(k + 2);
            char n1[64], n2[64], n3[64];
            snprintf(n1, sizeof n1, "policy.Wobs.%d", k);
            snprintf(n2, sizeof n2, "policy.Wmask.%d", k);
            snprintf(n3, sizeof n3, "policy.Wcodes.%d", k);
            ents.push_back({n1, wo});
            ents.push_back({n2, wm});
            ents.push_back({n3, wc});
        }
        WriteFile("refit_same_a.rw1", MakeRw1(ents));
        WriteFile("refit_same_b.rw1", MakeRw1(ents));
        for (auto& e : ents)
            for (auto& v : e.second) v = -v;
        WriteFile("refit_diff.rw1", MakeRw1(ents));

        auto run_with_refit = [&](const char* blob) {
            FarmConfig cfg;
            cfg.name = "refit";
            cfg.chains = 4;
            cfg.games = 24;
            cfg.seed0 = 4242;
            cfg.banks = 2;
            cfg.slots = 8;
            cfg.workers = 4;
            cfg.model.backend = "cpu";
            cfg.model.cpu = ToyModelDecl(cfg.slots);
            Farm farm;
            if (!farm.Init(cfg)) { g_fail++; return LegResult{0, 0, 0, 0, -1}; }
            if (!farm.RefitWeights(blob)) { g_fail++; return LegResult{0, 0, 0, 0, -2}; }
            farm.RunLeg(MakeToyAdapter, nullptr);
            const FarmTally& t = farm.tally();
            LegResult r{t.first_wins, t.first_total, t.second_wins, t.second_total,
                        t.decisions};
            r.fp = t.fingerprint;   // 指纹必须回填（漏填=异 blob 门空比较的教训）
            return r;
        };
        LegResult ra = run_with_refit("refit_same_a.rw1");
        LegResult rb = run_with_refit("refit_same_b.rw1");
        LegResult rd = run_with_refit("refit_diff.rw1");
        CHECK(ra.decisions > 0, "G5 refit 腿完成");
        CHECK(ra.fw == rb.fw && ra.sw == rb.sw && ra.decisions == rb.decisions
              && ra.fp == rb.fp,
              "G5 同 blob 两次换心=逐位同（含指纹）");
        CHECK(ra.fp != rd.fp,
              "G5 异 blob 必变（换心生效，指纹级）");

        // RW1 解析负路径（下面继续）
        // G6：五子棋范例——银行 vs inline 逐位（真实棋类的接缝验证）
        {
            auto gomoku_leg = [&](int banks) {
                FarmConfig cfg;
                cfg.name = "gomoku";
                cfg.chains = 4;
                cfg.games = 8;
                cfg.seed0 = 20260922u;
                cfg.banks = banks;
                cfg.slots = 8;
                cfg.workers = 4;
                cfg.stagger_ms = 1;
                cfg.model.backend = "cpu";
                cfg.model.cpu = gomoku::GomokuModelDecl(cfg.slots);
                Farm farm;
                if (!farm.Init(cfg)) { g_fail++; return LegResult{0, 0, 0, 0, -1}; }
                farm.RunLeg(gomoku::MakeGomokuAdapter, nullptr);
                const FarmTally& t = farm.tally();
                LegResult r{t.first_wins, t.first_total, t.second_wins, t.second_total,
                            t.decisions};
                r.fp = t.fingerprint;
                return r;
            };
            LegResult gb = gomoku_leg(2);
            LegResult gi = gomoku_leg(0);
            CHECK(gb.decisions > 0, "G6 五子棋腿完成（决策 > 0）");
            CHECK(gb.fw == gi.fw && gb.sw == gi.sw && gb.decisions == gi.decisions
                  && gb.fp == gi.fp,
                  "G6 五子棋银行 vs inline 逐位一致（含指纹）");
        }

        // RW1 解析负路径：坏 magic
        std::vector<char> bad = MakeRw1(ents);
        bad[0] = 'X';
        WriteFile("refit_bad.rw1", bad);
        std::vector<char> blob;
        std::vector<Rw1Entry> es;
        CHECK(!ParseRw1("refit_bad.rw1", blob, es), "G5 坏 magic fail fast");
        CHECK(!ParseRw1("no_such_file.rw1", blob, es), "G5 缺文件 fail fast");
        {
            // 截断（半条目）与 dtype 非法两条负路径
            std::vector<char> full = MakeRw1({ents_empty()});
            WriteFile("refit_trunc.rw1", std::vector<char>(full.begin(), full.begin() + full.size() / 2));
            CHECK(!ParseRw1("refit_trunc.rw1", blob, es), "G5 截断 blob fail fast");
            std::vector<char> bad_dt = full;
            size_t dt_off = 14 + std::string("policy.Wobs").size();   // 头12B+名长2B+名
            bad_dt[dt_off] = (char)9;   // 首条目 dtype 字段改非法值
            WriteFile("refit_dtype.rw1", bad_dt);
            CHECK(!ParseRw1("refit_dtype.rw1", blob, es), "G5 dtype 非法 fail fast");
        }
    }

    // G7：推理缓存（KataGo NNCacheTable 思想吸收，判决13）——键=组装行字节
    // 哈希+权重代次；命中=Abandon 弃槽+逐字节回放 dests。
    {
        // 单元门：命中回放 / 空表 / 代次门 / 同键新代覆盖
        {
            InferCache c;
            c.Init(4);
            CacheHasher h;
            h.Update("abc", 3);
            CacheKey128 k = h.Finalize();
            CHECK(!c.Lookup(k, 1), "G7 单元：空表未命中");
            c.Insert(k, 1, {{"policy", 2, {1.f, 2.f}}});
            auto p = c.Lookup(k, 1);
            CHECK(p && p->outs.size() == 1 && p->outs[0].name == "policy"
                      && p->outs[0].vals.size() == 2 && p->outs[0].vals[1] == 2.f,
                  "G7 单元：命中逐字节回放");
            CHECK(!c.Lookup(k, 2), "G7 单元：代次不符=未命中（换心失效）");
            c.Insert(k, 2, {{"policy", 2, {9.f, 9.f}}});
            auto p2 = c.Lookup(k, 2);
            CHECK(p2 && p2->outs[0].vals[0] == 9.f, "G7 单元：同键新代覆盖旧代");
        }
        // 行为门 A：玩具（3 输入）缓存开=关逐位同
        LegResult toy_on = RunOne(2, true, 4, 4242, false, 12);
        CHECK(bank1.fw == toy_on.fw && bank1.sw == toy_on.sw
              && bank1.decisions == toy_on.decisions && bank1.fp == toy_on.fp,
              "G7 玩具（多输入）缓存开=关逐位同（含指纹）");
        CHECK(toy_on.clo > 0, "G7 玩具缓存有查询样本");
        // 行为门 B：五子棋（空盘首决策跨局重放=天然命中面）开=关逐位同+真命中；
        // 同农场二腿=热缓存全命中路径也逐位同
        LegResult g_off, g_hot;
        LegResult g_on;
        {
            FarmConfig cfg;
            cfg.name = "gomoku-cache";
            cfg.chains = 4;
            cfg.games = 8;
            cfg.seed0 = 20260922u;
            cfg.banks = 2;
            cfg.slots = 8;
            cfg.workers = 4;
            cfg.stagger_ms = 1;
            cfg.model.backend = "cpu";
            cfg.model.cpu = gomoku::GomokuModelDecl(cfg.slots);
            Farm f;
            CHECK(f.Init(cfg), "G7 五子棋缓存关农场起");
            f.RunLeg(gomoku::MakeGomokuAdapter, nullptr);
            g_off = {f.tally().first_wins, f.tally().first_total,
                     f.tally().second_wins, f.tally().second_total,
                     f.tally().decisions};
            g_off.fp = f.tally().fingerprint;
        }
        {
            FarmConfig cfg;
            cfg.name = "gomoku-cache";
            cfg.chains = 4;
            cfg.games = 8;
            cfg.seed0 = 20260922u;
            cfg.banks = 2;
            cfg.slots = 8;
            cfg.workers = 4;
            cfg.stagger_ms = 1;
            cfg.model.backend = "cpu";
            cfg.model.cpu = gomoku::GomokuModelDecl(cfg.slots);
            cfg.cache_log2 = 12;
            Farm f;
            CHECK(f.Init(cfg), "G7 五子棋缓存开农场起");
            f.RunLeg(gomoku::MakeGomokuAdapter, nullptr);
            const FarmTally& t1 = f.tally();
            g_on = {t1.first_wins, t1.first_total, t1.second_wins, t1.second_total,
                    t1.decisions};
            g_on.fp = t1.fingerprint;
            g_on.clo = t1.cache_lookups;
            g_on.chi = t1.cache_hits;
            // 二腿（同农场同种子）：全部状态已在缓存=纯命中路径
            f.RunLeg(gomoku::MakeGomokuAdapter, nullptr);
            const FarmTally& t2 = f.tally();
            g_hot = {t2.first_wins, t2.first_total, t2.second_wins, t2.second_total,
                     t2.decisions};
            g_hot.fp = t2.fingerprint;
            g_hot.clo = t2.cache_lookups;
            g_hot.chi = t2.cache_hits;
        }
        CHECK(g_on.decisions > 0 && g_off.decisions == g_on.decisions,
              "G7 五子棋缓存腿完成（决策数=缓存关）");
        CHECK(g_off.fw == g_on.fw && g_off.sw == g_on.sw && g_off.fp == g_on.fp,
              "G7 五子棋缓存开=关逐位同（含指纹）");
        CHECK(g_on.chi > 0, "G7 五子棋真实命中（空盘首决策跨局重放）");
        CHECK(g_hot.fp == g_on.fp && g_hot.decisions == g_on.decisions,
              "G7 热缓存二腿逐位同（纯命中路径）");
        CHECK(g_hot.chi > 0 && g_hot.chi * 10 >= g_hot.clo * 9,
              "G7 热缓存二腿命中率高（≥90%）");
    }

    // G8a：多设备组（同构仿真：两组同模型 cpu）——分组机器（组池/组轮转/
    // Claim 组门/链钉扎）行为级等价：分组腿=单组腿逐位同，且重跑逐位同。
    // （异构真硬件门=本地 R3/gomoku --device 演示；CI 无双卡=同构仿真已覆盖
    // 分组协议面。）
    {
        auto dev_leg = [&](int n_groups, uint32_t seed0) {
            FarmConfig cfg;
            cfg.name = "g8";
            cfg.chains = 4;
            cfg.games = 16;
            cfg.seed0 = seed0;
            cfg.banks = 2;
            cfg.slots = 8;
            cfg.workers = 4;
            cfg.stagger_ms = 1;
            cfg.model.backend = "cpu";
            cfg.model.cpu = gomoku::GomokuModelDecl(cfg.slots);
            if (n_groups == 2) {
                DeviceConfig a, b;
                a.model = cfg.model;
                a.banks = 1;
                b.model = cfg.model;
                b.banks = 1;
                cfg.devices = {a, b};
            }
            Farm farm;
            if (!farm.Init(cfg)) { g_fail++; return LegResult{0, 0, 0, 0, -1}; }
            farm.RunLeg(gomoku::MakeGomokuAdapter, nullptr);
            const FarmTally& t = farm.tally();
            LegResult r{t.first_wins, t.first_total, t.second_wins, t.second_total,
                        t.decisions};
            r.fp = t.fingerprint;
            return r;
        };
        LegResult s1 = dev_leg(1, 4242);
        LegResult p1 = dev_leg(2, 4242);
        LegResult p2 = dev_leg(2, 4242);
        CHECK(s1.decisions > 0 && p1.decisions == s1.decisions,
              "G8a 分组腿完成（决策数=单组）");
        CHECK(s1.fw == p1.fw && s1.sw == p1.sw && s1.fp == p1.fp,
              "G8a 同构分组=单组逐位同（含指纹；组池/组门/链钉扎行为级等价）");
        CHECK(p1.fp == p2.fp,
              "G8a 分组腿重跑逐位同（钉扎确定性）");
        // G8b：异构批形状（同构后端仿真）——组 0 fb8 + 组 1 fb4：行宽/输入名/
        // 输出宽同、dim0 异；行独立 ⇒ 行的值与批形无关 ⇒ 结果=单组逐位同。
        // （真硬件异构小图=R5/gomoku --device slots=；本门 CI 可跑。）
        {
            FarmConfig cfg;
            cfg.name = "g8b";
            cfg.chains = 4;
            cfg.games = 16;
            cfg.seed0 = 4242;
            cfg.banks = 2;
            cfg.slots = 8;
            cfg.workers = 4;
            cfg.stagger_ms = 1;
            cfg.model.backend = "cpu";
            cfg.model.cpu = gomoku::GomokuModelDecl(8);
            DeviceConfig a, b;
            a.model.backend = "cpu";
            a.model.cpu = gomoku::GomokuModelDecl(8);
            a.banks = 1;
            b.model.backend = "cpu";
            b.model.cpu = gomoku::GomokuModelDecl(4);   // 小批形（dim0 异）
            b.banks = 1;
            b.slots = 4;
            cfg.devices = {a, b};
            Farm farm;
            CHECK(farm.Init(cfg), "G8b 混形状农场起（fb8+fb4）");
            farm.RunLeg(gomoku::MakeGomokuAdapter, nullptr);
            const FarmTally& t = farm.tally();
            CHECK(t.decisions == s1.decisions && t.fingerprint == s1.fp,
                  "G8b 混批形状=单组逐位同（含指纹；游标/窗满/越界三界按组）");
        }
    }

    // G9：population 路由（演化，判决16）——mid 缺省 0（Claim 清零）⇒ 均匀 pop
    // （各行=同 flat 权重）时全部行用 pop[0] ⇒ 与普通单模型腿同权重同输出。
    {
        const int kH = 6, kP = 4;
        auto mk_decl = [&](int slots, int pop_p) {
            CpuModelDecl d = ToyModelDecl(slots);
            d.hidden = kH;
            d.pop_p = pop_p;
            return d;
        };
        auto plain_leg = [&](uint32_t wseed) {
            FarmConfig cfg;
            cfg.name = "g9p";
            cfg.chains = 4;
            cfg.games = 16;
            cfg.seed0 = 4242;
            cfg.banks = 2;
            cfg.slots = 8;
            cfg.workers = 4;
            cfg.stagger_ms = 1;
            cfg.model.backend = "cpu";
            cfg.model.cpu = mk_decl(cfg.slots, 0);
            cfg.model.cpu.weight_seed = wseed;
            Farm farm;
            if (!farm.Init(cfg)) { g_fail++; return LegResult{0, 0, 0, 0, -1}; }
            farm.RunLeg(MakeToyAdapter, nullptr);
            const FarmTally& t = farm.tally();
            LegResult r{t.first_wins, t.first_total, t.second_wins, t.second_total,
                        t.decisions};
            r.fp = t.fingerprint;
            return r;
        };
        auto routed_leg = [&](uint32_t wseed, int cache_log2) {
            FarmConfig cfg;
            cfg.name = "g9r";
            cfg.chains = 4;
            cfg.games = 16;
            cfg.seed0 = 4242;
            cfg.banks = 2;
            cfg.slots = 8;
            cfg.workers = 4;
            cfg.stagger_ms = 1;
            cfg.cache_log2 = cache_log2;
            cfg.model.backend = "cpu";
            cfg.model.cpu = mk_decl(cfg.slots, kP);
            Farm farm;
            if (!farm.Init(cfg)) { g_fail++; return LegResult{0, 0, 0, 0, -1}; }
            std::vector<float> flat = CpuBuildMlpFlat(cfg.model.cpu, wseed);
            std::vector<float> pop(flat.size() * (size_t)kP);
            for (int p = 0; p < kP; p++)
                memcpy(pop.data() + p * flat.size(), flat.data(), flat.size() * 4);
            if (!farm.SetPopulation(pop.data())) { g_fail++; return LegResult{0, 0, 0, 0, -2}; }
            farm.RunLeg(MakeToyAdapter, nullptr);
            const FarmTally& t = farm.tally();
            LegResult r{t.first_wins, t.first_total, t.second_wins, t.second_total,
                        t.decisions};
            r.fp = t.fingerprint;
            return r;
        };
        LegResult plain = plain_leg(4242u);
        LegResult rt1 = routed_leg(4242u, 0);
        LegResult rt2 = routed_leg(4242u, 0);
        CHECK(plain.decisions > 0 && rt1.decisions == plain.decisions,
              "G9 路由腿完成（决策数=普通腿）");
        CHECK(plain.fp == rt1.fp && plain.fw == rt1.fw && plain.sw == rt1.sw,
              "G9a 均匀 population=普通单模型腿逐位同（路由不扰动行数学）");
        CHECK(rt1.fp == rt2.fp, "G9b 路由腿重跑逐位同");
        LegResult rt3 = routed_leg(4242u, 12);
        CHECK(rt3.fp == rt1.fp, "G9 路由+缓存=逐位同（pop 面不哈希、代次管）");
        LegResult alt = routed_leg(12345u, 0);
        CHECK(alt.fp != rt1.fp, "G9c SetPopulation 换代必变（新权重生效）");
        // G9d：同农场连换两代（缓存开）——第二代结果=新鲜农场第二代逐位同
        //（代次失效端到端：旧代缓存条目不得串门）
        {
            FarmConfig cfg;
            cfg.name = "g9d";
            cfg.chains = 4;
            cfg.games = 16;
            cfg.seed0 = 4242;
            cfg.banks = 2;
            cfg.slots = 8;
            cfg.workers = 4;
            cfg.stagger_ms = 1;
            cfg.cache_log2 = 12;
            cfg.model.backend = "cpu";
            cfg.model.cpu = mk_decl(cfg.slots, kP);
            Farm farm;
            CHECK(farm.Init(cfg), "G9d 农场起");
            std::vector<float> fa = CpuBuildMlpFlat(cfg.model.cpu, 4242u);
            std::vector<float> fb = CpuBuildMlpFlat(cfg.model.cpu, 12345u);
            std::vector<float> pa(fa.size() * (size_t)kP), pb(fb.size() * (size_t)kP);
            for (int p = 0; p < kP; p++) {
                memcpy(pa.data() + p * fa.size(), fa.data(), fa.size() * 4);
                memcpy(pb.data() + p * fb.size(), fb.data(), fb.size() * 4);
            }
            CHECK(farm.SetPopulation(pa.data()), "G9d 第一代写入");
            farm.RunLeg(MakeToyAdapter, nullptr);
            unsigned long long fp1 = farm.tally().fingerprint;
            CHECK(farm.SetPopulation(pb.data()), "G9d 第二代写入");
            farm.RunLeg(MakeToyAdapter, nullptr);
            unsigned long long fp2 = farm.tally().fingerprint;
            CHECK(fp1 == rt1.fp && fp2 == alt.fp,
                  "G9d 同农场连换两代=各自新鲜农场逐位同（代次失效端到端）");
        }
    }

    // G12：腿形状热调（回接方清单需求）——同农场腿1(形状A)→SetLegShape(形状B)
    // →腿2 == 新鲜农场(形状B)逐位同。refit 清单形态的根基：换形状不重建银行。
    {
        auto g12_farm = []() {
            FarmConfig cfg;
            cfg.name = "g12";
            cfg.chains = 4;
            cfg.games = 16;
            cfg.seed0 = 4242;
            cfg.banks = 2;
            cfg.slots = 8;
            cfg.workers = 4;
            cfg.stagger_ms = 1;
            cfg.model.backend = "cpu";
            cfg.model.cpu = gomoku::GomokuModelDecl(cfg.slots);
            Farm* f = new Farm();
            if (!f->Init(cfg)) { g_fail++; return (Farm*)nullptr; }
            return f;
        };
        auto leg_fp = [](Farm* f, AdapterFactory make) {
            f->RunLeg(make, nullptr);
            return f->tally().fingerprint;
        };
        Farm* a = g12_farm();
        CHECK(a != nullptr, "G12 农场起");
        if (a) {
            (void)leg_fp(a, gomoku::MakeGomokuAdapter);   // 腿1：形状 A（4 链 16 局）
            CHECK(a->SetLegShape(6, 24, 777u), "G12 SetLegShape(6,24,777) 成功");
            unsigned long long fp_re = leg_fp(a, gomoku::MakeGomokuAdapter);
            CHECK(!a->SetLegShape(0, 24, 1u), "G12 非法 chains 拒绝");
            CHECK(!a->SetLegShape(6, 0, 1u), "G12 非法 games 拒绝");
            Farm* b = g12_farm();
            CHECK(b != nullptr, "G12 对照农场起");
            if (b) {
                CHECK(b->SetLegShape(6, 24, 777u), "G12 对照农场同形状热调");
                unsigned long long fp_fresh = leg_fp(b, gomoku::MakeGomokuAdapter);
                CHECK(fp_re == fp_fresh,
                      "G12 热调续腿=新鲜农场同形状逐位同（含指纹）");
                delete b;
            }
            delete a;
        }
    }

    // ---------------- G13：批次/位置不变性门（后端契约级）----------------
    // 行独立契约的直接机器校验（Thinking Machines "batch invariance" 语境，
    // 2026-09-24 调研吸收）：同一行内容在 批大小变化（n=1..满）与 行位置变化
    // （同内容换槽）下输出逐位同。cpu 后端进 CI；ort/trt 同协议=R5。
    {
        InferBackend* be = CreateCpuBackend();
        ModelConfig mcfg;
        mcfg.backend = "cpu";
        mcfg.cpu.slots = 8;
        mcfg.cpu.poly_k = 32;
        mcfg.cpu.hidden = 16;
        mcfg.cpu.ins.push_back({"in", DTYPE_F32, {32}});
        mcfg.cpu.outs.push_back({"out", 16});
        ModelSpec spec;
        CHECK(be && be->LoadSpec(mcfg, mcfg.cpu.slots, spec) && spec.slots == 8,
              "G13 cpu 后端起（slots=8）");
        if (be && spec.slots == 8) {
            void* sess = be->CreateSession(mcfg, spec, /*for_bank=*/true);
            bool warm = sess && be->Warmup(sess);
            CHECK(warm, "G13 会话建+热身");
            if (warm) {
                // 内容 k = 种子 k 的确定性逐位模式（每槽内容互异）
                const int S = spec.slots;
                const size_t RB = 32 * sizeof(float);
                std::vector<std::vector<char>> content((size_t)S);
                for (int k = 0; k < S; k++) {
                    content[(size_t)k].resize(RB);
                    uint32_t x = 0x9E3779B9u * (uint32_t)(k + 1);
                    for (size_t i = 0; i < RB / sizeof(float); i++) {
                        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                        float v = (float)((int)(x & 0xFFFF) - 32768) / 32768.0f;
                        std::memcpy(content[(size_t)k].data() + i * sizeof(float),
                                    &v, sizeof(float));
                    }
                }
                auto run = [&](const std::vector<int>& slot_content, int slot_read,
                               std::vector<float>& out) -> bool {
                    for (size_t r = 0; r < slot_content.size(); r++) {
                        size_t rb = 0;
                        void* row = be->InputRow(sess, "in", (int)r, &rb);
                        if (!row || rb != RB) return false;
                        std::memcpy(row, content[(size_t)slot_content[r]].data(), RB);
                    }
                    unsigned seq = 0;
                    if (!be->SubmitBatch(sess, (int)slot_content.size(), seq))
                        return false;
                    while (!be->CompletionReached(sess, seq)) {}
                    be->CompletionFence();
                    int w = be->OutputWidth(sess, "out");
                    const float* src = w > 0 ? be->OutputRow(sess, "out", slot_read)
                                             : nullptr;
                    if (!src) return false;
                    out.assign(src, src + w);
                    return true;
                };
                auto biteq = [](const std::vector<float>& a,
                                const std::vector<float>& b) {
                    return a.size() == b.size()
                        && std::memcmp(a.data(), b.data(),
                                       a.size() * sizeof(float)) == 0;
                };
                // 基线：满批，槽 r ← 内容 r
                std::vector<std::vector<float>> ref((size_t)S);
                std::vector<int> full;
                for (int r = 0; r < S; r++) full.push_back(r);
                bool base = true;
                for (int r = 0; r < S; r++)
                    base = run(full, r, ref[(size_t)r]) && base;
                CHECK(base, "G13 满批基线（8 行逐槽采输出）");
                // 批大小变化：n=1 / n=3（前缀行数变，目标行输出必须逐位同）
                std::vector<float> got;
                bool ok = run({0}, 0, got) && biteq(got, ref[0]);
                CHECK(ok, "G13a 批大小不变性：n=1 行0 == 满批行0（逐位）");
                ok = run({0, 1, 2}, 2, got) && biteq(got, ref[2]);
                CHECK(ok, "G13b 批大小不变性：n=3 行2 == 满批行2（逐位）");
                // 行位置变化：内容 0 放到槽 p（槽 0..p-1 填其他内容），n=p+1
                ok = true;
                for (int p : {1, S / 2, S - 1}) {
                    std::vector<int> pc;
                    for (int i = 0; i < p; i++) pc.push_back(i + 1);   // 干扰内容
                    pc.push_back(0);                                   // 内容0→槽p
                    std::vector<float> g;
                    ok = run(pc, p, g) && biteq(g, ref[0]) && ok;
                }
                CHECK(ok, "G13c 行位置不变性：内容0@槽{1,4,7} == 满批行0（逐位）");
                // 会话内复跑稳定性：满批再来一遍全逐位同
                ok = true;
                for (int r = 0; r < S; r++) {
                    std::vector<float> g;
                    ok = run(full, r, g) && biteq(g, ref[(size_t)r]) && ok;
                }
                CHECK(ok, "G13d 会话内满批复跑逐位同");
            }
            if (sess) be->DestroySession(sess);
        }
        delete be;
    }

    // ---------------- G14：Farm 代际重建逐位门（清单模式契约）----------------
    // 同进程 建→跑→销毁 ×3 代同种子：跨代指纹/胜局/决策数逐位同——Farm/
    // fiber/bank 无静态残留。此前该契约只有 YGO --ort-jobs 侧对拍（跨代际
    // 同种子逐位三验全绿），框架自有 G 门补上。代间搀入异构形态腿（inline
    // banks=0，不同种子）弄脏全局静态后仍零携带；缓存开变体另验（缓存表随
    // Farm 代际重建，跨代零携带且须有真实命中）。
    {
        LegResult g[3];
        bool ok = true;
        for (int gen = 0; gen < 3; gen++) {
            g[(size_t)gen] = RunOne(2, true, 4, 20260924u);
            if (gen) {
                const LegResult& p = g[0];
                const LegResult& c = g[(size_t)gen];
                ok = c.fp == p.fp && c.fw == p.fw && c.sw == p.sw
                     && c.decisions == p.decisions && ok;
            }
            RunOne(0, true, 4, 777u);   // 代间弄脏：异构形态腿（inline）
        }
        CHECK(ok, "G14 Farm 代际重建×3 逐位同（代间搀 inline 腿弄脏静态）");
        LegResult c0 = RunOne(2, true, 4, 20260924u, false, 16);
        LegResult c1 = RunOne(2, true, 4, 20260924u, false, 16);
        CHECK(c1.fp == c0.fp && c1.decisions == c0.decisions,
              "G14a 缓存开代际重建逐位同");
    }

    std::printf("=== 完成：%s（%d 失败）===\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
