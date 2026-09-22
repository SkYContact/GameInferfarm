// farm_test.cpp — 确定性门与协议冒烟（G1/G2/F 系 + census X=0 + refit 语义）。
//
// 门语义（承接 YGO 产线 G1/F1/F2）：
//  G1 逐位：银行路径 vs inline 路径，同种子 → 每局 outcome/决策数全同
//       （行独立 + 零基组装 ⇒ 设计自带性质；失败=有 bug）
//  G2 确定性：同配置连跑两次 → 全同
//  G3 fiber vs 线程模式（banks=2）：全同（TLS 帧纪律的行为级验证）
//  G4 census：X（失踪人口）恒 0、腿末 live=0、全状态归零
//  G5 refit：同 blob 两次换心 → 逐位同；不同 blob → 结果必变（A1/A2 的 CPU 版）
#include "../examples/toy/toy_adapter.h"
#include "../examples/gomoku/gomoku_adapter.h"
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
};

static LegResult RunOne(int banks, bool fibers, int workers, uint32_t seed0,
                         bool census = false) {
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
    }

    std::printf("=== 完成：%s（%d 失败）===\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
