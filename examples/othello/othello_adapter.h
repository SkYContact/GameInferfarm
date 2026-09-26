// othello_adapter.h — 黑白棋接入：可路由策略（演化个体/当前策略）vs 三种对手
// （LCG 随机 / 锚点池 CPU 前向）。语义对齐实验09 批量环境。
//
// 升级（实验10 农场重做）：
//   1. 锚点对手：固定权重 41k MLP 在 C++ CPU 上前向（几十 µs，塞进 fiber 等待窗，
//      不占 GPU 路由）——锚点池协议原样复刻（exp09/10：锚点近贪心温度采样）；
//   2. 我方温度采样：softmax(logits/T) LCG 采样（确定性），记录所选 log 概率；
//   3. 轨迹落盘：每局终局写 {头 + 逐决策(own/opp 位掩码, act, logp)}（PG 训练数据）；
//   4. 位掩码打包：own/opp 平面 0/1 → 2×u64，轨迹 24B/决策。
// 确定性：单 LCG 流/局（对手随机/锚点采样/我方采样共用，Python 对拍逐 draw 复现）。
#pragma once
#include "inferfarm/inferfarm.h"
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace inferfarm {
namespace othello {

static const int kSize = 8;
static const int kCells = 64;
static const int kFlatW = 41280;        // 128→128→128→64 flat 布局（model.py 同序）
static const int kDir[8][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};

// ---- 全局驱动配置（main 按 CLI 设置；example 层全局足够）----
struct OthelloGlobals {
    int opp_mode = 0;                   // 0=LCG 随机 1=锚点池(CPU MLP)
    float our_temp = 0.0f;              // 0=argmax；>0 温度采样
    float anchor_temp = 0.1f;           // 锚点温度（exp09/10 口径）
    std::vector<float> anchor_w;        // [K*kFlatW] 锚点权重（代内可整块换）
    bool flip_colors = false;           // 白腿：我方执白（population 单色腿的补腿）
    uint32_t gen_mix = 0;               // 代数混淆（=torch 版 seed*100000+gen 语义：
                                        // 每代新随机流，防固定棋谱骨架被演化 exploiting）
    std::vector<float> score_sum;       // 按个体收账（胜1/平0.5/负0；主每腿清零）
    std::vector<int> score_games;
    FILE* dump = nullptr;               // 轨迹文件（null=不记）
    unsigned long long dump_bytes = 0;  // 累计落盘字节（驱动按段切轨迹）
    std::mutex dump_mx;
};
inline OthelloGlobals& OG() { static OthelloGlobals g; return g; }

// CPU MLP 前向（flat theta 布局 = model.py：W1,b1,W2,b2,W3,b3 行主序 [o,i]）
inline void MlpForward(const float* w, const float* x, float* out) {
    float h1[128], h2[128];
    const float *W1 = w, *b1 = w + 16384;
    const float *W2 = w + 16512, *b2 = w + 32896;
    const float *W3 = w + 33024, *b3 = w + 41216;
    for (int o = 0; o < 128; o++) {
        float s = b1[o];
        const float* row = W1 + (size_t)o * 128;
        for (int i = 0; i < 128; i++) s += row[i] * x[i];
        h1[o] = s > 0 ? s : 0;
    }
    for (int o = 0; o < 128; o++) {
        float s = b2[o];
        const float* row = W2 + (size_t)o * 128;
        for (int i = 0; i < 128; i++) s += row[i] * h1[i];
        h2[o] = s > 0 ? s : 0;
    }
    for (int o = 0; o < 64; o++) {
        float s = b3[o];
        const float* row = W3 + (size_t)o * 128;
        for (int i = 0; i < 128; i++) s += row[i] * h2[i];
        out[o] = s;
    }
}

struct OthelloAdapter : GameAdapter {
    uint8_t bd[kCells];
    int turn;
    bool we_black;
    uint32_t rng;
    int discs;                 // 总子数 = 4 + 落子数（翻子=变色不添子）
    bool over;
    int winner;                // 0 未终 1 我方 2 对手 3 平
    bool infer_failed;
    float policy[kCells];
    int decisions;
    int64_t mid_ = 0;          // population 路由个体号（演化，判决16）
    void SetModelId(int64_t m) override { mid_ = m; }

    // 轨迹（本局缓冲）
    struct DecRec { uint64_t own_m, opp_m; uint8_t act; float logp; };
    std::vector<DecRec> recs;
    int chain_id = 0, game_in_chain = 0;
    uint32_t anchor_idx = 0;

    static inline uint8_t last_board[kCells] = {0};

    static uint32_t Lcg(uint32_t& s) {
        s = s * 1664525u + 1013904223u;
        return s >> 8;
    }

    void NewGame(uint64_t seed, bool we_first) override {
        memset(bd, 0, sizeof bd);
        bd[28] = bd[35] = 1;   // 黑 e4 d5
        bd[27] = bd[36] = 2;   // 白 d4 e5
        turn = 0;
        we_black = OG().flip_colors ? !we_first : we_first;
        discs = 4;
        over = false;
        winner = 0;
        infer_failed = false;
        decisions = 0;
        rng = ((uint32_t)seed | 1u) ^ (OG().gen_mix * 2654435761u);
        game_in_chain++;
        recs.clear();
        if (OG().opp_mode == 1 && !OG().anchor_w.empty())
            anchor_idx = Lcg(rng) % (uint32_t)(OG().anchor_w.size() / kFlatW);
    }

    // 空格 cell 对 p 落子是否合法；flips 非空则写翻子格、返回翻子数。
    // path[7]：边格射线最长 7 格中间子（第 8 格出界），越界写=栈溢出 fast-fail
    static int LegalFlips(const uint8_t* bd, int cell, uint8_t p, int* flips) {
        if (bd[cell]) return 0;
        uint8_t o = (p == 1) ? 2 : 1;
        int r0 = cell / kSize, c0 = cell % kSize;
        int total = 0;
        for (int d = 0; d < 8; d++) {
            int r = r0 + kDir[d][0], c = c0 + kDir[d][1];
            int path[7], np = 0;
            while (r >= 0 && r < kSize && c >= 0 && c < kSize && bd[r * kSize + c] == o) {
                path[np++] = r * kSize + c;
                r += kDir[d][0]; c += kDir[d][1];
            }
            if (np > 0 && r >= 0 && r < kSize && c >= 0 && c < kSize
                && bd[r * kSize + c] == p) {
                if (flips) for (int i = 0; i < np; i++) flips[total + i] = path[i];
                total += np;
            }
        }
        return total;
    }
    static bool HasLegal(const uint8_t* bd, uint8_t p) {
        for (int i = 0; i < kCells; i++)
            if (LegalFlips(bd, i, p, nullptr) > 0) return true;
        return false;
    }

    void FinishByDiscs() {
        int mine = 0, opp = 0;
        uint8_t pme = we_black ? 1 : 2, pop = we_black ? 2 : 1;
        for (int i = 0; i < kCells; i++) {
            if (bd[i] == pme) mine++;
            else if (bd[i] == pop) opp++;
        }
        winner = (mine > opp) ? 1 : (opp > mine) ? 2 : 3;
        over = true;
        memcpy(last_board, bd, sizeof bd);
        if (mid_ >= 0 && (size_t)mid_ < OG().score_sum.size()) {
            OG().score_sum[(size_t)mid_] += (winner == 1) ? 1.0f : (winner == 3) ? 0.5f : 0.0f;
            OG().score_games[(size_t)mid_]++;
        }
        DumpGame();
    }

    void Place(int cell, uint8_t p, int* flips, int nf) {
        bd[cell] = p;
        for (int i = 0; i < nf; i++) bd[flips[i]] = p;
        discs += 1;                 // 翻子=变色不添子：总子数=4+落子数
        if (discs >= kCells) FinishByDiscs();
    }

    void PackPlanes(uint8_t pme, uint64_t& own_m, uint64_t& opp_m) {
        own_m = opp_m = 0;
        for (int i = 0; i < kCells; i++) {
            if (bd[i] == pme) own_m |= 1ull << i;
            else if (bd[i]) opp_m |= 1ull << i;
        }
    }
    static void FillX(uint64_t own_m, uint64_t opp_m, float* x) {
        for (int i = 0; i < 64; i++) {
            x[i] = (float)((own_m >> i) & 1);
            x[64 + i] = (float)((opp_m >> i) & 1);
        }
    }

    // ---- 对手着法：随机 或 锚点 CPU 前向（温度采样）——不经过农场 ----
    void OppMove() {
        uint8_t pop = we_black ? 2 : 1;
        int legal[kCells], ne = 0;
        for (int i = 0; i < kCells; i++)
            if (LegalFlips(bd, i, pop, nullptr) > 0) legal[ne++] = i;
        if (ne == 0) { turn ^= 1; return; }
        int cell;
        if (OG().opp_mode == 1) {
            float x[128], lg[64];
            uint64_t om, pm;
            PackPlanes(pop, om, pm);
            FillX(om, pm, x);
            MlpForward(OG().anchor_w.data() + (size_t)anchor_idx * kFlatW, x, lg);
            cell = SampleLegal(lg, legal, ne, OG().anchor_temp);
        } else {
            cell = legal[Lcg(rng) % (uint32_t)ne];
        }
        int flips[64];
        int nf = LegalFlips(bd, cell, pop, flips);
        Place(cell, pop, flips, nf);
        turn ^= 1;
    }

    // 合法格内温度采样；T<=0=argmax（平局取小格号）。确定性 LCG
    int SampleLegal(const float* lg, const int* legal, int ne, float T) {
        if (T <= 0.0f) {
            int best = legal[0];
            for (int e = 1; e < ne; e++)
                if (lg[legal[e]] > lg[best]) best = legal[e];
            return best;
        }
        float mx = lg[legal[0]];
        for (int e = 1; e < ne; e++) if (lg[legal[e]] > mx) mx = lg[legal[e]];
        float p[kCells], Z = 0;
        for (int e = 0; e < ne; e++) {
            p[e] = expf((lg[legal[e]] - mx) / T);
            Z += p[e];
        }
        uint32_t r = Lcg(rng);                       // 0..2^24-1
        float acc = 0;
        for (int e = 0; e < ne; e++) {
            acc += p[e] / Z;
            if ((float)(r + 1u) / 16777216.0f <= acc) return legal[e];
        }
        return legal[ne - 1];
    }

    bool AdvanceToDecision() override {
        if (over) return false;
        uint8_t pme = we_black ? 1 : 2, pop = we_black ? 2 : 1;
        for (;;) {
            if (over) return false;
            bool our_turn = ((turn == 0) == we_black);
            if (our_turn) {
                if (HasLegal(bd, pme)) return true;
                if (!HasLegal(bd, pop)) { FinishByDiscs(); return false; }
                turn ^= 1;
            } else {
                if (!HasLegal(bd, pop)) { turn ^= 1; continue; }
                OppMove();
            }
        }
    }

    void AssembleInto(SlotWriter& slot) override {
        uint8_t pme = we_black ? 1 : 2;
        float* own = (float*)slot.Row("own", nullptr);
        float* opp = (float*)slot.Row("opp", nullptr);
        if (!own || !opp) return;
        for (int i = 0; i < kCells; i++) {
            own[i] = bd[i] == pme ? 1.0f : 0.0f;
            opp[i] = (bd[i] != 0 && bd[i] != pme) ? 1.0f : 0.0f;
        }
        int64_t* mid = (int64_t*)slot.Row("mid", nullptr);
        if (mid) *mid = mid_;       // 路由键（图内取本行权重）
    }

    int CollectOutputs(OutputDest* dests, int cap) override {
        if (cap < 1) return 0;
        dests[0].name = "policy";
        dests[0].dst = policy;
        dests[0].n = kCells;
        return 1;
    }

    void ApplyResult() override {
        uint8_t pme = we_black ? 1 : 2;
        decisions++;
        int legal[kCells], ne = 0;
        for (int i = 0; i < kCells; i++)
            if (LegalFlips(bd, i, pme, nullptr) > 0) legal[ne++] = i;
        if (ne == 0) { FinishByDiscs(); return; }
        int best = SampleLegal(policy, legal, ne, OG().our_temp);
        float logp = 0.0f;
        if (OG().dump && OG().our_temp > 0.0f) {
            // 与 SampleLegal 同一归一的 log 概率（重算 max/Z——采样已消耗一次 draw）
            float mx = policy[legal[0]];
            for (int e = 1; e < ne; e++) if (policy[legal[e]] > mx) mx = policy[legal[e]];
            float Z = 0;
            for (int e = 0; e < ne; e++) Z += expf((policy[legal[e]] - mx) / OG().our_temp);
            logp = (policy[best] - mx) / OG().our_temp - logf(Z);
            DecRec r;
            PackPlanes(pme, r.own_m, r.opp_m);
            r.act = (uint8_t)best;
            r.logp = logp;
            recs.push_back(r);
        }
        int flips[64];
        int nf = LegalFlips(bd, best, pme, flips);
        Place(best, pme, flips, nf);
        turn ^= 1;
    }

    // 轨迹落盘：头{chain,game,先后手,结果,锚点号,n} + n×{own,opp 掩码,act,logp}
    void DumpGame() {
        if (!OG().dump || recs.empty()) return;
        #pragma pack(push, 1)
        struct Head { int32_t chain, game; uint8_t we_first, outcome; uint32_t anchor, n; };
        struct Rec { uint64_t om, pm; uint8_t act; uint8_t pad[3]; float logp; };
        #pragma pack(pop)
        static_assert(sizeof(Head) == 18 && sizeof(Rec) == 24, "轨迹二进制布局钉死");
        Head h{chain_id, game_in_chain, (uint8_t)we_black,
               (uint8_t)(winner == 1 ? 1 : winner == 3 ? 0xFF : 0), anchor_idx,
               (uint32_t)recs.size()};
        std::lock_guard<std::mutex> lk(OG().dump_mx);
        fwrite(&h, sizeof h, 1, OG().dump);
        for (auto& r : recs) {
            Rec q{r.own_m, r.opp_m, r.act, {0, 0, 0}, r.logp};
            fwrite(&q, sizeof q, 1, OG().dump);
        }
        OG().dump_bytes += sizeof(Head) + (unsigned long long)recs.size() * sizeof(Rec);
        fflush(OG().dump);
        recs.clear();
    }

    void OnInferFail() override { infer_failed = true; }
    bool IsDone() override { return over; }
    int Outcome() override {
        if (infer_failed || winner == 2) return 0;
        if (winner == 1) return 1;
        return -1;
    }
    bool WeAreFirst() override { return we_black; }
    long long GameFingerprint() override {
        return (long long)(discs - 4) * 1000003 + winner;
    }
    ITlsFrame* TlsFrame() override { return nullptr; }
};

inline GameAdapter* MakeOthelloAdapter(int chain, void*) {
    OthelloAdapter* a = new OthelloAdapter();
    a->chain_id = chain;
    return a;
}

inline CpuModelDecl OthelloModelDecl(int slots) {
    CpuModelDecl d;
    d.slots = slots;
    d.poly_k = kCells;
    d.hidden = 64;
    d.ins.push_back({"own", DTYPE_F32, {kCells}});
    d.ins.push_back({"opp", DTYPE_F32, {kCells}});
    d.outs.push_back({"policy", kCells});
    return d;
}

inline void PrintBoard(const uint8_t* bd) {
    std::printf("   ");
    for (int c = 0; c < kSize; c++) std::printf("%X", c);
    std::printf("\n");
    for (int r = 0; r < kSize; r++) {
        std::printf("%d  ", r);
        for (int c = 0; c < kSize; c++) {
            char ch = bd[r * kSize + c] == 0 ? '.'
                : bd[r * kSize + c] == 1 ? 'X' : 'O';
            std::putchar(ch);
        }
        std::printf("\n");
    }
}

} // namespace othello
} // namespace inferfarm
