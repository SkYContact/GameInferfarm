// othello_adapter.h — 黑白棋（8×8）接入：BC 策略网 vs 随机合法对手。
// 语义对齐实验09 的 Python 批量环境：初始 白d4=27/e5=36 黑e4=28/d5=35、黑先、
// 落子翻 8 方向、无棋 pass、双卡死或满盘(64子)终局、按子数定胜负。
// 确定性：对手随机源=种子派生 LCG（同种子逐位同）；我方=合法格内 argmax
// （平局取小格号）。种子协议 game seed = seed0 + chain*per + game（farm 侧）。
#pragma once
#include "inferfarm/inferfarm.h"
#include <cstdio>
#include <cstring>

namespace inferfarm {
namespace othello {

static const int kSize = 8;
static const int kCells = 64;
static const int kDir[8][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};

struct OthelloAdapter : GameAdapter {
    uint8_t bd[kCells];     // 0 空 1 黑 2 白
    int turn;               // 0 黑 1 白
    bool we_black;
    uint32_t rng;
    int discs;              // 盘面子数（初值 4）
    bool over;
    int winner;             // 0 未终 1 我方 2 对手 3 平
    bool infer_failed;
    float policy[kCells];
    int decisions;          // 我方决策计数（对拍口径）
    int64_t mid_ = 0;       // population 路由个体号（演化，判决16；非路由模型
                            // 无 "mid" 输入→AssembleInto 探测不到即跳过）
    void SetModelId(int64_t m) override { mid_ = m; }   // 框架演化模式直喂

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
        we_black = we_first;
        discs = 4;
        over = false;
        winner = 0;
        infer_failed = false;
        decisions = 0;
        rng = (uint32_t)seed | 1u;
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
    }

    void Place(int cell, uint8_t p, int* flips, int nf) {
        bd[cell] = p;
        for (int i = 0; i < nf; i++) bd[flips[i]] = p;
        discs += 1;                 // 翻子=变色不添子：总子数=4+落子数
        if (discs >= kCells) FinishByDiscs();   // 满盘终局
    }

    // 对手（随机合法）：无棋则 pass（换手）；有棋 LCG 均匀取一个
    void OppMove() {
        uint8_t pop = we_black ? 2 : 1;
        int legal[kCells], ne = 0;
        for (int i = 0; i < kCells; i++)
            if (LegalFlips(bd, i, pop, nullptr) > 0) legal[ne++] = i;
        if (ne == 0) { turn ^= 1; return; }
        int cell = legal[Lcg(rng) % (uint32_t)ne];
        int flips[64];
        int nf = LegalFlips(bd, cell, pop, flips);
        Place(cell, pop, flips, nf);
        turn ^= 1;
    }

    bool AdvanceToDecision() override {
        if (over) return false;
        uint8_t pme = we_black ? 1 : 2, pop = we_black ? 2 : 1;
        for (;;) {
            if (over) return false;
            bool our_turn = ((turn == 0) == we_black);
            if (our_turn) {
                if (HasLegal(bd, pme)) return true;    // 我方决策点
                if (!HasLegal(bd, pop)) { FinishByDiscs(); return false; }  // 双卡死
                turn ^= 1;                              // 我方 pass
            } else {
                if (!HasLegal(bd, pop)) { turn ^= 1; continue; }  // 对手 pass
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
        if (mid) *mid = mid_;   // 路由键（图内 Gather(pop, mid) 取本行权重）
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
        int best = -1;
        for (int i = 0; i < kCells; i++) {
            if (bd[i] || LegalFlips(bd, i, pme, nullptr) <= 0) continue;
            if (best < 0 || policy[i] > policy[best]) best = i;   // 平局取小格号
        }
        if (best < 0) { FinishByDiscs(); return; }
        int flips[64];
        int nf = LegalFlips(bd, best, pme, flips);
        Place(best, pme, flips, nf);
        turn ^= 1;
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

inline GameAdapter* MakeOthelloAdapter(int, void*) {
    return new OthelloAdapter();
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
    std::printf("X=黑 O=白（我方=%s）\n", "黑/白逐局交替");
}

} // namespace othello
} // namespace inferfarm
