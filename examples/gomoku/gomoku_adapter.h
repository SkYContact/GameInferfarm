// gomoku_adapter.h — 五子棋接入范例（inferfarm 最简参考实现）。
//
// 这个文件演示把一个真实游戏接进推理农场所需的全部工作：
//   1. 按队形实现 GameAdapter 七个钩子（约 200 行）；
//   2. 声明模型输入输出（两张己方/对方局面平面 → 225 点策略头）；
//   3. 规则对手（能赢就赢/必须堵就堵/启发式落子）——不需要任何训练。
//
// 神经网络侧用的是 CPU 后端的未训练确定性模型（权重由种子生成）：
// 策略=局面的固定线性函数。它不会下棋——本范例的目的是**跑通流程**：
// 种子协议、组装直写槽、银行攒批、收割回投、逐位确定性门全部照常工作。
// 换成真模型只改 GomokuModelDecl（或换 ort/trt 后端），适配器零改动。
//
// 确定性说明：对手的落子随机性来自种子派生的 LCG（同种子逐位同）；
// 我们侧 argmax 平局取小格号（确定）。
#pragma once
#include "inferfarm/inferfarm.h"
#include <cstdio>
#include <cstring>

namespace inferfarm {
namespace gomoku {

static const int kSize = 15;       // 15×15 标准盘
static const int kCells = kSize * kSize;

// ---- 规则工具：落子后过 (r,c) 的四方向最长连子 ----
inline int LineLen(const uint8_t* bd, int r, int c, uint8_t p, int dr, int dc) {
    int n = 1;
    for (int i = 1; i < 5; i++) {
        int rr = r + dr * i, cc = c + dc * i;
        if (rr < 0 || rr >= kSize || cc < 0 || cc >= kSize || bd[rr * kSize + cc] != p)
            break;
        n++;
    }
    for (int i = 1; i < 5; i++) {
        int rr = r - dr * i, cc = c - dc * i;
        if (rr < 0 || rr >= kSize || cc < 0 || cc >= kSize || bd[rr * kSize + cc] != p)
            break;
        n++;
    }
    return n;
}
inline bool MakesFive(const uint8_t* bd, int cell, uint8_t p) {
    int r = cell / kSize, c = cell % kSize;
    static const int kDir[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (auto& d : kDir)
        if (LineLen(bd, r, c, p, d[0], d[1]) >= 5) return true;
    return false;
}

struct GomokuAdapter : GameAdapter {
    uint8_t bd[kCells];        // 0 空 1 我方 2 对手
    uint32_t rng = 1;          // 对手随机源（种子派生——确定性契约）
    int moves = 0;             // 双方总落子数
    bool over = false;
    int winner = 0;            // 0 未终 1 我方 2 对手 3 平
    bool we_first_ = true;
    bool infer_failed = false;
    float policy[kCells] = {0};   // 模型输出（CollectOutputs 申报，收割回填）

    // 调试用棋盘快照（--show-board 用；链 0 末局）
    static inline uint8_t last_board[kCells] = {0};   // 调试快照（C++17 inline）

    int chain_id = 0;
    explicit GomokuAdapter(int chain) : chain_id(chain) {}

    static uint32_t Lcg(uint32_t& s) {
        s = s * 1664525u + 1013904223u;
        return s >> 8;
    }

    void NewGame(uint64_t seed, bool we_first) override {
        memset(bd, 0, sizeof bd);
        rng = (uint32_t)seed | 1u;
        moves = 0;
        over = false;
        winner = 0;
        we_first_ = we_first;
        infer_failed = false;
    }

    // ---- 对手（规则）：能赢就赢 → 必须堵就堵 → 启发式（邻子+中心+种子抖动）----
    void OppMove() {
        int empties[kCells], ne = 0;
        for (int i = 0; i < kCells; i++)
            if (!bd[i]) empties[ne++] = i;
        if (ne == 0) { over = true; winner = 3; return; }
        // ① 我方（=对手视角的玩家 2）一步取胜
        for (int e = 0; e < ne; e++)
            if (MakesFive(bd, empties[e], 2)) { Place(empties[e], 2); return; }
        // ② 堵掉我方的一步取胜
        for (int e = 0; e < ne; e++)
            if (MakesFive(bd, empties[e], 1)) { Place(empties[e], 2); return; }
        // ③ 启发式：8 邻域子数×3 + 中心偏好 + 种子抖动
        int best = -1;
        int best_score = -1;
        for (int e = 0; e < ne; e++) {
            int cell = empties[e];
            int r = cell / kSize, c = cell % kSize;
            int adj = 0;
            for (int dr = -1; dr <= 1; dr++)
                for (int dc = -1; dc <= 1; dc++) {
                    int rr = r + dr, cc = c + dc;
                    if ((dr || dc) && rr >= 0 && rr < kSize && cc >= 0 && cc < kSize
                        && bd[rr * kSize + cc]) adj++;
                }
            int dc_ = r < kSize - 1 - r ? r : kSize - 1 - r;
            int dcc = c < kSize - 1 - c ? c : kSize - 1 - c;
            int center = dc_ < dcc ? dc_ : dcc;          // 距边最小值=中心度
            int score = adj * 3 + center + (int)(Lcg(rng) % 5);
            if (score > best_score) { best_score = score; best = cell; }
        }
        Place(best, 2);
    }

    void Place(int cell, uint8_t p) {
        bd[cell] = p;
        moves++;
        if (MakesFive(bd, cell, p)) {
            over = true;
            winner = p;
            if (chain_id == 0) memcpy(last_board, bd, sizeof bd);   // 仅链 0 写
        } else if (moves >= kCells) {
            over = true;
            winner = 3;
            if (chain_id == 0) memcpy(last_board, bd, sizeof bd);   // 仅链 0 写
        }
    }

    // ---- GameAdapter 钩子 ----
    // 驱动环在每次我方决策前调用：先把对手走掉（对手先手时含开局），
    // 终局则返回 false。
    bool AdvanceToDecision() override {
        if (over) return false;
        if (moves == 0) {
            if (!we_first_) {          // 对手先攻：开局一手（确定=天元）
                OppOpen();
                if (over) return false;
            }
            return true;               // 我方（先攻）开局决策点
        }
        if (pend_opp_) {               // 我方上一手之后轮到对手
            OppMove();
            pend_opp_ = false;
            if (over) return false;
        }
        return true;   // 我方决策点（契约 1：本函数与 AssembleInto 无挂起点）
    }
    bool pend_opp_ = false;

    void OppOpen() {
        // 空盘开局：中心（确定；对手先攻时第一步走天元）
        Place(7 * kSize + 7, 2);
    }

    void AssembleInto(SlotWriter& slot) override {
        // 组装直写槽（零拷贝）：两张局面平面
        float* own = (float*)slot.Row("own", nullptr);
        float* opp = (float*)slot.Row("opp", nullptr);
        if (!own || !opp) return;
        for (int i = 0; i < kCells; i++) {
            own[i] = bd[i] == 1 ? 1.0f : 0.0f;
            opp[i] = bd[i] == 2 ? 1.0f : 0.0f;
        }
    }

    int CollectOutputs(OutputDest* dests, int cap) override {
        if (cap < 1) return 0;
        dests[0].name = "policy";
        dests[0].dst = policy;
        dests[0].n = kCells;
        return 1;
    }

    void ApplyResult() override {
        // 合法性过滤在适配器做：只在空格中取 argmax（平局取小格号——确定）
        int best = -1;
        for (int i = 0; i < kCells; i++) {
            if (bd[i]) continue;
            if (best < 0 || policy[i] > policy[best]) best = i;
        }
        if (best < 0) { over = true; winner = 3; return; }
        Place(best, 1);
        pend_opp_ = true;   // 我方落子完毕，下一轮 Advance 走对手
    }

    void OnInferFail() override { infer_failed = true; }
    bool IsDone() override { return over; }
    int Outcome() override {
        if (infer_failed || winner == 2) return 0;
        if (winner == 1) return 1;
        return -1;   // 平局
    }
    bool WeAreFirst() override { return we_first_; }
    long long GameFingerprint() override {
        return (long long)moves * 1000003 + winner;
    }
    // 棋类适配器无跨让出 thread_local 状态 → 无需帧（见 tls_frame.h 审计清单）
    ITlsFrame* TlsFrame() override { return nullptr; }

};

inline GameAdapter* MakeGomokuAdapter(int, void*) {
    return new GomokuAdapter(0);
}

inline CpuModelDecl GomokuModelDecl(int slots) {
    CpuModelDecl d;
    d.slots = slots;
    d.poly_k = kCells;               // 全局面输入
    d.hidden = 64;                   // 一层 MLP（450→64→225，未训练参考模型）
    d.ins.push_back({"own", DTYPE_F32, {kCells}});
    d.ins.push_back({"opp", DTYPE_F32, {kCells}});
    d.outs.push_back({"policy", kCells});
    return d;
}

// 调试打印（--show-board）
inline void PrintBoard(const uint8_t* bd) {
    std::printf("   ");
    for (int c = 0; c < kSize; c++) std::printf("%X", c % 16);
    std::printf("\n");
    for (int r = 0; r < kSize; r++) {
        std::printf("%2X ", r);
        for (int c = 0; c < kSize; c++) {
            char ch = bd[r * kSize + c] == 0 ? '.'
                : bd[r * kSize + c] == 1 ? 'O' : 'X';
            std::putchar(ch);
        }
        std::printf("\n");
    }
    std::printf("O=我方(神经网络) X=对手(规则)\n");
}

} // namespace gomoku
} // namespace inferfarm
