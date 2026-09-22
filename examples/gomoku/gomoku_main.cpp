// gomoku_main.cpp — 五子棋范例入口：未训练神经网络（CPU 后端确定性模型）
// vs 规则对手，跑通推理农场全流程（fiber/银行攒批/收割回投/种子协议）。
//
//   gomoku [--chains 8] [--games 16] [--banks 2] [--workers 4] [--slots 8]
//          [--inline] [--threads] [--census] [--show-board]
//
// 预期：未训练模型胜率很低（这是诚实的——模型只是局面的固定线性函数）；
// 本范例展示的是流程与逐位确定性：同参数复跑结果逐位同（FARM_CENSUS=1
// 可看取证层输出）。
#include "gomoku_adapter.h"
#include <cstdio>
#include <cstring>

using namespace inferfarm;
using namespace inferfarm::gomoku;

int main(int argc, char** argv) {
    FarmConfig cfg;
    cfg.name = "gomoku";
    cfg.chains = 8;
    cfg.games = 16;
    cfg.seed0 = 20260922u;
    cfg.fibers = true;
    cfg.workers = 4;
    cfg.banks = 2;
    cfg.slots = 8;
    cfg.window_ms = 0.2;
    cfg.stagger_ms = 1;
    cfg.model.backend = "cpu";
    cfg.model.cpu = GomokuModelDecl(cfg.slots);
    bool show_board = false;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--banks") && i + 1 < argc) cfg.banks = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--chains") && i + 1 < argc) cfg.chains = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--games") && i + 1 < argc) cfg.games = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--workers") && i + 1 < argc) cfg.workers = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--slots") && i + 1 < argc) { cfg.slots = atoi(argv[++i]); cfg.model.cpu = GomokuModelDecl(cfg.slots); }
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) cfg.seed0 = (uint32_t)strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--inline")) cfg.banks = 0;
        else if (!std::strcmp(argv[i], "--threads")) cfg.fibers = false;
        else if (!std::strcmp(argv[i], "--census")) cfg.census = true;
        else if (!std::strcmp(argv[i], "--show-board")) show_board = true;
        else { std::printf("未知参数 %s\n", argv[i]); return 2; }
    }
    Farm farm;
    if (!farm.Init(cfg)) return 1;
    double sec = farm.RunLeg(MakeGomokuAdapter, nullptr);
    const FarmTally& t = farm.tally();
    std::printf("[gomoku] %d 局 / %.2fs；先手 %d/%d 后手 %d/%d；指纹 %016llx\n",
                t.games_done, sec, t.first_wins, t.first_total, t.second_wins,
                t.second_total, (unsigned long long)t.fingerprint);
    if (show_board) {
        std::printf("\n终局样例棋盘（链 0 末局）：\n");
        PrintBoard(GomokuAdapter::last_board);
    }
    return 0;
}
