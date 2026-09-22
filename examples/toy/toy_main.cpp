// toy_main.cpp — 推理农场最小示范：玩具适配器 × CPU 后端 × fiber × 银行制。
// env 旋钮全可用（FARM_CENSUS=1 看取证层）。
#include "toy_adapter.h"
#include <cstdio>
#include <cstring>

using namespace inferfarm;
using namespace inferfarm::toy;

int main(int argc, char** argv) {
    FarmConfig cfg;
    cfg.name = "toy";
    cfg.chains = 8;
    cfg.games = 64;
    cfg.seed0 = 4242;
    cfg.fibers = true;
    cfg.workers = 4;
    cfg.banks = 2;
    cfg.slots = 8;
    cfg.window_ms = 0.2;
    cfg.stagger_ms = 1;
    cfg.model.backend = "cpu";
    cfg.model.cpu = ToyModelDecl(cfg.slots);
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--banks") && i + 1 < argc) cfg.banks = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--chains") && i + 1 < argc) cfg.chains = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--games") && i + 1 < argc) cfg.games = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--workers") && i + 1 < argc) cfg.workers = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--inline")) cfg.banks = 0;
        else if (!std::strcmp(argv[i], "--threads")) cfg.fibers = false;
        else if (!std::strcmp(argv[i], "--census")) cfg.census = true;
    }
    Farm farm;
    if (!farm.Init(cfg)) return 1;
    double sec = farm.RunLeg(MakeToyAdapter, nullptr);
    std::printf("[toy] 腿墙钟 %.2fs\n", sec);
    return 0;
}
