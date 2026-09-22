// gomoku_main.cpp — 五子棋范例入口：一层 MLP（未训练）vs 规则对手，
// 跑通推理农场全流程。三后端可换：cpu（默认，免 GPU）/ ort（onnx）/ trt（engine）。
//
//   gomoku [--backend cpu|ort|trt] [--model <fb.onnx>] [--engine <plan>]
//          [--chains 8] [--games 16] [--banks 2] [--workers 4] [--slots 8]
//          [--cache-log2 16] [--device <spec>]... [--inline] [--threads]
//          [--census] [--show-board]
//
//   --device 多设备组（判决15，可重复；首个替换主设备，后续追加设备组）：
//     spec = backend[,ep=cuda|dml][,dev=N][,banks=K][,share=W][,slots=S][,model=路径][,engine=路径][,dir=运行时目录]
//   例（NVIDIA 主卡 + AMD 核显，share=按算力配链、slots=核显小批图破形状墙）：
//     gomoku --backend ort --model m.fb16.onnx --banks 2 \
//            --device ort,ep=dml,dev=1,banks=1,share=0.4,slots=4,model=m.fb4.onnx,dir=D:/dml_rt
//
// ort/trt 模型工件由 tools/bake_gomoku_mlp.py 烤制（一层 MLP，权重由种子
// 生成——确定性）。三后端同架构；指纹只在与自身同后端双腿间可比。
#include "gomoku_adapter.h"
#include <cstdio>
#include <cstring>
#include <string>

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
        if (!std::strcmp(argv[i], "--backend") && i + 1 < argc) {
            cfg.model.backend = argv[++i];
            if (cfg.model.backend != "cpu" && cfg.model.backend != "ort"
                && cfg.model.backend != "trt") {
                std::printf("未知后端 %s（cpu|ort|trt）\n", cfg.model.backend.c_str());
                return 2;
            }
        }
        else if (!std::strcmp(argv[i], "--model") && i + 1 < argc) cfg.model.model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--engine") && i + 1 < argc) cfg.model.engine_path = argv[++i];
        else if (!std::strcmp(argv[i], "--ort-dir") && i + 1 < argc) cfg.model.ort_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--trt-dir") && i + 1 < argc) cfg.model.trt_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--cuda-dir") && i + 1 < argc) cfg.model.cuda_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--refit") && i + 1 < argc) cfg.model.refit_weights = argv[++i];
        else if (!std::strcmp(argv[i], "--banks") && i + 1 < argc) cfg.banks = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--chains") && i + 1 < argc) cfg.chains = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--games") && i + 1 < argc) cfg.games = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--workers") && i + 1 < argc) cfg.workers = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--slots") && i + 1 < argc) { cfg.slots = atoi(argv[++i]); cfg.model.cpu = GomokuModelDecl(cfg.slots); }
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) cfg.seed0 = (uint32_t)strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--cache-log2") && i + 1 < argc) cfg.cache_log2 = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--device") && i + 1 < argc) {
            // spec = backend[,ep=..][,dev=N][,banks=K][,model=..][,engine=..][,dir=..]
            std::string spec = argv[++i];
            DeviceConfig d;
            size_t pos = 0, first = spec.find(',');
            d.model.backend = spec.substr(0, first == std::string::npos ? spec.size() : first);
            bool have_model = false;
            auto field = [&](const std::string& k) -> std::string {
                std::string pat = k + "=";
                size_t p = 0;
                while ((p = spec.find(pat, p)) != std::string::npos) {
                    if (p != 0 && spec[p - 1] != ',') { p += pat.size(); continue; }
                    size_t v0 = p + pat.size(), v1 = spec.find(',', v0);
                    return spec.substr(v0, v1 == std::string::npos ? std::string::npos : v1 - v0);
                }
                return "";
            };
            std::string v;
            if (!(v = field("ep")).empty()) d.model.ort_ep = v;
            if (!(v = field("dev")).empty()) d.model.device_id = atoi(v.c_str());
            if (!(v = field("model")).empty()) { d.model.model_path = v; have_model = true; }
            if (!(v = field("engine")).empty()) { d.model.engine_path = v; have_model = true; }
            if (!(v = field("dir")).empty()) d.model.ort_dir = v;
            if (!(v = field("banks")).empty()) d.banks = atoi(v.c_str());
            else d.banks = 1;
            if (!(v = field("share")).empty()) d.share = atof(v.c_str());
            if (!(v = field("slots")).empty()) d.slots = atoi(v.c_str());
            if (!have_model) {
                if (d.model.backend == "ort") d.model.model_path = cfg.model.model_path;
                if (d.model.backend == "trt") d.model.engine_path = cfg.model.engine_path;
            }
            if (d.model.backend == "cpu") d.model.cpu = GomokuModelDecl(cfg.slots);
            if (cfg.devices.empty()) {   // 首个：替换主设备
                cfg.model = d.model;
                cfg.banks = d.banks;
                cfg.devices.push_back(d);   // 占位（Farm 展开时 devices 非空即走多组）
            } else {
                cfg.devices.push_back(d);
            }
        }
        else if (!std::strcmp(argv[i], "--inline")) cfg.banks = 0;
        else if (!std::strcmp(argv[i], "--threads")) cfg.fibers = false;
        else if (!std::strcmp(argv[i], "--census")) cfg.census = true;
        else if (!std::strcmp(argv[i], "--show-board")) show_board = true;
        else { std::printf("未知参数 %s\n", argv[i]); return 2; }
    }
    if (cfg.model.backend == "ort" && cfg.model.model_path.empty()) {
        std::fprintf(stderr, "[gomoku] ort 后端需 --model <fb.onnx>"
                     "（tools/bake_gomoku_mlp.py 烤制）\n");
        return 2;
    }
    if (cfg.model.backend == "trt" && cfg.model.engine_path.empty()) {
        std::fprintf(stderr, "[gomoku] trt 后端需 --engine <plan>"
                     "（tools/bake_gomoku_mlp.py --trt 烤制）\n");
        return 2;
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
