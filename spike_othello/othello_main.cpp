// othello_main.cpp — 黑白棋尖刺入口：BC MLP（41k 参数，实验09 检查点导出）
// vs 随机合法对手。结构与 gomoku_main 相同；ort 后端（cpu 后端仅支持一层 MLP）。
//
//   othello --backend ort --model models/othello_bc_mlp.fb16.onnx
//           [--chains 64] [--games 1280] [--banks 2] [--workers 14] [--slots 16]
//           [--device <spec>]... [--census] [--show-board]
#include "othello_adapter.h"
#include <cstdio>
#include <cstring>
#include <string>

using namespace inferfarm;
using namespace inferfarm::othello;

int main(int argc, char** argv) {
    FarmConfig cfg;
    cfg.name = "othello";
    cfg.chains = 64;
    cfg.games = 1280;
    cfg.seed0 = 20260922u;
    cfg.fibers = true;
    cfg.workers = 14;
    cfg.banks = 2;
    cfg.slots = 16;
    cfg.window_ms = 0.2;
    cfg.stagger_ms = 0;
    cfg.model.backend = "ort";
    cfg.model.cpu = OthelloModelDecl(cfg.slots);
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
        else if (!std::strcmp(argv[i], "--banks") && i + 1 < argc) cfg.banks = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--chains") && i + 1 < argc) cfg.chains = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--games") && i + 1 < argc) cfg.games = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--workers") && i + 1 < argc) cfg.workers = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--slots") && i + 1 < argc) { cfg.slots = atoi(argv[++i]); cfg.model.cpu = OthelloModelDecl(cfg.slots); }
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) cfg.seed0 = (uint32_t)strtoul(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--cache-log2") && i + 1 < argc) cfg.cache_log2 = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--device") && i + 1 < argc) {
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
            if (d.model.backend == "cpu") d.model.cpu = OthelloModelDecl(cfg.slots);
            if (cfg.devices.empty()) {
                cfg.model = d.model;
                cfg.banks = d.banks;
                cfg.devices.push_back(d);
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
        std::fprintf(stderr, "[othello] ort 后端需 --model <fb.onnx>（spike/export_bc_onnx.py）\n");
        return 2;
    }
    Farm farm;
    if (!farm.Init(cfg)) return 1;
    double sec = farm.RunLeg(MakeOthelloAdapter, nullptr);
    const FarmTally& t = farm.tally();
    std::printf("[othello] %d 局 / %.2fs = %.0f 局/s；先手 %d/%d 后手 %d/%d；"
                "决策 %lld；推理故障 %d；指纹 %016llx\n",
                t.games_done, sec, t.games_done / sec,
                t.first_wins, t.first_total, t.second_wins, t.second_total,
                (long long)t.decisions, t.infer_fails, (unsigned long long)t.fingerprint);
    if (show_board) {
        std::printf("\n终局样例棋盘（末局快照）：\n");
        PrintBoard(OthelloAdapter::last_board);
    }
    return 0;
}
