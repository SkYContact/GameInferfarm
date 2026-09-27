// gomoku_main.cpp — 五子棋范例入口：一层 MLP（未训练）vs 规则对手，
// 跑通推理农场全流程。三后端可换：cpu（默认，免 GPU）/ ort（onnx）/ trt（engine）。
//
//   gomoku [--backend cpu|ort|trt] [--model <fb.onnx>] [--engine <plan>]
//          [--chains 8] [--games 16] [--banks 2] [--workers 4] [--slots 8]
//          [--cache-log2 16] [--device <spec>]... [--inline] [--threads]
//          [--census] [--show-board] [--append-own]
//
//   --append-own 声明式增量 H2D 演示/A/B（判决25）：own 面点名 append +
//   包装适配器把 own 行改写为"深度前缀模式"（own[k]=HistPat(k), k<moves+1；
//   行其余=0）并逐行 FaceDepth 申报——content=f(k) 与局无关 ⇒ append-only
//   承诺在槽轮转下结构性成立。对照腿=同一命令去掉 --append-own（全量）。
//   配合 FARM_H2D_DELTA=0 / FARM_H2D_BATCH=1 等环境开关做 A/B；dep 三段
//   打印（h2d kB/批）为段字节测量面。
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
#include <cstdlib>
#include <cstring>
#include <string>


using namespace inferfarm;
using namespace inferfarm::gomoku;

// --append-mode=fill 的旗标（AdapterFactory=函数指针不可捕获——读全局）
static bool g_append_fill = false;

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
    bool append_own = false;
    const char* append_mode = "hist";
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--backend") && i + 1 < argc) {
            cfg.model.backend = argv[++i];
            if (cfg.model.backend != "cpu" && cfg.model.backend != "ort"
                && cfg.model.backend != "trt" && cfg.model.backend != "ncnn") {
                std::printf("未知后端 %s（cpu|ort|trt|ncnn）\n", cfg.model.backend.c_str());
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
            d.model.population_input = cfg.model.population_input;   // 路由模式组继承
            if (d.model.backend == "cpu") d.model.cpu = GomokuModelDecl(cfg.slots);
            if (d.model.backend == "ncnn") d.model.cpu = cfg.model.cpu;   // ncnn 声明式 spec 同样继承（否则 --device 组 LoadSpec 空声明必败）
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
        else if (!std::strcmp(argv[i], "--append-own")) {
            cfg.model.append_inputs.push_back("own");
            append_own = true;
        }
        else if (!std::strcmp(argv[i], "--append-mode") && i + 1 < argc) {
            // hist=深度=moves+1（低利用率：增长段个位数字节/行）
            // fill=深度=kCells（高利用率：每行整行重写=append 退化为 full 的
            //   最坏情况，考提交税/聚合收益）
            append_mode = argv[++i];
        }
        else { std::printf("未知参数 %s\n", argv[i]); return 2; }
    }
    // --device 组继承 append 声明（与 population_input 同规）
    for (auto& d : cfg.devices)
        d.model.append_inputs = cfg.model.append_inputs;
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
    // --append-own：包装适配器（own 行=深度前缀模式 + FaceDepth 申报）；
    // 未开=普通 GomokuAdapter（全量路径，零行为差）。A/B 强制 full 对照=
    // 同命令 + FARM_H2D_DELTA=0（同适配器同游戏，仅后端回落全量）。
    double sec;
    if (append_own) {
        g_append_fill = !std::strcmp(append_mode, "fill");
        struct AppendAdapter : GomokuAdapter {
            explicit AppendAdapter(int chain) : GomokuAdapter(chain) {}
            static float HistPat(int k) {
                uint32_t x = 0x9E3779B9u * (uint32_t)(k + 1);
                x ^= x << 13; x ^= x >> 17; x ^= x << 5;
                return (float)((int)(x & 0xFFFF) - 32768) / 32768.0f;
            }
            void AssembleInto(SlotWriter& slot) override {
                GomokuAdapter::AssembleInto(slot);
                float* own = (float*)slot.Row("own", nullptr);
                if (!own) return;
                // hist=低利用率（depth=moves+1，行动历史语义）；
                // fill=高利用率（depth=kCells 恒满=整行重写，最坏情况）
                const int depth = g_append_fill ? kCells
                    : (moves + 1 < kCells ? moves + 1 : kCells);
                for (int k = 0; k < depth; k++) own[k] = HistPat(k);
                std::memset(own + depth, 0,
                            (size_t)(kCells - depth) * sizeof(float));
                slot.FaceDepth("own", depth);
            }
        };
        sec = farm.RunLeg([](int chain, void*) -> GameAdapter* {
            return new AppendAdapter(chain);   // --append-own 的申报点
        }, nullptr);
    } else {
        sec = farm.RunLeg(MakeGomokuAdapter, nullptr);
    }
    const FarmTally& t = farm.tally();
    std::printf("[gomoku] RunLeg 返回\n");
    std::fflush(stdout);
    std::printf("[gomoku] %d 局 / %.2fs；先手 %d/%d 后手 %d/%d；指纹 %016llx\n",
                t.games_done, sec, t.first_wins, t.first_total, t.second_wins,
                t.second_total, (unsigned long long)t.fingerprint);
    std::fflush(stdout);
    if (show_board) {
        std::printf("\n终局样例棋盘（链 0 末局）：\n");
        PrintBoard(GomokuAdapter::last_board);
    }
    return 0;
}
