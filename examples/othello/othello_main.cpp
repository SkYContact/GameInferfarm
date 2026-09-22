// othello_main.cpp — 黑白棋尖刺入口：BC MLP（41k 参数，实验09 检查点导出）
// vs 随机合法对手。结构与 gomoku_main 相同；ort 后端（cpu 后端仅支持一层 MLP）。
//
//   othello --backend ort --model models/othello_bc_mlp.fb16.onnx
//           [--chains 64] [--games 1280] [--banks 2] [--workers 14] [--slots 16]
//           [--device <spec>]... [--census] [--show-board]
//
// ES 代际模式（population 路由，判决16）——模型=路由图（own/opp/pop/mid）：
//   othello --backend ort --model models/othello_pop.fb128.onnx
//           --population 128 --gens 4 [--pop-file <[P,flat_w] f32 bin>]
//   每代：SetPopulation（确定性扰动种群或文件载入）→ 一腿 128 个体×8 局 →
//   打印代墙钟（对标 torch 参考 1.07s/代）。链 c=个体 c%P（8 局同个体）。
#include "othello_adapter.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace inferfarm;
using namespace inferfarm::othello;

// 路由工厂：链 c → 个体 c%P（mid_ 由适配器在组装时写入行）
struct PopCtx { int P; };
static GameAdapter* MakeRoutedAdapter(int chain, void* p) {
    OthelloAdapter* a = new OthelloAdapter();
    a->mid_ = chain % ((PopCtx*)p)->P;
    return a;
}

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
    int pop_p = 0, gens = 1;
    std::string pop_file;
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
            d.model.population_input = cfg.model.population_input;   // 路由模式组继承
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
        else if (!std::strcmp(argv[i], "--population") && i + 1 < argc) pop_p = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--gens") && i + 1 < argc) gens = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--pop-file") && i + 1 < argc) pop_file = argv[++i];
        else if (!std::strcmp(argv[i], "--threads")) cfg.fibers = false;
        else if (!std::strcmp(argv[i], "--census")) cfg.census = true;
        else if (!std::strcmp(argv[i], "--show-board")) show_board = true;
        else { std::printf("未知参数 %s\n", argv[i]); return 2; }
    }
    if (cfg.model.backend == "ort" && cfg.model.model_path.empty()) {
        std::fprintf(stderr, "[othello] ort 后端需 --model <fb.onnx>（spike/export_bc_onnx.py）\n");
        return 2;
    }
    if (pop_p > 0) {
        // ES 代际模式（population 路由）：路由图 + 链=个体、每个体 8 局缺省
        cfg.model.population_input = "pop";
        for (auto& d : cfg.devices) d.model.population_input = "pop";   // 已解析组同步
        if (cfg.chains == 64) cfg.chains = pop_p;
        if (cfg.games == 1280) cfg.games = pop_p * 8;
    }
    Farm farm;
    if (!farm.Init(cfg)) return 1;
    if (pop_p > 0) {
        // ---- ES 代际模式：每代换种群跑一腿 ----
        ModelSpec* sp = farm.spec();
        size_t flat_w = 0;
        for (auto& m : sp->ins)
            if (m.population) flat_w = (size_t)m.dims[1];
        if (!flat_w) {
            std::fprintf(stderr, "[othello] 模型无 population 输入（须路由图 own/opp/pop/mid）\n");
            return 2;
        }
        // 种群源：--pop-file（[P, flat_w] f32 raw）或进程内确定性生成
        std::vector<float> pop_file_buf;
        const float* file_base = nullptr;
        if (!pop_file.empty()) {
            FILE* f = fopen(pop_file.c_str(), "rb");
            if (!f) { std::fprintf(stderr, "[othello] pop 文件打不开: %s\n", pop_file.c_str()); return 2; }
            pop_file_buf.resize((size_t)pop_p * flat_w);
            if (fread(pop_file_buf.data(), 4, pop_file_buf.size(), f) != pop_file_buf.size()) {
                std::fprintf(stderr, "[othello] pop 文件尺寸不符（须 %zu f32）\n", pop_file_buf.size());
                fclose(f);
                return 2;
            }
            fclose(f);
            file_base = pop_file_buf.data();
        }
        PopCtx pctx{pop_p};
        std::vector<float> pop((size_t)pop_p * flat_w);
        double gen_total = 0;
        for (int g = 0; g < gens; g++) {
            if (file_base) {
                pop.assign(file_base, file_base + pop.size());   // 每代同种群（驱动侧可自行扰动）
            } else {
                // 确定性扰动种群：个体 p = LCG(gen, p) 生成的 N(0, 0.05)-ish
                for (int p = 0; p < pop_p; p++) {
                    uint32_t st = 0x9E3779B9u ^ (uint32_t)g * 2654435761u ^ (uint32_t)p * 40503u;
                    float* row = pop.data() + (size_t)p * flat_w;
                    for (size_t e = 0; e < flat_w; e++) {
                        st = st * 1664525u + 1013904223u;
                        row[e] = ((float)(st >> 8) / (float)0x00FFFFFF - 0.5f) * 0.1f;
                    }
                }
            }
            if (!farm.SetPopulation(pop.data())) {
                std::fprintf(stderr, "[othello] SetPopulation 失败\n");
                return 1;
            }
            double sec = farm.RunLeg(MakeRoutedAdapter, &pctx);
            const FarmTally& t = farm.tally();
            gen_total += sec;
            std::printf("[es] 代 %d/%d: %d 局 / %.3fs（%d 个体×%d 局；累计 %.3fs）"
                        " 综合 %d/%d；决策 %lld；指纹 %016llx\n",
                        g + 1, gens, t.games_done, sec, pop_p,
                        t.games_done / (t.games_done ? pop_p : 1),
                        gen_total, t.first_wins + t.second_wins,
                        t.first_total + t.second_total, (long long)t.decisions,
                        (unsigned long long)t.fingerprint);
            std::fflush(stdout);
        }
        return 0;
    }
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
