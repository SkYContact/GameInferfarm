// probe_grpwait.cpp — SgWaitOthers 解链判决实验（90k 墙嫌疑定谳，2026-09-30）
//
// 单组 N 银行 TRT 成对状态行农场（state_toy 真引擎），W 工人线程各持一链
// 直驱 Claim/SubmitWait，测 dec/s。A/B 面：旧 SgWaitOthers=每批等全体他席
// 最近状态事件（银行流两两握手=串行链）；新=按池粒度+EventQuery 探完成
// 跳过。银行数>1 且批小批密时两案吞吐应可分辨。
//
// 用法: probe_grpwait [banks=8] [workers=8] [seconds=3] [slots=8] [groups=1]
// groups>1：每组 banks/G 家银行+独立池（主/池两组形态），工人按 wid%G 分组
// ——量单调度台/单收割串行多组的扩展性（per-group 拆分的判决前提）。
// GPU 协议：~/gpu_lock.sh acquire 后再跑，数字只在锁内有效。
#include "inferfarm/bank.h"
#include "inferfarm/backend_factory.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using namespace inferfarm;

int main(int argc, char** argv) {
    const int banks = argc > 1 ? atoi(argv[1]) : 8;
    const int workers = argc > 2 ? atoi(argv[2]) : 8;
    const double secs = argc > 3 ? atof(argv[3]) : 3.0;
    const int slots = argc > 4 ? atoi(argv[4]) : 8;
    const int groups = argc > 5 ? atoi(argv[5]) : 1;

    std::vector<InferBackend*> bes;
    for (int g = 0; g < groups; g++) {
        InferBackend* be = CreateTrtBackend();
        if (!be) { std::fprintf(stderr, "[probe] trt 后端不可用\n"); return 1; }
        bes.push_back(be);
    }
    ModelConfig mc;
    mc.backend = "trt";
    mc.engine_path = slots > 8 ? "models/state_toy.fb256.trt"
                               : "models/state_toy.fb8.trt";
    mc.state_pairs.push_back({"S_prev", "S_next"});
    mc.state_pool_rows = workers;
    ModelSpec spec;
    if (!bes[0]->LoadSpec(mc, slots, spec)) {
        std::fprintf(stderr, "[probe] LoadSpec 失败\n");
        return 1;
    }
    BankConfig bc;
    bc.banks = banks;
    bc.slots = slots;
    bc.window_ms = bc.window_floor = 0.2;
    BankScheduler sched;
    sched.Bind(*bes[0], nullptr);
    std::vector<BankGroupCfg> gs;
    for (int g = 0; g < groups; g++) {
        BankGroupCfg gcfg;
        gcfg.be = bes[(size_t)g];
        gcfg.model = mc;
        gcfg.banks = banks / groups;
        gcfg.slots = 0;
        gcfg.spec = spec;
        gs.push_back(gcfg);
    }
    if (!sched.InitGroups(bc, gs, &spec)) {
        std::fprintf(stderr, "[probe] 银行制启动失败\n");
        return 1;
    }

    std::atomic<long long> decs{0}, fails{0};
    std::atomic<bool> stop{false};
    double dummy[8] = {0};
    std::vector<std::thread> ws;
    for (int w = 0; w < workers; w++)
        ws.emplace_back([&, w] {
            const int g = w % groups;
            float out[1];
            OutputDest dests[1];
            dests[0].name = "policy";
            dests[0].dst = out;
            dests[0].n = 1;
            size_t rb = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                int bank = -1, slot = -1;
                if (!sched.Claim(bank, slot, g, w)) {
                    fails.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                // x 行：链号+低位翻转（真数据面；state 累加防常数折叠）
                if (void* xr = sched.InputRow(bank, slot, "x", &rb)) {
                    char* x = (char*)xr;
                    float v[4] = {1e-3f * (float)(w + 1), 0, 0,
                                  (float)(decs.load() & 1)};
                    memcpy(x, v, rb < sizeof(v) ? rb : sizeof(v));
                }
                if (!sched.SubmitWait(bank, slot, dests, 1))
                    fails.fetch_add(1, std::memory_order_relaxed);
                else
                    decs.fetch_add(1, std::memory_order_relaxed);
                dummy[w & 7] += out[0];
            }
        });
    std::this_thread::sleep_for(
        std::chrono::duration<double>(secs));
    stop.store(true);
    for (auto& t : ws) t.join();
    sched.Shutdown();
    // 防优化
    double s = 0;
    for (double v : dummy) s += v;
    for (auto& b : bes) delete b;
    std::printf("[probe] dec/s=%.0f fails=%lld (banks=%d workers=%d slots=%d "
                "groups=%d sink=%.3f)\n",
                (double)decs.load() / secs, fails.load(), banks, workers,
                slots, groups, s);
    return fails.load() > 0 ? 2 : 0;
}
