// ============================================================
//  inferfarm/census.h — 取证层（资产4）
//
//  纪律（fiber_census_contract 2026-09-21）：
//    · 全原子计数器 + 专职低频打印线程（100ms/行），不在热路径加锁——
//      不污染测量；census 自身 ~5% 税，仅取证时开；
//    · 状态机转移唯一属主：spawn→READY；工人取走→RUNNING；挂起→WAIT；
//      Post→READY；收卷→DONE；
//    · X = live − R − Q − W = 失踪人口，恒 0 是硬不变量（两级排队定谳案
//      的那把尺）；W 再拆 warr（到达/银行内未发车）与 wpipe（在飞未回信）；
//    · 复活路径直方图（投递→工人取走）：0.25ms 线性桶×256+溢出桶；
//    · 工人忙闲（busy/idle ns 累计）+ 进程 CPU 差分；
//    · 腿末：直方图汇总 + 线程级 CPU 普查（tid 表分组：工人/调度台/打印/
//      驱动杂项——CUDA 上下文线程、EP 线程池的 CPU 只有这里量得到）。
// ============================================================
#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>

namespace inferfarm {

class Census {
public:
    // ---- fiber 状态机（FiberPool 写；打印线程读）----
    std::atomic<int> live{0};          // 在册局 fiber（spawn++/收卷--）
    std::atomic<int> state[4];         // [1]=READY [2]=RUNNING [3]=WAIT
    // ---- W 拆账（银行层钩子写）----
    std::atomic<long long> arr{0};     // 已提交未发车的行（银行舱内）
    std::atomic<long long> pipe{0};    // 已发车未回信的行（在飞）
    // ---- 复活直方图（投递→工人取走，µs）----
    static const int kHistN = 257;     // 0.25ms×256 + 溢出桶
    std::atomic<long long> hist[kHistN];
    std::atomic<long long> rev_n{0}, rev_sum{0};
    // ---- per-worker（工人线程单写；打印线程原子读）----
    static const int kMaxWorkers = 512;
    std::atomic<int> q_len[kMaxWorkers];          // 就绪数快照
    std::atomic<uint64_t> busy_ns[kMaxWorkers];   // 忙累计
    std::atomic<uint64_t> idle_ns[kMaxWorkers];   // 闲累计
    // ---- 线程普查登记（tid 表：SetThreadDescription 实测本机不生效，tid 零依赖）----
    unsigned long tids_worker[kMaxWorkers];
    int tids_worker_n = 0;
    unsigned long tid_disp = 0, tid_printer = 0;
    // ---- 银行调度台分段（ns；调度台单写 relaxed，打印/汇总线程读——原子
    //      消除与 StopPrinter 的正式数据竞争）----
    std::atomic<long long> seg_wait_ns{0}, seg_poll_ns{0}, seg_close_ns{0};
    std::atomic<long long> seg_dep_disp_ns{0}, seg_dep_self_ns{0};
    std::atomic<long long> seg_harvest_ns{0}, seg_rot_ns{0}, seg_iter_ns{0};
    std::atomic<long long> seg_iter_n{0}, seg_disp_n{0}, seg_self_dep_n{0};
    // ---- 工人侧（原子；多工人累加）----
    std::atomic<long long> claim_n{0};
    std::atomic<long long> claim_try_ns{0}, claim_zero_ns{0};
    std::atomic<long long> claim_spin_ns{0}, claim_park_ns{0};
    std::atomic<long long> sub_n{0}, sub_ns{0};
    std::atomic<long long> copyslot_ns{0};
    std::atomic<long long> self_dep{0};

    // 选通与生命周期（FARM_CENSUS=1 / Farm 配置；默认关=各点一次可预测分支）
    bool on = false;

    void ResetLeg();                   // 腿清零（多腿进程内逐腿复位）
    void StartPrinter(std::chrono::steady_clock::time_point t0);  // 起 100ms 打印线程
    void StopPrinter();                // 停打印线程 + 腿末汇总（直方图/忙闲）
    void DumpThreads();                // 腿末线程级 CPU 普查（工人 join 前调）

    // 状态机转移（属主转移点调用；off=零开销立即返回）
    void OnSpawn();
    void OnPick(int worker, uint64_t ts_post_us);   // READY→RUNNING + 复活样
    void OnSuspend();                  // RUNNING→WAIT
    void OnPost(uint64_t& ts_post_out);// WAIT→READY（写投递时刻）
    void OnDone();                     // RUNNING→DONE + live--
    void OnArrPush() { if (on) arr.fetch_add(1); }
    void OnArrPop(int n) { if (on) { arr.fetch_sub(n); pipe.fetch_add(n); } }
    void OnPipeDone(int n) { if (on) pipe.fetch_sub(n); }

    static uint64_t NowUs();

private:
    void* printer_ = nullptr;   // CensusPrinter（内部）
};

// 便捷构造：环境 FARM_CENSUS=1 → on=true
Census* CensusGlobal();

} // namespace inferfarm
