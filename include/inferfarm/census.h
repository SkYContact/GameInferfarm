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
    std::atomic<int> q_peak[kMaxWorkers];         // 就绪数峰值（投递点 CAS 抬升
                                                  // ——100ms 快照会漏峰，复活
                                                  // 滞留=队深×回合的队深证据）
    std::atomic<int> ready_peak{0};               // 全局 READY 峰值（OnPost 维护）
    static const int kQHistN = 8;                 // 取走时剩余队深分布桶
    std::atomic<long long> qhist[kQHistN];        // 0/1/2/3/4-7/8-15/16-31/32+
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
    // 收割细分（唤醒链拆账，2026-09-30）：批 harvest 段内部——memcpy 逐行
    // 送回 vs FiberPost 逐行唤醒分开计时；post_n=真实回投行数（futex 假设
    // 的分母面：真 wake 应≈工件工人数而非行数）
    std::atomic<long long> seg_harv_copy_ns{0}, seg_harv_post_ns{0};
    std::atomic<long long> seg_harv_n{0}, seg_harv_post_n{0};
    std::atomic<long long> seg_harv_samp_n{0};   // 细分样本批数（抽样均分母）
    // FiberPost 三段细分（唤醒链定谳第二刀）：钩子/锁+入队/唤醒。
    // **细档分级（DATA14 判决：全量逐行三段计时=2.3× 观测税）**：细分计时
    // 仅在 fine 档打——fine=2 全量（FARM_CENSUS=2，取证短开）；fine=1 抽样
    // 1/256（FARM_CENSUS=1 缺省，样本均打印，post_samp_n=分母）；fine=0 不打。
    // 粗档总量计数（fetch_add 面）不受分级影响=恒全量。
    std::atomic<long long> post_hook_ns{0}, post_lock_ns{0}, post_wake_ns{0};
    std::atomic<long long> post_samp_n{0};   // 细分样本数（抽样均的分母）
    int fine = 0;                            // 0=关 1=抽样1/256 2=全量
    inline bool FineSample() {               // 热路径判定（单分支+线程局部位与）
        if (fine == 2) return true;
        if (fine != 1) return false;
        static thread_local uint32_t s = 0;
        return (++s & 0xFF) == 0;            // 1/256
    }
    std::atomic<long long> seg_iter_n{0}, seg_disp_n{0}, seg_self_dep_n{0};
    // ---- 工人侧（原子；多工人累加）----
    std::atomic<long long> claim_n{0};
    std::atomic<long long> claim_try_ns{0}, claim_zero_ns{0};
    std::atomic<long long> claim_spin_ns{0}, claim_park_ns{0};
    std::atomic<long long> sub_n{0}, sub_ns{0};
    std::atomic<long long> copyslot_ns{0};
    std::atomic<long long> self_dep{0};
    std::atomic<long long> susp_n{0};  // 挂起总次数（每决策纤程成本拆账的分母面）
    // ---- 乘客侧分段（第二刀：每决策量子=乘客份额 adv/coll/asm + 框架份额
    //      claim/提交/收割——farm.cpp 相位包裹；fiber 形态下 adv 干净，
    //      monolith 形态（advance 内 Claim/SubmitWait）adv 含等待，拆账以
    //      fiber 形态为准）----
    std::atomic<long long> seg_adv_ns{0}, seg_adv_n{0};    // AdvanceToDecision
    std::atomic<long long> seg_coll_ns{0}, seg_coll_n{0};  // CollectOutputs
    std::atomic<long long> seg_asm_ns{0}, seg_asm_n{0};    // AssembleInto

    // ---- 链钟（第三刀：链墙钟去向全埋点。链生命周期每段都在属主转移点
    //      入账，闭合等式 wall = run+ready+park+infer+other 恒成立）：
    //      wall  = 首局点火 → 末局收卷（OnChainSpawn/OnChainDone 记两端）
    //      run   = 工人切入→让出/收卷（在工人上真执行）
    //      ready = 入就绪队列→工人取走（首跑=点火→首取；复活=投递→取走）
    //      park  = 挂起中"等银行出池槽"（Claim 池空背压）
    //      infer = 挂起中"等推理在飞回信"（SubmitWait）
    //      other = 其余挂起（未知等待点——新挂起点忘传原因时在这里现形）----
    static const int kMaxChains = 16384;  // 超界链不记账（钩子侧钳掉，不崩；
                                          // 掼蛋高等待者世界 16384 链档，DATA14）
    std::atomic<long long> ch_wall_ns[kMaxChains];
    std::atomic<long long> ch_run_ns[kMaxChains];
    std::atomic<long long> ch_ready_ns[kMaxChains];
    std::atomic<long long> ch_park_ns[kMaxChains];
    std::atomic<long long> ch_infer_ns[kMaxChains];
    std::atomic<long long> ch_other_ns[kMaxChains];
    std::atomic<uint64_t> ch_spawn_us[kMaxChains];   // 首局点火时刻
    int ch_n = 0;                                    // 本腿链数（RunLeg 报备）
    void ChainLegBegin(int chains);                  // RunLeg 开头报备链数并清零
    void OnChainSpawn(int chain);                    // 首局入队（记点火时刻）
    void OnChainDone(int chain);                     // 末局收卷（wall 闭合）
    void OnChainRun(int chain, long long ns);
    void OnChainReady(int chain, long long ns);
    void OnChainWait(int chain, int reason, long long ns);  // reason=FiberWaitReason

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
    void NoteQLen(int worker);         // q_len++ 后调：CAS 抬本工人峰值（多投递
                                      // 线程并发，非单写者——必须 CAS）
    // 乘客侧三段累加（off=零开销立即返回）
    void OnAdv(long long d) { if (on) { seg_adv_ns.fetch_add(d, std::memory_order_relaxed);
                                        seg_adv_n.fetch_add(1, std::memory_order_relaxed); } }
    void OnColl(long long d) { if (on) { seg_coll_ns.fetch_add(d, std::memory_order_relaxed);
                                         seg_coll_n.fetch_add(1, std::memory_order_relaxed); } }
    void OnAsm(long long d) { if (on) { seg_asm_ns.fetch_add(d, std::memory_order_relaxed);
                                        seg_asm_n.fetch_add(1, std::memory_order_relaxed); } }
    void OnArrPush() { if (on) arr.fetch_add(1); }
    void OnArrPop(int n) { if (on) { arr.fetch_sub(n); pipe.fetch_add(n); } }
    void OnPipeDone(int n) { if (on) pipe.fetch_sub(n); }

    static uint64_t NowUs();
    static long long NowNsI();   // ns（相位包裹用；bank.cpp 有同名私有实现）

private:
    void* printer_ = nullptr;   // CensusPrinter（内部）
};

// 便捷构造：环境 FARM_CENSUS=1 → on=true
Census* CensusGlobal();

} // namespace inferfarm
