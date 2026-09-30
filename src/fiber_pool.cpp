// fiber_pool.cpp — 纤程池实现（ai_opp_loop.cpp FiWorker/FiTask/RunLegFibers 的
// 游戏无关抽取，2026-09-22。行为与 YGO 产线逐句同源：链-工人亲和、切换点装卸
// 帧、唤醒队列投递、错峰点火。）
//
// 切换后端可插拔（fiber_backend.h 五原语）：Windows=winfiber（现役）/
// fcontext 双选；POSIX x86_64=fcontext（Linux 移植面，2026-09-27——池机器
// 本体 std::thread/mutex/cv 全可移植，OS 触点只剩 tid 与睡眠，见
// platform_compat.h；工人绑核在非 Windows=降级跳过，affinity.cpp）。
#include "inferfarm/fiber_pool.h"
#include "inferfarm/affinity.h"
#include "inferfarm/census.h"
#include "fiber_backend.h"
#include "platform_compat.h"
#include <cassert>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace inferfarm {

// ---------------- 内部结构（FiTask/FiWorker/FiChain 原名原样）----------------
struct FiChain {                        // 链级共享（=线程模式的链线程栈角色）
    ITlsFrame* frame = nullptr;         // 链寿命帧（可为 null）
    int chain = 0, per = 0;
    void* user = nullptr;
};
struct alignas(64) FiTask {             // 一局一 fiber（64B 对齐：多 fiber 热字
                                        // 段不共享缓存行——P1-6 伪共享隔离）
    void* fiber = nullptr;
    FiChain* ch = nullptr;
    int gi = 0;
    int worker = 0;                     // 亲和工人（点火定终身，不迁移）
    void* sched = nullptr;              // 工人主 fiber（让出的回归目标）
    bool finished = false;              // fiber 体收尾置位（工人侧收卷判据）
    bool queued = false;                // debug：已在就绪队列（投递-挂起不变量：
                                        // 双重投递断言；w.mx 内读写=无竞争）
    uint64_t ts_post = 0;               // census：投递时刻（µs；w.mx 护送建立
                                        // happens-before，取走侧读）
    // 链钟埋点（cen->on 时有效；均在属主转移点单写）：
    uint64_t ts_spawn = 0;              // 入就绪队列时刻（首跑 ready 的起点）
    uint64_t ts_run0 = 0;               // 最近一次被工人取走时刻（run 段起点）
    uint64_t ts_susp = 0;               // 最近一次让出时刻（挂起段起点）
    int susp_reason = 0;                // FWait_*（挂起期入哪个桶）
};
struct FiWorker {
    std::mutex mx;
    std::condition_variable cv;
    std::deque<FiTask*> ready;          // 唤醒队列（per-worker）
    bool stop = false;
    std::atomic<bool> sleeping{false};  // 工人在 cv 睡眠中（TSAN 定谳转正：
                                        // FiberPostEnd 无锁读=真 race 面；
                                        // 语义仍同前论证——锁内读写+End 侧
                                        // 最坏多一次 notify，但存储必须是
                                        // atomic 防 O1 寄存器缓存）（唤醒跳过旗，DATA14
                                        // 回执后追加：投递侧跳过叫醒已醒工人
                                        // ——YGO 忙世界工人恒醒，notify 全是
                                        // 白打）。同步性：置位/清位/读（投递
                                        // 非批路径）全在 mx 临界区内=happens-
                                        // before 闭合，漏唤醒不可能；唯一
                                        // 无锁读点=FiberPostEnd（push-CS 之后
                                        // 同线程读），最坏=对已醒工人多发一次
                                        // notify（无害；用户拍板"极端多一次
                                        // 唤醒没啥"）。纯自旋档永不置位=投递
                                        // 零 notify（自旋工人轮询天然可见）。
    void* main_fib = nullptr;
};

struct FiberPoolState {
    std::deque<FiWorker> fiw;           // deque：mutex/cv 不可移动，元素原地构造
    std::vector<FiChain> fich;          // 腿寿命（索引=链号）
    std::mutex mx;
    std::condition_variable cv;
    int done = 0, total = 0;            // 收卷局数/总局数（mx 护）
    std::atomic<int> workers_ready{0};
    FiberGameFn game_fn = nullptr;
    FiberFrameFn frame_fn = nullptr;
    Census* cen = nullptr;
    std::vector<int> worker_aff;        // FARM_WORKER_AFFINITY 解析（RunLeg 开头
                                        // 刷新；工人线程内只读——绑核 id 模分配）
};
static FiberPoolState g_fps;
static thread_local FiTask* t_fi_task = nullptr;   // 本工人当前局（等待侧桥取 cookie）
// 批模式投递（FiberPostBegin/End；kMaxWorkers=512 → 512 位=8×u64 位图）
static thread_local bool t_post_batch = false;
static thread_local uint64_t t_post_wake[8] = {0, 0, 0, 0, 0, 0, 0, 0};
// 批模式按工人分桶（2026-09-30 唤醒链刀）：Begin/End 之间 FiberPost 只入
// 桶，End 按工人**一次锁批量入队**——原形态每 fiber 各过一次 w.mx（掼蛋
// rot 唤段 0.155ms/轮转 ≈ 384 席×0.4µs 逐席锁的账），批量化后每工人一次
// 锁+N 次 push_back。桶生命周期=投递线程（收割/调度台各持一份）。
static thread_local std::vector<std::vector<FiTask*>> t_post_q;
static thread_local int t_nosuspend = 0;           // 契约 1 断言计数（ScopedNoSuspend）
static IFiberBackend* g_be = nullptr;              // 切换后端（RunLeg 期选定，池寿命）
static inline void SpinPause() {                   // x86 PAUSE（SMT 对端不阻塞执行口）
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
    _mm_pause();   // MSVC：GCC builtin 不可用（09-30 合流首编译爆，老坑律）
#else
    __builtin_ia32_pause();
#endif
#else
    ;
#endif
}
#if defined(_MSC_VER)
#include <intrin.h>
static inline int Ctz64(unsigned long long m) {    // GCC __builtin_ctzll 的 MSVC 等价
    unsigned long i;
    _BitScanForward64(&i, m);
    return (int)i;
}
#else
static inline int Ctz64(unsigned long long m) { return __builtin_ctzll(m); }
#endif

// ---------------- 等待侧桥（银行层/任何等待点调用）----------------
void* FiberCurrent() { return t_fi_task; }

void FiberSuspend(FiberWaitReason why) {
    // 让出：Switch 回本工人调度器（恢复点=FiberPost 投递后工人再切入；
    // 恢复即结果就绪——收割侧先拷输出后投递）
    if (!t_fi_task) return;   // 线程腿误调=无操作（防御）
    assert(t_nosuspend == 0);   // 契约 1：组装直写槽窗口（ScopedNoSuspend
                                 // 只包 AssembleInto）内挂起=适配器违约
    FiTask* t = t_fi_task;
    if (g_fps.cen && g_fps.cen->on) {
        g_fps.cen->OnSuspend();
        // 链钟：run 段（工人切入→本让出）闭合；挂起段起点+原因登记
        const uint64_t now = Census::NowUs();
        if (t->ts_run0) {
            g_fps.cen->OnChainRun(t->ch->chain,
                                  (long long)(now - t->ts_run0) * 1000);
            t->ts_run0 = 0;
        }
        t->ts_susp = now;
        t->susp_reason = (int)why;
    }
    g_be->Switch(t->sched);
    // 恢复点：工人取走时已把状态翻回 RUNNING（见工人循环取走处）
}

void FiberPost(void* cookie) {
    // 收割侧投递：该局 fiber 进其所属工人的就绪队列+唤醒（唤醒队列本体）。
    // 投递-挂起不变量见 fiber_pool.h——此处断言防双重投递（目标尚未被取走
    // 时二次 Post=它会恢复两次=逻辑错）。
    FiTask* t = (FiTask*)cookie;
    if (g_fps.cen && g_fps.cen->on) {
        // 链钟：挂起段（让出→投递=答案就绪）按原因入账。ts_susp=0（登记→
        // 挂起窗口内被投回，本 fiber 还没真正让出）时无挂起段可记。
        if (t->ts_susp) {
            g_fps.cen->OnChainWait(t->ch->chain, t->susp_reason,
                                   (long long)(Census::NowUs() - t->ts_susp) * 1000);
            t->ts_susp = 0;
        }
        g_fps.cen->OnPost(t->ts_post);   // WAIT→READY + 投递时刻
        g_fps.cen->q_len[t->worker].fetch_add(1);
        g_fps.cen->NoteQLen(t->worker);  // 峰值 CAS（投递线程并发）
    }
    Census* cen = g_fps.cen;
    // 细分计时分级门（DATA14：全量逐行=2.3× 税）：fine 档抽样/全量才付
    // 时钟读+簿记；粗档只留下方 push 面的 fetch_add（无条件门旁路）
    const bool prof = cen && cen->on && cen->FineSample();
    const long long p0 = prof ? Census::NowNsI() : 0;
    FiWorker& w = g_fps.fiw[(size_t)t->worker];
    bool need_wake;
    if (t_post_batch) {   // 批模式：入桶不锁（FiberPostEnd 按工人一次锁批量入队）
        if (t_post_q.size() < g_fps.fiw.size()) t_post_q.resize(g_fps.fiw.size());
        t_post_q[(size_t)t->worker].push_back(t);
        t_post_wake[(size_t)t->worker >> 6] |= uint64_t(1) << (t->worker & 63);
        if (prof) {
            cen->post_hook_ns.fetch_add(Census::NowNsI() - p0,
                                        std::memory_order_relaxed);
            cen->post_samp_n.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }
    {
        std::lock_guard<std::mutex> lk(w.mx);
        const long long p1 = prof ? Census::NowNsI() : 0;
        assert(!t->queued);   // debug：同一 fiber 不得在队列中挂两条
        // 唤醒收敛（harvest 拆账判决 2026-09-30）：只有空→非空转变才
        // notify——Mesa 语义下队列非空=工人不可能在睡（谓词在锁内复检），
        // 多余的 notify_one 在工人停着时每次都是真 futex wake 系统调用
        //（strace 实锤每 post 一发 WAKE；批模式 FiberPostEnd 收敛 3×）。
        need_wake = w.sleeping.load(std::memory_order_acquire);   // 只叫醒在睡的（旗在锁内读写=无漏唤醒；
                                  // 已醒工人循环里必然再查队列）
        t->queued = true;
        w.ready.push_back(t);
        const long long p2 = prof ? Census::NowNsI() : 0;
        if (prof) cen->post_lock_ns.fetch_add(p2 - p1, std::memory_order_relaxed);
        if (need_wake) w.cv.notify_one();
        if (prof) {
            const long long p3 = Census::NowNsI();
            cen->post_hook_ns.fetch_add(p1 - p0, std::memory_order_relaxed);
            cen->post_wake_ns.fetch_add(p3 - p2, std::memory_order_relaxed);
            cen->post_samp_n.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void FiberPostBegin() {
    t_post_batch = true;
    for (auto& m : t_post_wake) m = 0;
    if (t_post_q.size() < g_fps.fiw.size()) t_post_q.resize(g_fps.fiw.size());
    for (auto& q : t_post_q) q.clear();
}

void FiberPostEnd() {
    t_post_batch = false;
    // 批量入队（唤醒链刀 2026-09-30）：按工人一次锁搬整桶——原形态每 fiber
    // 一次 lock/push/unlock。锁内仍逐条 assert+queued 置位（语义与逐条同）；
    // FIFO 序保持（桶内=投递序，桶间按工人独立队列无跨序）。
    for (size_t w = 0; w < t_post_q.size(); w++) {
        auto& q = t_post_q[w];
        if (q.empty()) continue;
        FiWorker& fw = g_fps.fiw[w];
        {
            std::lock_guard<std::mutex> lk(fw.mx);
            for (FiTask* t : q) {
                assert(!t->queued);   // debug：同一 fiber 不得在队列中挂两条
                t->queued = true;
                fw.ready.push_back(t);
            }
        }
        q.clear();
        if (fw.sleeping.load(std::memory_order_acquire))
            fw.cv.notify_one();
    }
    for (size_t i = 0; i < 8; i++) t_post_wake[i] = 0;
}

// ---------------- 契约 1 机器校验（作用域内挂起=断言）----------------
ScopedNoSuspend::ScopedNoSuspend() { ++t_nosuspend; }
ScopedNoSuspend::~ScopedNoSuspend() { --t_nosuspend; }

// ---------------- fiber 体与点火 ----------------
static void FiSpawnGame(FiChain* ch, int gi, int wid);

static void FI_API FiGameMain(void* p) {
    FiTask* tk = (FiTask*)p;
    FiChain* ch2 = tk->ch;
    g_fps.game_fn(ch2->chain, tk->gi, ch2->user);
    // 链续跑：未到 per 在同工人点火下一局（链内串行=线程模式同构；
    // 创建于工人=同链不跨工人）。到 per=链收卷。
    if (tk->gi + 1 < ch2->per)
        FiSpawnGame(ch2, tk->gi + 1, tk->worker);
    if (g_fps.cen && g_fps.cen->on) {
        g_fps.cen->OnDone();   // RUNNING→DONE + live--
        if (tk->gi + 1 >= ch2->per)
            g_fps.cen->OnChainDone(ch2->chain);   // 末局收卷：链墙钟闭合
    }
    tk->finished = true;
    g_be->Switch(tk->sched);   // 不归路（工人侧 Destroy；函数返回=杀线程）
}

static void FiSpawnGame(FiChain* ch, int gi, int wid) {
    // 创建于工人：fiber 由当前线程创建（首局=点火线程、后续=收卷局所在工人），
    // 只投 wid 工人队列=链-工人亲和。
    FiTask* t = new FiTask();
    t->ch = ch;
    t->gi = gi;
    t->worker = wid;
    t->sched = g_fps.fiw[(size_t)wid].main_fib;
    if (!t->sched) {
        // 工人 ConvertThreadToFiberEx 失败（已退线程）：本链无人供职——与
        // CreateFiberEx 失败同路：按剩余局记完成，防收卷谓词挂死
        std::printf("[fiber] 工人 %d 转换失败已退役（链 %d 局 %d 按已收卷计）\n",
                    wid, ch->chain, gi);
        std::fflush(stdout);
        delete t;
        {
            std::lock_guard<std::mutex> lk(g_fps.mx);
            g_fps.done += ch->per - gi;
        }
        g_fps.cv.notify_all();
        return;
    }
    t->fiber = g_be->Create(FiGameMain, t);
    if (!t->fiber) {
        std::printf("[fiber] Create 失败（后端 %s；链 %d 局 %d，本链放弃=按已收卷计）\n",
                    g_be->name(), ch->chain, gi);
        std::fflush(stdout);
        delete t;
        // 该链后续局不再点火——按剩余局数记完成，防收卷谓词挂死
        {
            std::lock_guard<std::mutex> lk(g_fps.mx);
            g_fps.done += ch->per - gi;
        }
        g_fps.cv.notify_all();
        return;
    }
    FiWorker& w = g_fps.fiw[(size_t)wid];
    bool need_wake;
    {
        std::lock_guard<std::mutex> lk(w.mx);
        // census 入册在入队之前（锁内）：工人取走前必已 live++/READY，
        // 消灭"先减后加"瞬时 -1 的 X≠0 假警报
        if (g_fps.cen && g_fps.cen->on) {
            g_fps.cen->OnSpawn();
            if (gi == 0) g_fps.cen->OnChainSpawn(ch->chain);   // 链钟：点火时刻
            g_fps.cen->q_len[wid].fetch_add(1);
            g_fps.cen->NoteQLen(wid);
        }
        t->ts_spawn = (g_fps.cen && g_fps.cen->on) ? Census::NowUs() : 0;
        need_wake = w.ready.empty();   // 空→非空才 notify（同 FiberPost 收敛）
        w.ready.push_back(t);
    }
    if (need_wake) w.cv.notify_one();
}

static void FiWorkerLoop(int wid, Census* cen) {
    FiWorker& w = g_fps.fiw[(size_t)wid];
    if (!g_fps.worker_aff.empty())
        PinThread(g_fps.worker_aff, wid, "worker");
    if (cen && cen->on && cen->tids_worker_n < Census::kMaxWorkers)
        cen->tids_worker[cen->tids_worker_n++] = CurrentTid();
    w.main_fib = g_be->ConvertThread();
    if (!w.main_fib) {
        std::printf("[fiber] 工人 %d ConvertThread 失败（后端 %s）\n",
                    wid, g_be->name());
        g_fps.workers_ready.fetch_add(1);
        return;
    }
    g_fps.workers_ready.fetch_add(1);   // 点火线程等到全体转换完（sched 指针就绪）
    // 纯自旋档（FARM_WORKER_SPIN=1，2026-09-30 判决）：空转 SpinPause
    //（x86 PAUSE——SMT 对端不阻塞执行口）不进 futex 睡眠，收割唤醒降为
    // 纯内存可见性（futex wait/wake 一对 ~5µs 系统调用对消失）。闲时烧
    // 一个核的Retired 位（用户拍板：农场机独占，烧得起）。stop 检查每
    // 千圈一次（volatile 读，无锁）；退出条件与 cv 档同=stop 且队列空。
    const bool spin_mode = [] {
        const char* e = std::getenv("FARM_WORKER_SPIN");
        return e && std::atoi(e) == 1;
    }();
    for (;;) {
        FiTask* t = nullptr;
        const bool con = cen && cen->on;   // census 门（默认关=零开销）
        uint64_t t_wait0 = con ? Census::NowUs() : 0;
        if (spin_mode) {
            for (int it = 0;; it++) {
                {
                    std::lock_guard<std::mutex> lk(w.mx);
                    if (!w.ready.empty()) { t = w.ready.front(); w.ready.pop_front(); t->queued = false; break; }
                    if (w.stop && w.ready.empty() && (it & 1023) == 1023) break;
                }
                for (int k = 0; k < 64; k++) SpinPause();
            }
            if (!t && w.ready.empty()) break;   // stop 且队列空：收工
        } else {
            std::unique_lock<std::mutex> lk(w.mx);
            w.sleeping.store(true, std::memory_order_release);   // 睡眠旗
            w.cv.wait(lk, [&w] { return w.stop || !w.ready.empty(); });
            w.sleeping.store(false, std::memory_order_release);  // wait 返回=持锁
            if (w.ready.empty()) break;   // stop 且队列空：收工
            t = w.ready.front();
            w.ready.pop_front();
            t->queued = false;   // debug 旗标：出队（FiberPost 断言的另一半）
        }
        const uint64_t t_run0 = con ? Census::NowUs() : 0;
        if (con) {
            // 闲段（cv 等待）入账 + 取走转移 READY→RUNNING + 复活样（投递→取走）
            cen->idle_ns[wid].fetch_add((t_run0 - t_wait0) * 1000);
            cen->OnPick(wid, t->ts_post);
            // 链钟：ready 段（入队→取走；首跑=ts_spawn、复活=ts_post）闭合
            const uint64_t rdy0 = t->ts_post ? t->ts_post : t->ts_spawn;
            if (rdy0)
                cen->OnChainReady(t->ch->chain, (long long)(t_run0 - rdy0) * 1000);
            t->ts_post = 0;
            t->ts_spawn = 0;
            t->ts_run0 = t_run0;   // run 段起点（让出/收卷处闭合）
        }
        // 切换点装卸（首跑/恢复同路）：帧=链寿命 → 同链跨局携带=线程模式语义
        t_fi_task = t;
        if (t->ch->frame) t->ch->frame->Install();
        g_be->Switch(t->fiber);
        if (t->ch->frame) t->ch->frame->Uninstall();
        t_fi_task = nullptr;
        if (t->finished) {
            if (con)   // 链钟：末段 run 闭合（收卷不经 FiberSuspend）
                cen->OnChainRun(t->ch->chain,
                                (long long)(Census::NowUs() - t->ts_run0) * 1000);
            g_be->Destroy(t->fiber);
            delete t;
            {
                std::lock_guard<std::mutex> lk(g_fps.mx);
                g_fps.done++;
            }
            g_fps.cv.notify_all();
        }
        // else：让出于等待点，FiberPost 投回本队列
        if (con)   // 忙段（含 fiber 全部执行）入账
            cen->busy_ns[wid].fetch_add((Census::NowUs() - t_run0) * 1000);
    }
    g_be->ConvertBack(w.main_fib);
}

void FiberPool::Configure(int workers, Census* census) {
    if (workers <= 0) {
        int hc = (int)std::thread::hardware_concurrency();
        workers = hc >= 2 ? hc / 2 : 1;   // 物理核≈hc/2（SMT 负资产判决）
    }
    if (workers > Census::kMaxWorkers)   // census 定长数组容量（防 OOB）
        workers = Census::kMaxWorkers;
    workers_ = workers;
    census_ = census;
}

double FiberPool::RunLeg(int chains, int per, FiberGameFn game_fn,
                         FiberFrameFn frame_fn, void* user, double stagger_ms,
                         int stagger_batch) {
    const int K = workers_;
    g_fps.worker_aff = ParseCpuList(std::getenv("FARM_WORKER_AFFINITY"));
    if (!g_fps.worker_aff.empty()) {
        std::printf("[fiber] 工人绑核 %d 项（K=%d 模分配）\n",
                    (int)g_fps.worker_aff.size(), K);
        std::fflush(stdout);
    }
    auto t0 = std::chrono::steady_clock::now();
    g_be = FiberBackendSelect();   // 每腿选定（env 可覆写；缺省=构建档决定）
    g_fps.fiw.clear();                   // FiWorker 含 mutex/cv：deque 原地构造免移动
    for (int w = 0; w < K; w++) g_fps.fiw.emplace_back();
    g_fps.done = 0;
    g_fps.total = 0;
    g_fps.workers_ready.store(0);
    g_fps.game_fn = game_fn;
    g_fps.frame_fn = frame_fn;
    g_fps.cen = census_;
    if (census_ && census_->on) census_->ChainLegBegin(chains);   // 链钟：报备本腿链数
    g_fps.fich.clear();
    g_fps.fich.resize((size_t)chains);
    std::vector<std::thread> wth;
    for (int w = 0; w < K; w++)
        wth.emplace_back(FiWorkerLoop, w, census_);
    // 等全体工人转 fiber 完（FiSpawnGame 要读 main_fib）
    while (g_fps.workers_ready.load() < K)
        FiSleepMs(1);
    // 逐链点火（错峰=stagger 语义：同步起跑=到达层羊群灾难；链 c → 工人
    // c%K=确定性亲和）。stagger_batch：每 tick 连点几条链再睡——4096 链×
    // 逐条睡 1ms 时点火拖 4s+（活口被掐死的假象来源）；批化后点火时长=
    // (chains/batch)×stagger。羊群度=batch×K 同时到达，batch=1 保持原语义。
    if (stagger_batch < 1) stagger_batch = 1;
    int spawned = 0;
    for (int c = 0; c < chains; c++) {
        FiChain& ch = g_fps.fich[(size_t)c];
        ch.chain = c;
        ch.per = per;
        ch.user = user;
        ch.frame = frame_fn ? frame_fn(c, user) : nullptr;
        g_fps.total += per;
        FiSpawnGame(&ch, 0, c % K);
        spawned++;
        if (stagger_ms > 0 && c + 1 < chains && spawned % stagger_batch == 0)
            FiSleepMs(stagger_ms);
    }
    std::printf("[fiber] %d 条链已点火（K=%d 工人，后端=%s，每局一 fiber，链-工人亲和 c%%K）\n",
                spawned, K, g_be->name());
    // 等全部局收卷
    {
        std::unique_lock<std::mutex> lk(g_fps.mx);
        g_fps.cv.wait(lk, [] { return g_fps.done >= g_fps.total; });
    }
    for (auto& w : g_fps.fiw) {
        {
            std::lock_guard<std::mutex> lk(w.mx);
            w.stop = true;
        }
        w.cv.notify_all();
    }
    for (auto& t : wth) t.join();
    // 帧所有权=调用方（适配器自持=随适配器回收；池不 delete——YGO 原版帧是
    // 独立堆对象，泛化后适配器常以成员地址作帧）
    g_fps.fich.clear();
    g_fps.fiw.clear();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ---------------- 线程模式腿（同构对照路径）----------------
struct ThreadLegCtx {
    FiberGameFn game_fn;
    FiberFrameFn frame_fn;
    void* user;
    int chain, per;
};

static void ThreadChainMain(ThreadLegCtx c) {
    // 线程模式同享 FARM_WORKER_AFFINITY（与 fiber 工人同一旋钮；链号=线程
    // 身份 id，模分配语义一致。每链解析：µs 级，与 fiber 路的动态刷新同语义）
    const std::vector<int> aff = ParseCpuList(std::getenv("FARM_WORKER_AFFINITY"));
    if (!aff.empty()) PinThread(aff, c.chain, "chain");
    ITlsFrame* frame = c.frame_fn ? c.frame_fn(c.chain, c.user) : nullptr;
    if (frame) frame->Install();   // 链寿命=线程寿命（所有权=调用方，不 delete）
    for (int gi = 0; gi < c.per; gi++)
        c.game_fn(c.chain, gi, c.user);
}

double RunLegThreads(int chains, int per, FiberGameFn game_fn, FiberFrameFn frame_fn,
                     void* user, double stagger_ms, int stagger_batch) {
    if (stagger_batch < 1) stagger_batch = 1;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ths;
    ThreadLegCtx base{game_fn, frame_fn, user, 0, per};
    for (int c = 0; c < chains; c++) {
        ThreadLegCtx lc = base;
        lc.chain = c;
        ths.emplace_back(ThreadChainMain, lc);
        if (stagger_ms > 0 && c + 1 < chains && (c + 1) % stagger_batch == 0)
            FiSleepMs(stagger_ms);
    }
    for (auto& t : ths) t.join();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace inferfarm
