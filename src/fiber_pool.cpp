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
static thread_local int t_nosuspend = 0;           // 契约 1 断言计数（ScopedNoSuspend）
static IFiberBackend* g_be = nullptr;              // 切换后端（RunLeg 期选定，池寿命）

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
    FiWorker& w = g_fps.fiw[(size_t)t->worker];
    {
        std::lock_guard<std::mutex> lk(w.mx);
        assert(!t->queued);   // debug：同一 fiber 不得在队列中挂两条
        t->queued = true;
        w.ready.push_back(t);
    }
    w.cv.notify_one();
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
        w.ready.push_back(t);
    }
    w.cv.notify_one();
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
    for (;;) {
        FiTask* t = nullptr;
        const bool con = cen && cen->on;   // census 门（默认关=零开销）
        uint64_t t_wait0 = con ? Census::NowUs() : 0;
        {
            std::unique_lock<std::mutex> lk(w.mx);
            w.cv.wait(lk, [&w] { return w.stop || !w.ready.empty(); });
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
                         FiberFrameFn frame_fn, void* user, double stagger_ms) {
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
    // c%K=确定性亲和）
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
        if (stagger_ms > 0 && c + 1 < chains)
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
                     void* user, double stagger_ms) {
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ths;
    ThreadLegCtx base{game_fn, frame_fn, user, 0, per};
    for (int c = 0; c < chains; c++) {
        ThreadLegCtx lc = base;
        lc.chain = c;
        ths.emplace_back(ThreadChainMain, lc);
        if (stagger_ms > 0 && c + 1 < chains)
            FiSleepMs(stagger_ms);
    }
    for (auto& t : ths) t.join();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace inferfarm
