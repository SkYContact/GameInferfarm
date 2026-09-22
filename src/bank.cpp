// bank.cpp — 零拷贝槽位银行制实现（ai_infer.cpp 银行段的游戏无关抽取，
// 2026-09-22。协议/并发结构与 YGO 产线逐句同源；GPU 面改经 InferBackend。）
#include "inferfarm/bank.h"
#include "inferfarm/fiber_pool.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <intrin.h>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace inferfarm {

static double NowMsD() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static long long NowNsI() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// 等待块（线程腿；栈上，submit 返回前有效）
struct BankDone {
    std::mutex mx;
    std::condition_variable cv;
    bool done = false;
    bool fail = false;
};

struct BankReq {
    int bank = 0, slot = 0;
    void* fiber = nullptr;               // Fiber cookie（fiber 腿）
    BankDone* ldone = nullptr;           // 线程腿等待块（栈上，收割侧回填）
    OutputDest dests[8];                 // 输出投递目的地（收割侧拷贝）
    int n_dests = 0;
    double t0 = 0;
};

enum { BK_POOL = 0, BK_FILL, BK_CLOSED, BK_FLIGHT };

struct BankCtl {
    int id = 0;
    void* sess = nullptr;                // 后端会话（地址终身固定，图一夫一妻）
    std::atomic<int> cursor{0};          // 本集会游标：fetch_add 领号（出池时归零）
    std::atomic<int> inflight{0};        // 在途写手（领号前 +1 / 行写完 -1）
    std::atomic<int> state{BK_POOL};
    std::vector<BankReq*> reqs;          // 槽→req（commit 前登记；null=作废槽）
    // 在途航班（单发=线性生命周期）
    unsigned flight_seq = 0;
    int flight_n = 0;
    double flight_t0 = 0;
    bool flight_warned = false;          // 看门狗打印去重
};

struct BankScheduler::Impl {
    InferBackend* be = nullptr;
    Census* cen = nullptr;
    BankConfig cfg;
    ModelSpec spec;
    static const int kMax = 32;
    BankCtl banks[kMax];   // 含原子不可移动：定长数组（原版同款）
    std::deque<int> pool;                // 空闲银行栈（mx 护）
    std::atomic<int> fill_idx{-1};       // 当前填充银行（-1=无，领号挂起等轮转）
    std::atomic<int> waiting{0};         // 等银行的写手数（背压观测）
    std::mutex mx;
    std::condition_variable cv;
    std::deque<void*> waiters;           // 挂起等槽的 fiber cookie（轮转时投回）
    std::thread disp;
    std::atomic<bool> stop{false};
    std::atomic<bool> running{false};
    // init 握手（会话建在调度台线程上：ORT 图会话 PerThreadContext 铁律）
    std::mutex init_mx;
    std::condition_variable init_cv;
    int init_rc = 0;                     // 0=进行中 1=ok -1=fail
    // 统计（收割/打印=调度台独占；发车计数=原子（写手自驱/调度台双源））
    std::vector<double> lat;
    double stat_t0 = 0, gpu_busy_sum = 0;
    std::atomic<long long> dep_us{0};
    std::atomic<long long> drain_us{0};
    std::atomic<long long> batches{0}, rows{0};
    std::atomic<int> self_dep{0};
    int n_banks = 0;

    void Notify() {
        std::lock_guard<std::mutex> lk(mx);
        cv.notify_all();
    }
};

// ---------------- 领号（游标制）----------------
// 无等待尝试：成功即得槽（inflight 已占，行清零由本函数完成=零基组装）。
// 领号序=在途序（先占名额再领号）⇒ drain 归零时游标终态、行前缀连续。
bool BankScheduler::Claim(int& bank, int& slot) {
    if (!banks_) return false;
    Impl& I = *impl_;
    const int S = cfg_.slots;
    const long long ts0 = I.cen && I.cen->on ? NowNsI() : 0;
    auto try_claim = [&](int& b_out, int& s_out) -> bool {
        const long long tp0 = I.cen && I.cen->on ? NowNsI() : 0;
        int fi = I.fill_idx.load(std::memory_order_acquire);
        if (fi < 0) return false;
        BankCtl& b = I.banks[(size_t)fi];
        if (b.state.load(std::memory_order_acquire) != BK_FILL) return false;  // 预检（不占名额）
        b.inflight.fetch_add(1, std::memory_order_acq_rel);   // 占在途名额（占额者必完工）
        int v = b.cursor.fetch_add(1, std::memory_order_acq_rel);
        if (v >= S || b.state.load(std::memory_order_acquire) != BK_FILL) {
            // 满座/闭舱/迟来残号：让出名额重试（v 的虚增量=发车侧作废槽跳过）
            b.inflight.fetch_sub(1, std::memory_order_acq_rel);
            return false;
        }
        b_out = fi;
        s_out = v;
        // 全行清零（零基契约）：未写区与"零垫基线"逐位同——适配器的清零类
        // 组装（bc71_clear_zone 式高水位清零）依赖"行起点为零"。
        {
            const long long tz0 = I.cen && I.cen->on ? NowNsI() : 0;
            for (size_t i = 0; i < I.spec.ins.size(); i++) {
                size_t rb = 0;
                void* row = I.be->InputRow(b.sess, I.spec.ins[i].name.c_str(), v, &rb);
                if (row) memset(row, 0, rb);
            }
            if (I.cen && I.cen->on) {
                I.cen->claim_zero_ns.fetch_add(NowNsI() - tz0, std::memory_order_relaxed);
                I.cen->claim_n.fetch_add(1, std::memory_order_relaxed);
                I.cen->claim_try_ns.fetch_add(NowNsI() - tp0, std::memory_order_relaxed);
            }
        }
        if (v == 0) I.Notify();   // 只在首行通知调度台起窗（每批一次，替代每行
                                  // notify_all 风暴——高行速时调度台被叫醒风暴拖垮）
        return true;
    };
    for (int spin = 0; spin < 4000; spin++) {   // 先短自旋：调度台通常 µs 级还池
        if (try_claim(bank, slot)) {
            if (I.cen && I.cen->on)
                I.cen->claim_spin_ns.fetch_add(NowNsI() - ts0, std::memory_order_relaxed);
            return true;
        }
        _mm_pause();
    }
    if (I.cen && I.cen->on)
        I.cen->claim_spin_ns.fetch_add(NowNsI() - ts0, std::memory_order_relaxed);
    // 池空背压：fiber 腿登记 cookie（还池/轮转时 FiberPost 投回）；线程腿 cv 等。
    // 等银行数入 waiting（[bank] 行 waiting 列=背压观测）。
    for (;;) {
        const long long tp0 = I.cen && I.cen->on ? NowNsI() : 0;
        if (try_claim(bank, slot)) return true;
        if (I.stop.load()) return false;   // 停机中：不再挂起（无人会投）
        if (void* fib = FiberCurrent()) {
            {
                std::lock_guard<std::mutex> lk(I.mx);
                I.waiters.push_back(fib);
            }
            I.waiting.fetch_add(1);
            if (I.cen && I.cen->on)
                I.cen->claim_park_ns.fetch_add(NowNsI() - tp0, std::memory_order_relaxed);
            FiberSuspend();   // 恢复=调度台已出池新银行（或再试）（挂起本身不计）
            I.waiting.fetch_sub(1);
        } else {
            std::unique_lock<std::mutex> lk(I.mx);
            I.waiting.fetch_add(1);
            if (I.cen && I.cen->on)
                I.cen->claim_park_ns.fetch_add(NowNsI() - tp0, std::memory_order_relaxed);
            I.cv.wait_for(lk, std::chrono::duration<double>(0.001),
                          [&I] { return I.fill_idx.load() >= 0 || I.stop.load(); });
            I.waiting.fetch_sub(1);
            if (I.stop.load()) return false;   // 停机中：弃领（判负纪律）
        }
    }
}

void* BankScheduler::InputRow(int bank, int slot, const char* name, size_t* row_bytes) {
    if (!banks_ || bank < 0 || bank >= banks_ || !name) return nullptr;
    return impl_->be->InputRow(impl_->banks[(size_t)bank].sess, name, slot, row_bytes);
}

// ---------------- 提交与收割 ----------------
static void BankFailComplete(BankReq* r) {
    r->ldone->fail = true;
    if (r->fiber) {
        FiberPost(r->fiber);
    } else {
        {
            std::lock_guard<std::mutex> lk2(r->ldone->mx);
            r->ldone->done = true;
        }
        r->ldone->cv.notify_one();
    }
}

static void BankDrainSubmit(BankScheduler::Impl& I, BankCtl& b, bool by_disp);   // 前向

static void BankCloseAndDispatch(BankScheduler::Impl& I, BankCtl& b);   // 前向

// 提交：登记 req → inflight--（=行写完，close-drain 的完工信号）→ 唤醒 →
// 挂起等收割回投。拆卸重定向（若适配器有）由适配器在 AssembleInto 后自清。
bool BankScheduler::SubmitWait(int bank, int slot, const OutputDest* dests, int n_dests) {
    if (!banks_) return false;
    Impl& I = *impl_;
    if (bank < 0 || bank >= banks_) return false;
    if (slot < 0 || slot >= cfg_.slots) {
        // 越界槽：Claim 的 inflight 占额必须照减（占额者必完工——否则 drain 卡死）
        I.banks[(size_t)bank].inflight.fetch_sub(1, std::memory_order_acq_rel);
        I.Notify();
        return false;
    }
    const long long ts0 = I.cen && I.cen->on ? NowNsI() : 0;
    BankCtl& b = I.banks[(size_t)bank];
    BankReq* r = new BankReq();
    r->bank = bank;
    r->slot = slot;
    if (n_dests > 8)
        std::fprintf(stderr, "[bank] SubmitWait n_dests=%d 超容量 8——截断"
                     "（OutputDest 上限=BankReq::dests[8]）\n", n_dests);
    for (int i = 0; i < n_dests && i < 8; i++) r->dests[i] = dests[i];
    r->n_dests = n_dests < 8 ? n_dests : 8;
    r->t0 = NowMsD();
    BankDone done;
    r->ldone = &done;
    r->fiber = FiberCurrent();
    int st = b.state.load(std::memory_order_acquire);
    if (st == BK_FLIGHT || st == BK_POOL) {
        // 防御（协议上不可达：占在途名额者必在 drain 内完工——state 复检在
        // 领号侧，drain 又等本名额）：槽已错过本航班，失败完成防挂死（判负
        // 该次前向=既有失败纪律），计数照减。
        std::printf("[bank] 协议防御：槽 %d 赶上 state=%d（本前向判负）\n", slot, st);
        std::fflush(stdout);
        b.inflight.fetch_sub(1, std::memory_order_acq_rel);
        done.fail = true;   // 等待者=调用者本人且仍在运行——直接置败，绝不
                            // FiberPost 自己（会把在跑的 fiber 投进就绪队列
                            // =双重调度/UAF；投递只留给真正挂起的写手）
        delete r;
        return false;
    }
    b.reqs[(size_t)slot] = r;                               // 登记（先于完工信号）
    std::atomic_thread_fence(std::memory_order_release);
    int prev_inf = b.inflight.fetch_sub(1, std::memory_order_acq_rel);   // 行写完
    if (I.cen) I.cen->OnArrPush();             // W 拆账：行入舱
    I.Notify();   // 承重通知：驱动调度台窗到期/drain/收割检查节拍（实测去掉掉 20%）
    // 满座自驱快路径（满了直接发，最后完笔者就地发车）："满=人人写完"由构造
    // 成立（+1 先于领号 ⇒ 满座时全部领号者已完工或在途；我是最后一个完工者
    // ⇒ 游标满 && 在途归零 ⇒ 不等 timer 不经他人手，自己发车。CAS 输=他人已关舱。
    if (prev_inf == 1 && b.cursor.load(std::memory_order_acquire) >= cfg_.slots) {
        I.self_dep.fetch_add(1);
        BankCloseAndDispatch(I, b);
    }
    if (I.cen && I.cen->on) {   // 前段=登记+commit+自驱判定+通知（不含挂起/恢复）
        I.cen->sub_ns.fetch_add(NowNsI() - ts0, std::memory_order_relaxed);
        I.cen->sub_n.fetch_add(1, std::memory_order_relaxed);
    }
    if (r->fiber) {
        FiberSuspend();
    } else {
        std::unique_lock<std::mutex> lk(done.mx);
        done.cv.wait(lk, [&] { return done.done; });
    }
    return !done.fail;
}

void BankScheduler::Abandon(int bank, int slot) {
    if (!banks_ || bank < 0 || bank >= banks_) return;
    Impl& I = *impl_;
    BankCtl& b = I.banks[(size_t)bank];
    if (slot < 0 || slot >= cfg_.slots) return;
    b.reqs[(size_t)slot] = nullptr;                        // 作废槽：发车跳过
    b.inflight.fetch_sub(1, std::memory_order_acq_rel);    // 完工照减（drain 不堵）
    I.Notify();
}

// 收割：完成旗标到（=输出已驻留主机）→ 逐 req 拷输出+回投 → 还池。
// **拷贝承重**：还池先于游戏恢复（新批可能立即复用槽行/输出 arena），
// 适配器不得直读银行内存——dests 缓冲收割侧回填。
static void BankHarvest(BankScheduler::Impl& I, BankCtl& b) {
    for (int s = 0; s < b.flight_n; s++) {
        BankReq* r = b.reqs[(size_t)s];
        b.reqs[(size_t)s] = nullptr;
        if (!r) continue;   // 作废槽
        for (int d = 0; d < r->n_dests; d++) {
            const OutputDest& od = r->dests[d];
            int w = I.be->OutputWidth(b.sess, od.name);
            if (w <= 0 || !od.dst) continue;
            const float* src = I.be->OutputRow(b.sess, od.name, s);
            int cn = w < od.n ? w : od.n;
            if (src && cn > 0) memcpy(od.dst, src, sizeof(float) * (size_t)cn);
        }
        r->ldone->fail = false;
        if (r->fiber) {
            FiberPost(r->fiber);
        } else {
            {
                std::lock_guard<std::mutex> lk2(r->ldone->mx);
                r->ldone->done = true;
            }
            r->ldone->cv.notify_one();
        }
        if (I.cen) I.cen->OnPipeDone(1);   // W 拆账：回信出账
        I.lat.push_back(NowMsD() - r->t0);
        delete r;
    }
    I.gpu_busy_sum += NowMsD() - b.flight_t0;
    // 还池（线性生命周期；mx 护——满座自驱路径也可能回池）
    b.state.store(BK_POOL, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(I.mx);
        I.pool.push_back(b.id);
    }
    I.Notify();   // 池非空=可轮转，叫醒调度台/等池写手
}

// 闭舱+drain+发车（幂等：CAS FILL→CLOSED，输者=他人已关舱直接走）。触发者
// 二选一：①timer=调度台（window 到期或观察到满座）；②满座自驱=最后完笔
// 的写手在 −1 后发现（游标满 && 在途归零）就地发车。可从写手 fiber 或调度台
// 线程调用：后端批发射线程安全（各家独立会话），CAS 保证一家银行只发一次。
static void BankCloseAndDispatch(BankScheduler::Impl& I, BankCtl& b) {
    int expected = BK_FILL;
    if (!b.state.compare_exchange_strong(expected, BK_CLOSED,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire))
        return;   // 已被他人关舱/发车
    int fi = I.fill_idx.load(std::memory_order_acquire);
    if (fi == b.id) I.fill_idx.compare_exchange_strong(fi, -1);
    BankDrainSubmit(I, b, false);   // 写手自驱路径（计时归自驱源）
}

// drain+发车本体（闭舱两路共用）
static void BankDrainSubmit(BankScheduler::Impl& I, BankCtl& b, bool by_disp) {
    Census* cen = I.cen;
    const long long tdp0 = cen && cen->on ? NowNsI() : 0;
    struct DepProf {   // census 分段（by_disp/自驱分源）
        bool on; bool by_disp; long long t0; Census* c;
        ~DepProf() {
            if (!on) return;
            if (by_disp) {
                c->seg_dep_disp_ns.fetch_add(NowNsI() - t0, std::memory_order_relaxed);
                c->seg_disp_n.fetch_add(1, std::memory_order_relaxed);
            } else {
                c->seg_dep_self_ns.fetch_add(NowNsI() - t0, std::memory_order_relaxed);
                c->seg_self_dep_n.fetch_add(1, std::memory_order_relaxed);
            }
        }
    } dep_prof{cen && cen->on, by_disp, tdp0, cen};
    // close-drain：等在途写手计数归零即发车。组装=同步纯计算无挂起点
    // （GameAdapter 契约 1），在途必在 ~几十 µs 完工=drain 有界；不写超时/
    // 迁移路径。归零时游标已终态，有效行=前缀连续 [0,n)。
    double drain_t0 = NowMsD();
    for (long long spins = 0;; spins++) {
        if (b.inflight.load(std::memory_order_acquire) == 0) break;
        if ((spins & 0x3FFFF) == 0x3FFFF) {   // 低频诊断（~每 5ms 一次）
            double waited = NowMsD() - drain_t0;
            if (waited > 2.0) {
                std::printf("[bank] drain 长等 %.1fms: bank=%d inflight=%d cursor=%d "
                            "pool=%zu fill=%d（若持续不归零=在途计数有漏减路径）\n",
                            waited, b.id, b.inflight.load(), b.cursor.load(),
                            I.pool.size(), I.fill_idx.load());
                std::fflush(stdout);
            }
        }
        _mm_pause();
    }
    I.drain_us.fetch_add((long long)((NowMsD() - drain_t0) * 1000.0));
    int n = b.cursor.load(std::memory_order_acquire);
    if (n > I.cfg.slots) n = I.cfg.slots;
    if (n <= 0) {   // 空舱（不可达防御）：直接回池
        b.state.store(BK_POOL, std::memory_order_release);
        std::lock_guard<std::mutex> lk(I.mx);
        I.pool.push_back(b.id);
        return;
    }
    // 发车：后端前缀 h2d（近满批整块）+ 异步发射（图回放优先）——不等回程
    double th0 = NowMsD();
    unsigned seq = 0;
    if (!I.be->SubmitBatch(b.sess, n, seq)) {
        // 发射失败（不可达防御）：本批全弃答（判负纪律），银行回池
        std::printf("[bank] 批异常（发射，本批 %d 行弃答）\n", n);
        std::fflush(stdout);
        for (int s = 0; s < n; s++) {
            BankReq* r = b.reqs[(size_t)s];
            b.reqs[(size_t)s] = nullptr;
            if (!r) continue;
            BankFailComplete(r);
            if (I.cen) I.cen->OnPipeDone(1);
            delete r;
        }
        b.state.store(BK_POOL, std::memory_order_release);
        std::lock_guard<std::mutex> lk(I.mx);
        I.pool.push_back(b.id);
        return;
    }
    double tl1 = NowMsD();
    I.dep_us.fetch_add((long long)((tl1 - th0) * 1000.0));
    b.flight_seq = seq;
    b.flight_n = n;
    b.flight_t0 = tl1;
    b.state.store(BK_FLIGHT, std::memory_order_release);
    I.batches.fetch_add(1);
    I.rows.fetch_add(n);
    if (I.cen) {
        int nrq = 0;
        for (int s = 0; s < n; s++) nrq += b.reqs[(size_t)s] ? 1 : 0;
        I.cen->OnArrPop(nrq);   // W 拆账：发车（舱→在飞）
    }
    I.Notify();   // 发车后立刻叫醒调度台（收割轮询/轮转出池要尽快跟上）
}

// 出池轮转（调度台线程）：池顶出一家（游标归零=新集会）→ 投回挂起写手
static void BankTryRotate(BankScheduler::Impl& I) {
    if (I.fill_idx.load(std::memory_order_acquire) >= 0) return;
    int i = -1;
    std::deque<void*> wake;
    {
        std::lock_guard<std::mutex> lk(I.mx);
        if (!I.pool.empty()) {
            i = I.pool.front();
            I.pool.pop_front();
            wake.swap(I.waiters);
            I.cv.notify_all();
        }
    }
    if (i < 0) return;
    BankCtl& b = I.banks[(size_t)i];
    b.cursor.store(0, std::memory_order_release);   // 新集会游标（图/地址一夫一妻）
    b.state.store(BK_FILL, std::memory_order_release);
    I.fill_idx.store(i, std::memory_order_release);
    for (void* fib : wake) FiberPost(fib);
}

// ---------------- 调度台 ----------------
// 事件驱动+短轮询（在途时 200µs 兜底叫醒——完成旗标无中断，只轻轮询）。
static void BankLoop(BankScheduler::Impl& I) {
    if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST))
        std::fprintf(stderr, "[bank] 调度台 SetThreadPriority 失败 GLE=%lu（继续，"
                     "仅性能层面影响）\n", GetLastError());
    if (I.cen) I.cen->tid_disp = GetCurrentThreadId();
    double window_ms = I.cfg.window_ms;
    if (window_ms < I.cfg.window_floor) window_ms = I.cfg.window_floor;
    double window_t0 = 0;
    bool window_open = false;
    I.stat_t0 = NowMsD();
    Census* cen = I.cen;
    for (;;) {
        const long long it0 = cen && cen->on ? NowNsI() : 0;
        long long seg_t0 = it0;
        double now = NowMsD();
        // ---- 收割：轮询在途银行完成旗标（FLIGHT 看门狗：>500ms 未到）----
        for (int i = 0; i < I.n_banks; i++) {
            BankCtl& b = I.banks[(size_t)i];
            if (b.state.load(std::memory_order_acquire) != BK_FLIGHT) continue;
            if (!I.be->CompletionReached(b.sess, b.flight_seq)) {
                if (now - b.flight_t0 > 500.0 && !b.flight_warned) {
                    b.flight_warned = true;
                    std::printf("[bank] FLIGHT 看门狗: bank=%d 已 %.0fms 未回信 seq=%u"
                                "（后端段卡死排查线索）\n",
                                i, now - b.flight_t0, b.flight_seq);
                    std::fflush(stdout);
                }
                continue;
            }
            I.be->CompletionFence();
            b.flight_warned = false;
            if (cen && cen->on) {
                cen->seg_poll_ns.fetch_add(NowNsI() - seg_t0, std::memory_order_relaxed);
                const long long th0 = NowNsI();
                BankHarvest(I, b);
                cen->seg_harvest_ns.fetch_add(NowNsI() - th0, std::memory_order_relaxed);
                seg_t0 = NowNsI();
            } else {
                BankHarvest(I, b);
            }
        }
        if (cen && cen->on && seg_t0 != it0)
            cen->seg_poll_ns.fetch_add(NowNsI() - seg_t0, std::memory_order_relaxed);
        if (I.stop.load()) {
            // 唤醒挂起写手（fiber 投回队列/线程腿 notify）：它们重试 Claim
            // 见 stop 即弃领退出——否则停机即挂死。前置条件仍是"腿已全部
            // 返回"（在途 FLIGHT 不保证收割，文档见 bank.h Shutdown 注）。
            std::deque<void*> wake;
            {
                std::lock_guard<std::mutex> lk(I.mx);
                wake.swap(I.waiters);
                I.cv.notify_all();
            }
            for (void* fib : wake) FiberPost(fib);
            break;
        }
        // ---- 出池：无填充银行 → 池顶出一家 → 投回挂起写手 ----
        if (cen && cen->on) seg_t0 = NowNsI();
        BankTryRotate(I);
        if (cen && cen->on) cen->seg_rot_ns.fetch_add(NowNsI() - seg_t0, std::memory_order_relaxed);
        // ---- timer 发车（满座通常已被写手自驱；此处兜底：到期或观察到满座）----
        int fi = I.fill_idx.load(std::memory_order_acquire);
        if (fi >= 0) {
            BankCtl& b = I.banks[(size_t)fi];
            int taken = b.cursor.load(std::memory_order_acquire);
            if (taken <= 0 || b.state.load(std::memory_order_acquire) != BK_FILL) {
                window_open = false;   // 空舱/已被自驱发走：窗口重置
            } else {
                if (!window_open) {
                    window_open = true;
                    window_t0 = now;
                }
                bool full = taken >= I.cfg.slots;
                bool expired = (now - window_t0) >= window_ms;
                if (full || expired) {
                    // 关舱（CAS 输=写手已自驱）→ 先轮转开新窗（drain/提交不堵
                    // 下一窗，批间流水重叠）→ 再 drain+发车本舱
                    window_open = false;
                    const long long tc0 = cen && cen->on ? NowNsI() : 0;
                    int expected = BK_FILL;
                    if (b.state.compare_exchange_strong(expected, BK_CLOSED,
                                                        std::memory_order_acq_rel,
                                                        std::memory_order_acquire)) {
                        int fi0 = I.fill_idx.load(std::memory_order_acquire);
                        if (fi0 == b.id) I.fill_idx.compare_exchange_strong(fi0, -1);
                        BankTryRotate(I);
                        if (cen && cen->on)
                            cen->seg_close_ns.fetch_add(NowNsI() - tc0, std::memory_order_relaxed);
                        BankDrainSubmit(I, b, true);
                    } else if (cen && cen->on) {
                        cen->seg_close_ns.fetch_add(NowNsI() - tc0, std::memory_order_relaxed);
                    }
                }
            }
        } else {
            window_open = false;
        }
        // ---- 汇报（每 300 个回信一行）----
        if ((int)I.lat.size() >= 300) {
            std::sort(I.lat.begin(), I.lat.end());
            double wall = NowMsD() - I.stat_t0;
            long long nb = I.batches.load();
            std::printf("[bank] srv-lat p50=%.1fms p90=%.1fms rows/batch=%.1f cycle=%.2fms "
                        "dep=%.2f drain=%.3f gpu_flight=%.0f%% waiting=%d "
                        "self_dep=%d batches=%lld\n",
                        I.lat[I.lat.size() / 2],
                        I.lat[(I.lat.size() * 9) / 10],
                        nb ? (double)I.rows.load() / (double)nb : 0.0,
                        nb ? wall / (double)nb : 0.0,
                        nb ? (double)I.dep_us.load() / 1000.0 / (double)nb : 0.0,
                        nb ? (double)I.drain_us.load() / 1000.0 / (double)nb : 0.0,
                        wall > 0 ? 100.0 * I.gpu_busy_sum / wall : 0.0,
                        I.waiting.load(),
                        (int)I.self_dep.load(),
                        nb);
            std::fflush(stdout);
            if (cen && cen->on && nb > 0) {
                double d = (double)nb;
                double wait = (double)cen->seg_wait_ns.load() / 1e6 / d;
                double poll = (double)cen->seg_poll_ns.load() / 1e6 / d;
                double close = (double)cen->seg_close_ns.load() / 1e6 / d;
                double dep = (double)cen->seg_dep_disp_ns.load() / 1e6 / d;
                double harv = (double)cen->seg_harvest_ns.load() / 1e6 / d;
                double rot = (double)cen->seg_rot_ns.load() / 1e6 / d;
                double itn = (double)cen->seg_iter_n.load();
                double resid = (double)cen->seg_iter_ns.load() / 1e6 / d - wait - poll
                    - close - dep - harv - rot;
                if (resid < 0) resid = 0;
                std::printf("[banksched] 段/周期: wait=%.3f poll=%.3f close=%.3f 发车=%.3f"
                            " harvest=%.3f rot=%.3f resid=%.3f | Σ=%.3f vs cycle=%.3f"
                            " 迭代/批=%.1f 自驱发车/批=%.2f\n",
                            wait, poll, close, dep, harv, rot, resid,
                            wait + poll + close + dep + harv + rot + resid,
                            wall / d, itn / d,
                            (double)cen->seg_self_dep_n.load() / d);
                std::fflush(stdout);
                for (auto* ctr : {&cen->seg_wait_ns, &cen->seg_poll_ns,
                                  &cen->seg_close_ns, &cen->seg_dep_disp_ns,
                                  &cen->seg_harvest_ns, &cen->seg_rot_ns,
                                  &cen->seg_iter_ns, &cen->seg_iter_n,
                                  &cen->seg_disp_n, &cen->seg_self_dep_n})
                    ctr->store(0, std::memory_order_relaxed);
            }
            I.lat.clear();
            I.stat_t0 = NowMsD();
            I.dep_us.store(0);
            I.drain_us.store(0);
            I.batches.store(0);
            I.rows.store(0);
            I.self_dep.store(0);
            I.gpu_busy_sum = 0;
        }
        // ---- 等待：事件（commit/领号/还池）cv 叫醒；窗内等窗到期；在途兜底轮询 ----
        {
            bool any_flight = false;
            for (int i = 0; i < I.n_banks; i++)
                if (I.banks[(size_t)i].state.load(std::memory_order_acquire) == BK_FLIGHT)
                    { any_flight = true; break; }
            double wait_ms = any_flight ? 0.1 : 2.0;
            if (window_open) {
                double rem = (window_t0 + window_ms) - NowMsD();
                if (rem < 0.02) rem = 0.02;
                if (rem < wait_ms) wait_ms = rem;
            }
            const long long tw0 = cen && cen->on ? NowNsI() : 0;
            std::unique_lock<std::mutex> lk(I.mx);
            I.cv.wait_for(lk, std::chrono::duration<double>(wait_ms / 1000.0));
            if (cen && cen->on) {
                cen->seg_wait_ns.fetch_add(NowNsI() - tw0, std::memory_order_relaxed);
                cen->seg_iter_ns.fetch_add(NowNsI() - it0, std::memory_order_relaxed);
                cen->seg_iter_n.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

// ---------------- init/shutdown ----------------
bool BankScheduler::Init(const BankConfig& cfg, const ModelConfig& mcfg, ModelSpec* spec_out) {
    Shutdown();
    cfg_ = cfg;
    if (cfg.banks <= 0 || cfg.slots < 1) return false;
    if (!spec_out) {
        std::fprintf(stderr, "[bank] Init 需要 spec_out（Farm 预先 LoadSpec 的模型规格）\n");
        return false;
    }
    impl_ = new Impl();
    Impl& I = *impl_;
    I.be = &backend();
    I.cen = cen_;
    I.cfg = cfg;
    I.spec = *spec_out;   // 由 Farm 预先 LoadSpec（slots 已核）
    *spec_out = I.spec;
    if (cfg.banks > 32) I.cfg.banks = 32;
    // 调度台线程上建会话（ORT 图会话 PerThreadContext 铁律：创建/热身/回放
    // 须同线程；TRT 同规更稳）
    I.disp = std::thread([&I, &mcfg] {
        double tb0 = NowMsD();
        bool ok = true;
        int n = I.cfg.banks;
        if (n > Impl::kMax) n = Impl::kMax;
        for (int i = 0; i < n && ok; i++) {
            BankCtl& b = I.banks[(size_t)i];
            b.id = i;
            b.sess = I.be->CreateSession(mcfg, I.spec, /*for_bank=*/true);
            if (!b.sess) { ok = false; break; }
            if (!I.be->Warmup(b.sess)) { ok = false; break; }
            b.cursor.store(0);
            b.inflight.store(0);
            b.state.store(BK_POOL);
            b.reqs.assign((size_t)I.cfg.slots, nullptr);
        }
        // 图地址烧死小实验（前两家）：任一不过=拒绝银行制启动（回不去旧路径
        // 的字节安全性不赌）
        for (int i = 0; ok && i < n && i < 2; i++) {
            if (!I.be->ProbeGraph(I.banks[(size_t)i].sess)) {
                std::fprintf(stderr, "[bank] 银行 %d 图地址小实验未过——拒绝银行制启动\n", i);
                ok = false;
            }
        }
        if (!ok) {
            for (auto& b : I.banks)
                if (b.sess) {
                    I.be->DestroySession(b.sess);
                    b.sess = nullptr;   // 防 Shutdown 二次销毁（双重释放案）
                }
            std::lock_guard<std::mutex> lk(I.init_mx);
            I.init_rc = -1;
            I.init_cv.notify_all();
            return;
        }
        for (int i = 0; i < n; i++) I.pool.push_back(i);
        I.n_banks = n;
        std::fprintf(stderr, "[bank] 建池 %d 家耗时 %.0fms（pinned 直写槽+专属批图）\n",
                     n, NowMsD() - tb0);
        {
            std::lock_guard<std::mutex> lk(I.init_mx);
            I.init_rc = 1;
        }
        I.init_cv.notify_all();
        BankLoop(I);
    });
    // 等建池结果
    {
        std::unique_lock<std::mutex> lk(I.init_mx);
        I.init_cv.wait(lk, [&I] { return I.init_rc != 0; });
    }
    if (I.init_rc != 1) {
        Shutdown();
        return false;
    }
    banks_ = I.n_banks;
    std::printf("[bank] 零拷贝槽位银行就绪: 池 %d 家 × %d 槽，window=%.2fms"
                "（游标领号/直写槽/满座或闹钟发车/旗标收割/还池；前缀 h2d+每家"
                "一图；池容量=在飞上限=背压）\n",
                banks_, cfg.slots, cfg.window_ms);
    std::fflush(stdout);
    return true;
}

void BankScheduler::Shutdown() {
    if (!impl_) return;
    Impl& I = *impl_;
    {
        std::lock_guard<std::mutex> lk(I.mx);
        I.stop.store(true);
        I.cv.notify_all();
    }
    if (I.disp.joinable()) I.disp.join();
    for (auto& b : I.banks)
        if (b.sess) {
            I.be->DestroySession(b.sess);
            b.sess = nullptr;
        }
    delete impl_;
    impl_ = nullptr;
    banks_ = 0;
}

// ---------------- inline 模式运行器 ----------------
struct InlineRunner::Session {
    InferBackend* be = nullptr;
    void* sess = nullptr;
    const ModelSpec* spec = nullptr;
    std::mutex mx;
    // row0 写面（SlotWriter 实现：整批照发=垃圾行无害）
    struct Writer0 : SlotWriter {
        InferBackend* be = nullptr;
        void* sess = nullptr;
        const ModelSpec* spec = nullptr;
        void* Row(const char* name, size_t* row_bytes) override {
            return be->InputRow(sess, name, 0, row_bytes);
        }
    } writer;
};

bool InlineRunner::Init(InferBackend& be, const ModelConfig& cfg, const ModelSpec& spec) {
    Shutdown();
    s_ = new Session();
    s_->be = &be;
    s_->spec = &spec;
    s_->sess = be.CreateSession(cfg, spec, /*for_bank=*/false);
    if (!s_->sess) { Shutdown(); return false; }
    if (!be.Warmup(s_->sess)) { Shutdown(); return false; }
    s_->writer.be = &be;
    s_->writer.sess = s_->sess;
    s_->writer.spec = &spec;
    return true;
}

void InlineRunner::Shutdown() {
    if (!s_) return;
    if (s_->sess) s_->be->DestroySession(s_->sess);
    delete s_;
    s_ = nullptr;
}

bool InlineRunner::Run(GameAdapter* g) {
    if (!s_) return false;
    std::lock_guard<std::mutex> lk(s_->mx);
    // 零基：row0 清零（与银行路径 claim 清零对齐——逐位一致性前提）
    for (size_t i = 0; i < s_->spec->ins.size(); i++) {
        size_t rb = 0;
        void* row = s_->be->InputRow(s_->sess, s_->spec->ins[i].name.c_str(), 0, &rb);
        if (row) memset(row, 0, rb);
    }
    OutputDest dests[8];
    int nd = g->CollectOutputs(dests, 8);
    g->AssembleInto(s_->writer);
    unsigned seq = 0;
    if (!s_->be->SubmitBatch(s_->sess, s_->spec->slots, seq)) return false;
    while (!s_->be->CompletionReached(s_->sess, seq)) _mm_pause();
    s_->be->CompletionFence();
    for (int d = 0; d < nd; d++) {
        int w = s_->be->OutputWidth(s_->sess, dests[d].name);
        if (w <= 0 || !dests[d].dst) continue;
        const float* src = s_->be->OutputRow(s_->sess, dests[d].name, 0);
        int cn = w < dests[d].n ? w : dests[d].n;
        if (src && cn > 0) memcpy(dests[d].dst, src, sizeof(float) * (size_t)cn);
    }
    return true;
}

} // namespace inferfarm
