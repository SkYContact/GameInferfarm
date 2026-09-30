// census.cpp — 取证层实现（fiber_census_contract 2026-09-21 的通用化抽取）
// 纪律：全原子计数器+专职低频打印线程（100ms/行），不在热路径加锁。
#include "inferfarm/census.h"
#include "platform_compat.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#else
#include <sys/resource.h>
#include <sys/time.h>
#endif

namespace inferfarm {

// 进程 CPU 累计（打印线程 pcpu 差分尺；Windows=GetProcessTimes 内核+用户，
// POSIX=getrusage(RUSAGE_SELF) 用户+系统——同口径）。返回秒。
static double ProcCpuSeconds() {
#ifdef _WIN32
    FILETIME ft, fe, fk, fu;
    if (!GetProcessTimes(GetCurrentProcess(), &ft, &fe, &fk, &fu)) return 0;
    ULARGE_INTEGER k, u;
    k.LowPart = fk.dwLowDateTime; k.HighPart = fk.dwHighDateTime;
    u.LowPart = fu.dwLowDateTime; u.HighPart = fu.dwHighDateTime;
    return (double)(k.QuadPart + u.QuadPart) / 1e7;   // 100ns → 秒
#else
    rusage ru;
    std::memset(&ru, 0, sizeof ru);
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
    return (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1e6
         + (double)ru.ru_stime.tv_sec + (double)ru.ru_stime.tv_usec / 1e6;
#endif
}

static Census* g_census = nullptr;

Census* CensusGlobal() {
    if (!g_census) {
        static Census c;
        g_census = &c;
        // 默认关；env 选通（各宿主也可显式置 on）
        // 二档（DATA14）：1=粗档+细分抽样1/256（缺省取证档）；2=细档全量
        //（三段计时逐行打，2.3× 税——只在短开取证用）
        if (const char* e = getenv("FARM_CENSUS")) {
            int v = atoi(e);
            c.on = v >= 1;
            c.fine = v >= 1 ? (v >= 2 ? 2 : 1) : 0;
        }
    }
    return g_census;
}

uint64_t Census::NowUs() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

long long Census::NowNsI() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}



void Census::ResetLeg() {
    if (!on) return;
    live.store(0);
    for (int i = 0; i < 4; i++) state[i].store(0);
    arr.store(0);
    pipe.store(0);
    for (int i = 0; i < kHistN; i++) hist[i].store(0);
    rev_n.store(0);
    rev_sum.store(0);
    for (int i = 0; i < kMaxWorkers; i++) {
        q_len[i].store(0);
        q_peak[i].store(0);
        busy_ns[i].store(0);
        idle_ns[i].store(0);
    }
    ready_peak.store(0);
    for (int i = 0; i < kQHistN; i++) qhist[i].store(0);
    seg_wait_ns.store(0); seg_poll_ns.store(0); seg_close_ns.store(0);
    seg_dep_disp_ns.store(0); seg_dep_self_ns.store(0);
    seg_harvest_ns.store(0); seg_rot_ns.store(0); seg_iter_ns.store(0);
    seg_harv_copy_ns.store(0); seg_harv_post_ns.store(0);
    seg_harv_n.store(0); seg_harv_post_n.store(0);
    post_hook_ns.store(0); post_lock_ns.store(0); post_wake_ns.store(0);
    post_samp_n.store(0); seg_harv_samp_n.store(0);
    seg_iter_n.store(0); seg_disp_n.store(0); seg_self_dep_n.store(0);
    claim_n.store(0);
    claim_try_ns.store(0); claim_zero_ns.store(0);
    claim_spin_ns.store(0); claim_park_ns.store(0);
    sub_n.store(0); sub_ns.store(0);
    copyslot_ns.store(0);
    self_dep.store(0);
    susp_n.store(0);
    seg_adv_ns.store(0); seg_adv_n.store(0);
    seg_coll_ns.store(0); seg_coll_n.store(0);
    seg_asm_ns.store(0); seg_asm_n.store(0);
    ChainLegBegin(0);   // 链钟清零（ch_n=0；RunLeg 会再 ChainLegBegin(chains)）
}

void Census::OnSpawn() {
    if (!on) return;
    live.fetch_add(1);
    state[1].fetch_add(1);
}
void Census::OnPick(int worker, uint64_t ts_post_us) {
    if (!on) return;
    q_len[worker].fetch_sub(1);
    state[1].fetch_sub(1);
    state[2].fetch_add(1);
    // 取走时剩余队深分布（复活滞留=队深×回合假设的直接证据面）
    {
        int q = q_len[worker].load(std::memory_order_relaxed);
        int b = q <= 0 ? 0 : q == 1 ? 1 : q == 2 ? 2 : q == 3 ? 3
              : q < 8 ? 4 : q < 16 ? 5 : q < 32 ? 6 : 7;
        qhist[b].fetch_add(1, std::memory_order_relaxed);
    }
    if (ts_post_us) {   // 复活样：投递→取走（srv-lat 看不见的那段）
        const uint64_t d = NowUs() - ts_post_us;
        int b = (int)(d / 250);
        if (b >= kHistN - 1) b = kHistN - 1;
        hist[b].fetch_add(1);
        rev_n.fetch_add(1);
        rev_sum.fetch_add((long long)d);
    }
}
void Census::OnSuspend() {
    if (!on) return;
    susp_n.fetch_add(1, std::memory_order_relaxed);
    state[2].fetch_sub(1);
    state[3].fetch_add(1);
}
void Census::OnPost(uint64_t& ts_post_out) {
    if (!on) return;
    ts_post_out = NowUs();
    state[3].fetch_sub(1);
    int r = state[1].fetch_add(1);
    int p = ready_peak.load(std::memory_order_relaxed);
    while (r > p && !ready_peak.compare_exchange_weak(p, r, std::memory_order_relaxed)) {}
}

// ---- 链钟（墙钟去向全埋点；写侧=属主转移点单写 relaxed，腿末汇总线程读）----
void Census::ChainLegBegin(int chains) {
    if (!on) return;
    ch_n = chains > kMaxChains ? kMaxChains : chains;
    for (int c = 0; c < ch_n; c++) {
        ch_wall_ns[c].store(0);
        ch_run_ns[c].store(0);
        ch_ready_ns[c].store(0);
        ch_park_ns[c].store(0);
        ch_infer_ns[c].store(0);
        ch_other_ns[c].store(0);
        ch_spawn_us[c].store(0);
    }
}
void Census::OnChainSpawn(int chain) {
    if (!on || chain < 0 || chain >= kMaxChains) return;
    ch_spawn_us[chain].store(NowUs(), std::memory_order_relaxed);
}
void Census::OnChainDone(int chain) {
    if (!on || chain < 0 || chain >= kMaxChains) return;
    const uint64_t sp = ch_spawn_us[chain].load(std::memory_order_relaxed);
    if (sp) ch_wall_ns[chain].fetch_add((long long)(NowUs() - sp) * 1000,
                                        std::memory_order_relaxed);
}
void Census::OnChainRun(int chain, long long ns) {
    if (!on || chain < 0 || chain >= kMaxChains) return;
    ch_run_ns[chain].fetch_add(ns, std::memory_order_relaxed);
}
void Census::OnChainReady(int chain, long long ns) {
    if (!on || chain < 0 || chain >= kMaxChains) return;
    ch_ready_ns[chain].fetch_add(ns, std::memory_order_relaxed);
}
void Census::OnChainWait(int chain, int reason, long long ns) {
    if (!on || chain < 0 || chain >= kMaxChains) return;
    std::atomic<long long>& bucket = reason == 1 ? ch_park_ns[chain]
                                  : reason == 2 ? ch_infer_ns[chain]
                                  : ch_other_ns[chain];
    bucket.fetch_add(ns, std::memory_order_relaxed);
}

void Census::NoteQLen(int worker) {
    if (!on) return;
    int q = q_len[worker].load(std::memory_order_relaxed);
    int p = q_peak[worker].load(std::memory_order_relaxed);
    while (q > p && !q_peak[worker].compare_exchange_weak(p, q, std::memory_order_relaxed)) {}
}
void Census::OnDone() {
    if (!on) return;
    state[2].fetch_sub(1);
    live.fetch_sub(1);
}

// ---- 打印线程（专职低频：只读原子，不加任何热路径锁）----
struct CensusPrinter {
    std::thread th;
    std::atomic<bool> stop{false};
};

static void CensusPrinterLoop(Census* c, std::chrono::steady_clock::time_point t0,
                              std::atomic<bool>* stop, int n_workers) {
    c->tid_printer = CurrentTid();
    long long prev[Census::kHistN] = {};
    std::vector<uint64_t> prev_busy, prev_idle;
    prev_busy.assign((size_t)n_workers, 0);
    prev_idle.assign((size_t)n_workers, 0);
    double prev_pcpu = ProcCpuSeconds();
    int line = 0;
    while (!stop->load()) {
        FiSleepMs(100);
        if (stop->load()) break;
        line++;
        const int lv = c->live.load();
        const int R = c->state[2].load(), Q = c->state[1].load(), W = c->state[3].load();
        int qmin = 1 << 30, qmax = -1;
        for (int wi = 0; wi < n_workers; wi++) {
            int q = c->q_len[wi].load();
            if (q < qmin) qmin = q;
            if (q > qmax) qmax = q;
        }
        // 复活 p50/p90（本打印周期增量直方图；0.25ms 桶粒度）
        long long delta[Census::kHistN], tot = 0;
        for (int i = 0; i < Census::kHistN; i++) {
            long long cn = c->hist[i].load();
            delta[i] = cn - prev[i];
            prev[i] = cn;
            tot += delta[i];
        }
        double p50 = -1, p90 = -1;
        if (tot > 0) {
            long long acc = 0;
            for (int i = 0; i < Census::kHistN; i++) {
                acc += delta[i];
                if (p50 < 0 && acc * 2 >= tot) p50 = (i + 0.5) * 0.25;
                if (acc * 10 >= tot * 9) { p90 = (i + 0.5) * 0.25; break; }
            }
        }
        const double tm = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("[fibq] t=%.0fms running=%d ready_total=%d wait_answer=%d "
                    "unaccounted=%d qlen=%d..%d warr=%lld wpipe=%lld "
                    "rev(p50=%.2f,p90=%.2f,n=%lld)\n",
                    tm, R, Q, W, lv - R - Q - W,
                    qmin == (1 << 30) ? 0 : qmin, qmax < 0 ? 0 : qmax,
                    c->arr.load(), c->pipe.load(), p50, p90, tot);
        if (line % 10 == 0) {
            // 工人忙闲分布（周期增量）+ 进程 CPU 差分
            std::vector<double> bp;
            for (int wi = 0; wi < n_workers; wi++) {
                uint64_t b = c->busy_ns[wi].load(), id = c->idle_ns[wi].load();
                uint64_t db = b - prev_busy[(size_t)wi], di = id - prev_idle[(size_t)wi];
                prev_busy[(size_t)wi] = b;
                prev_idle[(size_t)wi] = id;
                bp.push_back(db + di > 0 ? 100.0 * (double)db / (double)(db + di) : 0.0);
            }
            std::sort(bp.begin(), bp.end());
            const double pcpu = ProcCpuSeconds();
            std::printf("[fibq] busy%%[min..max/med]=%.0f..%.0f/%.0f pcpu=%.1fms/100ms "
                        "(srv-lat 边界: submit→回信完成；rev=投递→工人取走)\n",
                        bp.front(), bp.back(), bp[bp.size() / 2],
                        (pcpu - prev_pcpu) * 1000.0);
            prev_pcpu = pcpu;
            std::fflush(stdout);
        }
    }
}

void Census::StartPrinter(std::chrono::steady_clock::time_point t0) {
    if (!on) return;
    tids_worker_n = 0;
    auto* pr = new CensusPrinter();
    int nw = kMaxWorkers;   // 打印侧按登记的工人上限扫（未注册的读 0）
    pr->th = std::thread([this, t0, pr, nw] {
        CensusPrinterLoop(this, t0, &pr->stop, nw);
    });
    printer_ = pr;
    std::printf("[census] 人口普查开跑: 100ms/行 (live=R+Q+W+X；W 拆 warr=银行舱内"
                " / wpipe=在飞未回；rev=投递→工人取走。X 必须恒 0)\n");
    std::fflush(stdout);
}

void Census::StopPrinter() {
    if (!on || !printer_) return;
    CensusPrinter* pr = (CensusPrinter*)printer_;
    pr->stop.store(true);
    if (pr->th.joinable()) pr->th.join();
    // 腿末汇总：累计复活直方图 + 工人忙闲总账
    long long tot = rev_n.load(), s = rev_sum.load();
    std::printf("[census] 复活路径（投递→工人取走）累计: n=%lld 均值=%.2fms\n",
                tot, tot > 0 ? (double)s / tot / 1000.0 : 0.0);
    long long acc = 0;
    bool p50d = false, p90d = false;
    std::printf("[census] 直方图(ms:次数)");
    for (int i = 0; i < kHistN; i++) {
        long long cn = hist[i].load();
        if (!cn) continue;
        acc += cn;
        double edge = i < kHistN - 1 ? i * 0.25 : 64.0;
        std::printf(" %s%.2f:%lld", i == kHistN - 1 ? ">" : "", edge, cn);
        if (!p50d && acc * 2 >= tot) { std::printf(" |p50=%.2f|", edge); p50d = true; }
        if (!p90d && acc * 10 >= tot * 9) { std::printf("|p90=%.2f|", edge); p90d = true; }
    }
    std::printf("\n");
    // 乘客侧三段（第二刀）：量子构成拆账的乘客列；框架列=bankprof 行的
    // claim/提交/小拷贝+本行收割。fiber 形态下 adv=纯乘客（游戏逻辑+引擎
    // 步进）；monolith 形态 adv 含银行等待（挂起期时钟照走）不作拆账依据。
    if (seg_adv_n.load() || seg_coll_n.load() || seg_asm_n.load()) {
        auto avg = [](long long ns, long long n) {
            return n > 0 ? (double)ns / 1e6 / (double)n : 0.0;
        };
        std::printf("[census] 乘客侧/决策: adv=%.4fms(n=%lld) coll=%.4fms(n=%lld)"
                    " asm=%.4fms(n=%lld)（框架侧见 bankprof 行 claim/提交/收割）\n",
                    avg(seg_adv_ns.load(), seg_adv_n.load()), seg_adv_n.load(),
                    avg(seg_coll_ns.load(), seg_coll_n.load()), seg_coll_n.load(),
                    avg(seg_asm_ns.load(), seg_asm_n.load()), seg_asm_n.load());
    }
    // 队深峰值+取走时队深分布（复活滞留取证：100ms 快照漏的峰在这里）
    {
        int top1 = 0, top2 = 0, top3 = 0;
        for (int wi = 0; wi < kMaxWorkers; wi++) {
            int p = q_peak[wi].load();
            if (p > top1) { top3 = top2; top2 = top1; top1 = p; }
            else if (p > top2) { top3 = top2; top2 = p; }
            else if (p > top3) top3 = p;
        }
        long long qt = 0;
        for (int i = 0; i < kQHistN; i++) qt += qhist[i].load();
        std::printf("[census] 队深: ready_peak=%d 工人峰值 top3=%d/%d/%d | 取走时剩余深度:",
                    ready_peak.load(), top1, top2, top3);
        static const char* qbn[kQHistN] =
            {"0", "1", "2", "3", "4-7", "8-15", "16-31", "32+"};
        for (int i = 0; i < kQHistN; i++) {
            long long v = qhist[i].load();
            if (!v) continue;
            std::printf(" %s:%.1f%%", qbn[i], qt ? 100.0 * (double)v / (double)qt : 0.0);
        }
        std::printf("（n=%lld）\n", qt);
    }
    // 链钟汇总：每链墙钟去向（run/ready/park/infer/other），闭合账
    {
        long long wsum = 0, bsum[5] = {0, 0, 0, 0, 0};   // run ready park infer other
        int nch = 0;
        double wmax = 0;
        for (int c = 0; c < ch_n; c++) {
            const long long w = ch_wall_ns[c].load();
            if (!w) continue;
            nch++;
            wsum += w;
            if (w > wmax) wmax = (double)w / 1e9;
            bsum[0] += ch_run_ns[c].load();
            bsum[1] += ch_ready_ns[c].load();
            bsum[2] += ch_park_ns[c].load();
            bsum[3] += ch_infer_ns[c].load();
            bsum[4] += ch_other_ns[c].load();
        }
        if (nch > 0) {
            const long long btot = bsum[0] + bsum[1] + bsum[2] + bsum[3] + bsum[4];
            std::printf("[clock] 链墙钟去向 n=%d 链: 墙钟均值=%.3fs/链 峰值=%.3fs | "
                        "桶合计=%.3fs 闭合缺=%.2f%%\n"
                        "[clock]   run=%.1f%% ready=%.1f%% park(等池)=%.1f%% "
                        "infer(在飞)=%.1f%% other=%.1f%%\n",
                        nch, wsum / 1e9 / nch, wmax,
                        btot / 1e9, wsum > 0 ? 100.0 * (1.0 - (double)btot / (double)wsum) : 0.0,
                        wsum ? 100.0 * bsum[0] / wsum : 0.0,
                        wsum ? 100.0 * bsum[1] / wsum : 0.0,
                        wsum ? 100.0 * bsum[2] / wsum : 0.0,
                        wsum ? 100.0 * bsum[3] / wsum : 0.0,
                        wsum ? 100.0 * bsum[4] / wsum : 0.0);
            if (bsum[4] > 0)
                std::printf("[clock] ⚠ other>0：存在未传原因的挂起点（新挂起点应接"
                            "FiberSuspend(原因)）\n");
            // 链级明细：墙钟最长的前 8 条（极差侦查：哪条链拖尾）。选择式取
            // top8 免全排序；选中即清 wall 防重复选中（腿末一次性打印）。
            const int show = nch < 8 ? nch : 8;
            std::printf("[clock] 拖尾 top%d（chain: wall | run/ready/park/infer/other %%）:",
                        show);
            for (int k = 0; k < show; k++) {
                int best = -1; long long bw = -1;
                for (int c = 0; c < ch_n; c++) {
                    long long w = ch_wall_ns[c].load();
                    if (w > bw) { bw = w; best = c; }
                }
                if (best < 0 || bw <= 0) break;
                ch_wall_ns[best].store(0);
                const long long w = bw;
                std::printf(" [%d: %.3fs %.0f/%.0f/%.0f/%.0f/%.0f]",
                            best, w / 1e9,
                            100.0 * ch_run_ns[best].load() / w,
                            100.0 * ch_ready_ns[best].load() / w,
                            100.0 * ch_park_ns[best].load() / w,
                            100.0 * ch_infer_ns[best].load() / w,
                            100.0 * ch_other_ns[best].load() / w);
            }
            std::printf("\n");
        }
    }
    double bsum = 0, isum = 0;
    int nw = 0;
    for (int wi = 0; wi < kMaxWorkers; wi++) {
        uint64_t b = busy_ns[wi].load(), id = idle_ns[wi].load();
        if (!b && !id) continue;
        nw++;
        bsum += (double)(b / 1000) / 1e6;
        isum += (double)(id / 1000) / 1e6;
    }
    std::printf("[census] 工人忙闲总账: busy=%.1fs idle=%.1fs busy%%=%.0f（%d 工人）"
                "挂起=%lld 次\n",
                bsum, isum, bsum + isum > 0 ? 100.0 * bsum / (bsum + isum) : 0.0, nw,
                susp_n.load());
    // 银行调度台分段 + 工人侧计数（若银行用过本 census）
    long long cn = claim_n.load(), sn = sub_n.load();
    if (cn > 0 || sn > 0) {
        std::printf("[bankprof-worker] 领取 n=%lld 次/行: try=%.4fms/次[清零=%.4f 自旋外=%.4f]"
                    " 等池登记=%.5fms | 快自旋=%.4fms/决策组 | 提交前段 n=%lld %.4fms/次"
                    " | 小拷贝+卸载=%.4fms/次 | 自驱发车=%lld 次"
                    " | 第三刀: 跨挂起污染=已拆除(读数不可信,见 bank.cpp 批注)\n",
                    cn,
                    cn ? (double)claim_try_ns.load() / 1e6 / cn : 0.0,
                    cn ? (double)claim_zero_ns.load() / 1e6 / cn : 0.0,
                    cn ? ((double)claim_try_ns.load() - (double)claim_zero_ns.load()) / 1e6 / cn : 0.0,
                    cn ? (double)claim_park_ns.load() / 1e6 / cn : 0.0,
                    (double)claim_spin_ns.load() / 1e6,
                    sn, sn ? (double)sub_ns.load() / 1e6 / sn : 0.0,
                    sn ? (double)copyslot_ns.load() / 1e6 / sn : 0.0,
                    seg_self_dep_n.load());
        std::fflush(stdout);
    }
    delete pr;
    printer_ = nullptr;
    std::fflush(stdout);
}

void Census::DumpThreads() {
#ifdef _WIN32
    if (!on) return;
    // 线程表=最硬的一把尺：工人/调度台/打印之外的 CPU 全归"驱动/杂"
    // （CUDA 上下文线程、EP 线程池——它们的 CPU 只有这里量得到）。
    // 须在工人 join 前调（线程活着才量得到）。
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    DWORD pid = GetCurrentProcessId();
    struct Row { DWORD tid; double cpu; const char* grp; };
    std::vector<Row> rows;
    THREADENTRY32 te;
    te.dwSize = sizeof te;
    if (Thread32First(snap, &te)) {
        do {
            if (te.th32OwnerProcessID != pid || !te.th32ThreadID) continue;
            HANDLE h = OpenThread(THREAD_QUERY_INFORMATION, FALSE, te.th32ThreadID);
            if (!h) continue;
            FILETIME c, e, k, u;
            double cpu = 0;
            if (GetThreadTimes(h, &c, &e, &k, &u)) {
                ULARGE_INTEGER kk, uu;
                kk.LowPart = k.dwLowDateTime; kk.HighPart = k.dwHighDateTime;
                uu.LowPart = u.dwLowDateTime; uu.HighPart = u.dwHighDateTime;
                cpu = (double)(kk.QuadPart + uu.QuadPart) / 1e7;   // 100ns → 秒
            }
            CloseHandle(h);
            const char* grp = "driver/misc";
            for (int i = 0; i < tids_worker_n; i++)
                if (tids_worker[i] == te.th32ThreadID) { grp = "farm-worker"; break; }
            if (te.th32ThreadID == tid_disp) grp = "bank-disp";
            else if (te.th32ThreadID == tid_printer) grp = "farm-census";
            rows.push_back({te.th32ThreadID, cpu, grp});
        } while (Thread32Next(snap, &te));
    }
    CloseHandle(snap);
    std::sort(rows.begin(), rows.end(), [](const Row& x, const Row& y) { return x.cpu > y.cpu; });
    auto grp_sum = [&](const char* g) {
        double t = 0;
        for (const Row& r : rows) if (r.grp == g) t += r.cpu;
        return t;
    };
    double wk = grp_sum("farm-worker"), disp = grp_sum("bank-disp"), prn = grp_sum("farm-census");
    double all = 0;
    for (const Row& r : rows) all += r.cpu;
    double ptotal = 0;
    {
        FILETIME c, e, k, u;
        if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) {
            ULARGE_INTEGER kk, uu;
            kk.LowPart = k.dwLowDateTime; kk.HighPart = k.dwHighDateTime;
            uu.LowPart = u.dwLowDateTime; uu.HighPart = u.dwHighDateTime;
            ptotal = (double)(kk.QuadPart + uu.QuadPart) / 1e7;
        }
    }
    std::printf("[bankprof-thread] 线程分组CPU秒: 工人=%.1f 调度台=%.1f 打印=%.1f "
                "驱动/杂=%.1f | 全线程和=%.1f 进程总=%.1f（活线程=%zu）top:\n",
                wk, disp, prn, all - wk - disp - prn, all, ptotal, rows.size());
    int shown = 0;
    for (const Row& r : rows) {
        if (shown++ >= 24) break;
        std::printf("[bankprof-thread]   tid=%-6lu %-14.14s %8.2fs\n",
                    (unsigned long)r.tid, r.grp, r.cpu);
    }
    std::fflush(stdout);
#else
    // POSIX 降级点：线程级 CPU 普查（CreateToolhelp32Snapshot 通道）未上
    // 非 Windows——进程级 pcpu（getrusage）与工人忙闲账仍在（StartPrinter
    // 打印行）；分组普查（驱动/EP 线程识别）待接 /proc/<pid>/task 统计。
    (void)0;
#endif
}

} // namespace inferfarm
