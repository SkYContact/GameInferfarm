// census.cpp — 取证层实现（fiber_census_contract 2026-09-21 的通用化抽取）
// 纪律：全原子计数器+专职低频打印线程（100ms/行），不在热路径加锁。
#include "inferfarm/census.h"
#include <algorithm>
#include <cstdio>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace inferfarm {

static Census* g_census = nullptr;

Census* CensusGlobal() {
    if (!g_census) {
        static Census c;
        g_census = &c;
#ifdef _WIN32
        // 默认关；env 选通（各宿主也可显式置 on）
        if (const char* e = getenv("FARM_CENSUS")) c.on = atoi(e) == 1;
#endif
    }
    return g_census;
}

uint64_t Census::NowUs() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
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
        busy_ns[i].store(0);
        idle_ns[i].store(0);
    }
    seg_wait_ns.store(0); seg_poll_ns.store(0); seg_close_ns.store(0);
    seg_dep_disp_ns.store(0); seg_dep_self_ns.store(0);
    seg_harvest_ns.store(0); seg_rot_ns.store(0); seg_iter_ns.store(0);
    seg_iter_n.store(0); seg_disp_n.store(0); seg_self_dep_n.store(0);
    claim_n.store(0);
    claim_try_ns.store(0); claim_zero_ns.store(0);
    claim_spin_ns.store(0); claim_park_ns.store(0);
    sub_n.store(0); sub_ns.store(0);
    copyslot_ns.store(0);
    self_dep.store(0);
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
    state[2].fetch_sub(1);
    state[3].fetch_add(1);
}
void Census::OnPost(uint64_t& ts_post_out) {
    if (!on) return;
    ts_post_out = NowUs();
    state[3].fetch_sub(1);
    state[1].fetch_add(1);
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
    c->tid_printer = GetCurrentThreadId();
    long long prev[Census::kHistN] = {};
    std::vector<uint64_t> prev_busy, prev_idle;
    prev_busy.assign((size_t)n_workers, 0);
    prev_idle.assign((size_t)n_workers, 0);
    double prev_pcpu = 0;
    {
        FILETIME ft, fe, fk, fu;
        if (GetProcessTimes(GetCurrentProcess(), &ft, &fe, &fk, &fu)) {
            ULARGE_INTEGER k, u;
            k.LowPart = fk.dwLowDateTime; k.HighPart = fk.dwHighDateTime;
            u.LowPart = fu.dwLowDateTime; u.HighPart = fu.dwHighDateTime;
            prev_pcpu = (double)(k.QuadPart + u.QuadPart) / 10000.0;
        }
    }
    int line = 0;
    while (!stop->load()) {
        Sleep(100);
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
            double pcpu = 0;
            {
                FILETIME ft, fe, fk, fu;
                if (GetProcessTimes(GetCurrentProcess(), &ft, &fe, &fk, &fu)) {
                    ULARGE_INTEGER k, u;
                    k.LowPart = fk.dwLowDateTime; k.HighPart = fk.dwHighDateTime;
                    u.LowPart = fu.dwLowDateTime; u.HighPart = fu.dwHighDateTime;
                    pcpu = (double)(k.QuadPart + u.QuadPart) / 10000.0;
                }
            }
            std::printf("[fibq] busy%%[min..max/med]=%.0f..%.0f/%.0f pcpu=%.1fms/100ms "
                        "(srv-lat 边界: submit→回信完成；rev=投递→工人取走)\n",
                        bp.front(), bp.back(), bp[bp.size() / 2], pcpu - prev_pcpu);
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
    double bsum = 0, isum = 0;
    int nw = 0;
    for (int wi = 0; wi < kMaxWorkers; wi++) {
        uint64_t b = busy_ns[wi].load(), id = idle_ns[wi].load();
        if (!b && !id) continue;
        nw++;
        bsum += (double)(b / 1000) / 1e6;
        isum += (double)(id / 1000) / 1e6;
    }
    std::printf("[census] 工人忙闲总账: busy=%.1fs idle=%.1fs busy%%=%.0f（%d 工人）\n",
                bsum, isum, bsum + isum > 0 ? 100.0 * bsum / (bsum + isum) : 0.0, nw);
    // 银行调度台分段 + 工人侧计数（若银行用过本 census）
    long long cn = claim_n.load(), sn = sub_n.load();
    if (cn > 0 || sn > 0) {
        std::printf("[bankprof-worker] 领取 n=%lld 次/行: try=%.4fms/次[清零=%.4f 自旋外=%.4f]"
                    " 等池登记=%.5fms | 快自旋=%.4fms/决策组 | 提交前段 n=%lld %.4fms/次"
                    " | 小拷贝+卸载=%.4fms/次 | 自驱发车=%lld 次\n",
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
#endif
}

} // namespace inferfarm
