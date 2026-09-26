// fiber_bench.cpp — 纤程开销拆解微基准（判决实验"C++20 协程化"第 0 步）
//
// 掼蛋侧口径：纤程开销 ~12µs/决策。本基准把"每决策纤程成本"拆成可归因的
// 原语单价账（解释在后，数字先行——测量纪律）：
//   A. SwitchToFiber 往返单价（切换原语；FLOAT_SWITCH=产线旗标）
//   B. Post→pickup 全路径单价（mutex+deque+notify+cv 醒）——冷消费者（真睡
//      真醒=工人空闲形态）与热消费者（自旋不睡）分离：唤醒税 vs 队列操作税
//   C. 切换点 cache 冷却效应（往返两侧互踩 N KB 私有工作集后首触的 Reload 税
//      =多 fiber 共享一核的 L1/L2 逐出版本）
//   D. CreateFiberEx/DeleteFiber 单价（每局一次，摊到决策）
//
// 每决策成本 = 挂起次数 × (A + B + C) + D/每局决策数。
// 本基准不建模 bank 协议（那是 census 真负载的活）——只给原语单价。
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

static double g_inv_freq_ns = 0;
static double NowNs() {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * g_inv_freq_ns;
}

static unsigned volatile g_sink = 0;

// ---------------- A/C. SwitchToFiber 往返（可选双侧工作集互踩） ----------------
struct PingPongCtx {
    void* peer = nullptr;
    int touch_bytes = 0;               // 恢复后首触的私有工作集字节（0=纯切换）
    unsigned char* buf = nullptr;
};

static void TouchSum(PingPongCtx* c) {   // 触一圈缓存行（防 DCE：累加进 volatile）
    if (!c->touch_bytes) return;
    unsigned acc = 0;
    for (int i = 0; i < c->touch_bytes; i += 64) acc += c->buf[i];
    g_sink += acc;
}

static void WINAPI PingPongMain(void* p) {
    PingPongCtx* c = (PingPongCtx*)p;
    for (;;) {
        TouchSum(c);
        SwitchToFiber(c->peer);
    }
}

static void BenchSwitch(int touch_bytes, int rounds) {
    PingPongCtx peer_ctx;
    peer_ctx.touch_bytes = touch_bytes;
    PingPongCtx main_ctx;
    main_ctx.touch_bytes = touch_bytes;
    if (touch_bytes) {
        peer_ctx.buf = (unsigned char*)VirtualAlloc(nullptr, (SIZE_T)touch_bytes,
                                                    MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        main_ctx.buf = (unsigned char*)VirtualAlloc(nullptr, (SIZE_T)touch_bytes,
                                                    MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        for (int i = 0; i < touch_bytes; i += 4096) {
            peer_ctx.buf[i] = (unsigned char)i;
            main_ctx.buf[i] = (unsigned char)(i ^ 1);
        }
    }
    void* main_fib = ConvertThreadToFiberEx(nullptr, FIBER_FLAG_FLOAT_SWITCH);
    peer_ctx.peer = main_fib;
    void* peer = CreateFiberEx(0, 0, FIBER_FLAG_FLOAT_SWITCH, PingPongMain, &peer_ctx);
    for (int i = 0; i < 2000; i++) {   // 预热：页就位+TLB+分支预测
        SwitchToFiber(peer);
        TouchSum(&main_ctx);
    }
    const double t0 = NowNs();
    for (int i = 0; i < rounds; i++) {
        SwitchToFiber(peer);
        TouchSum(&main_ctx);
    }
    const double t1 = NowNs();
    DeleteFiber(peer);   // 挂起中的 peer（未运行中）——回收栈
    ConvertFiberToThread();
    if (peer_ctx.buf) VirtualFree(peer_ctx.buf, 0, MEM_RELEASE);
    if (main_ctx.buf) VirtualFree(main_ctx.buf, 0, MEM_RELEASE);
    const double per_rt = (t1 - t0) / rounds;   // ns/往返（=2 次切换；C 档含双侧首触）
    std::printf("[A/C] 往返（2 次切换%s）touch=%4dKB: %8.1f ns/往返 = %6.1f ns/切换\n",
                touch_bytes ? "，双侧首触" : "", touch_bytes / 1024, per_rt, per_rt / 2);
}

// ---------------- B. Post→pickup 全路径 ----------------
// 产线同构：FiWorker 的 mx/cv/ready 三件套 + FiberPost 的 投递→唤醒→取走。
struct BenchQueue {
    std::mutex mx;
    std::condition_variable cv;
    std::deque<long long> ready;
    bool stop = false;
};

static void BenchPostPickup(bool hot_consumer, int rounds) {
    BenchQueue q;
    std::atomic<int> ack{0};
    std::vector<double> lat((size_t)rounds);
    std::thread consumer([&] {
        if (hot_consumer) {
            int got = 0;
            while (got < rounds) {
                bool picked = false;
                {
                    std::lock_guard<std::mutex> lk(q.mx);
                    if (!q.ready.empty()) {
                        const double t0 = (double)q.ready.front();
                        q.ready.pop_front();
                        lat[(size_t)got] = NowNs() - t0;
                        picked = true;
                    }
                }
                if (picked) {
                    got++;
                    ack.store(1, std::memory_order_release);
                }
            }
            return;
        }
        int got = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(q.mx);
            q.cv.wait(lk, [&] { return q.stop || !q.ready.empty(); });
            if (q.ready.empty()) break;
            const double t0 = (double)q.ready.front();
            q.ready.pop_front();
            lk.unlock();
            lat[(size_t)got] = NowNs() - t0;
            got++;
            ack.store(1, std::memory_order_release);
            if (got >= rounds) break;
        }
    });
    Sleep(50);   // 让消费者入眠（冷档）
    for (int i = 0; i < rounds; i++) {
        const double t0 = NowNs();
        {
            std::lock_guard<std::mutex> lk(q.mx);
            q.ready.push_back((long long)t0);
        }
        q.cv.notify_one();
        while (!ack.load(std::memory_order_acquire))
            YieldProcessor();   // 串行化：每样=一次真醒（冷）/满速接力（热）
        ack.store(0, std::memory_order_release);
    }
    {
        std::lock_guard<std::mutex> lk(q.mx);
        q.stop = true;
    }
    q.cv.notify_all();
    consumer.join();
    std::sort(lat.begin(), lat.end());
    const double p10 = lat[(size_t)((double)rounds * 0.10)];
    const double p50 = lat[(size_t)((double)rounds * 0.50)];
    const double p90 = lat[(size_t)((double)rounds * 0.90)];
    const double p99 = lat[(size_t)((double)rounds * 0.99)];
    double sum = 0;
    for (double d : lat) sum += d;
    std::printf("[B] Post→pickup %s: p10=%6.0f p50=%6.0f p90=%6.0f p99=%7.0f ns（均 %6.0f）n=%d\n",
                hot_consumer ? "热自旋消费者" : "冷(cv真醒)消费者",
                p10, p50, p90, p99, sum / rounds, rounds);
}

// ---------------- B2. Post 队列操作本体（无消费者唤醒） ----------------
static void BenchPostOpOnly(int rounds) {
    BenchQueue q;
    const double t0 = NowNs();
    for (int i = 0; i < rounds; i++) {
        {
            std::lock_guard<std::mutex> lk(q.mx);
            q.ready.push_back((long long)i);
        }
        q.cv.notify_one();   // 无等待者=空操作路径
    }
    const double t1 = NowNs();
    g_sink += (unsigned)q.ready.size();
    std::printf("[B2] Post 队列操作本体（lock+push+notify 空场）: %6.0f ns/次\n",
                (t1 - t0) / rounds);
}

// ---------------- A2. fcontext 切换原语（同 harness 对照，原语门） ----------------
#if defined(_MSC_VER)
#include "../src/fcontext.h"

struct FcPing {
    inferfarm::fc::Ctx* peer = nullptr;   // 对端 ctx
    int touch_bytes = 0;
    unsigned char* buf = nullptr;
};
static inferfarm::fc::Ctx g_fc_peer_own;  // 对端自有 ctx（单对端基准专用）

static void FcTouchSum(FcPing* c) {
    if (!c->touch_bytes) return;
    unsigned acc = 0;
    for (int i = 0; i < c->touch_bytes; i += 64) acc += c->buf[i];
    g_sink += acc;
}

static void FcPeerMain(void* p) {   // 永不返回（收尾由宿主释放栈）
    FcPing* c = (FcPing*)p;
    for (;;) {
        FcTouchSum(c);
        inferfarm::fc::fi_swap(&g_fc_peer_own, c->peer);
    }
}

static void BenchSwitchFC(int touch_bytes, int rounds) {
    FcPing peer_arg;
    FcPing main_side;   // main 侧自有工作集（与 A 档 main_ctx 同形）
    peer_arg.touch_bytes = touch_bytes;
    main_side.touch_bytes = touch_bytes;
    if (touch_bytes) {
        peer_arg.buf = (unsigned char*)VirtualAlloc(nullptr, (SIZE_T)touch_bytes,
                                                    MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        main_side.buf = (unsigned char*)VirtualAlloc(nullptr, (SIZE_T)touch_bytes,
                                                     MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        for (int i = 0; i < touch_bytes; i += 4096) {
            peer_arg.buf[i] = (unsigned char)i;
            main_side.buf[i] = (unsigned char)(i ^ 1);
        }
    }
    const size_t stack_sz = 1024 * 1024;
    void* base = VirtualAlloc(nullptr, stack_sz, MEM_RESERVE | MEM_COMMIT,
                              PAGE_READWRITE);
    char* raw_top = (char*)base + stack_sz;
    void* top = (void*)((((uintptr_t)raw_top - 8) & ~(uintptr_t)15) + 8);
    inferfarm::fc::fi_make(&g_fc_peer_own, top, FcPeerMain, &peer_arg);
    inferfarm::fc::Ctx* peer = &g_fc_peer_own;
    inferfarm::fc::Ctx main_own{};
    peer_arg.peer = &main_own;
    for (int i = 0; i < 2000; i++) {
        inferfarm::fc::fi_swap(&main_own, peer);
        FcTouchSum(&main_side);
    }
    const double t0 = NowNs();
    for (int i = 0; i < rounds; i++) {
        inferfarm::fc::fi_swap(&main_own, peer);
        FcTouchSum(&main_side);
    }
    const double t1 = NowNs();
    VirtualFree(base, 0, MEM_RELEASE);
    if (peer_arg.buf) VirtualFree(peer_arg.buf, 0, MEM_RELEASE);
    if (main_side.buf) VirtualFree(main_side.buf, 0, MEM_RELEASE);
    const double per_rt = (t1 - t0) / rounds;
    std::printf("[A2] fcontext 往返（2 次切换%s）touch=%4dKB: %8.1f ns/往返 = %6.1f ns/切换\n",
                touch_bytes ? "，双侧首触" : "", touch_bytes / 1024, per_rt, per_rt / 2);
}

static void BenchCreateDeleteFC(int rounds) {
    const size_t stack_sz = 1024 * 1024;
    const double t0 = NowNs();
    for (int i = 0; i < rounds; i++) {
        void* base = VirtualAlloc(nullptr, stack_sz, MEM_RESERVE | MEM_COMMIT,
                                  PAGE_READWRITE);
        char* raw_top = (char*)base + stack_sz;
        void* top = (void*)((((uintptr_t)raw_top - 8) & ~(uintptr_t)15) + 8);
        inferfarm::fc::Ctx* c = new inferfarm::fc::Ctx{};
        inferfarm::fc::fi_make(c, top, FcPeerMain, nullptr);
        g_sink += (unsigned)(uintptr_t)c;
        delete c;
        VirtualFree(base, 0, MEM_RELEASE);
    }
    const double t1 = NowNs();
    std::printf("[D2] VirtualAlloc(1MB)+ctx+fi_make+Free: %8.0f ns/对\n",
                (t1 - t0) / rounds);
}
#endif // _MSC_VER

// ---------------- D. CreateFiberEx/DeleteFiber 单价 ----------------
static void WINAPI NullFiberMain(void*) {}   // 永不入内（只测建/删单价）

static void BenchCreateDelete(int rounds) {
    void* main_fib = ConvertThreadToFiberEx(nullptr, FIBER_FLAG_FLOAT_SWITCH);
    (void)main_fib;
    const double t0 = NowNs();
    for (int i = 0; i < rounds; i++) {
        void* f = CreateFiberEx(0, 0, FIBER_FLAG_FLOAT_SWITCH, NullFiberMain, nullptr);
        DeleteFiber(f);   // 未切入过=栈未提交，合法回收
    }
    const double t1 = NowNs();
    std::printf("[D] CreateFiberEx+DeleteFiber: %8.0f ns/对（每局一次，摊薄见正文）\n",
                (t1 - t0) / rounds);
    ConvertFiberToThread();
}

int main() {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    g_inv_freq_ns = 1e9 / (double)f.QuadPart;
    std::printf("== fiber_bench：纤程开销拆解（原语单价账）==\n");
    BenchSwitch(0, 100000);
    BenchSwitch(16 * 1024, 100000);
    BenchSwitch(64 * 1024, 100000);
    BenchSwitch(256 * 1024, 50000);
#if defined(_MSC_VER)
    BenchSwitchFC(0, 100000);
    BenchSwitchFC(64 * 1024, 100000);
    BenchCreateDeleteFC(20000);
#endif
    BenchPostPickup(false, 20000);
    BenchPostPickup(true, 20000);
    BenchPostOpOnly(200000);
    BenchCreateDelete(20000);
    std::printf("== 完 ==\n");
    return 0;
}
