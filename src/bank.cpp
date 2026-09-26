// bank.cpp — 零拷贝槽位银行制实现（ai_infer.cpp 银行段的游戏无关抽取，
// 2026-09-22。协议/并发结构与 YGO 产线逐句同源；GPU 面改经 InferBackend。）
#include "inferfarm/bank.h"
#include "inferfarm/affinity.h"
#include "inferfarm/fiber_pool.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

namespace inferfarm {

// 自旋等待原语（x86=PAUSE 指令；其余平台退化为让出——收口 POSIX 编译面，
// 语义不变：完成旗标轮询的读侧减速）
static inline void SpinPause() {
#if defined(_MSC_VER)
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

// 自旋/诊断节拍（原 4000/0x3FFFF 魔法数命名；来源=本机扫描：短自旋覆盖
// 调度台 µs 级还池的绝大多数，溢出才走挂起路径）
static constexpr int kClaimSpins = 4000;        // Claim 短自旋上限
static constexpr long long kDrainDiagMask = 0x3FFFF;   // drain 长等诊断打印分频
static constexpr int kInlineSpinBeforeYield = 4000;    // inline 完成等待转让出

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
    // 输出申报上限（与 BankScheduler::kMaxOutputDests 一致；超过=失败完成）
    static constexpr int kMaxDests = BankScheduler::kMaxOutputDests;
    int bank = 0, slot = 0;
    void* fiber = nullptr;               // Fiber cookie（fiber 腿）
    BankDone* ldone = nullptr;           // 线程腿等待块（栈上，收割侧回填）
    OutputDest dests[kMaxDests];         // 输出投递目的地（收割侧拷贝）
    int n_dests = 0;
    double t0 = 0;
};

enum { BK_POOL = 0, BK_FILL, BK_CLOSED, BK_FLIGHT };

struct alignas(64) BankCtl {            // 64B 对齐：相邻银行的 cursor/inflight/
                                        // state 热原子不与他行共享缓存行
                                        //（P1-6 伪共享隔离，2026-09-24 审计）
    int id = 0;
    int grp = 0;                          // 设备组号（多 GPU 判决15）
    InferBackend* be = nullptr;           // 组后端（会话仍按银行隔离）
    void* sess = nullptr;                // 后端会话（地址终身固定，图一夫一妻）
    int slots = 0;                        // 本银行批形状（=组模型 dim0；异构
                                          // 批形状：慢卡小图，游标/窗满/越界
                                          // 三处界全按此，不按全局）
    std::atomic<int> cursor{0};          // 本集会游标：fetch_add 领号（出池时归零）
    std::atomic<int> inflight{0};        // 在途写手（领号前 +1 / 行写完 -1）
    std::atomic<int> state{BK_POOL};
    std::vector<BankReq*> reqs;          // 槽→req（commit 前登记；null=作废槽）
    std::vector<BankReq> req_pool;       // 槽→req 对象池（与 reqs 一一对应；init
                                         // 期一次分配——req 与槽一一独占：领号
                                         // 串行发号+银行线性生命周期 ⇒ 同槽同时
                                         // 至多一个活 req，无需堆分配）
    // 槽基址预解（热路径去虚调用+名字串扫，2026-09-24 审计 P0）：行指针=
    // 基址+slot×row_bytes（InputRow 线性契约，三后端同式）。population 面
    // =nullptr（不走预解，回退后端直查）
    std::vector<char*> in_rows;                       // 下标=I.spec.ins 下标
    std::vector<std::pair<const char*, size_t>> in_idx;   // 名字指针→ins 下标
    // 在途航班（单发=线性生命周期）。非原子字段，同步边=state：
    // 写侧（发车者）先写 flight_* 再 state.store(FLIGHT, release)；
    // 读侧（收割）state.load(FLIGHT, acquire) 后读——release/acquire 配对。
    // 发射失败分支不复位：字段仅在 FLIGHT 态被读，回池后下批发车前重写。
    unsigned flight_seq = 0;
    int flight_n = 0;
    double flight_t0 = 0;
    bool flight_warned = false;          // 看门狗打印去重
};

struct BankScheduler::Impl {
    static const int kMax = 32;
    static const int kMaxGrp = 8;        // 设备组上限（双卡/显卡+核显均 ≤8 绰绰）
    Census* cen = nullptr;
    BankConfig cfg;
    ModelSpec spec;
    BankCtl banks[kMax];   // 含原子不可移动：定长数组（原版同款）
    int n_groups = 1;                     // 设备组数（池/领号/窗均按组分列）
    std::deque<int> pool_g[kMaxGrp];      // 各组空闲银行栈（mx 护）
    std::atomic<int> fill_idx[kMaxGrp];   // 各组当前填充银行（-1=无，领号挂起等轮转）
    std::deque<void*> waiters_g[kMaxGrp]; // 各组挂起等槽的 fiber cookie（轮转时投回）
    std::atomic<int> waiting{0};         // 等银行的写手数（背压观测；跨组合计）
    std::mutex mx;
    std::condition_variable cv;
    std::thread disp;
    std::vector<int> sched_aff;          // FARM_SCHED_AFFINITY 解析（Init 填、
                                         // disp 线程开头消费——spin=1 自旋核
                                         // 钉扎防迁移，判决17/18）
    void* notify_sem = nullptr;          // WMO 通知信号量（spin=2；Notify 释放，
                                         // 调度台 WMO 消费；Shutdown 兜底唤醒）
    std::atomic<bool> stop{false};
    std::atomic<bool> running{false};
    // init 握手（会话建在调度台线程上：ORT 图会话 PerThreadContext 铁律）
    std::mutex init_mx;
    std::condition_variable init_cv;
    int init_rc = 0;                     // 0=进行中 1=ok -1=fail
    // 统计（收割/打印=调度台独占；发车计数=原子（写手自驱/调度台双源））
    std::vector<double> lat;
    std::vector<int> bsz;                // 批大小样本（发车侧记；分布=稀释定律
                                         // 的测量面：p10 深度稀批即到达率绑定）
    std::vector<double> fl_ms;           // 每批在飞时长样本（发车→收割；围栏税
                                         // 与批延迟的分布面，均值 gpu_flight 之外）
    double stat_t0 = 0, gpu_busy_sum = 0;
    std::atomic<long long> dep_us{0};
    std::atomic<long long> drain_us{0};
    std::atomic<long long> batches{0}, rows{0};
    std::atomic<int> self_dep{0};
    int spin = 0;                        // 等待模式（BankConfig.spin：0/1；
                                         // 2=混合已判死拆除，判决17）
    int n_banks = 0;
    // HR waitable timer 等待面（FARM_BANK_HRTIMER=1，2026-09-24）：缺省
    // 路径的 cv.wait_for 受 ~1ms 定时量子税（判决3），换 高分辨率定时器
    // （Win10 1803+，实测过冲 50-600µs 文献口径）+调度台唤醒事件（cv.notify
    // 的 WMO 镜像——Notify 集中镜像+持锁点裸镜像）。opt-in 默认关=零行为差。
    void* hr_timer = nullptr;            // HR 定时器（自动重置窗闹钟）
    void* wake_ev = nullptr;             // 手动重置唤醒事件（notify 镜像）
    bool hr_mode = false;

    void Notify() {
        std::lock_guard<std::mutex> lk(mx);
        cv.notify_all();
#ifdef _WIN32
        if (notify_sem) ReleaseSemaphore((HANDLE)notify_sem, 1, nullptr);   // spin=2 WMO 唤醒
        if (wake_ev) SetEvent((HANDLE)wake_ev);   // HR timer 路径即时叫醒
#endif
    }
};

// ---------------- 领号（游标制）----------------
// 无等待尝试：成功即得槽（inflight 已占，行清零由本函数完成=零基组装）。
// 领号序=在途序（先占名额再领号）⇒ drain 归零时游标终态、行前缀连续。
// dev=设备组门：只在本组的填充银行领号（链→组钉扎=异构逐位钥匙）。
bool BankScheduler::Claim(int& bank, int& slot, int dev) {
    if (!banks_) return false;
    Impl& I = *impl_;
    const int g = dev < 0 ? 0 : dev;
    if (g >= Impl::kMaxGrp || g >= I.n_groups) return false;
    const long long ts0 = I.cen && I.cen->on ? NowNsI() : 0;
    auto try_claim = [&](int& b_out, int& s_out) -> bool {
        const long long tp0 = I.cen && I.cen->on ? NowNsI() : 0;
        int fi = I.fill_idx[g].load(std::memory_order_acquire);
        if (fi < 0) return false;
        BankCtl& b = I.banks[(size_t)fi];
        const int S = b.slots;   // 界=本银行形状（异构批形状：慢卡小图）
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
        // 组装（高水位清零式）依赖"行起点为零"。基址走预解表（空=回退直查）
        {
            const long long tz0 = I.cen && I.cen->on ? NowNsI() : 0;
            for (size_t i = 0; i < I.spec.ins.size(); i++) {
                if (I.spec.ins[i].population) continue;   // population 面不清零
                if (b.in_rows.empty()) {   // 预解未就绪防御：退回直查（语义同）
                    size_t rb = 0;
                    void* row = b.be->InputRow(b.sess, I.spec.ins[i].name.c_str(), v, &rb);
                    if (row) memset(row, 0, rb);
                    continue;
                }
                char* base = b.in_rows[i];
                if (base)
                    memset(base + (size_t)v * I.spec.ins[i].row_bytes, 0,
                           I.spec.ins[i].row_bytes);
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
    for (int spin = 0; spin < kClaimSpins; spin++) {   // 先短自旋：调度台通常 µs 级还池
        if (try_claim(bank, slot)) {
            if (I.cen && I.cen->on)
                I.cen->claim_spin_ns.fetch_add(NowNsI() - ts0, std::memory_order_relaxed);
            return true;
        }
        SpinPause();
    }
    if (I.cen && I.cen->on)
        I.cen->claim_spin_ns.fetch_add(NowNsI() - ts0, std::memory_order_relaxed);
    // 池空背压：fiber 腿登记 cookie（还池/轮转时 FiberPost 投回）；线程腿 cv 等。
    // 等银行数入 waiting（[bank] 行 waiting 列=背压观测）。
    // ⚠ 投递-挂起不变量（FiberPost 契约，fiber_pool.h）：登记→FiberSuspend 之间
    // 若调度台已把本 fiber 投回（BankTryRotate 的 wake swap 发生在此窗口），
    // 投递只是把 cookie 挂进就绪队列——本 fiber 随后的 FiberSuspend 被该记录
    // 唤醒=恰好一次恢复，不存在双重调度。前提=登记后无条件挂起（中途 return
    // =工人重复切入已收卷 fiber=UAF——两条登记点[此处/waiters 池]均满足）。
    for (;;) {
        const long long tp0 = I.cen && I.cen->on ? NowNsI() : 0;
        if (try_claim(bank, slot)) return true;
        if (I.stop.load()) return false;   // 停机中：不再挂起（无人会投）
        if (void* fib = FiberCurrent()) {
            {
                std::lock_guard<std::mutex> lk(I.mx);
                I.waiters_g[g].push_back(fib);
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
                          [&I, g] { return I.fill_idx[g].load() >= 0 || I.stop.load(); });
            I.waiting.fetch_sub(1);
            if (I.stop.load()) return false;   // 停机中：弃领（判负纪律）
        }
    }
}

int BankScheduler::GroupOf(int bank) const {
    if (!impl_ || bank < 0 || bank >= banks_) return 0;
    return impl_->banks[(size_t)bank].grp;
}

void* BankScheduler::InputRow(int bank, int slot, const char* name, size_t* row_bytes) {
    if (!banks_ || bank < 0 || bank >= banks_ || !name) return nullptr;
    BankCtl& b = impl_->banks[(size_t)bank];
    // 预解只读快路径（多工人线程并发安全——表 init 期填满后不可变）：
    // 名字→下标（指针同址优先，回退内容比较）→基址+slot×row_bytes。
    // population/未知名=不在表 → 回退后端直查（语义与直查完全一致）
    for (auto& e : b.in_idx) {
        if (!(e.first == name || (e.first && std::strcmp(e.first, name) == 0)))
            continue;
        char* base = b.in_rows[(size_t)e.second];
        if (!base) break;   // population 面 → 直查
        size_t rb = impl_->spec.ins[(size_t)e.second].row_bytes;
        if (row_bytes) *row_bytes = rb;
        return base + (size_t)slot * rb;
    }
    return b.be->InputRow(b.sess, name, slot, row_bytes);   // 按银行取组后端
}

bool BankScheduler::SetPopulation(int bank, const char* pop_input, const void* host) {
    if (!banks_ || bank < 0 || bank >= banks_ || !pop_input) return false;
    BankCtl& b = impl_->banks[(size_t)bank];
    return b.be->SetPopulation(b.sess, pop_input, host);
}

// ---------------- 提交与收割 ----------------
// 失败完成：fiber 腿 SwitchToFiber 全序免锁；线程腿 fail/done 同锁内置位
// （等待方锁内读——不依赖跨锁程序顺序）
static void BankFailComplete(BankReq* r) {
    if (r->fiber) {
        r->ldone->fail = true;
        FiberPost(r->fiber);
    } else {
        {
            std::lock_guard<std::mutex> lk2(r->ldone->mx);
            r->ldone->fail = true;
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
    BankCtl& b = I.banks[(size_t)bank];
    if (slot < 0 || slot >= b.slots) {
        // 越界槽：Claim 的 inflight 占额必须照减（占额者必完工——否则 drain 卡死）
        b.inflight.fetch_sub(1, std::memory_order_acq_rel);
        I.Notify();
        return false;
    }
    if (n_dests > BankReq::kMaxDests) {
        // 申报超额：失败完成（判负纪律）。静默截断=把缺失输出当有效结果——
        // 绝不。占额照减（本槽视为完工；reqs 槽位未登记，发车按作废槽跳过）。
        std::fprintf(stderr, "[bank] SubmitWait n_dests=%d > 容量 %d ——本前向失败"
                     "（适配器申报违约；占额已代减，勿再 Abandon 本槽）\n",
                     n_dests, BankReq::kMaxDests);
        std::fflush(stderr);
        b.inflight.fetch_sub(1, std::memory_order_acq_rel);
        I.Notify();
        return false;
    }
    const long long ts0 = I.cen && I.cen->on ? NowNsI() : 0;
    BankReq* r = &b.req_pool[(size_t)slot];   // 槽独占对象池（init 期一次分配）
    *r = BankReq{};
    r->bank = bank;
    r->slot = slot;
    for (int i = 0; i < n_dests; i++) r->dests[i] = dests[i];
    r->n_dests = n_dests;
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
                            // 而本 fiber 不再挂起=工人重复切入已收卷 fiber
                            // =UAF；投递只留给真正挂起的写手）
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
    // ⚠ 仅限写手线程可发车的后端（TRT/CPU）：ORT 图会话绑调度台线程，写手
    // 线程回放=ORT 重新捕获（CUDA 900/901）→ 此类后端只 Notify，调度台
    // "满座即发"兜底（唤醒延迟 µs 级）。
    if (b.be->DispatchFromWriterOk()
        && prev_inf == 1 && b.cursor.load(std::memory_order_acquire) >= b.slots) {
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
    if (slot < 0 || slot >= b.slots) return;
    b.reqs[(size_t)slot] = nullptr;                        // 作废槽：发车跳过
    b.inflight.fetch_sub(1, std::memory_order_acq_rel);    // 完工照减（drain 不堵）
    // 不 Notify：inflight-- 只被 close-drain 的自旋等待（不依赖 cv）；全弃批的
    // 关舱由窗闹钟兜底。缓存命中路径高频走此（判决13）——每次 notify_all 会把
    // 调度台打成唤醒风暴（实测 53% 命中反慢 2.4× 的主因）。
}

// 收割：完成旗标到（=输出已驻留主机）→ 逐 req 拷输出+回投 → 还池。
// **拷贝承重**：还池先于游戏恢复（新批可能立即复用槽行/输出 arena），
// 适配器不得直读银行内存——dests 缓冲收割侧回填。
// ldone 生命周期=SubmitWait 栈对象：本函数先回填后投递，协议保证 SubmitWait
// 未返回前回填完成（加超时/取消路径须先改本协议——bank.h Shutdown 前置条件）。
// req 对象=银行槽池所有（req_pool），此处只摘链不回收。
static void BankHarvest(BankScheduler::Impl& I, BankCtl& b) {
    // P0-1（2026-09-24 审计）：名字→(宽,基址) 每批 hoist——OutputWidth/
    // OutputRow 的名字查表×行数 收敛为 ×唯一名（OutputRow 线性契约：
    // 基址+slot×宽，三后端同式）。>16 唯一名退化为直查（正确性不变）
    struct Hoist { const char* name; int w; const float* base; };
    Hoist hoist[16];
    int n_hoist = 0;
    auto hoisted = [&](const OutputDest& od, int& w, const float*& base) -> bool {
        for (int i = 0; i < n_hoist; i++)
            if (hoist[i].name == od.name
                || std::strcmp(hoist[i].name, od.name) == 0) {
                w = hoist[i].w;
                base = hoist[i].base;
                return true;
            }
        if (n_hoist >= 16) return false;
        w = b.be->OutputWidth(b.sess, od.name);
        if (w <= 0) return false;
        base = b.be->OutputRow(b.sess, od.name, 0);
        if (!base) return false;
        hoist[n_hoist++] = {od.name, w, base};
        return true;
    };
    for (int s = 0; s < b.flight_n; s++) {
        BankReq* r = b.reqs[(size_t)s];
        b.reqs[(size_t)s] = nullptr;
        if (!r) continue;   // 作废槽
        for (int d = 0; d < r->n_dests; d++) {
            const OutputDest& od = r->dests[d];
            if (!od.name || !od.dst) continue;
            int w = 0;
            const float* src = nullptr;
            if (!hoisted(od, w, src)) continue;
            int cn = w < od.n ? w : od.n;   // od.n=调用方申报的 dst 容量（负值/空
                                            // 指针此行跳过——min(模型行宽, 容量)）
            if (src && cn > 0) memcpy(od.dst, src + (size_t)s * (size_t)w,
                                      sizeof(float) * (size_t)cn);
        }
        if (r->fiber) {
            r->ldone->fail = false;   // fiber 腿：SwitchToFiber 全序免锁
            FiberPost(r->fiber);
        } else {
            {
                std::lock_guard<std::mutex> lk2(r->ldone->mx);
                r->ldone->fail = false;
                r->ldone->done = true;
            }
            r->ldone->cv.notify_one();
        }
        if (I.cen) I.cen->OnPipeDone(1);   // W 拆账：回信出账
        I.lat.push_back(NowMsD() - r->t0);
    }
    I.gpu_busy_sum += NowMsD() - b.flight_t0;
    // 批大小/在飞时长样本：记在收割侧（BankHarvest 恒在调度台线程=与 lat 同源
    // 独占；勿移到发车侧——BankDrainSubmit 有写手自驱路径，plain vector 会竞争）
    const double flw = NowMsD() - b.flight_t0;
    I.bsz.push_back(b.flight_n);
    I.fl_ms.push_back(flw);
    // 还池（线性生命周期；mx 护——满座自驱路径也可能回池；按组还）
    b.state.store(BK_POOL, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(I.mx);
        I.pool_g[b.grp].push_back(b.id);
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
    int fi = I.fill_idx[b.grp].load(std::memory_order_acquire);
    if (fi == b.id) I.fill_idx[b.grp].compare_exchange_strong(fi, -1);
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
        if ((spins & kDrainDiagMask) == kDrainDiagMask) {   // 低频诊断（~每 5ms 一次）
            double waited = NowMsD() - drain_t0;
            if (waited > 2.0) {
                std::printf("[bank] drain 长等 %.1fms: bank=%d inflight=%d cursor=%d "
                            "pool=%zu fill=%d（若持续不归零=在途计数有漏减路径）\n",
                            waited, b.id, b.inflight.load(), b.cursor.load(),
                            I.pool_g[b.grp].size(), I.fill_idx[b.grp].load());
                std::fflush(stdout);
            }
        }
        SpinPause();
    }
    I.drain_us.fetch_add((long long)((NowMsD() - drain_t0) * 1000.0));
    int n = b.cursor.load(std::memory_order_acquire);
    if (n > b.slots) n = b.slots;
    if (n <= 0) {   // 空舱（不可达防御）：直接回池
        b.state.store(BK_POOL, std::memory_order_release);
        std::lock_guard<std::mutex> lk(I.mx);
        I.pool_g[b.grp].push_back(b.id);
        return;
    }
    // 发车：后端前缀 h2d（近满批整块）+ 异步发射（图回放优先）——不等回程
    double th0 = NowMsD();
    unsigned seq = 0;
    if (!b.be->SubmitBatch(b.sess, n, seq)) {
        // 发射失败（不可达防御）：本批全弃答（判负纪律），银行回池
        std::printf("[bank] 批异常（发射，本批 %d 行弃答）\n", n);
        std::fflush(stdout);
        for (int s = 0; s < n; s++) {
            BankReq* r = b.reqs[(size_t)s];
            b.reqs[(size_t)s] = nullptr;
            if (!r) continue;
            BankFailComplete(r);
            if (I.cen) I.cen->OnPipeDone(1);
        }
        b.state.store(BK_POOL, std::memory_order_release);
        std::lock_guard<std::mutex> lk(I.mx);
        I.pool_g[b.grp].push_back(b.id);
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

// 出池轮转（调度台线程，按组）：组池顶出一家（游标归零=新集会）→ 投回该组
// 挂起写手（组内等待——他组轮转不惊醒，免唤醒风暴）
static void BankTryRotate(BankScheduler::Impl& I, int g) {
    if (g < 0 || g >= I.n_groups) return;
    if (I.fill_idx[g].load(std::memory_order_acquire) >= 0) return;
    int i = -1;
    std::deque<void*> wake;
    {
        std::lock_guard<std::mutex> lk(I.mx);
        if (!I.pool_g[g].empty()) {
            i = I.pool_g[g].front();
            I.pool_g[g].pop_front();
            wake.swap(I.waiters_g[g]);
            I.cv.notify_all();
            if (I.wake_ev) SetEvent((HANDLE)I.wake_ev);   // HR 路径镜像（持锁点
                                                          // 裸加——Notify 有重锁）
        }
    }
    if (i < 0) return;
    BankCtl& b = I.banks[(size_t)i];
    b.cursor.store(0, std::memory_order_release);   // 新集会游标（图/地址一夫一妻）
    b.state.store(BK_FILL, std::memory_order_release);
    I.fill_idx[g].store(i, std::memory_order_release);
    for (void* fib : wake) FiberPost(fib);
}

// ---------------- 调度台 ----------------
// 事件驱动+短轮询（在途时 200µs 兜底叫醒——完成旗标无中断，只轻轮询）。
static void BankLoop(BankScheduler::Impl& I) {
#ifdef _WIN32
    if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST))
        std::fprintf(stderr, "[bank] 调度台 SetThreadPriority 失败 GLE=%lu（继续，"
                     "仅性能层面影响）\n", GetLastError());
    if (I.cen) I.cen->tid_disp = GetCurrentThreadId();
#else
    // POSIX：调度台优先级提升收口为 no-op（性能面，非正确性；tid 表=Windows
    // 普查通道，非 Windows 记 0）
    if (I.cen) I.cen->tid_disp = 0;
#endif
    double window_ms = I.cfg.window_ms;
    if (window_ms < I.cfg.window_floor) window_ms = I.cfg.window_floor;
    double window_t0[BankScheduler::Impl::kMaxGrp] = {0};
    bool window_open[BankScheduler::Impl::kMaxGrp] = {false};
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
            if (!b.be->CompletionReached(b.sess, b.flight_seq)) {
                if (now - b.flight_t0 > 500.0 && !b.flight_warned) {
                    b.flight_warned = true;
                    std::printf("[bank] FLIGHT 看门狗: bank=%d 已 %.0fms 未回信 seq=%u"
                                "（后端段卡死排查线索）\n",
                                i, now - b.flight_t0, b.flight_seq);
                    std::fflush(stdout);
                }
                continue;
            }
            b.be->CompletionFence();
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
                for (int g = 0; g < I.n_groups; g++) {
                    std::deque<void*> wg;
                    wg.swap(I.waiters_g[g]);
                    wake.insert(wake.end(), wg.begin(), wg.end());
                }
                I.cv.notify_all();
                if (I.wake_ev) SetEvent((HANDLE)I.wake_ev);   // stop 即时叫醒 HR 等待
            }
            for (void* fib : wake) FiberPost(fib);
            break;
        }
        // ---- 出池：各组独立轮转（无填充银行 → 组池顶出一家 → 投回该组写手）----
        if (cen && cen->on) seg_t0 = NowNsI();
        for (int g = 0; g < I.n_groups; g++) BankTryRotate(I, g);
        if (cen && cen->on) cen->seg_rot_ns.fetch_add(NowNsI() - seg_t0, std::memory_order_relaxed);
        // ---- timer 发车（满座通常已被写手自驱；此处兜底：到期或观察到满座）----
        for (int g = 0; g < I.n_groups; g++) {
            int fi = I.fill_idx[g].load(std::memory_order_acquire);
            if (fi < 0) { window_open[g] = false; continue; }
            BankCtl& b = I.banks[(size_t)fi];
            int taken = b.cursor.load(std::memory_order_acquire);
            if (taken <= 0 || b.state.load(std::memory_order_acquire) != BK_FILL) {
                window_open[g] = false;   // 空舱/已被自驱发走：窗口重置
            } else {
                if (!window_open[g]) {
                    window_open[g] = true;
                    window_t0[g] = now;
                }
                bool full = taken >= b.slots;   // 界=本银行形状（异构批形状）
                bool expired = (now - window_t0[g]) >= window_ms;
                if (full || expired) {
                    // 关舱（CAS 输=写手已自驱）→ 先轮转开新窗（drain/提交不堵
                    // 下一窗，批间流水重叠）→ 再 drain+发车本舱
                    window_open[g] = false;
                    const long long tc0 = cen && cen->on ? NowNsI() : 0;
                    int expected = BK_FILL;
                    if (b.state.compare_exchange_strong(expected, BK_CLOSED,
                                                        std::memory_order_acq_rel,
                                                        std::memory_order_acquire)) {
                        int fi0 = I.fill_idx[g].load(std::memory_order_acquire);
                        if (fi0 == b.id) I.fill_idx[g].compare_exchange_strong(fi0, -1);
                        BankTryRotate(I, g);
                        if (cen && cen->on)
                            cen->seg_close_ns.fetch_add(NowNsI() - tc0, std::memory_order_relaxed);
                        BankDrainSubmit(I, b, true);
                    } else if (cen && cen->on) {
                        cen->seg_close_ns.fetch_add(NowNsI() - tc0, std::memory_order_relaxed);
                    }
                }
            }
        }
        // ---- 汇报（每 300 个回信一行）----
        if ((int)I.lat.size() >= 3000) {   // P0-4（2026-09-24 审计）：3×sort+
                                           // printf+fflush 在调度台线程=行速
                                           // 高时每几 ms 抖一次——×10 节流
                                           //（细粒度看 census）
            std::sort(I.lat.begin(), I.lat.end());
            std::sort(I.bsz.begin(), I.bsz.end());
            std::sort(I.fl_ms.begin(), I.fl_ms.end());
            auto pct = [](const std::vector<int>& v, double p) -> double {
                return v.empty() ? 0.0 : (double)v[(size_t)(p * (double)(v.size() - 1))];
            };
            auto pctd = [](const std::vector<double>& v, double p) -> double {
                return v.empty() ? 0.0 : v[(size_t)(p * (double)(v.size() - 1))];
            };
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
            std::printf("[bank] 批分布: bsz p10/p50/p90=%.0f/%.0f/%.0f（slots=%d）"
                        " flw p50/p90=%.2f/%.2fms n=%zu\n",
                        pct(I.bsz, 0.10), pct(I.bsz, 0.50), pct(I.bsz, 0.90),
                        I.cfg.slots,
                        pctd(I.fl_ms, 0.50), pctd(I.fl_ms, 0.90),
                        I.bsz.size());
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
            I.bsz.clear();
            I.fl_ms.clear();
            I.stat_t0 = NowMsD();
            I.dep_us.store(0);
            I.drain_us.store(0);
            I.batches.store(0);
            I.rows.store(0);
            I.self_dep.store(0);
            I.gpu_busy_sum = 0;
        }
        // ---- 等待：事件（commit/领号/还池）cv 叫醒；窗内等窗到期（取各组最近
        //      到期）；在途兜底轮询 ----
        {
            bool any_flight = false;
            for (int i = 0; i < I.n_banks; i++)
                if (I.banks[(size_t)i].state.load(std::memory_order_acquire) == BK_FLIGHT)
                    { any_flight = true; break; }
            // ---- 等待策略三模式（P1 决策延迟链 v3，2026-09-24 接入方诊断）----
            // cv.wait_for(0.1ms) 在 Windows 实为 1-1.3ms 定时器量子（判决3），
            // 每决策吃两次（完成检出+下批发车处理）= srv-lat 决策延迟链主项。
            //   0=关：原路（忙 0.1 / 闲 2ms 的 cv——量子税在）
            //   1=纯自旋：busy 期不进 cv 纯轮询——延迟最优（实测 11.5×），
            //     忙时独烧调度台核（opt-in 代价）
            //   2=通知驱动（省核+同快）：WMO 等 [提交通知信号量 + 各在飞银行
            //     完成信号量]——提交/完成谁来叫醒谁，零轮询零量子零烧核；
            //     检出=回调延迟+唤醒（µs 级）。有轮询型在飞行（非 fence 后端）
            //     → cv 0.1 回退（量子税同原路）
            // 全闲一律回 cv 长等（不烧空核）。stop 有信号量唤醒+循环顶判断。
            // **混合等待（EMA 预测 deadline+余量）判死拆除（判决17 全账）**：
            // ①cv 睡眠段量子封底≈1ms ≥ fb 形状在飞时长；②EMA 把检测延迟吸进
            // 预测=自毒正反馈（实测 5.5ms 劣于原路）；③余量段 SwitchToThread
            // 把量子让给被服务的工人=饿死。勿再提案此族。
            bool has_fill = false;
            for (int g = 0; g < I.n_groups; g++)
                if (I.fill_idx[g].load(std::memory_order_acquire) >= 0)
                    { has_fill = true; break; }
            const bool busy = any_flight || has_fill;
            auto cv_long_wait = [&](double wait_ms) {
                const long long tw0 = cen && cen->on ? NowNsI() : 0;
                std::unique_lock<std::mutex> lk(I.mx);
                I.cv.wait_for(lk, std::chrono::duration<double>(wait_ms / 1000.0));
                if (cen && cen->on) {
                    cen->seg_wait_ns.fetch_add(NowNsI() - tw0, std::memory_order_relaxed);
                    cen->seg_iter_ns.fetch_add(NowNsI() - it0, std::memory_order_relaxed);
                    cen->seg_iter_n.fetch_add(1, std::memory_order_relaxed);
                }
            };
            if (!I.spin || !busy) {
                double wait_ms = any_flight ? 0.1 : 2.0;
                for (int g = 0; g < I.n_groups; g++) {
                    if (!window_open[g]) continue;
                    double rem = (window_t0[g] + window_ms) - NowMsD();
                    if (rem < 0.02) rem = 0.02;
                    if (rem < wait_ms) wait_ms = rem;
                }
                if (I.hr_mode) {
                    // HR 路径：WMO 等 [HR 定时器(相对 due=wait_ms) + 唤醒事件]。
                    // 醒因不重要——循环体自带全量状态复查，假唤醒无害；
                    // notify 落在 ResetEvent 前=丢事件但 timer 必醒（最坏等满
                    // wait_ms = 旧 cv 量子行为），语义安全无死等。
                    const long long tw0 = cen && cen->on ? NowNsI() : 0;
                    ResetEvent((HANDLE)I.wake_ev);
                    LARGE_INTEGER due;
                    due.QuadPart = -(LONGLONG)(wait_ms * 10000.0);   // ms→100ns 相对
                    if (SetWaitableTimer((HANDLE)I.hr_timer, &due, 0,
                                         nullptr, nullptr, FALSE)) {
                        HANDLE hs[2] = {(HANDLE)I.hr_timer, (HANDLE)I.wake_ev};
                        WaitForMultipleObjects(2, hs, FALSE,
                                               (DWORD)(wait_ms + 50.0));
                    } else {
                        Sleep((DWORD)(wait_ms + 0.5));   // Set 失败兜底
                    }
                    if (cen && cen->on) {
                        cen->seg_wait_ns.fetch_add(NowNsI() - tw0,
                                                   std::memory_order_relaxed);
                        cen->seg_iter_ns.fetch_add(NowNsI() - it0,
                                                   std::memory_order_relaxed);
                        cen->seg_iter_n.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    cv_long_wait(wait_ms);
                }
            } else if (I.spin == 1) {
                SpinPause();   // 纯轮询：量子彻底不沾，忙时独烧调度台核
            } else {
#ifdef _WIN32
                // 通知驱动（spin=2）：WMO 等 [通知信号量 + 各在飞银行完成
                // 信号量]——谁好了叫醒谁；银行 ≤32（kMax）+通知 1 ≤ 64 句柄限
                HANDLE handles[2 + 32];
                DWORD nh = 0;
                bool all_waitable = true;
                for (int i = 0; i < I.n_banks; i++) {
                    BankCtl& b = I.banks[(size_t)i];
                    if (b.state.load(std::memory_order_acquire) != BK_FLIGHT)
                        continue;
                    void* h = b.be->CompletionWaitHandle(b.sess);
                    if (!h) { all_waitable = false; break; }   // 轮询型在飞行→cv 回退
                    handles[(size_t)nh++] = (HANDLE)h;
                }
                if (I.notify_sem) handles[(size_t)nh++] = (HANDLE)I.notify_sem;
                if (all_waitable && nh >= 2) {
                    double timeout_ms = 2000.0;   // 兜底（stop 有信号量、窗有超时）
                    for (int g = 0; g < I.n_groups; g++) {
                        if (!window_open[g]) continue;
                        double remw = (window_t0[g] + window_ms) - NowMsD();
                        if (remw < 0.02) remw = 0.02;
                        timeout_ms = (std::min)(timeout_ms, remw);
                    }
                    const long long tw0 = cen && cen->on ? NowNsI() : 0;
                    WaitForMultipleObjects(nh, handles, FALSE, (DWORD)timeout_ms);
                    if (cen && cen->on) {
                        cen->seg_wait_ns.fetch_add(NowNsI() - tw0, std::memory_order_relaxed);
                        cen->seg_iter_ns.fetch_add(NowNsI() - it0, std::memory_order_relaxed);
                        cen->seg_iter_n.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    cv_long_wait(0.1);   // 混合农场有轮询型在飞行 → 原量子路
                }
#else
                SpinPause();   // 非 Windows：纯轮询回退（fence 通道暂未上 POSIX）
#endif
            }
        }
    }
}

// ---------------- init/shutdown ----------------
bool BankScheduler::Init(const BankConfig& cfg, const ModelConfig& mcfg, ModelSpec* spec_out) {
    BankGroupCfg g;
    g.be = primary_be_;
    g.model = mcfg;
    g.banks = cfg.banks;
    if (spec_out) g.spec = *spec_out;   // 便捷路径同组规格（组 slots=0=统一形状；
                                        // 缺此行会被 InitGroups 的 spec.slots
                                        // ≠cfg.slots 校验拒绝——旧坑）
    return InitGroups(cfg, std::vector<BankGroupCfg>{g}, spec_out);
}

bool BankScheduler::InitGroups(const BankConfig& cfg,
                               const std::vector<BankGroupCfg>& groups,
                               ModelSpec* spec_out) {
    Shutdown();
    cfg_ = cfg;
    if (cfg.slots < 1) return false;
    if (!spec_out) {
        std::fprintf(stderr, "[bank] Init 需要 spec_out（Farm 预先 LoadSpec 的模型规格）\n");
        return false;
    }
    if (groups.empty() || groups.size() > (size_t)Impl::kMaxGrp) {
        std::fprintf(stderr, "[bank] 设备组数 %zu ∉ [1,%d]\n",
                     groups.size(), Impl::kMaxGrp);
        return false;
    }
    int total = 0;
    for (auto& g : groups) {
        if (g.banks < 1 || g.banks > Impl::kMax || !g.be) {
            std::fprintf(stderr, "[bank] 组配置非法（banks=%d be=%p；界 [1,%d]）\n",
                         g.banks, (void*)g.be, Impl::kMax);
            return false;
        }
        if (g.slots < 0 || g.slots > 1024) {
            std::fprintf(stderr, "[bank] 组 slots=%d ∉ [0,1024]（0=统一形状）\n", g.slots);
            return false;
        }
        if (g.slots == 0 && g.spec.slots != cfg.slots) {
            std::fprintf(stderr, "[bank] 组 spec.slots=%d ≠ cfg.slots=%d\n",
                         g.spec.slots, cfg.slots);
            return false;
        }
        total += g.banks;
        if (total > Impl::kMax) {
            std::fprintf(stderr, "[bank] 总银行数 %d > %d（跨组合计上限）\n",
                         total, Impl::kMax);
            return false;
        }
    }
    impl_ = new Impl();
    Impl& I = *impl_;
    I.cen = cen_;
    I.cfg = cfg;
    I.spin = cfg.spin;
    I.sched_aff = ParseCpuList(std::getenv("FARM_SCHED_AFFINITY"));
    if (!I.sched_aff.empty()) {
        std::printf("[bank] 调度台绑核 %d 项\n", (int)I.sched_aff.size());
        std::fflush(stdout);
    }
#ifdef _WIN32
    I.notify_sem = CreateSemaphoreA(nullptr, 0, 0x7FFFFFFF, nullptr);
    // HR waitable timer 等待面（FARM_BANK_HRTIMER=1 opt-in；默认关=零行为差
    // ——量子税在 cv 里，判决3 的缺省档改造，2026-09-24）
    if (const char* e = std::getenv("FARM_BANK_HRTIMER"); e && *e && atoi(e) == 1) {
        I.wake_ev = CreateEventA(nullptr, TRUE, FALSE, nullptr);   // 手动重置
        // 旗标用数值：CREATE_MANUAL_RESET=0x1 / CREATE_WAITABLE_TIMER_HIGH_
        // RESOLUTION=0x2（Win10 1803+；SDK 常量被 WINVER 守卫，项目未提升）
        I.hr_timer = CreateWaitableTimerExW(nullptr, nullptr, 0x1 | 0x2,
                                            TIMER_ALL_ACCESS);
        I.hr_mode = I.wake_ev && I.hr_timer;
        if (!I.hr_mode)
            std::fprintf(stderr, "[bank] FARM_BANK_HRTIMER=1 但内核对象创建失败"
                         "（GLE=%lu，Win10 1803+ 才有 HR 旗标）——走原 cv 路\n",
                         GetLastError());
        else
            std::fprintf(stderr, "[bank] HR waitable timer 等待面启用（缺省档"
                         "量子税改造，实测过冲口径 50-600µs）\n");
    }
#endif
    I.spec = *spec_out;   // 由 Farm 预先 LoadSpec 并核各组结构一致
    *spec_out = I.spec;
    for (int g = 0; g < Impl::kMaxGrp; g++) I.fill_idx[g].store(-1);
    I.n_groups = (int)groups.size();
    // 调度台线程上建会话（ORT 图会话 PerThreadContext 铁律：创建/热身/回放
    // 须同线程；TRT 同规更稳；跨组同线程无碍——各组会话独立）
    I.disp = std::thread([&I, groups, spec_out] {
        if (!I.sched_aff.empty()) PinThread(I.sched_aff, 0, "sched");
        double tb0 = NowMsD();
        bool ok = true;
        int id = 0;
        ModelSpec g0_harvest;         // 探测砍除：组 0 spec 由首家会话产出
        for (int g = 0; g < I.n_groups && ok; g++) {
            const BankGroupCfg& gc = groups[(size_t)g];
            const int gslots = gc.slots > 0 ? gc.slots : gc.spec.slots;
            if (gc.spec.ins.empty() && g != 0) {   // 防御：延迟 spec 仅组 0
                std::fprintf(stderr, "[bank] 组 %d spec 占位非法（延迟 spec 仅组 0）\n", g);
                ok = false;
                break;
            }
            ModelSpec gspec = gc.spec;
            gspec.slots = gslots;   // 形状差只许 dim0：行宽/名字已由 Farm 核对
            for (int k = 0; k < gc.banks && ok; k++, id++) {
                BankCtl& b = I.banks[(size_t)id];
                b.id = id;
                b.grp = g;
                b.be = gc.be;
                b.slots = gslots;
                if (g == 0 && gc.spec.ins.empty()) {
                    // 探测砍除通道（能力位 ProbeFreeSpec）：首个真实银行会话
                    // 顺带产出组 0 spec——省一次建探测会话即毁（YGO 清单模式
                    // ≈0.1s×56 腿/代）。spec_out 只挂 k==0 首家：spec_out 的
                    // IO 枚举是追加式，banks>1 时第二家再传=同组 IO 重复入表
                    // （CNN 6 组形状首爆：组 0 spec 4 ins 2 outs，2026-09-24）
                    b.sess = b.be->CreateSessionWithSpec(gc.model, gslots,
                                                         /*for_bank=*/true,
                                                         k == 0 ? &g0_harvest
                                                                : nullptr);
                    if (!b.sess) { ok = false; break; }
                    if (g0_harvest.slots != gslots) {
                        std::fprintf(stderr, "[bank] 组 0 模型批形状 dim0=%d ≠ %d"
                                     "（探测砍除通道后校验）\n",
                                     (int)g0_harvest.slots, gslots);
                        ok = false;
                        break;
                    }
                    gspec = g0_harvest;
                    gspec.slots = gslots;
                    I.spec = g0_harvest;      // Claim 清零面立即可用
                    *spec_out = g0_harvest;   // Farm 侧 spec_ 回填
                } else {
                    b.sess = b.be->CreateSession(gc.model, gspec, /*for_bank=*/true);
                    if (!b.sess) { ok = false; break; }
                }
                if (!b.be->Warmup(b.sess)) { ok = false; break; }
                b.cursor.store(0);
                b.inflight.store(0);
                b.state.store(BK_POOL);
                b.reqs.assign((size_t)gslots, nullptr);
                b.req_pool.resize((size_t)gslots);   // 槽独占 req 对象池（一次分配）
            }
        }
        int built = id;
        // 槽基址预解（P0-2，2026-09-24 审计）：全部会话建好后一次性解析
        //（此时 I.spec 已终态——组 0 延迟收割在会话创建期完成）。**in_idx
        // 必须在此一次填满、运行期只读**：InputRow 会被多工人线程并发调用，
        // 运行期 push_back=堆损坏（首版间歇段错误案，2026-09-24）
        for (int i = 0; i < built; i++) {
            BankCtl& b = I.banks[(size_t)i];
            b.in_rows.assign(I.spec.ins.size(), nullptr);
            b.in_idx.reserve(I.spec.ins.size());
            for (size_t ii = 0; ii < I.spec.ins.size(); ii++) {
                if (I.spec.ins[ii].population) continue;
                size_t rb = 0;
                b.in_rows[ii] = (char*)b.be->InputRow(b.sess, I.spec.ins[ii].name.c_str(), 0, &rb);
                b.in_idx.push_back({I.spec.ins[ii].name.c_str(), ii});
            }
        }
        // 图地址烧死小实验（各组首家）：任一不过=拒绝银行制启动（回不去旧路径
        // 的字节安全性不赌；DML 路线同一实验兜底"同步 Run"假设）
        for (int i = 0; ok && i < built; i++) {
            if (I.banks[(size_t)i].id != 0
                && I.banks[(size_t)i].grp == I.banks[(size_t)(i - 1)].grp)
                continue;   // 非本组首家：跳过
            if (!I.banks[(size_t)i].be->ProbeGraph(I.banks[(size_t)i].sess)) {
                std::fprintf(stderr, "[bank] 银行 %d（组 %d）图地址小实验未过"
                             "——拒绝银行制启动\n", i, I.banks[(size_t)i].grp);
                ok = false;
            }
        }
        if (!ok) {
            for (int i = 0; i < built; i++)
                if (I.banks[(size_t)i].sess) {
                    I.banks[(size_t)i].be->DestroySession(I.banks[(size_t)i].sess);
                    I.banks[(size_t)i].sess = nullptr;   // 防 Shutdown 二次销毁（双重释放案）
                }
            std::lock_guard<std::mutex> lk(I.init_mx);
            I.init_rc = -1;
            I.init_cv.notify_all();
            return;
        }
        id = 0;
        for (int g = 0; g < I.n_groups; g++) {
            for (int k = 0; k < groups[(size_t)g].banks; k++, id++)
                I.pool_g[g].push_back(id);
        }
        I.n_banks = built;
        std::fprintf(stderr, "[bank] 建池 %d 家（%d 设备组）耗时 %.0fms"
                     "（pinned 直写槽+专属批图；组池独立轮转）\n",
                     built, I.n_groups, NowMsD() - tb0);
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
    n_groups_ = I.n_groups;
    std::printf("[bank] 零拷贝槽位银行就绪: 池 %d 家（%d 组）× %d 槽，window=%.2fms"
                "（游标领号/直写槽/满座或闹钟发车/旗标收割/还池；前缀 h2d+每家"
                "一图；组池独立=链→组钉扎；池容量=在飞上限=背压）\n",
                banks_, n_groups_, cfg.slots, cfg.window_ms);
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
    if (I.notify_sem) ReleaseSemaphore(I.notify_sem, 1, nullptr);   // 唤醒 WMO
    if (I.disp.joinable()) I.disp.join();
    for (auto& b : I.banks)
        if (b.sess) {
            b.be->DestroySession(b.sess);
            b.sess = nullptr;
        }
    if (I.notify_sem) { CloseHandle(I.notify_sem); I.notify_sem = nullptr; }
    if (I.hr_timer) { CloseHandle(I.hr_timer); I.hr_timer = nullptr; }
    if (I.wake_ev) { CloseHandle(I.wake_ev); I.wake_ev = nullptr; }
    delete impl_;
    impl_ = nullptr;
    banks_ = 0;
    n_groups_ = 0;
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
    OutputDest dests[BankScheduler::kMaxOutputDests];
    int nd = g->CollectOutputs(dests, BankScheduler::kMaxOutputDests);
    if (nd > BankScheduler::kMaxOutputDests) {
        // 适配器违约（契约=返回条数 ≤cap）：截断而非越界读，判负交上层
        std::fprintf(stderr, "[inline] CollectOutputs 返回 %d > cap %d——截断"
                     "（适配器违约）\n", nd, BankScheduler::kMaxOutputDests);
        nd = BankScheduler::kMaxOutputDests;
    }
    g->AssembleInto(s_->writer);
    unsigned seq = 0;
    if (!s_->be->SubmitBatch(s_->sess, s_->spec->slots, seq)) return false;
    long long spins = 0;
    while (!s_->be->CompletionReached(s_->sess, seq)) {
        // 短自旋+溢出让出：完成旗标轮询语义不变（结果与让出无关），inline
        // 模式不再烧满一个核
        if (++spins <= kInlineSpinBeforeYield) SpinPause();
        else std::this_thread::yield();
    }
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
