// ============================================================
//  inferfarm/fiber_pool.h — 纤程池（资产1：唤醒队列调度器）
//
//  形态（fiber_contract 2026-09-21，YGO 产线 42→317-368 局/s 的承重件）：
//    · K 工人线程（默认=物理核；SMT 是负资产，勿超物理核数）；
//      ConvertThreadToFiberEx 成主 fiber，循环取本工人就绪队列的对局
//      fiber SwitchToFiber 续跑；
//    · 每局一 fiber，创建于工人（链内串行续局=链-工人亲和，链不跨工人迁移
//      ——保住 thread_local 语义的另一半：帧不换工人）；
//    · 对局打到推理等待点 → fiber Switch 回调度器让出（不睡 cv、不抢锁、
//      不进 OS 运行队列）→ 收割侧 FiberPost 投回就绪队列+唤醒（这就是
//      "唤醒队列"，唤醒税由此消除）；
//    · 切换点装卸链寿命 TLS 帧（ITlsFrame）。
//
//  链（Chain）= 串行局的序列：链 c 的局 i 收卷处在同工人点火局 i+1。
//  链-工人亲和：首局 c%K 定终身。
// ============================================================
#pragma once
#include "tls_frame.h"
#include <cstdint>

namespace inferfarm {

class Census;

// 等待侧桥（银行层调用；三个函数都可在任意线程/fiber 上下文调用）：
//   Current(): 当前在对局 fiber 上 → 返回其 cookie；否则 nullptr（线程腿）。
//   Suspend(): 对局 fiber 内让出（Switch 回所属工人调度器）。
//   Post():    收割侧投递——该局 fiber 进其所属工人的就绪队列+唤醒。
//
// ⚠ Post 投递-挂起不变量（并发正确性的承重前提，2026-09-24 审计定案）：
// Post 允许投递**仍在运行**的 fiber（登记→挂起窗口内被投回=合法路径，如
// Claim 等池登记后调度台先轮转）——投递只是把 cookie 挂进就绪队列；该
// fiber 随后**恰一次** Suspend 时被该记录唤醒=恰好一次恢复，不存在双重
// 调度。对应义务：**Post 之后目标 fiber 必须恰好挂起一次**（登记后中途
// return/跑完收卷=工人重复切入已收卷 fiber=UAF）。SubmitWait 的协议防御
// 分支"绝不 FiberPost 自己"正是这条义务的执行点。debug 构建对双重投递
// 加断言（queued 旗标）。
// 挂起原因（链钟埋点：挂起期时长按此入账 census 的 park/infer/other 桶）。
// 新增挂起点必须传对原因——漏传落 other 桶即"未知等待点"警报。
enum FiberWaitReason {
    FWait_Other = 0,    // 未分类（默认；存在量应≈0）
    FWait_BankPark = 1, // Claim 池空背压：等银行出槽
    FWait_Infer = 2,    // SubmitWait：等推理在飞回信
};
void* FiberCurrent();
void FiberSuspend(FiberWaitReason why = FWait_Other);
void FiberPost(void* cookie);
// 批模式投递（收割唤醒收敛，2026-09-30）：Begin 后 FiberPost 只入队不唤醒
//（工人睡眠与否与队列可见性无关——Mesa 语义，队列非空=睡着的工人醒来必见）；
// End 对批内有投递的工人各 notify 一次。满舱批 256 行×16 工人：256 发真
// futex wake（实测 ~5µs/发=1.3ms 纯唤醒税）→ 16 发。Begin/End 必须同线程
// 配对；End 不调=批模式泄漏（用 RAII 守卫）。
void FiberPostBegin();
void FiberPostEnd();

// 契约 1 的机器校验（GameAdapter："advance 与 assemble 无挂起点"）：
// 作用域内任何 FiberSuspend=debug 断言失败（框架在 AdvanceToDecision/
// AssembleInto 调用点包裹）。Release 构建=计数器照走、断言编译出局（两次
// thread_local 自增/决策，ns 级）——"高质量契约文档"升级为"机器校验契约"。
class ScopedNoSuspend {
public:
    ScopedNoSuspend();
    ~ScopedNoSuspend();
    ScopedNoSuspend(const ScopedNoSuspend&) = delete;
    ScopedNoSuspend& operator=(const ScopedNoSuspend&) = delete;
};

// 局主体：运行一局（内部可任意 FiberSuspend）；返回时局已收卷。
// 参数：chain=链号，game=链内局号，user=RunLeg 传入的原样指针。
using FiberGameFn = void (*)(int chain, int game, void* user);

// 帧工厂：返回链 chain 的链寿命帧；可返回 nullptr。**所有权=调用方/适配器**
// （池只装卸不回收——适配器常以成员地址作帧）。
using FiberFrameFn = ITlsFrame* (*)(int chain, void* user);

class FiberPool {
public:
    // workers：工人线程数（调用方负责钳到物理核；0=用 hardware_concurrency/2）
    void Configure(int workers, Census* census);

    // 一腿：chains 条链 × per 局，逐链错峰点火（stagger_ms——到达层羊群判决：
    // 同步起跑=灾难，错峰=相位去同步，10ms 甜点）。阻塞至全部局收卷，
    // 返回墙钟秒。game_fn 在对局 fiber 上被调；首局在点火线程创建 fiber
    // （YGO 原版语义：首局主线程投、后续局由上一局在同工人创建）。
    double RunLeg(int chains, int per, FiberGameFn game_fn, FiberFrameFn frame_fn,
                  void* user, double stagger_ms, int stagger_batch = 1);

    bool enabled() const { return workers_ > 0; }
    int workers() const { return workers_; }

private:
    int workers_ = 0;
    Census* census_ = nullptr;
};

// 线程模式腿（不用 fiber 时的同构对照路径）：chains 条 OS 线程各串行跑 per 局，
// 帧在线程内 install 一次（链寿命语义一致）。返回墙钟秒。
double RunLegThreads(int chains, int per, FiberGameFn game_fn, FiberFrameFn frame_fn,
                     void* user, double stagger_ms, int stagger_batch = 1);

} // namespace inferfarm
