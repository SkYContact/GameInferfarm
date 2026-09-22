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
void* FiberCurrent();
void FiberSuspend();
void FiberPost(void* cookie);

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
                  void* user, double stagger_ms);

    bool enabled() const { return workers_ > 0; }
    int workers() const { return workers_; }

private:
    int workers_ = 0;
    Census* census_ = nullptr;
};

// 线程模式腿（不用 fiber 时的同构对照路径）：chains 条 OS 线程各串行跑 per 局，
// 帧在线程内 install 一次（链寿命语义一致）。返回墙钟秒。
double RunLegThreads(int chains, int per, FiberGameFn game_fn, FiberFrameFn frame_fn,
                     void* user, double stagger_ms);

} // namespace inferfarm
