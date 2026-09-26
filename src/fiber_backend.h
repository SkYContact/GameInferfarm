// fiber_backend.h — 有栈切换原语的可插拔后端（库内私有接口；判决实验
// docs/design-cpp20-coroutines.md 选型 (d)+(c)）
//
// 五原语覆盖池机器（fiber_pool.cpp）的全部切换面：工人入役/退役、局上下文
// 建/切/毁。接口 C++17 可编译，无任何协程对象——无栈路线经定理+拆解双判
// 死，这里只换"切换原语"本体：
//   winfiber = Windows Fibers 1:1（现役路径原样，C++17 回退档缺省=零行为差）
//   fcontext = 自写 MASM64 有栈切换（ABI 最小保存面；POSIX 路线的地基，
//              C++20 档缺省）
#pragma once

namespace inferfarm {

class IFiberBackend {
public:
    virtual ~IFiberBackend() = default;
    virtual const char* name() const = 0;
    // 工人线程入役：当前线程获得可被切入的上下文；nullptr=失败（工人退役路）
    virtual void* ConvertThread() = 0;
    // 工人退役（ConvertThread 成功后恰好一次；此后该上下文不可再用）
    virtual void ConvertBack(void* worker_ctx) = 0;
    // 一局的执行上下文。entry 永不返回（收卷经 Switch 回工人后由池 Destroy
    // ——fiber_pool.cpp FiGameMain 契约同源）；nullptr=失败（该局按已收卷计）
    virtual void* Create(void (*entry)(void*), void* arg) = 0;
    // 切到目标上下文；从目标切回=本调用返回（工人↔局通用）
    virtual void Switch(void* to) = 0;
    // 局上下文回收（挂起态亦可——栈不再被触碰，与 DeleteFiber 语义同）
    virtual void Destroy(void* game_ctx) = 0;
};

// 选后端：FARM_FIBER_BACKEND=winfiber|fcontext 显式选（双向，两构建档通用
// ——2×2 可隔离归因）；缺省=INFERFARM_CORO20 档 fcontext、C++17 回退档
// winfiber（零行为差）。显式选不可用=回退 winfiber+stderr 提示。
IFiberBackend* FiberBackendSelect();

// 各后端工厂（fiber_backend_winfiber.cpp / fiber_backend_fcontext.cpp；
// 不可用的编译面返回 nullptr）
IFiberBackend* MakeWinFiberBackend();
IFiberBackend* MakeFcontextBackend();

} // namespace inferfarm
