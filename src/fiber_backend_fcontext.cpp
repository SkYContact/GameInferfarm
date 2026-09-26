// fiber_backend_fcontext.cpp — 自写 fcontext 后端（MASM64 切换体，见
// src/fcontext.asm）。POSIX 路线的地基（SysV ABI 保存面更小，同构可移植）。
//
// 栈：VirtualAlloc reserve+全量 commit（FARM_FC_STACK_KB，缺省 1024KB，
// 钳 [64,8192]）——无 guard-page 生长（WinFiber 按需提交的足迹差见设计
// 评审第四节）；base/size 与 ctx 合并堆分配（Owned），供 Destroy 回收。
//
// ctx 不驻栈（血律：栈向下生长会踩穿驻栈 ctx——首实现 AV rip=0 定谳，
// farm_test/探针复现；Boost.Context 同款把 ctx 放堆/栈外）。
//
// 切换面：fi_swap 保存 ABI 最小集；"当前 ctx"=thread_local 指针，每次
// Switch 先更新再换——同工人多局复用一个线程，当前指针由切换本身维护
// （与 TLS 帧纪律同域：帧装卸仍在工人 Switch 调用点，fiber_pool.cpp 原样）。
#if defined(_WIN32) && defined(_MSC_VER)
#include "fiber_backend.h"
#include "fcontext.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <windows.h>

namespace inferfarm {
namespace {

// ctx 与栈头（base/size，供 Destroy 回收）合并一次堆分配
struct Owned {
    void* base;
    size_t size;
    fc::Ctx ctx;
};

size_t StackBytes() {
    size_t kb = 1024;
    if (const char* e = std::getenv("FARM_FC_STACK_KB")) {
        long v = atol(e);
        if (v >= 64 && v <= 8192) kb = (size_t)v;
    }
    return kb * 1024;
}

class FcontextBackend final : public IFiberBackend {
public:
    const char* name() const override { return "fcontext"; }
    void* ConvertThread() override {
        t_cur = new fc::Ctx{};   // 工人恢复点：首次被切回时由 fi_swap 填真值
        return t_cur;
    }
    void ConvertBack(void* worker_ctx) override {
        delete (fc::Ctx*)worker_ctx;
        t_cur = nullptr;
    }
    void* Create(void (*entry)(void*), void* arg) override {
        const size_t sz = StackBytes();
        void* base = VirtualAlloc(nullptr, sz, MEM_RESERVE | MEM_COMMIT,
                                  PAGE_READWRITE);
        if (!base) return nullptr;
        Owned* o = new Owned{};
        o->base = base;
        o->size = sz;
        // top ≡ 8 (mod 16)：trampoline 内 call 前 rsp≡0 (mod 16) 之必需
        // （x64 ABI：函数入口 rsp ≡ 8 (mod 16)）
        char* raw_top = (char*)base + sz;
        void* top = (void*)((((uintptr_t)raw_top - 8) & ~(uintptr_t)15) + 8);
        fc::fi_make(&o->ctx, top, entry, arg);
        return &o->ctx;
    }
    void Switch(void* to) override {
        fc::Ctx* from = t_cur;          // 本侧持久 ctx（线程内自识别）
        t_cur = (fc::Ctx*)to;           // 先更新后换：切回侧见到的是自己的 ctx
        fc::fi_swap(from, (const fc::Ctx*)to);
    }
    void Destroy(void* game_ctx) override {
        if (!game_ctx) return;
        Owned* o = (Owned*)((char*)game_ctx - offsetof(Owned, ctx));
        VirtualFree(o->base, 0, MEM_RELEASE);
        delete o;
    }

private:
    static thread_local fc::Ctx* t_cur;
};

thread_local fc::Ctx* FcontextBackend::t_cur = nullptr;

} // namespace

IFiberBackend* MakeFcontextBackend() { return new FcontextBackend(); }

} // namespace inferfarm

#else
namespace inferfarm {
IFiberBackend* MakeFcontextBackend() { return nullptr; }   // 限 MSVC/x64
} // namespace inferfarm
#endif
