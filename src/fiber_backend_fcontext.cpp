// fiber_backend_fcontext.cpp — 自写 fcontext 后端。切换体按平台分档：
//   Windows MSVC x64  → MASM64 src/fcontext.asm（248B ctx，XMM6-15+FP 控制字）
//   POSIX x86_64 ELF  → GNU as src/fcontext_sysv.S（64B ctx，零 FP 保存面
//                       ——SysV ABI 全 XMM/MXCSR caller-saved，语义差注记
//                       见 fcontext.h SysV 档头注释）
// 池机器（fiber_pool.cpp）零改动，五原语同接口（fiber_backend.h）。
//
// 栈：全量 commit（FARM_FC_STACK_KB，缺省 1024KB，钳 [64,8192]）——
// 无 guard-page 生长（WinFiber 按需提交的足迹差见设计评审第四节）。
// Windows=VirtualAlloc reserve+commit；POSIX 降级点=posix_memalign 一次
// 全量（语义同既定"全量 commit"，无 reserve/commit 两段面——glibc 大块
// 本就走 mmap，无额外动作）；base/size 与 ctx 合并堆分配（Owned），供
// Destroy 回收。
//
// ctx 不驻栈（血律：栈向下生长会踩穿驻栈 ctx——首实现 AV rip=0 定谳，
// farm_test/探针复现；Boost.Context 同款把 ctx 放堆/栈外）。
//
// 切换面：fi_swap 保存 ABI 最小集；"当前 ctx"=thread_local 指针，每次
// Switch 先更新再换——同工人多局复用一个线程，当前指针由切换本身维护
// （与 TLS 帧纪律同域：帧装卸仍在工人 Switch 调用点，fiber_pool.cpp 原样）。
#if (defined(_WIN32) && defined(_MSC_VER)) || \
    (defined(__x86_64__) && defined(__ELF__))
#include "fiber_backend.h"
#include "fcontext.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>

namespace inferfarm {
namespace {

void* AllocStack(size_t sz) {
    return VirtualAlloc(nullptr, sz, MEM_RESERVE | MEM_COMMIT,
                        PAGE_READWRITE);
}
void FreeStack(void* base, size_t /*sz*/) {
    VirtualFree(base, 0, MEM_RELEASE);
}

} // namespace
#else
#include <stdlib.h>

namespace inferfarm {
namespace {

// POSIX 降级点：posix_memalign(64) 全量一次分配（64B=缓存行对齐的好习惯，
// ABI 只需栈顶 16B——Create 里对齐计算两平台同款）。free 归还。
void* AllocStack(size_t sz) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, sz) != 0) return nullptr;
    return p;
}
void FreeStack(void* base, size_t /*sz*/) { free(base); }

} // namespace
#endif

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
        void* base = AllocStack(sz);
        if (!base) return nullptr;
        Owned* o = new Owned{};
        o->base = base;
        o->size = sz;
        // top ≡ 8 (mod 16)：trampoline 内 call 前 rsp≡0 (mod 16) 之必需
        // （两 ABI 同规：函数入口 rsp ≡ 8 (mod 16)）
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
        FreeStack(o->base, o->size);
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
IFiberBackend* MakeFcontextBackend() { return nullptr; }   // 限 MSVC/x64 或
                                    // POSIX x86_64 ELF 两档，其余面不可用
} // namespace inferfarm
#endif
