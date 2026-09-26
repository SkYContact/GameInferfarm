// fcontext.h — 自写 x64 有栈上下文（Boost.Context 思路；MASM64 切换体）
//
// 设计评审 docs/design-cpp20-coroutines.md：冻结面下"每挂起一次有栈切换"
// 不可消去（无栈不可达定理），故可移植后端=更轻的**有栈**原语，而非协程。
// 本实现与 WinFiber 的差：保存面=ABI 最小集（GP 非易变 8+XMM6-15+MXCSR/FCW），
// 不动 TEB/FLS（异常穿越切换点=未定义，两后端同罪——产线无 throw 面）。
#pragma once
#include <cstdint>

namespace inferfarm {
namespace fc {

// 上下文布局（偏移在 fcontext.asm 硬编码，两处必须同步改）：
//   0x000 rip  0x008 rsp  0x010 rbx 0x018 rbp 0x020 rdi 0x028 rsi
//   0x030 r12 0x038 r13 0x040 r14 0x048 r15
//   0x050..0x0EF xmm6..xmm15（各 16B）
//   0x0F0 fcw(4) 0x0F4 mxcsr(4)                       → 共 248B
struct Ctx {
    uint64_t rip;
    uint64_t rsp;
    uint64_t rbx, rbp, rdi, rsi, r12, r13, r14, r15;
    uint64_t xmm[10][2];
    uint32_t fcw;
    uint32_t mxcsr;
};
static_assert(sizeof(Ctx) == 248, "Ctx 布局与 fcontext.asm 硬编码偏移同步");

// 保存当前上下文进 *from，恢复 to 并跳其 rip；对端切回=本调用返回。
extern "C" void fi_swap(Ctx* from, const Ctx* to);

// 在堆置 ctx 上建"冷"上下文（首次被 fi_swap 切入时从 trampoline 起跑
// entry(arg)）。stack_top 必须满足 top ≡ 8 (mod 16)（见
// fiber_backend_fcontext.cpp）；栈顶只占 [top-0x10, top) 两个参数槽
// （ctx 不驻栈——栈向下生长会踩穿驻栈 ctx，首实现 AV rip=0 定谳）。
extern "C" void fi_make(Ctx* ctx, void* stack_top, void (*entry)(void*),
                        void* arg);

} // namespace fc
} // namespace inferfarm
