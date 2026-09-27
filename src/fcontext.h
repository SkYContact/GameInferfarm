// fcontext.h — 自写 x64 有栈上下文（Boost.Context 思路；切换体按平台分档：
// Windows=MASM64 src/fcontext.asm，POSIX x86_64 ELF=GNU as src/fcontext_sysv.S）
//
// 设计评审 docs/design-cpp20-coroutines.md：冻结面下"每挂起一次有栈切换"
// 不可消去（无栈不可达定理），故可移植后端=更轻的**有栈**原语，而非协程。
// 本实现与 WinFiber 的差：保存面=ABI 最小集，不动 TEB/FLS/TSD（异常穿越
// 切换点=未定义，两后端同罪——产线无 throw 面）。
#pragma once
#include <cstdint>

namespace inferfarm {
namespace fc {

#if defined(_WIN32)
// ---- MSVC x64 布局（fcontext.asm 硬编码偏移，两处必须同步改）----
//   0x000 rip  0x008 rsp  0x010 rbx 0x018 rbp 0x020 rdi 0x028 rsi
//   0x030 r12 0x038 r13 0x040 r14 0x048 r15
//   0x050..0x0EF xmm6..xmm15（各 16B）
//   0x0F0 fcw(4) 0x0F4 mxcsr(4)                       → 共 248B
// MSVC x64 ABI：xmm6-15 非易变 ⇒ 保存面含 XMM6-15+MXCSR/FCW（比 WinFiber
// FLOAT_SWITCH 的全 FP 窄且按 ABI 正确；易变寄存器由编译器跨 call 自理）。
struct Ctx {
    uint64_t rip;
    uint64_t rsp;
    uint64_t rbx, rbp, rdi, rsi, r12, r13, r14, r15;
    uint64_t xmm[10][2];
    uint32_t fcw;
    uint32_t mxcsr;
};
static_assert(sizeof(Ctx) == 248, "Ctx 布局与 fcontext.asm 硬编码偏移同步");
#else
// ---- SysV AMD64（ELF；Linux x64）布局（fcontext_sysv.S 硬编码偏移，
//      两处必须同步改）----
//   0x000 rip  0x008 rsp  0x010 rbx  0x018 rbp
//   0x020 r12  0x028 r13  0x030 r14  0x038 r15   → 共 64B（alignas 16）
// SysV AMD64 ABI：被调方保存=rbx/rbp/r12-r15/rsp；**全部 XMM 与
// MXCSR/FCW 均 caller-saved** ⇒ 切换面零 FP 保存（GCC/Clang 按 ABI 编译，
// 不跨 call 持 FP 态——与 WinFiber FLOAT_SWITCH 的语义差见下）。
// 【FP 语义差，切后端必读】SysV 面上协程切出再切回后，XMM 寄存器与
// MXCSR 的内容=对端（其他局/工人）遗留值——**不可信、也无需可信**：
// C++ 编译器本就把 XMM 当 caller-saved，任何跨 Switch() 存活的 FP 态
// 必须落内存（这是 ABI 义务，不是切换体的仁慈）。Windows 面则相反：
// xmm6-15 非易变，fi_swap 保存面显式携带。移植 X86 浮点自定义例程
// （MXCSR FTZ/DAZ 之类）时记得两侧语义不同。
struct alignas(16) Ctx {
    uint64_t rip;
    uint64_t rsp;
    uint64_t rbx, rbp, r12, r13, r14, r15;
};
static_assert(sizeof(Ctx) == 64, "Ctx 布局与 fcontext_sysv.S 硬编码偏移同步");
#endif

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
