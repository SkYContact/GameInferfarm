# fcontext_sysv.S — System V AMD64（ELF，Linux x64）有栈上下文切换
#                    （GNU as；CMake enable_language(ASM) 非 Windows 面）
#
# 布局与 src/fcontext.h 的 fc::Ctx（SysV 档，64B）同步（偏移硬编码，两处
# 必须同步改）。参考 Boost.Context jump_x86_64_sysv_elf_gas.S /
# make_x86_64_sysv_elf_gas.S（Boost Software License 1.0 / Oliver Kowalke），
# 本仓自写收敛为 fi_swap/fi_make/trampoline 直传 entry/arg 形态（无
# transfer_t）——布局更小（零 FP 面），trampoline 语义与 fcontext.asm 同源。
#
# 约定（SysV AMD64 ABI）：
#   · 被调方保存=rbx/rbp/r12-r15/rsp ⇒ 保存面就这 6 GP + 返回点 rip + rsp。
#     全部 XMM 与 MXCSR/FCW 均 caller-saved——**不保存**（ABI 义务：编译器
#     不跨 call 持 FP 态；切回后 XMM/MXCSR=对端遗留，不可信也无需可信，
#     语义差注记见 fcontext.h SysV 档头注释）。
#   · DF=0 进出（ABI 要求；两侧 C++ 面均不动 DF=自洽）。
#   · 不动 TSD/信号掩码（pthread 才动）——异常穿越切换点=未定义（产线
#     契约无 throw 面；见 docs/design-cpp20-coroutines.md 第四节）。
#   · ctx 在堆上（不驻栈）：栈向下生长会踩穿驻栈的 ctx（血律，pitfalls
#     2026-09-27）。栈顶只放 entry/arg 两个参数槽，trampoline 首跳即消费。
#
# CET/IBT：间接跳目标（fi_trampoline）须以 ENDBR64 起头——__CET__ 编译面
# （-fcf-protection）经 cet.h 取 _CET_ENDBR；非 CET 面=空宏（同 Boost 手法）。

#ifdef __CET__
#include <cet.h>
#define FI_CET_ENDBR _CET_ENDBR
#else
#define FI_CET_ENDBR
#endif

.text
.globl fi_swap
.type fi_swap,@function
.align 16

# void fi_swap(fc::Ctx* from /*%rdi*/, const fc::Ctx* to /*%rsi*/)
fi_swap:
    FI_CET_ENDBR
    # ---- 保存当前进 *from ----
    movq    %rbx, 0x10(%rdi)                # from->rbx
    movq    %rbp, 0x18(%rdi)                # from->rbp
    movq    %r12, 0x20(%rdi)                # from->r12
    movq    %r13, 0x28(%rdi)                # from->r13
    movq    %r14, 0x30(%rdi)                # from->r14
    movq    %r15, 0x38(%rdi)                # from->r15
    leaq    .Lfi_swap_resume(%rip), %rax
    movq    %rax, (%rdi)                    # from->rip = 恢复点
    movq    %rsp, 0x08(%rdi)                # from->rsp
    # ---- 恢复 *to ----
    movq    0x08(%rsi), %rsp
    movq    (%rsi), %rax                    # to->rip（冷 ctx=trampoline）
    movq    0x10(%rsi), %rbx
    movq    0x18(%rsi), %rbp
    movq    0x20(%rsi), %r12
    movq    0x28(%rsi), %r13
    movq    0x30(%rsi), %r14
    movq    0x38(%rsi), %r15
    jmp     *%rax                           # 切入对端
.Lfi_swap_resume:
    # 从对端切回：寄存器已由对端的"恢复段"装回=本调用入口时保存值
    xorl    %eax, %eax
    ret
.size fi_swap, .-fi_swap

.globl fi_make
.type fi_make,@function
.align 16

# void fi_make(fc::Ctx* ctx /*%rdi*/, void* stack_top /*%rsi*/,
#              void(*entry)(void*) /*%rdx*/, void* arg /*%rcx*/)
# stack_top ≡ 8 (mod 16)（C++ 侧保证）。参数槽 [top-0x10]=entry、
# [top-0x08]=arg；ctx->rsp=top-0x10。trampoline 首跳后栈帧从 top-0x10
# 向下生长，永不回到参数槽之上。
fi_make:
    FI_CET_ENDBR
    leaq    .Lfi_trampoline(%rip), %rax
    movq    %rax, (%rdi)                    # ctx->rip = trampoline
    leaq    -0x10(%rsi), %rax
    movq    %rax, 0x08(%rdi)                # ctx->rsp = top-0x10
    movq    %rdx, -0x10(%rsi)               # [top-0x10] = entry
    movq    %rcx, -0x08(%rsi)               # [top-0x08] = arg
    ret
.size fi_make, .-fi_make

# 冷上下文首跳落点：取栈上参数 → 调 entry(arg)。entry 不归路（收卷经
# fi_swap 回工人）——int3 仅为"违约返回"的显式爆点（SIGTRAP）。
# 对齐账：进入时 rsp=top-0x10（≡8 mod 16）→ pop/pop 后 rsp=top（≡8）→
# sub 8 后 ≡0（call 前的 ABI 对齐）→ call 压返回地址，entry 入口 ≡8。✓
# 局部符号（仅本文件 lea 引用，不出导出面）
.type .Lfi_trampoline,@function
.Lfi_trampoline:
    FI_CET_ENDBR                            # 间接跳目标（IBT 面必需）
    popq    %rax                            # entry   ([top-0x10])
    popq    %rdi                            # arg     ([top-0x08])
    subq    $8, %rsp                        # → ≡0 (mod 16)，call 后合规
    call    *%rax
    int3
.size .Lfi_trampoline, .-.Lfi_trampoline

.section .note.GNU-stack,"",@progbits
