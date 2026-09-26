; fcontext.asm — Windows x64 有栈上下文切换（MASM64；ml64）
; 布局与 src/fcontext.h 的 fc::Ctx 同步（偏移硬编码，两处必须同步改）。
; 约定：
;   · 保存面=ABI 最小集：GP 非易变（rbx/rbp/rdi/rsi/r12-r15）+ xmm6-15
;     + MXCSR/FCW。易变寄存器跨 fi_swap 调用边界由编译器按 ABI 自理。
;   · 不动 TEB/FLS（WinFiber 才动）——SEH/异常穿越切换点=未定义（产线
;     契约无 throw 面；见 docs/design-cpp20-coroutines.md 第四节）。
;   · ctx 在堆上（不驻栈）：栈向下生长会踩穿驻栈的 ctx（首实现踩过的坑，
;     farm_test/探针 AV rip=0 定谳）。栈顶只放 entry/arg 两个参数槽，
;     trampoline 首跳即消费。
.code

; void fi_swap(fc::Ctx* from /*RCX*/, const fc::Ctx* to /*RDX*/)
fi_swap proc
    ; ---- 保存当前进 *from ----
    lea     rax, fswap_resume
    mov     [rcx+000h], rax                 ; from->rip = 恢复点
    mov     [rcx+008h], rsp                 ; from->rsp
    mov     [rcx+010h], rbx
    mov     [rcx+018h], rbp
    mov     [rcx+020h], rdi
    mov     [rcx+028h], rsi
    mov     [rcx+030h], r12
    mov     [rcx+038h], r13
    mov     [rcx+040h], r14
    mov     [rcx+048h], r15
    movdqu  [rcx+050h], xmm6
    movdqu  [rcx+060h], xmm7
    movdqu  [rcx+070h], xmm8
    movdqu  [rcx+080h], xmm9
    movdqu  [rcx+090h], xmm10
    movdqu  [rcx+0A0h], xmm11
    movdqu  [rcx+0B0h], xmm12
    movdqu  [rcx+0C0h], xmm13
    movdqu  [rcx+0D0h], xmm14
    movdqu  [rcx+0E0h], xmm15
    stmxcsr [rcx+0F4h]
    fnstcw  [rcx+0F0h]
    ; ---- 恢复 *to ----
    ldmxcsr [rdx+0F4h]
    fldcw   [rdx+0F0h]
    movdqu  xmm6,  [rdx+050h]
    movdqu  xmm7,  [rdx+060h]
    movdqu  xmm8,  [rdx+070h]
    movdqu  xmm9,  [rdx+080h]
    movdqu  xmm10, [rdx+090h]
    movdqu  xmm11, [rdx+0A0h]
    movdqu  xmm12, [rdx+0B0h]
    movdqu  xmm13, [rdx+0C0h]
    movdqu  xmm14, [rdx+0D0h]
    movdqu  xmm15, [rdx+0E0h]
    mov     rbx, [rdx+010h]
    mov     rbp, [rdx+018h]
    mov     rdi, [rdx+020h]
    mov     rsi, [rdx+028h]
    mov     r12, [rdx+030h]
    mov     r13, [rdx+038h]
    mov     r14, [rdx+040h]
    mov     r15, [rdx+048h]
    mov     rsp, [rdx+008h]
    mov     rax, [rdx+000h]
    jmp     rax                             ; 切入对端（冷 ctx=trampoline）
fswap_resume:
    ; 从对端切回：寄存器已由对端的"恢复段"装回=本调用入口时保存值
    xor     eax, eax
    ret
fi_swap endp

; void fi_make(fc::Ctx* ctx /*RCX*/, void* stack_top /*RDX*/,
;              void(*entry)(void*) /*R8*/, void* arg /*R9*/)
; stack_top ≡ 8 (mod 16)（C++ 侧保证）。参数槽 [top-0x10]=entry、
; [top-0x08]=arg；ctx->rsp=top-0x10。trampoline 首跳后栈帧从 top-0x30
; 向下生长，永不回到参数槽之上。
fi_make proc
    lea     rax, fi_trampoline
    mov     [rcx+000h], rax                 ; ctx->rip = trampoline
    lea     rax, [rdx-010h]
    mov     [rcx+008h], rax                 ; ctx->rsp = top-0x10
    mov     [rdx-010h], r8                  ; [top-0x10] = entry
    mov     [rdx-008h], r9                  ; [top-0x08] = arg
    fnstcw  [rcx+0F0h]                      ; FP 环境初值（首次被切入时生效）
    stmxcsr [rcx+0F4h]
    ret
fi_make endp

; 冷上下文首跳落点：取栈上参数 → 调 entry(arg)。entry 不归路（收卷经
; fi_swap 回工人）——int 3 仅为"违约返回"的显式爆点。
fi_trampoline proc
    mov     rax, [rsp]                      ; entry   ([top-0x10])
    mov     rcx, [rsp+8]                    ; arg     ([top-0x08])
    add     rsp, 10h                        ; → stack_top
    sub     rsp, 28h                        ; → top-0x28（≡0 mod 16，call 后合规）
    call    rax
    int     3
fi_trampoline endp

end
