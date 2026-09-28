// segvcatch.c — LD_PRELOAD 信号抓栈小工具（Linux 取证面）。
//
// 用法（段错误/断言/总线错的最后一道取证——gdb 扰动现场、ASAN 与 CUDA
// 冲突半可信时的替代通路，掼蛋线幻影行定谳同款）：
//   gcc -shared -fPIC -o segvcatch.so segvcatch.c
//   LD_PRELOAD=$(pwd)/segvcatch.so ./your_app
// 目标程序建议 -g -rdynamic 编译（否则栈只有地址）；输出走 dprintf 直写
// stderr=async-signal-safe，退出码 128+信号（编排器认汇总行不认退出码时
// 仍可从码分辨信号类）。
#define _GNU_SOURCE
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>
#include <stdio.h>
static void handler(int sig) {
    void* bt[64]; int n = backtrace(bt, 64);
    dprintf(2, "== SIG%d backtrace ==\n", sig);
    backtrace_symbols_fd(bt, n, 2);
    _exit(128 + sig);
}
__attribute__((constructor)) static void init(void) {
    signal(SIGSEGV, handler); signal(SIGABRT, handler); signal(SIGBUS, handler);
}
