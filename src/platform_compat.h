// platform_compat.h — Win32/POSIX 最小兼容垫片（非 Windows 构建通路，
// 2026-09-27，判决 24 POSIX 路线）。只收"框架层"的 OS 触点（tid/睡眠/
// 调用约定宏）；GPU/EP 面的降级在各后端内就地注记（ort_backend 的 dlopen
// 面、census 的线程普查面）。
//
// 纪律：Windows 侧包装=原 API 逐字透传（零行为差红线）；POSIX 侧语义
// 尽量对齐，差异点在函数头注释里写明。
#pragma once
#include <cstdint>

#ifdef _WIN32
#include <windows.h>

namespace inferfarm {

// 纤程体/回调的调用约定（Win32 需 WINAPI，POSIX=空）
#define FI_API WINAPI

// 本线程 OS tid（census 线程普查登记用）
inline unsigned long CurrentTid() { return GetCurrentThreadId(); }

// 毫秒睡眠（含小数 ms；Windows=Sleep 四舍五入，与原直调
// Sleep((DWORD)(stagger_ms + 0.5)) 逐位同语义）
inline void FiSleepMs(double ms) { Sleep((DWORD)(ms + 0.5)); }

} // namespace inferfarm
#else
#include <chrono>
#include <thread>
#include <ctime>
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#else
#include <pthread.h>
#endif

namespace inferfarm {

#define FI_API

// 本线程 OS tid。POSIX 降级点：Linux=gettid 系统调用（与 Windows tid 同
// 为内核级线程号，census 线程普查的分组键）；非 Linux POSIX= pthread 自身
// 句柄哈希（仅区分用，非内核 tid——census 分组语义降级，注明）。
inline unsigned long CurrentTid() {
#if defined(__linux__)
    return (unsigned long)::syscall(SYS_gettid);
#else
    return (unsigned long)(uintptr_t)pthread_self();
#endif
}

// 毫秒睡眠。POSIX：sleep_for 浮点毫秒（Windows 面的 1ms 定时量子是
// Windows 现象——POSIX nanosleep 精度原生更高，无需 timeBeginPeriod 类
// 补偿；判决 3 的窗真相是平台属性，不是本函数的义务）。
inline void FiSleepMs(double ms) {
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(ms));
}

} // namespace inferfarm
#endif
