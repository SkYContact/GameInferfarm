// fiber_backend.cpp — 后端选择（FARM_FIBER_BACKEND；缺省=构建档决定）
// 双向回退：显式选/缺省选的后端在本构建面不可用（如 POSIX 面无 WinFiber、
// 非 x86_64 ELF 无 SysV 切换体）时回退另一路并 stderr 提示——绝不静默
// nullptr 出门（RunLeg 解引用即炸）。
#include "fiber_backend.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace inferfarm {

IFiberBackend* FiberBackendSelect() {
    const char* e = std::getenv("FARM_FIBER_BACKEND");
    const char* want = (e && *e) ? e
#if defined(INFERFARM_CORO20)
                                 : "fcontext";   // C++20 档缺省
#else
                                 : "winfiber";   // C++17 回退档缺省=现役
#endif
    IFiberBackend* be = nullptr;
    if (std::strcmp(want, "fcontext") == 0) {
        be = MakeFcontextBackend();
        if (!be) {
            std::printf("[fiber] fcontext 后端不可用（限 MSVC/x64 或 POSIX "
                        "x86_64 ELF 构建），回退 winfiber\n");
            be = MakeWinFiberBackend();
        }
    } else {
        be = MakeWinFiberBackend();
        if (!be) {
            std::printf("[fiber] winfiber 后端不可用（限 Windows；本面回退 "
                        "fcontext）\n");
            be = MakeFcontextBackend();
        }
    }
    return be;
}

} // namespace inferfarm
