// fiber_backend.cpp — 后端选择（FARM_FIBER_BACKEND；缺省=构建档决定）
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
        if (!be)
            std::printf("[fiber] fcontext 后端不可用（限 MSVC/x64 构建），回退 winfiber\n");
    }
    if (!be) be = MakeWinFiberBackend();
    return be;
}

} // namespace inferfarm
