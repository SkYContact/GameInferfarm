// fiber_backend_winfiber.cpp — WinFiber 后端：现役调用 1:1 提取
// （ConvertThreadToFiberEx/CreateFiberEx/SwitchToFiber/DeleteFiber/
//   ConvertFiberToThread，FIBER_FLAG_FLOAT_SWITCH 原样）。
// 行为零差：fiber_pool.cpp 走本后端=与提取前的直调逐句同源。
#ifdef _WIN32
#include "fiber_backend.h"
#include <windows.h>

namespace inferfarm {
namespace {

class WinFiberBackend final : public IFiberBackend {
public:
    const char* name() const override { return "winfiber"; }
    void* ConvertThread() override {
        return ConvertThreadToFiberEx(nullptr, FIBER_FLAG_FLOAT_SWITCH);
    }
    void ConvertBack(void* /*worker_ctx*/) override { ConvertFiberToThread(); }
    void* Create(void (*entry)(void*), void* arg) override {
        return CreateFiberEx(0, 0, FIBER_FLAG_FLOAT_SWITCH,
                             (LPFIBER_START_ROUTINE)entry, arg);
    }
    void Switch(void* to) override { SwitchToFiber(to); }
    void Destroy(void* game_ctx) override { DeleteFiber(game_ctx); }
};

} // namespace

IFiberBackend* MakeWinFiberBackend() { return new WinFiberBackend(); }

} // namespace inferfarm

#else
namespace inferfarm {
IFiberBackend* MakeWinFiberBackend() { return nullptr; }
} // namespace inferfarm
#endif
