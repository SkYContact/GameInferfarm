// cudart_dyn.h — cudart 动态通道（ort/trt 后端共用；ai_infer.cpp LoadCudart
// 的抽取）。只借 memcpy 家+分配+流+图；全部 GetProcAddress，零链接依赖。
// kind: 1=H2D 2=D2H（cudaMemcpyKind 值）。
#pragma once
#ifdef _WIN32
#include <windows.h>
#include <cstdio>
#include <string>

namespace inferfarm {

struct Cudart {
    void* (*Malloc)(void**, size_t) = nullptr;
    int (*Free)(void*) = nullptr;
    int (*HostAlloc)(void**, size_t, unsigned int) = nullptr;
    int (*FreeHost)(void*) = nullptr;
    void* (*Memcpy)(void*, const void*, size_t, int) = nullptr;
    int (*DeviceSynchronize)() = nullptr;
    int (*StreamCreate)(void**, unsigned int) = nullptr;
    int (*StreamCreateWithFlags)(void**, unsigned int) = nullptr;
    int (*StreamDestroy)(void*) = nullptr;
    int (*MemcpyAsync)(void*, const void*, size_t, int, void*) = nullptr;
    int (*SetDeviceFlags)(unsigned int) = nullptr;
    int (*GetDeviceFlags)(unsigned int*) = nullptr;
    int (*StreamSynchronize)(void*) = nullptr;
    int (*HostGetDevicePointer)(void**, void*, unsigned int) = nullptr;
    int (*StreamBeginCapture)(void*, unsigned int) = nullptr;
    int (*StreamEndCapture)(void*, void**) = nullptr;
    int (*GraphInstantiate)(void**, void*, unsigned long long) = nullptr;
    int (*GraphLaunch)(void*, void*) = nullptr;
    int (*GraphDestroy)(void*) = nullptr;
    int (*SetDevice)(int) = nullptr;            // 多卡守卫（可选符号：单卡行为不变）
    int (*GetDevice)(int*) = nullptr;
    int (*GetDeviceCount)(int*) = nullptr;
    // 事件族（ORT 零围栏实验用；可选符号——缺席=异步模式回落同步）
    int (*EventCreateWithFlags)(void**, unsigned int) = nullptr;
    int (*EventRecord)(void*, void*) = nullptr;
    int (*EventQuery)(void*) = nullptr;
    int (*EventSynchronize)(void*) = nullptr;
    int (*EventDestroy)(void*) = nullptr;
    // 流捕获态查询（fence 诊断；可选符号）：0=None 1=Global 2=ThreadLocal
    // 3=Relaxed（cudaStreamCaptureStatus）
    int (*StreamIsCapturing)(void*, int*) = nullptr;
    // CUDA 12.8+ 批拷贝（P1-5 稀疏批 H2D 判决实验；可选符号——缺席/失败自动
    // 回退逐输入路径）。签名（cuda_runtime_api.h:6435, v13.0）：
    //   cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs,
    //                        numAttrs, stream)
    int (*MemcpyBatchAsync)(void* const*, const void* const*, const size_t*,
                            size_t, void*, size_t*, size_t, void*) = nullptr;
    // 宿主函数入流（CUDA 10+；P1 决策延迟链通知驱动，可选符号）：流到达该点
    // 时宿主回调执行（CUDA 回调线程）——回调内禁调 CUDA API，只发 OS 信号量
    int (*LaunchHostFunc)(void*, void (*)(void*), void*) = nullptr;

    bool ok = false;

    // cuda_dir: cudart 所在目录（空=PATH 解析）；dll 文件名=env FARM_CUDART_DLL
    // （缺省 cudart64_12.dll——ORT wheel 可能是 CUDA13 构建，与 torch/lib 的
    // cudart12 混跑=双 runtime 进程，流归属/捕获行为会失真；复验时指 13）
    bool Load(const std::string& cuda_dir) {
        if (ok) return true;
        const char* dll_env = getenv("FARM_CUDART_DLL");
        std::string dll = dll_env && *dll_env ? dll_env : "cudart64_12.dll";
        std::string p = cuda_dir.empty() ? dll : cuda_dir + "\\" + dll;
        HMODULE h = cuda_dir.empty() ? LoadLibraryA(p.c_str())
                                     : LoadLibraryExA(p.c_str(), NULL,
                                                      LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!h) {
            std::fprintf(stderr, "[cudart] LoadLibrary %s 失败 GLE=%lu\n",
                         p.c_str(), GetLastError());
            return false;
        }
        auto g = [&](const char* n) { return (void*)GetProcAddress(h, n); };
        Malloc = (void* (*)(void**, size_t))g("cudaMalloc");
        Free = (int (*)(void*))g("cudaFree");
        HostAlloc = (int (*)(void**, size_t, unsigned int))g("cudaHostAlloc");
        FreeHost = (int (*)(void*))g("cudaFreeHost");
        Memcpy = (void* (*)(void*, const void*, size_t, int))g("cudaMemcpy");
        DeviceSynchronize = (int (*)())g("cudaDeviceSynchronize");
        StreamCreate = (int (*)(void**, unsigned int))g("cudaStreamCreate");
        StreamCreateWithFlags = (int (*)(void**, unsigned int))g("cudaStreamCreateWithFlags");
        StreamDestroy = (int (*)(void*))g("cudaStreamDestroy");
        MemcpyAsync = (int (*)(void*, const void*, size_t, int, void*))g("cudaMemcpyAsync");
        SetDeviceFlags = (int (*)(unsigned int))g("cudaSetDeviceFlags");
        GetDeviceFlags = (int (*)(unsigned int*))g("cudaGetDeviceFlags");
        StreamSynchronize = (int (*)(void*))g("cudaStreamSynchronize");
        HostGetDevicePointer = (int (*)(void**, void*, unsigned int))g("cudaHostGetDevicePointer");
        StreamBeginCapture = (int (*)(void*, unsigned int))g("cudaStreamBeginCapture");
        StreamEndCapture = (int (*)(void*, void**))g("cudaStreamEndCapture");
        GraphInstantiate = (int (*)(void**, void*, unsigned long long))g("cudaGraphInstantiate");
        GraphLaunch = (int (*)(void*, void*))g("cudaGraphLaunch");
        GraphDestroy = (int (*)(void*))g("cudaGraphDestroy");
        SetDevice = (int (*)(int))g("cudaSetDevice");
        GetDevice = (int (*)(int*))g("cudaGetDevice");
        GetDeviceCount = (int (*)(int*))g("cudaGetDeviceCount");
        EventCreateWithFlags = (int (*)(void**, unsigned int))g("cudaEventCreateWithFlags");
        EventRecord = (int (*)(void*, void*))g("cudaEventRecord");
        EventQuery = (int (*)(void*))g("cudaEventQuery");
        EventSynchronize = (int (*)(void*))g("cudaEventSynchronize");
        EventDestroy = (int (*)(void*))g("cudaEventDestroy");
        MemcpyBatchAsync = (int (*)(void* const*, const void* const*, const size_t*,
                                    size_t, void*, size_t*, size_t, void*))
            g("cudaMemcpyBatchAsync");
        LaunchHostFunc = (int (*)(void*, void (*)(void*), void*))g("cudaLaunchHostFunc");
        StreamIsCapturing = (int (*)(void*, int*))g("cudaStreamIsCapturing");
        if (!Malloc || !Free || !HostAlloc || !FreeHost || !Memcpy || !DeviceSynchronize
            || !StreamCreate || !StreamDestroy || !MemcpyAsync) {
            std::fprintf(stderr, "[cudart] 缺导出符号\n");
            return false;
        }
        ok = true;
        std::fprintf(stderr, "[cudart] loaded %s\n", p.c_str());   // 归属审计
        return true;
    }
};

} // namespace inferfarm
#endif // _WIN32
