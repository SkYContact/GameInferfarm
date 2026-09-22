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

    bool ok = false;

    // cuda_dir: cudart64_12.dll 所在目录（空=PATH 解析）
    bool Load(const std::string& cuda_dir) {
        if (ok) return true;
        std::string p = cuda_dir.empty() ? "cudart64_12.dll"
                                         : cuda_dir + "\\cudart64_12.dll";
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
        if (!Malloc || !Free || !HostAlloc || !FreeHost || !Memcpy || !DeviceSynchronize
            || !StreamCreate || !StreamDestroy || !MemcpyAsync) {
            std::fprintf(stderr, "[cudart] 缺导出符号\n");
            return false;
        }
        ok = true;
        return true;
    }
};

} // namespace inferfarm
#endif // _WIN32
