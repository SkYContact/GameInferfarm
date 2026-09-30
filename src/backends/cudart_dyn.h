// cudart_dyn.h — cudart 动态通道（ort/trt 后端共用；ai_infer.cpp LoadCudart
// 的抽取）。只借 memcpy 家+分配+流+图；全部动态符号查找，零链接依赖。
// kind: 1=H2D 2=D2H（cudaMemcpyKind 值）。
// 两平台装载面（2026-09-27 Linux 移植）：Windows=LoadLibraryA/GetProcAddress；
// POSIX=dlopen/dlsym（libcudart.so；目录语义 / 拼接，缺省基名 libcudart.so.12
// ——env FARM_CUDART_DLL 可覆写，同 Windows 面）。符号绑定块两平台共用。
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>

// 批拷贝属性（判决25 P1-5）：镜像 cudaMemcpyAttributes（CUDA v13.0
// driver_types.h；本仓不引 CUDA 头，布局手核：int enum + 2×cudaMemLocation
// {enum,uint} + uint = 24B @align4，static_assert 防布局漂移）。ort/trt 共用
struct HbAttr {
    int srcAccessOrder;        // 0x3=SrcAccessOrderAny（host 锚写稳、无在先流触碰）
    unsigned srcLocHint[2];    // cudaMemLocation（非托管/忽略场景全零）
    unsigned dstLocHint[2];
    unsigned flags;
};
static_assert(sizeof(HbAttr) == 24, "cudaMemcpyAttributes 布局漂移");
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

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
    int (*Memset)(void*, int, size_t) = nullptr;
    int (*MemsetAsync)(void*, int, size_t, void*) = nullptr;
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
    int (*EventElapsedTime)(float*, void*, void*) = nullptr;   // (ms, evA, evB) 排水拆解
    // 流捕获态查询（fence 诊断；可选符号）：0=None 1=Global 2=ThreadLocal
    // 3=Relaxed（cudaStreamCaptureStatus）
    int (*StreamIsCapturing)(void*, int*) = nullptr;
    // 跨流序（DATA17：共享状态池跨组换道=散射/gather 跨流无序——本符号+
    // EventRecord 补执行序；可选符号缺席=退化旧行为）
    int (*StreamWaitEvent)(void*, void*, unsigned int) = nullptr;
    // CUDA 12.8+ 批拷贝（P1-5 稀疏批 H2D 判决实验；可选符号——缺席/失败自动
    // 回退逐输入路径）。**ABI 双形态（2026-09-29 坑律入档，③D2D 批量化首跑
    // 段错误定谳）**：cu12(12.8/12.9)=9 参——numAttrs 后带 size_t* failIdx
    //（OUT 参数，须真实可写地址；cu12.9 头 cuda_runtime_api.h:7333）；
    // cu13=8 参（failIdx 移除；v13.0 头 6435）。单形态 typedef 跨版本=参数
    // 错位→stream 槽吃栈垃圾（驱动段错误/InvalidValue 双态——Linux cu12.9
    // 运行时+13.0 头镜像即中招）。MemcpyBatchAsyncV 按 cudaRuntimeGetVersion
    // 选形态调（ABI 归运行时库版本，非驱动版本）。
    int (*MemcpyBatch12)(void**, void**, size_t*, size_t, void*, size_t*,
                         size_t, size_t*, void*) = nullptr;
    int (*MemcpyBatch13)(void* const*, const void* const*, const size_t*,
                         size_t, void*, size_t*, size_t, void*) = nullptr;
    int (*RuntimeGetVersion)(int*) = nullptr;
    bool MemcpyBatchOk() const { return MemcpyBatch13 != nullptr; }
    int MemcpyBatchAsyncV(void* const* dsts, const void* const* srcs,
                          const size_t* sizes, size_t count, void* attrs,
                          size_t* attrIdxs, size_t numAttrs, void* stream) {
        int ver = 0;
        if (RuntimeGetVersion) RuntimeGetVersion(&ver);
        if (ver >= 13000)
            return MemcpyBatch13(dsts, srcs, sizes, count, attrs, attrIdxs,
                                 numAttrs, stream);
        size_t fail_idx = (size_t)-1;   // cu12：failIdx=OUT，须真实地址
        return MemcpyBatch12((void**)dsts, (void**)srcs,
                             const_cast<size_t*>(sizes), count, attrs,
                             attrIdxs, numAttrs, &fail_idx, stream);
    }
    // 宿主函数入流（CUDA 10+；P1 决策延迟链通知驱动，可选符号）：流到达该点
    // 时宿主回调执行（CUDA 回调线程）——回调内禁调 CUDA API，只发 OS 信号量
    int (*LaunchHostFunc)(void*, void (*)(void*), void*) = nullptr;

    bool ok = false;

    // 符号绑定块（两平台共用；g=按名取符号的原语——模板形参以兼容
    // Windows/POSIX 各自的 GetProcAddress/dlsym 捕获 lambda）
    template <class G>
    void Bind(G g) {
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
        Memset = (int (*)(void*, int, size_t))g("cudaMemset");
        MemsetAsync = (int (*)(void*, int, size_t, void*))g("cudaMemsetAsync");
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
        EventElapsedTime = (int (*)(float*, void*, void*))g("cudaEventElapsedTime");
        MemcpyBatch13 = (int (*)(void* const*, const void* const*, const size_t*,
                                 size_t, void*, size_t*, size_t, void*))
            g("cudaMemcpyBatchAsync");
        MemcpyBatch12 = (int (*)(void**, void**, size_t*, size_t, void*, size_t*,
                                 size_t, size_t*, void*))g("cudaMemcpyBatchAsync");
        RuntimeGetVersion = (int (*)(int*))g("cudaRuntimeGetVersion");
        LaunchHostFunc = (int (*)(void*, void (*)(void*), void*))g("cudaLaunchHostFunc");
        StreamIsCapturing = (int (*)(void*, int*))g("cudaStreamIsCapturing");
        StreamWaitEvent = (int (*)(void*, void*, unsigned int))g("cudaStreamWaitEvent");
    }

#ifdef _WIN32
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
        Bind(g);
        if (!Malloc || !Free || !HostAlloc || !FreeHost || !Memcpy || !DeviceSynchronize
            || !StreamCreate || !StreamDestroy || !MemcpyAsync
            || !Memset || !MemsetAsync) {
            std::fprintf(stderr, "[cudart] 缺导出符号\n");
            return false;
        }
        ok = true;
        std::fprintf(stderr, "[cudart] loaded %s\n", p.c_str());   // 归属审计
        return true;
    }
#else
    // POSIX 面（2026-09-27）：dlopen/dlsym，语义与 Windows 面对齐（目录空=
    // ld.so 搜索路径；FARM_CUDART_DLL 缺省 libcudart.so.12——pip 包
    // nvidia/cuda_runtime/lib 下，部署时设 FARM_CUDA_DIR 指它）
    bool Load(const std::string& cuda_dir) {
        if (ok) return true;
        const char* dll_env = getenv("FARM_CUDART_DLL");
        std::string dll = dll_env && *dll_env ? dll_env : "libcudart.so.12";
        std::string p = cuda_dir.empty() ? dll : cuda_dir + "/" + dll;
        void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            std::fprintf(stderr, "[cudart] dlopen %s 失败：%s\n",
                         p.c_str(), dlerror());
            return false;
        }
        auto g = [&](const char* n) { return (void*)dlsym(h, n); };
        Bind(g);
        if (!Malloc || !Free || !HostAlloc || !FreeHost || !Memcpy || !DeviceSynchronize
            || !StreamCreate || !StreamDestroy || !MemcpyAsync
            || !Memset || !MemsetAsync) {
            std::fprintf(stderr, "[cudart] 缺导出符号\n");
            return false;
        }
        ok = true;
        std::fprintf(stderr, "[cudart] loaded %s\n", p.c_str());   // 归属审计
        return true;
    }
#endif
};

} // namespace inferfarm
