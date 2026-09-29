// ort_backend.cpp — ONNX Runtime C++ 后端（不烤 TRT 的路线）。
//
// 与 TRT 后端的三个关键差异（实测/discipline 出处见 docs/design-judgments.md）：
//  1. **PerThreadContext 铁律**：ORT enable_cuda_graph 的会话绑线程。银行会话
//     （for_bank=true）的创建+热身+回放全在调度台线程上（BankScheduler::Init
//     已保证）；inline 会话强制关图（图会话不得跨线程用）。
//  2. **整设备同步血律**：ORT 不暴露内部流——不赌流序。前缀 H2D 用同步
//     cuMemcpy（返回即完成，与 ORT 流旗标无关=零竞态）；收割完成检测=
//     一次性 cuDeviceSynchronize + 前缀 D2H（围栏税 ~0.4-0.5ms/批，落在调度
//     台线程不在对局关键路径；GPU 侧多批仍并行——device sync 等的是 max 不是
//     sum）。v2 实验：legacy 默认流（synchronizing stream）邮箱盖章可去围栏
//     税，须 probe 门验证 ORT 流为 blocking 再开——此处不赌。
//  3. **权重热换（2026-09-29，推翻本行旧判"ORT 无热换 API 恒 false"）**：
//     升格权重（initializer 兼 graph input=ORT overridable initializer，
//     枚举期自动检测）绑定为本会话常驻设备缓冲（IOBinding 创建期钉地址），
//     RefitWeights(RW1)=validate-first 全会话先验后写 host+阻塞 H2D 覆写。
//     图捕获兼容：冻结地址原地改值，重放即读新权重。RW1 stash=实例级种子
//     （换心先于会话创建时暂存，Warmup memset 后播种）；未换心就发车=
//     SubmitBatch fail fast（零=垃圾，不许静默）。尺寸实测见
//     docs/reply-ygo-ort-hotswap.md §1（134MB=28.8ms，每 Run 0.525ms 零重传）。
//
// 图会话收益仍全额：批提交 3-4 次 WDDM 合成 1 次（ORT 内部 CUDA Graph）；
// 零拷贝直写槽/攒批/多批在飞等调度层收益与后端无关。
//
// 【PerThreadContext 与满座自驱的实测注记（2026-09-22）】铁律字面要求图会话
// 的创建/热身/回放同线程；本库银行会话的创建+热身在调度台线程，但**满座自驱
// 发车**让最后完笔的工人线程就地 RunWithBinding（回放跨线程）。已在
// onnxruntime 1.30.0 上以 FARM_BANK_WINDOW_FLOOR=1000 强制逐批自驱
// （self_dep=42/38）实测：结果逐位一致、无异常——即当前版本对回放的实际
// 约束比文档宽松。升级 ORT 版本时此结论须复验（自驱路径可退化为"置旗由
// 调度台发射"）。
//
// 【多设备改造（2026-09-22，判决15）】实例化：api/dll/env 计数全部从进程级
// 全局下沉为 OrtBackend 成员——一个实例=一套 ORT 运行时=一个设备组。双 ORT
// 共存改名律：CUDA 构建与 DML 构建是两颗同名 onnxruntime.dll，Windows 按基名
// 去重回柄——第二颗必须拷贝为 %TEMP%\inferfarm_ort_<n>.dll 再加载（依赖靠
// PATH 前插解析）。
//
// 【DML 路线（AMD/核显）】ort_ep="dml"：宿主绑定（输入输出直接绑我们的
// host arena，零 H2D/D2H）+ 同步 Run（返回即输出就绪——CompletionReached
// 恒真）。此同步假设由 ProbeGraph 门口实验兜底：若 Run 异步，两图案可分辨/
// 复跑稳定必挂，银行制拒绝启动。无图（graph 恒关）、无 cudart、写手自驱禁用
// （同步提交会阻塞写手 fiber 整个 GPU 时长）。实测 610M 核显 66K rows/s
// （玩具 MLP，DML 调度开销绑定）。
#include "inferfarm/backend.h"
#include "inferfarm/refit.h"
#include "cudart_dyn.h"
#include "onnxruntime_c_api.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
namespace inferfarm {
namespace ortplt {
// ---- 平台垫片（Windows=Win32 原样；POSIX=dlopen/无名信号量，Linux 移植
// 2026-09-27）。两侧同符号，调用点不设分叉。----
// CUDA host 回调的调用约定（x64 Win=唯一约定，POSIX=空；两义等价）
#define FI_STDCALL __stdcall
using HMODULE = ::HMODULE;
inline HMODULE LibLoad(const char* p) { return ::LoadLibraryA(p); }
inline void* LibSym(HMODULE h, const char* n) {
    return (void*)::GetProcAddress(h, n);
}
inline unsigned long LastErr() { return ::GetLastError(); }
// fence 完成信号量（FARM_ORT_ASYNC=3）：Win32 semaphore 原样
inline void* FenceSemCreate() {
    return (void*)::CreateSemaphoreA(nullptr, 0, 0x7FFFFFFF, nullptr);
}
inline bool FenceSemWait(void* sem, unsigned timeout_ms) {
    return ::WaitForSingleObject((HANDLE)sem, timeout_ms) == WAIT_OBJECT_0;
}
inline void FenceSemPost(void* sem) {
    ::ReleaseSemaphore((HANDLE)sem, 1, nullptr);
}
inline void FenceSemClose(void* sem) { ::CloseHandle((HANDLE)sem); }
} // namespace ortplt
} // namespace inferfarm
#else
// POSIX 降级面（Linux 移植 2026-09-27）：
//   · 动态装载=dlopen/dlsym（libonnxruntime.so；Windows 的 PATH 前插/基名
//     改名双 ORT 共存律是 Windows 专属病——System32 先于 PATH+按基名驻留
//     去重，POSIX 无此症，双 ORT 共存面暂未上 POSIX，见 LoadLib 注记）
//   · fence 完成信号量=无名 sem_t 堆置（sem_timedwait 100ms 烟雾等待）
//   · DML EP=Windows 专属（DirectML），POSIX 面配置级拒绝
#include <dlfcn.h>
#include <errno.h>
#include <semaphore.h>
#include <time.h>
namespace inferfarm {
namespace ortplt {
#define FI_STDCALL
using HMODULE = void*;
inline HMODULE LibLoad(const char* p) {
    void* h = ::dlopen(p, RTLD_NOW | RTLD_LOCAL);
    return h;
}
inline void* LibSym(HMODULE h, const char* n) { return ::dlsym(h, n); }
inline unsigned long LastErr() { return (unsigned long)errno; }
inline void* FenceSemCreate() {
    ::sem_t* s = new ::sem_t;
    if (::sem_init(s, 0, 0) != 0) { delete s; return nullptr; }
    return s;
}
inline bool FenceSemWait(void* sem, unsigned timeout_ms) {
    ::timespec ts;
    ::clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += (time_t)(timeout_ms / 1000);
    ts.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1000000000L;
    }
    return ::sem_timedwait((::sem_t*)sem, &ts) == 0;
}
inline void FenceSemPost(void* sem) { ::sem_post((::sem_t*)sem); }
inline void FenceSemClose(void* sem) {
    ::sem_destroy((::sem_t*)sem);
    delete (::sem_t*)sem;
}
} // namespace ortplt
} // namespace inferfarm
#endif

namespace inferfarm {

static Cudart g_cu;   // 进程一份（cudart 与 ORT 实例无关；DML 实例不加载）

// ORT C API 的 OrtStatus* 返回值带 warn_unused_result 属性（GCC/Clang 编译
// 必警，2026-09-27 上游转来）。fire-and-forget 调用点（失败语义由相邻显式
// 检查或上层错误路径覆盖：如 GetInputCount 后有范围闸、InputName 后有
// nm 判空）统一经此消费+释放；会致命的调用点仍走 if(status) 显式路径。
static void IgnoreStatus(const OrtApi* a, OrtStatus* st) {
    if (st) a->ReleaseStatus(st);
}

// DLL 注册表：双 ORT 共存改名律（基名冲突=拷贝改名再装；依赖 PATH 前插）
static std::mutex g_ort_dll_mx;
struct OrtDllRec { std::string base; std::string dir; ortplt::HMODULE h; };
static std::vector<OrtDllRec> g_ort_dlls;
static std::atomic<int> g_ort_dll_seq{0};

static std::string ToLower(std::string s) {
    for (auto& c : s) if (c >= 'A' && c <= 'Z') c += 32;
    return s;
}

static size_t OnnxElemSize(ONNXTensorElementDataType t) {
    switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return 1;
    default: return 0;
    }
}
static ElemDtype OnnxToElem(ONNXTensorElementDataType t) {
    switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return DTYPE_F32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return DTYPE_I64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return DTYPE_I32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return DTYPE_BOOL;
    default: return DTYPE_F32;
    }
}
static ONNXTensorElementDataType ElemToOnnx(ElemDtype t) {
    switch (t) {
    case DTYPE_F32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DTYPE_I64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DTYPE_I32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DTYPE_BOOL: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

// DML provider 挂载导出（dml_provider_factory.h 同签名；GetProcAddress 取）
typedef OrtStatus* (ORT_API_CALL* OrtDmlAppendFn)(OrtSessionOptions*, int);

// ScheduleSpin：设备等待恒忙等（Auto 策略睡 1-3ms/批）。须在 ORT 创建首个
// CUDA 上下文（首个 CUDA EP 会话）之前设——首个 CUDA 实例 LoadLib 时机即满足。
static void SetSpinFlagsOnce() {
    static bool done = false;
    if (done) return;
    done = true;
    if (g_cu.GetDeviceFlags && g_cu.SetDeviceFlags) {
        unsigned fl = 0;
        if (g_cu.GetDeviceFlags(&fl) == 0)
            g_cu.SetDeviceFlags(fl | 0x01 /*cudaDeviceScheduleSpin*/);
    }
}

// ORT 零围栏开关（env FARM_ORT_ASYNC，缺省关=现行为逐位不动）。语义：
//   1=实测判死通道（打印判决，不启用 async——2026-09-24 实验账见判决12）
//   2=翻案实验通道（用户流方案；判死保留复验用：换 cudart/ORT 版本后重跑）
//   3=fence 桥接通道（2026-09-24）：图尾 InferfarmFence custom op 在 ORT 自己
//     的流上盖事件章（tools/patch_fence.py 打补丁）——H2D/D2H 保持同步 memcpy
//     （确定性锚），Run 带 disable_synchronize_execution_providers，收割=
//     EventQuery（µs 级）。与 =2 的本质差：完全不碰 ORT 内部流序（上次跨流
//     无序判死的根治），只在其流尾偷听。每会话 CreateSession 读一次 env
//     （init 期调用，非热路径——static 缓存会挡住同进程 R6 门切档）。
static int AsyncEnvMode() {
    const char* e = getenv("FARM_ORT_ASYNC");
    return e && *e ? atoi(e) : 0;
}
static bool SharedEnv() {
    static const bool v = [] {
        const char* e = getenv("FARM_ORT_SHARED_ENV");
        return e && *e && atoi(e) == 1;
    }();
    return v;
}

// 声明式增量 H2D（判决25）env 面（均 CreateSession 期读取，非热路径）：
//   FARM_H2D_DELTA：总开关（缺省开；=0=杀手锏——点名面一律回落 full，零行为差
//     回退通道；配置声明仍进 spec 供观测）
//   FARM_H2D_DELTA_DEBUG：影子哨兵——每批 H2D 后影子 vs 宿主做全 face memcmp。
//     影子只同步"实际传输的段"=设备忠实镜像；非 0=设备侧与宿主漂移（声明漏
//     传/append-only 承诺违约/段规划 bug）。fprintf 报首个错+计数（R8 哨兵
//     门断言计数，防"打印没人看"的空过）。
static bool DeltaEnvOn() {
    const char* e = getenv("FARM_H2D_DELTA");
    return !(e && *e && atoi(e) == 0);
}
static bool DeltaDebugEnv() {
    const char* e = getenv("FARM_H2D_DELTA_DEBUG");
    return e && *e && atoi(e) == 1;
}
static std::atomic<long long> g_delta_violations{0};

// ==================== InferfarmFence custom op（FARM_ORT_ASYNC=3）====================
// 整设备同步围栏（~0.4-0.5ms/批）的拆除通道（判决12 翻案路标）：tools/
// patch_fence.py 在 fb onnx 图尾挂一个无输入 fence 节点（输出挂 graph output
// 防剪枝）。kernel 是**纯流探针**：Compute 里只从 KernelContext 拿 ORT 正在
// 执行的 GPU 流并登记进全局表——host 侧在 Run 提交后用这条流做 MemcpyAsync
// (D2H)+EventRecord（图外常规异步操作，零捕获语义参与）。确定性锚=H2D 同步
// memcpy；同流序 H2D(已完成)→GraphLaunch/eager 内核→D2H→事件 ⇒ 收割=
// EventQuery（µs 级）。
//
// 【为什么不把 EventRecord 放 kernel 里（v1 教训，2026-09-24 实测）】ORT 图
// 捕获（cudaStreamCaptureModeGlobal）窗口内的 cudaEventRecord 会把事件重置
// 为 pending 且不录进图（capture invalidated）——warmup 捕获跑后事件永
// notReady，另一形态直接 Concat cudaErrorInvalidValue（错误漂移）。eager
// 路径（图关）kernel 内 record 完全正常（inline 392 局/s 实测过）。v2 把
// record 挪到图外 host 侧后两种模式统一成立。
//
// 流指针生命周期=EP 统一流（enable_cuda_graph 时 ORT 自建 NonBlocking 流，
// 会话终身）——比事件还稳。host 认领在 Warmup（首跑 Compute 必已登记；
// CreateSession 后 kernel 尚未执行过）。ticket 握手与 v1 同：运行时指针
// 烤不进 onnx 属性，host 领票→kernel 按当前票号登记→Warmup 持票认领。
struct FenceKernelData {
    uint64_t ticket = 0;
    const OrtApi* api = nullptr;
};

static std::mutex g_fence_mx;
static uint64_t g_fence_armed_ticket = 0;   // 当前武装票号（0=无主）。仅 fence
                                            // 会话的 CreateSession 窗口内非零：
                                            // probe/inline 等其他会话对 patched
                                            // 模型也会实例化 fence kernel，武装
                                            // 窗口外的 kernel 拿到 0=不登记（堵
                                            // "他人登记进我票号"的污染/泄漏）
static std::unordered_map<uint64_t, std::vector<void*>> g_fence_streams;
static std::atomic<long long> g_fence_engaged{0};   // 累计真启用会话数（Warmup
                                                    // 烟雾通过才 ++；静默回落不
                                                    // 计——R6 门据此断言"真启用"）

static uint64_t FenceTicketArm() {
    std::lock_guard<std::mutex> lk(g_fence_mx);
    return ++g_fence_armed_ticket;
}
static void FenceTicketDisarm() {
    std::lock_guard<std::mutex> lk(g_fence_mx);
    g_fence_armed_ticket = 0;
}
static void FenceTicketDrop(uint64_t t) {
    std::lock_guard<std::mutex> lk(g_fence_mx);
    g_fence_streams.erase(t);   // 未认领条目回收（warmup 失败路径的微泄漏）
}
static void* FenceTicketClaim(uint64_t t) {
    std::lock_guard<std::mutex> lk(g_fence_mx);
    auto it = g_fence_streams.find(t);
    if (it == g_fence_streams.end() || it->second.empty()) return nullptr;
    void* st = it->second.back();
    it->second.pop_back();
    if (it->second.empty()) g_fence_streams.erase(it);
    return st;
}

// HbAttr 已提升 cudart_dyn.h（ort/trt 批拷贝共用；③批量 D2D）
// fence 完成回调（cudaLaunchHostFunc；CUDA 回调线程执行）：只发 OS 信号量
// ——回调内禁调 CUDA API（官方契约），ReleaseSemaphore 足够
// fence 完成回调（cudaLaunchHostFunc；CUDA 回调线程执行）：置 done 旗标 +
// 发 OS 信号量（唤醒 WMO 中的调度台）。回调内禁调 CUDA API（官方契约），
// 两者皆合法；ctx=FenceCbCtx（会话销毁前必无未执行回调——收割先于销毁）
struct FenceCbCtx {
    void* sem = nullptr;
    std::atomic<bool>* done = nullptr;
};
static void FI_STDCALL FenceReleaseCb(void* p) {
    FenceCbCtx* c = (FenceCbCtx*)p;
    c->done->store(true, std::memory_order_release);
    ortplt::FenceSemPost(c->sem);
}

static OrtStatusPtr FenceCreateKernel(const OrtCustomOp*, const OrtApi* api,
                                      const OrtKernelInfo*, void** kernel) {
    FenceKernelData* k = new FenceKernelData();
    k->api = api;
    {
        std::lock_guard<std::mutex> lk(g_fence_mx);
        k->ticket = g_fence_armed_ticket;   // 窗口外=0（无主，Compute 不登记）
    }
    *kernel = k;
    return nullptr;
}
static void FenceKernelDestroy(void* kernel) {
    delete (FenceKernelData*)kernel;
}
static OrtStatusPtr FenceCompute(void* kernel, OrtKernelContext* context) {
    FenceKernelData* k = (FenceKernelData*)kernel;
    if (k->ticket == 0) return nullptr;   // 无主 kernel（probe/inline 对 patched
                                          // 模型）——静默不登记
    // 诊断开关（FARM_ORT_FENCE_DBG=1）：流归属/捕获态一击定位
    static const bool dbg = [] {
        const char* e = getenv("FARM_ORT_FENCE_DBG");
        return e && *e && atoi(e) == 1;
    }();
    void* stream = nullptr;
    OrtStatusPtr st = k->api->KernelContext_GetGPUComputeStream(context, &stream);
    if (st) {
        if (dbg) std::fprintf(stderr, "[fence][dbg] GetStream 返回 status\n");
        return st;
    }
    if (!stream) {   // 非 CUDA 执行（EP 回落 CPU 等）——不登记，Warmup 自检
                     // 会抓住并回落同步
        if (dbg) std::fprintf(stderr, "[fence][dbg] stream=null（不登记）\n");
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(g_fence_mx);
        auto& v = g_fence_streams[k->ticket];
        if (v.empty() || v.back() != stream) v.push_back(stream);
    }
    if (dbg) {
        int cap = -1;
        if (g_cu.StreamIsCapturing) g_cu.StreamIsCapturing(stream, &cap);
        std::fprintf(stderr, "[fence][dbg] 登记流 %p ticket=%llu capturing=%d"
                     "（0=None 1=Global）\n", stream,
                     (unsigned long long)k->ticket, cap);
    }
    return nullptr;
}

// vtable：继承聚合零填 + 构造器逐槽位填（C++ 无指定初始化器）；可选回调
// （GetMayInplace 等）保持 null=ORT 判空跳过
struct FenceOp : OrtCustomOp {
    FenceOp() {
        std::memset(this, 0, sizeof(*this));
        version = ORT_API_VERSION;
        CreateKernelV2 = &FenceCreateKernel;
        GetName = [](const OrtCustomOp*) { return "InferfarmFence"; };
        GetExecutionProviderType =
            [](const OrtCustomOp*) { return "CUDAExecutionProvider"; };
        GetInputTypeCount = [](const OrtCustomOp*) -> size_t { return 0; };
        GetOutputTypeCount = [](const OrtCustomOp*) -> size_t { return 1; };
        GetOutputType = [](const OrtCustomOp*, size_t) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        };
        KernelComputeV2 = &FenceCompute;
        KernelDestroy = &FenceKernelDestroy;
        GetInputCharacteristic = [](const OrtCustomOp*, size_t) {
            return OrtCustomOpInputOutputCharacteristic::
                INPUT_OUTPUT_OPTIONAL;   // 实际不调（count=0）
        };
        GetOutputCharacteristic = [](const OrtCustomOp*, size_t) {
            return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
        };
    }
};
static FenceOp g_fence_op;   // 进程一份（注册进每个非 DML 会话的域）

struct OrtIn {
    InputMeta meta;
    void* host = nullptr;   // pinned carve（dml=普通页 carve，语义同）
    void* dev = nullptr;    // device carve（dml=host 同址，仅 CUDA 路径用）
    OrtValue* val = nullptr;
    // ---- 声明式增量 H2D（判决25 及其 headlive 扩展；旗标在 meta.append /
    //      meta.headlive——单一事实源，spec 与运行时同字段，无双写漂移）----
    int stride = 0;          // 一个深度单位的字节数 = row_bytes/dims[1]（nd<2 面不可 append）
    int max_depth = 0;       // 深度上限 = dims[1]
    std::vector<int> synced;     // 每行已同步深度（影子=设备的忠实镜像）；kFullSync=整行
    std::vector<int> declared;   // 每行本批申报深度（组装期写入；SubmitBatch 消费后
                                 // 复位 -1=未声明→整行兜底）
    void* shadow = nullptr;      // 影子缓冲 carve（宿主侧设备镜像，与面等字节）
    static constexpr int kFullSync = 1 << 30;   // "整行已同步"哨兵（>任何合法深度）
};
struct OrtOut {
    OutputMeta meta;
    void* dev = nullptr;
    float* host = nullptr;
    size_t bytes = 0;
    OrtValue* val = nullptr;   // 交 iob 钉住（进程寿命）
};
struct OrtSess {
    ortplt::HMODULE dll = nullptr;   // 持引用（进程寿命不卸；POSIX 面不 dlclose）
    const OrtApi* api = nullptr;   // 所属实例的 api（实例终身不毁=指针终身有效）
    OrtEnv* env = nullptr;
    OrtSession* sess = nullptr;
    OrtMemoryInfo* bind_mem = nullptr;   // CUDA 路径="Cuda"，DML 路径="Cpu"
    OrtIoBinding* iob = nullptr;
    std::vector<OrtIn> ins;
    std::vector<OrtOut> outs;
    void* in_h_arena = nullptr;  size_t in_h_bytes = 0;
    // DML dep 拆段累计（发射线程独写独读+打印即清零；可观测面 2026-09-26）
    long long dml_rebind_ns = 0, dml_run_ns = 0;
    unsigned dml_dep_cnt = 0;
    void* in_d_arena = nullptr;  size_t in_d_bytes = 0;
    void* out_h_arena = nullptr; size_t out_h_bytes = 0;
    void* out_d_arena = nullptr; size_t out_d_bytes = 0;
    bool graph_on = false;
    bool dml = false;
    int dev_id = 0;
    bool pop_dirty = false;   // population 面脏旗（cuda：下次 SubmitBatch 全量 H2D）
    bool pop_mode = false;    // population 路由模式（cfg.population_input 命中）
    std::vector<int> mid_like;   // 路由键输入下标（1-D i64 非 population——批尾
                                 // 毒化目标；路由图约定：i64 标量列=mid）
    // ---- 升格权重面（overridable initializer，RefitWeights 通道）----
    bool any_weight = false;      // 存在权重面（SubmitBatch fail-fast 门加速度器）
    bool weights_seeded = false;  // 全部权重面已从 stash 播种（Warmup 置位；
                                  // false 时 SubmitBatch 拒发车=零权重是垃圾）
    int slots = 64;
    // 完成协议（整设备同步血律）：Submit 后首个 CompletionReached 做一次
    // device sync + 前缀 D2H，随后同 seq 恒 true（dml：Run 同步=恒真）
    unsigned seq = 0;
    int last_n =  0;
    bool synced_for_seq = false;
    // DML 专属发射线程（队头阻塞解药，2026-09-22）：DML 的 Run 是同步的——
    // 若在共享调度台线程上执行，多 ms 的核显计算会把全部异构组的收割/发车
    // 挡在身后（实测 4cuda+dml(2链) 反而 1.87s vs 0.80s）。解法=每 DML 会话
    // 一条发射线程：SubmitBatch 投递即返回，完成旗标轮询（TRT 邮箱同款衣服）。
    std::thread* helper = nullptr;
    std::mutex h_mx;
    std::condition_variable h_cv;
    unsigned h_pending = 0;                 // 在飞作业 seq（0=无；银行单飞≤1）
    std::atomic<unsigned> h_done{0};        // 已完成作业 seq
    bool h_stop = false;
    // ORT 零围栏（FARM_ORT_ASYNC=1；判决12"整设备同步"结论的实测点补全，
    // 2026-09-24）：用户流（user_compute_stream）+ RunOptions 关 EP 同步
    // （disable_synchronize_execution_providers）+ 事件邮箱（TRT 同款收割
    // 通道）。H2D→Run→D2H→EventRecord 同流序 ⇒ 逐位不变性由序保证。
    // 探测会话恒同步（LoadSpec 语义依赖同步 Run）。
    bool async = false;
    // fence 桥接（FARM_ORT_ASYNC=3）：fence kernel 是流探针（Compute 只登记
    // ORT 内部流）；host 在 Run 提交后用该流做 D2H MemcpyAsync，再排
    // LaunchHostFunc 让 CUDA 回调线程 ReleaseSemaphore——**完成=通知驱动**
    // （零轮询零量子；调度台 WaitForMultipleObjects 直等信号量）。
    // 确定性锚=H2D 同步 memcpy；银行单飞 ⇒ 信号量计数∈{0,1} 精确映射本批。
    // 认领在 Warmup（首跑 Compute 必已登记；CreateSession 后 kernel 未跑过）。
    bool fence = false;
    uint64_t fence_ticket = 0;
    void* fence_stream = nullptr;   // 认领自注册表的 ORT EP 统一流
    void* fence_sem = nullptr;      // 完成信号量（Windows HANDLE；每批恰一次释放）
    FenceCbCtx fence_cb_ctx{};      // 回调 ctx（sem + done 旗标指针）
    std::atomic<bool> flight_done{false};   // 本批完成旗标（回调置位；CompletionReached 读）
    OrtCustomOpDomain* domain = nullptr;   // fence 域（ReleaseSession 后释放）
    // dep 三段分解累计（P1 仪器；调度台单线程写，打印即清零）
    unsigned long long dep_h2d_ns = 0, dep_run_ns = 0, dep_d2h_ns = 0;
    unsigned dep_cnt = 0;
    // P1-5 批拷贝 scratch（FARM_H2D_BATCH=1 且符号在）：稀疏批多输入 H2D
    // 合并为一次 cudaMemcpyBatchAsync（dep=宿主提交税∝提交次数）
    bool h2d_async = false;   // fence H2D 异步入流（FARM_H2D_ASYNC=1 opt-in；
                              // 缺省阻塞 memcpy=YGO 实测正解，见判决17 补记）
    bool h2d_batch = false;   // 依赖 h2d_async（批 API 只有异步形态）
    std::vector<void*> hb_dst, hb_src;
    std::vector<size_t> hb_sizes, hb_attridx;
    // 声明式增量 H2D（判决25）：append 面存在性（SubmitBatch 热路径分流的唯一
    // 门——无 append 面=与旧路径逐指令同）；debug 哨兵（影子 vs 宿主全 face
    // memcmp，非 0=声明漏传/承诺违约）；dep h2d 段字节数（A/B 测量面）
    bool any_append = false;
    bool delta_dbg = false;
    bool delta_vio_printed = false;   // 哨兵打印去重（首个错即可；计数不封顶）
    void* shadow_blk = nullptr;   // append 面影子总块（一次分配按面 carve）
    size_t shadow_blk_bytes = 0;
    unsigned long long dep_h2d_bytes = 0;   // 累计（与 dep_cnt 同节奏打印清零）
    void* stream = nullptr;
    void* event = nullptr;
    OrtRunOptions* ro = nullptr;
};

class OrtBackend : public InferBackend {
public:
    const char* Name() const override { return dml_ ? "ort-dml" : "ort"; }

    // ---- 会话注册表+RW1 stash（RefitWeights 广播面，镜像 cpu_backend）----
    // refit 与腿互斥（Farm 契约"前置=腿已返回"）⇒ mx_ 只护注册表与 stash
    // 的结构性读写，不护 SubmitBatch 热路径。
    std::mutex refit_mx_;
    std::vector<OrtSess*> sessions_;       // refit 广播面（Track/Untrack）
    std::vector<char> stash_blob_;         // 实例种子 RW1（整块持有；
                                           // Rw1Entry.data 指向此块内 carve）
    std::vector<Rw1Entry> stash_ents_;     // 种子条目（名字→权重字节）
    void Track(OrtSess* s) {
        std::lock_guard<std::mutex> lk(refit_mx_);
        sessions_.push_back(s);
    }
    void Untrack(OrtSess* s) {
        std::lock_guard<std::mutex> lk(refit_mx_);
        for (size_t i = 0; i < sessions_.size(); i++)
            if (sessions_[i] == s) { sessions_.erase(sessions_.begin() + (long)i); break; }
    }


    // ORT 图会话绑调度台线程（PerThreadContext 铁律）+ DML 同步提交不可自驱：
    // 两条路线统一禁写手自驱，发车一律走调度台
    bool DispatchFromWriterOk() const override { return false; }

    bool LoadSpec(const ModelConfig& cfg, int slots, ModelSpec& out) override {
        dml_ = (cfg.ort_ep == "dml");
        if (!LoadLib(cfg)) return false;
        if (!dml_) {
            if (!g_cu.Load(cfg.cuda_dir)) return false;
            SetSpinFlagsOnce();
        }
        const OrtApi* a = api_;
        // 探测会话（临时；用于元数据枚举）——图关（枚举无需图，避免 PerThread
        // Context 约束牵连 init 线程）
        OrtSess* probe = CreateSession(cfg, slots, /*for_bank=*/false, &out);
        if (!probe) return false;
        DestroySession(probe);
        return true;
    }

    void* CreateSession(const ModelConfig& cfg, const ModelSpec& spec,
                        bool for_bank) override {
        return CreateSession(cfg, spec.slots, for_bank, nullptr);
    }

    // 探测砍除通道（能力位）：LoadSpec 的探测会话可砍——首个真实银行会话
    // 顺带产出 spec（内部 4 参 CreateSession 本就支持 spec_out）
    bool ProbeFreeSpec() const override { return true; }
    void* CreateSessionWithSpec(const ModelConfig& cfg, int slots,
                                bool for_bank, ModelSpec* spec_out) override {
        return CreateSession(cfg, slots, for_bank, spec_out);
    }
    // 完成等待句柄（fence=v3 完成信号量；调度台 WMO 直等=通知驱动零轮询）
    void* CompletionWaitHandle(void* session) override {
        OrtSess* s = (OrtSess*)session;
        return (s && s->fence) ? s->fence_sem : nullptr;
    }

    bool Warmup(void* session) override {
        OrtSess* s = (OrtSess*)session;
        memset(s->in_h_arena, 0, s->in_h_bytes);
        // 升格权重播种（RW1 stash→host+H2D；memset 抹掉创建期种子，此处重播
        // 幂等）。stash 覆盖不全=权重面留零——发车侧 fail fast 兜底（不静默）。
        if (s->any_weight && !SeedWeights(s))
            std::fprintf(stderr, "[ort] 权重面播种不全（stash 空/名字覆盖不全）"
                         "——RefitWeights 前发车将被拒\n");
        // 一击必中探针（FARM_ORT_CAPTEST=1，外部审计建议的拦截器思路的零依赖
        // 版）：把流交给 ORT 前自捕一次。自捕 OK=流干净可捕获 ⇒ ORT 的 900
        // 来自它没用这条流；自捕 900=流已被捕/不可捕 ⇒ 有谁先动了它。
        {
            static const bool captest = [] {
                const char* e = getenv("FARM_ORT_CAPTEST");
                return e && *e && atoi(e) == 1;
            }();
            if (captest && s->async && s->graph_on && !s->dml) {
                void* g = nullptr;
                int rc1 = g_cu.StreamBeginCapture
                    ? g_cu.StreamBeginCapture(s->stream, 0) : -99;
                int rc2 = 0;
                if (rc1 == 0) {
                    rc2 = g_cu.StreamEndCapture(s->stream, &g);
                    if (g && g_cu.GraphDestroy) g_cu.GraphDestroy(g);
                }
                std::fprintf(stderr, "[ort][captest] stream=%p BeginCapture=%d "
                             "EndCapture=%d（0=干净可捕；900=已被捕/不可捕）\n",
                             s->stream, rc1, rc2);
                std::fflush(stderr);
            }
        }
        // 零填充 3 跑（enable_cuda_graph 内部前两跑构图/捕获——捕获窗口内
        // 不容地址/形状变化；地址已钉死=满足）
        for (int r = 0; r < 3; r++) {
            if (!s->dml) {
                if (g_cu.Memcpy(s->in_d_arena, s->in_h_arena, s->in_h_bytes, 1))
                    return false;
            }
            if (!RunOnce(s)) {
                std::fprintf(stderr, "[ort] Warmup 失败（async=%d graph=%d fence=%d）\n",
                             (int)s->async, (int)s->graph_on, (int)s->fence);
                return false;
            }
            if (!s->dml) g_cu.DeviceSynchronize();
        }
        // fence 认领+烟雾（FARM_ORT_ASYNC=3）：首跑 Compute 必已登记 ORT 流——
        // 此刻持票认领；烟雾=LaunchHostFunc 入流→等信号量（回调线程释放，
        // 100ms 超时）验证整链。失败（模型未打补丁/EP 回落 CPU）=回落同步
        if (s->fence) {
            s->fence_stream = FenceTicketClaim(s->fence_ticket);
            bool ok = s->fence_stream && g_cu.LaunchHostFunc && s->fence_sem;
            if (ok)
                ok = g_cu.LaunchHostFunc(s->fence_stream, &FenceReleaseCb,
                                         &s->fence_cb_ctx) == 0;
            if (ok) {
                ok = ortplt::FenceSemWait(s->fence_sem, 100);
                g_cu.DeviceSynchronize();
            }
            if (!ok) {
                std::fprintf(stderr, "[ort] FARM_ORT_ASYNC=3 fence 认领/烟雾失败"
                             "（流=%p——模型未打 fence 补丁？）→ 本会话回落同步"
                             "路径\n", s->fence_stream);
                s->fence = false;
                s->fence_stream = nullptr;
                if (s->ro) { api_->ReleaseRunOptions(s->ro); s->ro = nullptr; }
            } else {
                g_fence_engaged.fetch_add(1, std::memory_order_relaxed);
            }
        }
        std::fprintf(stderr, "[ort] 热身就绪 slots=%d%s%s%s\n", s->slots,
                     s->dml ? "，DML=宿主绑定+同步 Run（无图无围栏；dep 拆段每 64 批打印）"
                            : (s->graph_on ? "，CUDA Graph=开（银行会话：调度台线程绑定）"
                                           : "，CUDA Graph=关"),
                     s->async ? "，零围栏=开（用户流+事件收割）" : "",
                     s->fence ? "，fence 桥接=开（H2D 同步锚+信号量完成通知）" : "");
        return true;
    }

    bool ProbeGraph(void* session) override {
        // 图地址实验：绑定地址钉死后，Run 读当前设备内存值。
        // 两图案可分辨 + 复跑稳定 + 换数据输出跟着变。（dml：同一实验兜底
        // "Run 同步"假设——异步则两图案必同读陈旧宿主数据=可分辨挂=拒绝启动）
        OrtSess* s = (OrtSess*)session;
        const OrtApi* a = api_;
        std::vector<char> ref1, ref2, r2b;
        auto snap = [&](std::vector<char>& v) {
            if (!s->dml) {
                g_cu.DeviceSynchronize();
                for (size_t j = 0; j < s->outs.size(); j++)
                    g_cu.Memcpy(s->outs[j].host, s->outs[j].dev, s->outs[j].bytes, 2);
            }
            v.assign((const char*)s->out_h_arena,
                     (const char*)s->out_h_arena + s->out_h_bytes);
        };
        auto run_pat = [&]() {
            if (!s->dml)
                g_cu.Memcpy(s->in_d_arena, s->in_h_arena, s->in_h_bytes, 1);
            RunOnce(s);
        };
        FillPattern(s, 1);
        run_pat();
        snap(ref1);
        RunOnce(s);   // 复跑（不重拷——设备数据未变，纯图回放稳定性）
        snap(r2b);
        FillPattern(s, 2);
        run_pat();
        snap(ref2);
        bool diff = ref1 != ref2;
        bool stable = ref1 == r2b;
        bool bok = diff && stable;
        // 增量面 zero 基重建（判决25）：图案实验把显存写花——影子（恒未写=零）
        // 回拷恢复"设备=影子=零基"起步态（synced 恒 0 未动）。仅各组首家银行走
        // ProbeGraph；其余银行 Warmup 整块零拷后无人再碰=天然零基。
        if (s->any_append && !s->dml) {
            if (g_cu.SetDevice) g_cu.SetDevice(s->dev_id);
            for (auto& i : s->ins)
                if (i.meta.append || i.meta.headlive)
                    g_cu.Memcpy(i.dev, i.shadow,
                                i.meta.row_bytes * (size_t)s->slots, 1);
        }
        std::printf("[ort-probe] 两图案可分辨=%d 复跑稳定=%d%s\n",
                    (int)diff, (int)stable, bok ? "" : " ←FAIL");
        std::fflush(stdout);
        return bok;
    }

    void DestroySession(void* session) override {
        OrtSess* s = (OrtSess*)session;
        if (!s) return;
        Untrack(s);   // 注册表先摘除（refit 广播面不再指向将毁会话）
        if (s->helper) {   // 发射线程先停（Join 后再动会话对象）
            {
                std::lock_guard<std::mutex> lk(s->h_mx);
                s->h_stop = true;
            }
            s->h_cv.notify_all();
            s->helper->join();
            delete s->helper;
            s->helper = nullptr;
        }
        const OrtApi* a = api_;
        if (s->iob) a->ReleaseIoBinding(s->iob);
        if (s->ro) a->ReleaseRunOptions(s->ro);
        if (s->event && g_cu.EventDestroy) g_cu.EventDestroy(s->event);
        if (s->fence_sem) ortplt::FenceSemClose(s->fence_sem);   // fence 完成信号量
        // （=2 用户流与 fence 桥接的事件均 host 自建——此处统一销毁）
        if (s->stream && g_cu.StreamDestroy) g_cu.StreamDestroy(s->stream);
        for (auto& i : s->ins) if (i.val) a->ReleaseValue(i.val);
        for (auto& o : s->outs) if (o.val) a->ReleaseValue(o.val);
        if (s->sess) a->ReleaseSession(s->sess);
        if (s->domain) a->ReleaseCustomOpDomain(s->domain);   // 头文件契约：
        // 域须在所有用它的会话释放之后再删——放 ReleaseSession 后
        FenceTicketDrop(s->fence_ticket);   // 未认领流条目回收（ticket 0=无操作）
        if (s->env && !SharedEnv()) a->ReleaseEnv(s->env);   // 共享 env=进程寿命，不释放
        if (s->bind_mem) a->ReleaseMemoryInfo(s->bind_mem);
        if (s->dml) {
#ifdef _WIN32
            if (s->in_h_arena) VirtualFree(s->in_h_arena, 0, MEM_RELEASE);
            if (s->out_h_arena) VirtualFree(s->out_h_arena, 0, MEM_RELEASE);
#else
            // POSIX 不可达（DML 会话建不出来）；防御面 free 对齐分配
            if (s->in_h_arena) std::free(s->in_h_arena);
            if (s->out_h_arena) std::free(s->out_h_arena);
#endif
        } else {
            if (s->in_h_arena) g_cu.FreeHost(s->in_h_arena);
            if (s->in_d_arena) g_cu.Free(s->in_d_arena);
            if (s->out_h_arena) g_cu.FreeHost(s->out_h_arena);
            if (s->out_d_arena) g_cu.Free(s->out_d_arena);
        }
        if (s->shadow_blk) {   // 增量影子块（判决25）
#ifdef _WIN32
            VirtualFree(s->shadow_blk, 0, MEM_RELEASE);
#else
            std::free(s->shadow_blk);
#endif
        }
        delete s;
    }

    void* InputRow(void* session, const char* name, int slot, size_t* row_bytes) override {
        OrtSess* s = (OrtSess*)session;
        for (auto& i : s->ins)
            if (i.meta.name == name) {
                // 升格权重面=RefitWeights 专属，组装面不存在（nullptr=SlotWriter
                // "未知名"契约）。唯一写入口执法点：inline 清零/Claim 回退/bank
                // 视图回退经此全部自动安全。
                if (i.meta.weight) return nullptr;
                if (row_bytes) *row_bytes = i.meta.row_bytes;
                return (char*)i.host + (size_t)slot * i.meta.row_bytes;
            }
        return nullptr;
    }

    bool SubmitBatch(void* session, int n_rows, unsigned& seq_out) override {
        OrtSess* s = (OrtSess*)session;
        if (n_rows > s->slots) n_rows = s->slots;
        if (n_rows < 1) n_rows = 1;
        // 升格权重未换心=权重面全零（Warmup 零基 memset 后无种子）——零权重
        // 输出是垃圾，拒发车（loud，不静默）。首次 RefitWeights 即解除。
        if (s->any_weight && !s->weights_seeded) {
            std::fprintf(stderr, "[ort] 升格权重面未换心——拒发车（先 "
                         "RefitWeights(rw1)，或 init 期配 ModelConfig.refit_weights）\n");
            return false;
        }
        s->seq++;
        s->last_n = n_rows;
        s->synced_for_seq = false;
        const long long dep_t0 = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        // 批尾毒化（路由模式死行协议，判决16）：未领槽位 [n, slots) 的路由键
        // 置 -1（0xFF）——陈旧 mid 参与图内散射会破坏 (p,j) 唯一性（溢出/碰撞
        // 经 ScatterND 原子写污染新鲜行=活性非确定性）。图侧 v2.2 把 -1 路由
        // 到专属死块，毒行输出不被收割。dml 直读宿主=写完即生效。
        if (s->pop_mode && n_rows < s->slots) {
            for (int mi : s->mid_like) {
                OrtIn& m = s->ins[(size_t)mi];
                memset((char*)m.host + (size_t)n_rows * m.meta.row_bytes, 0xFF,
                       (size_t)(s->slots - n_rows) * m.meta.row_bytes);
            }
        }
        if (!s->dml) {
            // 多卡守卫：分配/拷贝作用于当前设备（会话设备）。dml 无 CUDA 面。
            if (g_cu.SetDevice) g_cu.SetDevice(s->dev_id);
            // 拷贝通道：异步模式=MemcpyAsync 上用户流（H2D→Run→D2H→事件同流序
            // ⇒收割语义与逐位结果均不变）；同步模式=阻塞 cudaMemcpy（原行为）。
            auto h2d = [&](void* dst, const void* src, size_t bytes) {
                if (s->async)
                    return g_cu.MemcpyAsync(dst, src, bytes, 1, s->stream) == 0;
                if (s->fence && s->h2d_async)
                    // fence 异步 H2D（FARM_H2D_ASYNC=1 opt-in）：与 replay/D2H
                    // 同流序保证正确性。**YGO fb64 实测回归 -17%**（PCIe 传输
                    // 与图回放在同流上串行化；旧同步 memcpy 的 PCIe DMA 与
                    // GPU 计算隐藏并行）——缺省=阻塞 memcpy 同步锚
                    return g_cu.MemcpyAsync(dst, src, bytes, 1, s->fence_stream) == 0;
                return g_cu.Memcpy(dst, src, bytes, 1) == nullptr;
            };
            // population 脏旗（演化路由，判决16）：代际换权重后的单次全量 H2D
            //（代间零拷贝——pop 非每槽输入，不参与前缀）
            if (s->pop_dirty) {
                for (size_t i = 0; i < s->ins.size(); i++)
                    if (s->ins[i].meta.population
                        && !h2d(s->ins[i].dev, s->ins[i].host,
                                s->ins[i].meta.row_bytes
                                    * (size_t)s->ins[i].meta.dims[0]))
                        return false;
                s->pop_dirty = false;
            }
            if (!s->any_append) {
            // 前缀 H2D：同步拷贝（返回即完成——与 ORT 内部流旗标无关，零竞态；
            // n>7/8·slots 走整块；population 面跳过）。无 append 面=原路径逐
            // 指令不变（any_append 门=零行为差保证，判决25）
            if (n_rows > (s->slots * 7) / 8) {
                if (!h2d(s->in_d_arena, s->in_h_arena, s->in_h_bytes))
                    return false;
                s->dep_h2d_bytes += s->in_h_bytes;
            } else if (s->fence && s->h2d_batch && s->ins.size() >= 2) {
                // P1-5 批拷贝判决实验（FARM_H2D_BATCH=1）：稀疏批多输入 H2D
                // 合并为一次 cudaMemcpyBatchAsync——dep=宿主提交税∝提交次数
                //（WDDM 下 5-10µs/次，多输入模型线性放大）。同流序不变 ⇒
                // 逐位等价；API 失败=整批判负（与单拷失败同纪律）。pop mid
                // 尾段毒化照旧单拷。
                size_t nb = 0;
                for (size_t i = 0; i < s->ins.size(); i++) {
                    if (s->ins[i].meta.population || s->ins[i].meta.weight) continue;
                    s->hb_dst[(size_t)nb] = s->ins[i].dev;
                    s->hb_src[(size_t)nb] = s->ins[i].host;
                    s->hb_sizes[(size_t)nb] = (size_t)n_rows * s->ins[i].meta.row_bytes;
                    s->hb_attridx[(size_t)nb] = 0;
                    nb++;
                }
                HbAttr attr;
                std::memset(&attr, 0, sizeof attr);
                attr.srcAccessOrder = 0x3;   // SrcAccessOrderAny：host 锚写稳、
                                             // 无在先流操作触碰 src
                // ⚠ 批拷贝必须与 ORT 同 cudart runtime：跨 runtime 调外部流
                // （本库 cudart12 调 ORT cudart13 的流）=段错误（2026-09-24
                // 实测；旧 MemcpyAsync 容忍、新批 API 不容忍）——生产契约
                // FARM_CUDART_DLL=cudart64_13.dll 已覆盖
                static const bool fdbg = [] {
                    const char* e = getenv("FARM_ORT_FENCE_DBG");
                    return e && *e && atoi(e) == 1;
                }();
                if (fdbg) {
                    std::fprintf(stderr, "[fence][dbg] batch nb=%zu dst0=%p "
                                 "src0=%p sz0=%zu\n", nb, s->hb_dst[0],
                                 s->hb_src[0], s->hb_sizes[0]);
                    std::fflush(stderr);
                }
                for (size_t i = 0; i < nb; i++) s->dep_h2d_bytes += s->hb_sizes[i];
                if (g_cu.MemcpyBatchAsyncV(s->hb_dst.data(), s->hb_src.data(),
                                          s->hb_sizes.data(), nb, &attr,
                                          s->hb_attridx.data(), 1,
                                          s->fence_stream))
                    return false;
                if (s->pop_mode && n_rows < s->slots)
                    for (int mi : s->mid_like) {
                        OrtIn& m = s->ins[(size_t)mi];
                        if (!h2d((char*)m.dev + (size_t)n_rows * m.meta.row_bytes,
                                 (char*)m.host + (size_t)n_rows * m.meta.row_bytes,
                                 (size_t)(s->slots - n_rows) * m.meta.row_bytes))
                            return false;
                    }
            } else {
                for (size_t i = 0; i < s->ins.size(); i++) {
                    if (s->ins[i].meta.population || s->ins[i].meta.weight) continue;
                    if (!h2d(s->ins[i].dev, s->ins[i].host,
                             (size_t)n_rows * s->ins[i].meta.row_bytes))
                        return false;
                    s->dep_h2d_bytes +=
                        (size_t)n_rows * s->ins[i].meta.row_bytes;
                }
                // 前缀路径补充：毒化后的路由键尾段同步到设备（整块路径全量拷贝已含）
                if (s->pop_mode && n_rows < s->slots)
                    for (int mi : s->mid_like) {
                        OrtIn& m = s->ins[(size_t)mi];
                        if (!h2d((char*)m.dev + (size_t)n_rows * m.meta.row_bytes,
                                 (char*)m.host + (size_t)n_rows * m.meta.row_bytes,
                                 (size_t)(s->slots - n_rows) * m.meta.row_bytes))
                            return false;
                    }
            }
            } else {
                // ---- 声明式增量 H2D（判决25）----
                // append 面逐行分流：增长只传 [synced,depth) 段（append-only
                // 承诺=前缀未变）；depth 递减=换局信号，影子重铸 [0,旧synced)
                // （新前段+零尾）一段式回传；未声明/满深行=整行兜底（永远正确，
                // 逐行混批）。full 面=前缀 [0,n)——append 面在批内，整块快捷会
                // 把增量面的陈旧行也重传（前功尽弃），故此模式不用整块判定。
                // 批拷贝通道可用=段聚合一次提交，否则逐段（先正确后优化）。
                const bool batch = s->fence && s->h2d_batch;
                size_t nb = 0;
                unsigned long long seg_bytes = 0;
                bool ok = true;
                auto emit = [&](void* dst, const void* src, size_t bytes) {
                    if (!ok || bytes == 0) return;
                    seg_bytes += bytes;
                    if (batch) {
                        s->hb_dst[nb] = dst;
                        s->hb_src[nb] = (void*)src;
                        s->hb_sizes[nb] = bytes;
                        s->hb_attridx[nb] = 0;
                        nb++;
                    } else if (!h2d(dst, src, bytes)) {
                        ok = false;
                    }
                };
                for (size_t i = 0; i < s->ins.size() && ok; i++) {
                    OrtIn& in = s->ins[i];
                    if (in.meta.population || in.meta.weight) continue;
                    if (!in.meta.append && !in.meta.headlive) {
                        emit(in.dev, in.host,
                             (size_t)n_rows * in.meta.row_bytes);
                        continue;
                    }
                    const size_t rb = in.meta.row_bytes;
                    const size_t st = (size_t)in.stride;
                    char* hb = (char*)in.host;
                    char* db = (char*)in.dev;
                    char* sb = (char*)in.shadow;
                    if (in.meta.headlive) {
                        // ---- headlive 面（判决25 扩展）：[0,depth) 本批新鲜
                        // 内容（可任意变化——逆序移位），[depth,slots) 宿主恒
                        // 零。每批全量传 [0,d)；缩深=换局信号，设备 [d,旧深)
                        // 补 memset（device-local 零 PCIe；与上传同流序，段
                        // 区不相交故与批聚合无序约束）；影子恒=[0,d)宿主+
                        // 零尾——哨兵 memcmp 全 face 照抓漂移（含宿主尾槽
                        // 非零=承诺违约）。
                        for (int r = 0; r < n_rows && ok; r++) {
                            const int d = in.declared[(size_t)r];
                            in.declared[(size_t)r] = -1;
                            const int so = in.synced[(size_t)r];
                            if (d < 0 || d >= in.max_depth) {
                                emit(db + (size_t)r * rb, hb + (size_t)r * rb, rb);
                                if (ok)
                                    std::memcpy(sb + (size_t)r * rb,
                                                hb + (size_t)r * rb, rb);
                                in.synced[(size_t)r] = OrtIn::kFullSync;
                                continue;
                            }
                            const size_t head = (size_t)d * st;
                            const int so_eff =
                                (so >= 0 && so <= in.max_depth) ? so : in.max_depth;
                            if (d < so_eff) {
                                char* dz = db + (size_t)r * rb + head;
                                size_t zb = (size_t)(so_eff - d) * st;
                                if (s->fence_stream) {
                                    if (g_cu.MemsetAsync(dz, 0, zb,
                                                         s->fence_stream))
                                        ok = false;
                                } else if (g_cu.Memset(dz, 0, zb)) {
                                    ok = false;
                                }
                            }
                            if (d > 0)
                                emit(db + (size_t)r * rb, hb + (size_t)r * rb,
                                     head);
                            if (ok) {
                                std::memcpy(sb + (size_t)r * rb, hb + (size_t)r * rb,
                                            head);
                                std::memset(sb + (size_t)r * rb + head, 0,
                                            rb - head);
                            }
                            in.synced[(size_t)r] = d;
                        }
                        continue;
                    }
                    for (int r = 0; r < n_rows && ok; r++) {
                        const int d = in.declared[(size_t)r];
                        in.declared[(size_t)r] = -1;   // 消费即复位（陈旧声明防线）
                        const int so = in.synced[(size_t)r];
                        if (d < 0 || d >= in.max_depth) {
                            // 未声明/满深：整行兜底
                            emit(db + (size_t)r * rb, hb + (size_t)r * rb, rb);
                            if (ok)
                                std::memcpy(sb + (size_t)r * rb,
                                            hb + (size_t)r * rb, rb);
                            in.synced[(size_t)r] = OrtIn::kFullSync;
                        } else if (d > so) {
                            // 增长：只传 [so,d)；影子同步仅此段（影子保持设备
                            // 镜像——哨兵 memcmp 才抓得到前缀违约）
                            const size_t off = (size_t)so * st;
                            const size_t bytes = (size_t)(d - so) * st;
                            emit(db + (size_t)r * rb + off,
                                 hb + (size_t)r * rb + off, bytes);
                            if (ok)
                                std::memcpy(sb + (size_t)r * rb + off,
                                            hb + (size_t)r * rb + off, bytes);
                            in.synced[(size_t)r] = d;
                        } else if (d < so) {
                            // 递减=换局：影子重铸本行 [0,有效旧深)（新前段+零尾）
                            // 后一段式回传。旧深以行界钳制——kFullSync（整行兜底
                            // 同步过，含虚增量未组装槽）或越界都按"整行"处理，
                            // 零尾到行尾为止（宿主侧该行=Claim 清零+[0,d) 新内容，
                            // 设备侧必须同样恢复零基）。无钳制=tail≈4GB 越界
                            // memset（虚增量+整行兜底组合实测段错误案）。
                            const int so_eff =
                                (so >= 0 && so <= in.max_depth) ? so : in.max_depth;
                            char* sh = sb + (size_t)r * rb;
                            const char* hh = hb + (size_t)r * rb;
                            const size_t head = (size_t)d * st;
                            const size_t tail = (size_t)(so_eff - d) * st;
                            std::memcpy(sh, hh, head);
                            std::memset(sh + head, 0, tail);
                            emit(db + (size_t)r * rb, sh, head + tail);
                            in.synced[(size_t)r] = d;
                        }
                        // d == so：零段（承诺=前缀未变、行尾已零）
                    }
                }
                // population 死行毒化尾段（判决16；宿主侧毒化已在批头完成）
                if (ok && s->pop_mode && n_rows < s->slots)
                    for (int mi : s->mid_like) {
                        OrtIn& m = s->ins[(size_t)mi];
                        emit((char*)m.dev + (size_t)n_rows * m.meta.row_bytes,
                             (char*)m.host + (size_t)n_rows * m.meta.row_bytes,
                             (size_t)(s->slots - n_rows) * m.meta.row_bytes);
                    }
                if (ok && batch && nb > 0) {
                    HbAttr attr;
                    std::memset(&attr, 0, sizeof attr);
                    attr.srcAccessOrder = 0x3;   // SrcAccessOrderAny（同 P1-5）
                    if (g_cu.MemcpyBatchAsyncV(s->hb_dst.data(),
                                              s->hb_src.data(),
                                              s->hb_sizes.data(), nb, &attr,
                                              s->hb_attridx.data(), 1,
                                              s->fence_stream))
                        ok = false;
                }
                if (!ok) return false;
                s->dep_h2d_bytes += seg_bytes;
                // debug 哨兵（FARM_H2D_DELTA_DEBUG=1）：影子=设备忠实镜像——
                // 全 face memcmp 宿主，非 0=设备侧与宿主漂移（声明漏传/
                // append-only 承诺违约/段规划 bug）。首个错打印；计数=门断言面
                //（OrtDeltaDebugViolations，防"打印没人看"的空过）。
                if (s->delta_dbg)
                    for (size_t i = 0; i < s->ins.size(); i++) {
                        OrtIn& in = s->ins[i];
                        if (!(in.meta.append || in.meta.headlive)) continue;
                        if (std::memcmp(in.shadow, in.host,
                                        in.meta.row_bytes
                                            * (size_t)s->slots) != 0) {
                            g_delta_violations.fetch_add(
                                1, std::memory_order_relaxed);
                            if (!s->delta_vio_printed) {
                                s->delta_vio_printed = true;
                                std::fprintf(stderr,
                                             "[ort] H2D 增量哨兵：face \"%s\" "
                                             "影子≠宿主（声明漏传或 append-only "
                                             "承诺违约——核对 FaceDepth 声明与"
                                             "行内容）\n",
                                             in.meta.name.c_str());
                                std::fflush(stderr);
                            }
                        }
                    }
            }
            const long long dep_t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (!RunOnce(s, s->ro)) return false;
            const long long dep_t2 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (s->fence) {
                // fence 桥接 v3：replay/eager 内核已提交到 ORT EP 统一流——D2H
                // 前缀异步挂同流，再排 LaunchHostFunc：流到达 D2H 之后时由
                // CUDA 回调线程置完成旗标+ReleaseSemaphore（调度台 WMO 直等
                // 信号量；零轮询零量子）。图外常规异步，零捕获语义参与。
                s->flight_done.store(false, std::memory_order_relaxed);
                for (size_t j = 0; j < s->outs.size(); j++)
                    if (g_cu.MemcpyAsync(s->outs[j].host, s->outs[j].dev,
                                         (size_t)s->last_n
                                             * (size_t)s->outs[j].meta.width * 4,
                                         2, s->fence_stream))
                        return false;
                if (g_cu.LaunchHostFunc(s->fence_stream, &FenceReleaseCb,
                                        &s->fence_cb_ctx))
                    return false;
            }
            // P1 dep 三段分解（H2D/Run/D2H+盖章；接入方 1.4ms 黑盒定位仪器）
            {
                long long t3 = dep_t2;
                if (s->fence) t3 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                s->dep_h2d_ns += dep_t1 - dep_t0;
                s->dep_run_ns += dep_t2 - dep_t1;
                s->dep_d2h_ns += t3 - dep_t2;
                if (++s->dep_cnt == 256) {
                    std::fprintf(stderr, "[ort] dep 三段 (n=%u, rows=%d): h2d=%.3f "
                                 "run=%.3f d2h=%.3f ms/均 h2d=%.1fkB/批\n", s->dep_cnt, s->last_n,
                                 s->dep_h2d_ns / 256e6, s->dep_run_ns / 256e6,
                                 s->dep_d2h_ns / 256e6,
                                 s->dep_h2d_bytes / 256.0 / 1024.0);
                    std::fflush(stderr);
                    s->dep_h2d_ns = s->dep_run_ns = s->dep_d2h_ns = 0;
                    s->dep_h2d_bytes = 0;
                    s->dep_cnt = 0;
                }
            }
            if (s->async) {
                // 前缀 D2H 同流入队（序=Run 后）+ 事件盖戳——事件完成=输出已
                // 驻留 host arena（pinned，标准可见性语义）。收割=EventQuery。
                for (size_t j = 0; j < s->outs.size(); j++)
                    if (g_cu.MemcpyAsync(s->outs[j].host, s->outs[j].dev,
                                         (size_t)s->last_n
                                             * (size_t)s->outs[j].meta.width * 4,
                                         2, s->stream))
                        return false;
                if (g_cu.EventRecord(s->event, s->stream)) return false;
            }
        } else {
            // DML：投递专属发射线程（同步 Run 不占调度台）；行数据在 host
            // arena，投递前的写在锁释放后对发射线程可见。
            {
                std::lock_guard<std::mutex> lk(s->h_mx);
                s->h_pending = s->seq;
            }
            s->h_cv.notify_one();
        }
        seq_out = s->seq;
        return true;   // cuda：Run 返回=已入队（收割侧 sync）；dml：已投递
    }

    bool CompletionReached(void* session, unsigned seq) override {
        OrtSess* s = (OrtSess*)session;
        if (s->dml) return s->h_done.load(std::memory_order_acquire) >= seq;
        if (seq < s->seq) return true;         // 旧序号（早已完成）
        if (s->synced_for_seq) return true;
        if (s->fence) {
            // fence 桥接 v3：done 旗标由 CUDA 回调线程置位（acquire 读见
            // release 写 ⇒ 前缀输出已驻留 host arena）。WMO 唤醒消费信号量、
            // 这里只读旗标——消费与状态分离，无计数竞态。
            if (s->flight_done.load(std::memory_order_acquire)) {
                s->synced_for_seq = true;
                return true;
            }
            return false;
        }
        if (s->async) {
            // 零围栏：事件在 D2H 之后入流——查询完成=前缀输出已驻留 host
            if (g_cu.EventQuery && g_cu.EventQuery(s->event) == 0) {
                s->synced_for_seq = true;
                return true;
            }
            return false;
        }
        // 整设备同步血律（不赌 ORT 内部流序）+ 前缀 D2H
        if (g_cu.SetDevice) g_cu.SetDevice(s->dev_id);
        g_cu.DeviceSynchronize();
        int n = s->last_n;
        for (size_t j = 0; j < s->outs.size(); j++)
            if (g_cu.Memcpy(s->outs[j].host, s->outs[j].dev,
                            (size_t)n * (size_t)s->outs[j].meta.width * 4, 2))
                return false;
        s->synced_for_seq = true;
        return true;
    }
    void CompletionFence() override {
        // 同步路径=无操作（CompletionReached 已整设备同步）；异步路径的收割
        // 侧调用发生在 EventQuery 成功之后（事件完成=流上 D2H 全部落定）——
        // 两路均无需额外等待
    }

    const float* OutputRow(void* session, const char* name, int slot) override {
        OrtSess* s = (OrtSess*)session;
        for (auto& o : s->outs)
            if (o.meta.name == name)
                return o.host + (size_t)slot * (size_t)o.meta.width;
        return nullptr;
    }
    int OutputWidth(void* session, const char* name) override {
        OrtSess* s = (OrtSess*)session;
        for (auto& o : s->outs)
            if (o.meta.name == name) return o.meta.width;
        return -1;
    }

    // ---- RefitWeights（升格权重热换，2026-09-29；接口语义与 TRT/CPU 同门：
    //      RW1 blob→全会话换心，跨会话共享=实例级）。前置条件=腿已返回
    //      （Farm 契约；与 SetPopulation 同纪律）。----
    // 相一 validate（全条目×全会话先验，零撕裂态）→ 相二 commit（host 覆写+
    // 阻塞 H2D——腿间流空闲，阻塞 memcpy=生产同款通道）。无会话=仅入 stash
    // （init 期换心先于建行：farm.cpp cfg.refit_weights 在建行前调——种子由
    // 后续会话 Warmup 播种；名字契约失配在 Warmup/发车侧 fail fast）。
    bool RefitWeights(const char* rw1_path) override {
        const double t0 = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        std::vector<char> blob;
        std::vector<Rw1Entry> ents;
        if (!ParseRw1(rw1_path, blob, ents)) return false;
        std::lock_guard<std::mutex> lk(refit_mx_);
        for (OrtSess* s : sessions_)
            if (!ValidateOrCommit(s, ents, false)) return false;
        int n_set = 0;
        for (OrtSess* s : sessions_)
            if (!ValidateOrCommit(s, ents, true, &n_set)) return false;
        // stash 换代（成功才动；未来会话播种源）。data 指针重基到 stash 块。
        std::vector<size_t> offs(ents.size());
        for (size_t k = 0; k < ents.size(); k++)
            offs[k] = (size_t)(ents[k].data - blob.data());
        stash_blob_ = std::move(blob);
        stash_ents_ = ents;
        for (size_t k = 0; k < stash_ents_.size(); k++)
            stash_ents_[k].data = stash_blob_.data() + offs[k];
        const double t1 = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        std::fprintf(stderr, "[ort] refit 换心 %d 项×%zu 会话耗时 %.1fms：%s\n",
                     n_set, sessions_.size(), t1 - t0, rw1_path);
        return true;
    }

private:
    // stash→会话播种（Warmup memset 后调用；幂等）。返回 true=全部权重面已获
    // 真实种子。stash 空/名字不命中=留零+false（发车侧 fail fast 兜底）。
    bool SeedWeights(OrtSess* s) {
        if (!s->any_weight) return true;
        std::lock_guard<std::mutex> lk(refit_mx_);
        bool all = true;
        for (auto& i : s->ins) {
            if (!i.meta.weight) continue;
            const Rw1Entry* e = nullptr;
            for (const auto& c : stash_ents_)
                if (c.name == i.meta.name) { e = &c; break; }
            const size_t bytes = i.meta.row_bytes * (size_t)i.meta.dims[0];
            if (!e || e->bytes != bytes) { all = false; continue; }
            std::memcpy(i.host, e->data, bytes);
            if (!s->dml) {
                if (g_cu.SetDevice) g_cu.SetDevice(s->dev_id);
                if (g_cu.Memcpy(i.dev, i.host, bytes, 1)) return false;
            }
        }
        s->weights_seeded = all;
        return all;
    }

    // 单会话 RW1 校验/提交（RefitWeights 两相共用；前置=持 refit_mx_）。
    // commit=false：只验——模型无权重面=拒（RW1 无落点）；有面但一项不中=空转
    // 拒载（e271290/DATA11 教训）；dtype/numel 不符=fail fast（TRT 原型校验同门）。
    // commit=true：host 覆写+阻塞 H2D（DML=纯宿主）。未播种会话要求全覆盖
    //（部分换心后余下面仍是零=垃圾）。
    bool ValidateOrCommit(OrtSess* s, const std::vector<Rw1Entry>& ents,
                          bool commit, int* n_set = nullptr) {
        int set = 0, faces = 0, matched = 0;
        for (auto& i : s->ins) {
            if (!i.meta.weight) continue;
            faces++;
            const Rw1Entry* hit = nullptr;
            for (const auto& e : ents)
                if (e.name == i.meta.name) { hit = &e; break; }
            if (!hit) continue;
            matched++;
            const size_t bytes = i.meta.row_bytes * (size_t)i.meta.dims[0];
            if ((size_t)Rw1DtypeSize(hit->dtype) != i.meta.esize
                || hit->bytes != bytes) {
                std::fprintf(stderr, "[ort] refit %s dtype/numel 与模型面不符"
                             "（blob %zuB/dtype%u vs 面 %zuB/esize%zu）——fail fast\n",
                             i.meta.name.c_str(), hit->bytes,
                             (unsigned)Rw1DtypeSize(hit->dtype), bytes, i.meta.esize);
                return false;
            }
            if (commit) {
                std::memcpy(i.host, hit->data, bytes);
                if (!s->dml) {
                    if (g_cu.SetDevice) g_cu.SetDevice(s->dev_id);
                    if (g_cu.Memcpy(i.dev, i.host, bytes, 1)) {
                        std::fprintf(stderr, "[ort] refit %s H2D 覆写失败\n",
                                     i.meta.name.c_str());
                        return false;
                    }
                }
                set++;
            }
        }
        if (!faces) {
            std::fprintf(stderr, "[ort] refit 拒载：模型无升格权重面"
                         "（RW1 %zu 项无落点）\n", ents.size());
            return false;
        }
        if (!matched) {
            std::fprintf(stderr, "[ort] refit 空转拒载：RW1 %zu 项与本会话 %d 个"
                         "权重面名字全不匹配\n", ents.size(), faces);
            return false;
        }
        if (commit) {
            if (!s->weights_seeded && set < faces) {
                std::fprintf(stderr, "[ort] refit 拒载：未播种会话仅覆盖 %d/%d 个"
                             "权重面（余下面为零）——首次换心须全量 RW1"
                             "（升格工具边车即全量）\n", set, faces);
                return false;
            }
            s->weights_seeded = true;
            if (n_set) *n_set += set;
        }
        return true;
    }

public:

    // population 面写入（演化路由）：宿主 arena 落盘 + cuda 置脏旗（下次批全量
    // H2D 一次）；dml=宿主绑定直读，写完即生效（无拷贝）
    bool SetPopulation(void* session, const char* pop_input, const void* host) override {
        OrtSess* s = (OrtSess*)session;
        for (auto& i : s->ins)
            if (i.meta.population && i.meta.name == pop_input) {
                memcpy(i.host, host, i.meta.row_bytes * (size_t)i.meta.dims[0]);
                s->pop_dirty = true;
                return true;
            }
        return false;
    }

    // 声明式增量 H2D（判决25）：组装期行深度申报。仅 append 面接收（深度单位
    // =行首维 dims[1]）；界外深度=拒绝（该行按未声明整行兜底——fail-safe）。
    // 写手 fiber 调用（Claim→Submit 窗口同槽独占=无并发）；SubmitBatch 消费后
    // 复位。无 append 面的会话=any_append 门一击即回（零占用）。
    void NoteFaceDepth(void* session, const char* name, int slot, int depth) override {
        OrtSess* s = (OrtSess*)session;
        if (!s || !s->any_append || !name || slot < 0 || slot >= s->slots) return;
        for (auto& i : s->ins)
            if ((i.meta.append || i.meta.headlive) && i.meta.name == name) {
                if (depth >= 0 && depth <= i.max_depth)
                    i.declared[(size_t)slot] = depth;
                return;
            }
    }
    // 声明会话起点复位（Claim 领槽调用）：上任写手的陈旧声明作废——否则新
    // 组装未重新声明的行会被陈旧深度误读=按错段传输。
    void ClearFaceDepths(void* session, int slot) override {
        OrtSess* s = (OrtSess*)session;
        if (!s || !s->any_append || slot < 0 || slot >= s->slots) return;
        for (auto& i : s->ins)
            if (i.meta.append || i.meta.headlive) i.declared[(size_t)slot] = -1;
    }

private:
    ortplt::HMODULE dll_ = nullptr;   // 持引用（进程寿命不卸；POSIX 不 dlclose）
    const OrtApi* api_ = nullptr;
    std::string ver_ = "?";
    std::string dll_dir_;
    int env_seq_ = 0;
    bool dml_ = false;

    // 实例级 DLL 加载（注册表：同(基名,目录)=复用；基名冲突他目录=改名装载）
    bool LoadLib(const ModelConfig& cfg) {
        if (api_) return true;
        // 目录解析链：cfg → env → 空（走系统 DLL 搜索，不写死开发机路径）。
        // 两者皆空时打印排查入口提示——loader 报错难读，提前一句话省一次迷路。
        const char* ed = getenv("FARM_ORT_DIR");
        std::string ort_dir = !cfg.ort_dir.empty() ? cfg.ort_dir
            : (ed && *ed ? ed : "");
        const char* cd = getenv("FARM_CUDA_DIR");
        std::string cuda_dir = !cfg.cuda_dir.empty() ? cfg.cuda_dir
            : (cd && *cd ? cd : "");
        if (ort_dir.empty())
            std::fprintf(stderr, "[ort] 未设 ModelConfig.ort_dir / env FARM_ORT_DIR"
                         "——ORT 动态库走系统库搜索（加载失败先查这里）\n");
        std::lock_guard<std::mutex> lk(g_ort_dll_mx);
        // 库基名（两平台各自的原生命名；注册表键=基名+目录）
#ifdef _WIN32
        std::string base = ToLower("onnxruntime.dll");
        // PATH 前插（每目录一次；onnxruntime 的 cudart/cublas/DML 依赖解析）
        {
            char old_path[8192];
            GetEnvironmentVariableA("PATH", old_path, sizeof old_path);
            std::string np = old_path;
            if (!ort_dir.empty()) np = ort_dir + ";" + np;
            if (!dml_ && !cuda_dir.empty()
                && np.find(cuda_dir) == std::string::npos)
                np = cuda_dir + ";" + np;
            SetEnvironmentVariableA("PATH", np.c_str());
        }
        ortplt::HMODULE h = nullptr;
        std::string load_path;
        for (auto& r : g_ort_dlls)
            if (r.base == base) {
                if (ToLower(r.dir) == ToLower(ort_dir)) { h = r.h; break; }
                // 基名冲突（Windows 按基名去重回柄）：拷贝改名再装。仅限有
                // 目录语义的实例——空目录=系统搜索装载，无"他目录"可冲突
                if (ort_dir.empty()) break;
                char tmp[MAX_PATH];
                GetTempPathA(MAX_PATH, tmp);
                load_path = std::string(tmp) + "inferfarm_ort_"
                    + std::to_string(g_ort_dll_seq.fetch_add(1)) + ".dll";
                if (!CopyFileA((ort_dir + "\\onnxruntime.dll").c_str(),
                               load_path.c_str(), FALSE)) {
                    std::fprintf(stderr, "[ort] 基名冲突改名拷贝失败 GLE=%lu（%s → %s）\n",
                                 ortplt::LastErr(), ort_dir.c_str(), load_path.c_str());
                    return false;
                }
                std::fprintf(stderr, "[ort] 双 ORT 共存：另一 onnxruntime.dll 已驻留，"
                             "本实例改载改名副本 %s（依赖走 PATH）\n", load_path.c_str());
                break;
            }
        if (!h) {
            // 目录空=裸名加载（Windows 标准搜索：应用目录→系统32→PATH——
            // 语义与 cudart_dyn 空 dir 一致）。此前空目录曾拼出 "\onnxruntime.dll"
            // 根路径必败（对外反馈 2026-09-24）。裸名搜索顺序含应用目录：exe
            // 同目录若有 CPU 版同名 dll 会先被命中——标准搜索的既定行为，
            // 部署上避免在 exe 旁放同名 dll。
            if (load_path.empty())
                load_path = ort_dir.empty() ? "onnxruntime.dll"
                                            : ort_dir + "\\onnxruntime.dll";
            h = ortplt::LibLoad(load_path.c_str());
        }
        if (!h) {
            std::fprintf(stderr, "[ort] LoadLibrary %s 失败 GLE=%lu\n",
                         load_path.c_str(), ortplt::LastErr());
            return false;
        }
#else
        // POSIX 降级点：无 PATH 前插/基名改名拷贝（Windows 专属病：System32
        // 先于 PATH + 按基名驻留去重；POSIX 依赖解析走 ld.so 搜索路径/rpath，
        // FARM_ORT_DIR 即 dlopen 全路径）。CUDA 依赖（libtorch 的 libcudart）
        // 需已在 ld 搜索路径（pip 包 lib 目录不自动进——CI/部署注意）。
        std::string base = ToLower("libonnxruntime.so");
        ortplt::HMODULE h = nullptr;
        std::string load_path = ort_dir.empty() ? std::string("libonnxruntime.so")
                                                : ort_dir + "/libonnxruntime.so";
        for (auto& r : g_ort_dlls)
            if (r.base == base && ToLower(r.dir) == ToLower(ort_dir)) {
                h = r.h;
                break;
            }
        if (!h) h = ortplt::LibLoad(load_path.c_str());
        if (!h) {
            std::fprintf(stderr, "[ort] dlopen %s 失败：%s（显式设 FARM_ORT_DIR"
                         " 指向 libonnxruntime.so 所在目录）\n",
                         load_path.c_str(), ::dlerror());
            return false;
        }
#endif
        auto fn = (const OrtApiBase*(ORT_API_CALL*)())ortplt::LibSym(h, "OrtGetApiBase");
        if (!fn) {
            std::fprintf(stderr, "[ort] DLL 无 OrtGetApiBase\n");
            return false;
        }
        const OrtApiBase* ab = fn();
        if (ab->GetVersionString) {
            const char* vs = ab->GetVersionString();
            if (vs && *vs) ver_ = vs;
        }
        api_ = ab->GetApi(ORT_API_VERSION);
        if (!api_) {
            std::fprintf(stderr, "[ort] GetApi(%d) 失败（头/DLL 版本不匹配）——系统搜索"
                         "命中的可能是旧版 ORT（System32 先于 PATH）：显式设 "
                         "FARM_ORT_DIR 指向你的 ORT 目录\n", ORT_API_VERSION);
            return false;
        }
        dll_ = h;
        dll_dir_ = ort_dir;
        bool known = false;
        for (auto& r : g_ort_dlls)
            if (r.h == h) known = true;
        if (!known) g_ort_dlls.push_back({base, ort_dir, h});
        std::fprintf(stderr, "[ort] %s 就绪%s（%s）\n", ver_.c_str(),
                     dml_ ? "（DML 路）" : "", load_path.c_str());
        return true;
    }

    static bool RunOnce(OrtSess* s, OrtRunOptions* ro = nullptr) {
        const OrtApi* a = s->api;
        if (s->dml) {
            // DML 血律（实测 2026-09-22）：iob 预绑的 CPU 输入被 EP 忽略（读到
            // 恒零设备缓冲——probe 两图案不可分辨即此症），输出预绑却正常回传。
            // 判决：每次 Run 前用**新鲜 CPU OrtValue** 重绑输入（python numpy
            // 路同款），EP 据值对象走 staging 拷入；输出维持预绑。
            // dep 拆段（2026-09-26，可观测面）：DML 的 flw 是黑盒墙钟，拆
            // rebind（CPU API 面）/run（RunWithBinding=EP 内部 staging memcpy
            // +dispatch+GPU 同步的总和）两段——"算力瓶颈 vs 提交瓶颈"的判决
            // 仪器（发射线程独写独读，Warmup 首次混入 1/256 可忽略）。
            const long long rb0 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            for (size_t i = 0; i < s->ins.size(); i++) {
                if (s->ins[i].val) a->ReleaseValue(s->ins[i].val);
                size_t bytes = s->ins[i].meta.population
                    ? s->ins[i].meta.row_bytes * (size_t)s->ins[i].meta.dims[0]
                    : s->ins[i].meta.row_bytes * (size_t)s->slots;   // DML 重绑同口径
                OrtValue* v = nullptr;
                if (a->CreateTensorWithDataAsOrtValue(
                        s->bind_mem, s->ins[i].host, bytes,
                        s->ins[i].meta.dims.data(), s->ins[i].meta.dims.size(),
                        ElemToOnnx(s->ins[i].meta.et), &v)) {
                    s->ins[i].val = nullptr;
                    return false;
                }
                s->ins[i].val = v;
                if (a->BindInput(s->iob, s->ins[i].meta.name.c_str(), v)) return false;
            }
            const long long rb1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            s->dml_rebind_ns += rb1 - rb0;
            OrtStatus* st = a->RunWithBinding(s->sess, ro, s->iob);
            const long long rb2 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            s->dml_run_ns += rb2 - rb1;
            // 阈值 64：大批形状（fb64×8 会话）总批数少，256 会整腿打不出
            if (++s->dml_dep_cnt == 64) {
                std::fprintf(stderr, "[ort-dml] dep (rows=%d): rebind=%.3f "
                             "run=%.3f ms/均\n", s->last_n,
                             s->dml_rebind_ns / 64e6, s->dml_run_ns / 64e6);
                std::fflush(stderr);
                s->dml_rebind_ns = s->dml_run_ns = 0;
                s->dml_dep_cnt = 0;
            }
            if (st) {
                std::printf("[ort] 批异常（RunWithBinding）：%s\n", a->GetErrorMessage(st));
                std::fflush(stdout);
                a->ReleaseStatus(st);
                return false;
            }
            return true;
        }
        return RunWithBindingCommon(s, ro);
    }

    static bool RunWithBindingCommon(OrtSess* s, OrtRunOptions* ro) {
        const OrtApi* a = s->api;
        OrtStatus* st = a->RunWithBinding(s->sess, ro, s->iob);
        if (st) {
            std::printf("[ort] 批异常（RunWithBinding）：%s\n", a->GetErrorMessage(st));
            std::fflush(stdout);
            a->ReleaseStatus(st);
            return false;
        }
        return true;
    }

    static void FillPattern(OrtSess* s, int seed) {
        for (size_t i = 0; i < s->ins.size(); i++) {
            OrtIn& oi = s->ins[i];
            // 升格权重面不进图案实验：ProbeGraph 的 run_pat 会整 arena H2D，
            // 垃圾图案上设备=后续真腿用垃圾权重（面恢复无门）
            if (oi.meta.weight) continue;
            // population 面=总量按 dim0（P≠slots；同族坑第二处——按 slots 会写爆堆）
            size_t n = (oi.meta.population
                            ? oi.meta.row_bytes * (size_t)oi.meta.dims[0]
                            : oi.meta.row_bytes * (size_t)s->slots)
                       / oi.meta.esize;
            if (oi.meta.et == DTYPE_F32) {
                float* p = (float*)oi.host;
                for (size_t e = 0; e < n; e++)
                    p[e] = (float)((int)((e * 31 + (size_t)seed * 997) % 2039) - 1019) / 1019.0f;
            } else if (oi.meta.et == DTYPE_I64) {
                long long* p = (long long*)oi.host;
                for (size_t e = 0; e < n; e++)
                    p[e] = (long long)((e * 7 + (size_t)seed * 13) % 14969);
            } else if (oi.meta.et == DTYPE_I32) {
                int* p = (int*)oi.host;
                for (size_t e = 0; e < n; e++) p[e] = (int)(e % 7);
            } else {
                unsigned char* p = (unsigned char*)oi.host;
                for (size_t e = 0; e < n; e++) p[e] = (unsigned char)((e + (size_t)seed) & 1);
            }
        }
    }

    // spec_out 非空=顺带枚举规格（LoadSpec 探测/探测砍除通道共用）：跑完即毁
    // 或常驻。**懒加载守卫必须置顶**：探测砍除通道下 LoadSpec 不再被调用，
    // 本函数就是 DLL/cudart 的首个触碰点（api_/g_cu 全 null 的段错误案
    // 2026-09-24）——LoadLib/g_cu.Load 均幂等，重复调用零成本。
    OrtSess* CreateSession(const ModelConfig& cfg, int slots, bool for_bank,
                              ModelSpec* spec_out) {
        dml_ = (cfg.ort_ep == "dml");
        // 指针返回函数：bool 字面量转空指针在严格编译器下报错（2d0d709
        // 862/864 接入方报告案）——一律 nullptr
        if (!LoadLib(cfg)) return nullptr;
        if (!dml_) {
            if (!g_cu.Load(cfg.cuda_dir)) return nullptr;
            SetSpinFlagsOnce();
        }
        const OrtApi* a = api_;
        if (cfg.model_path.empty()) {
            std::fprintf(stderr, "[ort] 缺 model_path（fb 烤死的 onnx）\n");
            return nullptr;
        }
        bool dml = dml_;
        OrtSess* s = new OrtSess();
        s->slots = slots;
        s->dll = dll_;
        s->api = api_;
        s->dml = dml;
        s->dev_id = cfg.device_id;
        // 每会话独立 env（会话/图/arena 全套自闭环，互不沾染共享态）；
        // FARM_ORT_SHARED_ENV=1=进程单 env（零围栏调试面：python 全局 env
        // 路径与 per-session env 的行为差异排查用）
        static OrtEnv* g_shared_env = nullptr;
        char env_name[32];
        std::snprintf(env_name, sizeof env_name, "inferfarm_%d", env_seq_++);
        if (SharedEnv() && g_shared_env) {
            s->env = g_shared_env;
        } else if (a->CreateEnv(ORT_LOGGING_LEVEL_ERROR, env_name, &s->env)) {
            DestroySession(s); return nullptr;
        }
        if (SharedEnv() && !g_shared_env) g_shared_env = s->env;
        OrtSessionOptions* opts = nullptr;
        if (a->CreateSessionOptions(&opts)) { DestroySession(s); return nullptr; }
        IgnoreStatus(a, a->SetIntraOpNumThreads(opts, cfg.ort_threads > 0 ? cfg.ort_threads : 1));
        IgnoreStatus(a, a->SetInterOpNumThreads(opts, 1));
        IgnoreStatus(a, a->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));
        if (dml) {
#ifdef _WIN32
            // ---- DML EP（AMD/核显路线）：挂 DirectML，device_id=适配器序号 ----
            OrtDmlAppendFn dml_append = (OrtDmlAppendFn)GetProcAddress(
                dll_, "OrtSessionOptionsAppendExecutionProvider_DML");
            if (!dml_append) {
                std::fprintf(stderr, "[ort] 本 DLL 无 DML 导出（须 onnxruntime-directml"
                             " 构建； ort_ep=dml 与 CUDA 构建互斥）\n");
                a->ReleaseSessionOptions(opts);
                DestroySession(s);
                return nullptr;
            }
            char did[16];
            std::snprintf(did, sizeof did, "%d", cfg.device_id);
            OrtStatus* st = dml_append(opts, cfg.device_id);
            if (st) {
                std::fprintf(stderr, "[ort] 挂 DML EP 失败（dev=%d）: %s\n",
                             cfg.device_id, a->GetErrorMessage(st));
                a->ReleaseStatus(st);
                a->ReleaseSessionOptions(opts);
                DestroySession(s);
                return nullptr;
            }
            s->graph_on = false;
#else
            // POSIX 降级点：DML EP=DirectML=Windows 专属（dlopen 面无此导出），
            // 配置级 fail fast 指路，不静默降级到 CPU（后端选型是调用方的决定）。
            std::fprintf(stderr, "[ort] ort_ep=dml 仅 Windows 构建可用（DirectML "
                         "是 Windows 专属 EP）；本面请选 cpu/cuda 路线\n");
            a->ReleaseSessionOptions(opts);
            DestroySession(s);
            return nullptr;
#endif
        } else {
            // ---- CUDA EP（+CUDA Graph——KV 串与 python providers=
            // {"enable_cuda_graph":"1"} 同义）。图仅银行会话开（PerThreadContext
            // 铁律：创建/回放同线程——银行会话全生命周期在调度台线程上）。----
            // 零围栏：1=判死打印不启用；2=翻案实验通道（NonBlocking 用户流 +
            // provider option + RunOptions 关 EP 同步 + 事件收割，探测会话除外）
            // 门按 for_bank 判（探测会话 for_bank=false 自然排除）——探测砍除
            // 通道下首个真实银行会话带 spec_out，照样吃 fence。
            if (for_bank && AsyncEnvMode() == 2) {
                if (g_cu.SetDevice) g_cu.SetDevice(s->dev_id);
                if (!g_cu.StreamCreateWithFlags
                    || g_cu.StreamCreateWithFlags(&s->stream, 0x01) != 0) {
                    std::fprintf(stderr, "[ort] FARM_ORT_ASYNC=2：StreamCreate 失败"
                                 "——本会话回落同步路径\n");
                    s->stream = nullptr;
                } else {
                    s->async = true;
                    std::fprintf(stderr, "[ort] FARM_ORT_ASYNC=2：启用（cudart=%s"
                                 "）\n", getenv("FARM_CUDART_DLL")
                                 && *getenv("FARM_CUDART_DLL")
                                 ? getenv("FARM_CUDART_DLL") : "cudart64_12.dll");
                }
            } else if (for_bank && AsyncEnvMode() == 3
                       && cfg.ort_cuda_graph
                       && g_cu.EventCreateWithFlags && g_cu.EventRecord
                       && g_cu.EventQuery) {
                // fence 桥接：不建用户流、不传 user_compute_stream——ORT 用它
                // 自己的 EP 统一流（enable_cuda_graph 才启用；eager=多流无探针
                // 可依，inline 实测漂移=回落同步），fence 探针在图尾暴露该流。
                // 事件族符号缺席=回落同步。
                if (g_cu.SetDevice) g_cu.SetDevice(s->dev_id);
                s->fence = true;
                s->fence_sem = ortplt::FenceSemCreate();
                if (s->fence_sem) {
                    s->fence_cb_ctx.sem = s->fence_sem;
                    s->fence_cb_ctx.done = &s->flight_done;
                } else {
                    s->fence = false;
                }
                // H2D 异步入流（FARM_H2D_ASYNC=1）：**YGO fb64 实测回归 -17%**
                // （PCIe 与图回放同流串行化；旧同步 memcpy 的 PCIe DMA 与 GPU
                // 计算隐藏并行）——缺省阻塞 memcpy=同步锚正解，开关留档复测
                s->h2d_async = s->fence
                    && [] {
                           const char* e = getenv("FARM_H2D_ASYNC");
                           return e && *e && atoi(e) == 1;
                       }();
                s->h2d_batch = s->fence && s->h2d_async
                    && [] {
                           const char* e = getenv("FARM_H2D_BATCH");
                           return e && *e && atoi(e) == 1;
                       }()
                    && g_cu.MemcpyBatchOk();   // 符号缺席=回退逐输入
                std::fprintf(stderr, "[ort] FARM_ORT_ASYNC=3：fence 桥接启用"
                             "（H2D 同步锚+信号量完成通知）%s\n",
                             s->h2d_async ? "+H2D 异步（FARM_H2D_ASYNC=1，YGO 实测"
                                           "回归档）"
                                          : "");
            } else if (for_bank && AsyncEnvMode() == 1) {
                std::fprintf(stderr, "[ort] FARM_ORT_ASYNC：实测判死（图=捕获不落"
                             "用户流 900；eager=逐位不确定 3 跑 3 指纹）——本会话"
                             "走原同步路径（判决12 全账；=2/=3 走翻案通道）\n");
            }
            // fence 域恒注册（非 DML 会话）：模型含 InferfarmFence 节点时探测
            // 会话（LoadSpec）也须能解析；无 fence 节点的模型注册域零副作用
            {
                OrtCustomOpDomain* dom = nullptr;
                if (!a->CreateCustomOpDomain("inferfarm", &dom)
                    && !a->CustomOpDomain_Add(dom, &g_fence_op)
                    && !a->AddCustomOpDomain(opts, dom)) {
                    s->domain = dom;
                } else {
                    if (dom) a->ReleaseCustomOpDomain(dom);
                    std::fprintf(stderr, "[ort] fence 域注册失败（含 InferfarmFence"
                                 " 节点的模型将拒载）\n");
                }
            }
            OrtCUDAProviderOptionsV2* co = nullptr;
            bool graph = for_bank && cfg.ort_cuda_graph;
            if (a->CreateCUDAProviderOptions(&co)) { a->ReleaseSessionOptions(opts); DestroySession(s); return nullptr; }
            char did[16];
            std::snprintf(did, sizeof did, "%d", cfg.device_id);
            char sptr[32];
            const char* keys[] = {"device_id", "enable_cuda_graph", "user_compute_stream"};
            const char* vals[3] = {did, graph ? "1" : "0", nullptr};
            int nk = 2;
            if (s->async) {
                std::snprintf(sptr, sizeof sptr, "%zu", (size_t)s->stream);
                vals[2] = sptr;
                nk = 3;
            }
            OrtStatus* st = a->UpdateCUDAProviderOptions(co, keys, vals, nk);
            if (st) {
                std::fprintf(stderr, "[ort] CUDA provider options: %s\n", a->GetErrorMessage(st));
                a->ReleaseStatus(st);
                a->ReleaseCUDAProviderOptions(co);
                a->ReleaseSessionOptions(opts);
                DestroySession(s);
                return nullptr;
            }
            st = a->SessionOptionsAppendExecutionProvider_CUDA_V2(opts, co);
            a->ReleaseCUDAProviderOptions(co);
            if (st) {
                std::fprintf(stderr, "[ort] 挂 CUDA EP 失败: %s\n", a->GetErrorMessage(st));
                a->ReleaseStatus(st);
                a->ReleaseSessionOptions(opts);
                DestroySession(s);
                return nullptr;
            }
            s->graph_on = graph;
        }
        if (s->fence) s->fence_ticket = FenceTicketArm();   // 武装窗口=CreateSession 一段
#ifdef _WIN32
        wchar_t wpath[1024];
        MultiByteToWideChar(CP_UTF8, 0, cfg.model_path.c_str(), -1, wpath, 1024);
        OrtStatus* stc = a->CreateSession(s->env, wpath, opts, &s->sess);
#else
        // POSIX 面 ORTCHAR_T=char：UTF-8 路径直传（Windows 面才需宽字符转换）
        OrtStatus* stc = a->CreateSession(s->env, cfg.model_path.c_str(),
                                          opts, &s->sess);
#endif
        if (s->fence) FenceTicketDisarm();   // 武装窗口=CreateSession 一段；
                                             // 窗口外其他会话的 kernel 一律无主
        a->ReleaseSessionOptions(opts);
        if (stc) {
            std::fprintf(stderr, "[ort] 建会话失败: %s\n", a->GetErrorMessage(stc));
            a->ReleaseStatus(stc);
            DestroySession(s);
            return nullptr;
        }
        // 零围栏件：RunOptions（关 EP 同步）+ 自建事件（disable timing；=2 与
        // fence 模式共用——fence 的事件在 Warmup 认领流后建）。失败=回落同步
        // （会话已建成，流无害留存）。
        if (s->async || s->fence) {
            // RO（关 EP 同步）对 =2/=3 都必须；事件仅 =2 用户流通道用（=3 完成走
            // fence 信号量，不再需要 CUDA 事件）
            if (a->CreateRunOptions(&s->ro)
                || a->AddRunConfigEntry(s->ro,
                                        "disable_synchronize_execution_providers",
                                        "1")
                || (s->async
                    && (!g_cu.EventCreateWithFlags
                        || g_cu.EventCreateWithFlags(&s->event, 0x02)))) {
                std::fprintf(stderr, "[ort] FARM_ORT_ASYNC：RunOptions/Event 失败"
                             "——本会话回落同步路径\n");
                if (s->ro) { a->ReleaseRunOptions(s->ro); s->ro = nullptr; }
                s->event = nullptr;
                s->async = false;
                s->fence = false;
            }
        }
        if (dml) {
            if (a->CreateMemoryInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault,
                                    &s->bind_mem)
                || a->CreateIoBinding(s->sess, &s->iob)) {
                DestroySession(s);
                return nullptr;
            }
        } else {
            if (a->CreateMemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault, &s->bind_mem)
                || a->CreateIoBinding(s->sess, &s->iob)) {
                DestroySession(s);
                return nullptr;
            }
        }
        // ---- 输入元数据 ----
        size_t n_in = 0;
        IgnoreStatus(a, a->SessionGetInputCount(s->sess, &n_in));
        if (n_in < 1 || n_in > 64) {
            std::fprintf(stderr, "[ort] 输入数 %zu ∉ [1,64]（模型不对？）\n", n_in);
            DestroySession(s);
            return nullptr;
        }
        s->ins.resize(n_in);
        OrtAllocator* alloc = nullptr;
        if (a->GetAllocatorWithDefaultOptions(&alloc) || !alloc) {
            DestroySession(s);
            return nullptr;
        }
        // （升格权重不在会话输入列表——ORT 把 overridable initializer 单列成
        //  独立类别（实测定谳 2026-09-29：SessionGetInputCount 只数真输入），
        //  见常规输入循环后的专列枚举段。）
        for (size_t i = 0; i < n_in; i++) {
            char* nm = nullptr;
            IgnoreStatus(a, a->SessionGetInputName(s->sess, i, alloc, &nm));
            s->ins[i].meta.name = nm ? nm : "?";
            if (nm) IgnoreStatus(a, a->AllocatorFree(alloc, nm));
            s->ins[i].meta.population =
                !cfg.population_input.empty()
                && s->ins[i].meta.name == cfg.population_input;   // 路由模式标记
            if (s->ins[i].meta.population && s->ins[i].meta.et != DTYPE_F32) {
                std::fprintf(stderr, "[ort] population 输入 %s 须 f32\n",
                             s->ins[i].meta.name.c_str());
                DestroySession(s);
                return nullptr;
            }
            OrtTypeInfo* ti = nullptr;
            if (a->SessionGetInputTypeInfo(s->sess, i, &ti)) { DestroySession(s); return nullptr; }
            const OrtTensorTypeAndShapeInfo* info = nullptr;
            if (a->CastTypeInfoToTensorInfo(ti, &info)) { a->ReleaseTypeInfo(ti); DestroySession(s); return nullptr; }
            ONNXTensorElementDataType et;
            IgnoreStatus(a, a->GetTensorElementType(info, &et));
            size_t nd = 0;
            IgnoreStatus(a, a->GetDimensionsCount(info, &nd));
            s->ins[i].meta.dims.resize(nd);
            IgnoreStatus(a, a->GetDimensions(info, s->ins[i].meta.dims.data(), nd));
            a->ReleaseTypeInfo(ti);
            s->ins[i].meta.et = OnnxToElem(et);
            s->ins[i].meta.esize = OnnxElemSize(et);
            if (!s->ins[i].meta.esize) {
                std::fprintf(stderr, "[ort] 输入 %s 非法元素类型\n", s->ins[i].meta.name.c_str());
                DestroySession(s);
                return nullptr;
            }
            if ((int)s->ins[i].meta.dims[0] != slots
                && !(s->ins[i].meta.population)) {
                std::fprintf(stderr, "[ort] 输入 %s dim0=%lld ≠ slots=%d——模型须先烤"
                             "成固定批形状（fb）\n", s->ins[i].meta.name.c_str(),
                             (long long)s->ins[i].meta.dims[0], slots);
                DestroySession(s);
                return nullptr;
            }
            size_t row = 1;
            for (size_t d = 1; d < nd; d++) row *= (size_t)s->ins[i].meta.dims[d];
            s->ins[i].meta.row_bytes = row * s->ins[i].meta.esize;
            // 声明式增量 H2D（判决25）：点名面标记。深度单位=行首维 dims[1]
            // （掼蛋形状 [256,18] → depth∈[0,256]、stride=18*esize 字节）。
            // population/weight 面与 <2 维面深度无定义=忽略点名（静默回落 full
            // ——full 永远正确，声明是加速非门槛）；
            // FARM_H2D_DELTA=0 杀手锏=全部忽略（零行为差回退）。
            if (!dml && !s->ins[i].meta.population && !s->ins[i].meta.weight
                && DeltaEnvOn()
                && (!cfg.append_inputs.empty() || !cfg.headlive_inputs.empty())
                && s->ins[i].meta.dims.size() >= 2
                && s->ins[i].meta.dims[1] > 0) {
                OrtIn& oi = s->ins[i];
                bool in_append = false, in_headlive = false;
                for (const auto& an : cfg.append_inputs)
                    if (an == oi.meta.name) { in_append = true; break; }
                for (const auto& hn : cfg.headlive_inputs)
                    if (hn == oi.meta.name) { in_headlive = true; break; }
                if (in_append && in_headlive) {
                    std::fprintf(stderr, "[ort] 面 %s 同时声明 append+headlive"
                                 "（互斥）——拒绝启动\n", oi.meta.name.c_str());
                    DestroySession(s);
                    return nullptr;
                }
                if (in_append || in_headlive) {
                    oi.meta.append = in_append;
                    oi.meta.headlive = in_headlive;
                    oi.max_depth = (int)oi.meta.dims[1];
                    oi.stride = (int)(oi.meta.row_bytes / (size_t)oi.meta.dims[1]);
                    oi.synced.assign((size_t)slots, 0);
                    oi.declared.assign((size_t)slots, -1);
                    s->any_append = true;   // any delta 面（append|headlive）
                }
            }
            if (spec_out) {
                InputMeta m = s->ins[i].meta;
                spec_out->ins.push_back(std::move(m));
            }
        }
        // ---- 升格权重面专列枚举（2026-09-29 实测定谳）：overridable initializer
        //      （initializer 兼 graph input=升格）不进会话输入列表，是独立类别
        //      ——按名 BindInput 即"覆写"语义（探针 P4 实证：覆写绑定缓冲→输出
        //      跟着变）。从专列 API 枚举后追加进 ins 尾部：arena carve/绑定/
        //      播种/换心/全豁免链自动覆盖。模型文件即声明面（升格工具产出），
        //      零配置；ORT 对升格权重仅 Warning"禁 const folding"不阻断。----
        {
            size_t n_ovr = 0;
            IgnoreStatus(a, a->SessionGetOverridableInitializerCount(s->sess, &n_ovr));
            for (size_t k = 0; k < n_ovr; k++) {
                char* nm = nullptr;
                IgnoreStatus(a, a->SessionGetOverridableInitializerName(s->sess, k, alloc, &nm));
                OrtIn wi;
                wi.meta.name = nm ? nm : "?";
                if (nm) IgnoreStatus(a, a->AllocatorFree(alloc, nm));
                if (!cfg.population_input.empty()
                    && wi.meta.name == cfg.population_input) {
                    std::fprintf(stderr, "[ort] 升格权重面 %s 与 population_input "
                                 "同名——两机制写同一面必打架，拒绝启动\n",
                                 wi.meta.name.c_str());
                    DestroySession(s);
                    return nullptr;
                }
                OrtTypeInfo* ti = nullptr;
                if (a->SessionGetOverridableInitializerTypeInfo(s->sess, k, &ti)) {
                    DestroySession(s);
                    return nullptr;
                }
                const OrtTensorTypeAndShapeInfo* info = nullptr;
                if (a->CastTypeInfoToTensorInfo(ti, &info) || !info) {
                    a->ReleaseTypeInfo(ti);
                    DestroySession(s);
                    return nullptr;
                }
                ONNXTensorElementDataType et;
                IgnoreStatus(a, a->GetTensorElementType(info, &et));
                size_t nd = 0;
                IgnoreStatus(a, a->GetDimensionsCount(info, &nd));
                wi.meta.dims.resize(nd);
                IgnoreStatus(a, a->GetDimensions(info, wi.meta.dims.data(), nd));
                a->ReleaseTypeInfo(ti);
                wi.meta.et = OnnxToElem(et);
                wi.meta.esize = OnnxElemSize(et);
                bool static_dims = nd >= 1;
                for (size_t d = 0; d < nd; d++)
                    if (wi.meta.dims[d] <= 0) static_dims = false;
                if (wi.meta.et != DTYPE_F32 || !wi.meta.esize || !static_dims) {
                    std::fprintf(stderr, "[ort] 升格权重面 %s 须 f32+全静态维"
                                 "（升格工具只升 float initializer）\n",
                                 wi.meta.name.c_str());
                    DestroySession(s);
                    return nullptr;
                }
                size_t row = 1;
                for (size_t d = 1; d < nd; d++) row *= (size_t)wi.meta.dims[d];
                wi.meta.row_bytes = row * wi.meta.esize;
                wi.meta.weight = true;
                if (spec_out) spec_out->ins.push_back(wi.meta);
                s->ins.push_back(std::move(wi));
                s->any_weight = true;
            }
        }
        // ---- 输出元数据（须 fp32）----
        size_t n_out = 0;
        IgnoreStatus(a, a->SessionGetOutputCount(s->sess, &n_out));
        if (n_out < 1 || n_out > 64) {
            std::fprintf(stderr, "[ort] 输出数 %zu ∉ [1,64]\n", n_out);
            DestroySession(s);
            return nullptr;
        }
        s->outs.resize(n_out);
        for (size_t j = 0; j < n_out; j++) {
            char* nm = nullptr;
            IgnoreStatus(a, a->SessionGetOutputName(s->sess, j, alloc, &nm));
            s->outs[j].meta.name = nm ? nm : "?";
            if (nm) IgnoreStatus(a, a->AllocatorFree(alloc, nm));
            OrtTypeInfo* ti = nullptr;
            if (a->SessionGetOutputTypeInfo(s->sess, j, &ti)) { DestroySession(s); return nullptr; }
            const OrtTensorTypeAndShapeInfo* info = nullptr;
            if (a->CastTypeInfoToTensorInfo(ti, &info) || !info) {
                a->ReleaseTypeInfo(ti);
                DestroySession(s);
                return nullptr;
            }
            ONNXTensorElementDataType et;
            IgnoreStatus(a, a->GetTensorElementType(info, &et));
            size_t nd = 0;
            IgnoreStatus(a, a->GetDimensionsCount(info, &nd));
            std::vector<int64_t> dims(nd);
            IgnoreStatus(a, a->GetDimensions(info, dims.data(), nd));
            a->ReleaseTypeInfo(ti);
            if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                std::fprintf(stderr, "[ort] 输出 %s 非 fp32（协议恒 f32）\n",
                             s->outs[j].meta.name.c_str());
                DestroySession(s);
                return nullptr;
            }
            if ((int)dims[0] != slots) {
                std::fprintf(stderr, "[ort] 输出 %s dim0=%lld ≠ slots=%d\n",
                             s->outs[j].meta.name.c_str(), (long long)dims[0], slots);
                DestroySession(s);
                return nullptr;
            }
            s->outs[j].meta.dims = dims;
            s->outs[j].meta.width = 1;
            for (size_t d = 1; d < nd; d++) s->outs[j].meta.width *= (int)dims[d];
            if (spec_out) spec_out->outs.push_back(s->outs[j].meta);
        }
        // ---- 输入单块 arena（256B 对齐 carve；CUDA=绑设备 carve，DML=纯宿主
        //      carve[VirtualAlloc 64K 对齐]，地址均终身固定）----
        const size_t kAlign = 256;
        size_t off = 0;
        for (size_t i = 0; i < s->ins.size(); i++)
            off = (off + ((s->ins[i].meta.population || s->ins[i].meta.weight)
                              ? s->ins[i].meta.row_bytes * (size_t)s->ins[i].meta.dims[0]
                              : s->ins[i].meta.row_bytes * (size_t)slots)
                      + kAlign - 1) / kAlign * kAlign;
        s->in_h_bytes = off;
        // P1-5 批拷贝 scratch（ins 数在此已终态）
        if (s->h2d_batch) {
            s->hb_dst.resize(s->ins.size());
            s->hb_src.resize(s->ins.size());
            s->hb_sizes.resize(s->ins.size());
            s->hb_attridx.resize(s->ins.size());
        }
        if (dml) {
#ifdef _WIN32
            s->in_d_bytes = 0;
            s->in_h_arena = VirtualAlloc(nullptr, s->in_h_bytes ? s->in_h_bytes : 1,
                                         MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            s->in_d_arena = s->in_h_arena;   // 宿主绑定：dev=host 同址（仅 CUDA 用区分）
            if (!s->in_h_arena) {
                std::fprintf(stderr, "[ort] DML 输入 arena(%zuB) 分配失败\n", s->in_h_bytes);
                DestroySession(s);
                return nullptr;
            }
#else
            // POSIX 不可达（DML 配置级拒绝于上方）——防御性留位
            DestroySession(s);
            return nullptr;
#endif
        } else {
            s->in_d_bytes = off;
            if (g_cu.SetDevice) g_cu.SetDevice(cfg.device_id);
            if (g_cu.HostAlloc(&s->in_h_arena, s->in_h_bytes, 0)
                || g_cu.Malloc(&s->in_d_arena, s->in_d_bytes)) {
                std::fprintf(stderr, "[ort] 输入 arena(%zuB) 分配失败\n", s->in_h_bytes);
                DestroySession(s);
                return nullptr;
            }
        }
        off = 0;
        for (size_t i = 0; i < s->ins.size(); i++) {
            // 绑定量=元数据口径：population/weight 面总量=row_bytes×dim0（dim0≠
            // slots！曾按 slots 统一乘→fb128 靠 P==slots 侥幸、fb1024 声明 8×
            // 实配→越界 700）
            size_t bytes = (s->ins[i].meta.population || s->ins[i].meta.weight)
                ? s->ins[i].meta.row_bytes * (size_t)s->ins[i].meta.dims[0]
                : s->ins[i].meta.row_bytes * (size_t)slots;
            off = (off + kAlign - 1) / kAlign * kAlign;
            s->ins[i].host = (char*)s->in_h_arena + off;
            s->ins[i].dev = (char*)s->in_d_arena + off;
            off += bytes;
            OrtValue* v = nullptr;
            if (a->CreateTensorWithDataAsOrtValue(s->bind_mem, s->ins[i].dev, bytes,
                                                  s->ins[i].meta.dims.data(),
                                                  s->ins[i].meta.dims.size(),
                                                  ElemToOnnx(s->ins[i].meta.et), &v)) {
                DestroySession(s);
                return nullptr;
            }
            s->ins[i].val = v;
            if (a->BindInput(s->iob, s->ins[i].meta.name.c_str(), v)) {
                DestroySession(s);
                return nullptr;
            }
        }
        // ---- 声明式增量影子块（判决25，仅 CUDA 路径）----
        // 影子=设备输入面的宿主镜像（只同步实际传输的段——这是哨兵有牙齿的
        // 前提：影子漂移=设备漂移）。VirtualAlloc 零页=零基起步（宿主零基组装
        // 起点=0、显存 zero 基=0、影子=0、synced=0 四方一致）；POSIX 面=堆零块
        // （增量通道暂未上非 Windows，防御面同语义）。
        if (s->any_append) {
            s->delta_dbg = DeltaDebugEnv();
            size_t tot = 0;
            for (size_t i = 0; i < n_in; i++)
                if (s->ins[i].meta.append || s->ins[i].meta.headlive)
                    tot = (tot + s->ins[i].meta.row_bytes * (size_t)slots
                           + kAlign - 1) / kAlign * kAlign;
#ifdef _WIN32
            s->shadow_blk = VirtualAlloc(nullptr, tot ? tot : 1,
                                         MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
            s->shadow_blk = std::malloc(tot ? tot : 1);
            if (s->shadow_blk) std::memset(s->shadow_blk, 0, tot ? tot : 1);
#endif
            if (!s->shadow_blk) {
                std::fprintf(stderr, "[ort] 增量影子块(%zuB) 分配失败\n", tot);
                DestroySession(s);
                return nullptr;
            }
            size_t soff = 0;
            for (size_t i = 0; i < n_in; i++) {
                if (!(s->ins[i].meta.append || s->ins[i].meta.headlive)) continue;
                soff = (soff + kAlign - 1) / kAlign * kAlign;
                s->ins[i].shadow = (char*)s->shadow_blk + soff;
                soff += s->ins[i].meta.row_bytes * (size_t)slots;
            }
            // 显存 zero 基（会话创建一次）：设备 append 面=影子（全零）。首批
            // 增量段只补 [0,depth)，行尾零由此保证（ Warmup 整块零拷幂等重做；
            // ProbeGraph 图案污染在彼处末尾重零）。
            if (g_cu.SetDevice) g_cu.SetDevice(s->dev_id);
            for (size_t i = 0; i < n_in; i++) {
                if (!(s->ins[i].meta.append || s->ins[i].meta.headlive)) continue;
                if (g_cu.Memcpy(s->ins[i].dev, s->ins[i].shadow,
                                s->ins[i].meta.row_bytes * (size_t)slots, 1)) {
                    std::fprintf(stderr, "[ort] 增量面 zero 基 memcpy 失败\n");
                    DestroySession(s);
                    return nullptr;
                }
            }
            // 批拷贝 scratch 扩容：增量段上限=每面每行至多一段+full 面前缀段
            if (s->h2d_batch) {
                size_t cap = s->ins.size() + 4;
                for (size_t i = 0; i < n_in; i++)
                    if (s->ins[i].meta.append || s->ins[i].meta.headlive)
                        cap += (size_t)slots;
                s->hb_dst.resize(cap);
                s->hb_src.resize(cap);
                s->hb_sizes.resize(cap);
                s->hb_attridx.resize(cap);
            }
            std::fprintf(stderr, "[ort] 声明式增量 H2D：%zu 个 append/headlive 面"
                         "（影子 %zuB，stride/depth 见面表%s）\n",
                         [&] { size_t k = 0;
                               for (auto& i : s->ins)
                                   if (i.meta.append || i.meta.headlive) k++;
                               return k; }(),
                         tot, s->delta_dbg ? "；哨兵=开" : "");
            std::fflush(stderr);
        }
        // ---- 输出单块 arena ----
        off = 0;
        for (size_t j = 0; j < n_out; j++) {
            s->outs[j].bytes = (size_t)s->outs[j].meta.width * 4 * (size_t)slots;
            off = (off + s->outs[j].bytes + kAlign - 1) / kAlign * kAlign;
        }
        s->out_h_bytes = off;
        if (dml) {
#ifdef _WIN32
            s->out_d_bytes = 0;
            s->out_h_arena = VirtualAlloc(nullptr, s->out_h_bytes ? s->out_h_bytes : 1,
                                          MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            s->out_d_arena = s->out_h_arena;
            if (!s->out_h_arena) {
                std::fprintf(stderr, "[ort] DML 输出 arena(%zuB) 分配失败\n", s->out_h_bytes);
                DestroySession(s);
                return nullptr;
            }
#else
            // POSIX 不可达（DML 配置级拒绝于上方）——防御性留位
            DestroySession(s);
            return nullptr;
#endif
        } else {
            s->out_d_bytes = off;
            if (g_cu.HostAlloc(&s->out_h_arena, s->out_h_bytes, 0)
                || g_cu.Malloc(&s->out_d_arena, s->out_d_bytes)) {
                std::fprintf(stderr, "[ort] 输出 arena(%zuB) 分配失败\n", s->out_h_bytes);
                DestroySession(s);
                return nullptr;
            }
        }
        off = 0;
        for (size_t j = 0; j < n_out; j++) {
            off = (off + kAlign - 1) / kAlign * kAlign;
            s->outs[j].host = (float*)((char*)s->out_h_arena + off);
            s->outs[j].dev = (char*)s->out_d_arena + off;
            off += s->outs[j].bytes;
            OrtValue* v = nullptr;
            if (a->CreateTensorWithDataAsOrtValue(s->bind_mem, s->outs[j].dev,
                                                  s->outs[j].bytes,
                                                  s->outs[j].meta.dims.data(),
                                                  s->outs[j].meta.dims.size(),
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v)) {
                DestroySession(s);
                return nullptr;
            }
            s->outs[j].val = v;
            if (a->BindOutput(s->iob, s->outs[j].meta.name.c_str(), v)) {
                DestroySession(s);
                return nullptr;
            }
        }
        if (spec_out) {
            spec_out->backend = dml ? "ort-dml" : "ort";
            spec_out->slots = slots;
        }
        // 路由键识别（population 模式）：1-D i64 非 population 输入=mid 类
        //（weight 面排除——升格的 1-D i64 权重若被当路由键批尾毒化=权重被打烂）
        if (!cfg.population_input.empty())
            for (size_t i = 0; i < n_in; i++) {
                if (s->ins[i].meta.population) s->pop_mode = true;
                else if (s->ins[i].meta.et == DTYPE_I64
                         && s->ins[i].meta.dims.size() == 1
                         && !s->ins[i].meta.weight)
                    s->mid_like.push_back((int)i);
            }
        if (s->any_weight) {
            std::fprintf(stderr, "[ort] 升格权重面 %zu 个：",
                         s->ins.size() - [&] { size_t k = 0;
                             for (auto& i : s->ins) if (!i.meta.weight) k++;
                             return k; }());
            for (auto& i : s->ins)
                if (i.meta.weight)
                    std::fprintf(stderr, " %s(%zuMB)", i.meta.name.c_str(),
                                 i.meta.row_bytes * (size_t)i.meta.dims[0] / (1 << 20));
            std::fprintf(stderr, "——RefitWeights(RW1) 换心通道；未换心发车将拒绝\n");
            std::fflush(stderr);
        }
        Track(s);
        if (dml) {
            // 专属发射线程（队头阻塞解药）：银行单飞（线性生命周期）⇒ 在飞
            // 作业 ≤1，投递槽单变量即够。失败仍记完成（loud print——银行无
            // 完成即败通道，看门狗/指纹门兜底）。
            s->helper = new std::thread([s] {
                for (;;) {
                    unsigned job = 0;
                    {
                        std::unique_lock<std::mutex> lk(s->h_mx);
                        s->h_cv.wait(lk, [s] { return s->h_stop || s->h_pending != 0; });
                        if (s->h_stop) return;
                        job = s->h_pending;
                    }
                    if (!RunOnce(s))
                        std::printf("[ort] DML 发射线程批异常（输出将陈旧）\n");
                    {
                        std::lock_guard<std::mutex> lk(s->h_mx);
                        s->h_pending = 0;
                    }
                    s->h_done.store(job, std::memory_order_release);
                }
            });
        }
        return s;
    }
};

InferBackend* CreateOrtBackend() { return new OrtBackend(); }

long long OrtFenceEngagedTotal() {
    return g_fence_engaged.load(std::memory_order_relaxed);
}

long long OrtDeltaDebugViolations() {
    return g_delta_violations.load(std::memory_order_relaxed);
}

} // namespace inferfarm
