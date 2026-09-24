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
//  3. **无 refit**：ORT 无权重热换 API——RefitWeights 恒 false（ES 候选迭代
//     每腿回退重载会话=秒级；演化场景走 TRT 后端）。
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
#endif

namespace inferfarm {

static Cudart g_cu;   // 进程一份（cudart 与 ORT 实例无关；DML 实例不加载）

// DLL 注册表：双 ORT 共存改名律（基名冲突=拷贝改名再装；依赖 PATH 前插）
static std::mutex g_ort_dll_mx;
struct OrtDllRec { std::string base; std::string dir; HMODULE h; };
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

// P1-5 批拷贝通道：镜像 cudaMemcpyAttributes（CUDA v13.0 driver_types.h:2345；
// 本仓不引 CUDA 头，布局手核：int enum + 2×cudaMemLocation{enum,uint} + uint
// = 24B @align4，static_assert 防 Layout 漂移）
struct HbAttr {
    int srcAccessOrder;        // 0x3=SrcAccessOrderAny（host 锚写稳、无在先流触碰）
    unsigned srcLocHint[2];    // cudaMemLocation（非托管/忽略场景全零）
    unsigned dstLocHint[2];
    unsigned flags;
};
static_assert(sizeof(HbAttr) == 24, "cudaMemcpyAttributes 布局漂移");

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
};
struct OrtOut {
    OutputMeta meta;
    void* dev = nullptr;
    float* host = nullptr;
    size_t bytes = 0;
    OrtValue* val = nullptr;   // 交 iob 钉住（进程寿命）
};
struct OrtSess {
    HMODULE dll = nullptr;      // 持引用（进程寿命不卸）
    const OrtApi* api = nullptr;   // 所属实例的 api（实例终身不毁=指针终身有效）
    OrtEnv* env = nullptr;
    OrtSession* sess = nullptr;
    OrtMemoryInfo* bind_mem = nullptr;   // CUDA 路径="Cuda"，DML 路径="Cpu"
    OrtIoBinding* iob = nullptr;
    std::vector<OrtIn> ins;
    std::vector<OrtOut> outs;
    void* in_h_arena = nullptr;  size_t in_h_bytes = 0;
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
    // ORT 内部流）；host 在 Run 提交后用该流做 D2H MemcpyAsync+EventRecord
    // （图外常规异步操作）。确定性锚=H2D 同步 memcpy；收割=纯 EventQuery。
    // 认领在 Warmup（首跑 Compute 必已登记；CreateSession 后 kernel 未跑过）。
    bool fence = false;
    uint64_t fence_ticket = 0;
    void* fence_stream = nullptr;   // 认领自注册表的 ORT EP 统一流
    OrtCustomOpDomain* domain = nullptr;   // fence 域（ReleaseSession 后释放）
    // dep 三段分解累计（P1 仪器；调度台单线程写，打印即清零）
    unsigned long long dep_h2d_ns = 0, dep_run_ns = 0, dep_d2h_ns = 0;
    unsigned dep_cnt = 0;
    // P1-5 批拷贝 scratch（FARM_H2D_BATCH=1 且符号在）：稀疏批多输入 H2D
    // 合并为一次 cudaMemcpyBatchAsync（dep=宿主提交税∝提交次数）
    bool h2d_batch = false;
    std::vector<void*> hb_dst, hb_src;
    std::vector<size_t> hb_sizes, hb_attridx;
    void* stream = nullptr;
    void* event = nullptr;
    OrtRunOptions* ro = nullptr;
};

class OrtBackend : public InferBackend {
public:
    const char* Name() const override { return dml_ ? "ort-dml" : "ort"; }

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

    bool Warmup(void* session) override {
        OrtSess* s = (OrtSess*)session;
        memset(s->in_h_arena, 0, s->in_h_bytes);
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
        // 此刻持票认领，record→sync→query 验证整链（流指针合法+事件真盖章）。
        // 失败（模型未打补丁/EP 回落 CPU）=回落同步，不等看门狗
        if (s->fence) {
            s->fence_stream = FenceTicketClaim(s->fence_ticket);
            bool ok = s->fence_stream && g_cu.EventRecord && g_cu.EventQuery;
            if (ok) ok = g_cu.EventRecord(s->event, s->fence_stream) == 0;
            if (ok) {
                g_cu.DeviceSynchronize();
                ok = g_cu.EventQuery(s->event) == 0;
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
                     s->dml ? "，DML=宿主绑定+同步 Run（无图无围栏）"
                            : (s->graph_on ? "，CUDA Graph=开（银行会话：调度台线程绑定）"
                                           : "，CUDA Graph=关"),
                     s->async ? "，零围栏=开（用户流+事件收割）" : "",
                     s->fence ? "，fence 桥接=开（图尾事件章+同步锚）" : "");
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
        std::printf("[ort-probe] 两图案可分辨=%d 复跑稳定=%d%s\n",
                    (int)diff, (int)stable, bok ? "" : " ←FAIL");
        std::fflush(stdout);
        return bok;
    }

    void DestroySession(void* session) override {
        OrtSess* s = (OrtSess*)session;
        if (!s) return;
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
            if (s->in_h_arena) VirtualFree(s->in_h_arena, 0, MEM_RELEASE);
            if (s->out_h_arena) VirtualFree(s->out_h_arena, 0, MEM_RELEASE);
        } else {
            if (s->in_h_arena) g_cu.FreeHost(s->in_h_arena);
            if (s->in_d_arena) g_cu.Free(s->in_d_arena);
            if (s->out_h_arena) g_cu.FreeHost(s->out_h_arena);
            if (s->out_d_arena) g_cu.Free(s->out_d_arena);
        }
        delete s;
    }

    void* InputRow(void* session, const char* name, int slot, size_t* row_bytes) override {
        OrtSess* s = (OrtSess*)session;
        for (auto& i : s->ins)
            if (i.meta.name == name) {
                if (row_bytes) *row_bytes = i.meta.row_bytes;
                return (char*)i.host + (size_t)slot * i.meta.row_bytes;
            }
        return nullptr;
    }

    bool SubmitBatch(void* session, int n_rows, unsigned& seq_out) override {
        OrtSess* s = (OrtSess*)session;
        if (n_rows > s->slots) n_rows = s->slots;
        if (n_rows < 1) n_rows = 1;
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
            // 前缀 H2D：同步拷贝（返回即完成——与 ORT 内部流旗标无关，零竞态；
            // n>7/8·slots 走整块；population 面跳过）
            if (n_rows > (s->slots * 7) / 8) {
                if (!h2d(s->in_d_arena, s->in_h_arena, s->in_h_bytes))
                    return false;
            } else if (s->fence && s->h2d_batch && s->ins.size() >= 2) {
                // P1-5 批拷贝判决实验（FARM_H2D_BATCH=1）：稀疏批多输入 H2D
                // 合并为一次 cudaMemcpyBatchAsync——dep=宿主提交税∝提交次数
                //（WDDM 下 5-10µs/次，多输入模型线性放大）。同流序不变 ⇒
                // 逐位等价；API 失败=整批判负（与单拷失败同纪律）。pop mid
                // 尾段毒化照旧单拷。
                size_t nb = 0;
                for (size_t i = 0; i < s->ins.size(); i++) {
                    if (s->ins[i].meta.population) continue;
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
                if (g_cu.MemcpyBatchAsync(s->hb_dst.data(), s->hb_src.data(),
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
                    if (s->ins[i].meta.population) continue;
                    if (!h2d(s->ins[i].dev, s->ins[i].host,
                             (size_t)n_rows * s->ins[i].meta.row_bytes))
                        return false;
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
            const long long dep_t1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (!RunOnce(s, s->ro)) return false;
            const long long dep_t2 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (s->fence) {
                // fence 桥接 v2：replay/eager 内核已提交到 ORT EP 统一流——D2H
                // 前缀异步挂同流（流序=图内核之后 ⇒ 读到本批输出）+ 事件盖章。
                // 图外常规异步操作，零捕获语义参与（v1 捕获窗内 record 判死）。
                for (size_t j = 0; j < s->outs.size(); j++)
                    if (g_cu.MemcpyAsync(s->outs[j].host, s->outs[j].dev,
                                         (size_t)s->last_n
                                             * (size_t)s->outs[j].meta.width * 4,
                                         2, s->fence_stream))
                        return false;
                if (g_cu.EventRecord(s->event, s->fence_stream)) return false;
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
                                 "run=%.3f d2h=%.3f ms/均\n", s->dep_cnt, s->last_n,
                                 s->dep_h2d_ns / 256e6, s->dep_run_ns / 256e6,
                                 s->dep_d2h_ns / 256e6);
                    std::fflush(stderr);
                    s->dep_h2d_ns = s->dep_run_ns = s->dep_d2h_ns = 0;
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
            // fence 桥接 v2：事件在 D2H 之后同流盖章——ready=前缀输出已驻留
            // host arena（pinned 完成可见性）。纯查询，µs 级。
            if (!g_cu.EventQuery || g_cu.EventQuery(s->event) != 0) return false;
            s->synced_for_seq = true;
            return true;
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

    bool RefitWeights(const char*) override {
        std::fprintf(stderr, "[ort] ORT 无权重热换 API——换心仅 TRT 后端支持"
                     "（演化场景走 TRT；ORT 候选迭代=每腿重载会话）\n");
        return false;
    }

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

private:
    HMODULE dll_ = nullptr;
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
                         "——onnxruntime.dll 走系统 DLL 搜索（加载失败先查这里）\n");
        std::lock_guard<std::mutex> lk(g_ort_dll_mx);
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
        std::string base = ToLower("onnxruntime.dll");
        HMODULE h = nullptr;
        std::string load_path;
        for (auto& r : g_ort_dlls)
            if (r.base == base) {
                if (ToLower(r.dir) == ToLower(ort_dir)) { h = r.h; break; }
                // 基名冲突（Windows 按基名去重回柄）：拷贝改名再装
                char tmp[MAX_PATH];
                GetTempPathA(MAX_PATH, tmp);
                load_path = std::string(tmp) + "inferfarm_ort_"
                    + std::to_string(g_ort_dll_seq.fetch_add(1)) + ".dll";
                if (!CopyFileA((ort_dir + "\\onnxruntime.dll").c_str(),
                               load_path.c_str(), FALSE)) {
                    std::fprintf(stderr, "[ort] 基名冲突改名拷贝失败 GLE=%lu（%s → %s）\n",
                                 GetLastError(), ort_dir.c_str(), load_path.c_str());
                    return false;
                }
                std::fprintf(stderr, "[ort] 双 ORT 共存：另一 onnxruntime.dll 已驻留，"
                             "本实例改载改名副本 %s（依赖走 PATH）\n", load_path.c_str());
                break;
            }
        if (!h) {
            if (load_path.empty()) load_path = ort_dir + "\\onnxruntime.dll";
            h = LoadLibraryA(load_path.c_str());   // 绝对路径：绕开 exe 同目录 CPU 版
        }
        if (!h) {
            std::fprintf(stderr, "[ort] LoadLibrary %s 失败 GLE=%lu\n",
                         load_path.c_str(), GetLastError());
            return false;
        }
        auto fn = (const OrtApiBase*(ORT_API_CALL*)())GetProcAddress(h, "OrtGetApiBase");
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
            std::fprintf(stderr, "[ort] GetApi(%d) 失败（头/DLL 版本不匹配）\n", ORT_API_VERSION);
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
        }
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
        a->SetIntraOpNumThreads(opts, cfg.ort_threads > 0 ? cfg.ort_threads : 1);
        a->SetInterOpNumThreads(opts, 1);
        a->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
        if (dml) {
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
                s->h2d_batch = [] {
                    const char* e = getenv("FARM_H2D_BATCH");
                    return e && *e && atoi(e) == 1;
                }() && g_cu.MemcpyBatchAsync != nullptr;   // 符号缺席=回退逐输入
                std::fprintf(stderr, "[ort] FARM_ORT_ASYNC=3：fence 桥接启用"
                             "（H2D 同步锚+图尾事件章）%s\n",
                             s->h2d_batch ? "+批拷贝 H2D（FARM_H2D_BATCH=1）" : "");
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
        wchar_t wpath[1024];
        MultiByteToWideChar(CP_UTF8, 0, cfg.model_path.c_str(), -1, wpath, 1024);
        if (s->fence) s->fence_ticket = FenceTicketArm();
        OrtStatus* stc = a->CreateSession(s->env, wpath, opts, &s->sess);
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
            if (a->CreateRunOptions(&s->ro)
                || a->AddRunConfigEntry(s->ro,
                                        "disable_synchronize_execution_providers",
                                        "1")
                || !g_cu.EventCreateWithFlags
                || g_cu.EventCreateWithFlags(&s->event, 0x02)) {
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
        a->SessionGetInputCount(s->sess, &n_in);
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
        for (size_t i = 0; i < n_in; i++) {
            char* nm = nullptr;
            a->SessionGetInputName(s->sess, i, alloc, &nm);
            s->ins[i].meta.name = nm ? nm : "?";
            if (nm) a->AllocatorFree(alloc, nm);
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
            a->GetTensorElementType(info, &et);
            size_t nd = 0;
            a->GetDimensionsCount(info, &nd);
            s->ins[i].meta.dims.resize(nd);
            a->GetDimensions(info, s->ins[i].meta.dims.data(), nd);
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
            if (spec_out) {
                InputMeta m = s->ins[i].meta;
                spec_out->ins.push_back(std::move(m));
            }
        }
        // ---- 输出元数据（须 fp32）----
        size_t n_out = 0;
        a->SessionGetOutputCount(s->sess, &n_out);
        if (n_out < 1 || n_out > 64) {
            std::fprintf(stderr, "[ort] 输出数 %zu ∉ [1,64]\n", n_out);
            DestroySession(s);
            return nullptr;
        }
        s->outs.resize(n_out);
        for (size_t j = 0; j < n_out; j++) {
            char* nm = nullptr;
            a->SessionGetOutputName(s->sess, j, alloc, &nm);
            s->outs[j].meta.name = nm ? nm : "?";
            if (nm) a->AllocatorFree(alloc, nm);
            OrtTypeInfo* ti = nullptr;
            if (a->SessionGetOutputTypeInfo(s->sess, j, &ti)) { DestroySession(s); return nullptr; }
            const OrtTensorTypeAndShapeInfo* info = nullptr;
            if (a->CastTypeInfoToTensorInfo(ti, &info) || !info) {
                a->ReleaseTypeInfo(ti);
                DestroySession(s);
                return nullptr;
            }
            ONNXTensorElementDataType et;
            a->GetTensorElementType(info, &et);
            size_t nd = 0;
            a->GetDimensionsCount(info, &nd);
            std::vector<int64_t> dims(nd);
            a->GetDimensions(info, dims.data(), nd);
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
        for (size_t i = 0; i < n_in; i++)
            off = (off + (s->ins[i].meta.population
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
            s->in_d_bytes = 0;
            s->in_h_arena = VirtualAlloc(nullptr, s->in_h_bytes ? s->in_h_bytes : 1,
                                         MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            s->in_d_arena = s->in_h_arena;   // 宿主绑定：dev=host 同址（仅 CUDA 用区分）
            if (!s->in_h_arena) {
                std::fprintf(stderr, "[ort] DML 输入 arena(%zuB) 分配失败\n", s->in_h_bytes);
                DestroySession(s);
                return nullptr;
            }
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
        for (size_t i = 0; i < n_in; i++) {
            // 绑定量=元数据口径：population 面总量=row_bytes×P（dim0=P≠slots！
            // 曾按 slots 统一乘→fb128 靠 P==slots 侥幸、fb1024 声明 8×实配→越界 700）
            size_t bytes = s->ins[i].meta.population
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
        // ---- 输出单块 arena ----
        off = 0;
        for (size_t j = 0; j < n_out; j++) {
            s->outs[j].bytes = (size_t)s->outs[j].meta.width * 4 * (size_t)slots;
            off = (off + s->outs[j].bytes + kAlign - 1) / kAlign * kAlign;
        }
        s->out_h_bytes = off;
        if (dml) {
            s->out_d_bytes = 0;
            s->out_h_arena = VirtualAlloc(nullptr, s->out_h_bytes ? s->out_h_bytes : 1,
                                          MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            s->out_d_arena = s->out_h_arena;
            if (!s->out_h_arena) {
                std::fprintf(stderr, "[ort] DML 输出 arena(%zuB) 分配失败\n", s->out_h_bytes);
                DestroySession(s);
                return nullptr;
            }
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
        if (!cfg.population_input.empty())
            for (size_t i = 0; i < n_in; i++) {
                if (s->ins[i].meta.population) s->pop_mode = true;
                else if (s->ins[i].meta.et == DTYPE_I64
                         && s->ins[i].meta.dims.size() == 1)
                    s->mid_like.push_back((int)i);
            }
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

} // namespace inferfarm
