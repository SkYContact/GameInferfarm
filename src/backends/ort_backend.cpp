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
// onnxruntime 1.30.0（q35 DLL）上以 FARM_BANK_WINDOW_FLOOR=1000 强制逐批自驱
// （self_dep=42/38）实测：结果逐位一致、无异常——即当前版本对回放的实际
// 约束比文档宽松。升级 ORT 版本时此结论须复验（自驱路径可退化为"置旗由
// 调度台发射"）。
#include "inferfarm/backend.h"
#include "cudart_dyn.h"
#include "onnxruntime_c_api.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace inferfarm {

static Cudart g_cu;
static std::mutex g_ort_mx;   // DLL/env 初始化互斥（会话各建 env；加载一次）
static HMODULE g_ort_dll = nullptr;
static const OrtApi* g_ort = nullptr;
static int g_ort_env_seq = 0;
static std::string g_ort_version = "?";

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

static bool LoadOrtLib(const ModelConfig& cfg) {
    std::lock_guard<std::mutex> lk(g_ort_mx);
    if (g_ort) return true;
    const char* ed = getenv("FARM_ORT_DIR");
    std::string ort_dir = !cfg.ort_dir.empty() ? cfg.ort_dir
        : (ed && *ed ? ed
           : "C:/Users/41601/Miniconda3/envs/q35/Lib/site-packages/onnxruntime/capi");
    const char* cd = getenv("FARM_CUDA_DIR");
    std::string cuda_dir = !cfg.cuda_dir.empty() ? cfg.cuda_dir
        : (cd && *cd ? cd : "C:/Users/41601/Miniconda3/envs/q35/Lib/site-packages/torch/lib");
    // PATH 前插（onnxruntime CUDA 版的 cudart/cublas 依赖解析）
    {
        char old_path[8192];
        GetEnvironmentVariableA("PATH", old_path, sizeof old_path);
        SetEnvironmentVariableA("PATH",
            (ort_dir + ";" + cuda_dir + ";" + old_path).c_str());
    }
    std::string dll = ort_dir + "\\onnxruntime.dll";
    HMODULE h = LoadLibraryA(dll.c_str());   // 绝对路径：绕开 exe 同目录 CPU 版
    if (!h) {
        std::fprintf(stderr, "[ort] LoadLibrary %s 失败 GLE=%lu\n", dll.c_str(), GetLastError());
        return false;
    }
    auto fn = (const OrtApiBase*(ORT_API_CALL*)())GetProcAddress(h, "OrtGetApiBase");
    if (!fn) {
        std::fprintf(stderr, "[ort] DLL 无 OrtGetApiBase\n");
        return false;
    }
    const OrtApiBase* base = fn();
    if (base->GetVersionString) {
        const char* vs = base->GetVersionString();
        if (vs && *vs) g_ort_version = vs;
    }
    g_ort = base->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        std::fprintf(stderr, "[ort] GetApi(%d) 失败（头/DLL 版本不匹配）\n", ORT_API_VERSION);
        return false;
    }
    g_ort_dll = h;
    std::fprintf(stderr, "[ort] %s 就绪（%s）\n", g_ort_version.c_str(), dll.c_str());
    return true;
}

// ScheduleSpin：设备等待恒忙等（Auto 策略睡 1-3ms/批）。须在 ORT 创建首个
// CUDA 上下文（首个 CUDA EP 会话）之前设——LoadOrtLib 时机即满足。
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

struct OrtIn {
    InputMeta meta;
    void* host = nullptr;   // pinned carve
    void* dev = nullptr;    // device carve
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
    OrtEnv* env = nullptr;
    OrtSession* sess = nullptr;
    OrtMemoryInfo* cuda_mem = nullptr;
    OrtIoBinding* iob = nullptr;
    std::vector<OrtIn> ins;
    std::vector<OrtOut> outs;
    void* in_h_arena = nullptr;  size_t in_h_bytes = 0;
    void* in_d_arena = nullptr;  size_t in_d_bytes = 0;
    void* out_h_arena = nullptr; size_t out_h_bytes = 0;
    void* out_d_arena = nullptr; size_t out_d_bytes = 0;
    bool graph_on = false;
    int slots = 64;
    // 完成协议（整设备同步血律）：Submit 后首个 CompletionReached 做一次
    // device sync + 前缀 D2H，随后同 seq 恒 true
    unsigned seq = 0;
    int last_n = 0;
    bool synced_for_seq = false;
};

class OrtBackend : public InferBackend {
public:
    const char* Name() const override { return "ort"; }

    bool LoadSpec(const ModelConfig& cfg, int slots, ModelSpec& out) override {
        if (!LoadOrtLib(cfg)) return false;
        if (!g_cu.Load(cfg.cuda_dir)) return false;
        SetSpinFlagsOnce();
        const OrtApi* a = g_ort;
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

    bool Warmup(void* session) override {
        OrtSess* s = (OrtSess*)session;
        memset(s->in_h_arena, 0, s->in_h_bytes);
        // 零填充 3 跑（enable_cuda_graph 内部前两跑构图/捕获——捕获窗口内
        // 不容地址/形状变化；地址已钉死=满足）
        for (int r = 0; r < 3; r++) {
            if (g_cu.Memcpy(s->in_d_arena, s->in_h_arena, s->in_h_bytes, 1))
                return false;
            if (!RunOnce(s)) return false;
            g_cu.DeviceSynchronize();
        }
        std::fprintf(stderr, "[ort] 热身就绪 slots=%d%s\n", s->slots,
                     s->graph_on ? "，CUDA Graph=开（银行会话：调度台线程绑定）" : "，CUDA Graph=关");
        return true;
    }

    bool ProbeGraph(void* session) override {
        // 图地址实验的 ORT 版：绑定地址钉死后，Run 读当前设备内存值。
        // 两图案可分辨 + 复跑稳定 + 换数据输出跟着变。
        OrtSess* s = (OrtSess*)session;
        std::vector<char> ref1, ref2, r2b;
        // 探针自己搬 D2H（生产路径的 D2H 在 CompletionReached——探针必须显式拷
        // 否则读到的是陈旧主机 arena）
        auto snap = [&](std::vector<char>& v) {
            g_cu.DeviceSynchronize();
            for (size_t j = 0; j < s->outs.size(); j++)
                g_cu.Memcpy(s->outs[j].host, s->outs[j].dev, s->outs[j].bytes, 2);
            v.assign((const char*)s->out_h_arena,
                     (const char*)s->out_h_arena + s->out_h_bytes);
        };
        // 跑图=H2D 整块 → RunOnce（生产路径同款：输入搬运在图外由调用方做）
        auto run_pat = [&]() {
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
        const OrtApi* a = g_ort;
        if (s->iob) a->ReleaseIoBinding(s->iob);
        for (auto& i : s->ins) if (i.val) a->ReleaseValue(i.val);
        for (auto& o : s->outs) if (o.val) a->ReleaseValue(o.val);
        if (s->sess) a->ReleaseSession(s->sess);
        if (s->env) a->ReleaseEnv(s->env);
        if (s->cuda_mem) a->ReleaseMemoryInfo(s->cuda_mem);
        if (s->in_h_arena) g_cu.FreeHost(s->in_h_arena);
        if (s->in_d_arena) g_cu.Free(s->in_d_arena);
        if (s->out_h_arena) g_cu.FreeHost(s->out_h_arena);
        if (s->out_d_arena) g_cu.Free(s->out_d_arena);
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
        // 前缀 H2D：同步拷贝（返回即完成——与 ORT 内部流旗标无关，零竞态；
        // n>7/8·slots 走整块）
        if (n_rows > (s->slots * 7) / 8) {
            if (g_cu.Memcpy(s->in_d_arena, s->in_h_arena, s->in_h_bytes, 1))
                return false;
        } else {
            for (size_t i = 0; i < s->ins.size(); i++)
                if (g_cu.Memcpy(s->ins[i].dev, s->ins[i].host,
                                (size_t)n_rows * s->ins[i].meta.row_bytes, 1))
                    return false;
        }
        if (!RunOnce(s)) return false;
        seq_out = s->seq;
        return true;   // Run 返回=已入队（不等 GPU 完成——收割侧 sync）
    }

    bool CompletionReached(void* session, unsigned seq) override {
        OrtSess* s = (OrtSess*)session;
        if (seq < s->seq) return true;             // 旧序号（早已完成）
        if (s->synced_for_seq) return true;
        // 整设备同步血律（不赌 ORT 内部流序）+ 前缀 D2H
        g_cu.DeviceSynchronize();
        int n = s->last_n;
        for (size_t j = 0; j < s->outs.size(); j++)
            if (g_cu.Memcpy(s->outs[j].host, s->outs[j].dev,
                            (size_t)n * (size_t)s->outs[j].meta.width * 4, 2))
                return false;
        s->synced_for_seq = true;
        return true;
    }
    void CompletionFence() override {}

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

private:
    static bool RunOnce(OrtSess* s) {
        const OrtApi* a = g_ort;
        OrtStatus* st = a->RunWithBinding(s->sess, nullptr, s->iob);
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
            size_t n = oi.meta.row_bytes * (size_t)s->slots / oi.meta.esize;
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

    // spec_out 非空=探测会话（LoadSpec 用）：跑完元数据枚举即毁
    OrtSess* CreateSession(const ModelConfig& cfg, int slots, bool for_bank,
                              ModelSpec* spec_out) {
        const OrtApi* a = g_ort;
        if (cfg.model_path.empty()) {
            std::fprintf(stderr, "[ort] 缺 model_path（fb 烤死的 onnx）\n");
            return nullptr;
        }
        OrtSess* s = new OrtSess();
        s->slots = slots;
        s->dll = g_ort_dll;
        // 每会话独立 env（会话/图/arena 全套自闭环，互不沾染共享态）
        char env_name[32];
        std::snprintf(env_name, sizeof env_name, "inferfarm_%d", g_ort_env_seq++);
        if (a->CreateEnv(ORT_LOGGING_LEVEL_ERROR, env_name, &s->env)) { DestroySession(s); return nullptr; }
        OrtSessionOptions* opts = nullptr;
        if (a->CreateSessionOptions(&opts)) { DestroySession(s); return nullptr; }
        a->SetIntraOpNumThreads(opts, cfg.ort_threads > 0 ? cfg.ort_threads : 1);
        a->SetInterOpNumThreads(opts, 1);
        a->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
        // CUDA EP（+CUDA Graph——KV 串与 python providers={"enable_cuda_graph":"1"}
        // 同义）。图仅银行会话开（PerThreadContext 铁律：创建/回放同线程——
        // 银行会话全生命周期在调度台线程上）。
        OrtCUDAProviderOptionsV2* co = nullptr;
        bool graph = for_bank && cfg.ort_cuda_graph;
        if (a->CreateCUDAProviderOptions(&co)) { a->ReleaseSessionOptions(opts); DestroySession(s); return nullptr; }
        const char* keys[] = {"device_id", "enable_cuda_graph"};
        const char* vals[] = {"0", graph ? "1" : "0"};
        OrtStatus* st = a->UpdateCUDAProviderOptions(co, keys, vals, 2);
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
        wchar_t wpath[1024];
        MultiByteToWideChar(CP_UTF8, 0, cfg.model_path.c_str(), -1, wpath, 1024);
        st = a->CreateSession(s->env, wpath, opts, &s->sess);
        a->ReleaseSessionOptions(opts);
        if (st) {
            std::fprintf(stderr, "[ort] 建会话失败: %s\n", a->GetErrorMessage(st));
            a->ReleaseStatus(st);
            DestroySession(s);
            return nullptr;
        }
        s->graph_on = graph;
        if (a->CreateMemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault, &s->cuda_mem)
            || a->CreateIoBinding(s->sess, &s->iob)) {
            DestroySession(s);
            return nullptr;
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
            if ((int)s->ins[i].meta.dims[0] != slots) {
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
        // ---- 输入单块 arena（256B 对齐 carve；绑设备 carve 地址，终身固定）----
        const size_t kAlign = 256;
        size_t off = 0;
        for (size_t i = 0; i < n_in; i++)
            off = (off + s->ins[i].meta.row_bytes * (size_t)slots + kAlign - 1)
                      / kAlign * kAlign;
        s->in_h_bytes = s->in_d_bytes = off;
        if (g_cu.HostAlloc(&s->in_h_arena, s->in_h_bytes, 0)
            || g_cu.Malloc(&s->in_d_arena, s->in_d_bytes)) {
            std::fprintf(stderr, "[ort] 输入 arena(%zuB) 分配失败\n", s->in_h_bytes);
            DestroySession(s);
            return nullptr;
        }
        off = 0;
        for (size_t i = 0; i < n_in; i++) {
            size_t bytes = s->ins[i].meta.row_bytes * (size_t)slots;
            off = (off + kAlign - 1) / kAlign * kAlign;
            s->ins[i].host = (char*)s->in_h_arena + off;
            s->ins[i].dev = (char*)s->in_d_arena + off;
            off += bytes;
            OrtValue* v = nullptr;
            if (a->CreateTensorWithDataAsOrtValue(s->cuda_mem, s->ins[i].dev, bytes,
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
        s->out_h_bytes = s->out_d_bytes = off;
        if (g_cu.HostAlloc(&s->out_h_arena, s->out_h_bytes, 0)
            || g_cu.Malloc(&s->out_d_arena, s->out_d_bytes)) {
            std::fprintf(stderr, "[ort] 输出 arena(%zuB) 分配失败\n", s->out_h_bytes);
            DestroySession(s);
            return nullptr;
        }
        off = 0;
        for (size_t j = 0; j < n_out; j++) {
            off = (off + kAlign - 1) / kAlign * kAlign;
            s->outs[j].host = (float*)((char*)s->out_h_arena + off);
            s->outs[j].dev = (char*)s->out_d_arena + off;
            off += s->outs[j].bytes;
            OrtValue* v = nullptr;
            if (a->CreateTensorWithDataAsOrtValue(s->cuda_mem, s->outs[j].dev,
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
            spec_out->backend = "ort";
            spec_out->slots = slots;
        }
        return s;
    }
};

InferBackend* CreateOrtBackend() { return new OrtBackend(); }

} // namespace inferfarm
