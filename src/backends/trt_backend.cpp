// trt_backend.cpp — TensorRT 后端（ai_infer.cpp TRT 机件的游戏无关抽取）。
//
// 承重三件套（YGO 产线实测定谳，勿凭直觉改）：
//  1. GPU 邮箱：mapped pinned 旗标 + 流末 4B 盖章拷贝 + CPU volatile 自旋
//     （~µs 检测），替代 cuStreamSynchronize 的 WDDM 围栏官道（0.36-0.5ms/批）。
//     正确性三支柱：①流序（盖章在 enqueueV3+输出 D2H 之后=旗标到即输出驻留）
//     ②x86 缓存相干（DMA 写 pinned 直达 CPU 视野；lfence=防御性载入序）
//     ③in-flight≤1/会话（staging 无覆写竞争，单调序号无 ABA）。
//  2. CUDA Graph 批捕获：[4B seq H2D→enqueueV3→输出 D2H→盖章] 四段一张图=
//     每批一次 WDDM 提交替代 3-4 次。图回放执行时读 staging/arena 当前值
//     （地址烧死≠值烧死——启动期 ProbeGraph 实验验证）。
//  3. refit 换心：refittable engine + RW1 blob，毫秒级权重热换；已捕获图
//     replay 读同一设备内存=新值（A4 门的性质支点）。
//
// ABI 手法（原样）：GetProcAddress 拿 createInferRuntime_INTERNAL /
// createInferRefitter_INTERNAL，显式传 DLL 真实版本整型（FARM_TRT_VERSION_INT
// 可覆盖；缺省头宏 NV_TENSORRT_VERSION）。TF32 纪律：烤制端
// NVIDIA_TF32_OVERRIDE=0，运行端必须一致（不一致拒建 context——fail fast）。
#include "inferfarm/backend.h"
#include "inferfarm/refit.h"
#include "cudart_dyn.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <intrin.h>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#if defined(INFERFARM_WITH_TRT)
#include <NvInferRuntime.h>
#define INFERFARM_TRT_OK 1
#else
#define INFERFARM_TRT_OK 0
#endif

namespace inferfarm {

#if !INFERFARM_TRT_OK
// 未编 TRT：返回空后端（MakeBackend 已打印"未知后端"路径——这里给明确报错）
InferBackend* CreateTrtBackend() {
    std::fprintf(stderr, "[trt] 本构建未开 TRT（CMake -DINFERFARM_WITH_TRT=ON）\n");
    return nullptr;
}
#else

// ---------------- 动态符号 ----------------
static Cudart g_cu;
static void* (*g_trt_create_runtime)(void*, int32_t) = nullptr;
static void* (*g_trt_create_refitter)(void*, void*, int32_t) = nullptr;
static int32_t g_trt_ver_int = 0;

// cuda_dir 缺省链：cfg / env FARM_CUDA_DIR / q35 torch/lib（与 ORT 后端同款）
static std::string DefaultCudaDir(const ModelConfig& cfg) {
    if (!cfg.cuda_dir.empty()) return cfg.cuda_dir;
    if (const char* e = getenv("FARM_CUDA_DIR")) if (*e) return e;
    return "C:/Users/41601/Miniconda3/envs/q35/Lib/site-packages/torch/lib";
}

static int32_t TrtVersionInt() {
    if (g_trt_ver_int) return g_trt_ver_int;
    if (const char* e = getenv("FARM_TRT_VERSION_INT")) {
        g_trt_ver_int = (int32_t)atoi(e);
        return g_trt_ver_int;
    }
#ifdef NV_TENSORRT_VERSION
    g_trt_ver_int = (int32_t)NV_TENSORRT_VERSION;
#else
    g_trt_ver_int = 10 * 10000 + 16 * 100 + 1;   // 10.16.1（本机 DLL 代）
#endif
    return g_trt_ver_int;
}

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING && msg && *msg)
            std::fprintf(stderr, "[trt] %s\n", msg);
    }
};
static TrtLogger g_trt_log;
static HMODULE g_trt_dll = nullptr;

struct TrtEngineCache {
    std::string path;
    nvinfer1::IRuntime* rt = nullptr;      // 全进程一份（create/deserialize 线程安全）
    nvinfer1::ICudaEngine* eng = nullptr;  // 一份权重；多会话各建 context
};
static TrtEngineCache g_trt_eng;

static bool LoadTrtLib(const ModelConfig& cfg) {
    if (g_trt_create_runtime) return true;
    const char* td = getenv("FARM_TRT_DIR");
    std::string trt_dir = !cfg.trt_dir.empty() ? cfg.trt_dir
        : (td && *td ? td
           : "C:/Users/41601/Miniconda3/envs/q35/Lib/site-packages/tensorrt_libs");
    // PATH 前插（nvinfer 的 cublas/cudart 依赖解析）——与 ORT 同款手法
    {
        char buf[8192];
        GetEnvironmentVariableA("PATH", buf, sizeof buf);
        std::string cuda_dir = DefaultCudaDir(cfg);
        SetEnvironmentVariableA("PATH",
            (cuda_dir + ";" + trt_dir + ";" + buf).c_str());
    }
    // TF32 纪律：烤制端 NVIDIA_TF32_OVERRIDE=0，运行端必须一致（Myelin 逐字
    // 比对 build/execution 两侧值，不一致拒建 context）。未设=自设 0；显式
    // 设 1=拒绝启动（防静默分叉）。
    {
        const char* tf = getenv("NVIDIA_TF32_OVERRIDE");
        if (!tf || !*tf) {
            _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
            std::fprintf(stderr, "[trt] NVIDIA_TF32_OVERRIDE 未设，已自设 0（与烤制端一致）\n");
        } else if (std::strcmp(tf, "0") != 0) {
            std::fprintf(stderr, "[trt] NVIDIA_TF32_OVERRIDE=%s 与烤制端(0)不一致，"
                         "拒绝启动（改 0 或 unset）\n", tf);
            return false;
        }
    }
    // 绝对路径 + LOAD_WITH_ALTERED_SEARCH_PATH：依赖解析先搜 nvinfer 自身目录
    // （普通 LoadLibrary 即便 PATH 前插仍 126——loader 对绝对路径加载的依赖
    // 解析不采纳进程内改写的 PATH；实测教训）
    std::string p = trt_dir + "\\nvinfer_10.dll";
    HMODULE h = LoadLibraryExA(p.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) {
        std::fprintf(stderr, "[trt] LoadLibrary %s 失败 GLE=%lu\n", p.c_str(), GetLastError());
        return false;
    }
    g_trt_create_runtime =
        (void* (*)(void*, int32_t))GetProcAddress(h, "createInferRuntime_INTERNAL");
    if (!g_trt_create_runtime) {
        std::fprintf(stderr, "[trt] nvinfer_10.dll 缺导出符号 createInferRuntime_INTERNAL\n");
        return false;
    }
    g_trt_create_refitter =
        (void* (*)(void*, void*, int32_t))GetProcAddress(h, "createInferRefitter_INTERNAL");
    g_trt_dll = h;
    return true;
}

static bool TrtDtypeToElem(nvinfer1::DataType dt, ElemDtype& out) {
    using nvinfer1::DataType;
    switch (dt) {
    case DataType::kFLOAT: out = DTYPE_F32; return true;
    case DataType::kINT64:  out = DTYPE_I64; return true;
    case DataType::kINT32:  out = DTYPE_I32; return true;
    case DataType::kBOOL:   out = DTYPE_BOOL; return true;
    default: return false;   // kHALF/kINT8 等不进 IO 面
    }
}

// ---------------- refit 换心 ----------------
static bool ApplyRefitWeights(nvinfer1::ICudaEngine* eng, const char* path) {
    double t0 = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    if (!g_trt_create_refitter) {
        std::fprintf(stderr, "[trt] DLL 缺 createInferRefitter_INTERNAL（换心不可用）\n");
        return false;
    }
    if (!eng->isRefittable()) {
        std::fprintf(stderr, "[trt] engine 不可 refit（烤制须开 refittable 旗标）\n");
        return false;
    }
    std::vector<char> blob;
    std::vector<Rw1Entry> ents;
    if (!ParseRw1(path, blob, ents)) return false;
    void* rp = g_trt_create_refitter(eng, &g_trt_log, TrtVersionInt());
    if (!rp)   // 版本整型回退（与 createInferRuntime 同款）
        rp = g_trt_create_refitter(eng, &g_trt_log, (int32_t)NV_TENSORRT_VERSION);
    if (!rp) {
        std::fprintf(stderr, "[trt] createInferRefitter 失败（版本整型 %d）\n",
                     (int)TrtVersionInt());
        return false;
    }
    nvinfer1::IRefitter* ref = (nvinfer1::IRefitter*)rp;
    bool failed = false;
    int n_set = 0, n_skip = 0;
    {
        // 引擎 refit 名单（名字集合校验：名单外 warning 跳过）
        int n_all = ref->getAllWeights(0, nullptr);
        std::vector<const char*> names((size_t)(n_all > 0 ? n_all : 0));
        if (n_all > 0) ref->getAllWeights(n_all, names.data());
        auto known = [&](const char* nm) {
            for (const char* k : names)
                if (k && std::strcmp(k, nm) == 0) return true;
            return false;
        };
        static const nvinfer1::DataType kDtypeMap[4] = {
            nvinfer1::DataType::kINT8, nvinfer1::DataType::kHALF,
            nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT64,
        };
        // 死区防御（与权威导出器同谓词；一般引擎无此名=零触发）：TRT 融合
        // 闭包把 dyt.alpha 族与 tmp_weight_N 锁成组，set 任一拖死整个 refit
        // （probe 实测），防御跳过。
        auto is_dead_zone = [](const std::string& nm) {
            static const char kSuf[] = "dyt.alpha";
            if (nm.size() >= sizeof kSuf - 1
                && nm.compare(nm.size() - (sizeof kSuf - 1), sizeof kSuf - 1, kSuf) == 0)
                return true;
            return nm.find("/m/dyt") != std::string::npos
                && nm.find("Constant_output_0") != std::string::npos;
        };
        for (const Rw1Entry& e : ents) {
            if (is_dead_zone(e.name)) {
                std::fprintf(stderr, "[trt] 死区跳过 %s（融合闭包锁死项）\n", e.name.c_str());
                n_skip++;
                continue;
            }
            if (!known(e.name.c_str())) {
                std::fprintf(stderr, "[trt] 名单外跳过 %s\n", e.name.c_str());
                n_skip++;
                continue;
            }
            // 原型校验（TRT 10.x：一参版返回 Weights，dtype/numel 都在里面）
            nvinfer1::Weights proto = ref->getWeightsPrototype(e.name.c_str());
            if (e.dtype < 4
                && (proto.type != kDtypeMap[e.dtype]
                    || (uint64_t)proto.count != (uint64_t)e.numel)) {
                std::fprintf(stderr, "[trt] %s dtype/numel 与引擎原型不符"
                             "（blob dtype=%u numel=%u 引擎 dtype=%d numel=%lld）"
                             "——fail fast\n",
                             e.name.c_str(), e.dtype, e.numel, (int)proto.type,
                             (long long)proto.count);
                failed = true;
                break;
            }
            nvinfer1::Weights w{};
            w.type = kDtypeMap[e.dtype];
            w.values = e.data;
            w.count = (int64_t)e.numel;
            if (!ref->setNamedWeights(e.name.c_str(), w)) {
                std::fprintf(stderr, "[trt] setNamedWeights(%s) 失败（dtype/numel 与引擎"
                             "原型不一致？numel=%u）\n", e.name.c_str(), e.numel);
                failed = true;
                break;
            }
            n_set++;
        }
    }
    if (!failed && !ref->refitCudaEngine()) {
        std::fprintf(stderr, "[trt] refit 失败: refitCudaEngine 返回 false");
        int n_miss = ref->getMissingWeights(0, nullptr);
        if (n_miss > 0) {   // 融合组不完整——列名字帮定位
            std::vector<const char*> miss((size_t)n_miss);
            ref->getMissingWeights(n_miss, miss.data());
            std::fprintf(stderr, "（缺 %d 项:", n_miss);
            for (int k = 0; k < n_miss && k < 8; k++)
                std::fprintf(stderr, "%s%s", k ? "," : "", miss[(size_t)k]);
            if (n_miss > 8) std::fprintf(stderr, ",…");
            std::fprintf(stderr, "）");
        }
        std::fprintf(stderr, "\n");
        failed = true;
    }
    delete ref;   // IRefitter 虚析构（v8+ 官方销毁道）
    if (failed) return false;
    double t1 = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::fprintf(stderr, "[trt] refit 换心 %d 项（名单外跳过 %d）耗时 %.0fms: %s\n",
                 n_set, n_skip, t1 - t0, path);
    return true;
}

// ---------------- 会话（InferCtx 的泛化）----------------
struct TrtIn {
    InputMeta meta;
    void* host = nullptr;   // pinned arena carve
    void* dev = nullptr;    // device arena carve
};
struct TrtOut {
    OutputMeta meta;
    void* dev = nullptr;
    float* host = nullptr;  // pinned
    size_t bytes = 0;
};
struct TrtSession {
    nvinfer1::IExecutionContext* ctx = nullptr;
    void* stream = nullptr;   // 专用流：enqueueV3 走默认流时 TRT 自插
                              // cudaStreamSynchronize（警告+税；非默认流实测净）
    std::vector<TrtIn> ins;
    std::vector<TrtOut> outs;
    void* in_h_arena = nullptr;  size_t in_h_bytes = 0;
    void* in_d_arena = nullptr;  size_t in_d_bytes = 0;
    void* out_h_arena = nullptr; size_t out_h_bytes = 0;
    void* out_d_arena = nullptr; size_t out_d_bytes = 0;
    // GPU 邮箱：128B mapped pinned（[0]=旗标 [64]=seq staging，分缓存行）
    void* mb_host = nullptr;
    void* mb_flag_dev = nullptr;
    void* mb_seq_dev = nullptr;
    unsigned mb_seq = 0;
    bool mb_ok = false;
    // CUDA Graph 批捕获
    void* graph = nullptr;
    bool graph_ok = false;
    int slots = 64;
    int last_n = 0;
};

class TrtBackend : public InferBackend {
public:
    const char* Name() const override { return "trt"; }

    bool LoadSpec(const ModelConfig& cfg, int slots, ModelSpec& out) override {
        if (!LoadTrtLib(cfg) || !g_cu.Load(DefaultCudaDir(cfg))) return false;
        if (!EnsureEngine(cfg)) return false;
        nvinfer1::ICudaEngine* eng = g_trt_eng.eng;
        out.backend = "trt";
        out.slots = slots;
        int n_io = eng->getNbIOTensors();
        for (int i = 0; i < n_io; i++) {
            const char* nm = eng->getIOTensorName(i);
            bool is_in = eng->getTensorIOMode(nm) == nvinfer1::TensorIOMode::kINPUT;
            if (is_in) {
                InputMeta m;
                m.name = nm;
                ElemDtype et;
                if (!TrtDtypeToElem(eng->getTensorDataType(nm), et)) {
                    std::fprintf(stderr, "[trt] 输入 %s 非法元素类型（f32/i64/i32/bool 之外）\n", nm);
                    return false;
                }
                m.et = et;
                m.esize = DtypeSize(et);
                nvinfer1::Dims d = eng->getTensorShape(nm);
                for (int j = 0; j < d.nbDims; j++) m.dims.push_back(d.d[j]);
                if ((int)m.dims[0] != slots) {
                    std::fprintf(stderr, "[trt] 输入 %s dim0=%lld ≠ slots=%d\n",
                                 nm, (long long)m.dims[0], slots);
                    return false;
                }
                size_t row = 1;
                for (size_t dd = 1; dd < m.dims.size(); dd++) {
                    if (m.dims[dd] < 0) {
                        std::fprintf(stderr, "[trt] 输入 %s 含动态维（须烤死静态形状）\n", nm);
                        return false;
                    }
                    row *= (size_t)m.dims[dd];
                }
                m.row_bytes = row * m.esize;
                out.ins.push_back(std::move(m));
            } else {
                if (eng->getTensorDataType(nm) != nvinfer1::DataType::kFLOAT) {
                    std::fprintf(stderr, "[trt] 输出 %s 非 fp32（烤制须保持 IO fp32）\n", nm);
                    return false;
                }
                OutputMeta m;
                m.name = nm;
                nvinfer1::Dims d = eng->getTensorShape(nm);
                for (int j = 0; j < d.nbDims; j++) m.dims.push_back(d.d[j]);
                if ((int)m.dims[0] != slots) {
                    std::fprintf(stderr, "[trt] 输出 %s dim0=%lld ≠ slots=%d\n", nm,
                                 (long long)m.dims[0], slots);
                    return false;
                }
                m.width = 1;
                for (size_t dd = 1; dd < m.dims.size(); dd++) m.width *= (int)m.dims[dd];
                out.outs.push_back(std::move(m));
            }
        }
        if (out.ins.empty() || out.outs.empty() || out.ins.size() > 64) {
            std::fprintf(stderr, "[trt] 输入数 %zu 非法（1..64）\n", out.ins.size());
            return false;
        }
        return true;
    }

    void* CreateSession(const ModelConfig& cfg, const ModelSpec& spec, bool for_bank) override {
        (void)cfg; (void)for_bank;
        if (!g_trt_eng.eng) return nullptr;
        nvinfer1::ICudaEngine* eng = g_trt_eng.eng;
        TrtSession* s = new TrtSession();
        s->slots = spec.slots;
        s->ctx = eng->createExecutionContext(
            nvinfer1::ExecutionContextAllocationStrategy::kSTATIC);
        if (!s->ctx) {
            std::fprintf(stderr, "[trt] createExecutionContext 失败（TF32 与烤制端不一致即此症）\n");
            delete s;
            return nullptr;
        }
        if (g_cu.StreamCreate(&s->stream, 0)) {
            std::fprintf(stderr, "[trt] 专用流创建失败\n");
            delete s;
            return nullptr;
        }
        const size_t kAlign = 256;
        // 输入单块 arena（256B 对齐 carve；IOBinding/setTensorAddress 绑 carve
        // 地址，地址终身固定）
        size_t off = 0;
        s->ins.resize(spec.ins.size());
        for (size_t i = 0; i < spec.ins.size(); i++) {
            s->ins[i].meta = spec.ins[i];
            off = (off + spec.ins[i].row_bytes * (size_t)spec.slots + kAlign - 1)
                      / kAlign * kAlign;
        }
        s->in_h_bytes = s->in_d_bytes = off;
        if (g_cu.HostAlloc(&s->in_h_arena, s->in_h_bytes, 0)
            || g_cu.Malloc(&s->in_d_arena, s->in_d_bytes)) {
            std::fprintf(stderr, "[trt] 输入 arena(%zuB) 分配失败\n", s->in_h_bytes);
            delete s;
            return nullptr;
        }
        off = 0;
        for (size_t i = 0; i < s->ins.size(); i++) {
            size_t bytes = s->ins[i].meta.row_bytes * (size_t)s->slots;
            off = (off + kAlign - 1) / kAlign * kAlign;
            s->ins[i].host = (char*)s->in_h_arena + off;
            s->ins[i].dev = (char*)s->in_d_arena + off;
            off += bytes;
            if (!s->ctx->setTensorAddress(s->ins[i].meta.name.c_str(), s->ins[i].dev)) {
                std::fprintf(stderr, "[trt] setTensorAddress(%s) 失败\n",
                             s->ins[i].meta.name.c_str());
                delete s;
                return nullptr;
            }
        }
        // 输出单块 arena
        s->outs.resize(spec.outs.size());
        off = 0;
        for (size_t j = 0; j < spec.outs.size(); j++) {
            s->outs[j].meta = spec.outs[j];
            s->outs[j].bytes = (size_t)spec.outs[j].width * 4 * (size_t)spec.slots;
            off = (off + s->outs[j].bytes + kAlign - 1) / kAlign * kAlign;
        }
        s->out_h_bytes = s->out_d_bytes = off;
        if (g_cu.HostAlloc(&s->out_h_arena, s->out_h_bytes, 0)
            || g_cu.Malloc(&s->out_d_arena, s->out_d_bytes)) {
            std::fprintf(stderr, "[trt] 输出 arena(%zuB) 分配失败\n", s->out_h_bytes);
            delete s;
            return nullptr;
        }
        off = 0;
        for (size_t j = 0; j < s->outs.size(); j++) {
            off = (off + kAlign - 1) / kAlign * kAlign;
            s->outs[j].host = (float*)((char*)s->out_h_arena + off);
            s->outs[j].dev = (char*)s->out_d_arena + off;
            off += s->outs[j].bytes;
            if (!s->ctx->setTensorAddress(s->outs[j].meta.name.c_str(), s->outs[j].dev)) {
                std::fprintf(stderr, "[trt] setTensorAddress(%s) 失败\n",
                             s->outs[j].meta.name.c_str());
                delete s;
                return nullptr;
            }
        }
        // GPU 邮箱：mapped pinned 128B + 设备 seq 槽（失败=回退流同步，不挡启动）
        if (g_cu.HostAlloc(&s->mb_host, 128, 4 /*cudaHostAllocMapped*/) == 0
            && g_cu.HostGetDevicePointer(&s->mb_flag_dev, s->mb_host, 0) == 0
            && g_cu.Malloc(&s->mb_seq_dev, 64) == 0) {
            memset(s->mb_host, 0, 128);   // 旗标初值 0（首个期望序号=1）
            s->mb_ok = true;
        } else {
            s->mb_ok = false;
            std::fprintf(stderr, "[trt] 邮箱分配失败——本会话回退流同步\n");
        }
        return s;
    }

    bool Warmup(void* session) override {
        TrtSession* s = (TrtSession*)session;
        memset(s->in_h_arena, 0, s->in_h_bytes);
        for (int r = 0; r < 3; r++) {
            if (g_cu.Memcpy(s->in_d_arena, s->in_h_arena, s->in_h_bytes, 1))
                return false;
            if (!s->ctx->enqueueV3((cudaStream_t)s->stream)) return false;
            if (g_cu.StreamSynchronize) g_cu.StreamSynchronize(s->stream);
            else g_cu.DeviceSynchronize();
        }
        // 3 跑后图捕获（惰性分配已落定，捕获窗口内不容分配）；验证发射+自旋
        if (s->mb_ok) {
            CaptureGraph(s);
            if (!s->graph_ok) {   // 在线邮箱冒烟（链路坏=回退流同步，不让首批挂）
                unsigned seq = 0;
                if (!MbSubmit(s, seq)) { s->mb_ok = false; return false; }
                if (!WaitFlag(s, seq, 5000.0)) { s->mb_ok = false; return true; }
            }
        }
        std::fprintf(stderr, "[trt] 热身就绪 slots=%d%s%s\n", s->slots,
                     s->mb_ok ? "，邮箱=开" : "", s->graph_ok ? "，批图捕获=开" : "");
        return true;
    }

    // 图地址烧死小实验（bank_contract 关键工程点 1）：两图案可分辨 + 图回放
    // ==在线参考 + 换数据图输出跟着变（非烧死快照）。任一不过=false。
    bool ProbeGraph(void* session) override {
        TrtSession* s = (TrtSession*)session;
        if (!s->graph_ok) {
            std::fprintf(stderr, "[trt-probe] 无批图——银行制要求图+邮箱，拒绝\n");
            return false;
        }
        std::vector<char> ref1, ref2, g1, g2;
        auto snap = [&](std::vector<char>& v) {
            v.assign((const char*)s->out_h_arena,
                     (const char*)s->out_h_arena + s->out_h_bytes);
        };
        auto online = [&](std::vector<char>& v) {
            g_cu.Memcpy(s->in_d_arena, s->in_h_arena, s->in_h_bytes, 1);
            s->ctx->enqueueV3((cudaStream_t)s->stream);
            g_cu.StreamSynchronize(s->stream);
            g_cu.Memcpy(s->out_h_arena, s->out_d_arena, s->out_h_bytes, 2);
            snap(v);
        };
        auto graphrun = [&](std::vector<char>& v) {
            // 发车同款：先整块 h2d（图内只有 4B seq H2D，输入搬运在图外）
            if (g_cu.Memcpy(s->in_d_arena, s->in_h_arena, s->in_h_bytes, 1)) {
                v.clear();
                return;
            }
            unsigned seq = 0;
            double spin = 0;
            if (!MbSubmit(s, seq) || !WaitFlag(s, seq, 5000.0, &spin)) {
                v.clear();
                return;
            }
            snap(v);
        };
        FillPattern(s, 1);
        online(ref1);
        FillPattern(s, 2);
        online(ref2);
        bool diff = ref1 != ref2;
        FillPattern(s, 1);
        graphrun(g1);
        bool g1ok = g1 == ref1;
        FillPattern(s, 2);
        graphrun(g2);
        bool g2ok = g2 == ref2;
        bool bok = diff && g1ok && g2ok;
        std::printf("[trt-probe] 两图案可分辨=%d 图回放1逐位=%d 图回放2逐位=%d%s\n",
                    (int)diff, (int)g1ok, (int)g2ok, bok ? "" : " ←FAIL");
        std::fflush(stdout);
        return bok;
    }

    void DestroySession(void* session) override {
        TrtSession* s = (TrtSession*)session;
        if (!s) return;
        if (s->graph && g_cu.GraphDestroy) g_cu.GraphDestroy(s->graph);
        if (s->ctx) delete s->ctx;
        if (s->stream && g_cu.StreamDestroy) g_cu.StreamDestroy(s->stream);
        if (s->in_h_arena) g_cu.FreeHost(s->in_h_arena);
        if (s->in_d_arena) g_cu.Free(s->in_d_arena);
        if (s->out_h_arena) g_cu.FreeHost(s->out_h_arena);
        if (s->out_d_arena) g_cu.Free(s->out_d_arena);
        if (s->mb_host) g_cu.FreeHost(s->mb_host);
        if (s->mb_seq_dev) g_cu.Free(s->mb_seq_dev);
        delete s;
    }

    void* InputRow(void* session, const char* name, int slot, size_t* row_bytes) override {
        TrtSession* s = (TrtSession*)session;
        for (auto& i : s->ins)
            if (i.meta.name == name) {
                if (row_bytes) *row_bytes = i.meta.row_bytes;
                return (char*)i.host + (size_t)slot * i.meta.row_bytes;
            }
        return nullptr;
    }

    // 前缀 h2d（n > 7/8·slots 走整块；尾行旧数据=行独立无害）+ 异步发射
    bool SubmitBatch(void* session, int n_rows, unsigned& seq_out) override {
        TrtSession* s = (TrtSession*)session;
        if (n_rows > s->slots) n_rows = s->slots;
        s->last_n = n_rows;
        if (n_rows > (s->slots * 7) / 8) {
            if (g_cu.MemcpyAsync(s->in_d_arena, s->in_h_arena, s->in_h_bytes, 1, s->stream))
                return false;
        } else {
            for (size_t i = 0; i < s->ins.size(); i++)
                if (g_cu.MemcpyAsync(s->ins[i].dev, s->ins[i].host,
                                     (size_t)n_rows * s->ins[i].meta.row_bytes,
                                     1, s->stream))
                    return false;
        }
        return MbSubmit(s, seq_out);
    }

    bool CompletionReached(void* session, unsigned seq) override {
        TrtSession* s = (TrtSession*)session;
        if (!s->mb_ok) {   // 降级流同步路径：发射即等完（SubmitBatch 已同步）
            g_cu.DeviceSynchronize();
            g_cu.Memcpy(s->out_h_arena, s->out_d_arena, s->out_h_bytes, 2);
            return true;
        }
        volatile unsigned* flag = (volatile unsigned*)s->mb_host;
        return *flag == seq;
    }
    void CompletionFence() override { _mm_lfence(); }

    const float* OutputRow(void* session, const char* name, int slot) override {
        TrtSession* s = (TrtSession*)session;
        for (auto& o : s->outs)
            if (o.meta.name == name)
                return o.host + (size_t)slot * (size_t)o.meta.width;
        return nullptr;
    }
    int OutputWidth(void* session, const char* name) override {
        TrtSession* s = (TrtSession*)session;
        for (auto& o : s->outs)
            if (o.meta.name == name) return o.meta.width;
        return -1;
    }

    bool RefitWeights(const char* rw1_path) override {
        if (!g_trt_eng.eng) {
            std::fprintf(stderr, "[trt] engine 未载——换心不可用\n");
            return false;
        }
        return ApplyRefitWeights(g_trt_eng.eng, rw1_path);
    }

private:
    static bool EnsureEngine(const ModelConfig& cfg) {
        if (g_trt_eng.eng) {
            if (!cfg.engine_path.empty() && g_trt_eng.path != cfg.engine_path) {
                std::fprintf(stderr, "[trt] 一进程只支持一个 engine（已载 %s，又要 %s）\n",
                             g_trt_eng.path.c_str(), cfg.engine_path.c_str());
                return false;
            }
            return true;
        }
        // ScheduleSpin：enqueueV3 异步返回后的设备等待恒忙等（Auto 策略会睡
        // 1-3ms/批——唤醒税）。须在首个 CUDA 调用（ctx 创建）前设。
        if (g_cu.GetDeviceFlags && g_cu.SetDeviceFlags) {
            unsigned fl = 0;
            if (g_cu.GetDeviceFlags(&fl) == 0)
                g_cu.SetDeviceFlags(fl | 0x01 /*cudaDeviceScheduleSpin*/);
        }
        void* rt = g_trt_create_runtime(&g_trt_log, TrtVersionInt());
        if (!rt)
            rt = g_trt_create_runtime(&g_trt_log, (int32_t)NV_TENSORRT_VERSION);
        if (!rt) {
            std::fprintf(stderr, "[trt] createInferRuntime 失败（版本整型 %d，DLL 与"
                         " engine 不同代？）\n", (int)TrtVersionInt());
            return false;
        }
        g_trt_eng.rt = (nvinfer1::IRuntime*)rt;
        const std::string& p = cfg.engine_path;
        FILE* f = fopen(p.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "[trt] 打不开 engine 文件: %s\n", p.c_str());
            return false;
        }
        // fseek 定长一次读入堆（栈缓冲 1MB 级会当场爆默认线程栈——0xC00000FD 实测）
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<char> blob((size_t)sz);
        size_t got = sz > 0 ? fread(blob.data(), 1, (size_t)sz, f) : 0;
        fclose(f);
        if (got != (size_t)sz || blob.empty()) {
            std::fprintf(stderr, "[trt] engine 读取不完整: %s\n", p.c_str());
            return false;
        }
        g_trt_eng.eng = g_trt_eng.rt->deserializeCudaEngine(blob.data(), blob.size());
        if (!g_trt_eng.eng) {
            std::fprintf(stderr, "[trt] 反序列化失败: %s（TF32 环境不一致会拒建 context）\n",
                         p.c_str());
            return false;
        }
        g_trt_eng.path = p;
        return true;   // 换心走 RefitWeights()（Farm 在 context 创建前调）
    }

    // 邮箱提交：序号 +1 写 staging → [图=graphLaunch（H2D 节点执行时读 staging
    // 当前值）] / [在线=4B H2D→enqueueV3→输出 D2H→盖章 逐个入队 stream]
    static bool MbSubmit(TrtSession* s, unsigned& seq_out) {
        seq_out = ++s->mb_seq;
        if (!s->mb_ok) {   // 流同步降级：enqueue+同步（提交税同步税都在）
            if (!s->ctx->enqueueV3((cudaStream_t)s->stream)) return false;
            g_cu.StreamSynchronize(s->stream);
            g_cu.Memcpy(s->out_h_arena, s->out_d_arena, s->out_h_bytes, 2);
            return true;
        }
        *(volatile unsigned*)((char*)s->mb_host + 64) = seq_out;
        if (s->graph_ok && s->graph)
            return g_cu.GraphLaunch(s->graph, s->stream) == 0;
        if (g_cu.MemcpyAsync(s->mb_seq_dev, (char*)s->mb_host + 64, 4, 1, s->stream))
            return false;
        if (!s->ctx->enqueueV3((cudaStream_t)s->stream)) return false;
        if (g_cu.MemcpyAsync(s->out_h_arena, s->out_d_arena, s->out_h_bytes, 2, s->stream))
            return false;
        if (g_cu.MemcpyAsync(s->mb_flag_dev, s->mb_seq_dev, 4, 2, s->stream)) return false;
        return true;
    }

    static bool WaitFlag(TrtSession* s, unsigned expect, double timeout_ms,
                         double* spin_ms = nullptr) {
        volatile unsigned* flag = (volatile unsigned*)s->mb_host;
        auto t0 = std::chrono::steady_clock::now();
        unsigned n = 0;
        auto elapsed = [&] {
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        };
        for (;;) {
            if (*flag == expect) {
                _mm_lfence();
                if (spin_ms) *spin_ms = elapsed();
                return true;
            }
            if ((++n & 0xFFF) == 0) {
                double w = elapsed();
                if (w > timeout_ms) {
                    if (spin_ms) *spin_ms = w;
                    return false;
                }
            }
            _mm_pause();
        }
    }

    static void FillPattern(TrtSession* s, int seed) {
        for (auto& i : s->ins) {
            size_t n = i.meta.row_bytes * (size_t)s->slots / i.meta.esize;
            if (i.meta.et == DTYPE_F32) {
                float* p = (float*)i.host;
                for (size_t e = 0; e < n; e++)
                    p[e] = (float)((int)((e * 31 + (size_t)seed * 997) % 2039) - 1019) / 1019.0f;
            } else if (i.meta.et == DTYPE_I64) {
                long long* p = (long long*)i.host;
                for (size_t e = 0; e < n; e++)
                    p[e] = (long long)((e * 7 + (size_t)seed * 13) % 14969);
            } else if (i.meta.et == DTYPE_I32) {
                int* p = (int*)i.host;
                for (size_t e = 0; e < n; e++) p[e] = 0;   // 行号 0=防越界 Gather
            } else {
                unsigned char* p = (unsigned char*)i.host;
                for (size_t e = 0; e < n; e++) p[e] = (unsigned char)((e + (size_t)seed) & 1);
            }
        }
    }

    // CUDA Graph 批捕获（[4B seq H2D→enqueueV3→输出 D2H→盖章] 四段一张图）
    static void CaptureGraph(TrtSession* s) {
        if (!s->mb_ok || !g_cu.StreamBeginCapture || !g_cu.StreamEndCapture
            || !g_cu.GraphInstantiate || !g_cu.GraphLaunch || !g_cu.GraphDestroy) {
            std::fprintf(stderr, "[trt] 图捕获前置不满足（邮箱/符号缺失）——降级仅邮箱\n");
            return;
        }
        if (g_cu.StreamBeginCapture(s->stream, 0 /*cudaStreamCaptureModeGlobal*/) != 0) {
            std::fprintf(stderr, "[trt] cudaStreamBeginCapture 失败——降级仅邮箱\n");
            return;
        }
        // 注：staging 不在捕获期写——H2D 节点在**执行时**读其当前值，MbSubmit
        // 每次发射前写好（in-flight≤1=无覆写竞争）。
        bool ok = true;
        if (g_cu.MemcpyAsync(s->mb_seq_dev, (char*)s->mb_host + 64, 4, 1, s->stream)) ok = false;
        if (ok && !s->ctx->enqueueV3((cudaStream_t)s->stream)) ok = false;
        if (ok && g_cu.MemcpyAsync(s->out_h_arena, s->out_d_arena, s->out_h_bytes, 2, s->stream))
            ok = false;
        if (ok && g_cu.MemcpyAsync(s->mb_flag_dev, s->mb_seq_dev, 4, 2, s->stream)) ok = false;
        void* g = nullptr;
        int ec = g_cu.StreamEndCapture(s->stream, &g);
        if (!ok || ec != 0 || !g) {
            std::fprintf(stderr, "[trt] CUDA Graph 捕获失败（ok=%d ec=%d）——降级仅邮箱\n",
                         (int)ok, ec);
            if (g) g_cu.GraphDestroy(g);
            return;
        }
        void* ge = nullptr;
        if (g_cu.GraphInstantiate(&ge, g, 0) != 0 || !ge) {
            std::fprintf(stderr, "[trt] cudaGraphInstantiate 失败——降级仅邮箱\n");
            g_cu.GraphDestroy(g);
            return;
        }
        g_cu.GraphDestroy(g);
        s->graph = ge;
        s->graph_ok = true;
        unsigned seq = 0;
        double spin = 0;
        if (!MbSubmit(s, seq) || !WaitFlag(s, seq, 5000.0, &spin)) {
            std::fprintf(stderr, "[trt] 图验证发射/旗标超时——降级仅邮箱\n");
            s->graph = nullptr;
            s->graph_ok = false;
            return;
        }
        std::fprintf(stderr, "[trt] CUDA Graph 批捕获成功（验证自旋 %.3fms）\n", spin);
    }
};

InferBackend* CreateTrtBackend() { return new TrtBackend(); }

#endif // INFERFARM_WITH_TRT

} // namespace inferfarm
