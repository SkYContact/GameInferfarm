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
//     （地址烧死≠值烧死——启动期 ProbeGraph 实验验证）。大输出模型非满批
//     另走计算图（前两段）+ 图外逐输出前缀 D2H（部分行档，见 MbSubmit）。
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
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>   // _mm_lfence/_mm_pause 的 GCC/Clang 面（用点=TRT 门内）
#endif
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>   // dlopen/dlsym：Linux 面（2026-09-27 移植）
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

// cuda_dir 缺省链：cfg / env FARM_CUDA_DIR / 空（走系统 DLL 搜索——不写死
// 开发机路径；与 ORT 后端同款）
static std::string DefaultCudaDir(const ModelConfig& cfg) {
    if (!cfg.cuda_dir.empty()) return cfg.cuda_dir;
    if (const char* e = getenv("FARM_CUDA_DIR")) if (*e) return e;
    return "";
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
#ifdef _WIN32
static HMODULE g_trt_dll = nullptr;
#else
static void* g_trt_dll = nullptr;
#endif

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
        : (td && *td ? td : "");
    if (trt_dir.empty())
        std::fprintf(stderr, "[trt] 未设 ModelConfig.trt_dir / env FARM_TRT_DIR"
                     "——nvinfer 运行库走系统库搜索（加载失败先查这里）\n");
#ifdef _WIN32
    // PATH 前插（nvinfer 的 cublas/cudart 依赖解析）——与 ORT 同款手法
    {
        char buf[8192];
        GetEnvironmentVariableA("PATH", buf, sizeof buf);
        std::string np = buf;
        std::string cuda_dir = DefaultCudaDir(cfg);
        if (!trt_dir.empty()) np = trt_dir + ";" + np;
        if (!cuda_dir.empty()) np = cuda_dir + ";" + np;
        SetEnvironmentVariableA("PATH", np.c_str());
    }
#else
    // Linux：运行期改 LD_LIBRARY_PATH 不影响本进程 dlopen 搜索（glibc 启动
    // 时快照）——nvinfer 依赖（cublas/cudart 等）须经进程启动时的
    // LD_LIBRARY_PATH（见 farm_env.sh）/ldconfig 缓存供面。
#endif
    // TF32 纪律：烤制端 NVIDIA_TF32_OVERRIDE=0，运行端必须一致（Myelin 逐字
    // 比对 build/execution 两侧值，不一致拒建 context）。未设=自设 0；显式
    // 设 1=拒绝启动（防静默分叉）。
    {
        const char* tf = getenv("NVIDIA_TF32_OVERRIDE");
        if (!tf || !*tf) {
#ifdef _WIN32
            _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
            setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif
            std::fprintf(stderr, "[trt] NVIDIA_TF32_OVERRIDE 未设，已自设 0（与烤制端一致）\n");
        } else if (std::strcmp(tf, "0") != 0) {
            std::fprintf(stderr, "[trt] NVIDIA_TF32_OVERRIDE=%s 与烤制端(0)不一致，"
                         "拒绝启动（改 0 或 unset）\n", tf);
            return false;
        }
    }
    // 目录非空：绝对路径 + LOAD_WITH_ALTERED_SEARCH_PATH（依赖解析先搜
    // nvinfer 自身目录——普通 LoadLibrary 即便 PATH 前插仍 126：loader 对
    // 绝对路径加载的依赖解析不采纳进程内改写的 PATH；实测教训）。
    // 目录空：裸名加载（Windows 标准搜索：应用目录→系统32→PATH——语义同
    // cudart_dyn 空 dir）。此前空目录曾拼出 "\nvinfer_10.dll" 根路径必败
    // （对外反馈 2026-09-24）。
#ifdef _WIN32
    HMODULE h;
    if (trt_dir.empty()) {
        h = LoadLibraryA("nvinfer_10.dll");
        if (!h)
            std::fprintf(stderr, "[trt] LoadLibrary nvinfer_10.dll（系统搜索）失败 GLE=%lu\n",
                         GetLastError());
    } else {
        std::string p = trt_dir + "\\nvinfer_10.dll";
        h = LoadLibraryExA(p.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!h)
            std::fprintf(stderr, "[trt] LoadLibrary %s 失败 GLE=%lu\n",
                         p.c_str(), GetLastError());
    }
#else
    // Linux 面（2026-09-27 移植）：dlopen libnvinfer.so.10；目录空=裸名
    // （ld.so 搜索，语义同 cudart_dyn 空 dir）。INTERNAL ABI 符号 Linux .so
    // 同样导出（nm -D 实证 10.16.1）。
    void* h;
    std::string p = trt_dir.empty() ? std::string("libnvinfer.so.10")
                                    : trt_dir + "/libnvinfer.so.10";
    h = ::dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h)
        std::fprintf(stderr, "[trt] dlopen %s 失败：%s（显式设 FARM_TRT_DIR"
                     " 指向 libnvinfer.so.10 所在目录）\n", p.c_str(), dlerror());
#endif
    if (!h) return false;
#ifdef _WIN32
    g_trt_create_runtime =
        (void* (*)(void*, int32_t))GetProcAddress(h, "createInferRuntime_INTERNAL");
    if (!g_trt_create_runtime) {
        std::fprintf(stderr, "[trt] nvinfer_10.dll 缺导出符号 createInferRuntime_INTERNAL\n");
        return false;
    }
    g_trt_create_refitter =
        (void* (*)(void*, void*, int32_t))GetProcAddress(h, "createInferRefitter_INTERNAL");
#else
    g_trt_create_runtime =
        (void* (*)(void*, int32_t))::dlsym(h, "createInferRuntime_INTERNAL");
    if (!g_trt_create_runtime) {
        std::fprintf(stderr, "[trt] libnvinfer.so.10 缺导出符号 createInferRuntime_INTERNAL\n");
        return false;
    }
    g_trt_create_refitter =
        (void* (*)(void*, void*, int32_t))::dlsym(h, "createInferRefitter_INTERNAL");
#endif
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
    void* graph_c = nullptr;   // 计算图（seq H2D→enqueueV3，无回拷/盖章）：
    bool graph_c_ok = false;   // 部分行 D2H 档的前缀图（大输出模型专用）
    // ③成对状态行（设备常驻池）：本会话解析结果（池指针拷贝自 backend 级
    // 分配——MbSubmit 是 static，经会话携带）
    bool has_state = false;              // 任一配对成立（恒图外尾段形态）
    std::vector<int> st_in, st_out;      // 输入/输出下标→池号（-1=非状态）
    std::vector<void*> pool_devs;        // 池号→设备池基址
    std::vector<int> pool_rows;          // 池号→池行数（幻影行卫的 static 面用）
    const std::atomic<int>* st_pids = nullptr;   // 银行槽→池下标（Claim 写/
                                                  // 发车读；未绑=Init 期冒烟走旧路）
    // ③批量 D2D scratch（填充/散射逐行小拷合并为单次 cudaMemcpyBatchAsync
    //——发车段 API 税∝提交次数；判决25 同机器。调度台线程独占=无锁）
    struct StBatch {
        std::vector<void*> dst, src;
        std::vector<size_t> sz, ai;
        void clear() { dst.clear(); src.clear(); sz.clear(); ai.clear(); }
        bool empty() const { return dst.empty(); }
    } sb;
    int slots = 64;
    int dev = 0;      // 会话设备（多卡：分配/流/图/邮箱全落此设备）
    int last_n = 0;
};

class TrtBackend : public InferBackend {
public:
    const char* Name() const override { return "trt"; }
    bool LoadSpec(const ModelConfig& cfg, int slots, ModelSpec& out) override {
        if (!cfg.population_input.empty()) {
            std::fprintf(stderr, "[trt] population 路由（演化）暂不支持 TRT 后端"
                         "——路由图走 ort/cpu（判决16：pop 为图输入，TRT 前缀拷"
                         "协议未覆盖该面）\n");
            return false;
        }
        if (!LoadTrtLib(cfg) || !g_cu.Load(DefaultCudaDir(cfg))) return false;
        dev_id_ = cfg.device_id;   // 多卡：engine 反序列化落定设备（同架构双卡
        if (g_cu.SetDevice) g_cu.SetDevice(dev_id_);   // 可共享 engine；>0 本机未测）
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
        (void)for_bank;
        if (!g_trt_eng.eng) return nullptr;
        nvinfer1::ICudaEngine* eng = g_trt_eng.eng;
        TrtSession* s = new TrtSession();
        s->slots = spec.slots;
        s->dev = cfg.device_id;
        if (g_cu.SetDevice) g_cu.SetDevice(s->dev);   // 上下文/流/arena 落对设备
        s->ctx = eng->createExecutionContext(
            nvinfer1::ExecutionContextAllocationStrategy::kSTATIC);
        if (!s->ctx) {
            std::fprintf(stderr, "[trt] createExecutionContext 失败（TF32 与烤制端不一致即此症）\n");
            delete s;
            return nullptr;
        }
        if (g_cu.StreamCreate(&s->stream, 0)) {
            std::fprintf(stderr, "[trt] 专用流创建失败\n");
            DestroySession(s);   // 中段失败完整回收（防泄漏）
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
            DestroySession(s);   // 中段失败完整回收（防泄漏）
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
                DestroySession(s);   // 中段失败完整回收（防泄漏）
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
            DestroySession(s);   // 中段失败完整回收（防泄漏）
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
                DestroySession(s);   // 中段失败完整回收（防泄漏）
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
        // ③成对状态行解析（docs/state-residency-design.md）：backend 级设备池
        // 首会话分配一次（零基同步 memset）；会话记录池指针（MbSubmit=static，
        // 经会话携带）。in/out 行字节不等或名字未命中=fail fast。
        s->st_in.assign(spec.ins.size(), -1);
        s->st_out.assign(spec.outs.size(), -1);
        if (!cfg.state_pairs.empty()) {
            if (cfg.state_pool_rows <= 0) {
                std::fprintf(stderr, "[trt] state_pairs 声明但 state_pool_rows=%d"
                             "（Farm 应填 chains）\n", cfg.state_pool_rows);
                DestroySession(s);
                return nullptr;
            }
            if (st_pools_.empty()) {
                for (const auto& pr : cfg.state_pairs) {
                    int ii = -1, jj = -1;
                    for (size_t k = 0; k < spec.ins.size() && ii < 0; k++)
                        if (spec.ins[k].name == pr.in) ii = (int)k;
                    for (size_t k = 0; k < spec.outs.size() && jj < 0; k++)
                        if (spec.outs[k].name == pr.out) jj = (int)k;
                    const size_t rb_in = ii >= 0 ? spec.ins[(size_t)ii].row_bytes : 0;
                    const size_t rb_out =
                        jj >= 0 ? (size_t)spec.outs[(size_t)jj].width * 4 : 1;
                    if (ii < 0 || jj < 0 || !rb_in || rb_in != rb_out) {
                        std::fprintf(stderr, "[trt] state_pair %s↔%s 非法（in_idx=%d"
                                     " out_idx=%d 行字节 %zu↔%zu 须相等且非零）\n",
                                     pr.in.c_str(), pr.out.c_str(), ii, jj, rb_in, rb_out);
                        DestroySession(s);
                        return nullptr;
                    }
                    StatePool p;
                    p.row_bytes = rb_in;
                    p.rows = cfg.state_pool_rows;
                    p.zero_pending = new std::atomic<char>[(size_t)p.rows];
                    for (int t = 0; t < p.rows; t++) p.zero_pending[t].store(0);
                    if (g_cu.Malloc(&p.dev, p.row_bytes * (size_t)(p.rows + 1))
                        || g_cu.Memset(p.dev, 0,
                                       p.row_bytes * (size_t)(p.rows + 1))) {
                        std::fprintf(stderr, "[trt] 状态池分配/零基失败（%d 行 × %zuB）\n",
                                     p.rows, p.row_bytes);
                        if (p.dev) g_cu.Free(p.dev);
                        delete[] p.zero_pending;
                        DestroySession(s);
                        return nullptr;
                    }
                    st_pools_.push_back(p);
                }
            } else if (st_pools_.size() != cfg.state_pairs.size()) {
                std::fprintf(stderr, "[trt] state_pairs 数目 %zu 与首会话 %zu 不一致\n",
                             cfg.state_pairs.size(), st_pools_.size());
                DestroySession(s);
                return nullptr;
            }
            for (size_t pi = 0; pi < cfg.state_pairs.size(); pi++) {
                const auto& pr = cfg.state_pairs[pi];
                for (size_t k = 0; k < spec.ins.size(); k++)
                    if (spec.ins[k].name == pr.in) s->st_in[k] = (int)pi;
                for (size_t k = 0; k < spec.outs.size(); k++)
                    if (spec.outs[k].name == pr.out) s->st_out[k] = (int)pi;
            }
            s->pool_devs.clear();
            s->pool_rows.clear();
            for (auto& p : st_pools_) {
                s->pool_devs.push_back(p.dev);
                s->pool_rows.push_back(p.rows);
            }
            s->has_state = true;
            std::fprintf(stderr, "[trt] 状态池生效: %zu 对 × %d 行"
                         "（提交侧 D2D 填充+批尾 D2D 散射，状态不过主机）\n",
                         st_pools_.size(), cfg.state_pool_rows);
        }
        if (for_bank) st_streams_.push_back(s->stream);   // ResetStatePool 全流面
        return s;
    }

    bool Warmup(void* session) override {
        TrtSession* s = (TrtSession*)session;
        if (g_cu.SetDevice) g_cu.SetDevice(s->dev);
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
                if (!MbSubmit(s, s->slots, seq)) { s->mb_ok = false; return false; }
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
        if (g_cu.SetDevice) g_cu.SetDevice(s->dev);
        // 状态会话只捕计算图（无 4 段图）——计算图同为准入对象
        if (!s->graph_ok && !s->graph_c_ok) {
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
            if (!s->graph_ok) {   // 状态会话：计算图直发+同步回拷（无 pids，
                v.clear();        // 尾段走同步面——探针只验图回放地址不烧死）
                if (g_cu.GraphLaunch(s->graph_c, s->stream)) return;
                g_cu.StreamSynchronize(s->stream);
                g_cu.Memcpy(s->out_h_arena, s->out_d_arena, s->out_h_bytes, 2);
                snap(v);
                return;
            }
            unsigned seq = 0;
            double spin = 0;
            if (!MbSubmit(s, s->slots, seq) || !WaitFlag(s, seq, 5000.0, &spin)) {
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
        for (size_t i = 0; i < st_streams_.size(); i++)   // ③流表摘除（防悬挂——
            if (st_streams_[i] == s->stream) {            // ResetStatePool 全流面）
                st_streams_.erase(st_streams_.begin() + (long)i);
                break;
            }
        if (s->graph && g_cu.GraphDestroy) g_cu.GraphDestroy(s->graph);
        if (s->graph_c && g_cu.GraphDestroy) g_cu.GraphDestroy(s->graph_c);
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

    // 前缀 h2d（n > 7/8·slots 走整块；尾行旧数据=行独立无害）+ 异步发射；
    // ③状态会话：状态输入行改设备池 D2D 填充（H2D 跳过，恒逐输入路径——
    // 聚合分支会整块 H2D 状态行）
    bool SubmitBatch(void* session, int n_rows, unsigned& seq_out) override {
        TrtSession* s = (TrtSession*)session;
        if (g_cu.SetDevice) g_cu.SetDevice(s->dev);   // 多卡守卫
        if (n_rows > s->slots) n_rows = s->slots;
        s->last_n = n_rows;
        const bool st = s->has_state && s->st_pids;
        if (!st && n_rows > (s->slots * 7) / 8) {
            if (g_cu.MemcpyAsync(s->in_d_arena, s->in_h_arena, s->in_h_bytes, 1, s->stream))
                return false;
        } else {
            for (size_t i = 0; i < s->ins.size(); i++) {
                const int pi = st ? (i < s->st_in.size() ? s->st_in[i] : -1) : -1;
                if (pi >= 0) {   // 状态输入行：池行→输入行（D2D，链粘滞行集）
                    char* dst = (char*)s->ins[i].dev;
                    char* pool = (char*)s->pool_devs[(size_t)pi];
                    const size_t rb = s->ins[i].meta.row_bytes;
                    const StatePool& pol = st_pools_[(size_t)pi];
                    const size_t prows = (size_t)pol.rows;
                    const bool batch = StBatchOn();
                    for (int r = 0; r < n_rows; r++) {
                        const int pid = s->st_pids[r].load(std::memory_order_relaxed);
                        if (pid < 0 || (size_t)pid >= prows)
                            continue;   // 幻影行（轮转归 -1：cursor 虚增未领
                                        // 号）——行内容垃圾无害，填充跳过
                        // DATA8 竞态修复：pending 行读池末保留零行（NewGame
                        // 语义=状态归零；散射随批覆写）——消费即清旗
                        const char* srcp = pool + (size_t)pid * rb;
                        if (pol.zero_pending[pid].load(std::memory_order_acquire)) {
                            srcp = pool + prows * rb;   // 保留零行
                            pol.zero_pending[pid].store(0, std::memory_order_relaxed);
                        }
                        if (batch) {
                            s->sb.dst.push_back(dst + (size_t)r * rb);
                            s->sb.src.push_back((void*)srcp);
                            s->sb.sz.push_back(rb);
                            s->sb.ai.push_back(0);
                        } else if (g_cu.MemcpyAsync(dst + (size_t)r * rb, srcp, rb,
                                                    3 /*D2D*/, s->stream)) {
                            return false;
                        }
                    }
                    continue;
                }
                if (g_cu.MemcpyAsync(s->ins[i].dev, s->ins[i].host,
                                     (size_t)n_rows * s->ins[i].meta.row_bytes,
                                     1, s->stream))
                    return false;
            }
        }
        if (!StBatchFlush(s)) return false;   // 状态填充批：单次提交
        return MbSubmit(s, n_rows, seq_out);
    }

    bool CompletionReached(void* session, unsigned seq) override {
        TrtSession* s = (TrtSession*)session;
        if (!s->mb_ok) {   // 降级流同步路径：发射即等完（SubmitBatch 已同步）
            if (g_cu.SetDevice) g_cu.SetDevice(s->dev);
            g_cu.DeviceSynchronize();
            if (PartialD2hOn(s, s->last_n)) {
                for (const auto& o : s->outs)
                    g_cu.Memcpy(o.host, o.dev,
                                (size_t)s->last_n * (size_t)o.meta.width * sizeof(float), 2);
            } else {
                g_cu.Memcpy(s->out_h_arena, s->out_d_arena, s->out_h_bytes, 2);
            }
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
        if (g_cu.SetDevice) g_cu.SetDevice(dev_id_);
        return ApplyRefitWeights(g_trt_eng.eng, rw1_path);
    }

private:
    int dev_id_ = 0;   // 实例设备（LoadSpec 落定；多卡守卫用）
    // ③状态池（backend 实例级——组内银行共享设备池；链→组钉扎=无跨组状态。
    // 首个带 state_pairs 的会话创建时分配一次，零基一次；析构释放）
    struct StatePool {
        void* dev = nullptr;      // 设备池 [(rows+1) × row_bytes]——末行=保留
                                  // 零行（init 清一次永不散射：pid<rows 恒真）
        size_t row_bytes = 0;
        int rows = 0;             // 逻辑行数（pool 行下标域）
        std::atomic<char>* zero_pending = nullptr;   // [rows] 行待清零旗
                                  // （NewGame 置 1=填充改读零行；DATA8 竞态修）
    };
    std::vector<StatePool> st_pools_;
    std::vector<void*> st_streams_;   // 全部银行会话流（ResetStatePool 全流
                                      // memset=任意下一读所在流自有序；同零值
                                      // 多流写良性）

public:
    ~TrtBackend() {
        for (auto& p : st_pools_) {
            if (p.dev && g_cu.Free) g_cu.Free(p.dev);
            delete[] p.zero_pending;
        }
    }
    // ③池行清零（NewGame）：链串行⇒该行无并发读者/写者；每条流各 memset
    // 一次⇒该行在任意银行的后续读（自流有序）之前完成
    // NewGame 池行清零（DATA8 竞态修复）：旧实现=全部会话流 memsetAsync——
    // 跨流无序，某流迟到 memset 会在新局首散射之后执行=清掉新状态（状态
    // 丢失→轨迹翻转；单银行全序安全/hands 高频 NewGame 放大窗口=DATA8 的
    // A/B/C 矩阵全解释）。修=延迟零行（设计文档 §5-2 备案方案）：置
    // pending 旗（无 CUDA 调用——worker 线程不再碰驱动），该行下次填充改
    // 读池末保留零行（散射随批覆写新态=语义等价旧 memset 且流序天然正确）
    bool ResetStatePool(int row) override {
        if (st_pools_.empty()) return false;
        for (auto& p : st_pools_) {
            if (row < 0 || row >= p.rows) return false;
            p.zero_pending[row].store(1, std::memory_order_release);
        }
        return true;
    }
    bool BindStatePids(void* session, const std::atomic<int>* pids) override {
        ((TrtSession*)session)->st_pids = pids;
        return true;
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
        if (sz < 0) {
            fclose(f);
            return false;
        }
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

    // 部分行输出 D2H（掼蛋线 §4-② 回传）：状态化/大输出模型每批全量回拷
    // out_arena（MB 级，与行数无关）是设计税——各输出段内行连续，逐输出只
    // 拷前 n_rows 行等价正确（收割只读本批行，尾行旧数据=行独立无害）。
    // 档位自动门控：仅全量回拷 ≥256KB 且非满批时启用（小输出模型走图外
    // 逐段拷的每段提交税 WDDM 5-10µs 大于省下的字节；满批走原四段图零扰动）。
    // FARM_D2H_PARTIAL=0 杀手锏回全量（A/B 口径）。
    static constexpr size_t kD2hPartialMinBytes = 256 * 1024;
    static bool PartialD2hOn(const TrtSession* s, int n_rows) {
        static const int mode = [] {
            const char* e = std::getenv("FARM_D2H_PARTIAL");
            return e ? std::atoi(e) : 1;
        }();
        if (mode <= 0 || n_rows <= 0 || n_rows >= s->slots) return false;
        return s->out_h_bytes >= kD2hPartialMinBytes;
    }
    // ③批量 D2D 档（FARM_STATE_D2D_BATCH=0 杀手锏回逐行=A/B 口径；符号
    // 缺席自动回逐行——批 API 为 CUDA12.8+ 可选符号）
    static bool StBatchOn() {
        static const int mode = [] {
            const char* e = std::getenv("FARM_STATE_D2D_BATCH");
            return e ? std::atoi(e) : 1;
        }();
        return mode > 0 && g_cu.MemcpyBatchOk();
    }
    // flush 批 scratch（单 attr 复用全条目；失败=scratch 清空后 false）
    static bool StBatchFlush(TrtSession* s) {
        if (s->sb.empty()) return true;
        HbAttr attr{};
        attr.srcAccessOrder = 0x3;
        int rc = g_cu.MemcpyBatchAsyncV(s->sb.dst.data(), s->sb.src.data(),
                                       s->sb.sz.data(), s->sb.dst.size(),
                                       &attr, s->sb.ai.data(), 1, s->stream);
        s->sb.clear();
        return rc == 0;
    }
    // 图外逐输出前缀 D2H + 盖章殿后（盖章在 D2H 之后=旗标到即输出驻留，
    // 流序契约与图内四段完全一致）
    static bool EnqueuePartialD2H(TrtSession* s, int n_rows) {
        for (const auto& o : s->outs)
            if (g_cu.MemcpyAsync(o.host, o.dev, (size_t)n_rows * (size_t)o.meta.width * sizeof(float),
                                 2, s->stream))
                return false;
        if (g_cu.MemcpyAsync(s->mb_flag_dev, s->mb_seq_dev, 4, 2, s->stream))
            return false;
        return true;
    }

    // ③状态尾段（状态会话恒图外；pids 未绑=Init 期冒烟走旧路）：状态输出行
    // D2D 散射回池（不过主机）+非状态输出前缀 D2H+盖章殿后（流序契约不变：
    // 旗标到=散射与 D2H 均已执行）
    static bool EnqueueStateTail(TrtSession* s, int n_rows) {
        s->sb.clear();   // 防御：早退不留陈旧条目（填充批已在 Submit 清空）
        const bool batch = StBatchOn();
        for (size_t oi = 0; oi < s->outs.size(); oi++) {
            const int pi = oi < s->st_out.size() ? s->st_out[oi] : -1;
            const size_t rb = (size_t)s->outs[oi].meta.width * sizeof(float);
            if (pi >= 0) {
                char* pool = (char*)s->pool_devs[(size_t)pi];
                const char* src = (const char*)s->outs[oi].dev;
                const size_t prows = (size_t)s->pool_rows[(size_t)pi];
                for (int r = 0; r < n_rows; r++) {
                    const int pid = s->st_pids[r].load(std::memory_order_relaxed);
                    if (pid < 0 || (size_t)pid >= prows)
                        continue;   // 幻影行：不散射（陈旧 pid 会把垃圾写进
                                    // 他链池行=状态投毒；与填充侧同卫）
                    if (batch) {
                        s->sb.dst.push_back(pool + (size_t)pid * rb);
                        s->sb.src.push_back((void*)(src + (size_t)r * rb));
                        s->sb.sz.push_back(rb);
                        s->sb.ai.push_back(0);
                    } else if (g_cu.MemcpyAsync(pool + (size_t)pid * rb,
                                                src + (size_t)r * rb, rb, 3,
                                                s->stream)) {
                        return false;
                    }
                }
            } else if (g_cu.MemcpyAsync(s->outs[oi].host, s->outs[oi].dev,
                                        (size_t)n_rows * rb, 2, s->stream)) {
                return false;
            }
        }
        if (!StBatchFlush(s)) return false;   // 散射批：单次提交
        if (g_cu.MemcpyAsync(s->mb_flag_dev, s->mb_seq_dev, 4, 2, s->stream))
            return false;
        return true;
    }

    // 邮箱提交：序号 +1 写 staging → [图=graphLaunch（H2D 节点执行时读 staging
    // 当前值）] / [在线=4B H2D→enqueueV3→输出 D2H→盖章 逐个入队 stream]；
    // 大输出模型非满批 → 计算图/在线 enqueue + 图外部分行 D2H；
    // ③状态会话（pids 已绑）→ 恒计算图/在线 + 图外状态尾段
    static bool MbSubmit(TrtSession* s, int n_rows, unsigned& seq_out) {
        seq_out = ++s->mb_seq;
        const bool st = s->has_state && s->st_pids;
        if (!s->mb_ok) {   // 流同步降级：发射即等完（提交税同步税都在）
            if (!s->ctx->enqueueV3((cudaStream_t)s->stream)) return false;
            g_cu.StreamSynchronize(s->stream);
            if (st) {   // 状态尾段（同步版）：状态行 D2D 散射+非状态前缀 D2H
                for (size_t oi = 0; oi < s->outs.size(); oi++) {
                    const int pi = oi < s->st_out.size() ? s->st_out[oi] : -1;
                    const size_t rb = (size_t)s->outs[oi].meta.width * sizeof(float);
                    if (pi >= 0) {
                        char* pool = (char*)s->pool_devs[(size_t)pi];
                        const char* src = (const char*)s->outs[oi].dev;
                        const size_t prows = (size_t)s->pool_rows[(size_t)pi];
                        for (int r = 0; r < n_rows; r++) {
                            const int pid = s->st_pids[r].load(
                                std::memory_order_relaxed);
                            if (pid < 0 || (size_t)pid >= prows) continue;
                            g_cu.Memcpy(pool + (size_t)pid * rb,
                                        src + (size_t)r * rb, rb, 3);
                        }
                    } else {
                        g_cu.Memcpy(s->outs[oi].host, s->outs[oi].dev,
                                    (size_t)n_rows * rb, 2);
                    }
                }
                return true;
            }
            if (PartialD2hOn(s, n_rows)) {
                for (const auto& o : s->outs)
                    g_cu.Memcpy(o.host, o.dev,
                                (size_t)n_rows * (size_t)o.meta.width * sizeof(float), 2);
            } else {
                g_cu.Memcpy(s->out_h_arena, s->out_d_arena, s->out_h_bytes, 2);
            }
            return true;
        }
        *(volatile unsigned*)((char*)s->mb_host + 64) = seq_out;
        if (st) {
            if (s->graph_c_ok && s->graph_c) {
                if (g_cu.GraphLaunch(s->graph_c, s->stream)) return false;
            } else {
                if (g_cu.MemcpyAsync(s->mb_seq_dev, (char*)s->mb_host + 64, 4,
                                     1, s->stream))
                    return false;
                if (!s->ctx->enqueueV3((cudaStream_t)s->stream)) return false;
            }
            return EnqueueStateTail(s, n_rows);
        }
        if (PartialD2hOn(s, n_rows)) {
            if (s->graph_c_ok && s->graph_c) {
                if (g_cu.GraphLaunch(s->graph_c, s->stream)) return false;
            } else {
                if (g_cu.MemcpyAsync(s->mb_seq_dev, (char*)s->mb_host + 64, 4,
                                     1, s->stream))
                    return false;
                if (!s->ctx->enqueueV3((cudaStream_t)s->stream)) return false;
            }
            return EnqueuePartialD2H(s, n_rows);
        }
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

    // CUDA Graph 批捕获：四段图 [4B seq H2D→enqueueV3→输出 D2H→盖章]（满批/
    // 小输出的正路）+ 计算图 [4B seq H2D→enqueueV3]（大输出模型部分行 D2H 档
    // 的前缀图——D2H/盖章由 MbSubmit 图外按 n_rows 动态入队，图内静态形状拷
    // 不了"前 n 行"）
    static void CaptureGraph(TrtSession* s) {
        if (!s->mb_ok || !g_cu.StreamBeginCapture || !g_cu.StreamEndCapture
            || !g_cu.GraphInstantiate || !g_cu.GraphLaunch || !g_cu.GraphDestroy) {
            std::fprintf(stderr, "[trt] 图捕获前置不满足（邮箱/符号缺失）——降级仅邮箱\n");
            return;
        }
        // 注：staging 不在捕获期写——H2D 节点在**执行时**读其当前值，MbSubmit
        // 每次发射前写好（in-flight≤1=无覆写竞争）。
        auto capture_one = [&](bool with_copy, void** gout) -> bool {
            if (g_cu.StreamBeginCapture(s->stream, 0 /*Global*/) != 0) return false;
            bool ok = true;
            if (g_cu.MemcpyAsync(s->mb_seq_dev, (char*)s->mb_host + 64, 4,
                                 1, s->stream))
                ok = false;
            if (ok && !s->ctx->enqueueV3((cudaStream_t)s->stream)) ok = false;
            if (ok && with_copy
                && g_cu.MemcpyAsync(s->out_h_arena, s->out_d_arena, s->out_h_bytes,
                                    2, s->stream))
                ok = false;
            if (ok && with_copy
                && g_cu.MemcpyAsync(s->mb_flag_dev, s->mb_seq_dev, 4, 2, s->stream))
                ok = false;
            void* g = nullptr;
            int ec = g_cu.StreamEndCapture(s->stream, &g);
            if (!ok || ec != 0 || !g) {
                if (g) g_cu.GraphDestroy(g);
                return false;
            }
            void* ge = nullptr;
            if (g_cu.GraphInstantiate(&ge, g, 0) != 0 || !ge) {
                g_cu.GraphDestroy(g);
                return false;
            }
            g_cu.GraphDestroy(g);
            *gout = ge;
            return true;
        };
        if (s->has_state) {
            // ③状态会话：图内静态拷贝装不下逐批动态行集（散射/D2H 行集随批
            // 变化）——恒计算图+图外状态尾段（MbSubmit 的 st 分支）
            if (capture_one(false, &s->graph_c)) s->graph_c_ok = true;
            else std::fprintf(stderr, "[trt] 计算图捕获失败——状态尾段走在线\n");
        } else {
        if (!capture_one(true, &s->graph)) {
            std::fprintf(stderr, "[trt] CUDA Graph 捕获失败——降级仅邮箱\n");
            return;
        }
        s->graph_ok = true;
        // 计算图只在大输出（部分行档潜在用户）才捕；失败=部分行档回在线路径
        if (s->out_h_bytes >= kD2hPartialMinBytes) {
            if (capture_one(false, &s->graph_c)) s->graph_c_ok = true;
            else std::fprintf(stderr, "[trt] 计算图捕获失败——部分行 D2H 档走在线\n");
        }
        }
        unsigned seq = 0;
        double spin = 0;
        if (!MbSubmit(s, s->slots, seq) || !WaitFlag(s, seq, 5000.0, &spin)) {
            std::fprintf(stderr, "[trt] 图验证发射/旗标超时——降级仅邮箱\n");
            g_cu.GraphDestroy(s->graph); s->graph = nullptr;
            s->graph_ok = false;
            if (s->graph_c) { g_cu.GraphDestroy(s->graph_c); s->graph_c = nullptr; }
            s->graph_c_ok = false;
            return;
        }
        std::fprintf(stderr, "[trt] CUDA Graph 批捕获成功（验证自旋 %.3fms%s）\n",
                     spin, s->graph_c_ok ? "，含计算图（部分行 D2H 档）" : "");
    }
};

InferBackend* CreateTrtBackend() { return new TrtBackend(); }

#endif // INFERFARM_WITH_TRT

} // namespace inferfarm
