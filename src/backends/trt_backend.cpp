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
#include "inferfarm/state_touch.h"
#include "cudart_dyn.h"
#include "state_gather.h"
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
#include <deque>
#include <map>
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
    nvinfer1::ICudaEngine* eng = nullptr;  // 一份权重；多会话各建 context。
                                          // 生命周期=进程（不释放——会话/context
                                          // 引用其设备权重；多引擎常驻=④/双模型
                                          // 的显存代价由调用方控制引擎数）
};
// W1 一进程多引擎（2026-09-30）：单例→按路径缓存。E/D 分频与双模型共根。
// runtime 全进程一份（create/deserialize 线程安全）；引擎指针归 backend 实例
// （LoadSpec 定 path，RefitWeights 随实例——多组各持不同引擎互不串）。
static nvinfer1::IRuntime* g_trt_rt_shared = nullptr;
static std::mutex g_engs_mx;
// deque（非 vector）：EnsureEngine 持有元素裸指针（eng_ = &back()），第二个
// 不同路径引擎 push_back 时 vector 会整体搬移=先注册实例的 eng_ 悬垂
// （W4 双模型两组两引擎首爆：组0 CreateSession SIGSEGV）。deque 的
// push_back 不失效既有元素指针（W5 双引擎门只跑单组农场，未覆盖此序）。
static std::deque<TrtEngineCache>* g_engs_ptr = nullptr;

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
            // 名字归一（掼蛋 API 直建图，2026-09-30）：TRT 网络 API 的
            // add_constant 在 REFIT 模式下成为可换权重，引擎名单名=层名+
            // " CONSTANT"（角色限定）；onnx 导入流=裸名。RW1 统一写裸名
            //（state_dict 键），此处先裸名后补后缀重试——两流同门。
            std::string eff = e.name;
            if (!known(eff.c_str())) {
                std::string with_role = e.name + " CONSTANT";
                if (known(with_role.c_str())) eff = with_role;
                else {
                    std::fprintf(stderr, "[trt] 名单外跳过 %s\n", e.name.c_str());
                    n_skip++;
                    continue;
                }
            }
            // 原型校验（TRT 10.x：一参版返回 Weights，dtype/numel 都在里面）
            nvinfer1::Weights proto = ref->getWeightsPrototype(eff.c_str());
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
            if (!ref->setNamedWeights(eff.c_str(), w)) {
                std::fprintf(stderr, "[trt] setNamedWeights(%s) 失败（dtype/numel 与引擎"
                             "原型不一致？numel=%u）\n", eff.c_str(), e.numel);
                failed = true;
                break;
            }
            n_set++;
        }
    }
    if (!failed && n_set == 0 && !ents.empty()) {
        // 换心空转=名字契约失配（RW1 项裸名与"+ CONSTANT"后缀均未命中名单）
        // ——静默返回成功是陷阱，掼蛋 DATA11 §2 实测：子代与父代逐决策全同
        // 才暴露。fail fast 拒当成功。
        std::fprintf(stderr, "[trt] refit 空转拒载：RW1 %d 项全部名单外"
                     "——名字契约失配，请核对引擎名单/RW1 名字\n",
                     (int)ents.size());
        failed = true;
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
    // 僵死诊断（2026-10-01 DATA20）：最近一次发射失败/看门狗探针快照
    char last_diag[192] = {0};
    // 排水拆解（09-29 飞行窗）：ev_a=发射前（h2d 入队后）、ev_b=发射后（盖章
    // 后=流尾）——EventElapsedTime=GPU 侧真实串行时长；与收割观测的飞行窗墙钟
    // 之差=观测/提交延迟。事件在旗标到（完成跃迁）时结算。
    void* ev_a = nullptr;
    void* ev_b = nullptr;
    void* ev_c = nullptr;   // 检出延迟仪器（DATA18 观测差拆解）：检测点回记
                            // 事件——EventElapsedTime(ev_b,ev_c)=本批 GPU 完
                            // 成→CPU 检出的墙钟差（设备钟），观测差减它=残差
                            // 归 ev_a 起跑延迟（流队列）。事件缺席=分段静默。
    bool ev_pending = false;
    double ev_t_launch = 0;
    long long fl_n = 0;
    double fl_gpu_ms = 0, fl_wall_ms = 0, fl_det_ms = 0, h2d_ms = 0, lch_ms = 0;
    // CUDA Graph 批捕获
    void* graph = nullptr;
    bool graph_ok = false;
    void* graph_c = nullptr;   // 计算图（seq H2D→enqueueV3，无回拷/盖章）：
    bool graph_c_ok = false;   // 部分行 D2H 档的前缀图（大输出模型专用）
    // ③成对状态行（设备常驻池）：本会话解析结果（池指针拷贝自 backend 级
    // 分配——MbSubmit 是 static，经会话携带）
    bool has_state = false;              // 任一配对成立（恒图外尾段形态）
    // population 路由（判决16，镜像 ort_backend）：pop 面=整平面 dim0=P≠slots，
    // 不参与前缀拷——代际换权重后脏旗全量 H2D；mid 路由键批尾毒化（死行协议）
    bool pop_dirty = false;
    // pop carve 在输入 arena 中的段（满批整块 H2D 跳过它——pop 面归脏旗管，
    // 不然每批白付 P×flat_w 的 PCIe 税；pop_end=对齐后段尾，两段夹出非 pop 区）
    bool pop_mode = false;
    size_t pop_off = 0, pop_end = 0;
    std::vector<int> mid_like;   // 路由键输入下标（1-D i64 非 population/weight）
    std::vector<int> st_in, st_out;      // 输入/输出下标→池号（-1=非状态）
    // （③池表已上收 backend 级 st_pools_——ShareStatePool 接线后无需重建会话）
    const std::atomic<int>* st_pids = nullptr;   // 银行槽→池下标（Claim 写/
                                                  // 发车读；未绑=Init 期冒烟走旧路）
                                                  // 读侧 acquire（09-30 加固，
                                                  // 掼蛋竞态案同款）：与 Claim/
                                                  // Abandon 的 release 写逐点配
                                                  // 对=局部自洽，不再单押 drain
                                                  // 握手的跨文件 happens-before
    // ③批量 D2D scratch（填充/散射逐行小拷合并为单次 cudaMemcpyBatchAsync
    //——发车段 API 税∝提交次数；判决25 同机器。调度台线程独占=无锁）
    struct StBatch {
        std::vector<void*> dst, src;
        std::vector<size_t> sz, ai;
        void clear() { dst.clear(); src.clear(); sz.clear(); ai.clear(); }
        bool empty() const { return dst.empty(); }
    } sb;
    // ③gather 内核行号表（DATA13；FARM_STATE_GATHER=1）：主机建表→H2D→
    // 单 launch 搬全面（CE 逐行固定开销 ~6.4µs×行数 → 一次 launch）。每池
    // 各建一张（zero_pending 属池）；缓冲按 slots 一次分配逐批复用——复用
    // 安全由会话生命周期保证（收割=旗标殿后，复用时上一批 H2D 必已执行）。
    int* sg_tbl_h = nullptr;   // 主机 staging（3KB/批级）
    int* sg_tbl_d = nullptr;   // 设备端表
    int sg_cap = 0;            // 已分配容量
    int slots = 64;
    int dev = 0;      // 会话设备（多卡：分配/流/图/邮箱全落此设备）
    int last_n = 0;
    // 跨流序（DATA17 定谳：共享状态池+跨组半局换道=散射(组0流)与 gather/
    // scatter(组1流) 无序——单流 FIFO 拦不住跨流。每会话一枚状态事件，
    // 状态尾散射后 record；其它会话的 fill/scatter 入流前 wait 全体他席
    // 事件（同组 FIFO 情形事件早已完成=零代价；跨组情形=补上丢失的执行序）。
    void* ev_state = nullptr;
    // 本会话触池表（池 dev 指针，去重）：SgWaitOthers 按池粒度登记的依据。
    // owner 会话=CreateSession 映射期填；共享组会话=ShareStatePool 接线期填。
    std::vector<void*> touch_pools;
};

// 触池注册表（2026-10-01 上收公共面 include/inferfarm/state_touch.h——ort
// 镜像共用同一张表=跨后端池共享同钥匙）。

static void StTouchAddTrt(TrtSession* s, void* pool_dev) {
    StTouchAdd(pool_dev, s, s->ev_state);
}

// 会话按 st_in/st_out 映射登记触池（pools=本实例池表；映射值=池下标）
static void StTouchMap(TrtSession* s,
                       const std::vector<void*>& pool_devs) {
    for (int pi : s->st_in)
        if (pi >= 0 && pi < (int)pool_devs.size())
            StTouchAddTrt(s, pool_devs[(size_t)pi]);
    for (int pi : s->st_out)
        if (pi >= 0 && pi < (int)pool_devs.size()) {
            bool dup = false;
            for (int q : s->st_in)
                if (q == pi) { dup = true; break; }
            if (!dup) StTouchAddTrt(s, pool_devs[(size_t)pi]);
        }
}

class TrtBackend : public InferBackend {
public:
    const char* Name() const override { return "trt"; }
    bool LoadSpec(const ModelConfig& cfg, int slots, ModelSpec& out) override {
        if (!LoadTrtLib(cfg) || !g_cu.Load(DefaultCudaDir(cfg))) return false;
        dev_id_ = cfg.device_id;   // 多卡：engine 反序列化落定设备（同架构双卡
        if (g_cu.SetDevice) g_cu.SetDevice(dev_id_);   // 可共享 engine；>0 本机未测）
        if (!EnsureEngine(cfg)) return false;
        nvinfer1::ICudaEngine* eng = eng_->eng;
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
                m.population = !cfg.population_input.empty()
                               && nm == cfg.population_input;   // 路由模式标记
                if (m.population) {
                    if (m.et != DTYPE_F32) {
                        std::fprintf(stderr, "[trt] population 输入 %s 须 f32\n", nm);
                        return false;
                    }
                    // dim0=P（种群数）≠ slots 合法——整平面面，不参与前缀拷
                    if (m.dims[0] < 1 || m.dims.size() < 2) {
                        std::fprintf(stderr, "[trt] population 输入 %s 形状非法"
                                     "（须 [P≥1, flat_w] 二维）\n", nm);
                        return false;
                    }
                } else if ((int)m.dims[0] != slots) {
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
        if (!eng_ || !eng_->eng) return nullptr;
        nvinfer1::ICudaEngine* eng = eng_->eng;
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
        // 排水拆解事件（可选符号：缺席=拆账静默缺席，行为零变化）
        if (g_cu.EventCreateWithFlags && g_cu.EventRecord) {
            int ra = g_cu.EventCreateWithFlags(&s->ev_a, 0);
            int rb = g_cu.EventCreateWithFlags(&s->ev_b, 0);
            int rc2 = g_cu.EventCreateWithFlags(&s->ev_c, 0);
            if (ra || rb || rc2)
                std::fprintf(stderr, "[trt-flight] 事件创建失败 ra=%d rb=%d rc=%d"
                             "（拆账缺席）\n", ra, rb, rc2);
        } else {
            std::fprintf(stderr, "[trt-flight] 事件符号缺席（拆账缺席）\n");
        }
        const size_t kAlign = 256;
        // 输入单块 arena（256B 对齐 carve；IOBinding/setTensorAddress 绑 carve
        // 地址，地址终身固定）
        size_t off = 0;
        s->ins.resize(spec.ins.size());
        for (size_t i = 0; i < spec.ins.size(); i++) {
            s->ins[i].meta = spec.ins[i];
            // population 面总量=行宽×dim0（P≠slots；同族坑：按 slots 会写爆堆）
            off = (off + spec.ins[i].row_bytes
                       * (size_t)(spec.ins[i].population
                                      ? spec.ins[i].dims[0] : spec.slots)
                   + kAlign - 1) / kAlign * kAlign;
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
            size_t bytes = s->ins[i].meta.row_bytes
                * (size_t)(s->ins[i].meta.population
                               ? s->ins[i].meta.dims[0] : s->slots);
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
        // 路由键识别（population 模式，镜像 ort_backend）：1-D i64 非
        // population 输入=mid 类（批尾毒化目标；weight 面排除）
        if (!cfg.population_input.empty())
            for (size_t i = 0; i < s->ins.size(); i++) {
                if (s->ins[i].meta.population) {
                    s->pop_mode = true;
                    const size_t pb = s->ins[i].meta.row_bytes
                        * (size_t)s->ins[i].meta.dims[0];
                    s->pop_off = (size_t)((char*)s->ins[i].host
                                          - (char*)s->in_h_arena);
                    s->pop_end = s->pop_off
                        + (pb + kAlign - 1) / kAlign * kAlign;
                } else if (s->ins[i].meta.et == DTYPE_I64
                         && s->ins[i].meta.dims.size() == 1
                         && !s->ins[i].meta.weight)
                    s->mid_like.push_back((int)i);
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
            if (cfg.state_share_grp >= 0) {
                // ③共享组：不分配池（ShareStatePool 由 Farm 接线；绑定前发车=
                // SubmitBatch fail fast）。只做面映射（会话须知道哪些是状态面）
            } else if (st_pools_.empty()) {
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
            s->has_state = true;
            if (!st_pools_.empty()) {   // owner 路径：池已在（首会话刚建或复用）
                                        // ——按映射登记触池（跨组 wait 面成员）
                std::vector<void*> devs;
                devs.reserve(st_pools_.size());
                for (const auto& p : st_pools_) devs.push_back(p.dev);
                StTouchMap(s, devs);
            }
            if (cfg.state_share_grp >= 0)
                std::fprintf(stderr, "[trt] 状态池共享组就绪: %zu 对（待 Farm"
                             " 接线绑定持有组）\n", cfg.state_pairs.size());
            else
                std::fprintf(stderr, "[trt] 状态池生效: %zu 对 × %d 行"
                             "（提交侧 D2D 填充+批尾 D2D 散射，状态不过主机）\n",
                             cfg.state_pairs.size(), cfg.state_pool_rows);
        }
        if (for_bank) {
            st_streams_.push_back(s->stream);   // ResetStatePool 全流面
            // 跨流序事件（DATA17 定谳；可选符号缺席=退化旧行为，单组无恙）
            if (g_cu.EventCreateWithFlags && g_cu.EventRecord)
                g_cu.EventCreateWithFlags(&s->ev_state, 0x2 /*DisableTiming*/);
            // 触池表刷新（登记在 ev_state 创建之前=存了 nullptr，Add 对已存
            // 条目原地更新事件——wait 面必须见真事件）
            if (s->has_state)
                for (void* pd : s->touch_pools) StTouchAddTrt(s, pd);
            st_sessions_.push_back(s);
        }
        return s;
    }

    bool Warmup(void* session) override {
        TrtSession* s = (TrtSession*)session;
        if (g_cu.SetDevice) g_cu.SetDevice(s->dev);
        // Init 窗时序探针（2026-10-01 DATA20 附二：僵死收窄 Init 窗，7 段
        // 精确 302-305s=超时+重试成功）：分段一次性打点，进程级相对秒——
        // 僵死 attempt 的部分 stderr（宿主转储）据此直接读出停在哪段。
        static const auto t0 = std::chrono::steady_clock::now();
        auto initlog = [](const char* stage) {
            std::fprintf(stderr, "[trt][init] t=%.2fs %s\n",
                         std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count(),
                         stage);
            std::fflush(stderr);
        };
        initlog("热身起（3 跑 enqueueV3+同步）");
        memset(s->in_h_arena, 0, s->in_h_bytes);
        for (int r = 0; r < 3; r++) {
            if (g_cu.Memcpy(s->in_d_arena, s->in_h_arena, s->in_h_bytes, 1))
                return false;
            if (!s->ctx->enqueueV3((cudaStream_t)s->stream)) return false;
            if (g_cu.StreamSynchronize) g_cu.StreamSynchronize(s->stream);
            else g_cu.DeviceSynchronize();
        }
        initlog("3 跑完成，进图捕获");
        // 3 跑后图捕获（惰性分配已落定，捕获窗口内不容分配）；验证发射+自旋
        if (s->mb_ok) {
            CaptureGraph(s);
            initlog(s->graph_ok ? "批图捕获=开" : "批图捕获=败（回退在线邮箱）");
            if (!s->graph_ok) {   // 在线邮箱冒烟（链路坏=回退流同步，不让首批挂）
                unsigned seq = 0;
                if (!this->MbSubmit(s, s->slots, seq)) { s->mb_ok = false; return false; }
                if (!WaitFlag(s, seq, 5000.0)) { s->mb_ok = false; return true; }
            }
        }
        initlog(s->mb_ok ? (s->graph_ok ? "热身就绪（邮箱+批图）" : "热身就绪（邮箱）")
                         : "热身就绪（降级流同步）");
        std::fprintf(stderr, "[trt] 热身就绪 slots=%d%s%s\n", s->slots,
                     s->mb_ok ? "，邮箱=开" : "", s->graph_ok ? "，批图捕获=开" : "");
        return true;
    }

    // 图地址烧死小实验（bank_contract 关键工程点 1）：两图案可分辨 + 图回放
    // ==在线参考 + 换数据图输出跟着变（非烧死快照）。任一不过=false。
    bool ProbeGraph(void* session) override {
        TrtSession* s = (TrtSession*)session;
        if (g_cu.SetDevice) g_cu.SetDevice(s->dev);
        // Init 窗时序探针（同 Warmup，DATA20 附二）
        static const auto t0 = std::chrono::steady_clock::now();
        auto initlog = [](const char* stage) {
            std::fprintf(stderr, "[trt][init] t=%.2fs %s\n",
                         std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count(),
                         stage);
            std::fflush(stderr);
        };
        initlog("图探针起（两图案+回放逐位）");
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
            if (!this->MbSubmit(s, s->slots, seq) || !WaitFlag(s, seq, 5000.0, &spin)) {
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
        initlog(bok ? "图探针过" : "图探针 FAIL");
        return bok;
    }

    void DestroySession(void* session) override {
        TrtSession* s = (TrtSession*)session;
        if (!s) return;
        StTouchRemove(s);   // 触池登记摘除（wait 面防悬挂）
        for (size_t i = 0; i < st_streams_.size(); i++)   // ③流表摘除（防悬挂——
            if (st_streams_[i] == s->stream) {            // ResetStatePool 全流面）
                st_streams_.erase(st_streams_.begin() + (long)i);
                break;
            }
        for (size_t i = 0; i < st_sessions_.size(); i++)
            if (st_sessions_[i] == s) { st_sessions_.erase(st_sessions_.begin() + (long)i); break; }
        if (s->ev_state && g_cu.EventDestroy) g_cu.EventDestroy(s->ev_state);
        if (s->graph && g_cu.GraphDestroy) g_cu.GraphDestroy(s->graph);
        if (s->graph_c && g_cu.GraphDestroy) g_cu.GraphDestroy(s->graph_c);
        if (s->ctx) delete s->ctx;
        if (s->stream && g_cu.StreamDestroy) g_cu.StreamDestroy(s->stream);
        if (s->in_h_arena) g_cu.FreeHost(s->in_h_arena);
        if (s->in_d_arena) g_cu.Free(s->in_d_arena);
        if (s->out_h_arena) g_cu.FreeHost(s->out_h_arena);
        if (s->out_d_arena) g_cu.Free(s->out_d_arena);
        if (s->mb_host) g_cu.FreeHost(s->mb_host);
        if (s->ev_a && g_cu.EventDestroy) g_cu.EventDestroy(s->ev_a);
        if (s->ev_b && g_cu.EventDestroy) g_cu.EventDestroy(s->ev_b);
        if (s->ev_c && g_cu.EventDestroy) g_cu.EventDestroy(s->ev_c);
        if (s->mb_seq_dev) g_cu.Free(s->mb_seq_dev);
        if (s->sg_tbl_d) g_cu.Free(s->sg_tbl_d);
        delete[] s->sg_tbl_h;
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

    // population 面写入（演化路由，判决16）：宿主 arena 落盘 + 置脏旗
    //（下次 SubmitBatch 全量 H2D；流序先于发射=与在飞批无竞态）
    bool SetPopulation(void* session, const char* pop_input, const void* host) override {
        TrtSession* s = (TrtSession*)session;
        for (size_t i = 0; i < s->ins.size(); i++)
            if (s->ins[i].meta.population
                && s->ins[i].meta.name == pop_input) {
                memcpy(s->ins[i].host, host,
                       s->ins[i].meta.row_bytes
                           * (size_t)s->ins[i].meta.dims[0]);
                s->pop_dirty = true;
                return true;
            }
        return false;
    }

    // 前缀 h2d（n > 7/8·slots 走整块；尾行旧数据=行独立无害）+ 异步发射；
    // ③状态会话：状态输入行改设备池 D2D 填充（H2D 跳过，恒逐输入路径——
    // 聚合分支会整块 H2D 状态行）
    bool SubmitBatch(void* session, int n_rows, unsigned& seq_out) override {
        TrtSession* s0 = (TrtSession*)session;
        if (!SubmitBatchImpl(s0, n_rows, seq_out)) {
            // DATA20：发射失败不再裸奔——当下探针（错误码/流状态/旗标）随
            // 批异常一起打印（cudaGetLastError 会清错，此处消费=诊断专用）
            char db[192];
            ProbeDiag(s0, -1, db, (int)sizeof db);
            std::fprintf(stderr, "[trt][发射诊断] %s\n", db);
            std::fflush(stderr);
            return false;
        }
        return true;
    }
    // 当下探针：expect<0=读会话已发 seq。写 s->last_diag 供 bank 看门狗转印。
    void ProbeDiag(TrtSession* s, long long expect, char* buf, int cap) {
        if (!buf || cap <= 0) return;
        char errbuf[32] = "-";
        if (g_cu.GetLastError)
            std::snprintf(errbuf, sizeof errbuf, "%d", g_cu.GetLastError());
        const int sq = g_cu.StreamQuery ? g_cu.StreamQuery(s->stream) : -999;
        const unsigned flag = s->mb_host ? *(volatile unsigned*)s->mb_host : 0;
        // ev_b=本批流尾事件（发射时回记）——NotReady=GPU 真未完成（图回放
        // 段卡死）；Success=GPU 已完而旗标未达=盖章/D2H 段问题
        const int evb = (s->ev_b && g_cu.EventQuery)
                            ? g_cu.EventQuery(s->ev_b) : -999;
        const unsigned exp = expect >= 0 ? (unsigned)expect : s->mb_seq;
        std::snprintf(buf, (size_t)cap,
                      "cuda_err=%s stream_qry=%d ev_b_qry=%d flag=%u seq=%u",
                      errbuf, sq, evb, flag, exp);
        std::snprintf(s->last_diag, sizeof s->last_diag, "%s", buf);
    }
    bool SubmitBatchImpl(TrtSession* s, int n_rows, unsigned& seq_out) {
        if (g_cu.SetDevice) g_cu.SetDevice(s->dev);   // 多卡守卫
        if (n_rows > s->slots) n_rows = s->slots;
        s->last_n = n_rows;
        const bool st = s->has_state && s->st_pids;
        bool st_fill_waited = false;   // 跨流序 wait 每批一次
        if (s->has_state && st_pools_.empty()) {   // ③共享组未接线（Farm 配置
            // 错误/序错）——池行指针不存在，静默旧路=状态面垃圾，loud 快败
            std::fprintf(stderr, "[trt] state_pairs 会话未绑定状态池（跨组共享"
                         "未接线？）——拒发车\n");
            return false;
        }
        const double tw0 = s->ev_a ? std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count() : 0;
        // （旧 ev_a 流头记录点已撤——事件差口径改"引擎+散射纯 GPU 段"，记录
        // 点移至填充入流后/发射前，见 SubmitBatch 尾段注释）
        // population 脏旗（判决16）：代际换权重后的单次全量 H2D（pop 非每槽
        // 输入，不参与前缀拷；同流序先于发射=正确性由流序保证）
        if (s->pop_dirty) {
            for (size_t i = 0; i < s->ins.size(); i++)
                if (s->ins[i].meta.population
                    && g_cu.MemcpyAsync(s->ins[i].dev, s->ins[i].host,
                                        s->ins[i].meta.row_bytes
                                            * (size_t)s->ins[i].meta.dims[0],
                                        1, s->stream))
                    return false;
            s->pop_dirty = false;
        }
        // 批尾毒化（路由模式死行协议，判决16）：未领槽位 [n, slots) 的路由键
        // 置 -1（0xFF）——陈旧 mid 参与图内散射会破坏 (p,j) 唯一性。图侧把
        // -1 路由到专属死块，毒行输出不被收割。整块路径全量拷贝自带尾段。
        if (s->pop_mode && n_rows < s->slots)
            for (int mi : s->mid_like) {
                const TrtIn& m = s->ins[(size_t)mi];
                memset((char*)m.host + (size_t)n_rows * m.meta.row_bytes, 0xFF,
                       (size_t)(s->slots - n_rows) * m.meta.row_bytes);
            }
        if (!st && n_rows > (s->slots * 7) / 8) {
            if (!s->pop_mode) {
                if (g_cu.MemcpyAsync(s->in_d_arena, s->in_h_arena,
                                     s->in_h_bytes, 1, s->stream))
                    return false;
            } else {
                // pop 面整平面不走前缀（脏旗管）——两段夹出非 pop 区，
                // 满批免搬 P×flat_w（YGO 5.3MB/批的 PCIe 税）
                if (s->pop_off > 0
                    && g_cu.MemcpyAsync(s->in_d_arena, s->in_h_arena,
                                        s->pop_off, 1, s->stream))
                    return false;
                if (s->pop_end < s->in_h_bytes
                    && g_cu.MemcpyAsync((char*)s->in_d_arena + s->pop_end,
                                        (char*)s->in_h_arena + s->pop_end,
                                        s->in_h_bytes - s->pop_end,
                                        1, s->stream))
                    return false;
            }
        } else {
            for (size_t i = 0; i < s->ins.size(); i++) {
                const int pi = st ? (i < s->st_in.size() ? s->st_in[i] : -1) : -1;
                if (pi >= 0) {   // 状态输入行：池行→输入行（D2D，链粘滞行集）
                    char* dst = (char*)s->ins[i].dev;
                    char* pool = (char*)st_pools_[(size_t)pi].dev;
                    const size_t rb = s->ins[i].meta.row_bytes;
                    const StatePool& pol = st_pools_[(size_t)pi];
                    const size_t prows = (size_t)pol.rows;
                    if (!st_fill_waited) {   // 本批一次（多状态面共享同序）
                        st_fill_waited = true;
                        if (!SgWaitOthers(s)) return false;
                    }
                    if (StGatherOn()) {   // DATA13：单 launch gather 全面
                        if (!SgRun(s, dst, pool, pol, /*zero_ok=*/true, n_rows,
                                   rb, 0))
                            return false;
                        continue;
                    }
                    const bool batch = StBatchOn();
                    for (int r = 0; r < n_rows; r++) {
                        const int pid = s->st_pids[r].load(std::memory_order_acquire);
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
                if (s->ins[i].meta.population) continue;   // 整平面面不走前缀（脏旗 H2D 已覆盖）
                if (g_cu.MemcpyAsync(s->ins[i].dev, s->ins[i].host,
                                     (size_t)n_rows * s->ins[i].meta.row_bytes,
                                     1, s->stream))
                    return false;
            }
            // 前缀路径补充：毒化后的路由键尾段同步到设备（整块路径全量拷贝已含）
            if (s->pop_mode && n_rows < s->slots)
                for (int mi : s->mid_like) {
                    const TrtIn& m = s->ins[(size_t)mi];
                    if (g_cu.MemcpyAsync((char*)m.dev
                                             + (size_t)n_rows * m.meta.row_bytes,
                                         (char*)m.host
                                             + (size_t)n_rows * m.meta.row_bytes,
                                         (size_t)(s->slots - n_rows)
                                             * m.meta.row_bytes,
                                         1, s->stream))
                        return false;
                }
        }
        if (!StBatchFlush(s)) return false;   // 状态填充批：单次提交
        // 排水拆解：ev_a=**填充入流后、发射前**、ev_b=流尾（盖章后）——事件
        // 差=引擎+散射纯 GPU 段（h2d/fill 排除——CPU-paced fill 会把等待灌进
        // 事件差=负观测差怪象根因，2026-10-01 移位定谳；引擎侧记账口径自此
        // 干净，旧数字含 fill 不可直比）。h2d 段仍由 CPU 墙钟单列（h2d_ms）。
        // 事件在旗标跃迁处结算。
        if (s->ev_a) {
            g_cu.EventRecord(s->ev_a, s->stream);
            const double th1 = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            const bool ok2 = MbSubmit(s, n_rows, seq_out);
            g_cu.EventRecord(s->ev_b, s->stream);
            const double tl1 = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            s->ev_pending = true;
            s->ev_t_launch = tl1;
            s->h2d_ms += th1 - tw0;
            s->lch_ms += tl1 - th1;
            return ok2;
        }
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
        if (*flag != seq) return false;
        // 排水拆解结算（完成跃迁）：GPU 侧串行时长（事件差）vs 飞行窗墙钟
        // （发射→观测）——差值=观测/收割延迟。每 1024 批打一行。
        if (s->ev_pending && g_cu.EventElapsedTime) {
            float gpu_ms = -1.f;
            int rc = g_cu.EventElapsedTime(&gpu_ms, s->ev_a, s->ev_b);
            if (rc == 0 && gpu_ms >= 0) {
                const double now = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                s->fl_n++;
                s->fl_gpu_ms += gpu_ms;
                s->fl_wall_ms += now - s->ev_t_launch;
                // 检出延迟分段（DATA18）：检测点在会话流回记 ev_c——
                // elapsed(ev_b,ev_c)=本批 GPU 完成→检测的设备钟差（本流空闲
                // 事件即执行≈墙钟）。观测差−检出=残差归 ev_a 起跑延迟（同流
                // FIFO/设备队列）。旗标到⇒ev_b 必已完成（盖章在 ev_b 前入流）。
                if (s->ev_c && g_cu.EventRecord) {
                    g_cu.EventRecord(s->ev_c, s->stream);
                    for (int spin = 0; spin < 10000; spin++) {
                        if (!g_cu.EventQuery
                            || g_cu.EventQuery(s->ev_c) == 0) break;
                        _mm_pause();
                    }
                    float det = -1.f;
                    if (g_cu.EventElapsedTime(&det, s->ev_b, s->ev_c) == 0
                        && det >= 0)
                        s->fl_det_ms += det;
                }
                if (s->fl_n % 128 == 0) {
                    std::printf("[trt-flight] n=%lld GPU侧串行=%.3fms 飞行墙=%.3fms "
                                "观测差=%.3fms 检出=%.3fms 起跑残差=%.3fms"
                                " | 发射前段: h2d入队=%.3fms 发射调用=%.3fms\n",
                                s->fl_n, s->fl_gpu_ms / s->fl_n,
                                s->fl_wall_ms / s->fl_n,
                                (s->fl_wall_ms - s->fl_gpu_ms) / s->fl_n,
                                s->fl_det_ms / s->fl_n,
                                (s->fl_wall_ms - s->fl_gpu_ms
                                 - s->fl_det_ms) / s->fl_n,
                                s->h2d_ms / s->fl_n, s->lch_ms / s->fl_n);
                    std::fflush(stdout);
                }
            } else {
                static std::atomic<int> once{0};
                if (once.fetch_add(1) == 0)
                    std::fprintf(stderr, "[trt-flight] 首次结算失败 rc=%d gpu_ms=%f"
                                 "（后续静默）\n", rc, gpu_ms);
            }
            s->ev_pending = false;
        }
        return true;
    }
    void CompletionFence() override { _mm_lfence(); }

    // bank 看门狗/批异常回调（2026-10-01 DATA20）：当下探针快照
    void DiagnoseSubmit(void* session, char* buf, int cap) override {
        ProbeDiag((TrtSession*)session, -1, buf, cap);
    }

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
        if (!eng_ || !eng_->eng) {
            std::fprintf(stderr, "[trt] engine 未载——换心不可用\n");
            return false;
        }
        if (g_cu.SetDevice) g_cu.SetDevice(dev_id_);
        if (!ApplyRefitWeights(eng_->eng, rw1_path)) return false;
        // 图失效+全量重捕（2026-10-01 僵死案根治刀）：批图在捕获时刻烧死
        // 内核参数（含权重设备地址），refit 原地改权重显存——TRT 内部
        // staging 重排/tactic 状态变化（候选离基座越远越易触发）时图参数
        // 指向旧地址/半更新状态，某次 GraphLaunch 永不成完（死锁形态=旗标
        // 停写+全员 futex）。重捕把新地址重新烧死=整类交互消灭；尾部自带
        // 发射冒烟（5s 界）——不过的会话毁图降级邮箱=响败不挂死。契约：
        // 换心只在腿间调用（无在飞批）=重捕窗口安全。
        for (TrtSession* rs : st_sessions_) {
            if (!rs->mb_ok) continue;   // 已降级会话无图可捕（在线路径天然新地址）
            if (rs->graph) { g_cu.GraphDestroy(rs->graph); rs->graph = nullptr; }
            if (rs->graph_c) { g_cu.GraphDestroy(rs->graph_c); rs->graph_c = nullptr; }
            rs->graph_ok = false;
            rs->graph_c_ok = false;
            if (g_cu.SetDevice) g_cu.SetDevice(rs->dev);
            CaptureGraph(rs);
        }
        if (g_cu.SetDevice) g_cu.SetDevice(dev_id_);
        return true;
    }

private:
    int dev_id_ = 0;   // 实例设备（LoadSpec 落定；多卡守卫用）
    // R10 判据钩子：按实例路径重扫缓存容器取当前元素地址（不读 eng_——
    // vector 搬移后它是悬垂值，读即 UB；重扫地址对比才确定性暴露"元素搬移"）
    const void* DebugEngineCookie() const override {
        if (!g_engs_ptr || eng_path_.empty()) return nullptr;
        std::lock_guard<std::mutex> lk(g_engs_mx);
        for (const auto& e : *g_engs_ptr)
            if (e.path == eng_path_ && e.eng) return (const void*)&e;
        return nullptr;
    }

    TrtEngineCache* eng_ = nullptr;   // W1：本实例引擎（EnsureEngine 落定；
    std::string eng_path_;            // R10 钩子面：实例路径独立副本（eng_ 悬垂
                                      // 时 path 不可从缓存元素读）
                                      // RefitWeights/会话建 context 随实例）
    // ③状态池（backend 实例级——组内银行共享设备池；链→组钉扎=无跨组状态。
    // 首个带 state_pairs 的会话创建时分配一次，零基一次；析构释放）
    struct StatePool {
        void* dev = nullptr;      // 设备池 [(rows+1) × row_bytes]——末行=保留
                                  // 零行（init 清一次永不散射：pid<rows 恒真）
        size_t row_bytes = 0;
        int rows = 0;             // 逻辑行数（pool 行下标域）
        std::atomic<char>* zero_pending = nullptr;   // [rows] 行待清零旗
                                  // （NewGame 置 1=填充改读零行；DATA8 竞态修）
        bool external = false;    // ③跨组共享（ShareStatePool 绑定）——析构不释放
    };
    std::vector<StatePool> st_pools_;
    std::vector<TrtSession*> st_sessions_;   // 状态会话表（跨流 wait 面）
    std::vector<void*> st_streams_;   // 全部银行会话流（ResetStatePool 全流
                                      // memset=任意下一读所在流自有序；同零值
                                      // 多流写良性）

public:
    ~TrtBackend() {
        for (auto& p : st_pools_) {
            if (p.dev && !p.external && g_cu.Free) g_cu.Free(p.dev);
            if (!p.external) delete[] p.zero_pending;
        }
    }
    // ③跨组共享池：持有组导出/共享组绑定（同 GPU 设备行直读；决策级组路由
    // W4 形态——单一正典状态=与主机路径语义逐位等价，docs/state-residency §7）
    int StatePoolInfo(SharedStatePool* out, int cap) override {
        if ((int)st_pools_.size() > cap) return -1;
        for (size_t i = 0; i < st_pools_.size(); i++) {
            out[i].dev = st_pools_[i].dev;
            out[i].zero_pending = st_pools_[i].zero_pending;
            out[i].row_bytes = st_pools_[i].row_bytes;
            out[i].rows = st_pools_[i].rows;
        }
        return (int)st_pools_.size();
    }
    bool ShareStatePool(const SharedStatePool* pools, int n) override {
        if (!st_pools_.empty()) return false;   // 已持有/已绑定=重复接线
        if (n <= 0) return false;
        for (int i = 0; i < n; i++) {
            StatePool p;
            p.dev = pools[i].dev;
            p.zero_pending = pools[i].zero_pending;
            p.row_bytes = pools[i].row_bytes;
            p.rows = pools[i].rows;
            p.external = true;
            st_pools_.push_back(p);
        }
        // 触池登记（跨组洞闭合）：共享组全部状态会话入 wait 面——他组散射
        // 事件此后对本组 gather/scatter 可见（此前跨组形态零序=DATA17 同源）
        if (!st_sessions_.empty()) {
            std::vector<void*> devs;
            devs.reserve(st_pools_.size());
            for (const auto& p : st_pools_) devs.push_back(p.dev);
            for (TrtSession* s : st_sessions_)
                if (s->has_state) StTouchMap(s, devs);
        }
        std::fprintf(stderr, "[trt] 状态池跨组共享生效: %d 对（绑定他组设备行，"
                     "本组不持有）\n", n);
        return true;
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
    bool EnsureEngine(const ModelConfig& cfg) {   // 实例方法：eng_ 落定
        // 已绑定本实例引擎且路径一致=幂等快路
        if (eng_ && eng_->eng
            && (cfg.engine_path.empty() || eng_->path == cfg.engine_path))
            return true;
        if (cfg.engine_path.empty()) {
            std::fprintf(stderr, "[trt] 缺 engine 路径\n");
            return false;
        }
        // ScheduleSpin：enqueueV3 异步返回后的设备等待恒忙等（Auto 策略会睡
        // 1-3ms/批——唤醒税）。须在首个 CUDA 调用（ctx 创建）前设（一次）。
        if (g_cu.GetDeviceFlags && g_cu.SetDeviceFlags && !g_trt_rt_shared) {
            unsigned fl = 0;
            if (g_cu.GetDeviceFlags(&fl) == 0)
                g_cu.SetDeviceFlags(fl | 0x01 /*cudaDeviceScheduleSpin*/);
        }
        if (!g_trt_rt_shared) {
            void* rt = g_trt_create_runtime(&g_trt_log, TrtVersionInt());
            if (!rt)
                rt = g_trt_create_runtime(&g_trt_log, (int32_t)NV_TENSORRT_VERSION);
            if (!rt) {
                std::fprintf(stderr, "[trt] createInferRuntime 失败（版本整型 %d，DLL 与"
                             " engine 不同代？）\n", (int)TrtVersionInt());
                return false;
            }
            g_trt_rt_shared = (nvinfer1::IRuntime*)rt;
        }
        std::lock_guard<std::mutex> lk(g_engs_mx);
        if (!g_engs_ptr) g_engs_ptr = new std::deque<TrtEngineCache>();
        for (auto& e : *g_engs_ptr)
            if (e.path == cfg.engine_path && e.eng) {
                eng_ = &e;
                eng_path_ = e.path;
                return true;   // 同路径复用（多组同引擎：权重共享，refit 随实例）
            }
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
        nvinfer1::ICudaEngine* eng =
            g_trt_rt_shared->deserializeCudaEngine(blob.data(), blob.size());
        if (!eng) {
            std::fprintf(stderr, "[trt] 反序列化失败: %s（TF32 环境不一致会拒建 context）\n",
                         p.c_str());
            return false;
        }
        g_engs_ptr->push_back(TrtEngineCache{p, eng});
        eng_ = &g_engs_ptr->back();
        eng_path_ = p;
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
    // ③批量 D2D 档：**缺省关**（2026-09-29 判决：掼蛋 PFD workload 池路径
    // 罕见指纹漂移逐臂排除后定谳=批 API 本身——逐行 MemcpyAsync 4/4×57k 局
    // 零翻面且实测零差（892.7 vs 893.4 局/s，D2D 本就在显存、批量省的提交
    // 税在 Linux 不存在）。FARM_STATE_D2D_BATCH=1 opt-in 仅供 A/B 回归
    static bool StBatchOn() {
        static const int mode = [] {
            const char* e = std::getenv("FARM_STATE_D2D_BATCH");
            return e ? std::atoi(e) : 0;
        }();
        return mode > 0 && g_cu.MemcpyBatchOk();
    }
    // ③gather 内核档（DATA13，2026-09-30 判决）：FARM_STATE_GATHER=1 opt-in。
    // 逐行 D2D 的执行端 CE 固定开销（~6.4µs×1536 行/批 ≈10ms）是吞吐真墙；
    // 批 API 只省提交端不省执行端。内核路径=主机建行号表→H2D→单 launch。
    // 装载失败自动回落旧路（batch/逐行）——旋钮是加速器不是依赖。
    static bool StGatherOn() {
        static const bool ready = [] {
            const char* e = std::getenv("FARM_STATE_GATHER");
            if (!e || std::atoi(e) != 1) return false;
            if (!StateGatherInit()) {
                std::fprintf(stderr, "[trt] state gather 内核装载失败——状态面回落逐行/batch 路径\n");
                return false;
            }
            std::printf("[trt] state gather 内核就绪（行号表→单 launch）\n");
            std::fflush(stdout);
            return true;
        }();
        return ready;
    }
    // 主机侧行号表（语义全在这定，内核零分支零原子）：
    //   -1=幻影行跳过（目标留陈旧内容，与逐行路径 continue 逐位同）；
    //   zero_ok（仅填充侧）：pending 行改用池末保留零行（行号=prows）且消费
    //   即清旗——load(acquire)→决策→store(0) 顺序与逐行路径逐位同；
    //   散射侧 zero_ok=false：pending 不碰（散射覆写池行即 NewGame 语义兑现）。
    static void SgBuildTbl(TrtSession* s, int n_rows, const StatePool& pol,
                           bool zero_ok, int* tbl) {
        const int prows = pol.rows;
        for (int r = 0; r < n_rows; r++) {
            const int pid = s->st_pids[r].load(std::memory_order_acquire);
            if (pid < 0 || pid >= prows) { tbl[r] = -1; continue; }
            if (zero_ok && pol.zero_pending[pid].load(std::memory_order_acquire)) {
                pol.zero_pending[pid].store(0, std::memory_order_relaxed);
                tbl[r] = prows;
            } else {
                tbl[r] = pid;
            }
        }
    }
    // 跨流序（DATA17 定谳）：本会话入流任何"读/写共享池行"的操作前，等全体
    // 触池他席的最近状态事件。两级收窄（原=实例内全体无差别）：
    //   ①按池粒度：只等真正共享池的会话（全局注册表——跨组共享形态由此入
    //     面，洞闭合）；
    //   ②EventQuery 探完成：事件已完成（或从未 record）=散射 GPU 侧已全局
    //     收敛，跳过 StreamWaitEvent——流内不挂无谓依赖，各银行流自由重叠
    //     （原实现 N 条银行流每批两两握手=退化串行链，90k 墙主嫌疑）。
    // 事件未完成才真 wait——序保证与原实现逐位同（完成=无需序，未完成=补序）。
    bool SgWaitOthers(TrtSession* s) {
        if (!g_cu.StreamWaitEvent) return true;
        StTouchEntry others[64];
        int n_others = 0;
        for (void* pd : s->touch_pools) {
            StTouchEntry sub[64];
            const int ns = StTouchOthers(s, pd, sub, 64);
            for (int i = 0; i < ns && n_others < 64; i++) {
                bool dup = false;
                for (int k = 0; k < n_others; k++)
                    if (others[k].sess == sub[i].sess) { dup = true; break; }
                if (!dup) others[n_others++] = sub[i];
            }
        }
        for (int k = 0; k < n_others; k++) {
            if (!others[k].ev_state) continue;
            if (g_cu.EventQuery
                && g_cu.EventQuery(others[k].ev_state) == 0 /*cudaSuccess=已完成*/)
                continue;
            if (g_cu.StreamWaitEvent(s->stream, others[k].ev_state, 0)) return false;
        }
        return true;
    }
    // 建表+H2D+单 launch（rc!=0 即 false——失败纪律同 memcpy，上层判负）。
    // 表缓冲按 slots 一次分配逐批复用；安全由会话生命周期保证（旗标殿后=
    // 复用时上一批 H2D 必已执行）。
    static bool SgRun(TrtSession* s, void* rows_base, void* pool_base,
                      const StatePool& pol, bool zero_ok, int n_rows,
                      size_t rb, int mode) {
        if (n_rows > s->sg_cap) {
            if (s->sg_tbl_d && g_cu.Free(s->sg_tbl_d)) return false;
            delete[] s->sg_tbl_h;
            s->sg_tbl_h = new int[s->slots];
            s->sg_tbl_d = nullptr;
            if (g_cu.Malloc((void**)&s->sg_tbl_d, (size_t)s->slots * sizeof(int)))
                return false;
            s->sg_cap = s->slots;
        }
        SgBuildTbl(s, n_rows, pol, zero_ok, s->sg_tbl_h);
        if (g_cu.MemcpyAsync(s->sg_tbl_d, s->sg_tbl_h,
                             (size_t)n_rows * sizeof(int), 1, s->stream))
            return false;
        return StateGatherLaunch(rows_base, pool_base, s->sg_tbl_d, n_rows, rb,
                                 s->stream, mode);
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
    bool EnqueueStateTail(TrtSession* s, int n_rows) {
        s->sb.clear();   // 防御：早退不留陈旧条目（填充批已在 Submit 清空）
        if (!SgWaitOthers(s)) return false;   // 本散射等他席最近散射（跨流
        // 行序：旧局末散射不得晚于新局散射覆写同行——NewGame 延迟零行案）
        const bool batch = StBatchOn();
        for (size_t oi = 0; oi < s->outs.size(); oi++) {
            const int pi = oi < s->st_out.size() ? s->st_out[oi] : -1;
            const size_t rb = (size_t)s->outs[oi].meta.width * sizeof(float);
            if (pi >= 0) {
                char* pool = (char*)st_pools_[(size_t)pi].dev;
                const char* src = (const char*)s->outs[oi].dev;
                const size_t prows = (size_t)st_pools_[(size_t)pi].rows;
                if (StGatherOn()) {   // DATA13：单 launch scatter 全面
                    if (!SgRun(s, s->outs[oi].dev, pool, st_pools_[(size_t)pi],
                               /*zero_ok=*/false, n_rows, rb, 1))
                        return false;
                    continue;
                }
                for (int r = 0; r < n_rows; r++) {
                    const int pid = s->st_pids[r].load(std::memory_order_acquire);
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
        if (s->ev_state && g_cu.EventRecord
            && g_cu.EventRecord(s->ev_state, s->stream))
            return false;   // 他席可见点（散射全部入流后）
        if (g_cu.MemcpyAsync(s->mb_flag_dev, s->mb_seq_dev, 4, 2, s->stream))
            return false;
        return true;
    }

    // 邮箱提交：序号 +1 写 staging → [图=graphLaunch（H2D 节点执行时读 staging
    // 当前值）] / [在线=4B H2D→enqueueV3→输出 D2H→盖章 逐个入队 stream]；
    // 大输出模型非满批 → 计算图/在线 enqueue + 图外部分行 D2H；
    // ③状态会话（pids 已绑）→ 恒计算图/在线 + 图外状态尾段
    bool MbSubmit(TrtSession* s, int n_rows, unsigned& seq_out) {
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
                        char* pool = (char*)st_pools_[(size_t)pi].dev;
                        const char* src = (const char*)s->outs[oi].dev;
                        const size_t prows = (size_t)st_pools_[(size_t)pi].rows;
                        for (int r = 0; r < n_rows; r++) {
                            const int pid = s->st_pids[r].load(
                                std::memory_order_acquire);
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
            // pop 面=整平面 P≠slots（同族坑第三处：按 slots 填=写 P 倍界外
            // ——YGO pop[2,660303] slots=64 时 169MB 越界砸穿进程，热身期
            // 秒崩；玩具床 P/slots 比小+flat 短=溢出落自家 arena 尾页被掩盖，
            // R12 门全绿的验收盲区，2026-09-30）
            const size_t rows = (size_t)(i.meta.population ? i.meta.dims[0]
                                                           : s->slots);
            size_t n = i.meta.row_bytes * rows / i.meta.esize;
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
    void CaptureGraph(TrtSession* s) {
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
        if (!this->MbSubmit(s, s->slots, seq) || !WaitFlag(s, seq, 5000.0, &spin)) {
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
