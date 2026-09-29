// ============================================================
//  state_gather.cpp — 状态面索引 gather/scatter 行拷内核实现（DATA13）
//
//  见 state_gather.h 契约。本件零 nvcc 依赖：内嵌 PTX 字符串经
//  cuModuleLoadData 装载（driver API 动态 dlopen libcuda，零链接依赖，
//  装载风格对齐 cudart_dyn.h 的两平台面但独立成 struct——别处不许动）。
//
//  PTX 设计要点（为何这样写）：
//    - grid=n blocks × 256 threads，每 block 负责一行 r=blockIdx.x，
//      读设备端 idx[r]：-1 直接 ret（幻影行目标留陈旧内容=逐行路径
//      continue 语义，天然同位）；其余线程组协作搬 rb 字节。
//    - 对齐陷阱：行起点=base+r*rb，rb 非 16 倍数（如 1000）时行起点
//      16B 不对齐；且 gather/scatter 两侧 src/dst 分属两个独立
//      cudaMalloc 池，(src&15)!=(dst&15) 完全可能。故每 block 运行时
//      检测：src/dst 16B 残差相等 → 16B 头对齐 + v4 主循环 + 字节尾；
//      残差不等 → 全程 u8 循环（stride 256，连续线程连续字节=天然
//      合并访问）。正确性优先于极限带宽——0.15ms 目标对逐字节路径
//      仍有 ~30 倍余量，两条路径都不必赌对齐。
//    - 参数面：rows/pool/idx 用 .u64，n/rb/mode 用 .u32（rb 传参前在
//      主机侧截断检查，>4GB 直接报错——状态面行不可能是这个量级）。
// ============================================================
#include "state_gather.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace inferfarm {

// ------------------------------------------------------------
// 内嵌 PTX（.target sm_86=3060M 是 Ampere；驱动 JIT 会为更低 cc 透明
// 降编，set 目标向上不兼容但本仓部署面就是本机单卡，写死 sm_86）。
// ------------------------------------------------------------
static const char kStateGatherPtx[] = R"PTX(
.version 7.4
.target sm_86
.address_size 64

.visible .entry farm_state_gather(
    .param .u64 p_rows,
    .param .u64 p_pool,
    .param .u64 p_idx,
    .param .u32 p_n,
    .param .u32 p_rb,
    .param .u32 p_mode)
{
    .reg .pred  %p<8>;
    .reg .b32   %r<32>;
    .reg .b64   %rd<40>;

    ld.param.u64    %rd1, [p_rows];
    ld.param.u64    %rd2, [p_pool];
    ld.param.u64    %rd3, [p_idx];
    ld.param.u32    %r1, [p_n];
    ld.param.u32    %r2, [p_rb];
    ld.param.u32    %r3, [p_mode];

    // r = blockIdx.x; guard r >= n (defensive, grid is always n)
    mov.u32         %r4, %ctaid.x;
    setp.ge.u32     %p1, %r4, %r1;
    @%p1 bra        END;

    mov.u32         %r5, %tid.x;
    cvt.u64.u32     %rd20, %r5;          // 64-bit shadow of tid (for address math)

    // i = idx[r] (device-side int32 table)
    mul.wide.u32    %rd4, %r4, 4;
    add.u64         %rd5, %rd3, %rd4;
    ld.global.s32   %r6, [%rd5];
    setp.lt.s32     %p2, %r6, 0;
    @%p2 bra        END;                 // -1 = phantom row: target keeps stale content

    // row bases: rows_row = rows_base + r*rb; pool_row = pool_base + i*rb
    // (i may be prows = reserved zero row; to the kernel it is just a row id)
    cvt.u64.u32     %rd6, %r2;           // rb64
    cvt.u64.u32     %rd7, %r4;           // r64
    mul.lo.u64      %rd8, %rd7, %rd6;
    add.u64         %rd9, %rd1, %rd8;    // rows_row
    cvt.s64.s32     %rd10, %r6;
    mul.lo.u64      %rd11, %rd10, %rd6;
    add.u64         %rd12, %rd2, %rd11;  // pool_row

    // mode: 0=gather(dst=rows_row,src=pool_row) 1=scatter(swapped)
    setp.ne.u32     %p3, %r3, 0;
    @%p3 bra        SCATTER;
    mov.u64         %rd13, %rd9;         // dst
    mov.u64         %rd14, %rd12;        // src
    bra             COPY;
SCATTER:
    mov.u64         %rd13, %rd12;
    mov.u64         %rd14, %rd9;
COPY:
    // resid mismatch (separate allocs + rb%16!=0) ->
    // full byte path (still coalesced, just more instructions)
    and.b64         %rd15, %rd13, 15;    // dres
    and.b64         %rd16, %rd14, 15;    // sres
    setp.ne.u64     %p4, %rd15, %rd16;
    @%p4 bra        BYTELOOP;

    // ---- 16B aligned fast path ----
    // h = (16 - dres) & 15, capped at rb (rb<16 degenerates to head-only)
    mov.u64         %rd17, 16;
    sub.u64         %rd18, %rd17, %rd15;
    and.b64         %rd19, %rd18, 15;    // h
    setp.gt.u64     %p5, %rd19, %rd6;
    @%p5 mov.u64    %rd19, %rd6;         // h = min(h, rb)
    cvt.u32.u64     %r7, %rd19;

    // head bytes: one byte per thread for tid < h
    setp.ge.u32     %p6, %r5, %r7;
    @%p6 bra        AL_MAIN;
    add.u64         %rd21, %rd14, %rd20;
    add.u64         %rd22, %rd13, %rd20;
    ld.global.u8    %r8, [%rd21];
    st.global.u8    [%rd22], %r8;
AL_MAIN:
    // main loop: u = (rb-h) >> 4 units of 16B, thread stride 256
    sub.u64         %rd23, %rd6, %rd19;  // rem = rb - h
    shr.u64         %rd24, %rd23, 4;     // u
    mov.u64         %rd25, %rd20;        // j = tid
AL_LOOP:
    setp.ge.u64     %p7, %rd25, %rd24;
    @%p7 bra        AL_TAIL;
    mul.lo.u64      %rd26, %rd25, 16;
    add.u64         %rd27, %rd26, %rd19; // off = h + j*16
    add.u64         %rd28, %rd14, %rd27;
    add.u64         %rd29, %rd13, %rd27;
    ld.global.v4.u32 {%r9, %r10, %r11, %r12}, [%rd28];
    st.global.v4.u32 [%rd29], {%r9, %r10, %r11, %r12};
    add.u64         %rd25, %rd25, 256;
    bra             AL_LOOP;
AL_TAIL:
    // byte tail: (rb-h) & 15 bytes, one per thread for tid < tail
    and.b64         %rd30, %rd23, 15;
    setp.ge.u64     %p7, %rd20, %rd30;
    @%p7 bra        END;
    mul.lo.u64      %rd31, %rd24, 16;
    add.u64         %rd32, %rd31, %rd19; // off = h + u*16 + tid
    add.u64         %rd33, %rd32, %rd20;
    add.u64         %rd34, %rd14, %rd33;
    add.u64         %rd35, %rd13, %rd33;
    ld.global.u8    %r13, [%rd34];
    st.global.u8    [%rd35], %r13;
    bra             END;

    // ---- byte fallback path ----
BYTELOOP:
    mov.u64         %rd36, %rd20;        // j = tid
BL_LOOP:
    setp.ge.u64     %p7, %rd36, %rd6;    // j >= rb ?
    @%p7 bra        END;
    add.u64         %rd37, %rd14, %rd36;
    add.u64         %rd38, %rd13, %rd36;
    ld.global.u8    %r14, [%rd37];
    st.global.u8    [%rd38], %r14;
    add.u64         %rd36, %rd36, 256;
    bra             BL_LOOP;
END:
    ret;
}
)PTX";

// ------------------------------------------------------------
// libcuda 驱动 API 动态装载（只取本件用到的 8 个符号；装载风格对齐
// cudart_dyn.h 两平台面，独立 struct 不共肉）
// ------------------------------------------------------------
struct DriverApi {
    int (*Init)(unsigned) = nullptr;
    int (*GetDevice)(int*) = nullptr;            // cuDeviceGet
    int (*GetErrorString)(int, const char**) = nullptr;
    int (*DevicePrimaryCtxRetain)(void*, int) = nullptr;
    int (*CtxGetCurrent)(void**) = nullptr;
    int (*CtxSetCurrent)(void*) = nullptr;
    int (*ModuleLoadData)(void**, const void*) = nullptr;
    int (*ModuleGetFunction)(void**, void*, const char*) = nullptr;
    int (*LaunchKernel)(void*, unsigned, unsigned, unsigned,
                        unsigned, unsigned, unsigned,
                        unsigned, void*, void**, void**) = nullptr;
    bool ok = false;

    template <class G>
    void Bind(G g) {
        Init = (int (*)(unsigned))g("cuInit");
        GetDevice = (int (*)(int*))g("cuDeviceGet");
        GetErrorString = (int (*)(int, const char**))g("cuGetErrorString");
        DevicePrimaryCtxRetain = (int (*)(void*, int))g("cuDevicePrimaryCtxRetain");
        CtxGetCurrent = (int (*)(void**))g("cuCtxGetCurrent");
        CtxSetCurrent = (int (*)(void*))g("cuCtxSetCurrent");
        ModuleLoadData = (int (*)(void**, const void*))g("cuModuleLoadData");
        ModuleGetFunction = (int (*)(void**, void*, const char*))g("cuModuleGetFunction");
        LaunchKernel = (int (*)(void*, unsigned, unsigned, unsigned,
                                unsigned, unsigned, unsigned,
                                unsigned, void*, void**, void**))g("cuLaunchKernel");
    }
};

static DriverApi g_dv;
static void* g_module = nullptr;      // CUmodule
static void* g_fn = nullptr;          // CUfunction farm_state_gather
static void* g_ctx = nullptr;         // primary context（dev 0；单卡钉死）
static bool g_init_done = false;
static bool g_init_ok = false;

static const char* CuErr(int rc) {
    const char* s = nullptr;
    if (g_dv.GetErrorString && g_dv.GetErrorString(rc, &s) == 0 && s) return s;
    return "未知驱动错误";
}

// Init：dlopen libcuda + cuInit + 装载 PTX + retain primary context。
// 幂等（二次调用返回缓存结果，不重复装载）。失败 stderr 一行原因。
bool StateGatherInit() {
    if (g_init_done) return g_init_ok;
    g_init_done = true;

#ifdef _WIN32
    const char* env = std::getenv("FARM_CUDA_DRIVER_DLL");
    const char* dll = env && *env ? env : "nvcuda.dll";
    HMODULE h = LoadLibraryA(dll);
    if (!h) {
        std::fprintf(stderr, "[state_gather] LoadLibrary %s 失败 GLE=%lu\n",
                     dll, GetLastError());
        return false;
    }
    auto g = [&](const char* n) { return (void*)GetProcAddress(h, n); };
#else
    const char* env = std::getenv("FARM_CUDA_DRIVER_DLL");
    std::string dll = env && *env ? env : "libcuda.so.1";
    void* h = dlopen(dll.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        std::fprintf(stderr, "[state_gather] dlopen %s 失败：%s\n",
                     dll.c_str(), dlerror());
        return false;
    }
    auto g = [&](const char* n) { return (void*)dlsym(h, n); };
#endif
    g_dv.Bind(g);
    if (!g_dv.Init || !g_dv.GetDevice || !g_dv.DevicePrimaryCtxRetain
        || !g_dv.CtxGetCurrent || !g_dv.CtxSetCurrent || !g_dv.ModuleLoadData
        || !g_dv.ModuleGetFunction || !g_dv.LaunchKernel) {
        std::fprintf(stderr, "[state_gather] libcuda 缺导出符号\n");
        return false;
    }

    int rc = g_dv.Init(0);
    if (rc != 0) {
        std::fprintf(stderr, "[state_gather] cuInit 失败(%d): %s\n",
                     rc, CuErr(rc));
        return false;
    }
    int dev = 0;
    rc = g_dv.GetDevice(&dev);
    if (rc != 0) {
        std::fprintf(stderr, "[state_gather] cuDeviceGet 失败(%d): %s\n",
                     rc, CuErr(rc));
        return false;
    }
    // primary context retain：调用方（TRT 工作线程）此前只走过 runtime API
    //（隐式绑 primary context）——驱动侧 launch 须同上下文。retain 一次存住，
    // Launch 时按需 SetCurrent，与 runtime 侧本就是同一 primary context。
    rc = g_dv.DevicePrimaryCtxRetain(&g_ctx, dev);
    if (rc != 0) {
        std::fprintf(stderr, "[state_gather] cuDevicePrimaryCtxRetain 失败(%d): %s\n",
                     rc, CuErr(rc));
        return false;
    }

    rc = g_dv.ModuleLoadData(&g_module, kStateGatherPtx);
    if (rc != 0) {
        std::fprintf(stderr, "[state_gather] cuModuleLoadData 失败(%d): %s\n",
                     rc, CuErr(rc));
        return false;
    }
    rc = g_dv.ModuleGetFunction(&g_fn, g_module, "farm_state_gather");
    if (rc != 0) {
        std::fprintf(stderr, "[state_gather] cuModuleGetFunction 失败(%d): %s\n",
                     rc, CuErr(rc));
        g_module = nullptr;
        return false;
    }
    g_init_ok = true;
    std::fprintf(stderr, "[state_gather] PTX 模块装载完成 (dev %d)\n", dev);
    return true;
}

bool StateGatherReady() {
    return g_init_ok && g_module != nullptr && g_fn != nullptr;
}

bool StateGatherLaunch(void* rows_base, void* pool_base, const int* idx_dev,
                       int n, size_t rb, void* stream, int mode) {
    if (!StateGatherReady()) {
        std::fprintf(stderr, "[state_gather] launch 前模块未就绪\n");
        return false;
    }
    if (n < 0 || rb > 0xFFFFFFFFu) {   // PTX 参数面 rb 是 .u32
        std::fprintf(stderr, "[state_gather] 非法参数 n=%d rb=%zu\n", n, rb);
        return false;
    }

    // 驱动 API 须当前线程有 CUDA 上下文：调用方是 runtime API 侧线程，
    // 隐式 primary context 对驱动 API 未必已绑——查空则补绑 retain 的那个。
    void* cur = nullptr;
    int rc = g_dv.CtxGetCurrent(&cur);
    if (rc != 0) {
        std::fprintf(stderr, "[state_gather] cuCtxGetCurrent 失败(%d): %s\n",
                     rc, CuErr(rc));
        return false;
    }
    if (!cur) {
        rc = g_dv.CtxSetCurrent(g_ctx);
        if (rc != 0) {
            std::fprintf(stderr, "[state_gather] cuCtxSetCurrent 失败(%d): %s\n",
                         rc, CuErr(rc));
            return false;
        }
    }

    unsigned rb32 = (unsigned)rb;
    unsigned un = (unsigned)n;
    void* params[] = { &rows_base, &pool_base, &idx_dev, &un, &rb32, &mode };
    // grid=n blocks × 256 threads，0 共享内存，extra 参数面不用
    rc = g_dv.LaunchKernel(g_fn, un, 1, 1, 256, 1, 1, 0,
                           stream, params, nullptr);
    if (rc != 0) {
        std::fprintf(stderr, "[state_gather] cuLaunchKernel 失败(%d): %s\n",
                     rc, CuErr(rc));
        return false;   // 绝不静默——调用方回退逐行路径
    }
    return true;
}

} // namespace inferfarm
