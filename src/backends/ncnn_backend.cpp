// ncnn_backend.cpp — ncnn Vulkan 后端（AMD 核显路线，2026-09-26）。
//
// 与 ort/trt 后端对称的第三条 GPU 路线：直连 ncnn.dll 的 C API（动态加载，
// GetProcAddress 函数表——c_api.h 的签名手工抄录，零编译期依赖）。
// 执行模型：ncnn 的 extract 是同步的 → DML 同款专属发射线程（SubmitBatch
// 投递即返回，CompletionReached 轮询 h_done 旗标，TRT 邮箱同款衣服）。
// batch 语义：ncnn 无 batch 维，用 Mat 的 h 维承批（w=行宽元素数，h=本批
// 行数）——external_2d_elem 直接包装宿主槽 arena，前缀 n 行语义与 ORT 的
// "只拷前 n 行"对齐（行独立 ⇒ 垃圾行无害）。
// spec：声明式（ModelConfig.cpu 的 ins/outs——ncnn param 无 shape 枚举；
// blob 名须与 pnnx/param 一致）。
//
// env：FARM_NCNN_DIR=ncnn.dll 所在目录（vulkan wheel/官方 shared 包）。
#include "inferfarm/backend.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace inferfarm {

namespace {

// ---------------- 动态加载的 C API 函数表（签名照抄 c_api.h）----------------
struct NcnnApi {
    const char* (*version)(void) = nullptr;
    void* (*net_create)(void) = nullptr;
    void (*net_destroy)(void*) = nullptr;
    int (*net_load_param)(void*, const char*) = nullptr;
    int (*net_load_model)(void*, const char*) = nullptr;
    int (*net_get_input_count)(const void*) = nullptr;
    const char* (*net_get_input_name)(const void*, int) = nullptr;
    int (*net_get_output_count)(const void*) = nullptr;
    const char* (*net_get_output_name)(const void*, int) = nullptr;
    void (*net_set_vulkan_device)(void*, int) = nullptr;
    void* (*net_get_option)(void*) = nullptr;
    void* (*option_create)(void) = nullptr;
    void (*option_destroy)(void*) = nullptr;
    void (*option_set_num_threads)(void*, int) = nullptr;
    void (*option_set_use_vulkan_compute)(void*, int) = nullptr;
    void (*option_set_use_fp16_packed)(void*, int) = nullptr;
    void (*option_set_use_fp16_storage)(void*, int) = nullptr;
    void (*option_set_use_fp16_arithmetic)(void*, int) = nullptr;
    void (*option_set_use_cooperative_matrix)(void*, int) = nullptr;
    void* (*mat_create_external_2d_elem)(int, int, void*, size_t, int, void*) = nullptr;
    void (*mat_destroy)(void*) = nullptr;
    void* (*mat_get_data)(const void*) = nullptr;
    int (*mat_get_w)(const void*) = nullptr;
    int (*mat_get_h)(const void*) = nullptr;
    void* (*extractor_create)(void*) = nullptr;
    void (*extractor_destroy)(void*) = nullptr;
    int (*extractor_input)(void*, const char*, const void*) = nullptr;
    int (*extractor_extract)(void*, const char*, void**) = nullptr;

    bool ok = false;

    bool Load(const std::string& dir) {
        if (ok) return true;
        std::string dll = "ncnn.dll";
        std::string p = dir.empty() ? dll : dir + "\\" + dll;
#ifdef _WIN32
        HMODULE h = dir.empty() ? LoadLibraryA(p.c_str())
                                : LoadLibraryExA(p.c_str(), NULL,
                                                 LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!h) {
            std::fprintf(stderr, "[ncnn] LoadLibrary %s 失败 GLE=%lu\n",
                         p.c_str(), GetLastError());
            return false;
        }
        auto g = [&](const char* n) { return (void*)GetProcAddress(h, n); };
        auto req = [&](void* pfn, const char* n) {
            if (!pfn) {
                std::fprintf(stderr, "[ncnn] 缺导出符号 %s\n", n);
                return false;
            }
            return true;
        };
        version = (const char* (*)(void))g("ncnn_version");
        net_create = (void* (*)(void))g("ncnn_net_create");
        net_destroy = (void (*)(void*))g("ncnn_net_destroy");
        net_load_param = (int (*)(void*, const char*))g("ncnn_net_load_param");
        net_load_model = (int (*)(void*, const char*))g("ncnn_net_load_model");
        net_get_input_count = (int (*)(const void*))g("ncnn_net_get_input_count");
        net_get_input_name = (const char* (*)(const void*, int))g("ncnn_net_get_input_name");
        net_get_output_count = (int (*)(const void*))g("ncnn_net_get_output_count");
        net_get_output_name = (const char* (*)(const void*, int))g("ncnn_net_get_output_name");
        net_set_vulkan_device = (void (*)(void*, int))g("ncnn_net_set_vulkan_device");
        net_get_option = (void* (*)(void*))g("ncnn_net_get_option");
        option_set_num_threads = (void (*)(void*, int))g("ncnn_option_set_num_threads");
        option_set_use_vulkan_compute = (void (*)(void*, int))g("ncnn_option_set_use_vulkan_compute");
        option_set_use_fp16_packed = (void (*)(void*, int))g("ncnn_option_set_use_fp16_packed");
        option_set_use_fp16_storage = (void (*)(void*, int))g("ncnn_option_set_use_fp16_storage");
        option_set_use_fp16_arithmetic = (void (*)(void*, int))g("ncnn_option_set_use_fp16_arithmetic");
        option_set_use_cooperative_matrix = (void (*)(void*, int))g("ncnn_option_set_use_cooperative_matrix");
        mat_create_external_2d_elem = (void* (*)(int, int, void*, size_t, int, void*))g("ncnn_mat_create_external_2d_elem");
        mat_destroy = (void (*)(void*))g("ncnn_mat_destroy");
        mat_get_data = (void* (*)(const void*))g("ncnn_mat_get_data");
        mat_get_w = (int (*)(const void*))g("ncnn_mat_get_w");
        mat_get_h = (int (*)(const void*))g("ncnn_mat_get_h");
        extractor_create = (void* (*)(void*))g("ncnn_extractor_create");
        extractor_destroy = (void (*)(void*))g("ncnn_extractor_destroy");
        extractor_input = (int (*)(void*, const char*, const void*))g("ncnn_extractor_input");
        extractor_extract = (int (*)(void*, const char*, void**))g("ncnn_extractor_extract");
        bool good = version && req((void*)net_create, "ncnn_net_create")
            && req((void*)net_load_param, "ncnn_net_load_param")
            && req((void*)net_load_model, "ncnn_net_load_model")
            && req((void*)net_get_input_count, "ncnn_net_get_input_count")
            && req((void*)net_get_input_name, "ncnn_net_get_input_name")
            && req((void*)net_get_output_count, "ncnn_net_get_output_count")
            && req((void*)net_get_output_name, "ncnn_net_get_output_name")
            && req((void*)mat_create_external_2d_elem, "ncnn_mat_create_external_2d_elem")
            && req((void*)mat_destroy, "ncnn_mat_destroy")
            && req((void*)mat_get_data, "ncnn_mat_get_data")
            && req((void*)extractor_create, "ncnn_extractor_create")
            && req((void*)extractor_destroy, "ncnn_extractor_destroy")
            && req((void*)extractor_input, "ncnn_extractor_input")
            && req((void*)extractor_extract, "ncnn_extractor_extract");
        if (!good) return false;
        ok = true;
        std::fprintf(stderr, "[ncnn] loaded %s（%s）\n", version ? version() : "?",
                     p.c_str());
        return true;
#else
        (void)dir; (void)p; (void)dll;
        return false;
#endif
    }
    // 注：不做会话归零 FreeLibrary——实测 ncnn.dll 卸载 detach 在本机挂死
    // （vk 清理等齐队列死锁）；保持进程尾随 loader 卸载，退出期 AV 见判决 23。
};

NcnnApi g_ncnn;   // 进程一份（同 cudart/ORT 模式；同 dll 同运行时）

// ---------------- 会话 ----------------
struct NcnnSess {
    // 模型
    void* net = nullptr;
    // 声明式 IO 面（LoadSpec 展开）
    struct InSlot {
        std::string name;
        size_t row_elems = 0;               // 行宽（f32 元素数）
        char* host = nullptr;               // 槽基址（256B carve）
    };
    std::vector<InSlot> ins;
    struct OutSlot {
        std::string name;
        int width = 0;
        char* host = nullptr;               // 输出 arena 段
    };
    std::vector<OutSlot> outs;
    int slots = 0;
    char* in_arena = nullptr;
    size_t in_bytes = 0;
    char* out_arena = nullptr;
    size_t out_bytes = 0;

    // 常量输入（v1.2 批量图：MatMul 的权重 blob 作为 Input 节点，数据来自
    // pnnx 约定的 <basename>_<blob>.npy，f32 C-order）——常驻 external Mat
    // （零拷贝：数据终身固定，Mat 只建一次）
    struct ConstIn {
        std::string name;
        std::vector<char> data;             // 保活（external mat 不拥有数据）
        void* mat = nullptr;
        int w = 0, h = 0;
    };
    std::vector<ConstIn> consts;

    // 发射线程（DML 同款：同步 extract 不占调度台）
    std::thread helper;
    std::mutex h_mx;
    std::condition_variable h_cv;
    unsigned h_pending = 0;                 // 在飞作业 seq（0=无；银行单飞≤1）
    std::atomic<unsigned> h_done{0};
    bool h_stop = false;
    int last_n = 0;                         // 本批行数（SubmitBatch 设置，发射线程读）
};

long long NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

class NcnnBackend : public InferBackend {
public:
    const char* Name() const override { return "ncnn"; }

    bool LoadSpec(const ModelConfig& cfg, int slots, ModelSpec& out) override {
        if (!g_ncnn.ok && !g_ncnn.Load(ResolveDir(cfg))) return false;
        // 声明式 spec（cfg.cpu 的 ins/outs；ncnn param 无 shape 枚举）
        if (cfg.cpu.ins.empty() || cfg.cpu.outs.empty()) {
            std::fprintf(stderr, "[ncnn] 需要 ModelConfig.cpu 的 ins/outs 声明"
                         "（名字+行宽；blob 名须与 param 一致）\n");
            return false;
        }
        out = ModelSpec{};
        out.slots = slots;
        for (const auto& in : cfg.cpu.ins) {
            InputMeta m;
            m.name = in.name;
            m.et = in.et;
            m.esize = 4;
            m.dims = in.row_dims;
            if (!m.dims.empty()) m.dims[0] = slots;   // dim0=槽数（fb 语义）
            size_t row_elems = 1;
            for (size_t d = 1; d < in.row_dims.size(); d++)
                row_elems *= (size_t)in.row_dims[d];
            if (in.row_dims.empty()) row_elems = 1;
            m.row_bytes = row_elems * 4;   // 声明面恒 f32（ncnn 输入 Mat f32）
            m.population = false;
            out.ins.push_back(m);
        }
        for (const auto& o : cfg.cpu.outs) {
            OutputMeta m;
            m.name = o.name;
            m.dims = {slots};
            m.width = o.width;
            out.outs.push_back(m);
        }
        param_path_ = cfg.model_path;
        bin_path_ = cfg.model_path.substr(0, cfg.model_path.rfind('.')) + ".bin";
        gpu_device_ = cfg.device_id;
        threads_ = cfg.ort_threads > 0 ? cfg.ort_threads : 1;
        return true;
    }

    void* CreateSession(const ModelConfig& cfg, const ModelSpec& spec,
                        bool for_bank) override {
        (void)for_bank;
        if (!g_ncnn.ok && !g_ncnn.Load(ResolveDir(cfg))) return nullptr;
        NcnnSess* s = new NcnnSess();
        s->slots = spec.slots;
        s->net = g_ncnn.net_create();
        if (!s->net) { delete s; return nullptr; }
        if (g_ncnn.net_set_vulkan_device) g_ncnn.net_set_vulkan_device(s->net, cfg.device_id);
        std::fprintf(stderr, "[ncnn] dbg: net created vk=%d\n", cfg.device_id);
        std::fflush(stderr);
        // option 必须在 load_param/load_model 之前定死（判决 23）：ncnn 的
        // layer pipeline 在 load 期按 option 烧制（fp16 存储格式/协作矩阵
        // 特化都在 create_pipeline 选型）；load 后再改 option=pipeline 与
        // 运行期数据格式错配。实测（判决 23）：dll 缺省 use_vulkan_compute=
        // 0（官方默认），旧版 load 后才开 vk+fp16 → 旧数字产生于非受控
        // 状态且 fp32 档崩；显式先设后，fp32 档全卡可跑。
        if (void* opt = g_ncnn.net_get_option(s->net)) {
            g_ncnn.option_set_num_threads(opt, threads_);
            // fp16/协作矩阵开关（FARM_NCNN_FP16=1 开；FARM_NCNN_CM 缺省关）：
            // RDNA2 实测（610M，批量图 64 行）：fp16+CM=6.8ms/批 vs CM 关
            // 1.4ms/批=协作矩阵在 Gemm 批形状上是 4.7× 减速器（16x16x16 tile
            // 对 M=64 欠利用，与 llama.cpp 在非 NVIDIA 卡禁 CM 同型）；5070Ti
            // 上 CM 中性。CM 只在乘客显式 FARM_NCNN_CM=1 时启用。fp16 归约/
            // CM/纯 fp32 三档指纹均非确定（判决 22 扩展：fp32 也不确定）。
            const char* fp16e = std::getenv("FARM_NCNN_FP16");
            const bool fp16 = !(fp16e && *fp16e && std::atoi(fp16e) == 0);   // 缺省开
            const char* cme = std::getenv("FARM_NCNN_CM");
            const bool cm = cme && *cme && std::atoi(cme) == 1;              // 缺省关
            if (g_ncnn.option_set_use_vulkan_compute) g_ncnn.option_set_use_vulkan_compute(opt, 1);
            if (g_ncnn.option_set_use_fp16_packed) g_ncnn.option_set_use_fp16_packed(opt, fp16 ? 1 : 0);
            if (g_ncnn.option_set_use_fp16_storage) g_ncnn.option_set_use_fp16_storage(opt, fp16 ? 1 : 0);
            if (g_ncnn.option_set_use_fp16_arithmetic) g_ncnn.option_set_use_fp16_arithmetic(opt, fp16 ? 1 : 0);
            if (g_ncnn.option_set_use_cooperative_matrix) g_ncnn.option_set_use_cooperative_matrix(opt, cm ? 1 : 0);
        }
        if (g_ncnn.net_load_param(s->net, param_path_.c_str()) != 0) {
            std::fprintf(stderr, "[ncnn] load_param 失败: %s\n", param_path_.c_str());
            g_ncnn.net_destroy(s->net); delete s; return nullptr;
        }
        std::fprintf(stderr, "[ncnn] dbg: param loaded\n");
        std::fflush(stderr);
        if (g_ncnn.net_load_model(s->net, bin_path_.c_str()) != 0) {
            std::fprintf(stderr, "[ncnn] load_model 失败: %s\n", bin_path_.c_str());
            g_ncnn.net_destroy(s->net); delete s; return nullptr;
        }
        // IO 面（声明驱动）
        for (const auto& m : spec.ins) {
            NcnnSess::InSlot is;
            is.name = m.name;
            is.row_elems = m.row_bytes / 4;
            s->ins.push_back(is);
        }
        for (const auto& m : spec.outs) {
            NcnnSess::OutSlot os;
            os.name = m.name;
            os.width = m.width;
            s->outs.push_back(os);
        }
        // arena carve（VirtualAlloc 64K 对齐；DML 同款宿主面）
        const size_t kAlign = 256;
        size_t off = 0;
        for (auto& i : s->ins)
            off = (off + i.row_elems * 4 * (size_t)s->slots + kAlign - 1) / kAlign * kAlign;
        s->in_bytes = off ? off : kAlign;
        off = 0;
        for (auto& o : s->outs) {
            o.host = nullptr;
            off = (off + (size_t)o.width * 4 * (size_t)s->slots + kAlign - 1) / kAlign * kAlign;
        }
        s->out_bytes = off ? off : kAlign;
#ifdef _WIN32
        s->in_arena = (char*)VirtualAlloc(nullptr, s->in_bytes,
                                          MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        s->out_arena = (char*)VirtualAlloc(nullptr, s->out_bytes,
                                           MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
        s->in_arena = (char*)malloc(s->in_bytes);
        s->out_arena = (char*)malloc(s->out_bytes);
#endif
        if (!s->in_arena || !s->out_arena) {
            std::fprintf(stderr, "[ncnn] arena 分配失败\n");
            DestroySession(s);
            return nullptr;
        }
        memset(s->in_arena, 0, s->in_bytes);
        memset(s->out_arena, 0, s->out_bytes);
        off = 0;
        for (auto& i : s->ins) {
            off = (off + kAlign - 1) / kAlign * kAlign;
            i.host = s->in_arena + off;
            off += i.row_elems * 4 * (size_t)s->slots;
        }
        off = 0;
        for (auto& o : s->outs) {
            off = (off + kAlign - 1) / kAlign * kAlign;
            o.host = s->out_arena + off;
            off += (size_t)o.width * 4 * (size_t)s->slots;
        }
        // ---- 常量输入（v1.2 批量图）：net 的 input blob 中，凡不属于数据
        // 声明（cfg.cpu.ins）的=权重/常量节点，从 pnnx 约定的
        // <basename>_<blob>.npy（f32 C-order）加载，常驻 external Mat。
        // 逐位确定性不在此层保证（见判决 22 非确定档）。
        {
            std::vector<std::string> data_names;
            for (const auto& i : s->ins) data_names.push_back(i.name);
            const int nin = g_ncnn.net_get_input_count(s->net);
            for (int i = 0; i < nin; i++) {
                const char* nm = g_ncnn.net_get_input_name(s->net, i);
                if (!nm) continue;
                std::string name(nm);
                bool is_data = false;
                for (const auto& d : data_names) is_data = is_data || d == name;
                if (is_data) continue;
                // npy 路径：<param basename>_<blob>.npy
                std::string base = param_path_.substr(0, param_path_.rfind('.'));
                std::string npy = base + "_" + name + ".npy";
                FILE* f = fopen(npy.c_str(), "rb");
                if (!f) {
                    std::fprintf(stderr, "[ncnn] 常量输入 %s 缺 %s（pnnx 转换产物）\n",
                                 name.c_str(), npy.c_str());
                    DestroySession(s);
                    return nullptr;
                }
                fseek(f, 0, SEEK_END);
                long sz = ftell(f);
                fseek(f, 0, SEEK_SET);
                std::vector<char> raw((size_t)sz);
                if (fread(raw.data(), 1, raw.size(), f) != raw.size() || sz < 10
                    || memcmp(raw.data(), "\x93NUMPY", 6) != 0) {
                    std::fprintf(stderr, "[ncnn] %s 不是 npy\n", npy.c_str());
                    fclose(f); DestroySession(s);
                    return nullptr;
                }
                fclose(f);
                // header：magic(6) ver(2) hlen(2) then dict；只认 v1.0 + <f4 + C-order
                unsigned short hlen = 0;
                memcpy(&hlen, raw.data() + 8, 2);
                std::string hdr(raw.data() + 10, hlen);
                if (hdr.find("'|f4'") == std::string::npos
                    && hdr.find("'<f4'") == std::string::npos) {
                    std::fprintf(stderr, "[ncnn] %s 非 f32 npy\n", npy.c_str());
                    DestroySession(s);
                    return nullptr;
                }
                if (hdr.find("True") != std::string::npos) {
                    std::fprintf(stderr, "[ncnn] %s 为 fortran_order，需 C-order\n", npy.c_str());
                    DestroySession(s);
                    return nullptr;
                }
                int w = 1, h = 1;
                {
                    auto grab = [&](const char* key) -> bool {
                        auto p = hdr.find("'" + std::string(key) + "': (");
                        if (p == std::string::npos) return false;
                        auto q1 = hdr.find('(', p) + 1;
                        auto q2 = hdr.find(')', q1);
                        std::string body = hdr.substr(q1, q2 - q1);
                        auto c = body.find(',');
                        if (c == std::string::npos) { w = atoi(body.c_str()); h = 1; }
                        else { h = atoi(body.c_str()); w = atoi(body.c_str() + c + 1); }
                        return true;
                    };
                    // shape=(h, w)（np 二维=行,列）
                    if (!grab("shape")) { DestroySession(s); return nullptr; }
                }
                NcnnSess::ConstIn ci;
                ci.name = name;
                ci.w = w;
                ci.h = h;
                ci.data.assign(raw.end() - (long)(w * h * 4), raw.end());
                ci.mat = g_ncnn.mat_create_external_2d_elem(w, h, ci.data.data(), 4, 1, nullptr);
                if (!ci.mat) { DestroySession(s); return nullptr; }
                s->consts.push_back(std::move(ci));
                std::fprintf(stderr, "[ncnn] 常量输入 %s <- %s（%dx%d）\n",
                             name.c_str(), npy.c_str(), w, h);
            }
        }
        // 发射线程（extract 同步；投递即返回）
        s->helper = std::thread([s] {
            for (;;) {
                unsigned job = 0;
                {
                    std::unique_lock<std::mutex> lk(s->h_mx);
                    s->h_cv.wait(lk, [s] { return s->h_stop || s->h_pending != 0; });
                    if (s->h_stop) return;
                    job = s->h_pending;
                }
                RunBatch(s);
                {
                    std::lock_guard<std::mutex> lk(s->h_mx);
                    s->h_pending = 0;
                }
                s->h_done.store(job, std::memory_order_release);
            }
        });
        return s;
    }

    bool Warmup(void* session) override {
        NcnnSess* s = (NcnnSess*)session;
        unsigned seq = 0;
        if (!SubmitBatch(s, s->slots, seq)) return false;
        while (!CompletionReached(s, seq)) {}
        CompletionFence();
        return true;
    }

    bool ProbeGraph(void* session) override {
        (void)session;
        return true;   // ncnn 无图捕获/地址烧死语义——检查无意义，恒过
    }

    void DestroySession(void* session) override {
        NcnnSess* s = (NcnnSess*)session;
        if (!s) return;
        if (s->helper.joinable()) {
            {
                std::lock_guard<std::mutex> lk(s->h_mx);
                s->h_stop = true;
            }
            s->h_cv.notify_all();
            s->helper.join();
        }
        for (auto& c : s->consts)
            if (c.mat) g_ncnn.mat_destroy(c.mat);
#ifdef _WIN32
        if (s->in_arena) VirtualFree(s->in_arena, 0, MEM_RELEASE);
        if (s->out_arena) VirtualFree(s->out_arena, 0, MEM_RELEASE);
#else
        free(s->in_arena);
        free(s->out_arena);
#endif
        if (s->net) g_ncnn.net_destroy(s->net);
        delete s;
    }

    void* InputRow(void* session, const char* name, int slot, size_t* row_bytes) override {
        NcnnSess* s = (NcnnSess*)session;
        for (auto& i : s->ins)
            if (i.name == name) {
                if (row_bytes) *row_bytes = i.row_elems * 4;
                return i.host + (size_t)slot * i.row_elems * 4;
            }
        return nullptr;
    }

    bool SubmitBatch(void* session, int n_rows, unsigned& seq_out) override {
        NcnnSess* s = (NcnnSess*)session;
        unsigned seq = s->h_done.load(std::memory_order_relaxed) + 1;
        s->last_n = n_rows;
        {
            std::lock_guard<std::mutex> lk(s->h_mx);
            s->h_pending = seq;
        }
        s->h_cv.notify_one();
        seq_out = seq;
        return true;
    }

    bool CompletionReached(void* session, unsigned seq) override {
        NcnnSess* s = (NcnnSess*)session;
        return s->h_done.load(std::memory_order_acquire) == seq;
    }

    void CompletionFence() override {
        std::atomic_thread_fence(std::memory_order_acquire);
    }

    const float* OutputRow(void* session, const char* name, int slot) override {
        NcnnSess* s = (NcnnSess*)session;
        for (auto& o : s->outs)
            if (o.name == name)
                return (const float*)(o.host + (size_t)slot * o.width * 4);
        return nullptr;
    }

    int OutputWidth(void* session, const char* name) override {
        NcnnSess* s = (NcnnSess*)session;
        for (auto& o : s->outs)
            if (o.name == name) return o.width;
        return 0;
    }

    bool RefitWeights(const char* rw1_path) override {
        (void)rw1_path;
        return false;   // ncnn 权重热换不在 v1 面（reload net=冷换，留给乘客侧）
    }

private:
    // 一批 = 单次 extract（v1.2 批量图，缺省）：数据输入 external_2d(h=本批行数)
    // 直包槽面，常量输入（权重）复用常驻 external Mat。逐行保留为
    // FARM_NCNN_BATCH=0 回退档（另：无权重输入 blob 的旧 InnerProduct 图无
    // 批量形，自动走逐行——见门）。实测（4096 局 ×3 交替，判决 23 后基线）：
    // 批量 5070Ti 0.15s / 610M 0.94-0.99s vs 逐行 5070Ti 4.3s / 610M 30.7s
    // ≈29×/32×（逐行每次 extract 重传 1.38MB 权重，批量摊薄 64×）。
    static void RunBatch(NcnnSess* s) {
        const char* be = std::getenv("FARM_NCNN_BATCH");
        const bool want_batch = (be && *be) ? std::atoi(be) == 1 : true;   // 缺省批量
        if (!want_batch || s->consts.empty()) {
            RunBatchRowwise(s);
            return;
        }
        const int n = s->last_n;
        void* ex = g_ncnn.extractor_create(s->net);
        if (!ex) {
            std::fprintf(stderr, "[ncnn] extractor 创建失败\n");
            return;
        }
        for (auto& c : s->consts)
            g_ncnn.extractor_input(ex, c.name.c_str(), c.mat);
        for (auto& i : s->ins) {
            void* m = g_ncnn.mat_create_external_2d_elem(
                (int)i.row_elems, n, i.host, 4, 1, nullptr);
            g_ncnn.extractor_input(ex, i.name.c_str(), m);
            g_ncnn.mat_destroy(m);
        }
        for (auto& o : s->outs) {
            void* om = nullptr;
            if (g_ncnn.extractor_extract(ex, o.name.c_str(), &om) != 0 || !om) {
                std::fprintf(stderr, "[ncnn] extract %s 失败\n", o.name.c_str());
                continue;
            }
            const size_t rowb = (size_t)o.width * 4;
            if (g_ncnn.mat_get_w(om) == o.width && g_ncnn.mat_get_h(om) == n) {
                memcpy(o.host, g_ncnn.mat_get_data(om), rowb * (size_t)n);
            } else {
                std::fprintf(stderr, "[ncnn] 输出面意外: %s w=%d h=%d（期望 w=%d h=%d）\n",
                             o.name.c_str(), g_ncnn.mat_get_w(om),
                             g_ncnn.mat_get_h(om), o.width, n);
            }
            g_ncnn.mat_destroy(om);
        }
        g_ncnn.extractor_destroy(ex);
    }

    // v1.1 逐行（正确性参照）：ncnn InnerProduct flatten 语义下唯一确定形
    static void RunBatchRowwise(NcnnSess* s) {
        const int n = s->last_n;
        for (int r = 0; r < n; r++) {
            void* ex = g_ncnn.extractor_create(s->net);
            if (!ex) return;
            // 常量输入（v1.2 批量图的权重 blob）逐行同样要喂——图把权重声明
            // 成 Input blob 时，缺喂=空 blob 进 GPU 层=AV（实测段错误点）
            for (auto& c : s->consts)
                g_ncnn.extractor_input(ex, c.name.c_str(), c.mat);
            for (auto& i : s->ins) {
                void* m = g_ncnn.mat_create_external_2d_elem(
                    (int)i.row_elems, 1, i.host + (size_t)r * i.row_elems * 4,
                    4, 1, nullptr);
                g_ncnn.extractor_input(ex, i.name.c_str(), m);
                g_ncnn.mat_destroy(m);
            }
            for (auto& o : s->outs) {
                void* om = nullptr;
                if (g_ncnn.extractor_extract(ex, o.name.c_str(), &om) != 0 || !om)
                    continue;
                memcpy(o.host + (size_t)r * o.width * 4,
                       g_ncnn.mat_get_data(om), (size_t)o.width * 4);
                g_ncnn.mat_destroy(om);
            }
            g_ncnn.extractor_destroy(ex);
        }
    }

    std::string ResolveDir(const ModelConfig& cfg) {
        if (!cfg.ncnn_dir.empty()) return cfg.ncnn_dir;
        const char* e = std::getenv("FARM_NCNN_DIR");
        return e && *e ? std::string(e) : std::string();
    }

    std::string param_path_, bin_path_;
    int gpu_device_ = 0;
    int threads_ = 1;
};

} // namespace inferfarm

namespace inferfarm {
InferBackend* CreateNcnnBackend() { return new NcnnBackend(); }
} // namespace inferfarm
