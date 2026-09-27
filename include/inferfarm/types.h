// ============================================================
//  inferfarm/types.h — 基础类型：dtype/张量元数据/模型规格/槽位读写视图
//
//  行独立假设（全框架的支点，见 docs/design-judgments.md）：
//  推理图必须逐行独立（无 batchnorm 类跨行算子；训练图不限，导出折叠即可）。
//  银行不满也整批照发，尾行读显存上批旧数据无害——每发车只拷前 n 行正是
//  靠它成立。
// ============================================================
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace inferfarm {

// 元素类型（覆盖 TRT 会话面所需；INT8/F16 不进输入输出面，仅 refit blob 用）
enum ElemDtype : uint8_t {
    DTYPE_F32 = 0,
    DTYPE_I64 = 1,
    DTYPE_I32 = 2,
    DTYPE_BOOL = 3,
};
inline size_t DtypeSize(ElemDtype t) {
    static const size_t s[4] = {4, 8, 4, 1};
    return s[(int)t];
}

struct InputMeta {
    std::string name;
    ElemDtype et = DTYPE_F32;
    size_t esize = 4;
    std::vector<int64_t> dims;   // 全量形状：dim0=槽数（fb），其余=行形状
    size_t row_bytes = 0;        // 一行的字节数（dims[1:] 乘积 × esize）
    bool population = false;     // population 输入面（演化路由，判决16）：
                                 // 整张量=[P, flat_w] 种群权重平面——Claim 不清零/
                                 // 缓存不哈希（代次 gen 已在键）/SubmitBatch 脏旗
                                 // 全量拷（cuda）；dim0=P ≠ slots 合法
    bool append = false;         // 声明式增量 H2D 面（判决25）：模型配置
                                 // append_inputs 点名的 append-only 面（后端在
                                 // 枚举期标记；未点名=full 现状行为）。正确性前提
                                 // =乘客声明即承诺：行内容演化是前缀增长，depth
                                 // 递减=换局信号（见 SlotWriter::FaceDepth）
    bool headlive = false;       // 头部活跃面（判决25 扩展，2026-09-27）：配置
                                 // headlive_inputs 点名的 newest-first 面。行内
                                 // [0,depth)=本批新鲜内容（每批可任意变化，逆序
                                 // 移位 OK），[depth,dims[1])=恒零（乘客承诺，
                                 // 哨兵必校）。设备侧尾槽恒零（zero 基+缩深
                                 // memset），每批只传 [0,depth)。与 append 互斥
                                 // （同面双声明=枚举期拒绝）。模型零改动，
                                 // b4 parity 不破（掼蛋 chain/chattr 形态）
};

struct OutputMeta {
    std::string name;
    std::vector<int64_t> dims;   // dim0=槽数
    int width = 0;               // 行宽（f32 个数；输出协议恒 fp32）
};

// 模型规格：后端 LoadSpec 产出（TRT=engine 枚举；CPU=ModelConfig 声明）。
struct ModelSpec {
    std::string backend;              // "cpu" / "trt"
    std::vector<InputMeta> ins;
    std::vector<OutputMeta> outs;
    int slots = 64;                   // dim0（=银行槽位数=批形状）
};

// CPU 后端的模型声明（确定性玩具/稠密模型：无需 GPU 即可验证全链与确定性门）
struct CpuModelDecl {
    struct In { std::string name; ElemDtype et; std::vector<int64_t> row_dims; };
    struct Out { std::string name; int width; };
    std::vector<In> ins;
    std::vector<Out> outs;
    int slots = 64;
    int poly_k = 8;                    // 每输入参与点积的行首元素数上限（棋类
                                      // 全局面输入=行宽，如五子棋 225）
    int hidden = 0;                    // >0=一层 MLP（H 隐藏单元：relu(W1·x+b1)
                                      // → tanh(W2·h+b2)）；0=纯线性（refit 协议
                                      // 仅覆盖线性模式）
    int pop_p = 0;                     // >0=population 路由模式（演化，判决16）：
                                      // 追加输入 "pop"[pop_p, flat_w]（f32 种群权重
                                      // 平面，population=true）与 "mid"[slots]
                                      // （i64，每行个体号）；权重=pop[mid 行]，
                                      // flat 布局=CpuBuildMlpFlat 序。须 hidden>0
    uint32_t weight_seed = 0xC0FFEEu; // 权重种子（同种子=同权重=逐位确定）
};

// population 路由模式的 flat 权重序列化（与 cpu 后端 BuildMlpWeights 同生成序）：
// b1[H] | w1[t][i][L] | b2[j][k] | w2[j][k][H]——测试/驱动侧用来构造 pop 平面，
// 使"均匀 pop（各行同权重）"与普通单模型腿逐位可比（门 G9a）
std::vector<float> CpuBuildMlpFlat(const CpuModelDecl& d, uint32_t seed);

// 模型配置：Farm 初始化时交给后端（多设备：每设备组一份，见 FarmConfig.devices）
struct ModelConfig {
    std::string backend = "cpu";      // "cpu" | "ort" | "trt" | "ncnn"
    std::string model_path;           // ort: fb 烤死的 onnx；ncnn: .param
                                      // （权重=同 basename .bin，pnnx 约定）
    std::string engine_path;          // trt: engine 文件
    std::string ort_dir;              // ort: onnxruntime.dll 所在目录（缺省 env FARM_ORT_DIR）
    std::string cuda_dir;             // ort/trt: cudart64_12.dll 所在目录（缺省 env FARM_CUDA_DIR）
    std::string trt_dir;              // trt: nvinfer_10.dll 所在目录（缺省 env FARM_TRT_DIR）
    std::string ncnn_dir;             // ncnn: ncnn.dll 所在目录（缺省 env FARM_NCNN_DIR）
    int device_id = 0;                // 设备序号：trt=cudaSetDevice；ort_cuda=EP device_id；
                                      // ort_dml=DML 适配器序号（异构双卡的关键面）；
                                      // ncnn=Vulkan 设备序号（如 1=610M 核显）
    std::string ort_ep = "cuda";      // ort 执行提供器："cuda" | "dml"（AMD/核显路线：
                                      // DML=宿主绑定+同步 Run，无图无 cudart）
    int ort_threads = 1;              // ort: IntraOp 线程数（纪律=1，防多局互踩）
    bool ort_cuda_graph = true;       // ort: enable_cuda_graph（仅银行会话生效——图会话
                                      // 绑线程[PerThreadContext 铁律]，inline 会话强制关；
                                      // dml 恒无图）
    std::string population_input;     // population 路由（演化，判决16）：模型里种群
                                      // 权重平面的输入名（如 "pop"；空=普通单模型）。
                                      // 配套 Farm::SetPopulation 代际换权重
    std::vector<std::string> append_inputs;   // 声明式增量 H2D（判决25）：append-only
                                      // 输入面名单（空=全 full=零行为差；点名面须
                                      // ≥2 维，dim1=行首维=深度单位）。适配器组装期
                                      // 经 SlotWriter::FaceDepth 逐行申报有效深度，
                                      // 后端按 [synced, depth) 段增量传输省 PCIe。
                                      // 承诺与边界见 FaceDepth 注释
    std::vector<std::string> headlive_inputs; // 头部活跃面名单（判决25 扩展）：
                                      // newest-first 面（新内容压行首、尾槽恒零）。
                                      // 每批只传 [0,depth)，缩深走设备侧 memset
                                      // （零 PCIe）。与 append_inputs 同面互斥。
                                      // 承诺=宿主行尾槽 [depth,slots) 恒零
    std::string refit_weights;        // 可选：init 期一次性换心（RW1 blob 路径；多设备组
                                      // 不支持=fail fast）
    CpuModelDecl cpu;                 // backend=="cpu" 时生效；backend=="ncnn" 时
                                      // 复用其 ins/outs 声明（名字+行宽+et）——
                                      // ncnn param 无 shape 枚举，spec 靠乘客声明
                                      //（blob 名须与 pnnx/param 一致）
};

// 输出投递目的地：收割时由调度台逐行拷进这些缓冲（bank 回池先于游戏恢复，
// 拷贝承重——不能让适配器直接读银行显存/arena，见 bank.cpp 收割注释）
struct OutputDest {
    const char* name = nullptr;       // 输出名（后端模型规格内）
    float* dst = nullptr;             // 适配器自有缓冲（生命周期须跨 SubmitWait：
                                      // 收割侧回填后才可读）
    int n = 0;                        // dst 容量（float 元素数）。收割侧拷
                                      // min(模型行宽, n)：行宽>n 取 n（截断方向
                                      // =按容量），n≤0 或 dst 空=该输出跳过
};

// 组装视图：适配器在 AssembleInto 里按名取行指针，直写槽位（零拷贝契约）
class SlotWriter {
public:
    virtual ~SlotWriter() = default;
    // 返回本槽该输入的行首指针；未知名字返回 nullptr（row_bytes 可空）
    virtual void* Row(const char* name, size_t* row_bytes) = 0;
    // 声明式增量 H2D（判决25）：组装本行时声明该行该输入面的当前有效深度。
    //   depth 单位=行首维（dims[1]）条目数，值域 [0, dims[1]]；行字节段
    //   = [depth 元素之前全有效, 之后全零]——零基组装（Claim 清零）下"未写区
    //   =0"与该声明互为充要。缺省不实现（适配器从不调用）=零开销，该行该面
    //   按 full 现状处理。逐行也可混批：同批内部分行声明部分行不声明=声明行
    //   增量、未声明行整行传输。
    // **append-only 承诺（乘客声明即承诺）**：声明行跨批的 [0, synced) 前缀
    //   字节不变（同一局的行动历史增长）；depth 递减=换局信号（新内容自
    //   [0,depth) 重写+尾部清零）。违反承诺=设备侧旧前缀与宿主漂移——输出
    //   错而指纹门红，且 FARM_H2D_DELTA_DEBUG=1 哨兵当场报非 0。无法承诺的
    //   行就别声明（full 兜底永远正确）。实践形态见判决 25 接入指引。
    // **headlive 承诺（头部活跃面，2026-09-27）**：行内 [0,depth) 是本批
    //   新鲜内容（每批可任意变化——newest-first 逆序移位 OK），
    //   [depth,slots) 宿主恒零（zero 基组装天然满足）。后端每批传 [0,depth)、
    //   缩深走设备 memset；尾槽非零=哨兵违约。depth 语义与 append 相同：
    //   单位=行首维条目数，值域 [0,dims[1]]，递减=换局。
    virtual void FaceDepth(const char* name, int depth) {
        (void)name; (void)depth;
    }
};

// 收割完成后适配器读自有缓冲——无需框架视图；ApplyResult() 直接读成员即可。

} // namespace inferfarm
