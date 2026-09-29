// ============================================================
//  inferfarm/backend.h — 推理后端接口（银行机件的唯一 GPU/CPU 面）
//
//  职责边界（与 ai_infer.cpp 现役机件一一对应）：
//    LoadSpec       = engine/onnx 元数据枚举（名字/形状/dtype）
//    CreateSession  = 建会话：pinned 输入 arena + 设备 arena + 输出 arena +
//                     专属执行上下文，**地址终身固定**（图与地址一夫一妻）
//    Warmup         = 3 跑热身 + 批图捕获 + 邮箱冒烟（降级链：图→邮箱→流同步）
//    ProbeGraph     = 图地址烧死小实验（两图案可分辨/图回放逐位）——不过=拒绝银行制
//    SubmitBatch    = 前缀 h2d（近满批走整块）+ 异步批发射；回 seq 供完成轮询
//    CompletionReached = 完成旗标的廉价 volatile 轮询（邮箱；流序保证旗标到=输出驻留）
//    RefitWeights   = RW1 blob 换心（引擎级，跨会话共享）
//
//  线律：SubmitBatch/OutputRow 可从调度台线程调用；InputRow 可从任意游戏
//  fiber/线程调用（不同槽行 disjoint）。CreateSession/Warmup/Probe 仅 init
//  期单线程调用。
// ============================================================
#pragma once
#include "types.h"
#include <atomic>   // ③BindStatePids 的槽→池下标数组

namespace inferfarm {

class InferBackend {
public:
    virtual ~InferBackend() = default;
    virtual const char* Name() const = 0;

    // 元数据发现（engine 枚举或声明展开）；slots 需与配置一致
    virtual bool LoadSpec(const ModelConfig& cfg, int slots, ModelSpec& out) = 0;

    // 会话=一家银行的全部私有资源（arena/上下文/图/邮箱）。返回句柄。
    // for_bank=true：银行会话（发车线程上创建+热身+回放——ORT 图会话的
    // PerThreadContext 铁律要求同线程；TRT 图回放本就线程无关，同规更稳）。
    // for_bank=false：inline 会话（任意线程经锁使用；ORT 强制关图）。
    virtual void* CreateSession(const ModelConfig& cfg, const ModelSpec& spec,
                                bool for_bank) = 0;
    virtual bool Warmup(void* session) = 0;          // 含图捕获+邮箱冒烟+降级链
    virtual bool ProbeGraph(void* session) = 0;      // 图地址烧死小实验（银行制准入门）
    virtual void DestroySession(void* session) = 0;

    // 槽行访问（组装直写面）
    virtual void* InputRow(void* session, const char* name, int slot, size_t* row_bytes) = 0;

    // 发车：前缀 n 行 H2D（n > 7/8·slots 时整块）+ 异步发射（图回放优先）
    virtual bool SubmitBatch(void* session, int n_rows, unsigned& seq_out) = 0;
    // 完成轮询：旗标==seq（volatile 读，µs 级；真=输出已驻留主机 arena）
    virtual bool CompletionReached(void* session, unsigned seq) = 0;
    virtual void CompletionFence() = 0;              // 检出后的载入序栅（lfence）

    // 输出读取（收割期调度台调用）：行首指针（f32）
    virtual const float* OutputRow(void* session, const char* name, int slot) = 0;
    virtual int OutputWidth(void* session, const char* name) = 0;

    // 换心（RW1 blob；TRT=refitter，CPU=直接改权重）。失败=false。
    virtual bool RefitWeights(const char* rw1_path) = 0;

    // 诊断钩子（R10 门用）：实例缓存引擎元素的地址（容器稳定性回归观测——
    // 仅比对值，不解引用；缺省 nullptr=后端无引擎缓存概念）。
    virtual const void* DebugEngineCookie() const { return nullptr; }

    // 写手线程能否就地发车（满座自驱）：ORT 图会话=PerThreadContext 铁律
    // （创建/热身/回放须同线程=调度台线程），写手线程回放会触发 ORT 侧
    // 重新捕获（CUDA failure 900/901，2026-09-22 五子棋 CNN chains=64 实测）
    // → false：满座不自驱，Notify 调度台发车。TRT 图回放线程无关、CPU 无图
    // → true（默认）。
    virtual bool DispatchFromWriterOk() const { return true; }

    // population 面写入（演化路由，判决16）：把 host 指向的 [P, flat_w] 种群
    // 权重平面拷入本会话的 population 输入。cuda 路线置脏旗（下次 SubmitBatch
    // 全量 H2D 一次）；cpu/dml 直读宿主=写完即生效。前置条件=腿已返回
    // （与 RefitWeights 同纪律）。不支持 population 的后端/找不到该输入=false
    virtual bool SetPopulation(void* session, const char* pop_input, const void* host) {
        (void)session; (void)pop_input; (void)host;
        return false;
    }

    // ③成对状态行（docs/state-residency-design.md）——设备池接线面。
    // BindStatePids：银行把本行槽→池下标数组（Claim 写/发车读，槽独占期
    // 单写者）交给后端，SubmitBatch 据此做状态输入行 D2D 填充（数组生命周期
    // =银行池，会话销毁前有效）。ResetStatePool：NewGame 清零池行（后端在
    // 全部会话流上 memsetAsync=任意下一读所在流自有序；链串行⇒无并发读者）。
    // 缺省 no-op/拒（cpu/ort v1 不支持=Farm Init 时 fail fast）。
    virtual bool BindStatePids(void* session, const std::atomic<int>* pids) {
        (void)session; (void)pids;
        return false;
    }
    virtual bool ResetStatePool(int row) {
        (void)row;
        return false;
    }

    // ③跨组共享状态池（决策级组路由 × 池路径，W4 形态）：持有组导出池表，
    // 共享组绑定同一批设备行（同 GPU D2D；链串行+收割完成序=跨流安全；
    // 单一正典状态=与主机路径语义逐位等价）。零基旗（zero_pending）也共享
    // ——NewGame 双组复位=同一原子旗双写，良性。缺省拒（不支持的后端
    // Farm 接线时 fail fast）。
    struct SharedStatePool {
        void* dev;                      // 池基址 [(rows+1) × row_bytes]
        std::atomic<char>* zero_pending; // [rows] NewGame 延迟清零旗（共享）
        size_t row_bytes;
        int rows;
    };
    // 持有组导出（out 由调用方分配；cap≥对数）。返回对数（0=无池，-1=cap
    // 不足，负值一律按不支持处理）。
    virtual int StatePoolInfo(SharedStatePool* out, int cap) {
        (void)out; (void)cap;
        return 0;
    }
    // 共享组绑定（须在首批发车前；绑定后本组会话的填充/散射直接读共享行）。
    virtual bool ShareStatePool(const SharedStatePool* pools, int n) {
        (void)pools; (void)n;
        return false;
    }

    // 能力位：LoadSpec 需要建"探测会话"（枚举元数据即毁——ORT≈0.1s/次，
    // 清单模式 56 腿/代≈5.6s/代纯探测税）的后端可声明 true：Farm 对组 0
    // 砍探测，改由首个真实银行会话经 CreateSessionWithSpec 顺带产出 spec。
    // cpu（声明展开，零成本）/trt 不必强推——缺省 false 走老路。
    virtual bool ProbeFreeSpec() const { return false; }
    // 探测砍除通道：建会话并顺带枚举模型规格写入 *spec_out。仅
    // ProbeFreeSpec()==true 的后端会被调用；其余后端缺省返回 nullptr。
    virtual void* CreateSessionWithSpec(const ModelConfig& cfg, int slots,
                                        bool for_bank, ModelSpec* spec_out) {
        (void)cfg; (void)slots; (void)for_bank; (void)spec_out;
        return nullptr;
    }

    // 完成等待句柄（可选能力）：返回可被 WaitForMultipleObjects 等待的 OS
    // 句柄（fence 桥接=完成信号量，每批恰一次释放），调度台据此做通知驱动
    // 等待（零轮询零量子）；nullptr=仅支持轮询。
    virtual void* CompletionWaitHandle(void* session) {
        (void)session;
        return nullptr;
    }

    // 声明式增量 H2D（判决25，可选能力）：组装期申报某槽某输入面的有效深度
    // （单位/承诺见 types.h SlotWriter::FaceDepth）。缺省=忽略（不实现增量的
    // 后端对声明零反应，行为=full——声明本身永远安全）。只在 config
    // append_inputs 点名且后端支持的面上生效。
    virtual void NoteFaceDepth(void* session, const char* name, int slot, int depth) {
        (void)session; (void)name; (void)slot; (void)depth;
    }
    // 声明会话起点复位（Claim 领槽时调用=该槽声明随新组装作废；漏复位会让
    // 上一任写手的陈旧声明被本批判读=静默漏传）。缺省=无操作。
    virtual void ClearFaceDepths(void* session, int slot) {
        (void)session; (void)slot;
    }
};

} // namespace inferfarm
