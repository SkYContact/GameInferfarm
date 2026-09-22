// ============================================================
//  inferfarm/game_adapter.h — GameAdapter 接缝（新库的提取面）
//
//  框架驱动环（每局，在对局 fiber 上）：
//    NewGame → loop { AdvanceToDecision → [Claim 领槽] → AssembleInto(直写槽)
//              → SubmitWait(挂起/收割回投) → ApplyResult } → Outcome
//
//  三条契约（违反=框架的正确性前提破洞，见 docs/design-judgments.md）：
//  1. **Advance 与 Assemble 无挂起点**——银行 close-drain 有界的前提
//     （在途写手必在几十 µs 完工）。组装内不得 FiberSuspend/阻塞 IO。
//  2. **行独立**——模型必须逐行独立（无 batchnorm 类跨行算子），银行不满
//     整批照发、尾行旧数据无害，全系于此。
//  3. **逐位确定性由适配器保证**——种子协议（game seed = seed0 + chain*per
//     + game）给定后，同种子双腿必须逐局同结果；框架的确定性门据此验。
//
//  YGO 参考实现：ocgcore+ai_bot 组装+对手执行器（D:/ygo/ygopro/ai_core，
//  现役生产代码）；本仓 examples/toy 是最小示范。
// ============================================================
#pragma once
#include "tls_frame.h"
#include "types.h"

namespace inferfarm {

class GameAdapter {
public:
    virtual ~GameAdapter() = default;

    // 开新局（同适配器跨局复用；NewGame 须把上局状态清干净=确定性前提）
    virtual void NewGame(uint64_t seed, bool we_first) = 0;

    // 推进到下一个决策点。返回 false=本局无更多决策（跳到收卷）。
    virtual bool AdvanceToDecision() = 0;

    // 组装直写槽：按名取行指针写特征（零拷贝契约——大数组在此直写；
    // 未写完的小输入可在 CollectOutputs 阶段后补，见 bank 层兜底拷贝）。
    virtual void AssembleInto(SlotWriter& slot) = 0;

    // 申报输出缓冲（每次决策调用；dst=适配器自有缓冲，收割侧回填）。
    // 返回条数（≤cap）。
    virtual int CollectOutputs(OutputDest* dests, int cap) = 0;

    // 消费输出（此刻 dests 缓冲已回填）——推进游戏内部状态。
    virtual void ApplyResult() = 0;

    // 推理链路故障（弃答纪律）：默认实现=标记本局判负。适配器可覆盖
    // （如改走安全兜底策略），但不得静默重试（会撕裂确定性）。
    virtual void OnInferFail() = 0;

    virtual bool IsDone() = 0;
    virtual int  Outcome() = 0;          // 我方视角：1=胜 0=负 -1=平/异常
    virtual bool WeAreFirst() = 0;       // 收账用（先手/后手胜率分列）
    // 逐局指纹（收卷时取一次）：确定性门的强比较面——胜率聚合太粗，
    // 指纹=比分类更强的逐位信号（如比分）。默认 0=不参与比较。
    virtual long long GameFingerprint() { return 0; }

    // 链寿命 TLS 帧（fiber 模式切换点装卸）；无跨让出 TLS 的游戏返回 nullptr。
    virtual ITlsFrame* TlsFrame() = 0;

    // population 路由（演化，判决16）：框架在 NewGame 后按链→个体映射喂入
    // 本局的模型号（路由图 mid 列的来源）。适配器可在 AssembleInto 里把它写进
    // "mid" 输入。非演化负载不调用。缺省=忽略（适配器自管 mid 亦兼容）。
    virtual void SetModelId(int64_t /*mid*/) {}
};

// 适配器工厂：为链 chain 造一个适配器（链内串行复用；腿末 delete）
using AdapterFactory = GameAdapter* (*)(int chain, void* user);

} // namespace inferfarm
