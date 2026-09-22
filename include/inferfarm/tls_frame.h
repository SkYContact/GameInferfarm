// ============================================================
//  inferfarm/tls_frame.h — TLS 帧纪律的结构化落点（血律第一工序）
//
//  fiber 化的第一道工序就是 TLS 审计表（见 docs/pitfalls.md）：同工人多局
//  fiber 共享线程 TLS，跨让出点存活的局内态必须收进"帧"，在工人切换点
//  装卸——否则 t_ctx.bot 指向他局这类串局事故（YGO 案例册有账）。
//  本接口把该纪律变成结构：适配器把链寿命状态装进 ITlsFrame，FiberPool
//  在每次 SwitchToFiber 前后调用 install/uninstall，帧寿命=链（复刻线程
//  模式跨局携带语义）。
//
//  审计清单（适配器作者必过）：
//  1. 枚举自己（与所依赖游戏库）的全部 thread_local / TLS 用点；
//  2. 逐点判定：仅局内使用不跨让出？→ 留 TLS；跨让出存活？→ 进帧；
//  3. 进程级（g_ 前缀 + 锁保护）→ 留全局，注明线程安全性依据；
//  4. 不确定的点：保守进帧 + 兜底 reset。
// ============================================================
#pragma once

namespace inferfarm {

class ITlsFrame {
public:
    virtual ~ITlsFrame() = default;
    // 工人切入对局 fiber 前调用（首跑/恢复同路）：把帧内状态装回线程 TLS
    virtual void Install() = 0;
    // 工人从对局 fiber 返回后调用（让出/收卷同路）：卸载/复位
    virtual void Uninstall() = 0;
};

} // namespace inferfarm
