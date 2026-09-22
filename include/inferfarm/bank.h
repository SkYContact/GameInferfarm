// ============================================================
//  inferfarm/bank.h — 零拷贝槽位银行制（资产2，bank_contract 2026-09-22）
//
//  用户定案：组装直写固定内存+定时发车有多少发多少；同进程内拷来拷去
//  是不对的。结构（协调人三笔定案，勿凭直觉推翻）：
//
//    N 家银行 × slots 槽 × 行宽，pinned、启动一次分配、**地址终身固定**；
//    图回放烧死输入地址 → **每家银行一张专属预捕获图**（图与地址一夫一妻）；
//    银行池**线性生命周期**：出池→填充→发车→收割→还池（不复用不轮替，
//    还池点在单线程收货路径=零同步；池容量=在飞上限=天然背压）；
//
//  领号（游标制）：填充银行唯一；**先占在途名额 inflight++ 再 cursor 领号**
//  （防撕裂竞态：占额者必完工）；领号序=在途序 ⇒ close-drain 归零时游标
//  已终态，有效行=前缀连续 [0,n)。闭舱后迟来残号由 state 复检拒绝=虚增量
//  在发车侧按作废槽跳过，账号自洽。
//
//  发车：满座或 window 闹钟 → 闭舱（CAS 幂等）→ 自旋等 inflight==0
//  （组装无挂起点 ⇒ drain 有界 µs 级，**不写超时不写迁移**）→ 前缀部分
//  h2d（n>7/8·slots 走整块；尾行读显存旧数据无害——行独立前提）→ 异步
//  发射不等回程。满座**自驱发车**：最后完笔者就地发（"满=人人写完"由
//  +1 先于领号的构造成立，CAS 输=他人已关舱）。
//
//  收割：调度台轮询在途银行完成旗标（volatile 读）→ 逐 req 拷输出到
//  适配器缓冲（**拷贝承重：回池先于游戏恢复，适配器不得直读银行内存**）
//  → FiberPost 回投 → 还池。
//
//  窗的真相（Windows）：timeBeginPeriod(1) 下 cv 定时量子 ≈1ms——µs 精钟
//  忠实执行 0.1ms 会批量饿死；window_floor=0.2ms 实测最优（本机扫：
//  0.2→129.6 / 1.0→100.4=旧路径平价点）。
// ============================================================
#pragma once
#include "backend.h"
#include "game_adapter.h"
#include "census.h"
#include "types.h"
#include <deque>
#include <mutex>

namespace inferfarm {

struct BankConfig {
    int banks = 4;               // N：银行家数=池容量=在飞批上限（本机最优 4）
    int slots = 64;              // 每银行槽数（=模型批形状 dim0）
    double window_ms = 0.2;      // 攒批窗
    double window_floor = 0.2;   // 有效窗底限（ms；Windows 定时量子勘误）
};

class BankScheduler {
public:
    BankScheduler() = default;
    void Bind(InferBackend& be, Census* cen) { be_ = &be; cen_ = cen; }

    // 建池+起调度台。**在专用调度台线程上建会话**（ORT 图会话 PerThreadContext
    // 铁律：创建/热身/回放须同线程；TRT 同规更稳）。含图地址烧死小实验门：
    // 不过=拒绝启动（字节安全性不赌）。阻塞至就绪或失败。
    bool Init(const BankConfig& cfg, const ModelConfig& mcfg, ModelSpec* spec_out = nullptr);
    void Shutdown();

    bool active() const { return banks_ > 0; }

    // ---- 写手侧（对局 fiber 或 OS 线程）----
    // 领槽（自旋+等池背压；成功即行清零=零基组装契约：未写区与"python
    // 零垫"逐位同）。false 仅当银行未启用。
    bool Claim(int& bank, int& slot);
    // 组装直写面：该槽该输入的行首指针（零拷贝——大数组组装期直接写这里）
    void* InputRow(int bank, int slot, const char* name, size_t* row_bytes);
    // 组装兜底拷贝：小输入从临时缓冲拷进槽行+行尾清零（YGO 的 scal/act_code
    // 后补路径；dst=Row(name)；rb<bytes 时补零）。
    // 提交+等待：登记 dests → inflight--（完工信号）→ 满座自驱判定 →
    // fiber 让出/cv 等 → 恢复即输出已拷进 dests。false=本前向判负纪律。
    // ⚠ 组装与提交之间不得有挂起点（drain 有界的前提）。
    bool SubmitWait(int bank, int slot, const OutputDest* dests, int n_dests);
    // 弃槽（异常路径）：作废槽（发车跳过）+完工照减（drain 不堵）
    void Abandon(int bank, int slot);

    // ---- 观测（[bank] 行每 300 回信一行：srv-lat/批均/在飞/背压）----
    void PrintStatsIfDue(bool force = false);

    // inline（无银行）路径的会话/锁：Farm 用（row0 专用，整批照发=垃圾行无害）
    InferBackend& backend() { return *be_; }

    struct Impl;   // 公有：调度台/收割自由函数引用（bank.cpp 内）
private:
    Impl* impl_ = nullptr;
    BankConfig cfg_;
    int banks_ = 0;
    InferBackend* be_ = nullptr;
    Census* cen_;
    ModelSpec spec_;
};

// inline 模式运行器：单会话+互斥，整批照发（fb 行，垃圾行无害），读 row0。
// 与银行路径逐位一致的前提=行独立（同模型同图形状）。仅供对照/低负载/
// 确定性门；生产形态=银行制。
class InlineRunner {
public:
    bool Init(InferBackend& be, const ModelConfig& cfg, const ModelSpec& spec);
    void Shutdown();
    // 整个决策在锁内（正确性优先；生产性能形态=银行制）：清零 row0 → 组装
    // → 整批照发（垃圾行无害）→ 等完 → 回填 dests。与银行路径逐位一致
    // 的前提=行独立+同批形状。
    bool Run(GameAdapter* g);
    struct Session;
private:
    Session* s_ = nullptr;
};

} // namespace inferfarm
