// ============================================================
//  inferfarm/farm.h — 编排器：适配器 × 纤程池 × 银行制 × 取证 一站装配
//
//  种子协议（与 YGO 产线逐位同公式）：per=ceil(games/chains)，链 c 局 i
//  种子 = seed0 + c*per + i；先后攻默认逐局交替（偶数局我方先攻）。
//
//  选通（Config 全显式；env FARM_* 可覆盖，便于快速实验）：
//    FARM_FIBERS=1            fiber 模式（默认开）
//    FARM_FIBER_WORKERS=K     工人数（缺省=物理核≈hc/2；SMT 负资产勿超）
//    FARM_BANKS=N             银行家数（0=inline 模式；本机最优 4）
//    FARM_BANK_WINDOW_FLOOR   有效窗底限 ms（0.2）
//    FARM_STAGGER_MS          点火错峰（10ms 甜点）
//    FARM_CENSUS=1            取证层（~5% 税，仅取证开）
// ============================================================
#pragma once
#include "backend.h"
#include "bank.h"
#include "census.h"
#include "fiber_pool.h"
#include "game_adapter.h"
#include "types.h"

namespace inferfarm {

struct FarmConfig {
    std::string name = "farm";
    int chains = 8;
    int games = 64;               // 总局数（每链 ceil(games/chains)）
    uint32_t seed0 = 4242;
    bool alternate_first = true;  // 偶数局我方先攻
    bool fibers = true;
    int workers = 0;              // 0=物理核≈hc/2
    int banks = 4;                // 0=inline
    int slots = 64;
    double window_ms = 0.2;
    double window_floor = 0.2;
    double stagger_ms = 10;       // 到达层羊群判决：错峰甜点
    long long max_decisions = 1000000;   // 对局决策数护栏（防适配器死循环）
    bool census = false;
    ModelConfig model;
};

struct FarmTally {
    int first_wins = 0, first_total = 0;
    int second_wins = 0, second_total = 0;
    int games_done = 0, infer_fails = 0;
    long long decisions = 0;
    unsigned long long fingerprint = 0;   // 逐局指纹 XOR（顺序无关；逐位门用）
};

class Farm {
public:
    ~Farm() { Shutdown(); }

    // env 覆盖（FARM_*）→ 建后端/银行/inline → 就绪。多腿复用同一 Farm
    // （refit-jobs 形态：RefitWeights→RunLeg→…）。
    bool Init(FarmConfig cfg);

    // 一腿：chains×games 跑完收卷。返回墙钟秒。make: AdapterFactory。
    double RunLeg(AdapterFactory make, void* user);

    // 运行期换心（RW1 blob；仅后端支持时生效——TRT=refitter，CPU=直改，
    // ORT=不支持返回 false）。成功后后续腿用新权重。
    bool RefitWeights(const char* rw1_path) {
        return backend_ ? backend_->RefitWeights(rw1_path) : false;
    }

    const FarmTally& tally() const { return tally_; }
    const FarmConfig& config() const { return cfg_; }
    ModelSpec* spec() { return spec_ok_ ? &spec_ : nullptr; }
    InferBackend* backend() { return backend_; }
    BankScheduler* bank() { return bank_ ? &bank_obj_ : nullptr; }
    Census* census() { return &census_; }

    void Shutdown();

    // 驱动环本体（对局 fiber 上；也供 inline/线程模式同构调用）
    void DriveGame(GameAdapter* g, uint64_t seed, bool we_first);
    // 决策一步（银行/inline 分流；返回 false=判负纪律已触发）
    bool DriveDecision(GameAdapter* g);

private:
    FarmConfig cfg_;
    InferBackend* backend_ = nullptr;
    BankScheduler bank_obj_;
    BankScheduler* bank_ = nullptr;
    InlineRunner inline_;
    FiberPool pool_;
    Census census_;
    ModelSpec spec_;
    bool spec_ok_ = false;
    bool timer_armed_ = false;
    FarmTally tally_;
    std::mutex tally_mx_;
    void NoteGameDone(bool we_first, int outcome, long long dec, bool infer_fail,
                      long long fingerprint);
    friend struct FarmGameCtx;
};

} // namespace inferfarm
