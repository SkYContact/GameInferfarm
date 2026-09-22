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
//    FARM_CACHE_LOG2          推理缓存表容 log2（0=关；如 16=64K 条）——
//                             键=组装行字节哈希+权重代次（KataGo NNCache
//                             思想吸收，判决13；有重复状态的游戏红利大）
// ============================================================
#pragma once
#include "backend.h"
#include "bank.h"
#include "cache.h"
#include "census.h"
#include "fiber_pool.h"
#include "game_adapter.h"
#include "types.h"

namespace inferfarm {

// 多权重模式（population 路由的一等负载面，判决16）：这不是"演化功能"——
// 演化/权重评测/批量测试都是本模式的乘客。框架只管：模型数×每模型局数=链数、
// 链 c → 模型 c%P（映射+SetModelId 喂适配器）、按 Outcome() 契约逐模型计数
// （与全局胜负统计同一泛型层）。**结算语义与框架无关**：适应度/模型对比/
// 通过率等解释归驱动侧读 tally 自行定义。
struct PopulationConfig {
    int models = 0;        // P（0=关=普通单权重负载）；须与路由图 pop 面容量一致
    int games_each = 8;    // 每模型局数（链数=P×games_each，每链 1 局全并发）
};

// 设备组配置（多 GPU）：一组=一套后端+模型配置+银行数（如 NVIDIA 主卡 2 家 +
// AMD 核显 1 家；或双 NVIDIA 各 N 家）。backend/device_id/ort_ep/路径全组独立。
struct DeviceConfig {
    ModelConfig model;
    int banks = 2;
    double share = 1.0;   // 链分配权重（平滑加权轮询）：异构卡速差大时按算力
                          // 配比（如主卡 share=4 核显 share=1）；0=该组不接链。
                          // 缺省 1=均分（两组时≡c%n_groups 老行为）
    int slots = 0;        // 0=统一形状（cfg.slots）；>0=本组批形状（异构小图，
                          // 仅非主组有意义；行宽/输入名/输出宽须与主组一致）
};

struct FarmConfig {
    std::string name = "farm";
    int chains = 8;
    int games = 64;               // 总局数（每链 ceil(games/chains)）
    uint32_t seed0 = 4242;
    bool alternate_first = true;  // 偶数局我方先攻
    bool fibers = true;
    int workers = 0;              // 0=物理核≈hc/2
    int banks = 4;                // 0=inline；多设备时被 devices 各组 banks 覆盖
    int slots = 64;
    double window_ms = 0.2;
    double window_floor = 0.2;
    double stagger_ms = 10;       // 到达层羊群判决：错峰甜点
    long long max_decisions = 1000000;   // 对局决策数护栏（防适配器死循环）
    bool census = false;
    int cache_log2 = 0;          // 推理缓存：0=关（缺省零行为差）；如 16=64K 条
    PopulationConfig population;   // 多权重路由负载（models>0=启用；与 population_input 家族对齐）
    // 多设备组（判决15；空=单设备老行为=cfg.model+cfg.banks）。链 c 钉扎到
    // 组 c%devices.size()——异构设备（如 NVIDIA+AMD）下保跨跑逐位的关键。
    // 各组模型结构须一致（输入名/行宽、输出名/宽、slots）；权重可不同
    // （int8/fp16 各卡一档）——钉扎保证每链恒用同组。
    std::vector<DeviceConfig> devices;
    ModelConfig model;
};

struct FarmTally {
    int first_wins = 0, first_total = 0;
    int second_wins = 0, second_total = 0;
    int games_done = 0, infer_fails = 0;
    long long decisions = 0;
    unsigned long long fingerprint = 0;   // 逐局指纹 XOR（顺序无关；逐位门用）
    unsigned long long cache_lookups = 0, cache_hits = 0;   // 推理缓存（开后才有数）
    // 演化模式按个体收账（适应度）：链 c → 个体 c%P（演化模式才非空）
    std::vector<int> model_wins, model_games;
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
    // ORT=不支持返回 false）。多设备组=全组成功才算成（任一组不支持即 false，
    // 不留半换心农场）；成功后缓存代次同步失效。
    bool RefitWeights(const char* rw1_path) {
        if (group_bes_.empty()) return false;
        for (InferBackend* be : group_bes_)
            if (!be->RefitWeights(rw1_path)) return false;
        infer_gen_++;   // 换心=旧缓存全表逻辑失效（代次门，不清表）
        return true;
    }

    // population 代际换权重（演化路由，判决16）：host=[P, flat_w] f32 种群平面
    // → 写满全部银行会话的 pop 输入 + 缓存代次失效（SetPopulation 后新代
    // 决策用新权重；旧代条目永不再命中）。前置条件=腿已返回（同 RefitWeights
    // 纪律）。false=非路由模式/任一银行写入失败
    bool SetPopulation(const void* host) {
        if (!bank_ || cfg_.model.population_input.empty()) return false;
        for (int bk = 0; bk < bank_->banks(); bk++)
            if (!bank_->SetPopulation(bk, cfg_.model.population_input.c_str(), host))
                return false;
        infer_gen_++;
        return true;
    }

    const FarmTally& tally() const { return tally_; }
    const FarmConfig& config() const { return cfg_; }
    ModelSpec* spec() { return spec_ok_ ? &spec_ : nullptr; }
    InferBackend* backend() { return backend_; }
    BankScheduler* bank() { return bank_ ? &bank_obj_ : nullptr; }
    Census* census() { return &census_; }

    void Shutdown();

    // 驱动环本体（对局 fiber 上；也供 inline/线程模式同构调用）
    void DriveGame(GameAdapter* g, uint64_t seed, bool we_first,
                    int chain_id = 0, int game_id = 0);
    // 决策一步（银行/inline 分流；grp=设备组（链钉扎）；返回 false=判负纪律已触发）
    bool DriveDecision(GameAdapter* g, int grp = 0);

private:
    // 组装行字节 → 缓存键（逐输入 InputRow 全行宽；槽独占期内调用安全；
    // grp 掺入键=缓存设备命名空间——异构组同字节行输出逐位可异，不共享条目）
    CacheKey128 HashSlot(int bk, int sl, int grp);
    FarmConfig cfg_;
    InferBackend* backend_ = nullptr;      // =组 0 后端（inline/兼容面）
    std::vector<InferBackend*> group_bes_; // 每设备组一个后端实例（Farm 建/毁）
    std::vector<ModelSpec> group_specs_;   // 每组模型规格（slots=组实际形状）
    int n_dev_ = 1;                        // 设备组数（链钉扎 c%n_dev_）
    std::vector<int> chain_grp_;           // 链→组（平滑加权轮询，share 配比）
    BankScheduler bank_obj_;
    BankScheduler* bank_ = nullptr;
    InlineRunner inline_;
    FiberPool pool_;
    Census census_;
    ModelSpec spec_;
    bool spec_ok_ = false;
    bool timer_armed_ = false;
    InferCache cache_;            // 推理缓存（cache_log2>0 时启）
    uint64_t infer_gen_ = 1;      // 权重代次：换心成功即 ++
    FarmTally tally_;
    std::mutex tally_mx_;
    void NoteGameDone(bool we_first, int outcome, long long dec, bool infer_fail,
                      long long fingerprint, int chain_id, int game_id);
    friend struct FarmGameCtx;
};

} // namespace inferfarm
