// toy_adapter.h — 最小示范适配器：宝石收集 MDP。
//
// 演示面：具名多输入（f32/i64）多输出、掩码决策、确定性种子协议、链寿命
// TLS 帧的正确用法（thread_local 指针在工人切换点装卸）、推理故障判负。
// 每局 T_TOYS 回合，每回合 1 决策；特征=状态的确定函数（零随机之外来）。
#pragma once
#include "inferfarm/inferfarm.h"
#include <cstdio>
#include <cstring>
#include <random>

namespace inferfarm {
namespace toy {

static const int kObsDim = 8;    // "obs" 行宽（f32）
static const int kActN = 4;      // 动作数（=policy 宽 + mask 宽）
static const int kCodes = 3;     // "codes" 行宽（i64）
static const int kTurns = 8;     // 每局决策数

// ---- 链寿命 TLS 帧示范：跨让出存活的"链内计数"住帧不住 thread_local ----
struct ToyChainTls {
    int chain = 0;
    long long decisions_this_chain = 0;   // 链累计（跨局携带=线程模式语义）
};
static thread_local ToyChainTls* t_toy_tls = nullptr;

struct ToyFrame : ITlsFrame {
    ToyChainTls* st = nullptr;
    ToyChainTls* saved = nullptr;
    explicit ToyFrame(ToyChainTls* s) : st(s) {}
    void Install() override { saved = t_toy_tls; t_toy_tls = st; }
    void Uninstall() override { t_toy_tls = saved; }
};

// 确定性 LCG（32 位；种子协议唯一随机源）
static inline uint32_t ToyLcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s >> 8;
}

struct ToyAdapter : GameAdapter {
    // 链寿命状态（帧携带；帧由适配器提供）
    ToyChainTls chain_tls;
    ToyFrame frame{&chain_tls};

    // 局内状态
    uint32_t rng = 0;
    int turn = 0;
    int score_us = 0, score_them = 0;
    bool we_first_ = true;
    bool infer_failed = false;

    // 输出缓冲（CollectOutputs 申报；收割回填）
    float policy[kActN] = {0};
    float value = 0;

    explicit ToyAdapter(int chain) { chain_tls.chain = chain; }

    void NewGame(uint64_t seed, bool we_first) override {
        rng = (uint32_t)seed | 1u;
        turn = 0;
        score_us = 0;
        score_them = 0;
        we_first_ = we_first;
        infer_failed = false;
    }

    bool AdvanceToDecision() override {
        if (turn >= kTurns) return false;
        return true;
    }

    void AssembleInto(SlotWriter& slot) override {
        // obs：状态确定函数（rng 派生但不推进决策随机源——特征只读）
        float* obs = (float*)slot.Row("obs", nullptr);
        int64_t* codes = (int64_t*)slot.Row("codes", nullptr);
        float* mask = (float*)slot.Row("mask", nullptr);
        if (!obs || !codes || !mask) return;   // 模型签名错位=写不进（判负可见）
        uint32_t r = rng;   // 拷贝推进（不动决策随机源）
        for (int i = 0; i < kObsDim; i++) {
            obs[i] = (float)(ToyLcg(r) % 97) / 97.0f;
            obs[i] += (float)(score_us + score_them + turn) * 0.001f * (float)(i + 1);
        }
        // mask：回合晚期动作 3 合法（确定规则）
        for (int a = 0; a < kActN; a++)
            mask[a] = (a == 3 && turn < kTurns - 2) ? 0.0f : 1.0f;
        codes[0] = turn;
        codes[1] = we_first_ ? 1 : 0;
        codes[2] = (int64_t)(t_toy_tls ? t_toy_tls->decisions_this_chain : 0) % 7;
    }

    int CollectOutputs(OutputDest* dests, int cap) override {
        int n = 0;
        if (n < cap) { dests[n].name = "policy"; dests[n].dst = policy; dests[n].n = kActN; n++; }
        if (n < cap) { dests[n].name = "value"; dests[n].dst = &value; dests[n].n = 1; n++; }
        return n;
    }

    void ApplyResult() override {
        // 掩码 argmax（平局取小号动作——确定）
        int best = -1;
        for (int a = 0; a < kActN; a++) {
            bool legal = !(a == 3 && turn < kTurns - 2);
            if (!legal) continue;
            if (best < 0 || policy[a] > policy[best]) best = a;
        }
        if (best < 0) best = 0;
        if (t_toy_tls) t_toy_tls->decisions_this_chain++;
        // 记分：确定表 + 决策随机源推进（先手有确定微加成）
        static const int kGain[kActN] = {3, 5, 2, 11};
        int gain = kGain[best] + (int)(ToyLcg(rng) % 4);
        int first_bonus = we_first_ ? 1 : 0;
        if ((turn % 2) == 0) score_us += gain + first_bonus;
        else score_them += gain;
        turn++;
    }

    void OnInferFail() override { infer_failed = true; }
    bool IsDone() override { return turn >= kTurns; }
    int Outcome() override {
        if (infer_failed) return 0;
        if (score_us > score_them) return 1;
        if (score_us < score_them) return 0;
        return -1;
    }
    bool WeAreFirst() override { return we_first_; }
    long long GameFingerprint() override {
        return (long long)score_us * 1000003 + score_them;
    }
    ITlsFrame* TlsFrame() override { return &frame; }
};

inline GameAdapter* MakeToyAdapter(int chain, void*) {
    return new ToyAdapter(chain);
}

inline CpuModelDecl ToyModelDecl(int slots) {
    CpuModelDecl d;
    d.slots = slots;
    d.ins.push_back({"obs", DTYPE_F32, {kObsDim}});
    d.ins.push_back({"mask", DTYPE_F32, {kActN}});
    d.ins.push_back({"codes", DTYPE_I64, {kCodes}});
    d.outs.push_back({"policy", kActN});
    d.outs.push_back({"value", 1});
    return d;
}

} // namespace toy
} // namespace inferfarm
