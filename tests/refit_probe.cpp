// refit_probe.cpp — 隔离验证：CPU 后端会话权重在 RefitWeights 后是否真被计算消费。
#include "../examples/toy/toy_adapter.h"
#include "inferfarm/backend_factory.h"
#include <cstdio>
#include <vector>
using namespace inferfarm;
using namespace inferfarm::toy;

int main() {
    ModelConfig cfg;
    cfg.backend = "cpu";
    cfg.cpu = ToyModelDecl(4);
    InferBackend* be = CreateCpuBackend();
    ModelSpec spec;
    if (!be->LoadSpec(cfg, 4, spec)) return 2;
    void* s1 = be->CreateSession(cfg, spec, true);
    void* s2 = be->CreateSession(cfg, spec, true);
    if (!s1 || !s2) return 2;
    be->Warmup(s1);
    be->Warmup(s2);
    // 填两行不同 obs
    float* obs1 = (float*)be->InputRow(s1, "obs", 0, nullptr);
    float* obs2 = (float*)be->InputRow(s1, "obs", 1, nullptr);
    for (int e = 0; e < 8; e++) {
        obs1[(size_t)e] = 0.3f * (float)(e + 1);
        obs2[(size_t)e] = -0.2f * (float)(e + 1);
    }
    unsigned seq = 0;
    be->SubmitBatch(s1, 2, seq);
    const float* p1_0 = be->OutputRow(s1, "policy", 0);
    const float* p1_1 = be->OutputRow(s1, "policy", 1);
    std::vector<float> before0(p1_0, p1_0 + 4), before1(p1_1, p1_1 + 4);
    std::printf("s1 row0 policy: %.4f %.4f %.4f %.4f\n", before0[0], before0[1], before0[2], before0[3]);
    const float* p2_0 = be->OutputRow(s2, "policy", 0);
    std::printf("s2 row0 policy（同输入应为同值）: %.4f %.4f %.4f %.4f\n",
                p2_0[0], p2_0[1], p2_0[2], p2_0[3]);
    // 会话2未写输入——先给它同输入再比
    float* obs2b = (float*)be->InputRow(s2, "obs", 0, nullptr);
    for (int e = 0; e < 8; e++) obs2b[(size_t)e] = 0.3f * (float)(e + 1);
    unsigned seq2 = 0;
    be->SubmitBatch(s2, 1, seq2);
    p2_0 = be->OutputRow(s2, "policy", 0);
    std::printf("s2 row0 policy（写入同输入后）: %.4f %.4f %.4f %.4f\n",
                p2_0[0], p2_0[1], p2_0[2], p2_0[3]);
    // RW1：行 0 取反
    std::vector<char> blob;
    auto put32 = [&](uint32_t v) { for (int i = 0; i < 4; i++) blob.push_back((char)(v >> (8 * i))); };
    auto put16 = [&](uint16_t v) { for (int i = 0; i < 2; i++) blob.push_back((char)(v >> (8 * i))); };
    std::vector<float> w8(8);
    for (int e = 0; e < 8; e++) w8[(size_t)e] = 0.25f * ((e % 2) ? -1.0f : 1.0f);
    blob.insert(blob.end(), {'R', 'W', '1', '\0'});
    put32(1); put32(1);
    std::string nm = "policy.Wobs.0";
    put16((uint16_t)nm.size());
    blob.insert(blob.end(), nm.begin(), nm.end());
    blob.push_back((char)2);
    put32(8);
    const char* wp = (const char*)w8.data();
    blob.insert(blob.end(), wp, wp + 32);
    FILE* f = fopen("probe.rw1", "wb");
    fwrite(blob.data(), 1, blob.size(), f);
    fclose(f);
    if (!be->RefitWeights("probe.rw1")) { std::printf("refit FAIL\n"); return 1; }
    be->SubmitBatch(s1, 2, seq);
    p1_0 = be->OutputRow(s1, "policy", 0);
    p1_1 = be->OutputRow(s1, "policy", 1);
    std::printf("s1 row0 after refit: %.4f %.4f %.4f %.4f（期望≠before）\n",
                p1_0[0], p1_0[1], p1_0[2], p1_0[3]);
    // 权重全行共享：行 1 的 action0（refit 的 k=0）会变，action1-3 不变
    bool row0_changed = std::vector<float>(p1_0, p1_0 + 4) != before0;
    bool row1_tail_same = p1_1[1] == before1[1] && p1_1[2] == before1[2]
        && p1_1[3] == before1[3];
    std::printf("s1 row1 after refit: %.4f %.4f %.4f %.4f（action1-3 应不变；"
                "action0 允许变=权重全行共享）\n",
                p1_1[0], p1_1[1], p1_1[2], p1_1[3]);
    std::printf("%s\n", row0_changed && row1_tail_same ? "PROBE PASS" : "PROBE FAIL");
    be->DestroySession(s1);
    be->DestroySession(s2);
    delete be;

    // ---- Farm 全链段：银行协议下 refit 输出对比（same vs diff blob）----
    auto farm_policy = [&](const char* blob, float* out4) {
        FarmConfig cfg;
        cfg.name = "probe";
        cfg.chains = 1;
        cfg.games = 1;
        cfg.banks = 2;
        cfg.slots = 4;
        cfg.workers = 2;
        cfg.stagger_ms = 0;
        cfg.model.backend = "cpu";
        cfg.model.cpu = ToyModelDecl(4);
        Farm farm;
        if (!farm.Init(cfg)) { std::printf("farm init FAIL\n"); std::exit(1); }
        if (blob && !farm.RefitWeights(blob)) { std::printf("farm refit FAIL\n"); std::exit(1); }
        BankScheduler* bk = farm.bank();
        int b = -1, sl = -1;
        if (!bk->Claim(b, sl)) { std::printf("claim FAIL\n"); std::exit(1); }
        float* obs = (float*)bk->InputRow(b, sl, "obs", nullptr);
        for (int e = 0; e < 8; e++) obs[(size_t)e] = 0.3f * (float)(e + 1);
        OutputDest d[1];
        d[0].name = "policy";
        d[0].dst = out4;
        d[0].n = 4;
        if (!bk->SubmitWait(b, sl, d, 1)) { std::printf("submit FAIL\n"); std::exit(1); }
    };
    {
        float pa[4], pb[4];
        farm_policy("probe.rw1", pa);        // 行0=+0.25 交替
        // 反转 blob：同法重造（行0=-0.25 交替）
        std::vector<float> w8n(8);
        for (int e = 0; e < 8; e++) w8n[(size_t)e] = -w8[(size_t)e];
        std::vector<char> nblob;
        nblob.insert(nblob.end(), {'R', 'W', '1', '\0'});
        auto np32 = [&](uint32_t v) { for (int i = 0; i < 4; i++) nblob.push_back((char)(v >> (8 * i))); };
        auto np16 = [&](uint16_t v) { for (int i = 0; i < 2; i++) nblob.push_back((char)(v >> (8 * i))); };
        np32(1);
        np32(1);
        np16((uint16_t)nm.size());
        nblob.insert(nblob.end(), nm.begin(), nm.end());
        nblob.push_back((char)2);
        np32(8);
        const char* np = (const char*)w8n.data();
        nblob.insert(nblob.end(), np, np + 32);
        FILE* fneg = fopen("probe_neg.rw1", "wb");
        fwrite(nblob.data(), 1, nblob.size(), fneg);
        fclose(fneg);
        farm_policy("probe_neg.rw1", pb);
        bool farm_differs = std::vector<float>(pa, pa + 4) != std::vector<float>(pb, pb + 4);
        std::printf("farm 链路 policy(+0.25): %.4f %.4f %.4f %.4f / (-0.25): %.4f %.4f %.4f %.4f → %s\n",
                    pa[0], pa[1], pa[2], pa[3], pb[0], pb[1], pb[2], pb[3],
                    farm_differs ? "DIFF(好)" : "SAME(坏=腿没吃到新权重)");
    }
    // ---- G5 完全复刻：完整腿 + 玩具适配器 + same/diff blob ----
    {
        auto farm_leg_fp = [&](const char* blob) {
            FarmConfig cfg;
            cfg.name = "probe5";
            cfg.chains = 4;
            cfg.games = 24;
            cfg.banks = 2;
            cfg.slots = 8;
            cfg.workers = 4;
            cfg.stagger_ms = 1;
            cfg.model.backend = "cpu";
            cfg.model.cpu = ToyModelDecl(cfg.slots);
            Farm farm;
            if (!farm.Init(cfg)) { std::printf("init FAIL\n"); std::exit(1); }
            if (blob && !farm.RefitWeights(blob)) { std::printf("refit FAIL\n"); std::exit(1); }
            farm.RunLeg(MakeToyAdapter, nullptr);
            return farm.tally().fingerprint;
        };
        // 造 same/diff（与 farm_test 同构：12 条目全取反=diff）
        std::vector<char> sb, db;
        auto mk = [](std::vector<char>& b, bool neg) {
            auto p32 = [&](uint32_t v) { for (int i = 0; i < 4; i++) b.push_back((char)(v >> (8 * i))); };
            auto p16 = [&](uint16_t v) { for (int i = 0; i < 2; i++) b.push_back((char)(v >> (8 * i))); };
            float sgn = neg ? -1.0f : 1.0f;
            b.insert(b.end(), {'R', 'W', '1', '\0'});
            p32(1);
            p32(12);
            for (int k = 0; k < 4; k++) {
                char n[64];
                snprintf(n, sizeof n, "policy.Wobs.%d", k);
                std::vector<float> wo(8);
                for (int e = 0; e < 8; e++)
                    wo[(size_t)e] = sgn * 0.25f * (float)(k + 1) * ((e % 2) ? -1.0f : 1.0f);
                p16((uint16_t)strlen(n));
                b.insert(b.end(), n, n + strlen(n));
                b.push_back((char)2);
                p32(8);
                const char* p = (const char*)wo.data();
                b.insert(b.end(), p, p + 32);
                snprintf(n, sizeof n, "policy.Wmask.%d", k);
                std::vector<float> wm(4);
                for (int e = 0; e < 4; e++) wm[(size_t)e] = sgn * 0.2f * (float)(k + 2);
                p16((uint16_t)strlen(n));
                b.insert(b.end(), n, n + strlen(n));
                b.push_back((char)2);
                p32(4);
                p = (const char*)wm.data();
                b.insert(b.end(), p, p + 16);
                snprintf(n, sizeof n, "policy.Wcodes.%d", k);
                std::vector<float> wc(1, sgn * 0.3f);
                p16((uint16_t)strlen(n));
                b.insert(b.end(), n, n + strlen(n));
                b.push_back((char)2);
                p32(1);
                p = (const char*)wc.data();
                b.insert(b.end(), p, p + 4);
            }
        };
        mk(sb, false);
        mk(db, true);
        FILE* f1 = fopen("probe5_same.rw1", "wb");
        fwrite(sb.data(), 1, sb.size(), f1);
        fclose(f1);
        FILE* f2 = fopen("probe5_diff.rw1", "wb");
        fwrite(db.data(), 1, db.size(), f2);
        fclose(f2);
        unsigned long long fa = farm_leg_fp("probe5_same.rw1");
        unsigned long long fb = farm_leg_fp("probe5_same.rw1");
        unsigned long long fd = farm_leg_fp("probe5_diff.rw1");
        std::printf("G5 复刻: same=%016llx same2=%016llx(%s) diff=%016llx(%s)\n",
                    fa, fb, fa == fb ? "同✓" : "异✗", fd,
                    fa != fd ? "变✓" : "同✗");
    }
    return (row0_changed && row1_tail_same) ? 0 : 1;
}
