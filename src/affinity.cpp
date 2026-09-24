// affinity.cpp — 绑核实现（Windows；2026-09-24）。契约见 affinity.h。
#include "inferfarm/affinity.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace inferfarm {

namespace {
std::atomic<int>& PinnedCounter() {
    static std::atomic<int> n{0};
    return n;
}

void WarnSeg(const char* spec, const std::string& seg) {
    std::fprintf(stderr, "[affinity] 段 \"%s\"（源 \"%s\"）非法，跳过\n",
                 seg.c_str(), spec ? spec : "");
}

// "phys" 段：每物理核取最低逻辑位（全局核号=64×组号+组内位号；单组机=朴素
// 编号）。任何失败路径都留 stderr 痕迹——静默空列表曾把 G15 门空过成假绿
// （首查 GLE 非预期值，2026-09-24）。
void AppendPhysicalCores(std::vector<int>* out, bool* ok) {
#ifdef _WIN32
    *ok = false;
    DWORD len = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len)) {
        const DWORD gle = GetLastError();
        if (gle != ERROR_INSUFFICIENT_BUFFER || len == 0) {
            std::fprintf(stderr, "[affinity] phys 枚举：首查失败 GLE=%lu len=%lu\n",
                         gle, len);
            return;
        }
    }
    std::vector<BYTE> buf((size_t)len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)buf.data(), &len)) {
        std::fprintf(stderr, "[affinity] phys 枚举：二查失败 GLE=%lu len=%lu\n",
                     GetLastError(), len);
        return;
    }
    int n_cores = 0;
    BYTE* p = buf.data();
    BYTE* end = buf.data() + len;
    // 步进锚=记录头 8 字节（Relationship+Size），**不能用 sizeof(EX)**：
    // VS18 新 SDK 结构扩容到 80 而实记录 48——sizeof 当最小步长把首条
    // 误判截断，phys 枚举恒空（2026-09-24 实案）。字段同理**手动偏移读**
    // （x64 实布局，python ctypes 逐字节核过）：头 8 + Flags@8 + EffClass@9
    // + Reserved[20]@10 + GroupCount@30 + GroupMask[0]@32（Mask@32、
    // Group@40）——新 SDK 头的字段偏移与实记录脱节，按头读会越界。
    constexpr size_t kRecHdr = sizeof(DWORD) * 2;
    while (p + kRecHdr <= end) {
        auto* rec = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)p;
        if (rec->Size < kRecHdr)
            break;
        if (rec->Relationship == RelationProcessorCore
            && rec->Size >= 48) {   // 单组面满记录：8 头 + 24 处理器段 + 16 组亲和
            ULONG64 m = 0;
            WORD grp = 0;
            std::memcpy(&m, p + 32, sizeof(m));
            std::memcpy(&grp, p + 40, sizeof(grp));
            if (m) {
                int idx = 0;
                for (ULONG64 b = m; !(b & 1); b >>= 1) idx++;
                out->push_back(64 * (int)grp + idx);
            }
            n_cores++;
        }
        p += rec->Size;
    }
    if (n_cores == 0)
        std::fprintf(stderr, "[affinity] phys 枚举：记录遍历零核（len=%lu）\n", len);
    *ok = true;
#else
    (void)out;
    *ok = false;
#endif
}
} // namespace

std::vector<int> ParseCpuList(const char* spec) {
    std::vector<int> out;
    if (!spec || !*spec) return out;
    size_t i = 0;
    const std::string s(spec);
    while (i <= s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        std::string seg = s.substr(i, j - i);
        // trim 空白
        const auto n1 = seg.find_first_not_of(" \t");
        const auto n2 = seg.find_last_not_of(" \t");
        seg = (n1 == std::string::npos) ? std::string()
                                        : seg.substr(n1, n2 - n1 + 1);
        if (!seg.empty()) {
            if (seg == "phys") {
                bool ok = false;
                AppendPhysicalCores(&out, &ok);
                if (!ok)
                    std::fprintf(stderr, "[affinity] phys 枚举失败（源 \"%s\"），段跳过\n", spec);
            } else if (seg.find_first_not_of("0123456789-") == std::string::npos) {
                const size_t dash = seg.find('-');
                const int a = atoi(seg.c_str());
                const int b = dash == std::string::npos
                                  ? a : atoi(seg.c_str() + dash + 1);
                // 负核号拒绝（"-1" 曾被当区间 [-1,1] 吃成假元素，G15 门空过）
                if (a < 0 || b < 0 || (dash == std::string::npos && seg[0] == '-')) {
                    WarnSeg(spec, seg);
                } else if (dash == std::string::npos || a <= b) {
                    for (int c = a; c <= b; c++) {
                        // 去重保序
                        size_t k = 0;
                        for (; k < out.size() && out[k] != c; k++) {}
                        if (k == out.size()) out.push_back(c);
                    }
                } else {
                    WarnSeg(spec, seg);
                }
            } else {
                WarnSeg(spec, seg);
            }
        }
        if (j == s.size()) break;
        i = j + 1;
    }
    return out;
}

bool PinThread(const std::vector<int>& cpus, int id, const char* role) {
    if (cpus.empty()) return false;
    const int cpu = cpus[(size_t)id % cpus.size()];
#ifdef _WIN32
    ULONG_PTR proc_mask = 0, sys_mask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask)) {
        std::fprintf(stderr, "[affinity] %s#%d GetProcessAffinityMask 失败 GLE=%lu\n",
                     role, id, GetLastError());
        return false;
    }
    // 范围检查先于移位（越界移位是 UB：x86 移位量 mod 64 会静默绕回——
    // 9999 号核曾借此混过检查，G15 越界门抓到）
    if (cpu < 0 || cpu > 62) {
        std::fprintf(stderr, "[affinity] %s#%d -> cpu %d 越界（单组面 0-62），跳过\n",
                     role, id, cpu);
        return false;
    }
    const ULONG_PTR m = (ULONG_PTR)1 << cpu;
    if (!(m & proc_mask)) {
        std::fprintf(stderr, "[affinity] %s#%d -> cpu %d 不在进程亲和集，跳过\n",
                     role, id, cpu);
        return false;
    }
    const ULONG_PTR old = SetThreadAffinityMask(GetCurrentThread(), m);
    if (!old) {
        std::fprintf(stderr, "[affinity] %s#%d -> cpu %d SetThreadAffinityMask 失败 GLE=%lu\n",
                     role, id, cpu, GetLastError());
        return false;
    }
    PinnedCounter().fetch_add(1, std::memory_order_relaxed);
    std::printf("[affinity] %s#%d -> cpu %d\n", role, id, cpu);
    return true;
#else
    std::fprintf(stderr, "[affinity] %s#%d 非平台支持，跳过\n", role, id);
    return false;
#endif
}

int AffinityPinnedCount() { return PinnedCounter().load(std::memory_order_relaxed); }

} // namespace inferfarm
