// ============================================================
//  inferfarm/refit.h — RW1 权重 blob（资产3：毫秒级换心）
//
//  RW1 格式（小端）：
//    magic "RW1\0"(4B) | u32 version=1 | u32 n_entries
//    entry: u16 name_len | name(utf8) | u8 dtype(0=int8,1=f16,2=f32,3=int64)
//         | u32 numel | 原始字节(C 序)
//  权威导出器=tools/refit_blob.py（θ→blob；量化尺度 parent 定死=候选比较
//  共模自洽；np.round 银行家舍入与 ORT 逐位一致）。
//
//  语义（换心门的支点）：
//    · blob 名字不在引擎 refit 名单 → warning 跳过；
//    · 引擎需要但 blob 没有 → 保持引擎原值（部分换心合法）；
//    · 名单内 dtype/numel 与原型不一致 → fail fast（绝不静默错心）；
//    · 死区（TRT 融合闭包锁死项，如 dyt.alpha 族）→ 防御跳过。
// ============================================================
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace inferfarm {

struct Rw1Entry {
    std::string name;
    uint8_t dtype = 0;        // 0=int8 1=f16 2=f32 3=int64
    uint32_t numel = 0;
    const char* data = nullptr;   // 指向 blob 整块内 carve（blob 持有至换心返回）
    size_t bytes = 0;
};

// 解析 RW1（文件→blob 字节+条目表）。逐字节校验：越界/截断/尾部残余/版本
// 不符全部 fail fast。
bool ParseRw1(const std::string& path, std::vector<char>& blob,
              std::vector<Rw1Entry>& out);

// 小工具：dtype→字节数（0,1,2,3 → 1,2,4,8）；未知=0
size_t Rw1DtypeSize(uint8_t dtype);

} // namespace inferfarm
