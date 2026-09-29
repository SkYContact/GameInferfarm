// ============================================================
//  state_gather.h — 状态面索引 gather/scatter 行拷内核（DATA13）
//
//  病灶：填充+散射两侧 768+768 次逐行 cudaMemcpyAsync D2D，每次 ~6.4µs
//  CE 固定开销，22MB 本该 0.15ms 搬完实际 ~10ms/批（提交端批化
//  FARM_STATE_D2D_BATCH 只省提交，执行端税原封）。本件一次 launch 搬全面。
//
//  语义契约（与逐行路径逐位同）：
//    idx_dev[i] ∈ {-1, [0, prows]}，i=行号 r：
//      -1       → 跳过（幻影行：目标行留陈旧内容，与逐行路径 continue 同）
//      prows    → 源/目标 = pool_base + prows*rb（池末保留零行；zero_pending
//                 行语义——旗标清理由调用方在**主机侧建表时**完成，内核零原子）
//      [0,prows)→ 正常池行
//    mode 0=gather（pool→rows：dst=rows_base+r*rb，src=pool+idx*rb）
//    mode 1=scatter（rows→pool：dst=pool+idx*rb，src=rows_base+r*rb）
//
//  实现面：libcuda 驱动 API 动态装载（dlopen/LoadLibrary，零链接依赖）+
//  内嵌 PTX（cuModuleLoadData，无需 nvcc）。launch 前须有当前 CUDA 上下文
//  （调用方=trt_backend 工作线程，runtime API 已隐式绑主上下文——本件用
//  cuDevicePrimaryCtxRetain+cuCtxSetCurrent 补齐驱动侧同上下文，幂等）。
// ============================================================
#pragma once
#include <cstddef>

namespace inferfarm {

// 一次性装载（libcuda + PTX 模块；线程安全由调用方序列化——Init 均在
// 后端 Init 路径）。返回 false=装载失败（调用方回退旧逐行路径）。
bool StateGatherInit();

// 单次行集拷贝（流内序；rc=false=launch 失败，调用方回退）。
// idx_dev：设备端 int32 表（n 项）；rb=行字节数（内核内 16B 主循环+尾字节）。
bool StateGatherLaunch(void* rows_base, void* pool_base, const int* idx_dev,
                       int n, size_t rb, void* stream, int mode);

// 自检（bench 用）：编译目标/模块加载是否可用。
bool StateGatherReady();

} // namespace inferfarm
