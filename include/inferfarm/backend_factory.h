// backend_factory.h — 后端工厂（cpu|ort|trt|ncnn；trt 需 -DINFERFARM_WITH_TRT=ON，
// ncnn 需 -DINFERFARM_WITH_NCNN=ON 构建）
#pragma once

namespace inferfarm {
class InferBackend;
InferBackend* CreateCpuBackend();
InferBackend* CreateOrtBackend();
InferBackend* CreateTrtBackend();
InferBackend* CreateNcnnBackend();

// fence 桥接诊断（FARM_ORT_ASYNC=3 专属）：累计"真启用"的 ORT 会话数——
// Warmup 烟雾（认领流→record→sync→query）通过才计数，静默回落同步不计。
// 门的空过防线：fence 腿断言该计数增长，而非只看逐位（回落也会逐位同）。
long long OrtFenceEngagedTotal();

// 声明式增量 H2D 的 debug 哨兵计数（FARM_H2D_DELTA_DEBUG=1，判决25）：累计
// "影子≠宿主"的全 face memcmp 违约次数（影子=设备忠实镜像；非 0=声明漏传
// 或 append-only 承诺违约）。门的空过防线：R8 哨兵门断言该计数增长——
// fprintf 只是运维可见，计数才是机器可断言面。
long long OrtDeltaDebugViolations();
} // namespace inferfarm
