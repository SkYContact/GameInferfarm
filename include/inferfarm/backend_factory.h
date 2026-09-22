// backend_factory.h — 后端工厂（cpu|ort|trt；trt 需 -DINFERFARM_WITH_TRT=ON 构建）
#pragma once

namespace inferfarm {
class InferBackend;
InferBackend* CreateCpuBackend();
InferBackend* CreateOrtBackend();
InferBackend* CreateTrtBackend();
} // namespace inferfarm
