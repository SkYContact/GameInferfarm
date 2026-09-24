# 贡献指南

## 开发纪律（本仓的承重墙，改动前请读）

1. **先读判决书再动手**：[docs/design-judgments.md](docs/design-judgments.md)
   的每条判决都有实测依据或构造论证（含负结果/墓碑）——推翻判决需要新的
   测量，不需要直觉。
2. **行为改动必须过确定性门**：`farm_test`（G1-G12）全绿才算数——银行 vs
   inline 逐位一致、复跑全同、fiber vs 线程全同。改 bank/fiber/farm 后先跑：
   ```bash
   cmake -S . -B build && cmake --build build --config Release
   build/Release/farm_test.exe
   ```
3. **每个性能数字 ≥3 跑取中位+指纹全同**；A/B 对比交替 ≥4 腿防热偏置
   （吞吐基准必须 `FARM_STAGGER_MS=0`）。负结果同样值得提交——判决书里
   一半是墓碑。
4. **先修完再跑，勿带病发车**；fail fast 优于静默容错（推理故障=判负纪律，
   不静默重试）。
5. 提交信息写清"改了什么+为什么+回归门结果"。

## 环境要求

- C++17，CMake ≥3.16，Windows（fiber 语义与后端 DLL 装载为 Windows 实现，
  见 README"约束与路线"）。
- 无 GPU 也能开发/回归：cpu 后端覆盖全部门（G1-G12）；ort/trt 需运行时
  DLL（`FARM_ORT_DIR`/`FARM_TRT_DIR`），真模型门无工件自动 SKIP。

## 提交什么

- bug 修复（附复现+回归门结果）
- 新后端/新游戏范例（走 GameAdapter 三契约，见
  [include/inferfarm/game_adapter.h](include/inferfarm/game_adapter.h)）
- 文档/基准口径修正（欢迎挑错——口径表就是被审核逼出来的）
