# 种群逐行路由 ONNX 对拍报告（2026-09-22，负载 A / 候选设计 1）

## 命令

```
cd D:\inferfarm\spike_othello
C:/Users/41601/Miniconda3/envs/q35/python.exe export_pop_onnx.py   # 全程 CPU EP
```

pop = [128, 41280]（0 号个体=bc_t3k theta_final.pt 真权重，1..127=随机 init_theta），
own/opp 随机 fp32，mid 随机 int64 ∈ [0,128)，每档 4 轮。产出模型输入名次序
own/opp/pop/mid（mid=INT64 已断言），pop 形状钉 [128, 41280]，输出 policy [S,64]。

## 三个对拍（max|diff|；①另有差异位数口径）

| 对拍 | fb128 (S=128) | fb16 (S=16) |
|---|---|---|
| ① ONNX vs torch 路由 forward_pop | 1.53e-05 | 7.63e-06 |
| ② ONNX vs torch 单模型 forward_one(mm) | 1.53e-05（差异 25338/32768 位） | 7.63e-06（差异 3235/4096 位） |
| ③ torch 路由(bmm) vs 单模型(mm) | **0（0/32768 位，逐位同）** | **0（0/4096 位，逐位同）** |
| 附 torch 路由 vs model.forward_pop 对角(baddbmm) | 3.05e-05 | 3.05e-05 |

口径背景：|logit|max=226（fb128）/270（fb16），①的**相对**差异 6.8e-08 / 2.8e-08
——末位量级。①②的差异源=ORT MatMul(MLAS) 与 torch bmm/mm kernel 累加顺序不同，
与 REPORT 先例（CUDA/CPU、跨批形末位差）同类；torch 侧 bmm 与 mm 完全逐位同（③=0）。

## argmax 口径（下游只吃 argmax）

①ONNX vs 单模型：argmax 翻转 **0/512 行**（fb128）、**0/64 行**（fb16）——末位差
未翻转任何决策。

## 结论

- 逐行路由数学正确：torch 侧路由 vs 逐模型单独跑**逐位同**（③=0），满足
  ENGINEER_NOTE 验收门槛 1 的参考标准（forward_pop/forward_gather 一致性）；
- ONNX（CPU EP）与 torch 有纯 kernel 末位差（相对 ~1e-8，argmax 零翻转），属
  REPORT 既有口径内，非缺陷；
- 两档钉批模型可用：权重为运行时输入（文件仅 ~10KB，无常量权重），一次 build
  全代 128 个体热换，适配器只需在组装行时带个体号进 mid。

## 工件

- `D:\inferfarm\spike_othello\export_pop_onnx.py`（导出+对拍一体）
- `D:\inferfarm\models\othello_pop.fb128.onnx`（S=128, P=128）
- `D:\inferfarm\models\othello_pop.fb16.onnx`（S=16, P=128）
