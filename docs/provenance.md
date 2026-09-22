# 来历与账本（provenance）

## 本库从哪来
2026-09-22 用户裁决：推理农场框架库升格新主线——YGO 只是第一个乘客。本仓
（D:/inferfarm）从 D:/ygo 主线交出的资产抽层而成，协议与并发结构**逐句同源**于
现役生产代码：

| 本库 | YGO 源（D:/ygo/ygopro/ai_core/） |
|---|---|
| `src/fiber_pool.cpp` | `ai_opp_loop.cpp` FiWorker/FiTask/FiChain/RunLegFibers（2026-09-21 fiber_contract） |
| `src/bank.cpp` | `ai_infer.cpp` 银行段 BankCtl/Claim/SubmitWait/Harvest/BankLoop（2026-09-22 bank_contract） |
| `src/census.cpp` | `ai_opp_loop.cpp` census 埋点 + `ai_infer.cpp` 线程普查/调度台分段（fiber_census_contract） |
| `src/refit.cpp` + `tools/refit_blob.py` | `ai_infer.cpp` ParseRw1/ApplyRefitWeights + 权威导出器（refit v1 contract） |
| `src/backends/trt_backend.cpp` | `ai_infer.cpp` TRT 机件（LoadTrtLib/BuildTrtSession/Warmup/TrtCaptureGraph/MbSubmit/邮箱） |
| `src/backends/ort_backend.cpp` | `ai_infer.cpp` BuildSession 的 ORT 咒语（CUDA EP V2+enable_cuda_graph+IOBinding）——完成检测改为显式整设备同步（ORT 后端 v1 纪律） |
| `src/farm.cpp` 驱动环 | `ai_opp_loop.cpp` OppRunOneGame/OppCppChain 种子协议与收账语义 |

施工契约原文：`D:/ygo_data/refit_v1/*.md`（bank/fiber/census/refit/integration）。
交接书：`D:/ygo/.zcode/memory/inferfarm-handover.md`。

## 数字底账（YGO 产线，本机 9955HX 16C/32T + 5070Ti Laptop）
- 两夜 42→**317-368 局/s（7.5-8.7×）**；51.4ms CPU/局全账平：工人 48.2
  （组装 8-11 含清零 1.31+自旋 1.78｜游戏本体 33=lua16+引擎/泵 17）+调度台
  2.5（大半闲等）+驱动 0.5。理论极限 ~360，已贴。
- 拷贝链勘误：行宽 176.9KB（非 85KB）；老路径两跳拷贝 ~7ms/局 → 银行 1.5。
- 银行窗扫描（pool4，512@256 stagger10）：floor 0.2→129.6 最优（见判决 3）。
- 形态内余量 ~10-20%（引擎非 lua 瘦身/slim 行/自旋修）；再往上=加物理核
  （EPYC 128C ≈2500 局/s）或 gpu 侧（闲）。
- 确定性史：同种子逐位门贯穿所有重构（G1/F1/I1/A4 全绿史）。

## env 旋钮对照（YGO_ → FARM_）
| YGO 产线 | 本库 |
|---|---|
| YGO_OPP_FIBERS=1 / YGO_OPP_FIBER_WORKERS=K | FARM_FIBERS / FARM_FIBER_WORKERS |
| YGO_INFER_BANKS=N / YGO_BANK_WINDOW_FLOOR | FARM_BANKS / FARM_BANK_WINDOW_FLOOR |
| YGO_FIB_CENSUS=1 | FARM_CENSUS |
| YGO_INFER_ENGINE / YGO_INFER_REFIT_WEIGHTS | ModelConfig.engine_path / model.refit_weights |
| YGO_TRT_DIR / YGO_ORT_THREADS | FARM_TRT_DIR / ModelConfig.ort_threads |

## 本仓首日验收（2026-09-22）
- 主构建（cpu+ort 编译，trt 关）+ build-trt（trt 编译）全绿。
- `farm_test`：G1 银行 vs inline 逐位（含指纹）/G2 复跑/G3 fiber vs 线程/
  G4 census X=0+复活样本/G5 refit 同 blob 逐位同+异 blob 必变+RW1 负路径
  ——**ALL PASS**。
- `refit_probe`：会话级 refit 生效（行0变/行1 非对应位不变）+ Farm 全链
  （银行协议 Claim/SubmitWait 直读）+ G5 完全复刻三腿。
- 首夜血案两起入坑目录：帧所有权 delete、测试指纹忘回填（空过+假败一对）。

## 开源化（2026-09-22 同日）
- 五子棋接入范例 examples/gomoku（未训练 CPU 模型 vs 规则对手——流程示范，
  不赢棋是特性说明）；farm_test 增 G6（五子棋银行 vs inline 逐位门）。
- README 重写为开源中文主文档 + README.en.md 英文副本；LICENSE=MIT。
- CPU 后端点积宽度可配（CpuModelDecl.poly_k，棋类全局面输入=行宽）。

## 未决
- run25 发车仍在 YGO 侧停车等用户口令（发车卡 D:/ygo_data/es_run25_launch.txt）
  ——**框架会话勿动它**。
- 框架正式命名待定（暂名 inferfarm/推理农场）。
- ORT+银行制吞吐实测点缺失（见判决 12）；YGO 适配器回接待做。
