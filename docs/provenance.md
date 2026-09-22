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

## 真模型实测（2026-09-22 三后端五子棋）
五子棋一层 MLP（450-64-225，未训练、种子权重；tools/bake_gomoku_mlp.py 烤
fb8 onnx + TRT engine，TF32 关）x2 银行 x8 槽 x4 工人 x32 局：

- **ORT 后端首跑通**：CUDA EP + enable_cuda_graph（银行会话钉调度台线程）；
  探针修两处潜伏 bug 后（探针漏 D2H、探针跑图漏 H2D）全链绿。
  R1 门：复跑逐位同 + 银行 vs inline 逐位同（指纹 7f1cb31d293c961c）。
  **账本缺口"ORT+银行制"实测点落地**：银行 508-722 局/s / inline 606-659 局/s。
- **TRT 后端首跑通**：图捕获+邮箱照常；R2 门全绿（同指纹 7f1cb31d293c961c）。
  银行 432-618 局/s；[bank] 行 gpu_flight 68%、dep 0.08ms（vs ORT 0.48——
  图回放提交更便宜的直接证据）。
- **跨后端逐位一致（观察项）**：ORT 与 TRT 对同一 fp32 模型指纹逐位同
  （TF32 双端关闭的前提下）。
- **诚实结论（玩具尺度）**：CPU 后端最快（约 1000 局/s）——MLP 太小，GPU
  每批门票纯开销；GPU 后端的价值在 YGO 级模型（行宽 176.9KB）——产线账本
  317-368 局/s 才是它的战场。银行制下三后端行为逐位可比本身就是可交付性质。
- 门进套件：tests/gomoku_backend_test.cpp（工件缺席=SKIP 退 0，无 GPU/CI
  不拦路）；主构建（ORT）与 build-trt（ORT+TRT）双 ALL PASS。

## 未决
- run25 发车仍在 YGO 侧停车等用户口令（发车卡 D:/ygo_data/es_run25_launch.txt）
  ——**框架会话勿动它**。
- 框架正式命名待定（暂名 inferfarm/推理农场）。
- YGO 适配器回接待做（现役 ai_core 即参考实现）；TRT refit 真引擎换心冒烟（RW1 名单对齐 torch 权重名）待做。

## KataGo 调研与吸收（2026-09-22）
- 调研动机：KataGo 是固定游戏（围棋），inferfarm 是任意游戏——但推理服务的
  思维应当可迁移。通读 cpp/neuralnet/nneval.h/.cpp（NNCacheTable /
  NNResultBuf / 服务器线程拉模型）+ 论文 arXiv:1902.10565。
- **吸收**：推理缓存（判决13，include/inferfarm/cache.h）——键=组装行字节
  +权重代次，直接索引+条锁+shared_ptr 出借（KataGo NNCacheTable 结构同款）。
  门 G7 全绿：五子棋热缓存腿 100% 命中 430 局/s vs 关缓存 153 局/s，指纹
  逐位同；玩具（多输入）开=关逐位同；代次失效门过。
- **互证收获**（判决14）：KataGo fixed-shape 补齐注释=我们 fb 钉死判决的
  英文镜像；后处理在客户端=防退化备忘；多 GPU/fp16/纯前向基准范式入路标。
- **明确不吸收**：MCTS 树层技巧（乘客的事）、对称随机化（游戏特定）、
  客户端行缓冲单跳拷贝（我们零拷贝直写更优）。
- **负结果同样入账**：CPU 玩具尺度 512 局 A/B（chains=32 banks=4 slots=8），
  缓存 53% 命中反而 1079 vs 2709 局/s——CPU 后端批计算按领号前缀行算（弃槽
  行照算垃圾）+ 命中纤维不挂起踩踏领号机器（银行 cycle 0.17→0.79ms）。GPU
  后端批形状钉死不受此税（机械论证；YGO/TRT 实测数待回接时补）。
- **ORT(GPU) 后端 A/B（4 腿交替）**：缓存关 573/651 局/s，开（57% 命中）
  633/605——±10% 效应 < ±13% 跑间噪声，玩具尺度**不可判**；四腿指纹全部
  dbe3e3c5e59ba566（行为不变性成立）。缓存主战场=YGO 尺度（真模型+GPU 忙），
  回接时补数。ORT=q35 环境自带 onnxruntime 1.30（capi 目录）。

## 多 GPU：设备挂银行（2026-09-22，判决15）
- 动机：开源框架的泛用性（用户判据：显卡+核显跑通 ⇒ 别人双开直接用）。
  KataGo gpuIdxByServerThread 同构吸收；610M 核显当验证台（非算力）。
- **实现**：BankGroupCfg（组=后端+模型+银行数）；组池/组轮转/组窗分列；
  Claim 组门（链→组钉扎 c%n_groups）；Farm devices 配置面+组间结构核对；
  ORT 实例化（api/dll/env 下沉成员）+双 dll 改名共存；DML EP 分支（宿主
  绑定+同步 Run+输入重绑血律）；TRT device_id 守卫（cudaSetDevice 入口化，
  dev>0 本机未测——单卡行为不变）。
- **实测**（5070Ti+610M）：双 ORT 同进程（1.30 CUDA+1.24 DML）；异构 64 局
  315 局/s、指纹 a343f5f21dc49872 三跑全同（跨厂商钉扎确定性）；R4 门绿；
  **观察**：该工件 CUDA/DML/TRT 三家指纹逐位一致（a3535388ff95061c，与
  单设备腿同——大模型不保证，仅证明协议通）。DML 双设备初测 96K/66K
  rows/s（玩具，调度开销绑定）。
- 门：G8a（同构分组=单组逐位同，CI 可跑）；R4（真异构，FARM_DML_DIR
  选通，缺席 SKIP）；gomoku --device 语法演示。
- 途中案：InputRow 残留旧单组成员=空指针虚调用（坑目录）；DML iob 输入
  忽略案（probe 哨兵抓到，判决15 血律 2）。
