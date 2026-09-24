# 来历与账本（provenance）

## 本库从哪来
2026-09-22 用户裁决：推理农场框架库升格新主线——YGO 只是第一个乘客。本仓
从 YGO 产线主线仓（私有）交出的资产抽层而成，协议与并发结构**逐句同源**
于现役生产代码：

| 本库 | YGO 源（产线 ai_core/，私有主线仓） |
|---|---|
| `src/fiber_pool.cpp` | `ai_opp_loop.cpp` FiWorker/FiTask/FiChain/RunLegFibers（2026-09-21 fiber_contract） |
| `src/bank.cpp` | `ai_infer.cpp` 银行段 BankCtl/Claim/SubmitWait/Harvest/BankLoop（2026-09-22 bank_contract） |
| `src/census.cpp` | `ai_opp_loop.cpp` census 埋点 + `ai_infer.cpp` 线程普查/调度台分段（fiber_census_contract） |
| `src/refit.cpp` + `tools/refit_blob.py` | `ai_infer.cpp` ParseRw1/ApplyRefitWeights + 权威导出器（refit v1 contract） |
| `src/backends/trt_backend.cpp` | `ai_infer.cpp` TRT 机件（LoadTrtLib/BuildTrtSession/Warmup/TrtCaptureGraph/MbSubmit/邮箱） |
| `src/backends/ort_backend.cpp` | `ai_infer.cpp` BuildSession 的 ORT 咒语（CUDA EP V2+enable_cuda_graph+IOBinding）——完成检测改为显式整设备同步（ORT 后端 v1 纪律） |
| `src/farm.cpp` 驱动环 | `ai_opp_loop.cpp` OppRunOneGame/OppCppChain 种子协议与收账语义 |

施工契约原文在 YGO 侧私有库（refit_v1/*.md，bank/fiber/census/refit/
integration 五篇）；交接书同在私有记忆库（未公开）。

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
- ~~框架正式命名待定~~ 已定：GameInferfarm（仓）/ inferfarm（命名空间）。
- ~~YGO 适配器回接待做~~ 已回接（产线现役 ai_core 即参考实现，演化回接走
  population 路由+SetLegShape 清单形态）。
~~- TRT refit 真引擎换心冒烟~~ 已清账（见 B5 段，2026-09-24）。

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
- **share 配比落地+批密度稀释定谳**（2026-09-22 追加）：DeviceConfig.share
  （平滑加权轮询，0=不接链）；--device share= 语法。实测 32局/8链：单卡 533、
  异构均分 267、share=4:1 229、share=0 空转 457、双 CUDA 同构组 320——
  病根=批密度稀释（到达流切薄），非核显算力/非调度台共享。512局/32链反转：
  均分异构 1113 > 单卡 654（双并行填充管线）。三形状指纹全部
  dbe3e3c5e59ba566/a3535388ff95061c（share 不破逐位）。
- **同卡双组红利（2026-09-22 三跑 A/B 定谳）**：512局/32链——单卡单组
  602-683 局/s；**同卡双 CUDA 组 1163-1313**；cuda+dml 1113-1138。结论：
  玩具尺度的"多卡翻倍"机制=**第二条填充管线**（单组同时只开一家填充银行），
  非核显算力（同卡双组同样快甚至略快——dml 同步提交有少量队头税）。副产品：
  **单卡用户把 banks 拆两组即可吃到管线并行红利**（--device 两次同 dev 不同
  组）。九跑指纹全同 dbe3e3c5e59ba566（分组确定性再证）。
- **CNN 极限冲刺（2026-09-22 终章）**：冠军=同卡 6 组×2 银行×256 链+stagger0
  ——4096 局 2026 / 8192 局 2167 局/s（各 3 跑指纹同；815.5 旧纪录的 2.66×，
  python 串行 111×）。腿长-速率曲线：1280→1600 / 4096→2026 / 8192→2167（爬坡
  摊薄+链密度）。核显 CNN 全配置空间零正收益定谳（批形×银行×share×异步全扫，
  机理三层入 benchmark.md）；每组独立批形状落地（G8b 门+fb1/2/4/8 小图烤制
  逐位验证）。stagger=10ms 默认在 128 链=1.08s 纯税——bench 必须 0。

## population 路由落地（2026-09-22，spike_othello 演化需求）
- 子代理产出：export_pop_onnx.py + models/othello_pop.fb{128,16}.onnx；
  对拍（CPU EP）：torch 路由 vs 单模型=逐位同（验收参考标准达成）；ONNX vs
  torch=1.5e-05 末位差（MLAS/kernel 序），argmax 翻转 0。
- 框架：InputMeta.population 三豁免+SetPopulation API+cpu 路由模式+G9 门
  （4+4 检查全绿，均匀 pop=单模型逐位同一次过）。
- 实测：12 银行 fb128×1024 链 → 0.52-0.54s/代（2.0× torch 1.07s）；2048 局
  腿=1.03s 线性（60K 决策/s 持续=routed 图吞吐绑定）；指纹跨银行数全同、
  逐代必变。途中案：--device 组不继承 population_input（CLI 两处修）；
  othello ES 模式缺省链=个体/局=8×P。
- 负载 B 侧：SetPopulation=毫秒级换心原语（21MB/代 H2D ~1ms），锚点对手
  =适配器侧后续（对手行走推理的 adapter 改造，spike 侧工作）。
- **网格路由终局（2026-09-22 深夜）**：v2.2 图（死路由块+越界同行+权重补行）
  + 后端批尾毒化 → 活性非确定性根治（同配置×3/跨银行数/跨图型指纹全同
  36a139d97396ec2d 系）。终版代频 **0.115-0.138s/代（torch 1.07s 的 8×，
  2 银行最优）**；45 门 ALL PASS。途中三雷：population 绑定量/探针 FillPattern
  两处按 slots 误乘（fb128 靠 P==slots 掩盖的陈年坑，fb1024 声明 8×实配）、
  批尾陈旧行散射污染（死行协议根治）。网格图 CPU 回落算子与 ORT 图捕获
  不兼容（--no-graph）；OneHot CUDA EP 内核 bug 绕行（Equal/Cast 替代）。
- **SetLegShape 腿形状热调**（2026-09-22 回接方需求，清单模式真墙）：腿间改
  chains/games/seed0 不重建银行池（Shutdown/Init=建池+热身+探针秒级开销每作业
  付一次，恰是 refit-jobs 要消灭的）。物理面 Init 烧死不可动；population 模式
  链数派生拒改。G12 门：热调续腿=新鲜农场同形状逐位同（含指纹）+非法拒绝。
  回接方清单循环形态：SetLegShape(作业形) → RefitWeights(blob) → RunLeg。

## 真模型门矩阵补全（2026-09-24）
- 烤制 `models/gomoku_mlp.fb8.trt`（tensorrt 10.16.1 python 包，TF32 关，
  引擎 0.3MB；本地工件不入库——TRT engine 机器相关，缺失=门自动 SKIP）。
- **R2 trt**：银行腿+复跑+银行 vs inline 三重逐位全绿（银行 515 / inline
  647 局/s，共享 GPU 时段读数仅供参考）。
- **R3 跨后端**：ort vs trt 指纹逐位一致 `a3535388ff95061c`——与本账本
  早前记录跨日跨会话复现（fp32+TF32 关的强性质第三次成立）。
- **R5 批次/位置不变性门**：ort 与 trt 双后端全绿——批大小 n=1/3/满与
  行位置变化下逐位同。至此 G13/R5 家族在 cpu/ort/trt 三后端全部实证，
  "batch invariance"从声明变成三后端被守住的契约。
- 真模型可选门矩阵现状：R1/R2/R3/R5 全绿；R4（cuda+dml 异构）待双卡环境。

## 绑核落地与判决 18（2026-09-24）
- 接入方需求：worker 支持绑核。新增 include/inferfarm/affinity.h +
  src/affinity.cpp（ParseCpuList/PinThread/pinned 探针）；接入点三处：
  fiber 工人（FiWorkerLoop 头）、调度台（disp 线程头）、线程模式链线程；
  env=FARM_WORKER_AFFINITY/FARM_SCHED_AFFINITY，值 "0,2-7,phys"。
- G15 门（farm_test 61→70 ok）：解析器规格/空串/垃圾段全拒/phys 真枚举
  +绑核腿指纹逐位同+pinned 探针防空过+越界软失败。
- 途中四案（全部当轮修复，坑目录有档）：①越界移位 UB 绕过范围检查；
  ②G15 门 sched env 残留假红；③phys 门被 "-1" 区间假元素空过成假绿
  （phys 枚举因 SDK sizeof(EX)=80>实记录 48 恒空——python ctypes 逐字
  节核布局后手动偏移读修复）；④探测砍除 spec_out 追加式枚举 ×banks>1
  =组 0 spec IO 重复入表（CNN 6 组形状首爆=多组盲区，只挂首银行修复，
  修复后 CNN 512 局腿 85.5% 胜率吻合训练记录 85.9%）。
- 判决 18 数字（fb8 fence 4096 局 / CNN 6 组 w14 4096 局，独占，指纹
  全同）：MLP 闲形状工人钉 phys +3.5% 弱正；调度台与工人同核 -94%
  崩盘（红线）；CNN 满载工人钉 -27%、+调度台钉空核=持平。绑核=opt-in
  工具非默认。
- A1 spin=2 独占复测（4096 局 ×4 交替）：中位 -15% vs spin=1——从
  "判死拆码"改判"留档省核选项"，默认仍 spin=1（判决 17 追记）。

## B4 banks×slots 二维重扫（2026-09-24 晚，fb16/fb32 工件烤制后）
- 工件：bake_gomoku_mlp.py --slots 16/32（权重与 fb8 sha 一致=同种子，
  仅批维钉死不同；模型不入库）。
- 矩阵（同步契约 spin=1，128 链 4096 局，九腿指纹全同 0a7be41d93eeda56
  ——同权重跨批形逐位同=R5 批次不变性再证）：
  | slots\banks | 1 | 2 | 4 |
  |---|---|---|---|
  | 8 | 923 | **12346** | 1352 |
  | 16 | 1103 | **2146** | 1515 |
  | 32 | 2036 | **4166** | 3001 |
- 判决：banks=2 全档最优（fence v4 的"fb8 2 流饱和"结论在 fb16/32 推
  广成立）；**同步契约对 banks 远比 fence 敏感**（banks=1 掉 51-93% vs
  fence 契约的 -30%——整设备同步税在无第二流重叠时全额裸露）；
  banks=4 反降（同步 Run 调度台串行提交下银行数=窗碎片化+轮转税，无
  并行收益）。slots↑ 恒增（批密满 rows/batch 8.0/16.0/32.0），但绝对
  值仍低于 fence fb8 档（fence 桥接=吞吐王者不变）。

## B5 TRT refit 真引擎换心冒烟（2026-09-24 晚，未决项清账）
- 根因：bake_gomoku_mlp.py 烤引擎从未设 refittable 旗标（kREFIT_NONE，
  createInferRefitter 拒建）——"真引擎冒烟"一直无从谈起。
- 修：烤制端 config.set_flag(BuilderFlag.REFIT)（legacy 模式=引擎自带权重
  运行端名单式重供，配 ApplyRefitWeights 现有逻辑；STRIP_PLAN+REFIT_
  IDENTIFIERS 需全量重供不采用）。重烤 fb8.trt（0.4MB，4 refittable 权重
  =onnx initializer 名单 fc1.weight/fc1.bias/fc2.weight/fc2.bias）。
- 名单权威链：TRT python get_all_weights()=onnx initializer 同 4 项（legacy
  REFIT 引擎现值不可读——getNamedWeights internal error，值一律以 onnx 侧
  为准）；导出器 tools/refit_mlp_rw1.py（--scale 扰动因子）。
- 冒烟三连（仓根腿）：原引擎指纹 502013814d87ca76 → 换心（scale=0.5）4 项
  全中 29ms、指纹必变 8776d14b497cb876 → 复跑逐位同。29ms=毫秒级热换兑现
  （演化场景选 TRT 的判决 12 依据落地）；已捕获图换心后照常跑（无 900）。
- **R7 门**（gomoku_backend_test，trt 腿可用才跑）：换心必变/复采逐位同/
  同 blob 二次 refit 幂等/名单外假名=引擎不动（负路径），全绿；旧工件
  （不可 refit）=SKIP。
- 新坑：FARM_CUDART_DLL 是全局 env——ORT fence 契约（cudart64_13）与 TRT
  cu12 构建（cudart64_12）冲突，两后端混用的进程不能同设（本门 R2 曾被
  此 env 打成 SKIP）。

## 对外反馈批：空目录解析修复（2026-09-24 晚）
- 反馈：f699bd3 的"cfg→env→空（系统 DLL 搜索）"意图未兑现——空目录拼出
  "\onnxruntime.dll" 根路径必败（GLE=126）。修=ort/trt 两处空目录走裸名
  LoadLibrary（标准搜索：应用目录→System32→PATH；语义对齐 cudart_dyn 原
  生正确写法），ort 基名冲突改名分支对空目录跳过（无"他目录"可冲突），
  GetApi 版本门失败信息补指路（显式设 FARM_ORT_DIR）。
- 实测：本机 System32 真有陈年 ORT 1.17.1——修复后裸名加载命中它并被
  版本门清晰拒载（此前根路径 126 掩盖了整个事实）；trt 空 env 裸名 126
  排查面正常；全 env 回归（fence ort 指纹 6da7de2ad81bdf36 / trt 指纹
  502013814d87ca76 / farm_test / gomoku_backend_test）全绿。
