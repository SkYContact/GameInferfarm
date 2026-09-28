# 棋类 AlphaZero 框架调研 → inferfarm 吸收判决（2026-09-28）

出处：lc0 × dlshogi × rela/ELF 四仓框架层调研（全报告 `D:/ygo_data/research/framework_survey_lc0_dlshogi_rela_0928.md`；
四仓全量克隆 `D:/refs/{lc0,dlshogi,rela,ELF}`）。本文=十二条可偷清单对 inferfarm 的逐条判决：
**已吸收 / 不吸收（带理由）/ 挂账（带触发条件）**。

## 总判决

inferfarm 与三仓**独立收敛**：推理缓存（KataGo 式判决13）、多 bank 同 GPU（B4）、
HR timer（判决20）、census 拆账、refit 热换+代际门（G14/B5）、probe+指纹门纪律——
三仓的共同结论我们基本都有了，且部分更好（fence 桥接 FARM_ORT_ASYNC=3 对应的
异步面 dlshogi 至今没做）。净新增价值集中在三件：**自动调参器（本批落地）**、
**在体影子哨兵（挂账）**、**位压缩输入（挂账大件）**。

## 逐条判决

| # | 调研清单条目（出处仓） | 判决 | 依据 |
|---|---|---|---|
| ① | 前置 NN 结果缓存+命中直推接口（lc0/dlshogi） | **已吸收** | `include/inferfarm/cache.h`（2026-09-22 判决13，KataGo NNCacheTable 同构：直接索引+碰撞即逐出+256 条锁+gen 代次失效）；`FARM_CACHE_LOG2`；与状态池互斥已显式声明（farm.cpp 池路径键失真警告）。lc0 的"命中即接口枚举"增量=我们的 Lookup 返回 shared_ptr 出借，同一语义 |
| ② | 变长凑批：首件无限等+后件限时（ELF） | **不吸收（哲学相反）**+挂账 | 银行=满座自驱+窗闹钟（window 缺省 0.2ms）+惊群修复，吞吐优先拓扑已固化；ELF"首件无限等"是吞吐至上路线，与我们低延迟窗相反。挂账：**空闲即刻发车**（池空在飞零时首请求免窗直发）——会改批组成=指纹面，必须 opt-in 门+指纹豁免论证后才做 |
| ③ | 后端自报批长/线程数 BackendAttributes（lc0） | **不吸收** | 银行批=slots 结构决定，无 minibatch 手调面可消；能力位已有一半（`DispatchFromWriterOk`/`ProbeFreeSpec`/`CompletionWaitHandle`）。若未来出现多 minibatch 后端再启此条 |
| ④ | TRT 每线程独立 profile/context/stream（dlshogi 2026-05） | **已吸收（等价面）** | 同 GPU 多 bank 组池独立轮转（B4 判决 banks=2 全档最优）=dlshogi 多 slot 的框架级等价物；组间零共享=无 host 锁 |
| ⑤ | Optuna+MedianPruner 参数自动调参器（dlshogi 102 行） | **本批落地** | `tools/tune_farm.py`：目标=farm 臂局/s；trial0=全缺省锚点；sublegs 段中位抗热毛刺；**指纹/决策数违约 trial 自动判剪**（"时序旋钮不改算术"红线变成执法面）；--storage 可分布式。取代 CLAIM_SPINS/SPIN/HRTIMER 类手扫（c8a1cc3、判决17/20 的梯子已内置为候选档） |
| ⑥ | 模型 .ini 捎带推理超参（dlshogi） | **挂账（乘客面）** | 框架无需改动；触发=YGO 演化导出器同目录发 ini（σ/温度等），Farm 侧可选读取。等第一个多代乘客要求再做 |
| ⑦ | recordreplay/check 影子比对（lc0） | **部分已有+挂账** | 离线面已覆盖：refit R7 门+`tools/probe_partial_d2h`（FNV 四档双门）+R3 三平台逐位同。挂账：**在体影子哨兵** `FARM_CHECK_BACKEND=<名>`——长演化运行中逐 N 批跨后端抽检 FNV，抓 INT8/refit/驱动的在体漂移；触发=演化代际热换出现无 traceback 读数漂移（呼应 ygo 侧 g166 静默死亡案） |
| ⑧ | ModelLocker 三缓冲引用计数热换（rela） | **已吸收（等价面）** | RefitWeights 29ms 热换+G14 代际门+懒加载守卫+缓存 gen 失效=同一"新版本就绪前旧版持续服务"语义家族；无 ABA 面（引擎级换心非多版本并存） |
| ⑨ | 配置三元组哈希决定重建 vs 原地更新（lc0） | **不另做** | 会话生命周期=银行制一夫一妻（地址烧死图），重建面已由代际门管辖；lc0 那套是为长活引擎频繁改配设计的，我们的形态不需要 |
| ⑩ | hcpe3 访问分布格式+裸结构体 gz chunk（dlshogi/lc0） | **乘客面** | 框架无关。YGO/gd 学习侧复活时的对局落盘格式参考，挂到各自车道 |
| ⑪ | 自对弈质检三件套（lc0 resign playthrough/fp_threshold/弃开局池） | **乘客面** | 同上，挂 YGO DPO 分叉挖矿/winbot 线 |
| ⑫ | Tachometer+优雅停机+pause-eval-resume（rela/ELF） | **已吸收** | census 三段乘客计时+banksched 队深取证（9120735）已超 Tachometer；停机=全量 swap 唤醒路径已有（0e19323）；pause-eval-resume 对应 inline 会话+population 面（判决16） |

## 调研中超出清单的观察（备忘）

- **位压缩输入（dlshogi 1bit/格+GPU unpack kernel，PCIe ÷8~32）→ 挂账大件**：
  ModelConfig 输入 dtype=packed bitplane + TRT 图前奏 unpack 层（烤图时烘焙）/
  ORT custom op。与判决25 声明式增量 H2D（5b28419，FaceDepth 申报=少传字节第一刀）
  同方向；位压缩是第二档（少传 bit 视图）。触发=掼蛋/YGO 侧 PCIe 再度成瓶颈且
  增量 H2D 吃不下。
- dlshogi"凑批加 mutex 上线 8 天即 revert"（24b2065→1fb5157）=我们对照门纪律的
  同款，无动作。
- ELF→rela 演进判决（回调式训练被对局节拍绑架→replay 解耦）支持现有乘客制拓扑，
  无动作。
- lc0 smart pruning 勿映射"验收对局早停"——ygo 侧 evo-efficiency-menu ③号已判死
  （2048 局 CI 只能决 ≥2.5pp），与本仓无关但防误吸，记录在案。

## 本批变更

- `src/bank.cpp`：win 编译修复（ab6ad15，裸 std::min 宏劫持+%d/size_t）。
- `tools/tune_farm.py`：清单⑤落地（冒烟 4 trials×2 sublegs 全绿，指纹全程
  da7855e72a80b6b4 逐位同；ort GPU 腿前置=FARM_ORT_DIR+FARM_CUDART_DLL+torch/lib
  入 PATH，见工具头注）。
