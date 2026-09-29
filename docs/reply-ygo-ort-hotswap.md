# 回执：ORT 会话常驻+权重热换工单（YGO 粗筛腿，2026-09-29 框架车道）

> 工单：`ai_duel --ort-jobs` 作业切换 2.3s/次（ORT 会话逐作业重建）→ 提案
> 同架构会话池+权重外置初始化器原地热换。**本回执结论：机制成立，判决探针
> 已过，可以立项**；另有一条零框架工作的 Option 0 请 YGO 侧先核对。

## 0. Option 0（先核对，可能不用写码）

**TRT refit-jobs 形态已在框架现役**：Farm 常驻+RefitWeights+SetLegShape+
RunLeg（每作业只换心不重建，银行池只建一次）——TRT 换心 53ms 实测在档
（掼蛋 0.7M 参 43 项；YGO 34.5M 参=上传受限约 20-40ms，同数量级）。
粗筛腿若能接受"每架构烤一次 REFIT 旗标引擎"（烤制 ~150s 一次性），今天
就能切，零框架工作。**问 YGO：粗筛腿停留 ORT 的硬约束是什么？**（若无，
Option 0 最快；若 ORT 是硬约束——如与在产 fb64 管线同一套工件——走 §2。）

## 1. 机制判决（探针实证，非推断）

"权重外置初始化器"在 ORT 的官方姿势 = **权重升格为可覆写图输入**
（initializer 同时列入 graph.input）+ **IOBinding 裸指针绑定我们自有的
常驻缓冲**。探针 `tools/probe_ort_weightswap.py`（CPU EP，ort 1.26.0）：

| 判据 | 结果 |
|---|---|
| P1 可覆写初始化器存活且喂值生效 | ✓（ORT 仅 Warning"不再是常量/禁 const folding"，不阻断） |
| P2 裸指针绑定+复跑逐位稳 | ✓ |
| **P4 原地改缓冲→下个 Run 读到新值（热换核心语义）** | ✓ |
| P3 换回幂等（G5 语义：A→B→A 逐位还原） | ✓ |
| 图优化级别敏感性（ALL/BASIC/DISABLE 三档） | 全过=不敏感 |
| 热换世界 == 重建会话世界 | ✓（同图同优化=同 float 世界，跨作业恒定） |

CUDA 段实测（`tools/probe_ort_weightswap_cuda.py` v2.1，2026-09-29 GPU 锁内，
ORT 1.30/farm_env.sh 现役，RTX 3060M）：

| 判据 | 结果 |
|---|---|
| P_swap 热换生效（134MB pinned H2D 覆写绑定缓冲→下个 Run 读新值） | ✓ |
| P3 换回幂等（A→B→A 输出逐位还原） | ✓ |
| 换心计时 | **28.8 ms**（134MB pinned H2D+同步；换回 28.7ms） |
| 每 Run 开销（含 134MB 权重绑定） | **0.525 ms**=常规会话同量级（绑定零重传实证） |
| 会话创建冷启（含 CUDA EP init） | 0.55–2.08 s（跑间波动；与 YGO 2.3s/作业读数同域） |
| 会话创建温启（同进程第二会话） | 0.33 s |

对账：124 作业/代 × 29ms ≈ **3.6s/代**（vs 现行 285s，-99%）。热换成本
实测落在预判带内（138MB≈15-30ms §2），且低于 TRT refit 的 53ms——两后端
"换心不换会话"成本同数量级，粗筛腿留 ORT 无成本罚。

**环境怪癖面修正（v2.1 顺带定谳）**：同步 `cudaMemcpy` 在 ORT CUDA EP init
后原生崩的怪癖，实测**pinned 源/128KB 也崩**（旧档"pageable ≥1MB"记窄了）；
同窗 `cudaMalloc`/`cudaHostAlloc` 皆活=纯 sync-memcpy 入口问题。探针改
MemcpyAsync+流同步（框架生产同款路径）后全绿。凡 ctypes/cudart_dyn 直接
调 CUDA 的探针一律走 MemcpyAsync+同步，别碰同步 cudaMemcpy 入口。

## 2. 设计（ort_backend 立项面）——**已落地（2026-09-29，用户令"框架先立能力，
不依赖 YGO 用不用"；实现差异见 §2a）**

- **导出侧**：bake 脚本加"权重升格"出口（torch 导出后把 initializers 挪进
  graph inputs——onnx 官方工具 remove_initializer_from_input 的逆操作，
  ~20 行）；权重名=state_dict 键（RW1 同款纪律）。
- **ort_backend**：`RefitWeights(rw1)` 从恒 false 变真实现——RW1 名单→
  逐权重 H2D 覆写进绑定缓冲（判决 25 批拷贝通道可复用，138MB≈15-30ms）。
  会话创建时一次性 bind（IOBinding 钉死地址），**图捕获（fence 桥）兼容**：
  CUDA Graph 冻结地址下原地改值合法，重放即读新权重。
- **银行面豁免**（承重墙，三处）：权重面=**会话级绑定面**，不进槽行体系
  ——Claim 清零豁免、HashSlot 豁免、H2D 循环豁免（population 面同款三豁免
  机制复用；否则 138MB×每决策 memset=灾难）。
- **const folding 损失**：升格后 ORT 禁用权重相关折叠（Warning 在案）——
  YGO 模型 BN 折叠在训练侧已完成，残余影响待 fb64 真模型 A/B（探针小模型
  三档优化全过=方向性乐观）。

### 2a. 实现定稿（与 §2 设计的差异，2026-09-29）

- **声明面=模型文件本体，零配置**：检测用 ORT 官方 overridable initializer
  枚举（`SessionGetOverridableInitializer{Name,Count}`，v19 头在）——升格的
  onnx 自带声明，无 ModelConfig 新旋钮（population 需配置因文件里无从分辨；
  升格权重文件里自明）。`InputMeta.weight` 旗标随 spec 流动。
- **种子语义（v19 头无 initializer 读值 API 的解法）**：RefitWeights 在无会话
  时入实例级 stash（init 期换心先于建行的现役时序正好吻合）；后续会话
  Warmup memset 后从 stash 播种。**升格模型未换心就发车=SubmitBatch fail
  fast 拒批**（零权重=垃圾，不许静默；⑥负路径门）——首换心用升格工具的
  RW1 边车（as-baked 值）或 `ModelConfig.refit_weights` 指向它。
- **validate-first 两相提交**：全条目×全会话先验（dtype/numel 不符/无落点/
  空转=拒，e271290 同款）后写，零撕裂态；未播种会话的部分换心=拒（余下面
  仍为零）。DML=纯宿主覆写；CUDA=阻塞 H2D（生产通道，腿间流空闲）。
- **豁免接线（population 同款+一处执法）**：Claim 清零/HashSlot/H2D 三循环/
  槽基址预解/FillPattern/mid_like 毒化排除全按 `population||weight`；
  **ort InputRow 对权重面返回 nullptr**（唯一写入口执法——inline 清零/Claim
  回退/bank 视图回退经此自动安全）。fillwrite/append/headlive/state_pairs
  不受影响（权重面与各声明面互斥或静默回落 full=永远安全）。
- **工具**：`tools/promote_weights_to_inputs.py`（升格+RW1 边车一次产出）；
  门=`gomoku_backend_test` R11（换心必变/复采同/幂等/换回还原/假名负路径/
  未换心拒批/stash 播种重建世界对拍）。

## 3. 验收门（按 YGO 契约+框架纪律）

1. **G5-ort**：同 blob 双换幂等（哈希/输出逐位不变）+ 换心后复采同 + 假名
   负路径拒载（对齐 trt R7 契约）；
2. **同种子逐局同**（现有契约）：热换腿 == 重建会话腿逐位对拍——若 float
   世界有漂（升格改变 kernel 选择），按"表达式变形改变 float 世界"旧律，
   锚点重测一遍一次性入账；
3. **计时门**：换心 ≤100ms 量级（138MB H2D 上限 ~15-30ms+refit 开销）；
   每 Run 开销与重建基线持平（绑定缓冲零重传）；
4. **A/B 终验**：粗筛腿 124 作业/代全程热换 vs 现行 --ort-jobs，同种子
   outcome/指纹逐位。

## 4. 工作量

探针（已完）→ ort_backend RefitWeights+绑定面（~1 天）→ bake 出口+
G5-ort 门（~半天）→ YGO 接线联调（你们侧半天，pol==3 式档位预留同款）。
收益对账按你们口径：124 换/代 × 2.3s → ~4s/代（H2D 覆写），单代 ~-45%。

## 5. 附

探针 CPU 版四前提×三档优化全过 + CUDA 版计时/生效/幂等全绿（§1）=判决
完备，工单可随时立项（等 YGO Option 0 答复定路线）。TRT 侧同思路先例
（判决 12"演化场景选 TRT"+53ms 实测）不变——本工单是把同一思想移植到
ORT 现役管线，两后端此后同享"换心不换会话"。
