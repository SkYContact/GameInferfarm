# 掼蛋线交接档 — 2026-09-30 晨（会话归档，新会话入口）

> 十轮交接（gd:notes/FRAMEWORK_HANDOFF.md + DATA1-10 / REPLY1-13）的累计
> 状态与挂账清单。新会话从本文件起步。全部数字锁内实测；回执链在
> 掼蛋仓 gd/notes/，判决原文在 docs/design-judgments.md 与
> docs/state-residency-design.md。

> **清账补记（2026-09-29 晨会话，无 GPU 批）**：§2 挂账 B/D/E 的 CPU 可解
> 面已清——W2 组路由+异 IO+固批按组+G19 门（a87e4ac，接口冻结，掼蛋
> W4=RouteGroup return 1 即接）/CI 五笔实为全绿（当时漏查）/refit 空转
> fail-fast（e271290；DATA11 §2 报修=旧快照误读，重试本体 7a48eef 已有）/
> segvcatch 入仓（1bb506b）/patch_fence population 支持（7013cd3）/
> /tmp 易失件已归档 ~/inferfarm-project/archive/gd-tmp-forensics-20260929
> .tar.gz。明细=gd:notes/FRAMEWORK_REPLY14.md。
>
> **清账补记二（2026-09-29 GPU 锁内会话）：A 项 DATA10 结案**——根因=
> 掼蛋侧 adapter 的 CollectOutputs 在首决策漏报状态 dest（stateful_ 探测
> 跑在首次 AssembleInto 之前），主机路径从第一天起丢 dec-1 状态输出、
> dec-2 喂零；**池路径全程忠实（散射无损+填充忠实+图/context 终审无罪，
> 设备级三读回点实证）**。修复=乘客侧三行（合成图契约下 dest 按模型真相
> 申报），修复后 4245 逐决策同+4279 seat-pool 判别局翻面消失。方向反转：
> 池无回归，评估腿可切回池配置。全文=gd:notes/FRAMEWORK_REPLY15.md；
> 修复件=archive/gd-sandbox-fixed-gdstate-20260929.tar.gz。ort/trt 面
> R 门复验同步清零（34+15 ok ALL PASS，R7 负路径按 fail-fast 契约更新
> a17088d）。C 项无触发维持按需。

> **清账补记三（2026-09-29 会话四）**：①REPLY16——W1 引擎缓存 vector 悬垂
> 认领（掼蛋代理捡漏第 5 个真 bug；deque 修复三行逐字一致+R10 交错序门，
> cec7d34；红光三连假门教训=取样点/成员值自比/UB 堆运气，终版判据=
> DebugEngineCookie 地址稳定钩子）。②**仓库改单主线**：master 快进并删除
> feature/cpp20-coroutines 分支（51 笔一次到位），此后工作直接在 master。
> ③**ORT 1.30 转正**（farm_env.sh 已切 farm_pkg_130/cu13，指纹 a3535388
> 零漂移×2+55 ok ALL PASS×2；回滚=farm_env_126.sh）。④**YGO ORT 热换工单
> 判决+探针入仓**（docs/reply-ygo-ort-hotswap.md + tools/probe_ort_weightswap
> {,_cuda}.py；四语义前提全立；Option 0=TRT refit-jobs 现役待 YGO 答复硬约束；
> 2.3s 大头=CUDA EP init 实测 381-1836ms）。⑤ncnn 核显热换判决：C API 零权重
> 口、reload 式可行但**判决 22 非确定挡死 YGO bitwise 验收**——挂账触发=出现
> 无 bitwise 契约的纯吞吐换模型负载。⑥GPU_PROTOCOL v1.1（release 校验+who
> 会话可区分名；本次校验测试误放 PFD 锁已代管恢复）。**挂账**：热换探针
> 134MB pinned 计时段（tools/probe_ort_weightswap_cuda.py v2 已就绪，锁空即
> 跑）；YGO Option 0 答复；③v2 维持按需。
>
## 1. 速度谱终账（gd 复合图 fb64，4096 链）

| 站 | dec/s | 提交 |
|---|---|---|
| 诊断起点 | 22,419 | — |
| +快刀（fullwrite+spins1000） | 25,286 | 0f29ec2/c8a1cc3 |
| +③设备池 | 33,251 | f7be552/f6a8807 |
| +D2D 批量化 | 34,027（+52%） | d45fa29 |
| refit 热换 | 53ms/次（代成本-62%） | 7a48eef |
| W1 一进程多引擎 | 双引擎同进程门绿 | b86e387 |
| 固批 FARM_FIXED_BATCH | 确定性旋钮 | dd9fd38 |

## 2. 挂账清单（新会话按此清理，按优先级）

### A. DATA10 两局分歧谜题（池路径回归的最后一块）
- **谜题**：决策 1 两路径输入逐位同（状态零/特征同哈希/rst 同），但引擎
  输出 vlen_new 主机会话=1、池会话=0（整数差）；vlen 轨迹 1,4,7 vs 0,3,6。
- **已排除**（勿重走）：批拷 API / 邻行独立 / 游戏世界分岔（落牌流水
  逐位同）/ 图形态（主机强制计算图=基线同）/ 散射完整性（完成点池==输出）/
  evmask-eva-evc 行宽错配（引擎枚举全匹配）。
- **下一步**：①锚行哈希加 vlen 分桶单独再证图内路径；②混合形态对照
  （池会话用 4 段图+图外散射——hack 已在 /tmp/gd_dbg 试过雏形）；③都排除
  → TRT 双 context 同 engine 差异域（版本敏感，备选升级 TRT 复验）。
- **取证树**：/tmp/gd_dbg（gd HEAD+WIP 四文件+上游最新 src；含 GD_DEC_DUMP/
  PLAY/INPUT/STATE_TRACE 仪器与 GST 系 trace——**易失，新会话先打包**）。
  复现：REPLY13 §2 引 DATA8 §3 命令。
- **仪器教训**（勿再踩）：dump 必须放写入之后（写入前=跨链槽残留噪声，
  已误导一轮）；真值读点=完成点（flag 观测后）；submit 时同步读=上批残留。

### B. W2 组间异引擎 + RouteGroup（双模型/④ 的最后一件）
- W1 引擎面已就绪；剩：Farm 组间同 IO 契约解禁 + GameAdapter::RouteGroup()
  （缺省 -1=链钉扎）+ 固批按组化（W3）+ 双引擎门（W5）。设计=REPLY10 §2。

### C. ③v2 清单（按需）：页池 / ort 面状态池 / 中途复位 API
（SlotWriter::StateReset 设计已入 docs/state-residency-design.md §5-2）

### D. CI 状态
- d45fa29 及之前绿；b60008c..dd9fd38 五笔 **Windows 面未验**（多引擎重构
  编译面 Linux 三象限绿；Win 若红大概率还是平台宏/头文件面——Win 工程
  师上次修的是 (std::min) 宏劫持同款）。

### E. 掼蛋侧待办（他们的账，REPLY12/13 已交）
- refit 接入三步（烤图 REFIT 旗标+命名、--rw1 出口、es.py 循环；
  /tmp/refit_lab/bake_refit.py 实验版可参考——**易失**）；
- 池路径回归等 A 闭合（评估腿 B 配置不受阻）。

## 3. 方法论沉淀（本线新增判决/坑律，已入 commit message 与 docs）

- 变批 tactic 世界差（n<8 vs ≥8）→ 固批旋钮；
- 表达式等价变形改变 float 世界（预转置常量 vs TRANSPOSE op）；
- refit≠重烤 float 世界（tactic 依赖权重值）→ 演化全走 refit 路径；
- cudaMemcpyBatchAsync cu12/cu13 ABI 双形态（failIdx 参数）；
- add_constant+REFIT=可换权重（引擎名=层名+" CONSTANT"，RW1 裸名+归一）；
- 幻影行（cursor 虚增未领号）×第二消费者（池下标）=野指针/投毒双态；
- ResetStatePool 跨流 memset 竞态 → 延迟零行；
- 默认流同步 D2H=观测者效应（抚平竞态）；
- fullwrite/append/headlive/state_pairs 四声明面与互斥护栏。

## 4. 关键工件索引

- 框架仓 tools/：probe_partial_d2h / probe_batch_invariance / probe_refit_world；
- gd 仓 notes/：FRAMEWORK_{HANDOFF,DATA1-10,REPLY1-13}（对话全史）；
- /tmp（易失）：gd_dbg（A 项取证树）/ refit_lab（refit 实验烤版）/ probe_*.cpp
  （已入仓的可删）/ segvcatch（信号抓栈小工具，值得入仓）。
