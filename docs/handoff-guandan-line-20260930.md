# 掼蛋线交接档 — 2026-09-30 晨（会话归档，新会话入口）

> 十轮交接（gd:notes/FRAMEWORK_HANDOFF.md + DATA1-10 / REPLY1-13）的累计
> 状态与挂账清单。新会话从本文件起步。全部数字锁内实测；回执链在
> 掼蛋仓 gd/notes/，判决原文在 docs/design-judgments.md 与
> docs/state-residency-design.md。

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
