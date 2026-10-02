# FRAMEWORK_DATA21 — 掼蛋线 err700 僵死案：框架侧修复清单+仪器+未决（移交文档）

> 掼蛋侧 → 框架侧移交。10-01~10-02 掼蛋演化产线持续遭遇 GPU 非法访存
> （cuda err700）导致的批次僵死。本文档汇总：已定谳并修复的框架 bug 五项、
> 常驻仪器四件、一项新功能、以及**仍未定谳的 err700 真凶**（含排除清单与
> 下一步）。所有修复在 gd 仓 vendored 副本验证（指纹锚逐位复现），框架本仓
> 已提交 591c544/4e8703d 两笔，其余见 §6 回灌清单。

## 0. 症状与现场签名（所有发作共有的指纹）

```
[bank] FLIGHT 死信: bank=N 已 30000ms 未回信 seq=XXX n=256
  | cuda_err=700 stream_qry=700 ev_b_qry=700
  | flag=4294967295(=毒化值=本批发射后 GPU 从未写完旗标)
  | site=state_tail(=发射线程最后被打的站点标签) seq=XXX
```

- 异步错：批发射完后 GPU 侧某 kernel 越界，旗标永不回写，收割线程 30s
  死信击杀进程（修复 #2 之前=整场僵死 300s 靠外层超时收尸）。
- 发作位置：腿内深批（seq 346~771），非 Init 期。
- 频率：单实例约每 25~100 腿一发（历史上多实例并行时显著放大）。
- **跨路径**：TRT 合成图路径与 K1/K23 全自定义路径**都发**——这是排查
  路线几次反转的关键约束。

## 1. 已修复的框架 bug（按严重度）

### 1.1 图捕获必败：sg_tbl 非锁页 H2D + 懒分配落入流捕获期【最重】

- **位置**：`trt_backend.cpp` `SgRun()`（sg_tbl_h/sg_tbl_d 分配）
- **根因**（两层）：
  1. `sg_tbl_h` 原为 `new int[]` 堆内存——`MemcpyAsync` 从非锁页内存在
     **流捕获期间非法**。状态会话（合成图/K23 两形态）的图捕获路径里含
     SgRun → 捕获必败， longstanding 日志
     `[trt][init] 批图捕获=败（回退在线邮箱）` 的真因。状态会话因此
     **100% 回退在线路径**——而在线路径正是 DATA20 时代挂死案的现场。
  2. 修锁页后发现第二层：池引擎会话走另一条创建路径（无 k23 分支），
     预分配漏罩 → 懒分配（HostAlloc+Malloc）发生在捕获期=非法 API。
- **修复**：sg_tbl_h 改 `HostAlloc` 锁页；预分配提前到会话创建期、
  全会话通用（ WarmupSite 统一入口）。
- **验证**：修复后"批图捕获=败"清零；gd 锚指纹逐位复现（行为中性）。
- **附带**：会话销毁路径同步改 FreeHost。

### 1.2 FLIGHT 死信不营救：卡死批次=整场僵死

- **位置**：`bank.cpp` 收割线程看门狗。
- **根因**：看门狗只 `flight_warned` 打一行报警，永不营救；卡死批次
  挂死整进程（外层 300s 超时才收尸）。
- **修复**：`BankFlightDeadExit()`——超过 `FARM_FLIGHT_DEAD_MS`（缺省
  30000，0=关）→ 全量诊断（DiagnoseSubmit 快照）落 stderr + `_Exit(86)`。
  调用方（gd 侧 es.py）视 rc=86 为瞬时故障：转储 stderr 尾+GPU 上下文
  释放等待+指数退避重拉。效果：wedge 损失从 ~5min 降到 ~40s，战役不再瘫。

### 1.3 冒烟假绿：陈旧旗标冒充新序号，图验证空转

- **位置**：`trt_backend.cpp` `MbSubmit()`/`CaptureGraph()` 冒烟。
- **根因**：冒烟的 `WaitFlag(seq)` 可能命中上一批残留旗标（实测重捕后
  "验证自旋 0.002ms"×4——真发射 3-5ms 不可能），图验证=没验证。
- **修复**：MbSubmit 发射前旗标毒化 `*mb_host = 0xFFFFFFFF`（不可能值），
  陈旧值永不可能匹配新序号。in-flight≤1 契约下安全。

### 1.4 跨组共享池零校验：潜伏的系统性 OOB footgun

- **位置**：`TrtBackend::ShareStatePool()`。
- **根因**：借组按**下标**绑定他组状态池，不校验本组 state_pair 行宽/
  行数与借来池是否一致。两引擎独立烤制（M 面 36864B/Kr 面 24576B 宽度
  各异），声明错位或版本不一致时，散射按本组 rb 写对方 stride 的池
  =**每批系统性越界**。当前掼蛋两引擎实测宽度一致未触发，但结构裸露。
- **修复**：逐对断言 `row_bytes`/`rows`，不等=响亮拒绑（stderr 指明
  pair# 与两侧数值）。

### 1.5 图双重销毁：销毁后不清指针

- **位置**：`DestroySession()` 的 graph/graph_c 销毁。
- **根因**：销毁不置空；Shutdown 与析构两路都走 → 第二次
  `cudaGraphDestroy` 返回 InvalidValue。compute-sanitizer 实测 6 发
  （纯噪音不伤运行），但污染错误流、可能掩盖真错、也是诊断里 err=1
  残留的来源之一。
- **修复**：销毁后置空。

## 2. 常驻仪器（破案所装，建议保留）

1. **首现场埋点**（框架仓 4e8703d）：会话级 `last_site`（发射站点标签，
   relaxed atomic）+ `gpu_err`/`err_seen`（sticky 错误首见快照）；
   ProbeDiag 随行输出 site/gpu_err。收割线程首见错误即落盘
   `[trt] GPU错误首现场: sess= site= err= seq=`。
2. **发射点标签全集**：st_graphc_launch/st_online_enqueue/p_*（MbSubmit
   各分支）+ k1_submit/k23a_launch/k23b_launch/k23_state_d2d（K23 链）
   + sg_gather/sg_scatter（SgRun）+ state_tail/partial_d2h。
3. **发射后错误探针** `LaunchProbe()`：每个带标签发射点后 `GetLastError`，
   **每个不同错误码各报一次**（初版一次性闸门被 err=1 残留吃掉导致真凶
   700 被噤声的教训）。
4. **rc=86 stderr 转储**（调用方侧）：死信击杀后自动转储 stderr 尾 25 行
   （含首现场+诊断）。

## 3. 新功能（掼蛋 DPO 铺轨，暂闲置）

`farm.h/cpp` 腿级分叉覆盖：`Farm::SetForkLeg(we_first, force_dec)` +
`FarmGameMain` we_first 覆盖；gd 侧清单第 6 字段 `"K:wf"` 解析（单局腿
chains=1 + 决策级强制换轨）。用途：同 seed 重放到指定决策点强制换动作
（DPO 偏好对产数）。配套 gd_adapter `g_gd_force_dec`（拒绝首采重抽=确定性
换轨）。**注意 gd 仓 gd_farm_main.cpp 还有同 rw1 连续腿跳过重复 refit 的
小优化**（DPO 清单 2F 行同权重场景）。

## 4. err700 真凶：仍未定谳（排查进展与排除清单）

**已静态/动态排除**：
- SgRun 内核行内寻址：PTX 五条拷贝路径（16B 快路 head/main/tail、字节
  兜底、f16 cvt）全部 `j<rb` 有界；grid<n；宿主 pid 表值域 {-1,pid,prows}
  有界；池分配含保留零行（rows+1）。
- 同组 state_pair：rb_in==rb_out 创建期强制相等。
- 跨组宽度错位：#1.4 校验上线后零报警（当前两引擎宽度一致）。
- **SgRun 生态整体**：`FARM_STATE_GATHER=0`（内核完全不跑，逐行 memcpy
  路径）下 wedge 照发（seq=752/771 实锤）。
- StBatchFlush/cuMemcpyBatchAsync：`FARM_STATE_D2D_BATCH` 缺省 0=该路径
  从未启用。
- 多实例互删（曾污染判据）：fixed_es 已加 flock 单实例锁，现单实例环境
  下 wedge 仍发。

**剩余嫌疑（收窄到二）**：
1. 状态池簿记的逐行 D2D memcpy（fill 侧 pool→行 / 尾侧行→pool，宿主算
   pid 地址——静态有界，但 pid/zero_pending 的跨流时序竞争仍是黑箱）。
2. TRT 引擎本体（enqueueV3+refit，fp16）——特定权重触发 tactic 内核
   故障的先例存在。

**进行中**：compute-sanitizer memcheck 常驻狩猎循环（256 局/腿，~4-5min/腿
instrumented）——非法访存会**当场**报出内核名+地址+host 栈，无需再猜。
初步短跑只复现了 §1.5 的销毁噪音（已修），真凶尚未在仪器下发作。

## 5. 对框架侧的请求

1. **回灌确认**（§6 清单）：vendored 与框架本仓的差异请框架侧 review 合入。
2. **SgRun/状态池时序**的专业复核：特别是 zero_pending 消费即清
   （fill 侧 store(0)）与散射侧读的 happens-before，以及 NewGame 的
   ResetStatePool 跨流 memset 与在飞批的序。
3. 若 sanitizer 捕获指向 TRT 本体：需要框架侧判断是否走 TRT issue
   （refit+fp16+固定 batch tactic 的非法访存有已知案例可查）。

## 6. 提交状态与回灌清单

框架本仓已提交：`591c544`（死信营救+旗标毒化）、`4e8703d`（首现场埋点）。
**仅在 gd vendored 副本、待回灌框架本仓的改动**（trt_backend.cpp）：
- sg_tbl 锁页化 + 全会话预分配（gd 仓 ccce58d/826061e 同步记录）
- K23/K1 发射点标签（0789e82 起）
- LaunchProbe 发射后探针（含"每错误码各报一次"升级）
- ShareStatePool 跨组校验
- 图销毁置空（双销毁修复）
gd 仓参考提交：ccce58d、826061e、938d5e5、0789e82；掼蛋侧调用方语义
（rc=86 重试链/GPU 释放等待）在 gd 仓 evo/es.py、evo/fixed_es.py。

— 掼蛋侧 2026-10-02
