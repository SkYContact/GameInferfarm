# 设计评审：C++20 协程版纤程池（判决实验，2026-09-27）

> 状态：设计定稿于实现前（判决实验纪律：测量→结论→设计→实现→判决）。
> 掼蛋侧反馈：纤程开销 ~12µs/决策，是 30k 局/s 负载的最大单项成本。
> 本文分三部分：①12µs 拆解账（实测）；②无栈不可达定理（冻结面下的构造
> 论证）；③选型与实现边界。

## 一、12µs/决策拆解账（测量优先）

### 1.1 原语单价（tests/fiber_bench.cpp，本机 9955HX，2026-09-27）

**机器态双跑注记**：首跑（冷态，WinFiber 切换 62ns/建删 17.8µs）与暖态
稳态（三跑稳定）差 2.5-4×——微基准绝对数必须报稳态，冷态数仅作漂移带。
下表=暖态三跑中位。

| 项 | 实测（暖态稳态） | 备注 |
|---|---|---|
| A. SwitchToFiber 往返 | **46.6 ns/往返 = 23.3 ns/切换** | FLOAT_SWITCH=产线旗标 |
| A/C +双侧首触 16KB | +51.6 ns/往返（≈+26 ns/切换） | 小帧工作集（L1 重灌面） |
| A/C +双侧首触 64KB | +569 ns/往返（≈+284 ns/切换） | L1 逐出→L2 带宽重灌 |
| A/C +双侧首触 256KB | +1640 ns/往返（≈+820 ns/切换） | 大帧最坏面（L2→L3 延迟带） |
| A2. fcontext 往返 | **13.9 ns/往返 = 7.0 ns/切换** | 自写 MASM64（3.3× 于 WinFiber） |
| A2 +双侧首触 64KB | +570 ns/往返 | 缓存项与 WinFiber 同阶（非原语项） |
| B. Post→pickup 冷（cv 真醒） | **p50 100-400 ns / p99 1-1.6K** | 工人空闲形态真醒税（双态带） |
| B2. Post 队列操作本体 | **22-134 ns/次** | lock+push+notify 空场 |
| D. CreateFiberEx+DeleteFiber | **~4-5.7 µs/对**（暖态） | 每局一次；冷态 17.8µs=伪高勿引 |
| D2. VirtualAlloc(1MB)+fi_make+Free | ~3.2-5.1 µs/对 | 同阶（VirtualAlloc 主导） |

### 1.2 真负载 census（gomoku fb8 fence，--chains 256 --games 4096 --banks 4
--workers 4，FARM_STAGGER_MS=0 FARM_BANK_SPIN=1 FARM_CENSUS=1）

| 项 | 实测（快腿 1.26s / 1092 局/s 共享 GPU 慢腿对照） |
|---|---|
| 决策数 | 38,629（9.43 决策/局） |
| **挂起次数** | **190,332-263,462 = 4.9-6.8 次/决策**（两腿）——不是 2！ |
| rev（投递→取走） | 均值 0.85-1.83 ms，p50 0.25-1.25 ms（队列驻留为主，qlen 峰值 57） |
| 工人 busy% | 73-75%（idle 25% = 就绪队列空等的可见面） |
| claim 等池登记 | 2.42 µs/决策（~5 次挂起的登记合计） |
| claim 快自旋 | **2963 ms/3.86 万决策 = 76.7 µs/决策**（kClaimSpins=4000 次自旋，
  占全部工人 CPU ~27%） |
| 提交前段 | 2.0 µs/次 |

### 1.3 归因账（每决策，gomoku 形状，暖态原语价）

```
挂起次数 n≈5-6.8（Claim 等池重试 churn 主导，见 1.4）：
  切换原语   n × 2 × 23.3ns          ≈ 0.25-0.3 µs  （原语本体——尾项；
                                       fcontext 7ns 仅再省 ~0.2µs）
  Post 路径  n × 22-134ns            ≈ 0.15-0.9 µs  （队列操作）
  切换点缓存 n × 2 × 26~820ns        ≈ 0.3-11 µs    （帧+栈工作集重灌——
                                       宽度由帧尺寸定；YGO 大帧=上限区，
                                       是可归因纤程 CPU 的主体）
  帧 Install/Uninstall 2n 次虚调用  ≈ 0.1 µs
建删摊销    ~4µs ÷ 9.43 决策/局      ≈ 0.4 µs       （每局一次）
唤醒延迟    rev 0.4µs~ms             = 延迟非 CPU——链深流水消化；只在就绪
                                       队列排空时浮出为工人 idle（25%）
bank 协议 CPU（非纤程项，同链常被误记进"纤程开销"）：
  claim 快自旋 76.7µs/决策（4 工人摊 ≈19µs）+ 登记/提交前段 ~4.4µs
```

**结论**：切换原语（无论 WinFiber 还是 fcontext 还是协程 resume）都是
**尾项**（<0.3µs）；可归因纤程 CPU 的主体 = **挂起次数 × 切换点缓存
重灌**，加上 bank 协议侧的自旋/登记 CPU 常被误记进"纤程账"。
掼蛋 12µs 口径与"中大帧 × 高挂起次数 + 协议 CPU 误记"的组合相容。
**开销不在"纤程"这个名字上，在挂起次数、缓存亲缘与协议自旋上。**

### 1.4 为什么挂起是 5-6.8 次而不是 2 次（bank.cpp BankTryRotate）

轮转出池=**全群惊醒**：`wake.swap(waiters_g[g])` 后逐个 FiberPost——一家
银行回池把整组 waiters（数十个）全部投回就绪队列；只有 1 个在重试
try_claim 时抢到，其余全部再登记+再挂起（=每人再交一轮切换+Post+缓存税）。
4 银行 × 256 链下每次决策平均 5-6.8 次挂起。**次数杠杆在银行协议侧
（投递收敛/领票制），不在切换原语侧。**

## 二、无栈不可达定理（冻结面下的构造论证）

**命题**：在 fiber_pool.h 冻结面（Configure/RunLeg/FiberGameFn/
FiberCurrent/FiberSuspend/FiberPost 签名与契约不动）下，纯 C++20 无栈
协程不能替换现役挂起机制。

**证明**（构造）：
1. C++20 无栈协程的挂起只能发生在协程函数体内；挂起传播要求从顶层到
   挂起点之间的**每一帧**都是协程帧。
2. 挂起点的调用栈：worker→FiGameMain→game_fn(=FarmGameMain)→DriveGame
   →DriveDecision→Claim/SubmitWait→FiberSuspend——全部是框架代码
   （bank.cpp/farm.cpp），表面上看"翻 Chain 即可"。
3. 但 `FiberGameFn` 是**普通函数指针**且其契约是"**返回时局已收卷**"
   （fiber_pool.h）：池以 game_fn 返回=局终来做链续跑/收卷。
   - 若 FarmGameMain 变协程：game_fn 指针签名冻结，不成立；
   - 若 FarmGameMain 保持普通函数、内部启动 DriveGame 协程：协程在深处
     挂起时控制流**返回到 FarmGameMain**，此时局未收卷——FarmGameMain
     要么返回（违约：池按收卷处理=错）要么阻塞等待（等价线程模式=灾难）
     要么**有栈切换让出工人**（=保留有栈原语，且每次挂起一次）。
4. 因此：**任意 C++20 化方案中，"每次挂起一次有栈切换"不可消去**——
   无栈层至多把有栈切换从"深层"挪到"浅层"（FarmGameMain 处），次数不变。
   ∎

**推论**：C++20 化的可得收益上限 = 有栈切换**原语单价差**（62ns vs
~20ns，×2×n ≈ 0.5µs/决策）——对拆解账的每一主导项（缓存重灌、建删、
队列驻留）**全部无贡献**：栈还是那条栈（缓存面不变）、栈还得分配
（建删面不变）、队列还得投递（唤醒面不变）。

## 三、四方案对照与选型

| 方案 | 消掉哪部分 | 省多少（对 1.3 账） | 判定 |
|---|---|---|---|
| (a) 框架内决策等待点协程化（反转 DriveDecision） | 无——定理 3：次数不变，仅换层 | ≈0；且 bank/farm 全链 ifdef 分叉，投递-挂起不变量双份维护 | **判死**（定理+账） |
| (b) 唤醒队列存 coroutine_handle 顶层协程 | 无——深层挂起仍走有栈原语 | 换皮 | **判死**（定理） |
| (c) 自写 x64 fcontext（Boost.Context 思路） | 切换原语 23.3→7.0ns | ≈0.2-0.3µs/决策（<2.5%，跑间噪声带内）；**POSIX 路线的真地基**（SysV ABI 同构更简） | **实现**（作可移植性+原语差实测载体；端到端预期=不可判） |
| (d) IFiberBackend 抽象 + C++17 回退档 | 无直接收益 | 可移植性交付：WinFiber（现役）与 fcontext 同接口互换 | **实现**（主体交付） |

**选型 = (d) 主体 + (c) 载体**：
1. `IFiberBackend`（库内私有接口）：ConvertThread / ConvertBack /
   Create / Switch / Destroy 五原语；WinFiberBackend=现役调用 1:1 提取
   （C++17 构建零行为差）；FcontextBackend=自写 MASM64 有栈切换
   （248B 上下文：GP 非易变 8 + XMM6-15 + FCW/MXCSR；全量 commit 栈）。
2. 构建档：缺省 C++17（回退档=现役原样，缺省后端 winfiber）；
   `INFERFARM_CORO20=ON` → /std:c++20 + 编译定义 + **缺省后端=fcontext**
   （FARM_FIBER_BACKEND env 双向覆写，2×2 可隔离归因）。
3. **档内无协程对象**——如实声明：协程 facade 经定理与拆解双判死，
   本档的实体=可移植有栈后端 + C++20 工具链通路（facade 类实验的
   编译期地基）。把 (c) 冠以"C++20 封装"只剩伪装价值，不做。

**真正的杠杆（本次不动码，入判决书路标）**：
- 挂起次数收敛（轮转投递从全群惊醒改领票/限量投递）≈ 省 n 项的 3-5×；
- fiber 建/删复用（每链一 fiber 循环用）≈ 省 1.9µs/决策；
- 切换点缓存亲缘（工人优先取"刚让出的那个 task"=L1 热恢复）≈ 省缓存项的大头。
三项都是队列/协议侧改造，任何后端（含 C++20 协程）都替代不了。

## 四、实现边界（如实声明）

- fcontext 后端与 WinFiber 的语义差：
  - 栈=全量 commit（FARM_FC_STACK_KB，缺省 1024KB），无 guard-page 自动
    生长——内存足迹 256 局 × 1MB commit 面大于 WinFiber 的按需提交；
    Windows=VirtualAlloc reserve+commit，POSIX=posix_memalign 全量（降级点
    已注记，2026-09-27 Linux 移植）；
  - TEB StackBase/Limit 不随切换更新（WinFiber 会）：**异常/SEH 穿越
    切换点=未定义**（两后端同罪——产线契约本就无 throw 面）；
    栈溢出无硬件兜底（全 commit 下越界=相邻页踩踏）；
  - 切换保存面=ABI 最小集且**正确**：Windows=GP 非易变+XMM6-15+
    MXCSR/FCW（MSVC ABI xmm6-15 非易变，比 FIBER_FLAG_FLOAT_SWITCH 全 FP
    窄）；SysV AMD64（Linux）=GP 非易变 6+rip+rsp 共 64B，**零 FP 面**
    （SysV 全 XMM/MXCSR caller-saved——切出切回后 XMM 内容不可信也无需
    可信，ABI 义务；语义差注记见 src/fcontext.h SysV 档头注释）。
- 深栈限制覆盖面：本实现是**全量替换切换原语**，无"部分挂起点"问题
  （选的是有栈路线；无栈路线的覆盖面问题见定理，已判死）。
- **POSIX 面已落地（2026-09-27，本设计 (c) 的"可移植性载体"兑现）**：
  src/fcontext_sysv.S（GNU as，SysV AMD64，参考 Boost.Context
  jump/make_x86_64_sysv_elf_gas.S，BSD 风格许可 attribution 在源文件头）+
  fiber_pool 去 Windows-only + ort dlopen 面 + CI ubuntu job（双标准档跑
  farm_test cpu 门）。验收协议第五节由 CI 常驻执行。

## 五、验收协议

1. 行为门：farm_test ALL PASS，四象限 {C++17,C++20}×{winfiber,fcontext}
   全绿+指纹同（fcontext 档逐位=winfiber 档：切换原语不改任何算术）。
2. 性能门：gomoku fb8 fence（--chains 256 --games 4096 --banks 4，
   FARM_STAGGER_MS=0 FARM_BANK_SPIN=1）**交替腿 ≥3 取中位**+指纹全同
   （共享 GPU 时段两侧对称污染——方向可信幅度存疑，判决书标注）。
3. 原语门：fiber_bench 增补 fcontext 档，直接量切换单价差。
