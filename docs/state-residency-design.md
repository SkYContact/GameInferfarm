# ③ 成对状态行设备常驻 — 设计定稿（2026-09-29，接掼蛋 FRAMEWORK 线四轮交接）

> 裁决输入：掼蛋 DATA3（asm 120µs=量子第二大块、满批 5.3MB 状态行税）。
> 本文替换 FRAMEWORK_REPLY4 §4 的 ③a/③b 阶梯——**③a（host arena echo）
> 判死，直接落设备池形态**。vLLM 式（链粘滞+池管理）为既定架构方向
>（KV 归属定案：语义归游戏侧、机制归框架侧）。

## 1. ③a host echo 判死（构造性论证，不动 GPU）

状态行每决策两份主机拷贝：收割 out_arena→乘客缓冲（86KB）、asm 乘客
缓冲→输入槽行（~90KB，asm 120µs 的大头）。融合条件=两拷贝共享稳定端点。
但**槽行按批借还且跨链复用**（链 X 本批行 3、下批行 7），收割时不知道
下批行号；银行无链身份（设计使然）。按链存中间仓=两份拷贝换两个执行者，
字节数不变——纯主机侧任何排布都省不了。**结论：省拷贝的充分条件=状态
不过主机；身份问题的关键在"谁的池行"，不在"哪个槽行"。**

## 2. 设计：设备状态池 + 提交/收割两侧逐行 D2D 接线

**链稳定身份现成就有：chain_id（v1 池下标=chain_id，零分配零回收）。**

引擎契约**零改动**（M_prev 进 / M_new 出，arena 张量照旧）——掼蛋不用
重烤图。框架在两端把状态行的 PCIe/主机段替换为设备本地搬运：

```
提交（发车前，银行流序）：
  非状态输入：H2D 前缀拷（现状）
  状态输入行 s：MemcpyAsync(in_arena 行 s ← pool[pid[s]] 行)   // D2D
收割（批尾，盖章前，同流）：
  非状态输出：部分行 D2H（判决②）
  状态输出行 s：MemcpyAsync(pool[pid[s]] ← out_arena 行 s)     // D2D 散射
  盖章殿后（流序契约不变）
NewGame：框架对全部银行会话流的 pool 行 chain_id memsetAsync
  （同零值多流写=良性；链串行⇒无并发读者；自流有序⇒后继读安全）
```

- **pid[slot] 数组**：Claim 时框架写（DriveGame 知 chain_id，穿参到
  Claim），BankCtl 持有（atomic，claim 写/发车读，drain 握手给 happens-before）；
  后端 Init 期 BindPoolPids 拿数组地址——**每批零接口流量**；
- **池归属=后端实例**（组内银行共享同一设备池；链→组钉扎⇒无跨组状态）；
- **跨流安全**：散射在批尾盖章前（同流）——host 见旗标⇒散射已执行⇒
  数据全局可见⇒链的下一决策（必然晚于旗标观测）在任意银行的读安全；
- **吃掉的四样**：asm 状态回写（120→仅特征 ~30µs 量级）、CollectOutputs
  状态回拷 86KB、状态行 H2D/D2H PCIe、满批税的状态行份额（out D2H 只剩
  logits）；
- **乘客契约窄化**：声明配对后，状态输入行**不得写**（框架 D2D 覆盖前
  内容未定义）、配对输出**可不注册 dest**（不拷主机；注册了=照拷，调试
  通道）；Claim 清零对状态输入行自动豁免（同 population 面先例）。

### 声明面（判决25 同门）

```cpp
struct StatePairDecl { std::string in, out; };
ModelConfig.state_pairs = {{"M_prev","M_new"}, {"Kr_prev","Kr_new"}, ...};
// 池行宽=配对行宽（LoadSpec 校验 in/out row_bytes 相等，不等=fail fast）
// FARM_STATE_POOL=0 杀手锏回主机往返（A/B 口径）
```

### 实现分层

1. **trt 后端**（掼蛋现役路径先行）：池分配（HostAlloc 零基一次 memset）、
   BindPoolPids、SubmitBatch 状态输入行 D2D 填充（H2D 跳过）、批尾散射+
   状态输出 D2H 跳过、ResetStatePool(row)（全流 memset）、状态会话恒走
   计算图/在线+图外尾段形态（4 段图内静态拷装不下动态行集）；
2. **ort 后端**：同机制后续补（fence 尾段同位插散射）；
3. **cpu 后端**：不适用（无设备池概念），声明即拒绝；
4. **农场/银行**：chain_id 穿参 Claim→pids；NewGame 重置钩子。

### 门（R9，状态化玩具）

- 玩具引擎：S_next=S_prev+x（跨决策累加=任何池错必爆）、policy=f(S_next)；
  onnx 手写（Add/ReduceSum，tools/bake_state_toy.py）；
- 双适配器对拍：池路径（不写 S_prev、不注册 S_next dest）vs 主机路径
  （写 S_prev+读 S_next 自累积）同 seed 指纹逐位同=主门；
- FARM_STATE_POOL=0 杀手锏腿；复跑腿；多链多局腿（池行粘滞+换局清零）。

## 3. 判决实验预告（落地后掼蛋侧）

census 四看：asm 120→~30µs、coll 回拷消失、[bank] 行 D2H 字节塌缩、
满批与非满批同税（状态行不再进 D2H）。预期端到端 +10-20% 起步（量子
570→~450 量级），叠加②的批容量余量。

## 4. 排期

本文件=动工令记录；实现按 §2 分层顺序，trt 面+R9 门完整落地后交掼蛋
rsync A/B（DATA4）。v2（页池/换心失效/refit 交互）按 v1 判决结果排。
