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

## 5. 手起复位（多手 episode；掼蛋 DATA4 实战回填，2026-09-29）

池清零挂在 NewGame——**换手发生在 episode 内部**（掼蛋 continue_next_hand）
的乘客，池状态会跨手泄漏（首测指纹分叉实锤）。两条路：

1. **图内复位（推荐，掼蛋已上线）**：引擎加 `rst[B,1]` 输入行（换手后
   首个决策=1），图入口对全部配对状态 select 清零。框架无关、两路径
   自动同语义（rst=0 恒等——旧引擎指纹逐位同为证）、新旧引擎/适配器
   双向兼容（rst 行 null 跳过 / 缺省 0=恒等）。状态行数+1 的行宽代价
   可忽略。
2. **框架中途复位 API（v2 挂账）**：`SlotWriter::StateReset()`——适配器
   在换手后首个 AssembleInto 里调用；银行记本槽复位旗，后端提交侧填充
   该行改读**保留零行**（池末行，init memset 一次、永不散射——pid 均
   <rows 故零行恒零），批尾散射照常覆写池行=新状态落池。无 memset、无
   跨流问题、与图内 rst 互为替代。触发条件=真有第二个多手乘客且不愿
   动图时再动工。

（Farm::ResetStatePool 现仅 NewGame 钩——中途面即上述 v2，不在 v1。）

## 6. 实战回执（掼蛋 DATA4，v1 首个用户）

- 指纹：池==主机逐位同（512 链 A/B 双腿 + 4096 链头条，恒等性三证）；
- census 四看全兑现且超预告：asm 120→**4.9µs**（预告 ~30）、coll 1.6µs、
  复活 12.14→4.59ms、乘客量子残余 ~24µs；
- 端到端：22,419 → **33,251 dec/s（累计 +48.7%）**；
- ①残余战场移交框架侧（claim/收割/自旋均摊）——调度台拆账仪器现成
  （FARM_CENSUS=1 的 `[banksched] 段/周期` 行：wait/poll/close/发车/
## 7. 排期（v1 已落地，2026-09-29 f7be552+f6a8807）

实现按 §2 分层：trt 面+R9 门已落地并经掼蛋 DATA4 实战验收（§6）。v2
（页池/换心失效/refit 交互/ort 面/中途复位 API=§5-2）按后续判决排。
