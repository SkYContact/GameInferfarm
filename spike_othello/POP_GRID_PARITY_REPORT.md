# 种群网格路由（v2）ONNX 对拍报告（2026-09-22，负载 A / 路由第二代）

## 命令

```
cd D:\inferfarm\spike_othello
C:/Users/41601/Miniconda3/envs/q35/python.exe export_pop_grid_onnx.py   # 全程 CPU EP
```

pop = [128, 41280]（0 号个体=bc_t3k theta_final.pt 真权重，1..127=随机 init_theta），
own/opp 随机 fp32，mid int64 ∈ [0,128)。两模型输入名次序 **own,opp,pop,mid**（已断言），
mid=**INT64**（已断言），pop 形状钉 **[128,41280]**，输出 policy [S,64]，S=1024 钉批。

## v2 网格图 vs v1 gather 图（同接口换代）

v1 每行 `Gather(pop, mid)` 物化 [S,41280] 行权重再逐层 bmm；v2 把行散射进
[P=128, J=8] 网格后整网格 bmm——**pop 原地 Slice 当批权重，零 Gather 零物化**：

1. x = cat(own,opp) [S,128]；j = onehot(mid)→CumSum(axis0) 取 mid 列 −1（个体批内
   出现序号，torch 侧 assert j<8）；idx = mid\*8+j；
2. X_grid = **ScatterND**(zeros[1024,128], idx[:,None], x) → [128,8,128]（没到的格子
   保持零，垃圾行无害——行独立契约，最后不 Gather 回）；
3. 每层 W = Slice(pop, 层偏移, axis=1)→view(P,o,i)ᵀ [P,i,o]，b=[P,1,o]，
   h = relu(bmm(X_grid, W) + b)（层序/偏移=v1 flat 布局一致）；
4. Y [128,8,64]→flat[1024,64]，**行序还原 policy = Gather(Y_flat, idx)**——散射与
   还原同一 idx，天然互逆。

实测导出算子集（opset 17，torch.onnx dynamo=False 与 v1 同款，v2.2 版）：
`Add, Cast, Concat, Constant, CumSum, Equal, Gather, GatherElements, Greater,
Less, MatMul, Max, Min, Mod, Mul, Or, Relu, Reshape, ScatterND, Shape, Slice,
Squeeze, Sub, Transpose, Unsqueeze, Where`——全部标准 ONNX 算子，无自定义；
ScatterND/CumSum/Equal/Where/Mod/Min/Max 均在图内（脚本内白名单断言 + **OneHot
已剔除断言**兜底，见 v2.1 修订）。

## A. 三个对拍（钉批交付图 S=1024，4 轮满占用+行洗牌——j 跳号 128/128 个体）

| 对拍 | max\|diff\| | 差异位数 | argmax 一致率 |
|---|---|---|---|
| ① ONNX 网格 vs torch 网格（同逻辑） | **0（逐位同）** | 0/262,144 | 1.000000 (4096/4096) |
| ② torch 网格 vs torch v1 路由（同数学不同批形 [P,8,·] vs [S,1,·]） | 9.16e-05 | 224,073/262,144 | **1.000000 (4096/4096)** |
| ③ 附 torch 网格 vs 单模型 forward_one（金标准） | 9.16e-05 | 224,073/262,144 | 1.000000 |
| ④ 附 ONNX 网格 vs ONNX v1 fb1024 对照（同 EP 同 kernel，纯图形状效应） | 9.16e-05 | 223,267/262,144 | 1.000000 |
| ⑤ 附 ONNX v1 对照 vs torch v1 路由（对照组自检，v1 口径） | 4.58e-05 | 203,135/262,144 | 1.000000 |

口径背景：|logit|max=294，②的**相对**差异 3.1e-7——末位量级（GEMM 批形 M=8×P vs
M=1 的 kernel 分块/累加顺序不同，与 v1 报告/CUDA-CPU 末位差先例同类）。②④的 argmax
翻转行 top1−top2 间隙 max=0（即无翻转，无近平局风险行）。**①=0 逐位同**：网格批形下
ORT(CPU/MLAS) MatMul 与 torch bmm 结果逐位一致——比 v1 fb128 的 1.53e-05 更干净。

## B. 散射-还原专项（动态 S 孪生图，同结构仅 axis0 动态，跑完即删）

S=P·J=1024 且不变量 ≤8 行时占用被算术强制全满（Σcnt=1024=128×8）——**空格子只在
S<1024（真实对局中局数递减）出现**，故用孪生图测 S<1024 各档随机占用：

| S | 空格子 | 不足 8 行 | 满 8 行 | 跳号个体 |
|---|---|---|---|---|
| 1023 | 0 | 1 | 127 | 128 |
| 999 | 0 | 12 | 116 | 128 |
| 512 | 20 | 91 | 17 | 94 |
| 100 | 95 | 33 | 0 | 26 |
| 37 | 111 | 16 | 1 | 11 |
| 8 | 123 | 5 | 0 | 2 |
| 1 | 127 | 1 | 0 | 0 |

结果：**ONNX 孪生 vs torch 网格 = 0（逐位同，全档累计）**；torch 网格 vs 金标准
forward_one = 9.16e-05（末位，同上口径）；argmax 一致率 1.0（全档断言）。每轮均通过
散射安全性显式断言：j∈[0,8) 且 **idx=mid\*8+j 无重复**（ScatterND 重复索引=未定义，
torch 侧 `assert j<8` 是不变量的守门人）。

## v2.1 修订（同日）：OneHot → Equal+Cast（CUDA EP 修复）

**故障**：农场 GPU 跑 fb1024 网格图，ORT 1.30 CUDA EP 的 OneHot 内核非法访存
（sequential executor 报 CUDA error 700，/OneHot 节点）——CUDA EP OneHot 实现的
已知毛病，非图逻辑错误（本地同版本 ORT 1.30.0 + CUDA 未复现，3/3 通过；应与
GPU/kernel 版本相关，农场为准）。

**修法**（export_pop_grid_onnx.py，等价标准算子组合，其余路径逐字节不动）：
`Equal(mid[:,None] [S,1] int64, arange(P)[None,:] [1,128] int64 常量)` → `Cast
int64` → 原 CumSum 路径不变。列号 arange 以 module buffer 钉死，导出后为图中
[1,128] int64 常量（导出函数有 `_has_const_cols` 断言 + "OneHot 不在图内"断言）。
Equal/Cast 均为 int64 精确运算 → j 逐位同 → idx 及全图结果逐位同。

**修复后全量对拍重跑**（口径与 v2.0 完全一致）：
① ONNX 网格 vs torch 网格 = **0（逐位同，0/262,144）**；② vs torch v1 路由 =
9.16e-05；③ 金标准 = 9.16e-05；④ vs ONNX v1 对照 = 9.16e-05；⑤ 4.58e-05；
B 散射-还原专项（S∈{1023..1} 含空格子/跳号）ONNX vs torch = 0 逐位、argmax
全档一致；argmax 一致率 1.000000（4096/4096）维持。

**本地 CUDA EP 冒烟**（新图，4 轮满占用随机 pop/mid）：PASS 4/4，
CUDA vs CPU max|diff| = 2.29e-05（末位量级，v1 REPORT 的 CUDA/CPU 先例同类），
argmax 翻转 1/4096 行且该行 top1−top2 间隙 7.63e-06 < 末位差（近平局行）。

**图内其余算子的 CUDA EP 支持面检查**（先只修 OneHot，以下为风险记录）：
- **ScatterND**：CUDA 内核存在且本图跑通；运行时打警告"reduction=='none' 仅在
  索引无重复时保证正确"——本图靠结构不变量守住（每个体 ≤J=8 行 → idx=mid*8+j
  数学上无重复；torch 侧 `assert j<8` 守门 + 对拍每轮显式 `idx.unique` 断言）。
  **唯一需要持续盯的契约**：适配器组装批时必须维持该不变量，违反=结果未定义；
- **CumSum**：CUDA 内核存在（含 int64）；axis 是图内 Constant 标量（CPU 驻留），
  标准形态无坑；未发现现行版本已知缺陷；
- **GatherElements**：CUDA 内核存在（int64 索引）；索引形状契约 [S,1] ⊆ 数据
  [S,P] 满足，无坑；
- **Equal(int64)/Cast/MatMul(批 cuBLAS)/Gather/Relu/Add/Mul/Sub**：CUDA EP 均
  有实现；Shape/Reshape/Slice/Squeeze/Unsqueeze/Concat 等形状类算子会被分到 CPU
  侧执行并隐式插拷贝——影响的是性能（可在农场 A/B 时观察），不是正确性。

## v2.2 修订（同日）：死路由块（修农场活性非确定性）

**根因**：农场批可部分填充（n<1024），行 n..1023 为陈旧行（旧 mid 仍是合法个体号）
→ CumSum→j→ScatterND 后 (a) 同个体行数可超 8 → idx 溢出到下一个体块；(b) 与新鲜行
格子碰撞 → ScatterND 重复索引原子写 → 新鲜行被覆盖（同配置重跑指纹不同）。

**改法**（农场把批尾 mid 毒化为 -1/int64 全 0xFF，配合图内四件事）：
1. 活值钳制 mid_live = Max(Min(mid, P-1), 0)（int64 走 Min/Max，Clip 不支持整型）
   ——保证取列索引合法；
2. 计数用**原始 mid**（Equal+CumSum 不变）：毒化行/越界行不落任何列 → 新鲜行计数
   只含新鲜行（毒化保证），任意行 j_raw ∈ [-1, 7]；
3. **死路由块**：mid<0 行 → 第 P+1 块 idx_dead = P·8+(j_raw % 8)（Mod Python 语义，
   -1%8=7）；死行间允许碰撞（输出不被收割；死块零权重 → 死行输出**恒 0**，碰撞后
   仍确定）；活行 idx = mid_live·8+j_raw。**越界行（mid≥P，如探针 999999）同入死块**
   ——按字面"钳制后当活行"其 j 取自 P-1 列纯新鲜计数：无先行新鲜行时 j=-1 →
   idx=(P-1)·8-1 反渗进 P-2 块；有 c 个先行新鲜行时 j=c-1 与第 c 个新鲜行撞格——
   均违反"活行输出与无死行版逐位同"，故为本版必要修正（记录在案）；
4. 权重面补零行：网格 [P,8]→[P+1,8]；每层 W=Slice(pop)ᵀ 后 Concat [1,i,o] 零行、
   b Concat [1,1,o] 零行（死块算垃圾=0，行独立不影响活块）；输出 Gather 同一 idx。

**A/B 回归**（全活行口径，与 v2.1 **逐位一致**——129 块 bmm 未扰动活块）：
①=0（逐位同）；②=9.16e-05（224,073/262,144 位）；③=9.16e-05；④=9.16e-05；
⑤=4.58e-05；B 专项 ONNX vs torch=0、vs 金标准=9.16e-05；argmax 一致率 1.000000
（4096/4096）+ 专项全档。

**C. 死行鲁棒性（新增对拍，钉批图，先造合法全活批取参考再毒化/撒越界）**：

| 档 | 死行(-1) | 越界行(999999) | 活行 | 活行差异（torch/ONNX 双口径） | 死行输出 | 复跑 |
|---|---|---|---|---|---|---|
| C0 | 200 | 50 | 774 | **0 位（逐位同）** | 恒 0 | 稳定 |
| C1 | 513 | 3 | 508 | **0 位** | 恒 0 | 稳定 |
| C2 | **1000** | 0 | 24 | **0 位** | 恒 0 | 稳定 |
| C3 | 0 | 500 | 524 | **0 位** | 恒 0 | 稳定 |

"复跑稳定"= 同 session 5 次 + 新建 session 1 次，全输出（含死行）逐位同。
每轮通过散射安全断言：活行 j∈[0,8)、活格 idx 唯一、死/越界行全落死块。

**本地 CUDA EP 冒烟**（v2.2 图，900 死行/轮）：3/3 OK，活行 CUDA vs CPU
max|diff|=2.29e-05（末位量级）、argmax 零翻转（0/372）、死行输出恒 0、CUDA 复跑
稳定；新算子 Mod/Min/Max/Where/Or/Less/Greater 在 CUDA EP 均可用。ScatterND 的
"duplicate indices" 警告仍在（死块碰撞，规格允许；输出恒 0 不受写序影响）。

**性能注记**：死块使每层 W Concat 多拷一份权重切片（[1,128,128]/[1,128,128]/
[1,128,64] ≈ 0.16 MB 常量 + 每前向 ~20MB 拷贝）；如成瓶颈可改为激活侧补零
（bmm 只算 [P,·] 活块，死块行 Concat 零激活），数学等价——本版严格按"权重面补
一行"实现，未做此优化。文件 0.70 MB（zeros [1032,128] 常量 + 死块权重）。

## 结论

- **网格路由数学正确**：散射(ScatterND)→网格 bmm→同 idx Gather 还原，在满占用、
  空格子、部分占用、j 跳号、S=1 边界全档与金标准/对照逐位或末位一致，argmax
  零翻转（4096/4096 + 专项全档）；
- **交付图可用**：othello_pop_grid.fb1024.onnx（v2.2，0.70 MB，已无 OneHot，CUDA EP
  可跑，**死/越界行免疫：活行输出与无死行版逐位同、复跑稳定**——活性非确定性已修）
  与 v1 完全同接口（own/opp/pop/mid，mid=INT64，pop=[128,41280]），适配器零改动
  （农场只需维持两件套：批尾毒化 -1 + 每个体 ≤8 行不变量）；热换不变（权重仍是
  运行时输入）；
- **对照隔离就绪**：othello_pop.fb1024.onnx（v1 gather 图同 S 钉批，0.01 MB）与
  网格图同 S/同 P/同 EP——农场侧 A/B 可把"批合并收益"与"图形状收益"分开计量；
- 与 v1 的 9.16e-05 末位差属既有口径（相对 ~3e-7，argmax 零翻转），非缺陷。

## 工件

- `D:\inferfarm\spike_othello\export_pop_grid_onnx.py`（导出+对拍一体，含动态 S 专项）
- `D:\inferfarm\models\othello_pop_grid.fb1024.onnx`（v2 网格图，S=1024/P=128/J=8）
- `D:\inferfarm\models\othello_pop.fb1024.onnx`（对照组，v1 gather 图 S=1024 钉批）
