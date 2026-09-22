# 五子棋公开基准（Inferfarm Gomoku Bench）

> 一个**可复现**的对局吞吐基准：同一五子棋负载（同模型、同对手、同种子、
> 同局数）下，对比五种实现形态的吞吐。目的：回答"自博弈/评估/对局生成这
> 类'大量同构游戏 + 神经网络批量决策'负载，不同写法差多少"。
> 任何实现都可以来跑——负载定义、模型工件、评估口径全部公开（见下）。

## 负载定义

- **游戏**：15×15 五子棋，先连五者胜；我方=神经策略（argmax，合法过滤，
  平局取小格号），对手=内置规则玩家（能赢就赢 → 必须堵 → 8 邻域×3 + 中心
  + 种子抖动 LCG）。
- **模型**：4 层 CNN + 全局 FC，6.85M 参数，~90M MACs/决策（`own/opp` 两
  平面输入 → 225 点策略头）。权重由行为克隆训练（老师=连型评估启发式，
  4,000 局自博弈 341,043 样本 + 8 对称增强；`tools/gomoku_teacher.py` +
  `tools/train_gomoku_policy.py` 一键再生）。**训练后对内置规则对手胜率
  85.9%（1100/1280；先手 87.2%、后手 84.7%）**；未训练参考模型 0/16。
- **协议**：chains=64 / games=1280 / seed0=20260922 / 偶数局我方先攻；
  腿指纹（逐局 XOR + 局身份混淆）跨实现应一致——**吞吐与正确性一起交账**。
- **指标**：局/s（主）与 决策/s（对局长度无关的硬件口径）。本负载实测
  12.72 决策/局（16276 决策 / 1280 局）。

## 实现谱系与结果（2026-09-22，4 腿交替，中位数）

环境：i9-9955HX 16C/32T + RTX 5070 Ti Laptop 12GB；ORT CUDA EP
（onnxruntime-gpu，CUDA 12）；torch cu128 训练侧。

| # | 实现 | 形态 | 局/s | 决策/s | 相对最慢 |
|---|---|---|---:|---:|---:|
| 1 | python 串行 | 单局循环，每决策一次 batch=1 GPU 推理 | 19.4 | 236.7 | 1× |
| 2 | python 向量化 | 64 环境同步推步，每步 batch≤64（gymnasium VectorEnv / EnvPool sync 语义） | 26.2 | 357.1 | 1.3× |
| 3 | C++ 每链线程 + 逐次推理 | 64 线程，inline 会话（`--threads --inline`） | 200.7 | 2,552 | 10.3× |
| 4 | C++ fiber 池 + 逐次推理 | 16 工人 fiber，inline（`--inline`） | 200.1 | 2,544 | 10.3× |
| 5 | **C++ 推理农场** | fiber 池 + 2 银行×16 槽攒批 + CUDA Graph（默认） | **815.5** | **10,370** | **42.0×** |

- 行 1→2：python 内把推理攒成批只换来 1.3×——瓶颈不在 GPU 在解释器与
  同步屏障（所有环境等最慢者）。
- 行 3→5：同样的 C++、同一个 ORT、同一份 onnx，**银行攒批一项 = 4.06×**
  （4 腿批均 11.7-12.8 行/发车，门票摊薄 + GPU 多批在飞）。
- 谱系首尾（1→5）：**42.0×**。
- 正确性：12 腿（3 实现 × 4 腿）指纹全部一致（`5cf30db420b33388`）、
  决策数全部 16276、推理故障 0——三种并发形态行为逐位等价。

## 复现

```bash
# 数据 → 训练 → 导出（GPU，~7 分钟）
python tools/gomoku_teacher.py --games 4000 --procs 8 --out bench_data/gomoku_bc.npz
python tools/train_gomoku_policy.py --data bench_data/gomoku_bc.npz --epochs 3 \
    --export --slots 16 --out models/gomoku_cnn.fb16.onnx
# python 基线
python tools/bench_gomoku_python.py --games 1280 --envs 64 --arms serial,vec
# C++ 三臂 × 4 腿（腿间冷却，交替防热偏置）
python tools/bench_gomoku.py --legs 4 --games 1280 --chains 64 \
    --banks 2 --slots 16 --workers 16
# 胜率评估（vs 内置规则对手）
gomoku.exe --backend ort --model models/gomoku_cnn.fb16.onnx \
    --chains 64 --games 1280 --banks 2 --workers 16 --slots 16
```

`models/gomoku_cnn.fb16.onnx` 已随仓分发（27MB）——没有 GPU 训练条件也可
直接跑后三行。CPU 后端（`--backend cpu`）不支持该 CNN（点积后端仅一层
MLP）；未训练 MLP 范例走 `tools/bake_gomoku_mlp.py`。

## 生态位（为什么不用现成框架跑这个基准）

| 项目 | 定位 | 与本基准负载的关系 |
|---|---|---|
| EnvPool（NeurIPS 2022） | C++ 环境执行并行（~百万 fps 环境步） | 只解决环境侧；推理侧仍要自己攒批 |
| SampleFactory | 单机高吞吐 RL 训练系统（batched inference + rollout workers） | 绑定训练算法栈；"带自己的游戏来跑对局"需按其 env API 重写 |
| OpenSpiel | 博弈算法/环境广度（AlphaZero/MCTS 等） | 以广度为目标，非吞吐 |
| KataGo | 围棋专用自博弈引擎 | 单游戏，不可接入自己的游戏 |
| **GameInferfarm** | **游戏无关的推理层库：你的 GameAdapter + 你的模型 → 并发/攒批/图/取证全托管** | 本基准的主角 |

## 与产线的关系

本基准是"公开可复现缩小版"。产线同协议代码（YGO 游戏引擎，行宽 176.9KB
真模型）实测 42 → 317-368 局/s（7.5-8.7×），见
[docs/design-judgments.md](design-judgments.md) 与 [docs/provenance.md](provenance.md)。
