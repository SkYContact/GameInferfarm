# 坑目录（血律——新库施工直接继承；每条都付过学费）

## 并发/协议
- **fiber 化必做 TLS 审计表**：同工人多局 fiber 共享线程 TLS，跨让出点存活的局内态
  必须收进帧（`ITlsFrame`），工人切换点装卸。危险=跨让出读写 thread_local。
  YGO 首版 27 项审计（AiBotTls）；本库把纪律做成了结构（tls_frame.h 的审计清单）。
- **帧所有权=适配器**：帧常是适配器成员的地址——调度器不得 delete 调用方帧
  （本库首夜崩溃案：RunLeg 末尾 delete 成员地址=无效释放）。
- **代理 subprocess 必带 timeout**（死锁人质案）；**20s 判死纪律**。
- **census 自身 ~5% 税**，仅取证开；埋点全原子+专职低频打印线程，不在热路径加锁。
- **收割先拷后投递**：bank 回池先于游戏恢复（新批可能立即复用槽行/输出 arena），
  适配器不得直读银行内存——OutputDest 缓冲由收割侧回填（拷贝是承重的，不是优化）。
- **ORT 图会话绑线程**（PerThreadContext 铁律）：创建/热身/回放必须同线程。
  银行会话钉在调度台线程；inline 会话强制关图。
- **CUDA context 创建前设 ScheduleSpin**（cuSetDeviceFlags），否则设备等待=Auto
  策略睡 1-3ms/批。
- **TF32 纪律**：TRT engine 烤制端 NVIDIA_TF32_OVERRIDE=0，运行端不一致=拒建
  context（fail fast，绝不静默分叉）。
- **栈缓冲读 engine 文件会爆栈**（1MB 级 0xC00000FD）——fseek 定长一次读入堆。
- **Git Bash heredoc 折叠反斜杠**：python heredoc 里 `\n` 会变成真换行写进源码
  （本仓首日连踩三次）；含反斜杠的补丁一律用编辑工具或 chr() 构造。

## 测量纪律
- **解释前先测量**：吞吐读数 ≥48 局/链才饱和；A/B 交替 ≥4 腿防热偏置（±10% 波动带）。
- **行为门必须同 games+同 chains+种子对齐**（种子宇宙=games×chains 拆分；
  YGO 侧 YGO_OPP_SEED=1+同 --seed）。
- **逐位门比较面要够强**：胜率聚合太粗（本仓 G5 假绿案：常量向量抹平动作区分度、
  胜场计数巧合相等）；用逐局指纹（GameFingerprint XOR）。
- **判定门空过的味道**：某 CHECK 恒真=比较面没喂到（本仓 run_with_refit 忘回填
  fp，同/异 blob 门双双失真——"同"空过、"异"假败）。
- **census X=0 是硬不变量**：live−R−Q−W 必须恒 0，否则人口失踪=状态机有漏转移。

## 游戏/模型契约
- **advance 与 assemble 无挂起点**（银行 drain 有界的前提）——组装内不得
  FiberSuspend/阻塞 IO。
- **行独立**：模型无 batchnorm 类跨行算子。银行不满整批照发、尾行旧数据无害，
  全系于此。RW1/CPU 后端的权重是**全行共享**的——换心影响所有行的对应动作位。
- **零基组装**：领槽即清零行——未写区与"零垫基线"逐位同（适配器的高水位清零
  类组装依赖行起点为零）。
- **推理故障=判负纪律**：不静默重试（会撕裂确定性），OnInferFail 显式处理。

## 构建/环境
- MSVC 聚合初始化带基类不行；`delete` 内部成员地址=UB；含原子/互斥的类型不可
  移动（vector 扩容炸）——用定长数组或 deque。
- TRT 版本宏 `NV_TENSORRT_VERSION_INT(major,minor,patch)` 是**函数式宏**，裸用=
  编译错；值宏是 `NV_TENSORRT_VERSION`。`getWeightsPrototype(name)` 一参返回
  Weights（10.16 面）。
- Windows 线程级 CPU 只有 CreateToolhelp32Snapshot 普查量得到（CUDA 上下文线程、
  EP 线程池）；GetThreadTimes 15.625ms 量化不可用于 µs 段——用 QueryThreadCycleTime。
- 交替腿防热偏置；引擎/量化烤制须 GPU 空载时进行。
