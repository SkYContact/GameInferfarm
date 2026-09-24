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

## 评审轮教训（2026-09-22 子代理 codereview 后修复批次）
- **错误路径与快乐路径的纪律差**：P0/P1 全部集中在 init 失败清理（双重
  DestroySession）、中段失败泄漏（TRT 五处只 delete 结构体不回收资源）、
  工人退役无记账（RunLeg 永久挂死）——fail-fast 纪律必须同样覆盖回收路径：
  **销毁后置空指针；中段失败统一走 DestroySession；退役者按剩余工作量记账**。
- **"防挂死"防御不得引入新死法**：SubmitWait 防御分支对**正在运行**的 fiber
  调 FiberPost = 把在跑的 fiber 投进就绪队列（双重调度/UAF）。等待者=调用者
  本人时直接置败返回，投递只留给真正挂起的写手。
- **停机要叫醒所有睡着的人**：Shutdown 路径须唤醒挂起领槽者（弃领退出），
  并在头文件声明前置条件（腿全部返回）——否则停机=换一种挂死。
- **数组写点与"默认关=零开销"要同门**：census 定长数组的写点全部以 on 为门
  （指针非空≠开启）；workers 配置钳到数组容量。
- **聚合掺混只能用身份不能用序**：完成序在并发两腿间不同（已入坑），本轮
  再确认：掺链号+局号。

## 后端能力位教训（2026-09-22 五子棋 CNN 基准首跑抓出 P0）
- **满座自驱发车 × ORT 图会话 = 线程违规**：写手 fiber 就地
  BankCloseAndDispatch → ORT 图会话在非创建线程回放 → ORT 内部触发**重新
  捕获**（`CUDA failure 900: operation not permitted when stream is
  capturing` + `901: previous error during capture`）→ 整批弃答 → 判负纪律
  连坐（chains=64 实测推理故障局 160-272/320，且故障数逐跑不同=指纹不可
  复现）。**低负载（chains=16）timer 发车为主不触发**——高压才显形，静态
  评审看不见，只有高压基准能抓。修复：`DispatchFromWriterOk()` 能力位
  （ORT=false 满座只 Notify，调度台"满座即发"兜底 µs 级；TRT 图回放线程
  无关/CPU 无图=true）。教训：**"后端线程亲和"必须是后端声明的显式能力，
  协议层不得默认全体后端同权**；修后银行/inline/复跑三者指纹在 CNN 上
  逐位同（分歧源就是故障批，此前的"conv kernel 数值差"假说不成立）。

## 测量纪律
- **解释前先测量**：吞吐读数 ≥48 局/链才饱和；A/B 交替 ≥4 腿防热偏置（±10% 波动带）。
- **行为门必须同 games+同 chains+种子对齐**（种子宇宙=games×chains 拆分；
  YGO 侧 YGO_OPP_SEED=1+同 --seed）。
- **逐位门比较面要够强**：胜率聚合太粗（本仓 G5 假绿案：常量向量抹平动作区分度、
  胜场计数巧合相等）；用逐局指纹（GameFingerprint XOR）。
- **判定门空过的味道**：某 CHECK 恒真=比较面没喂到（本仓 run_with_refit 忘回填
  fp，同/异 blob 门双双失真——"同"空过、"异"假败）。
- **顺序无关聚合不得掺完成序**：腿指纹用 XOR（顺序无关）是为并发两腿可比；
  掺入 games_done 计数器防抵消——计数是**完成序**，并发下两腿完成序不同
  → 指纹假异（G1/G2/G6 三门连红）。正解=掺**局身份**（链号+局号）。
- **GPU 后端探针要自己搬数据**：探针直跑 Run/enqueue 而不经生产提交路径时，
  H2D/D2H 都要显式做（ORT 探针两连坑：漏 D2H 读陈旧主机 arena、跑图漏 H2D
  导致两图案不可分辨）。
- **census X=0 是硬不变量**：live−R−Q−W 必须恒 0，否则人口失踪=状态机有漏转移。

## 游戏/模型契约
- **缓存键必须含全部语义维度**（KataGo release note 真事故：eval cache 跨
  搜索参数错误复用）：本仓键=组装行字节+权重代次 gen+设备组号+mid 行号。
  新增任何"影响输出但不在输入字节里"的维度（换心/population 换代/异构设备）
  都必须掺进键或推代次——少一个维度=静默错答案，且指纹门会照常绿（同错）。
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

## 多设备改造（2026-09-22）
- **残留成员=空指针虚调用**：多组化时 BankScheduler::InputRow（公共包装）仍
  用旧 impl_->be，而 InitGroups 从未赋它——首个组装即段错误且崩点在适配器
  帧内（误导）。教训：删成员别留成员，让编译器逼你交出所有用点。
- **Windows 基名去重**：同名 dll 不同目录只能驻留一颗——双 ORT 共存必须
  拷贝改名（%TEMP%），依赖解析靠 PATH 前插（改名副本自身目录无依赖）。
- **DML EP 的 iob 输入忽略**：预绑 CPU 输入读恒零（输出绑定却通）——每次
  Run 前新鲜 CPU OrtValue 重绑输入即愈；probe 两图案门是此症的哨兵。
- **pip --target 装 ORT-DML**：勿装进主环境（顶掉 onnxruntime-gpu）；
  `pip install --target <目标目录> onnxruntime-directml`，dll 在
  `<目标目录>/onnxruntime/capi/`。
- DML 设备枚举序号=DXGI 适配器序号（dev0/dev1 哪个是核显枚举定，本机
  dev1=610M）；DML 打印的错误消息可能因系统 locale 非 utf-8 解码失败——
  属包装层噪音，不是失败原因。
## 绑核/affinity 批（2026-09-24）
- **变长系统记录的步进锚=头 8 字节，不是 sizeof(结构)**：VS18 新 SDK 的
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX sizeof=80 而实记录 48——
  `p + sizeof <= end` 把首条误判截断=phys 枚举恒空。字段偏移同理脱节
  （GroupMask 按头文件读会越界到下条记录）——python ctypes 逐字节核出
  实布局后手动偏移读（GroupCount@30/Mask@32/Group@40）。
- **越界移位先于范围检查=UB 静默绕过**：`(ULONG_PTR)1 << 9999` 在 x86
  按移位量 mod 64 回绕，9999 号核曾借此混过检查真绑上——范围检查必须
  在移位前。
- **追加式枚举的收割点必须唯一**：探测砍除的 spec_out 传给组 0 每家银行
  =IO 重复入表（own/opp/policy ×2）——单组腿无结构对拍从未暴露（多组
  盲区）。收割点唯一化（只挂首银行）。
- **性能门的断言面自身可空过**：phys 门曾靠垃圾段 "-1" 被当区间
  [-1,1] 造出的假元素空过成假绿（枚举恒空无人知）——解析器拒负核号+
  断言面只认真枚举（R6 空过防线同源教训：比较面要喂到）。
- **G15 门自身残留 env 假红**：越界门只清 worker env 不清 sched——调度
  台照常绑上把 pinned 探针抬高。门改 env 前先盘全部门面。
- Git Bash heredoc 传 python 源码含 `\n` 的 pattern 会被折叠成真换行
  （第 4 次踩）——替换/匹配一律走编辑工具。
## DLL 解析链批（2026-09-24 晚）
- **空目录拼分隔符=根路径**：`dir.empty() ? dll : dir + "\\" + dll` 的三分
  写法才是对的（cudart_dyn 原生正确）；`dir + "\\" + dll` 在 dir="" 时落成
  "\onnxruntime.dll" 根路径必败——"空=系统搜索"的意图从未兑现（对外反馈
  2026-09-24）。ort/trt 两处已修齐。
- **System32 永远先于 PATH**（标准搜索顺序）：裸名加载会先命中 System32
  的陈年同名 dll（实测本机 System32 有 ORT 1.17.1，PATH 前插 1.30 永远
  轮不到）。框架侧正确应对=版本门 fail fast+指路（GetApi 失败信息提示
  显式设 FARM_ORT_DIR），不静默拿旧版。
