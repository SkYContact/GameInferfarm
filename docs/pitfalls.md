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

## ncnn Vulkan 批（2026-09-26）
- **手写 param 的"底数 顶数"是双计数字段**：top 名给少了不报错——多出的
  参数 token 被吞成第二个 top blob 名（"0=1"成了 blob）、ParamDict 静默
  全空（transB 回落 0）、blob 总数超声明 → load_param 越界写堆。崩相=
  数百毫秒后的段错误/静默腐坏，与病灶相距十万八千里（dll 构建当场崩、
  pip 静态构建延迟崩，同错两脸）。写 param 后必须 dump `net.layers()`
  核对 bottoms/tops 计数。
- **ncnn Concat 的 axis 按 ncnn 维序不按 torch 直觉**：2D 输入 axis=0=
  沿 h 堆叠、axis=1=沿 w 拼接（"沿宽度拼"要写 0=1）。h=1 时两种写法
  扁平序相同=侥幸正确，批量化（h>1）立刻翻车。
- **ncnn 的 option 必须在 load_param/load_model 之前设**：pipeline 在
  load 期按 option 烧制（fp16 存储格式/CM 特化在 create_pipeline 选型），
  load 后改 option=pipeline 与数据格式错配（fp32 数据进 fp16 pipeline=
  挂死/AV）。且 dll 缺省 use_vulkan_compute=0——不显式设 1，load 出来
  的是 CPU 层图，运行期再开=混合路径非受控状态。
- **ncnn.dll（20260526）进程退出清理有 rip=0 空 函数指针 AV**（上游
  #2733，open）：全部工作完成后才发生，编排器认汇总行不认退出码；勿试
  FreeLibrary 提前卸载（vk 清理死锁）。python 侧"析构段错误"同源。

## 有栈上下文/纤程后端批（2026-09-27，判决 24）
- **ctx 不驻栈**：自写 context-switch 的上下文块放在栈顶下方=被首局前几
  层调用帧向下生长直接踩穿（症状=复活时 AV rip=0、rsp=栈帧垃圾；ping-
  pong 微基准不暴露——对端 ctx 恰好放全局）。ctx 放堆（或栈外），栈顶
  只留首跳参数槽（trampoline 首跳即消费）。Boost.Context 同款布局。
- **冷机微基准是伪高**：同一机器冷/暖态绝对数差 2.5-4×（SwitchToFiber
  62→23.3ns、CreateFiber 17.8→~4µs）——微基准必须暖机后三跑中位，冷态
  数只配进漂移带注记，勿入正账。
- **共享 GPU 时段跨窗口数字不可比**：同命令吞吐差 4×（他会话占卡）——
  A/B 只在同窗口交替腿内比（判决 12/24 两次踩实）。

## SysV AMD64 移植批（2026-09-27，fcontext Linux 面）
- **SysV 上切回后 XMM/MXCSR 不可信（与 WinFiber FLOAT_SWITCH 语义差）**：
  SysV AMD64 ABI 把**全部 XMM 与 MXCSR/FCW 定为 caller-saved**——fcontext
  的 SysV 切换体（src/fcontext_sysv.S）零 FP 保存面是按 ABI 正确，不是偷工：
  GCC/Clang 编译的 C++ 本就不跨 call 持 XMM 态。但含义必须写明：协程切出
  再切回后，XMM 内容=对端遗留值，不可信也无需可信。Windows 面语义相反
  （xmm6-15 非易变，MASM64 切换体显式保存）——**同一份框架代码在两平台
  的"切换保存 FP 态"承诺宽度不同**，移植依赖 FP 控制寄存器的自定义例程
  （FTZ/DAZ、x87 精度）时逐侧核对。
- **.S 汇编的 CMake ASM 编译器要钉到 C 编译器**：`.S`（大写）需要预处理
  （本仓有 `__CET__` 门），CMake `enable_language(ASM)` 在 Unix 缺省可能
  选中裸 `as`——预处理行被当注释吞掉，`FI_CET_ENDBR` 宏展开失败当场报
  错（未静默腐坏，但第一现场难读）。configure 前置
  `set(CMAKE_ASM_COMPILER "${CMAKE_C_COMPILER}")`（gcc/clang 驱动对 .S 先
  跑 cpp）。
- **间接跳目标必须 ENDBR64（CET/IBT）**：Ubuntu 的 GCC 缺省 `-fcf-protection`
  编译面，fi_swap 用 `jmp *%rax` 切进冷 ctx 的 trampoline=间接跳目标——
  无 ENDBR64 在 IBT 实启的机器上=#CP 故障。手法同 Boost.Context：
  `#ifdef __CET__ #include <cet.h>` 取 `_CET_ENDBR`，非 CET 面空宏。
- **缺少 `.note.GNU-stack` 段=可执行栈**：binutils ≥2.41 对缺段的 .o 告警
  并给产物打可执行栈标记——汇编文件尾部补
  `.section .note.GNU-stack,"",@progbits`。
- **非 Windows 的后端选择必须双向回退**：Linux 面 winfiber 工厂返回
  nullptr——选择层若只做"fcontext 失败回 winfiber"单向回退，C++17 缺省档
  （want=winfiber）在 Linux 直接 nullptr 出门=首次 Switch 解引用即炸。
  回退必须双向（fiber_backend.cpp），且每路回退留 stderr 注记。
- **宿主平台从未编译过的 #else 面=首编译即爆**：winfiber 空工厂引用
  IFiberBackend 但头包含在 #ifdef _WIN32 内、bank 两处 SetEvent 在
  HR 镜像路径裸奔、ort CreateSession 的 wchar 路径——全是"Windows 上
  永远不编译的分支"里潜伏的雷，一次 Linux 构建全炸出来。收口纪律：
  条件编译面的**两侧**都要能编译（有空工厂就得有头；Win32 调用点跟随
  对象创建点同门），移植批的验证载体=一台真 Linux 盒（本批用 8.222.179.238
  relay 盒 git archive 直传，cmake+2 核构建+全门运行 ~3 分钟/轮）。
- **基准"留位=编译过"的降级点会过期**：fiber_bench 首版把 A2 档门写成
  `#if defined(_MSC_VER)`，POSIX 面只打印一句指路——后端通了、基准没跟上，
  头注释里的降级注记成了唯一线索。移植收官批把门改成"切换体所在构建面"
  宏（FI_FC_BENCH：MASM64 与 SysV 两面开测），降级点拆除。教训：写
  降级注记时带上拆除条件（"等切换体过 POSIX 面"），移植批 grep 注记
  逐条清账。
- **跨平台"建删账"差一阶（含分配语义差）**：D2 档（1MB 栈+ctx+回收）
  Linux posix_memalign 实测 0.14µs/对 vs Windows VirtualAlloc ~3.3µs/对
  （23×）——glibc malloc 复用 vs 内核页 commit；且含义差一阶：Windows
  reserve+commit 按需提交 vs Linux 全量一次分配（fcontext Linux 档内存
  足迹注记的另一半）。建删摊销账跨平台不可直比，比账只同平台内 A/B。

## 声明式增量 H2D 批（2026-09-27，判决 25）
- **省字节≠省时间（逐段 memcpy 的 WDDM 提交税）**：增量 H2D 缺省逐段
  通道下，每段固定 ~5-10µs 提交税×每批段数——fb8 实测 8 段/批把 dep h2d
  抬 2.6×、吞吐反降 1.75-2×（PCIe 字节省了一半，时间反而更差）。增量
  传输上产线必须配套段聚合（FARM_H2D_BATCH=1 的 cudaMemcpyBatchAsync
  一次提交）；字节账与时间账分开算，dep h2d 是宿主提交面不是 GPU 时间线
  （判决 17"fence v4 判死"同款教训在增量通道复现）。
- **「旧深」先钳行界再算尾长**：synced 里的哨兵值（kFullSync=1<<30）与
  越界值直接当深度参与 `(旧深-新深)×stride` → tail≈4GB 的
  memset/memcpy 越界（虚增量+整行兜底组合实测段错误案）。哨兵/外来深度
  先钳到 [0,max_depth] 再做段运算。
- **影子必须只同步实传段**：增量 H2D 的 debug 哨兵靠"影子=设备忠实镜像"
  才有牙齿——若影子图省事整行/整面同步宿主，哨兵 memcmp 恒 0=空过假绿
  （"判定门空过"同源：比较面要喂到）。
- **性能 A/B 的通道开关要显式隔离**：append vs full 对比若混入
  FARM_H2D_BATCH 开关差异，会把"通道收益"误记成"增量收益"（本批靠补跑
  full+批拷贝腿才分离出通道自身 0.38→0.26s 的贡献）。每换一个 env 维度
  补一条对照腿。
- **声明消费即复位**：逐行深度声明是"本批语义"——SubmitBatch 消费后
  复位 -1+Claim 领槽再清一道。漏复位=上一任写手的陈旧声明被本批判读=
  按错段传输=静默漏传（正确性由整行兜底的 fail-safe 挡住，但增量收益
  静默消失且哨兵会报良性违约——两头难查）。
