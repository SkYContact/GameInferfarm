# inferfarm（推理农场）

**通用 C++ 游戏决策推理框架库**——把"大量同构游戏并发推进 + 批量神经网络决策"这件事做成
游戏无关的库。YGO（游戏王）产线是第一个乘客与参考实现（42→317-368 局/s，7.5-8.7×）；
其他游戏按 `GameAdapter` 接缝接入。

## 四件游戏无关资产（全部跑过生产）

1. **fiber 调度器**（`fiber_pool.h`）：K 工人线程（默认=物理核）+ 每局一 fiber +
   per-worker 唤醒队列。推理等待=Switch 回调度器让出，不睡 cv、不进 OS 运行队列
   ——唤醒税由此消除（实测 ort_Run 4.55ms 里 3.3ms 是 OS 调度等待）。
   链-工人亲和保住 thread_local 语义；切换点装卸链寿命 TLS 帧（`tls_frame.h`）。
2. **零拷贝槽位银行制**（`bank.h`）：N 家银行 × slots 槽 pinned、地址终身固定、
   **每家一张专属预捕获图**（图与地址一夫一妻）。原子游标领号（先占在途再领号）
   →组装**直写槽**（零同进程拷贝）→满座自驱/闹钟发车→close-drain（有界 µs，
   不写超时不写迁移）→只拷前 n 行（行独立前提）→异步整批回放→旗标收割→还池。
   池容量=在飞上限=天然背压。
3. **refit 热换**（`refit.h` + `tools/refit_blob.py`）：RW1 权重 blob 毫秒级换心；
   量化尺度 parent 定死=候选比较共模自洽。TRT 后端独有（ORT 无此 API）。
4. **census 取证**（`census.h`）：全原子状态机（running/ready/wait_answer，X 恒 0
   硬不变量）+复活路径直方图+工人忙闲+线程级 CPU 普查+调度台循环分段。
   `FARM_CENSUS=1` 选通，默认关=零开销。

## 三后端（`InferBackend` 接口并列）

| 后端 | 图会话 | 完成检测 | refit | 用途 |
|---|---|---|---|---|
| `cpu` | 无需 | 即时 | ✓（toy 协议） | 无 GPU 全链验证、确定性门、refit 语义试验 |
| `ort` | enable_cuda_graph（银行会话绑调度台线程——PerThreadContext 铁律） | 整设备同步（不赌 ORT 流序） | ✗ | 不烤 TRT 的路线：调度/拷贝/攒批层收益全额，围栏税留在调度台线程 |
| `trt` | CUDA Graph 批捕获（一银行一图） | GPU 邮箱 4B 盖章+volatile 自旋（µs 级，绕开 WDDM 围栏） | ✓ | 生产路线：提交合并+围栏消灭+INT8 换心 |

## 快速开始

```bash
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release
build/Release/farm_test.exe      # 确定性门（G1-G5，全绿才算数）
build/Release/toy.exe            # 示范：玩具适配器 × CPU 后端 × fiber × 银行
```

TRT 后端：`cmake -B build-trt -DINFERFARM_WITH_TRT=ON`
（`INFERFARM_TRT_INCLUDE_DIR` 指向含 NvInferRuntime.h 的目录）。

## 接一个游戏

实现 `GameAdapter`（见 `game_adapter.h` 与 `examples/toy/toy_adapter.h`）：

```cpp
struct MyAdapter : GameAdapter {
    void NewGame(uint64_t seed, bool we_first) override;
    bool AdvanceToDecision() override;          // 契约1：无挂起点
    void AssembleInto(SlotWriter& slot) override;  // 直写槽行（零拷贝）
    int  CollectOutputs(OutputDest* d, int cap) override;
    void ApplyResult() override;
    bool IsDone() override;  int Outcome() override;
    ITlsFrame* TlsFrame() override;             // 跨让出 TLS 装进帧（血律）
};

FarmConfig cfg;   // workers/banks/slots/window/stagger/census/model...
Farm farm;
farm.Init(cfg);
farm.RunLeg([](int chain, void*) -> GameAdapter* { return new MyAdapter(); }, nullptr);
```

三条契约（违反=框架正确性前提破洞）：**①advance 与 assemble 无挂起点**（drain 有界的前提）；
**②行独立**（无 batchnorm 类跨行算子——银行不满整批照发、尾行旧数据无害）；
**③逐位确定性由适配器保证**（种子协议：game seed = seed0 + chain*per + game）。

## 目录

```
include/inferfarm/   公共头（types/backend/fiber_pool/bank/census/refit/game_adapter/farm）
src/                 实现（bank.cpp=银行协议；backends/=cpu|ort|trt）
examples/toy/        最小示范适配器（含 TLS 帧用法）
tests/farm_test.cpp  确定性门（G1 逐位/G2 复跑/G3 fiber vs 线程/G4 census X=0/G5 refit）
tools/refit_blob.py  RW1 权威导出器（θ→blob；银行家舍入与 ORT 逐位一致）
docs/                design-judgments（12 条实测判决）/ pitfalls（血律）/ provenance（来历与账本）
```

## 环境旋钮（`FARM_*`，显式 Config 为准、env 快速实验）

`FARM_FIBERS` `FARM_FIBER_WORKERS` `FARM_BANKS` `FARM_BANK_WINDOW_FLOOR`
`FARM_STAGGER_MS` `FARM_CENSUS` `FARM_ORT_DIR` `FARM_CUDA_DIR` `FARM_TRT_DIR`
`FARM_TRT_VERSION_INT`（与 YGO 产线 `YGO_*` 旋钮一一对应，见 docs/provenance.md）

## 状态与路线

- [x] 四资产抽层成库（协议逐句同源于 D:/ygo/ygopro/ai_core 现役代码）
- [x] 三后端接口并列（cpu/ort 编译+cpu 全门绿；trt 编译通过）
- [x] 确定性门：G1-G5 全绿（银行 vs inline 逐位、fiber vs 线程、census X=0、refit 同/异 blob）
- [ ] ORT/TRT 后端真模型实测（本机 q35 环境就绪即可跑；读数纪律见 docs/pitfalls.md）
- [ ] YGO 适配器回接（现役代码按 GameAdapter 接缝改写——参考实现已在 D:/ygo）
- [ ] 低负载混合发车（solo 银行慢 65% 的解药：内联发车+自旋，已设计未建）
- [ ] POSIX fiber 移植面（fiber_pool.cpp 内 Switch 族一文件收口）

命名 inferfarm（推理农场）为暂名（交接书 2026-09-22 待定项）；仓库独立于 D:/ygo。
