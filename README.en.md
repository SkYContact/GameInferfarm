# GameInferfarm

**A general-purpose C++ inference framework for game-playing agents** — it takes
"run a huge number of homogeneous games concurrently, batch every neural-net
decision" and turns it into a game-agnostic library. Implement one
`GameAdapter` for your game; the farm handles concurrency, batching, GPU
submission and instrumentation.

The batched-inference layer for self-play training, RL evaluation, evolutionary
population evaluation and game-data generation: measured **116× over a naive
python loop** on the public Gomoku benchmark ([docs/benchmark.md](docs/benchmark.md);
numerator/denominator of every headline multiplier is tabulated there);
NVIDIA (ONNX Runtime CUDA / TensorRT) **and AMD (DirectML)** devices, including
mixed-vendor multi-GPU; bitwise determinism gates enforced in CI.

Keywords: self-play acceleration, batched neural network inference, game AI
throughput, RL rollout generation, evolutionary search, CUDA Graph, fiber
scheduler.

```
[中文版（主文档）](README.md)
```

## Why it exists

For self-play / RL evaluation / game-generation workloads, the bottleneck is
almost never the network itself. It is the OS scheduler tax of one thread
wakeup per decision, the in-process copy chain of feature assembly, and paying
a full batch's fixed GPU ticket for tiny batches. inferfarm turns these three
layers into game-agnostic machinery (measured 7.5-8.7×; see
[docs/design-judgments.md](docs/design-judgments.md), in Chinese):

1. **Fiber scheduler** — K worker threads + one fiber per game + per-worker
   wakeup queues. Waiting on inference = yielding the fiber (no cv sleep, no
   OS run queue): the wakeup tax is gone. Chain-to-worker affinity preserves
   thread_local semantics; chain-lifetime TLS frames are installed/uninstalled
   at every switch point.
2. **Zero-copy slot banking** — N banks × slots pinned rows, addresses fixed
   for life, one captured CUDA graph per bank. Atomic-cursor slot claim →
   **assemble directly into the slot** (zero in-process copies) → dispatch on
   full or timer → close-drain (bounded µs) → prefix-only H2D → async whole-
   batch replay → flag harvest → return to pool. Pool size = max in-flight
   batches = natural backpressure.
3. **Refit hot-swap** — RW1 weight blobs swap engine weights in milliseconds
   (evolution/evaluation loops skip re-baking engines per candidate).
4. **Census forensics** — all-atomic state machine (population identity X≡0),
   revival-latency histogram, per-thread CPU census, dispatcher-loop
   segmentation: measure first, explain later.

## Gomoku sample (no training required)

[examples/gomoku](examples/gomoku) demonstrates the full pipeline with a real
complete game: an **untrained one-layer MLP** (450→64→225, weights generated
from a seed = bitwise deterministic) vs a **rule-based opponent** (win if
possible, block if forced, heuristic otherwise). It loses — on purpose. The
seed protocol, direct-write slots, bank batching, harvest, and bitwise
determinism all genuinely work; swapping in your trained model is a one-line
model declaration change (or switch to the ort/trt backend) with zero adapter
changes.

```bash
build/Release/gomoku.exe --chains 8 --games 16 --show-board          # cpu backend (default, no GPU)
# Real-model artifacts (one-layer MLP -> batch-pinned onnx + TRT engine):
python tools/bake_gomoku_mlp.py --slots 8 --hidden 64 --out models/gomoku_mlp.fb8.onnx --trt models/gomoku_mlp.fb8.trt
build/Release/gomoku.exe --backend ort --model models/gomoku_mlp.fb8.onnx   # ORT (with CUDA Graph)
build-trt/Release/gomoku.exe --backend trt --engine models/gomoku_mlp.fb8.trt  # TRT (graph + mailbox)
```

Real-model measurements (local, 32 games × 2 banks): all three backends are
**bitwise comparable** (ORT and TRT produce identical fingerprints for the same
fp32 model; bank vs inline bitwise identical per backend). At toy scale the
CPU backend is fastest — the MLP is tiny, so the per-batch GPU ticket is pure
overhead; the GPU backends pay off at real-model scale (see
[docs/provenance.md](docs/provenance.md), in Chinese).

### Integrating your game

```cpp
#include "inferfarm/inferfarm.h"

struct MyGame : inferfarm::GameAdapter {
    void NewGame(uint64_t seed, bool we_first) override;      // start (reset fully = determinism)
    bool AdvanceToDecision() override;                        // advance to our next decision
    void AssembleInto(inferfarm::SlotWriter& slot) override;  // write features into slot rows
    int  CollectOutputs(inferfarm::OutputDest* d, int cap) override;  // declare output buffers
    void ApplyResult() override;                              // consume outputs, advance game
    void OnInferFail() override;                              // inference failure = loss discipline
    bool IsDone() override;  int Outcome() override;          // 1 win 0 loss -1 draw
    bool WeAreFirst() override;  inferfarm::ITlsFrame* TlsFrame() override;
};

int main() {
    inferfarm::FarmConfig cfg;              // workers/banks/slots/window/stagger/model...
    cfg.model.backend = "cpu";              // "cpu" | "ort" | "trt"
    inferfarm::Farm farm;
    farm.Init(cfg);
    farm.RunLeg([](int, void*) -> inferfarm::GameAdapter* { return new MyGame(); },
                nullptr);
}
```

Three contracts (full text in
[include/inferfarm/game_adapter.h](include/inferfarm/game_adapter.h)):
1. **advance and assemble contain no suspension points** — this is what keeps
   the bank's close-drain bounded;
2. **rows are independent** — no batchnorm-style cross-row ops (banks fire
   partial batches; stale tail rows are harmless);
3. **bitwise determinism is the adapter's responsibility** — seed protocol:
   `game seed = seed0 + chain*per + game`.

Games with thread_local state that survives a yield (script engines etc.)
wrap it in an [`ITlsFrame`](include/inferfarm/tls_frame.h); the scheduler
installs/uninstalls it at every fiber switch.

## Quick start

```bash
git clone https://github.com/SkYContact/GameInferfarm.git && cd GameInferfarm
cmake -S . -B build -G "Visual Studio 18 2026" -A x64   # or any supported generator
cmake --build build --config Release

build/Release/farm_test.exe    # determinism gates (G1-G7, all green or it ships not)
build/Release/toy.exe          # minimal toy (TLS-frame usage)
build/Release/gomoku.exe       # the Gomoku sample
```

TRT backend: `cmake -B build-trt -DINFERFARM_WITH_TRT=ON`
(`INFERFARM_TRT_INCLUDE_DIR` points at a dir containing NvInferRuntime.h).

## Backends

| Backend | Batch graph | Completion | refit | Role |
|---|---|---|---|---|
| `cpu` | — | immediate | ✓ (demo protocol) | full-stack validation without a GPU, CI |
| `ort` | enable_cuda_graph (bank sessions pinned to the dispatcher thread) | full device sync | ✗ (no such ORT API) | the no-TRT route: scheduling/copy/batching gains in full |
| `trt` | CUDA Graph capture (one per bank) | GPU mailbox 4B stamp + volatile spin | ✓ | production: fused submissions, fence tax gone, ms hot-swap |

All three implement the same `InferBackend` interface over the same banking
protocol — throughput differs in the submission layer, behavior stays bitwise
comparable.

## Inference cache (optional; idea absorbed from KataGo's NNCache)

Set `FarmConfig.cache_log2` (env `FARM_CACHE_LOG2`) > 0 to enable: the key is a
128-bit hash of the assembled input-row bytes plus a weight generation counter
(invalidated automatically on refit); a hit skips harvest/roundtrip and replays
the outputs byte-for-byte. **Where it pays (measured)**: the CPU backend
loses at low hit rates (53% hits ran 2.4x slower — abandoned rows still compute,
plus claim-machinery stampede) and wins at high rates (fully warm cache: 430 vs
153 games/s). GPU backends run a fixed batch shape, so garbage rows are free —
mechanically a pure win (at toy scale the effect is below run-to-run noise; the
real verdict waits for YGO-scale models). Bitwise identity is guaranteed by gate
G7 either way. Off by default.

## Multi-GPU (banks attached to devices; judgment 15)

`FarmConfig.devices`: one entry per device group — `(backend, device_id,
ort_ep, model, bank count)`. Each group has its own bank pool/window rotation,
and **chain c is pinned to group c%n_groups** — the key to cross-vendor rerun
bitwise determinism (measured: NVIDIA CUDA + AMD DML, identical fingerprints
across three runs). Two same-model NVIDIA cards keep the bitwise gate for free.
The inference cache namespaces keys per group.

Sample `--device` syntax (first replaces the primary device, later ones append;
`share=` is the chain-assignment weight — default splits evenly, set by
compute ratio on heterogeneous rigs, `share=0` parks a group):

```
gomoku --device ort,ep=cuda,dev=0,banks=2,share=4,model=m.onnx \
       --device ort,ep=dml,dev=1,banks=1,share=1,model=m.onnx,dir=D:/dml_rt/capi
```

**Partitioning law (measured)**: splitting chains thins each group's arrival
stream — at low load partitioning loses (even two CUDA groups on the same
card run slower; not an iGPU problem), while at high load two groups are two
parallel fill pipelines (512 games: even-split heterogeneous 1113 games/s vs
single-GPU 654). Weight by `share` (or park weak GPUs at 0) on real models.

AMD GPUs go through `ep=dml` (DirectML: host binding + synchronous Run; needs
an onnxruntime-directml build — `pip install --target <dir>
onnxruntime-directml`, point `dir=` at its capi directory; the name clash
with the CUDA build's onnxruntime.dll is handled automatically by the
framework via a renamed copy). TRT `device_id>0` guards are in place
(untested on this single-GPU machine; ready for twin same-arch cards).

## Environment knobs (explicit Config wins; env for quick experiments)

`FARM_FIBERS` `FARM_FIBER_WORKERS` `FARM_BANKS` `FARM_BANK_WINDOW_FLOOR`
`FARM_STAGGER_MS` `FARM_CENSUS` `FARM_CACHE_LOG2` `FARM_ORT_DIR`
`FARM_CUDA_DIR` `FARM_TRT_DIR`

## Layout

```
include/inferfarm/    public headers: types / backend / fiber_pool / bank /
                      cache / census / refit / game_adapter / tls_frame / farm
src/                  implementation (bank.cpp = banking protocol; backends/ = cpu|ort|trt)
examples/toy/         minimal toy adapter (TLS frames)
examples/gomoku/      the Gomoku integration sample
tests/                farm_test determinism gates + refit_probe
tools/refit_blob.py   authoritative RW1 weight-blob exporter
docs/                 design judgments / pitfalls / provenance (Chinese)
```

## Tests and gates

`farm_test` carries over the production verification discipline:
- **G1** bank vs inline bitwise identical (outcome + decision counts + per-game
  fingerprints) — a designed-in property; failure means a bug;
- **G2** identical reruns; **G3** fiber mode vs thread mode identical
  (behavioral proof of the TLS-frame discipline);
- **G4** census on = bitwise-identical results + population identity X≡0 +
  revival samples;
- **G5** refit: same blob twice = bitwise identical; different blob = must
  change; RW1 negative paths fail fast;
- **G6** a real game (Gomoku): bank vs inline bitwise identical;
- **G7** inference cache: on = off bitwise identical (including a second leg on a
  fully warm cache) + generation-invalidation gate + real hits;
- **G8a** multi-device groups (homogeneous simulation): partitioned = single-group
  bitwise identical + rerun identical (real heterogeneous = R4 / `--device`);
- **R1/R2** (optional; runs only when real-model artifacts exist, otherwise
  SKIP): ORT/TRT backends — identical reruns + bank vs inline bitwise
  identical (`gomoku_backend_test`);
- **R4** (optional; `FARM_DML_DIR` pointing at an onnxruntime-directml capi
  directory): cuda+dml heterogeneous leg completes + rerun bitwise identical
  (cross-vendor pinning determinism).

## Constraints and roadmap

- **Windows-first** for now. Platform-surface status: fiber semantics
  (fiber_pool.cpp, Windows Fibers) and backend DLL loading (LoadLibrary in the
  ort/trt backends) are Windows implementations; the platform bits in
  bank/census/farm (spin primitive / thread priority / timer) are already
  gated. A POSIX port = those two spots: the Switch family
  (ucontext / boost::context) + dlopen loading. C++17, CMake ≥3.16.
- Roadmap: ORT/TRT real-model benchmarks (paradigm borrowed from KataGo's
  benchmarkPureForward: barrier start + per-thread medians + wall clock),
  an fp16 bake tier, hybrid low-load dispatch
  (the cure for the banking tax in single-game scenarios), POSIX fibers, more
  game samples. (Multi-GPU has landed: cuda+dml heterogeneous verified with
  cross-vendor rerun bitwise identity; TRT device_id guards await a twin-card
  machine.)

## Naming

The repository is **GameInferfarm**; the C++ namespace/target keeps the short name `inferfarm`.

## License

[MIT](LICENSE). TensorRT / ONNX Runtime / CUDA are not redistributed —
install them from their official channels and point at them via
`ModelConfig` / `FARM_*`.
