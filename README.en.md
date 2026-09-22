# inferfarm

**A general-purpose C++ inference framework for game-playing agents** — it takes
"run a huge number of homogeneous games concurrently, batch every neural-net
decision" and turns it into a game-agnostic library. Implement one
`GameAdapter` for your game; the farm handles concurrency, batching, GPU
submission and instrumentation.

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
complete game: an **untrained network** (deterministic CPU backend model,
weights generated from a seed) vs a **rule-based opponent** (win if possible,
block if forced, heuristic otherwise). It loses — on purpose. The seed
protocol, direct-write slots, bank batching, harvest, and bitwise determinism
all genuinely work; swapping in your trained model is a one-line model
declaration change (or switch to the ort/trt backend) with zero adapter
changes.

```bash
build/Release/gomoku.exe --chains 8 --games 16 --show-board
```

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
git clone <this repo> && cd inferfarm
cmake -S . -B build -G "Visual Studio 18 2026" -A x64   # or any supported generator
cmake --build build --config Release

build/Release/farm_test.exe    # determinism gates (G1-G6, all green or it ships not)
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

## Environment knobs (explicit Config wins; env for quick experiments)

`FARM_FIBERS` `FARM_FIBER_WORKERS` `FARM_BANKS` `FARM_BANK_WINDOW_FLOOR`
`FARM_STAGGER_MS` `FARM_CENSUS` `FARM_ORT_DIR` `FARM_CUDA_DIR` `FARM_TRT_DIR`

## Layout

```
include/inferfarm/    public headers: types / backend / fiber_pool / bank /
                      census / refit / game_adapter / tls_frame / farm
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
- **G6** a real game (Gomoku): bank vs inline bitwise identical.

## Constraints and roadmap

- **Windows-first** for now (fibers via Windows Fibers; the POSIX port surface
  is confined to the Switch-family in fiber_pool.cpp). C++17, CMake ≥3.16.
- Roadmap: ORT/TRT real-model benchmarks, hybrid low-load dispatch (the cure
  for the banking tax in single-game scenarios), POSIX fibers, more game
  samples.

## License

[MIT](LICENSE). TensorRT / ONNX Runtime / CUDA are not redistributed —
install them from their official channels and point at them via
`ModelConfig` / `FARM_*`.
