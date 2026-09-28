#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tune_farm.py —— 农场旋钮自动调参（Optuna + MedianPruner）

出处：棋类框架调研清单⑤——dlshogi utils/usi_params_optimizer.py（102 行：
trial 0 固定已知好值 / 每局回调报中间值 / MedianPruner 提前砍无望 trial /
支持分布式 storage）的吸收版，目标从"引擎对抗胜率"换成"farm 臂吞吐"。
见 docs/framework-survey-absorption.md。

设计：
  - 每 trial = sublegs 段独立短腿（同 seed 同负载），段末上报局/s 作中间值
    → MedianPruner 从第 2 段起砍低于已完成中位的 trial（sublegs=1 无剪枝面）；
  - trial 0 = 旋钮全缺省（基线锚点，dlshogi --init_params 同款思路）；
  - 纪律执法：同 seed 下指纹与决策数必须跨 trial 逐位同（"时序旋钮不改
    算术"红线）——首个成功段定锚，此后任何违约 trial 立即判剪并记
    user_attr，比人工扫描更严；
  - 缺省空间=纯时序旋钮（env）；--full 追加结构面（banks/slots/workers，
    指纹不变由 bench 三臂门背书，但单腿时长敏感，慎在共享机上扫）。

用法：
  # ort GPU 腿前置（本机 q35 env 为例）：ORT 目录 + cudart + cudnn(走 PATH)
  export FARM_ORT_DIR="C:/Users/41601/miniconda3/envs/q35/Lib/site-packages/onnxruntime/capi"
  export FARM_CUDART_DLL="C:/Users/41601/miniconda3/envs/q35/Lib/site-packages/torch/lib/cudart64_12.dll"
  export PATH="/c/Users/41601/miniconda3/envs/q35/Lib/site-packages/torch/lib:$PATH"
  python tools/tune_farm.py --trials 12 --sublegs 2
  python tools/tune_farm.py --trials 30 --storage sqlite:///bench_data/tune.db  # 可分布式
  # cpu 腿免上述前置：--backend cpu --model models/gomoku_mlp.fb16.onnx（slots 需配平 fb 值）

热噪声纪律：别拿相邻 trial 的差值当结论（GPU 温度漂移可比旋钮效应大）——
看 best vs 缺省基线的中位差，必要时 --trials 加倍复跑取稳定胜者。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import optuna

# 扫描空间：env 旋钮 → 候选档（档位来自仓内判决史的手扫梯子：
# CLAIM_SPINS=c8a1cc3（4000→1000 判决）、BANK_SPIN=判决17 链、HRTIMER=判决20 旗标）
TIMING_SPACE = {
    "FARM_CLAIM_SPINS": [0, 250, 1000, 4000],
    "FARM_BANK_SPIN": [0, 1, 2],
    "FARM_BANK_HRTIMER": [0, 1, 3],
}
# --full 追加：结构面（exe 参数；重，热噪声敏感）
STRUCT_SPACE = {
    "banks": [1, 2, 3],
    "slots": [8, 16, 32],
    "workers": [8, 16, 32],
}
# 参照系：首个成功段定锚（指纹/决策数），此后逐 trial 强制校验
REF = {"fp": "", "dec": -1}


def run_leg(exe, backend, model, chains, games, seed, banks, slots, workers,
            env_knobs, ort_dir, timeout):
    cmd = [exe, "--backend", backend, "--model", model,
           "--chains", str(chains), "--games", str(games), "--seed", str(seed)]
    if banks:
        cmd += ["--banks", str(banks)]
    if slots:
        cmd += ["--slots", str(slots)]
    if workers:
        cmd += ["--workers", str(workers)]
    env = dict(os.environ)
    # 受控环境：被扫旋钮先剥离（"缺省基线"的语义才成立），再上本 trial 值
    for k in TIMING_SPACE:
        env.pop(k, None)
    env.update(env_knobs)
    if ort_dir:
        env["FARM_ORT_DIR"] = ort_dir
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace", timeout=timeout)
    out = p.stdout + p.stderr
    if p.returncode != 0:
        print(out[-1500:])
        raise RuntimeError("[tune] 退出码 %d" % p.returncode)
    m = re.search(r"全链结束: (\d+) 局用时 ([\d.]+)s（([\d.]+) 局/秒）", out)
    fp = re.search(r"指纹 ([0-9a-f]{16})", out)
    dec = re.search(r"决策 (\d+)", out)
    if not m or not fp:
        print(out[-1500:])
        raise RuntimeError("[tune] 输出未匹配（指纹/汇总缺失）")
    return {"gps": float(m.group(3)), "fingerprint": fp.group(1),
            "decisions": int(dec.group(1)) if dec else -1,
            "wall": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="build/Release/gomoku.exe")
    ap.add_argument("--model", default="models/gomoku_cnn.fb16.onnx")
    ap.add_argument("--backend", default="ort")
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--sublegs", type=int, default=2, help="每 trial 短腿数（剪枝粒度）")
    ap.add_argument("--chains", type=int, default=64)
    ap.add_argument("--games", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--banks", type=int, default=2)
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--full", action="store_true", help="追加结构面 banks/slots/workers 进空间")
    ap.add_argument("--ort-dir", default=os.environ.get("FARM_ORT_DIR", ""))
    ap.add_argument("--cool", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--storage", default=None, help="如 sqlite:///bench_data/tune.db（可分布式）")
    ap.add_argument("--out", default="bench_data/tune_farm.json")
    args = ap.parse_args()

    space = dict(TIMING_SPACE)
    struct = dict(STRUCT_SPACE) if args.full else {}

    def objective(trial):
        if trial.number == 0:
            # 基线锚点：时序旋钮全缺省、结构面取本次 CLI 缺省
            env_knobs = {}
            banks, slots, workers = args.banks, args.slots, args.workers
            trial.set_user_attr("baseline", True)
        else:
            env_knobs = {k: str(trial.suggest_categorical(k, [str(c) for c in ch]))
                         for k, ch in space.items()}
            if struct:
                banks = trial.suggest_categorical("banks", struct["banks"])
                slots = trial.suggest_categorical("slots", struct["slots"])
                workers = trial.suggest_categorical("workers", struct["workers"])
            else:
                banks, slots, workers = args.banks, args.slots, args.workers
        fps, gps_all = set(), []
        for seg in range(args.sublegs):
            rec = run_leg(args.exe, args.backend, args.model, args.chains,
                          args.games, args.seed, banks, slots, workers,
                          env_knobs, args.ort_dir, args.timeout)
            fps.add(rec["fingerprint"])
            if len(fps) > 1:
                trial.set_user_attr("fingerprint_mismatch", sorted(fps))
                raise optuna.TrialPruned("指纹违约（同 trial 内段间不一致）")
            if not REF["fp"]:
                REF["fp"], REF["dec"] = rec["fingerprint"], rec["decisions"]
            elif rec["fingerprint"] != REF["fp"]:
                trial.set_user_attr("fingerprint_mismatch", [REF["fp"], rec["fingerprint"]])
                raise optuna.TrialPruned("指纹违约（vs 锚点）——该旋钮改了算术")
            if REF["dec"] > 0 and rec["decisions"] != REF["dec"]:
                trial.set_user_attr("decisions", rec["decisions"])
                raise optuna.TrialPruned("决策数违约（%d≠%d，负载漂移）"
                                         % (rec["decisions"], REF["dec"]))
            gps_all.append(rec["gps"])
            trial.report(rec["gps"], seg)
            if trial.should_prune():
                raise optuna.TrialPruned("MedianPruner：低于已完成中位")
            label = " ".join("%s=%s" % (k[5:].lower(), v)
                             for k, v in sorted(env_knobs.items())) or "defaults"
            if struct:
                label += " banks=%d slots=%d workers=%d" % (banks, slots, workers)
            print("[tune] t%-3d seg%d %8.1f 局/s（%s | 指纹 %s）"
                  % (trial.number, seg, rec["gps"], label, rec["fingerprint"]))
            sys.stdout.flush()
            time.sleep(args.cool)
        return sorted(gps_all)[len(gps_all) // 2]  # 段中位（抗单段热毛刺）

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3,
                                                                   n_warmup_steps=1),
                                storage=args.storage,
                                study_name="tune_farm" if args.storage else None,
                                load_if_exists=bool(args.storage))
    study.optimize(objective, n_trials=args.trials)

    df = study.trials
    base_t = next((t for t in df if t.user_attrs.get("baseline")), None)
    base_v = base_t.value if base_t and base_t.value else 0.0
    best = study.best_trial
    print("\n[tune] ==== 汇总（%d trials，完成 %d，剪枝 %d）===="
          % (len(df), sum(1 for t in df if t.state.name == "COMPLETE"),
             sum(1 for t in df if t.state.name == "PRUNED")))
    print("[tune] 缺省基线 %8.1f 局/s" % base_v)
    print("[tune] 最优   %8.1f 局/s（%.2f× 基线）params=%s"
          % (best.value, best.value / base_v if base_v else 0, best.params))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"best": {"value": best.value, "params": best.params},
                   "baseline": base_v,
                   "trials": [{"number": t.number, "state": t.state.name,
                               "value": t.value, "params": t.params,
                               "user_attrs": t.user_attrs} for t in df]},
                  f, ensure_ascii=False, indent=1)
    print("[tune] → %s" % args.out)


if __name__ == "__main__":
    main()
