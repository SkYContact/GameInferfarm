#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_gomoku.py —— 三臂吞吐对照：朴素线程 / fiber inline / 完整农场（银行攒批）

同一模型、同 chains/games/seed 三臂对照（对局路径逐位同=同负载），多腿交替
防热偏置；腿末校验跨臂指纹一致（并发正确性随吞吐一起交账）。

臂：
  naive  = --threads --inline   （每链一线程 + 逐次推理 = 常规写法基线）
  fiber  = --inline             （fiber 池消唤醒税，推理仍逐次）
  farm   = 默认                 （fiber + 银行攒批 + CUDA Graph）

用法（Git Bash / Windows，q35 只为借 onnxruntime.dll）：
  python tools/bench_gomoku.py --exe build/Release/gomoku.exe \
      --model models/gomoku_cnn.fb16.onnx --legs 4 --chains 64 --games 1280 \
      --banks 2 --slots 16 --workers 16 --out bench_data/bench.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ARMS = [
    ("naive", ["--threads", "--inline"]),
    ("fiber", ["--inline"]),
    ("farm", []),
]


def run_leg(exe, extra, args, arm):
    cmd = [exe, "--backend", "ort", "--model", args.model,
           "--chains", str(args.chains), "--games", str(args.games),
           "--workers", str(args.workers), "--slots", str(args.slots)]
    if arm == "farm":
        cmd += ["--banks", str(args.banks)]
    cmd += extra
    env = dict(os.environ)
    if args.ort_dir:
        env["FARM_ORT_DIR"] = args.ort_dir
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace", timeout=1800)
    wall = time.time() - t0
    out = p.stdout + p.stderr
    if p.returncode != 0:
        print(out[-2000:])
        raise SystemExit("[bench] %s 臂退出码 %d" % (arm, p.returncode))
    m = re.search(r"全链结束: (\d+) 局用时 ([\d.]+)s（([\d.]+) 局/秒）", out)
    fp = re.search(r"指纹 ([0-9a-f]{16})", out)
    dec = re.search(r"决策 (\d+)", out)
    fail = re.search(r"推理故障局 (\d+)", out)
    if not m:
        print(out[-2000:])
        raise SystemExit("[bench] %s 臂输出未匹配" % arm)
    rec = {"arm": arm, "games": int(m.group(1)), "sec_reported": float(m.group(2)),
           "games_per_sec": float(m.group(3)), "wall": round(wall, 3),
           "decisions": int(dec.group(1)) if dec else -1,
           "fingerprint": fp.group(1) if fp else "",
           "infer_fails": int(fail.group(1)) if fail else -1}
    bank = re.search(r"rows/batch=([\d.]+)", out)
    if bank:
        rec["rows_per_batch"] = float(bank.group(1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="build/Release/gomoku.exe")
    ap.add_argument("--model", default="models/gomoku_cnn.fb16.onnx")
    ap.add_argument("--legs", type=int, default=4)
    ap.add_argument("--chains", type=int, default=64)
    ap.add_argument("--games", type=int, default=1280)
    ap.add_argument("--banks", type=int, default=2)
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--ort-dir", default=os.environ.get("FARM_ORT_DIR", ""))
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--cool", type=float, default=3.0, help="腿间冷却秒")
    ap.add_argument("--out", default="bench_data/bench.json")
    args = ap.parse_args()

    legs = []
    for leg in range(args.legs):
        for arm, extra in ARMS:
            rec = run_leg(args.exe, extra + ["--seed", str(args.seed)], args, arm)
            rec["leg"] = leg + 1
            legs.append(rec)
            print("[bench] 腿%d %-5s %8.1f 局/s（决策 %d，指纹 %s，rows/batch %s）"
                  % (rec["leg"], arm, rec["games_per_sec"], rec["decisions"],
                     rec["fingerprint"], rec.get("rows_per_batch", "-")))
            sys.stdout.flush()
            time.sleep(args.cool)

    # 汇总：各臂中位吞吐 + 加速比；指纹跨臂一致校验
    by = {}
    for r in legs:
        by.setdefault(r["arm"], []).append(r)
    med = {a: sorted(x["games_per_sec"] for x in rs)[len(rs) // 2]
           for a, rs in by.items()}
    fps = {a: set(x["fingerprint"] for x in rs) for a, rs in by.items()}
    all_fps = set().union(*fps.values())
    base = med.get("naive", 0.0)
    print("\n[bench] ==== 汇总（%d 腿 × %d 局）====" % (args.legs, args.games))
    for a in ("naive", "fiber", "farm"):
        print("[bench] %-5s 中位 %8.1f 局/s（相对 naive %.2f×；各腿 %s）"
              % (a, med[a], med[a] / base if base else 0,
                 ["%.1f" % x for x in sorted(r["games_per_sec"] for r in by[a])]))
    print("[bench] 指纹：跨臂集合 %s → %s"
          % (sorted(all_fps), "一致 ✓" if len(all_fps) == 1 else "不一致 ✗"))
    decs = set(r["decisions"] for r in legs)
    print("[bench] 决策数：集合 %s → %s"
          % (sorted(decs), "同负载 ✓" if len(decs) == 1 else "不同 ✗"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "legs": legs, "median_gps": med,
                   "fingerprints": sorted(all_fps)}, f, ensure_ascii=False, indent=1)
    print("[bench] → %s" % args.out)


if __name__ == "__main__":
    main()
