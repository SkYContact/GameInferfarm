#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_gomoku_python.py —— python 实现谱系基线（与 C++ 三臂同负载对照）

臂（同一 torch ckpt、同一 GPU、同一对局协议=模型 vs 规则对手）：
  serial  串行单局，每决策一次 batch=1 GPU 推理（"最朴素"）
  vec     同步向量化 N 环境：所有环境推到决策点 → 攒 batch=N 一次推理
          （python 生态标准姿势：gymnasium VectorEnv / EnvPool sync 模式语义）
输出局/s + 决策/s 双口径（局/s 受对局长度影响，决策/s 是硬件口径）。

用法（q35 环境）：
  python tools/bench_gomoku_python.py --ckpt bench_data/gomoku_cnn.pt \
      --games 1280 --envs 64 --out bench_data/bench_python.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
from train_gomoku_policy import GomokuCNN  # noqa: E402

SIZE, CELLS = 15, 225
DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]


# ---- 对局逻辑（C++ gomoku_adapter 语义的 python 复刻）----
def line_len(bd, r, c, p, dr, dc):
    n = 1
    for i in range(1, 5):
        rr, cc = r + dr * i, c + dc * i
        if rr < 0 or rr >= SIZE or cc < 0 or cc >= SIZE or bd[rr, cc] != p:
            break
        n += 1
    open_a = 1
    rr, cc = r + dr * n, c + dc * n
    if rr < 0 or rr >= SIZE or cc < 0 or cc >= SIZE or bd[rr, cc] != 0:
        open_a = 0
    m = 1
    for i in range(1, 5):
        rr, cc = r - dr * i, c - dc * i
        if rr < 0 or rr >= SIZE or cc < 0 or cc >= SIZE or bd[rr, cc] != p:
            break
        m += 1
    open_b = 1
    rr, cc = r - dr * m, c - dc * m
    if rr < 0 or rr >= SIZE or cc < 0 or cc >= SIZE or bd[rr, cc] != 0:
        open_b = 0
    return n + m - 1, open_a + open_b


def makes_five(bd, r, c, p):
    return any(line_len(bd, r, c, p, d[0], d[1])[0] >= 5 for d in DIRS)


class PyGomoku:
    """一局：我方=模型 argmax（合法过滤，平局取小格号）；对手=C++ 内置规则
    对手复刻（能赢就赢/必堵/8邻域×3+中心+LCG 抖动=同种子同着法）。"""

    def __init__(self, seed, we_first):
        self.bd = np.zeros((SIZE, SIZE), dtype=np.int8)
        self.rng = seed | 1
        self.moves = 0
        self.over = False
        self.winner = 0
        self.we_first = we_first
        self.decisions = 0
        if not we_first:                     # 对手先攻：天元
            self._place(7 * SIZE + 7, 2)

    def _lcg(self):
        self.rng = (self.rng * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.rng >> 8

    def _place(self, cell, p):
        r, c = divmod(cell, SIZE)
        self.bd[r, c] = p
        self.moves += 1
        if makes_five(self.bd, r, c, p):
            self.over, self.winner = True, p
        elif self.moves >= CELLS:
            self.over, self.winner = True, 3

    def _opp(self):
        empt = np.argwhere(self.bd == 0)
        if len(empt) == 0:
            self.over, self.winner = True, 3
            return
        for r, c in empt:                    # ① 对手一步取胜
            if makes_five(self.bd, r, c, 2):
                self._place(r * SIZE + c, 2)
                return
        for r, c in empt:                    # ② 堵我方一步取胜
            if makes_five(self.bd, r, c, 1):
                self._place(r * SIZE + c, 2)
                return
        best, bs = -1, -1                    # ③ 邻域×3+中心+抖动
        for r, c in empt:
            adj = int(np.sum(self.bd[max(0, r-1):r+2, max(0, c-1):c+2] != 0))
            ctr = min(r, SIZE - 1 - r, c, SIZE - 1 - c)
            s = adj * 3 + ctr + self._lcg() % 5
            if s > bs:
                bs, best = s, int(r) * SIZE + int(c)
        self._place(best, 2)

    def features(self):
        return ((self.bd == 1).astype(np.float32).reshape(-1),
                (self.bd == 2).astype(np.float32).reshape(-1))

    def advance(self):
        """推到我方决策点；终局返回 False（每决策=对手先走一步再轮我）"""
        while True:
            if self.over:
                return False
            if self.moves == 0:
                return True
            self._opp()
            if self.over:
                return False
            return True

    def apply(self, move):
        self.decisions += 1
        self._place(move, 1)

    def outcome(self):
        return 1 if self.winner == 1 else 0


def bench_serial(model, games, seed0, dev):
    t0 = time.time()
    done, dec, wins = 0, 0, 0
    for i in range(games):
        g = PyGomoku(seed0 + i, i % 2 == 0)
        while g.advance():
            own, opp = g.features()
            with torch.no_grad():
                logit = model(torch.from_numpy(own).unsqueeze(0).to(dev),
                              torch.from_numpy(opp).unsqueeze(0).to(dev))
            pol = logit[0].cpu().numpy()
            pol[g.bd.reshape(-1) != 0] = -1e30        # 合法过滤
            g.apply(int(pol.argmax()))
        done += 1
        dec += g.decisions
        wins += g.outcome()
    return {"games": done, "sec": time.time() - t0, "decisions": dec, "wins": wins}


def bench_vec(model, games, nenv, seed0, dev):
    t0 = time.time()
    envs = [None] * nenv
    ptr = [0] * nenv
    idx = list(range(nenv))
    queue = list(range(nenv))               # 待开局的槽
    game_i = [0] * nenv
    done, dec, wins = 0, 0, 0
    active = {}
    while done < games:
        # ① 所有空槽开新局，推到首个决策点
        for s in list(queue):
            gi = game_i[s]
            if gi >= games:
                queue.remove(s)
                continue
            g = PyGomoku(seed0 + gi, gi % 2 == 0)
            game_i[s] += 1
            if g.advance():
                active[s] = g
            else:
                done += 1
                dec += g.decisions
                wins += g.outcome()
            if s in queue:
                queue.remove(s)
        if not active:
            break
        # ② 攒批：所有活跃环境的特征 → 一次 GPU 推理
        own = np.stack([active[s].features()[0] for s in active])
        opp = np.stack([active[s].features()[1] for s in active])
        with torch.no_grad():
            logit = model(torch.from_numpy(own).to(dev),
                          torch.from_numpy(opp).to(dev))
        pol = logit.cpu().numpy()
        # ③ 回投：合法过滤 argmax，推进一步（含对手），回决策点或收局
        for j, s in enumerate(list(active.keys())):
            g = active[s]
            p = pol[j].copy()
            p[g.bd.reshape(-1) != 0] = -1e30
            g.apply(int(p.argmax()))
            if g.advance():
                continue
            done += 1
            dec += g.decisions
            wins += g.outcome()
            del active[s]
            queue.append(s)
    return {"games": done, "sec": time.time() - t0, "decisions": dec, "wins": wins}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="bench_data/gomoku_cnn.pt")
    ap.add_argument("--games", type=int, default=1280)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--arms", default="serial,vec")
    ap.add_argument("--out", default="bench_data/bench_python.json")
    args = ap.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = GomokuCNN().to(dev).eval()
    model.load_state_dict(ck["state_dict"])
    torch.cuda.synchronize()

    res = {}
    for arm in args.arms.split(","):
        if arm == "serial":
            r = bench_serial(model, args.games, args.seed, dev)
        elif arm == "vec":
            r = bench_vec(model, args.games, args.envs, args.seed, dev)
        else:
            raise SystemExit("未知臂 %s" % arm)
        r["games_per_sec"] = round(r["games"] / r["sec"], 2)
        r["decisions_per_sec"] = round(r["decisions"] / r["sec"], 1)
        r["win_rate"] = round(r["wins"] / max(1, r["games"]), 4)
        res[arm] = r
        print("[py-bench] %-6s %5d 局 / %6.1fs = %8.2f 局/s（%.1f 决策/s，胜率 %.1f%%）"
              % (arm, r["games"], r["sec"], r["games_per_sec"],
                 r["decisions_per_sec"], r["win_rate"] * 100))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": res}, f, ensure_ascii=False, indent=1)
    print("[py-bench] → %s" % args.out)


if __name__ == "__main__":
    main()
