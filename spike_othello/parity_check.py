# -*- coding: utf-8 -*-
"""尖刺对拍：①ONNX(fb16) vs torch(theta) 逐位对比；②全协议 Python 复算 1280 局
（LCG/种子协议/argmax/指纹公式与 C++ adapter 逐行对齐）对 C++ 腿的
胜率/决策数/指纹。用法：python spike/parity_check.py（在 runs/40_engine_battle 下）"""
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RUN40 = os.path.dirname(HERE)
sys.path.insert(0, RUN40)
from model import init_theta, forward_one  # noqa: E402

CHAINS, GAMES, SEED0 = 64, 1280, 20260922
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
M64 = (1 << 64) - 1


def legal_flips(bd, cell, p):
    if bd[cell]:
        return None
    o = 3 - p
    r0, c0 = cell // 8, cell % 8
    flips = []
    for dr, dc in DIRS:
        r, c = r0 + dr, c0 + dc
        path = []
        while 0 <= r < 8 and 0 <= c < 8 and bd[r * 8 + c] == o:
            path.append(r * 8 + c)
            r += dr
            c += dc
        if path and 0 <= r < 8 and 0 <= c < 8 and bd[r * 8 + c] == p:
            flips += path
    return flips


class Lcg:
    def __init__(self, s):
        self.s = (s | 1) & 0xFFFFFFFF

    def draw(self):
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s >> 8


def play_game(theta, seed, we_first):
    bd = np.zeros(64, dtype=np.uint8)
    bd[28] = bd[35] = 1     # 黑
    bd[27] = bd[36] = 2     # 白
    turn = 0                # 0 黑
    we_black = we_first
    discs = 4
    decisions = 0

    def finish():
        pme = 1 if we_black else 2
        mine = int((bd == pme).sum())
        opp = int((bd == (3 - pme)).sum())
        return (mine > opp and 1) or (opp > mine and 2) or 3

    rng = Lcg(seed)
    while True:
        our_turn = (turn == 0) == we_black
        pme = 1 if we_black else 2
        pop = 3 - pme
        if our_turn:
            mine_moves = [i for i in range(64) if legal_flips(bd, i, pme)]
            if mine_moves:
                # 我方决策：own/opp 平面 → argmax（平局取小格号）
                own = (bd == pme).astype(np.float32)
                opp = (bd == pop).astype(np.float32)
                logits = forward_one(theta, torch.from_numpy(
                    np.concatenate([own, opp])[None])).numpy()[0]
                best = -1
                for i in mine_moves:
                    if best < 0 or logits[i] > logits[best]:
                        best = i
                fl = legal_flips(bd, best, pme)
                bd[best] = pme
                bd[fl] = pme
                discs += 1
                decisions += 1
                turn ^= 1
                if discs >= 64:
                    return finish(), decisions, discs
                continue
            if not any(legal_flips(bd, i, pop) for i in range(64)):
                return finish(), decisions, discs
            turn ^= 1      # 我方 pass
        else:
            opp_moves = [i for i in range(64) if legal_flips(bd, i, pop)]
            if not opp_moves:
                turn ^= 1  # 对手 pass
                continue
            cell = opp_moves[rng.draw() % len(opp_moves)]
            fl = legal_flips(bd, cell, pop)
            bd[cell] = pop
            bd[fl] = pop
            discs += 1
            turn ^= 1
            if discs >= 64:
                return finish(), decisions, discs


def main():
    theta = torch.load(os.path.join(RUN40, "bc_out", "bc_t3k", "theta_final.pt"),
                       map_location="cpu")
    # ① ONNX vs torch 逐位对比
    import onnxruntime as ort
    sess = ort.InferenceSession(
        r"D:\inferfarm\models\othello_bc_mlp.fb16.onnx",
        providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    maxdiff = 0.0
    for _ in range(8):
        own = rng.integers(0, 2, (16, 64)).astype(np.float32)
        opp = rng.integers(0, 2, (16, 64)).astype(np.float32)
        o1 = sess.run(None, {"own": own, "opp": opp})[0]
        o2 = forward_one(theta, torch.from_numpy(
            np.concatenate([own, opp], axis=1))).numpy()
        maxdiff = max(maxdiff, float(np.abs(o1 - o2).max()))
    print("[parity1] ONNX vs torch max|diff| = %.3g %s"
          % (maxdiff, "(逐位同)" if maxdiff == 0 else "(有差异!)"))

    # ② 全协议复算
    per = (GAMES + CHAINS - 1) // CHAINS
    fp = 0
    fw = sw = ft = st = 0
    tot_dec = 0
    for chain in range(CHAINS):
        for game in range(per):
            seed = SEED0 + chain * per + game
            we_first = game % 2 == 0
            winner, dec, discs = play_game(theta, seed, we_first)
            g = (discs - 4) * 1000003 + winner
            mixed = ((g + chain * 0xD1B54A32D192ED03 +
                      game * 0xC2B2AE3D27D4EB4F) * 0x9E3779B97F4A7C15) & M64
            fp ^= mixed
            if we_first:
                ft += 1
                fw += winner == 1
            else:
                st += 1
                sw += winner == 1
            tot_dec += dec
    print("[parity2] 先手 %d/%d 后手 %d/%d 综合 %d/%d (%.1f%%) 决策 %d 指纹 %016x"
          % (fw, ft, sw, st, fw + sw, ft + st, 100.0 * (fw + sw) / (ft + st),
             tot_dec, fp))


if __name__ == "__main__":
    main()
