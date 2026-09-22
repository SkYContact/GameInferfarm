#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gomoku_teacher.py —— 五子棋老师：连型评估 + 一步杀/防 + 自博弈数据生成

老师强度定位：比内置规则对手（能赢就赢/必须堵/8 邻域启发）高一档——
多的是连型评估（活三/冲四/活四/双威胁打分）与防守价值项。它不是完美
棋手；它的职责是给 BC 提供比"随机"强得多、比"规则"稳的着法分布。

产出（--out xxx.npz）：
  X_own (N,225) f32 / X_opp (N,225) f32 / y (N,) i64 / 局数与终局统计
  视角=当前行动方（own=行动方棋子平面）——与 inferfarm 适配器口径一致。

用法（q35 环境）：
  python tools/gomoku_teacher.py --games 3000 --seed 20260922 \
      --out bench_data/gomoku_bc.npz --procs 8
冒烟：--games 100 --eval-rule（老师 vs 复刻的内置规则对手 200 局胜率）
"""
import argparse
import os
import time

import numpy as np

SIZE = 15
CELLS = SIZE * SIZE
DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]

# 连型分：len=连长, open=两端空数（落子后口径）
SCORE = {
    (5, 2): 10_000_000_000, (5, 1): 10_000_000_000, (5, 0): 10_000_000_000,
    (4, 2): 1_000_000, (4, 1): 100_000, (4, 0): 0,
    (3, 2): 10_000, (3, 1): 1_000, (3, 0): 0,
    (2, 2): 100, (2, 1): 10, (2, 0): 0,
    (1, 2): 1, (1, 1): 1, (1, 0): 0,
}


def line_len(bd, r, c, p, dr, dc):
    """以 (r,c) 为中心（已放 p）向两侧延伸的连长与两端开放数"""
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
    for dr, dc in DIRS:
        ln, _ = line_len(bd, r, c, p, dr, dc)
        if ln >= 5:
            return True
    return False


def candidates(bd):
    """非空邻域 2 格内的空点（棋盘空时=天元）"""
    if bd.sum() == 0:
        return [7 * SIZE + 7]
    out = []
    idx = np.argwhere(bd != 0)
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    for r, c in idx:
        r0, r1 = max(0, r - 2), min(SIZE, r + 3)
        c0, c1 = max(0, c - 2), min(SIZE, c + 3)
        mask[r0:r1, c0:c1] = True
    mask &= (bd == 0)
    for r, c in np.argwhere(mask):
        out.append(int(r) * SIZE + int(c))
    return out


def attack_score(bd, cell, p):
    """p 方在 cell 落子的连型攻击分（四方向求和；成五单列大分）"""
    r, c = divmod(cell, SIZE)
    s = 0
    best = 0
    for dr, dc in DIRS:
        ln, op = line_len(bd, r, c, p, dr, dc)
        if ln >= 5:
            return SCORE[(5, 2)]
        v = SCORE.get((ln, op), 0)
        s += v
        best = max(best, v)
    return best * 2 + s  # 主方向加倍：突出双威胁点


def teacher_move(bd, me, rng):
    """老师着法：一步杀 → 堵一步杀 → 攻防综合 argmax"""
    cand = candidates(bd)
    # ① 我方一步取胜
    for cell in cand:
        r, c = divmod(cell, SIZE)
        if makes_five(bd, r, c, me):
            return cell
    opp = 3 - me
    # ② 堵对方一步取胜（多个时取攻击分最高的堵点）
    blocks = []
    for cell in cand:
        r, c = divmod(cell, SIZE)
        if makes_five(bd, r, c, opp):
            blocks.append(cell)
    if blocks:
        return max(blocks, key=lambda x: attack_score(bd, x, me))
    # ③ 攻防综合：我的攻击分 + 0.85 × 对方在此点的攻击分（防守价值）
    best, best_s = cand[0], -1.0
    for cell in cand:
        s = attack_score(bd, cell, me) + 0.85 * attack_score(bd, cell, opp)
        if s > best_s:
            best_s, best = s, cell
    return best


def rule_move(bd, me, rng):
    """复刻内置规则对手（胜率冒烟自测用）：能赢→必堵→8邻域×3+中心+抖动"""
    cand = candidates(bd)
    for cell in cand:
        r, c = divmod(cell, SIZE)
        if makes_five(bd, r, c, me):
            return cell
    opp = 3 - me
    for cell in cand:
        r, c = divmod(cell, SIZE)
        if makes_five(bd, r, c, opp):
            return cell
    best, best_s = cand[0], -1
    for cell in cand:
        r, c = divmod(cell, SIZE)
        adj = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr or dc) and 0 <= rr < SIZE and 0 <= cc < SIZE and bd[rr, cc]:
                    adj += 1
        ctr = min(r, SIZE - 1 - r, c, SIZE - 1 - c)
        s = adj * 3 + ctr + rng.integers(0, 5)
        if s > best_s:
            best_s, best = s, cell
    return best


def play_game(seed, open_moves=2):
    """老师自博弈一局，返回样本列表 [(own,opp,move),...] 与终局"""
    rng = np.random.default_rng(seed)
    bd = np.zeros((SIZE, SIZE), dtype=np.int8)
    samples = []
    winner = 0
    move_count = 0
    while True:
        me = 1 if move_count % 2 == 0 else 2
        cand_ok = bd.sum() > 0
        if move_count < open_moves:
            # 开局随机（中心 5×5）保证数据多样性
            while True:
                r, c = rng.integers(5, 10, 2)
                if bd[r, c] == 0:
                    cell = int(r) * SIZE + int(c)
                    break
        else:
            cell = teacher_move(bd, me, rng)
        r, c = divmod(cell, SIZE)
        own = (bd == me).astype(np.float32).reshape(-1)
        opp = (bd == (3 - me)).astype(np.float32).reshape(-1)
        samples.append((own, opp, cell))
        bd[r, c] = me
        move_count += 1
        if makes_five(bd, r, c, me):
            winner = me
            break
        if bd.sum() >= CELLS:
            winner = 3
            break
    return samples, winner, move_count


def gen_batch(args):
    seed0, games = args
    all_own, all_opp, all_y = [], [], []
    results = {1: 0, 2: 0, 3: 0}
    for i in range(games):
        samples, winner, _ = play_game(seed0 + i)
        results[winner] += 1
        for own, opp, y in samples:
            all_own.append(own)
            all_opp.append(opp)
            all_y.append(y)
    return (np.stack(all_own), np.stack(all_opp), np.array(all_y, dtype=np.int64),
            results)


def eval_rule(games, seed):
    """老师 vs 规则对手（老师执双先后手）——胜率冒烟"""
    res = {"w": 0, "l": 0, "d": 0}
    for g in range(games):
        rng = np.random.default_rng(seed + g)
        bd = np.zeros((SIZE, SIZE), dtype=np.int8)
        teacher_first = g % 2 == 0
        winner = 0
        mc = 0
        while True:
            me = 1 if mc % 2 == 0 else 2
            is_teacher = (me == 1) == teacher_first
            if mc == 0:
                cell = 7 * SIZE + 7
            else:
                cell = teacher_move(bd, me, rng) if is_teacher else rule_move(bd, me, rng)
            r, c = divmod(cell, SIZE)
            bd[r, c] = me
            mc += 1
            if makes_five(bd, r, c, me):
                winner = me
                break
            if bd.sum() >= CELLS:
                winner = 3
                break
        tw = (winner == 1) == teacher_first if winner in (1, 2) else False
        if winner == 3:
            res["d"] += 1
        elif tw:
            res["w"] += 1
        else:
            res["l"] += 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", default="bench_data/gomoku_bc.npz")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--eval-rule", action="store_true", help="只跑老师vs规则冒烟")
    ap.add_argument("--eval-games", type=int, default=200)
    args = ap.parse_args()

    if args.eval_rule:
        t0 = time.time()
        res = eval_rule(args.eval_games, args.seed)
        n = sum(res.values())
        print("[teacher-vs-rule] %d 局: 胜 %d / 负 %d / 平 %d（胜率 %.1f%%，%.2fs）"
              % (n, res["w"], res["l"], res["d"], res["w"] * 100.0 / n,
                 time.time() - t0))
        return

    import multiprocessing as mp
    t0 = time.time()
    per = args.games // args.procs
    chunks = [(args.seed + k * 1_000_000, per) for k in range(args.procs)]
    with mp.Pool(args.procs) as pool:
        results = pool.map(gen_batch, chunks)
    owns = np.concatenate([r[0] for r in results])
    opps = np.concatenate([r[1] for r in results])
    ys = np.concatenate([r[2] for r in results])
    tot = {1: 0, 2: 0, 3: 0}
    for r in results:
        for k, v in r[3].items():
            tot[k] += v
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, X_own=owns, X_opp=opps, y=ys,
                        games=args.games, seed=args.seed)
    print("[teacher] %d 局 → %d 样本（先手胜 %d / 后手胜 %d / 平 %d）；%.1fs → %s"
          % (args.games, len(ys), tot[1], tot[2], tot[3], time.time() - t0,
             args.out))


if __name__ == "__main__":
    main()
