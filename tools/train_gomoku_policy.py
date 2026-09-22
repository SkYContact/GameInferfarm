#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_gomoku_policy.py —— BC 训练五子棋策略 CNN + 导出 inferfarm 工件

数据：tools/gomoku_teacher.py 产出的 npz（own/opp 两平面 + 老师着法）。
模型：纯 Conv+ReLU+Linear（**无 batchnorm**——inferfarm 行独立契约：银行
不满整批照发、尾行旧数据无害）。尺寸对齐"GPU 有意义"的档位（~6.9M 参数，
~83M MACs/决策）：CPU 单样本几十 ms 出局，GPU 攒批摊门票才是主场。

工件（--export 时）：
  钉批 onnx：输入 own[slots,225]/opp[slots,225]，输出 policy[slots,225]
  （批维钉死=银行 slots；与 bake_gomoku_mlp.py 同约定，适配器零改动）

用法（q35 环境，GPU）：
  python tools/train_gomoku_policy.py --data bench_data/gomoku_bc.npz \
      --epochs 3 --export --slots 16 --out models/gomoku_cnn.fb16.onnx
"""
import argparse
import os

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")   # TF32 纪律：烤制/导出前设

import numpy as np
import torch
import torch.nn as nn

BOARD = 225


class GomokuCNN(nn.Module):
    """own/opp [N,225] → reshape 15×15 → concat 2 通道 → 4 层 conv → policy"""

    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch * 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(ch * 2 * BOARD, BOARD),
        )

    def forward(self, own, opp):
        x = torch.stack([own, opp], dim=1).reshape(-1, 2, 15, 15)
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench_data/gomoku_bc.npz")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--augment", type=int, default=1, help="8 对称增强（先切分后增强，防泄漏）")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--out", default="models/gomoku_cnn.fb16.onnx")
    ap.add_argument("--ckpt", default="bench_data/gomoku_cnn.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[train] device=%s" % dev)

    d = np.load(args.data)
    X_own, X_opp, y = d["X_own"], d["X_opp"], d["y"]
    n_raw = len(y)
    # 先切分后增强（对称副本进 val=泄漏，指标假高）
    idx = np.random.default_rng(args.seed).permutation(n_raw)
    n_val = int(n_raw * args.val_frac)
    vi, ti = idx[:n_val], idx[n_val:]
    # 盘面 8 对称增强（4 旋转 × 2 镜像）：BC 泛化标配，棋规不变性
    def sym_batch(idx_sel):
        G = X_own[idx_sel].reshape(-1, 15, 15)
        H = X_opp[idx_sel].reshape(-1, 15, 15)
        M = np.zeros((len(idx_sel), 15, 15), dtype=np.float32)
        M[np.arange(len(idx_sel)), y[idx_sel] // 15, y[idx_sel] % 15] = 1.0
        outs = []
        for k in range(4):
            for flip in (False, True):
                g, h, m = np.rot90(G, k), np.rot90(H, k), np.rot90(M, k)
                if flip:
                    g, h, m = g[:, :, ::-1], h[:, :, ::-1], m[:, :, ::-1]
                outs.append((np.ascontiguousarray(g).reshape(-1, BOARD),
                             np.ascontiguousarray(h).reshape(-1, BOARD),
                             np.ascontiguousarray(m).reshape(-1, BOARD).argmax(1)))
        return outs
    if args.augment:
        trios = sym_batch(ti)
        own_t_np = np.concatenate([t[0] for t in trios])
        opp_t_np = np.concatenate([t[1] for t in trios])
        y_t_np = np.concatenate([t[2] for t in trios])
        print("[train] 8 对称增强: %d → %d 训练样本（val %d 保持原始分布）"
              % (len(ti), len(y_t_np), len(vi)))
    else:
        own_t_np, opp_t_np, y_t_np = X_own[ti], X_opp[ti], y[ti]
    n = n_raw
    own_t = torch.from_numpy(own_t_np).to(dev)
    opp_t = torch.from_numpy(opp_t_np).to(dev)
    y_t = torch.from_numpy(y_t_np).to(dev)
    print("[train] %d 原始样本 → train %d / val %d" % (n, len(y_t), len(vi)))

    model = GomokuCNN().to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print("[train] 参数量 %.2fM" % (nparam / 1e6))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * ((len(y_t) + args.batch - 1) // args.batch))
    lossf = nn.CrossEntropyLoss()

    step = 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(y_t), device=dev)
        tot_loss, tot_hit, nb = 0.0, 0, 0
        for b0 in range(0, len(y_t), args.batch):
            b = perm[b0:b0 + args.batch]
            logit = model(own_t[b], opp_t[b])
            loss = lossf(logit, y_t[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(b)
            tot_hit += (logit.argmax(1) == y_t[b]).sum().item()
            nb += len(b)
            step += 1
            if step % 200 == 0:
                print("[train] ep%d step%d loss %.4f acc %.3f"
                      % (ep + 1, step, tot_loss / nb, tot_hit / nb))
        model.eval()
        with torch.no_grad():
            hit, nv = 0, 0
            for b0 in range(0, len(vi), 2048):
                b = slice(b0, b0 + 2048)
                logit = model(torch.from_numpy(X_own[vi[b]]).to(dev),
                              torch.from_numpy(X_opp[vi[b]]).to(dev))
                hit += (logit.argmax(1).cpu() == torch.from_numpy(y[vi[b]])).sum().item()
                nv += len(vi[b0:b0 + 2048])
            print("[train] ep%d val top-1（老师着法命中）= %.4f" % (ep + 1, hit / nv))

    os.makedirs(os.path.dirname(os.path.abspath(args.ckpt)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, args.ckpt)
    print("[train] checkpoint → %s" % args.ckpt)

    if args.export:
        model = model.cpu()   # 导出在 CPU 面（权重在 cuda 时 dummy 输入跨设备）
        model.eval()
        own = torch.zeros(args.slots, BOARD)
        opp = torch.zeros(args.slots, BOARD)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        try:
            torch.onnx.export(model, (own, opp), args.out,
                              input_names=["own", "opp"], output_names=["policy"],
                              opset_version=17, dynamo=False)
        except TypeError:
            torch.onnx.export(model, (own, opp), args.out,
                              input_names=["own", "opp"], output_names=["policy"],
                              opset_version=17)
        import onnx
        m = onnx.load(args.out)
        dims = {i.name: [dd.dim_value for dd in i.type.tensor_type.shape.dim]
                for i in m.graph.input}
        for name, dd in dims.items():
            assert dd[0] == args.slots, "批维未钉死: %s=%s" % (name, dd)
        print("[export] onnx → %s（%.1f MB）形状 %s"
              % (args.out, os.path.getsize(args.out) / 1e6, dims))


if __name__ == "__main__":
    main()
