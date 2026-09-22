#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bake_gomoku_mlp.py —— 五子棋范例的一层 MLP 模型工件烤制（inferfarm）

一层 MLP（未训练，权重由种子生成=逐位确定）：
    own[*,225] + opp[*,225] → concat → Linear(450,H) → ReLU → Linear(H,225) → policy
行独立（无 batchnorm）——满足推理农场 GameAdapter 契约 2。

工件：
  1. onnx：批维钉死 fb（银行 slots），输入名 own/opp，输出名 policy——ORT 后端用
  2. trt engine（--trt）：从该 onnx 构建，IO fp32，TF32 关（与运行端纪律一致）

用法（q35 环境）：
  python tools/bake_gomoku_mlp.py --slots 8 --hidden 64 \
      --out models/gomoku_mlp.fb8.onnx --trt models/gomoku_mlp.fb8.trt

坑（血律）：引擎/模型烤制须 GPU 空载时进行；NVIDIA_TF32_OVERRIDE 必须为 0
（脚本自设——运行端 trt_backend 同纪律，不一致=拒建 context）。
"""
import argparse
import hashlib
import os
import sys

# TF32 纪律：须在 torch/trt 任何 CUDA 面初始化前设
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import torch
import torch.nn as nn

BOARD = 225  # 15×15


class GomokuMlp(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(2 * BOARD, hidden)
        self.fc2 = nn.Linear(hidden, BOARD)

    def forward(self, own, opp):
        x = torch.cat([own, opp], dim=1)
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=8, help="批维（=银行 slots）")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", required=True, help="输出 onnx 路径")
    ap.add_argument("--trt", default=None, help="可选：同时输出 TRT engine 路径")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = GomokuMlp(args.hidden).eval()
    own = torch.zeros(args.slots, BOARD)
    opp = torch.zeros(args.slots, BOARD)
    with torch.no_grad():
        ref = model(own, opp)
    wsum = hashlib.sha256()
    for p in model.state_dict().values():
        wsum.update(p.detach().cpu().numpy().tobytes())
    print("[bake] seed=%d hidden=%d slots=%d 权重sha256=%s 参考输出[0,:3]=%s"
          % (args.seed, args.hidden, args.slots, wsum.hexdigest()[:16],
             ref[0, :3].tolist()))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # 经典导出器（torch 2.x：dynamo=False 保形状钉死行为）；失败回退默认路径
    try:
        torch.onnx.export(
            model, (own, opp), args.out,
            input_names=["own", "opp"], output_names=["policy"],
            opset_version=17, dynamo=False)
    except TypeError:
        torch.onnx.export(
            model, (own, opp), args.out,
            input_names=["own", "opp"], output_names=["policy"],
            opset_version=17)
    print("[bake] onnx → %s" % args.out)

    # 形状自检（钉死校验）
    import onnx
    m = onnx.load(args.out)
    dims = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
            for i in m.graph.input}
    for name, d in dims.items():
        assert d[0] == args.slots, "批维未钉死: %s=%s" % (name, d)
    print("[bake] 形状自检:", dims)

    if args.trt:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        # TRT 10：显式批恒开（EXPLICIT_BATCH 旗标已移除）
        flags = 0
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        with open(args.out, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print("[trt-parse] %s" % parser.get_error(i))
                raise SystemExit("ONNX 解析失败")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise SystemExit("engine 构建失败")
        with open(args.trt, "wb") as f:
            f.write(plan)
        print("[bake] trt engine → %s（%.1f MB）"
              % (args.trt, os.path.getsize(args.trt) / 1e6))


if __name__ == "__main__":
    main()
