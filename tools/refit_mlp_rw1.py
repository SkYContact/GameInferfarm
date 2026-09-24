# -*- coding: utf-8 -*-
"""refit_mlp_rw1.py — 五子棋 MLP 引擎的 RW1 refit blob 导出器（B5 真引擎冒烟）

从 onnx initializer 取权威名单与值（TRT refittable 权重名=onnx 导入后的
initializer 名，实测 fb8 引擎 get_all_weights()=同一名单 4 项），值乘 --scale
生成扰动候选。legacy REFIT 模式引擎现值不可读（getNamedWeights internal
error，refit_blob.py 判据 1 同款）——onnx 侧是唯一权威源。

用法：
  python tools/refit_mlp_rw1.py --onnx models/gomoku_mlp.fb8.onnx \
      --out build/mlp_delta.rw1 --scale 0.5
"""
import argparse
import struct

import numpy as np
import onnx
from onnx import numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="扰动因子：w' = w * scale（1.0=原值照抄）")
    args = ap.parse_args()

    m = onnx.load(args.onnx)
    out = bytearray()
    out += b"RW1\x00"
    out += struct.pack("<II", 1, len(m.graph.initializer))
    for init in m.graph.initializer:
        arr = (numpy_helper.to_array(init) * args.scale).astype(np.float32)
        name = init.name.encode("utf-8")
        out += struct.pack("<H", len(name)) + name
        out += struct.pack("<BI", 2, arr.size)   # 2 = f32
        out += arr.tobytes()
    with open(args.out, "wb") as f:
        f.write(out)
    print("[rw1] %s 项 → %s（scale=%s）" % (len(m.graph.initializer), args.out, args.scale))


if __name__ == "__main__":
    main()
