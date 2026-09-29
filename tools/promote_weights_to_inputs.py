#!/usr/bin/env python3
# promote_weights_to_inputs.py — 权重升格出口（ORT 权重热换通道的烤制侧，
# 2026-09-29；设计=docs/reply-ygo-ort-hotswap.md §2）。
#
# 把 ONNX 模型的 float initializers 升格为可覆写图输入（initializer 兼
# graph.input = ORT 官方 "overridable initializer"；ORT 仅 Warning"禁 const
# folding"不阻断）。升格后框架侧自动检测（零配置），RefitWeights(RW1) 即换心。
#
# 同时产出 RW1 边车（<out>.rw1）：as-baked 权重值，供首次换心/种子
#（ModelConfig.refit_weights 指向它，或首腿前显式 RefitWeights——未换心发车
# =框架拒批，零权重是垃圾）。RW1 格式见 include/inferfarm/refit.h。
#
# 用法：python promote_weights_to_inputs.py <in.onnx> <out.onnx> [--names n1,n2]
#   缺省=全部 float initializer；--names 显式点名（须全为 float）。
import sys
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def rw1_dtype(np_dtype):
    if np_dtype == np.float32:
        return 2
    raise SystemExit(f"RW1 仅收 f32（IO 面协议），{np_dtype} 不支持")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    names = None
    if "--names" in sys.argv:
        names = set(sys.argv[sys.argv.index("--names") + 1].split(","))

    m = onnx.load(inp)
    g = m.graph
    in_names = {i.name for i in g.input}
    picked = []
    for init in g.initializer:
        if init.name in in_names:
            continue  # 已升格（重复运行幂等）
        if names is not None and init.name not in names:
            continue
        if init.data_type != TensorProto.FLOAT:
            if names is not None and init.name in names:
                raise SystemExit(f"点名 {init.name} 非 float（data_type={init.data_type}）")
            continue  # 缺省只升 float（权重/偏置；int64 形状常量等不动）
        picked.append(init)
    if names is not None:
        missing = names - {p.name for p in picked}
        if missing:
            raise SystemExit(f"点名未命中任何 initializer: {sorted(missing)}")
    if not picked:
        raise SystemExit("没有可升格的 float initializer（全升过了？）")

    total = 0
    for init in picked:
        arr = numpy_helper.to_array(init)
        g.input.append(onnx.helper.make_tensor_value_info(
            init.name, TensorProto.FLOAT, list(arr.shape)))
        total += arr.nbytes
    onnx.save(m, outp)

    # RW1 边车（as-baked 值；名字=升格图输入名=state_dict 键纪律）
    blob = bytearray(b"RW1\x00")
    blob += (1).to_bytes(4, "little")
    blob += len(picked).to_bytes(4, "little")
    for init in picked:
        arr = np.ascontiguousarray(numpy_helper.to_array(init))
        nb = init.name.encode()
        blob += len(nb).to_bytes(2, "little") + nb
        blob += bytes([rw1_dtype(arr.dtype)])
        blob += np.uint32(arr.size).tobytes()
        blob += arr.tobytes()
    side = outp + ".rw1"
    with open(side, "wb") as f:
        f.write(blob)

    print(f"升格 {len(picked)} 项 {total/1e6:.1f}MB → {outp}")
    print(f"RW1 边车（as-baked 种子）→ {side}")
    for init in picked:
        print(f"  {init.name} {list(numpy_helper.to_array(init).shape)}")


if __name__ == "__main__":
    main()
