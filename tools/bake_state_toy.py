#!/usr/bin/env python
# bake_state_toy.py — ③成对状态行（设备池）R9 门的玩具引擎。
# 图：S_next = S_prev + x（跨决策累加——任何池错/粘滞错/清零漏必爆）；
#     policy = ReduceSum(S_next)（适配器指纹掺入 policy 位=逐位观测面）。
# 产出 onnx；trt 烤制走 ~/bake_fb8_trt.py（TF32 关/REFIT 开纪律同门）。
# 用法: trt_venv/bin/python tools/bake_state_toy.py [slots] models/state_toy.fb8.onnx
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto

slots = int(sys.argv[1]) if len(sys.argv) > 1 else 8
dst = sys.argv[2] if len(sys.argv) > 2 else "models/state_toy.fb8.onnx"
w = 4

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [slots, w])
sp = helper.make_tensor_value_info("S_prev", TensorProto.FLOAT, [slots, w])
pol = helper.make_tensor_value_info("policy", TensorProto.FLOAT, [slots, 1])
sn = helper.make_tensor_value_info("S_next", TensorProto.FLOAT, [slots, w])
axes = helper.make_tensor("axes1", TensorProto.INT64, [1], [1])

nodes = [
    helper.make_node("Add", ["S_prev", "x"], ["S_next"]),
    helper.make_node("ReduceSum", ["S_next", "axes1"], ["red"], keepdims=1),
    helper.make_node("Identity", ["red"], ["policy"]),
]
graph = helper.make_graph(nodes, "state_toy", [x, sp], [pol, sn],
                          initializer=[axes])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, dst)
print(f"[bake] {dst} written (slots={slots}: x/S_prev[slots,4] -> policy[slots,1] S_next[slots,4])")
