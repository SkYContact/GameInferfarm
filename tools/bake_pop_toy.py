#!/usr/bin/env python
# bake_pop_toy.py — population 路由（演化，判决16）TRT 覆盖门的玩具引擎。
# 路由图（v2.2 口径最小版）：每行权重按 mid[r] 从 pop 平面取行——
#   y[r] = reshape(pop[mid[r]], [O, F]) @ x[r]
# mid 恒 0 + 均匀 pop ⇒ 全场同权重（路由不扰动行数学的对照根基）；
# mid[r]=r%P + 异权重 pop ⇒ 路由生效（换代 fp 必变）。
# 产出 onnx；trt 烤制走 ~/bake_fb8_trt.py（TF32 关/REFIT 开纪律同门）。
# 用法: trt_venv/bin/python tools/bake_pop_toy.py [slots] models/pop_toy.fb8.onnx
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto

slots = int(sys.argv[1]) if len(sys.argv) > 1 else 8
dst = sys.argv[2] if len(sys.argv) > 2 else "models/pop_toy.fb8.onnx"
P, F, O = 4, 6, 4

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [slots, F])
mid = helper.make_tensor_value_info("mid", TensorProto.INT64, [slots])
pop = helper.make_tensor_value_info("pop", TensorProto.FLOAT, [P, O * F])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [slots, O])

nodes = [
    helper.make_node("Gather", ["pop", "mid"], ["Wsel"], axis=0),   # [slots, O*F]
    helper.make_node("Reshape", ["Wsel", "c3"], ["W3"]),            # [slots, O, F]
    helper.make_node("Reshape", ["x", "c4"], ["x3"]),               # [slots, F, 1]
    helper.make_node("MatMul", ["W3", "x3"], ["y3"]),               # [slots, O, 1]
    helper.make_node("Reshape", ["y3", "c5"], ["y"]),               # [slots, O]
]
graph = helper.make_graph(nodes, "pop_toy", [x, mid, pop], [y],
                          initializer=[
                              onnx.numpy_helper.from_array(
                                  np.array([slots, O, F], dtype=np.int64), "c3"),
                              onnx.numpy_helper.from_array(
                                  np.array([slots, F, 1], dtype=np.int64), "c4"),
                              onnx.numpy_helper.from_array(
                                  np.array([slots, O], dtype=np.int64), "c5")])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, dst)
print(f"[bake] {dst} written (slots={slots}, P={P}, F={F}, O={O}: "
      f"x+mid+pop -> y[slots,{O}])")
