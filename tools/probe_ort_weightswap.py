# probe_ort_weightswap.py — ORT 权重外置热换可行性探针（2026-09-29 YGO 工单）
# 判据：P1 可覆写初始化器存活且喂值生效；P2 同输入复跑逐位稳；P3 换回幂等
#      （G5 语义）；P4 裸指针绑定+原地改缓冲→下个 Run 读到新值（热换核心语义）
#      P5 图优化级别敏感性（防折叠）
import numpy as np, onnx, onnx.helper as oh, onnxruntime as ort

def build(path):
    din, dh, dout, N = 16, 32, 4, 8
    rs = np.random.RandomState(7)
    w1 = rs.randn(din, dh).astype(np.float32) * 0.3
    b1 = rs.randn(dh).astype(np.float32) * 0.1
    w2 = rs.randn(dh, dout).astype(np.float32) * 0.3
    inits = [onnx.numpy_helper.from_array(w1, "w1"),
             onnx.numpy_helper.from_array(b1, "b1"),
             onnx.numpy_helper.from_array(w2, "w2")]
    nodes = [oh.make_node("MatMul", ["x", "w1"], ["h"]),
             oh.make_node("Add", ["h", "b1"], ["hb"]),
             oh.make_node("Relu", ["hb"], ["hr"]),
             oh.make_node("MatMul", ["hr", "w2"], ["y"])]
    ins = [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [N, din]),
           oh.make_tensor_value_info("w1", onnx.TensorProto.FLOAT, [din, dh]),
           oh.make_tensor_value_info("b1", onnx.TensorProto.FLOAT, [dh]),
           oh.make_tensor_value_info("w2", onnx.TensorProto.FLOAT, [dh, dout])]
    outs = [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [N, dout])]
    graph = oh.make_graph(nodes, "swap", ins, outs, inits)
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)

def run_probe(opt_level, tag):
    so = ort.SessionOptions()
    so.graph_optimization_level = opt_level
    sess = ort.InferenceSession("/tmp/ortswap/mlp_swap.onnx", so,
                                providers=["CPUExecutionProvider"])
    x = np.random.RandomState(9).randn(8, 16).astype(np.float32)
    w1 = np.random.RandomState(7).randn(16, 32).astype(np.float32) * 0.3
    b1 = np.zeros(32, np.float32)
    w2 = np.random.RandomState(2).randn(32, 4).astype(np.float32) * 0.3
    io = ort.IOBinding(sess)
    # P4 核心：裸指针绑定我们自己的 numpy 缓冲（IOBinding 语义=每 Run 直读指针）
    io.bind_input("x", "cpu", 0, np.float32, x.shape, x.ctypes.data)
    io.bind_input("w1", "cpu", 0, np.float32, w1.shape, w1.ctypes.data)
    io.bind_input("b1", "cpu", 0, np.float32, b1.shape, b1.ctypes.data)
    io.bind_input("w2", "cpu", 0, np.float32, w2.shape, w2.ctypes.data)
    out_buf = np.zeros((8, 4), np.float32)
    io.bind_output("y", "cpu", 0, np.float32, out_buf.shape, out_buf.ctypes.data)
    sess.run_with_iobinding(io, None); o_base = out_buf.copy()
    # P2 复跑稳
    sess.run_with_iobinding(io, None); o_rerun = out_buf.copy()
    p2 = np.array_equal(o_base, o_rerun)
    # 换权重（原地写我们自己的缓冲——这就是"热换"本体）
    w1_new = np.random.RandomState(99).randn(16, 32).astype(np.float32) * 0.3
    np.copyto(w1, w1_new)
    sess.run_with_iobinding(io, None); o_swapped = out_buf.copy()
    p_swap = not np.array_equal(o_base, o_swapped)
    # P3 换回幂等
    np.copyto(w1, np.random.RandomState(7).randn(16, 32).astype(np.float32) * 0.3)
    sess.run_with_iobinding(io, None); o_back = out_buf.copy()
    p3 = np.array_equal(o_base, o_back)
    # 交叉验证：喂同样权重，ORT 重建的"常规可覆写喂值会话"输出应同世界
    sess2 = ort.InferenceSession("/tmp/ortswap/mlp_swap.onnx", so,
                                 providers=["CPUExecutionProvider"])
    o_fresh = sess2.run(None, {"x": x, "w1": w1, "b1": b1, "w2": w2})[0]
    p_world = np.array_equal(o_back, o_fresh)
    print(f"[{tag}] P2复跑稳={p2} P3换回幂等={p3} 热换生效={p_swap} 世界==重建会话={p_world}")
    return p2 and p3 and p_swap

build("/tmp/ortswap/mlp_swap.onnx")
ok_all = True
for lv, tag in [(ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "ALL"),
                (ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, "BASIC"),
                (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "DISABLE")]:
    ok_all &= run_probe(lv, tag)
print("VERDICT:", "PASS 语义前提全立" if ok_all else "FAIL 见上")
