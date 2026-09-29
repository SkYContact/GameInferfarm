# probe_ort_weightswap_cuda.py — CUDA EP 设备驻留热换+计时（YGO 工单判决探针 GPU 段）
# 前置：锁内跑；环境=farm_env + trt_venv python（onnxruntime 走 farm_pkg）。
# 语义前提（CPU EP 版 probe_ort_weightswap.py 已证）：可覆写初始化器+裸指针
# 绑定+原地改+跨 Run 常驻+换回幂等。本段补：设备缓冲驻留、H2D 覆写热换耗时、
# 每 Run 零额外流量（绑定权重不重传）、会话创建耗时构成（2.3s 的分解对照）。
import sys, ctypes, time
sys.path.append('/home/wrp/farm_pkg')
import numpy as np, onnx, onnx.helper as oh, onnxruntime as ort

RT = ctypes.CDLL("/home/wrp/farm_pkg/nvidia/cuda_runtime/lib/libcudart.so.12")
def cuda_alloc(n):
    p = ctypes.c_void_p()
    assert RT.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)) == 0
    return p
def cuda_upload(dst, arr):
    assert RT.cudaMemcpy(dst, arr.ctypes.data, ctypes.c_size_t(arr.nbytes), 4) == 0  # H2D
def cuda_download(src, n):
    h = np.empty(n // 4, np.float32)
    assert RT.cudaMemcpy(h.ctypes.data, src, ctypes.c_size_t(n), 2) == 0  # D2H
    return h

# 模型：YGO fb64 权重量级模拟（~34M 参两层，138MB fp32）
din, dh, dout, N = 1024, 8192, 64, 8
rs = np.random.RandomState(7)
w1 = (rs.randn(din, dh) * 0.02).astype(np.float32)
b1 = np.zeros(dh, np.float32)
w2 = (rs.randn(dh, dout) * 0.02).astype(np.float32)
mb = w1.nbytes + b1.nbytes + w2.nbytes
print(f"权重总量 = {mb/1e6:.0f} MB（对齐 YGO fb64 量级）")
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
model = oh.make_model(oh.make_graph(nodes, "swap", ins, outs, inits),
                      opset_imports=[oh.make_opsetid("", 13)])
model.ir_version = 8
onnx.save(model, "/tmp/ortswap/mlp_cuda.onnx")

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
t0 = time.perf_counter()
sess = ort.InferenceSession("/tmp/ortswap/mlp_cuda.onnx", so,
                            providers=["CUDAExecutionProvider"])
print(f"会话创建（含 CUDA EP init）= {(time.perf_counter()-t0)*1e3:.0f} ms ← --ort-jobs 2.3s 的构成大头即此")

x = np.random.RandomState(9).randn(N, din).astype(np.float32)
px, pw1, pb1, pw2 = (cuda_alloc(a.nbytes) for a in (x, w1, b1, w2))
cuda_upload(px, x); cuda_upload(pw1, w1); cuda_upload(pb1, b1); cuda_upload(pw2, w2)
py = cuda_alloc(N * dout * 4)
io = ort.IOBinding(sess)
io.bind_input("x", "cuda", 0, np.float32, x.shape, px)
io.bind_input("w1", "cuda", 0, np.float32, w1.shape, pw1)
io.bind_input("b1", "cuda", 0, np.float32, b1.shape, pb1)
io.bind_input("w2", "cuda", 0, np.float32, w2.shape, pw2)
io.bind_output("y", "cuda", 0, np.float32, (N, dout), py)
sess.run_with_iobinding(io, None)
o_base = cuda_download(py, N * dout * 4)

def swap(w1new):
    t = time.perf_counter()
    cuda_upload(pw1, w1new)                    # 热换本体=一次 H2D 覆写
    RT.cudaDeviceSynchronize()
    return (time.perf_counter() - t) * 1e3

w1b = (np.random.RandomState(99).randn(din, dh) * 0.02).astype(np.float32)
ms = swap(w1b)
sess.run_with_iobinding(io, None)
o_swap = cuda_download(py, N * dout * 4)
print(f"P_swap 热换生效={not np.array_equal(o_base, o_swap)}  换心耗时={ms:.1f} ms（H2D 上限≈{mb/11e9*1e3:.0f} ms@11GB/s）")
ms2 = swap(w1)
sess.run_with_iobinding(io, None)
o_back = cuda_download(py, N * dout * 4)
print(f"P3 换回幂等={np.array_equal(o_base, o_back)}  耗时={ms2:.1f} ms")
for _ in range(5): sess.run_with_iobinding(io, None)
t = time.perf_counter()
for _ in range(100): sess.run_with_iobinding(io, None)
print(f"每 Run（含权重输入）= {(time.perf_counter()-t)*10:.3f} ms —— 绑定缓冲不重传=应与常规会话同量级")
