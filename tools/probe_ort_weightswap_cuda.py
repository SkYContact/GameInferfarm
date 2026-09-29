# probe_ort_weightswap_cuda.py — CUDA EP 设备驻留热换+计时（YGO 工单判决探针 GPU 段）
# v2（2026-09-29 会话四）：①权重全住 pinned 暂存（生产语义；naive pageable 大块
#   H2D 在本机 ORT CUDA EP init 后原生崩——1.26/1.30 一致的环境怪癖，与版本无关）
#   ②尺寸对齐 YGO fb64 真量级（134MB 权重）③环境=farm_env.sh 现役（ORT 1.30）。
# 用法：source ~/farm_env.sh && PYTHONPATH=$FARM_ORT_DIR/.. \
#        ~/trt_venv/bin/python -u tools/probe_ort_weightswap_cuda.py
import sys, ctypes, time
sys.path.insert(0, '/home/wrp/farm_pkg_130')
import numpy as np, onnx, onnx.helper as oh, onnxruntime as ort

RT = ctypes.CDLL("/home/wrp/farm_pkg_130/nvidia/cu13/lib/libcudart.so.13")
RT.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
RT.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
RT.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
RT.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
RT.cudaDeviceSynchronize.argtypes = []
# v2.1（09-29 锁内实跑定谳）：同步 cudaMemcpy 入口在本机 ORT CUDA EP init 后原生崩
#   ——pinned 源/128KB 也崩（怪癖面比旧档"pageable ≥1MB"更宽：同窗 cudaMalloc/
#   cudaHostAlloc 皆活，唯 sync-memcpy 死）。改 MemcpyAsync+流同步=框架生产同款
#   路径（fence 门 Linux 全绿实证），免疫。
def cuda_alloc(n):
    p = ctypes.c_void_p()
    assert RT.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)) == 0
    return p
def pinned_view(nbytes, dtype=np.float32):
    p = ctypes.c_void_p()
    assert RT.cudaHostAlloc(ctypes.byref(p), ctypes.c_size_t(nbytes), 0) == 0
    return np.ctypeslib.as_array(ctypes.cast(p, ctypes.POINTER(ctypes.c_float)),
                                 shape=(nbytes // 4,))
def h2d(dst, src_pinned, nbytes):
    assert RT.cudaMemcpyAsync(dst, ctypes.c_void_p(src_pinned), ctypes.c_size_t(nbytes), 4, None) == 0  # pinned→dev
    assert RT.cudaStreamSynchronize(None) == 0
def d2h(dst_pinned, src, nbytes):
    if isinstance(src, ctypes.c_void_p):
        src = src.value
    assert RT.cudaMemcpyAsync(dst_pinned, ctypes.c_void_p(src), ctypes.c_size_t(nbytes), 2, None) == 0
    assert RT.cudaStreamSynchronize(None) == 0

# 模型：YGO fb64 权重量级（w1 4096×8192 fp32 = 134MB，两层 MLP）
din, dh, dout, N = 4096, 8192, 64, 8
rs = np.random.RandomState(7)
w1 = (rs.randn(din, dh) * 0.02).astype(np.float32)
b1 = np.zeros(dh, np.float32)
w2 = (rs.randn(dh, dout) * 0.02).astype(np.float32)
mb = w1.nbytes + b1.nbytes + w2.nbytes
print(f"权重总量 = {mb/1e6:.0f} MB（YGO fb64 量级）", flush=True)
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

t0 = time.perf_counter()
so = ort.SessionOptions()
sess = ort.InferenceSession("/tmp/ortswap/mlp_cuda.onnx", so,
                            providers=["CUDAExecutionProvider"])
print(f"会话创建冷启（含 CUDA EP init）= {(time.perf_counter()-t0)*1e3:.0f} ms", flush=True)
t0 = time.perf_counter()
sess2 = ort.InferenceSession("/tmp/ortswap/mlp_cuda.onnx", so,
                             providers=["CUDAExecutionProvider"])
print(f"会话创建温启（同进程第二会话）= {(time.perf_counter()-t0)*1e3:.0f} ms", flush=True)

x = np.random.RandomState(9).randn(N, din).astype(np.float32)
# 权重全住 pinned 暂存（生产语义：H2D 全程 pinned 源）
pw1h = pinned_view(w1.nbytes); pw1h[:] = w1.ravel()
pb1h = pinned_view(b1.nbytes); pb1h[:] = b1.ravel()
pw2h = pinned_view(w2.nbytes); pw2h[:] = w2.ravel()
pxh = pinned_view(x.nbytes);  pxh[:] = x.ravel()
px, pw1, pb1, pw2 = (cuda_alloc(a.nbytes) for a in (x, w1, b1, w2))
py = cuda_alloc(N * dout * 4)
h2d(px, pxh.ctypes.data, x.nbytes)
h2d(pw1, pw1h.ctypes.data, w1.nbytes)
h2d(pb1, pb1h.ctypes.data, b1.nbytes)
h2d(pw2, pw2h.ctypes.data, w2.nbytes)
io = ort.IOBinding(sess)
# 1.30 pybind 新重载：buffer_ptr 须纯 int（c_void_p 实例不满足 SupportsInt）
io.bind_input("x", "cuda", 0, np.float32, x.shape, px.value)
io.bind_input("w1", "cuda", 0, np.float32, w1.shape, pw1.value)
io.bind_input("b1", "cuda", 0, np.float32, b1.shape, pb1.value)
io.bind_input("w2", "cuda", 0, np.float32, w2.shape, pw2.value)
io.bind_output("y", "cuda", 0, np.float32, (N, dout), py.value)
oh16 = pinned_view(N * dout * 4)
sess.run_with_iobinding(io, None)
d2h(oh16.ctypes.data, py, N * dout * 4)
o_base = oh16.copy()

def swap(w1new):
    """热换本体：新权重→pinned 暂存→一次 pinned H2D 覆写绑定缓冲"""
    t = time.perf_counter()
    pw1h[:] = w1new.ravel()
    h2d(pw1, pw1h.ctypes.data, w1.nbytes)
    RT.cudaDeviceSynchronize()
    return (time.perf_counter() - t) * 1e3

w1b = (np.random.RandomState(99).randn(din, dh) * 0.02).astype(np.float32)
ms = swap(w1b)
sess.run_with_iobinding(io, None)
d2h(oh16.ctypes.data, py, N * dout * 4)
o_swap = oh16.copy()
print(f"P_swap 热换生效={not np.array_equal(o_base, o_swap)}  换心(134MB pinned H2D)={ms:.1f} ms", flush=True)
ms2 = swap(w1)
sess.run_with_iobinding(io, None)
d2h(oh16.ctypes.data, py, N * dout * 4)
o_back = oh16.copy()
print(f"P3 换回幂等={np.array_equal(o_base, o_back)}  耗时={ms2:.1f} ms", flush=True)
for _ in range(5): sess.run_with_iobinding(io, None)
t = time.perf_counter()
for _ in range(100): sess.run_with_iobinding(io, None)
per = (time.perf_counter() - t) * 10
print(f"每 Run（含 134MB 权重输入绑定）= {per:.3f} ms —— 绑定缓冲不重传=应与常规会话同量级", flush=True)
print(f"对账：2.3s 重建 → 热换 {ms:.0f}ms + 零会话创建；124 作业/代 ≈ 285s → {124*ms/1e3:.1f}s", flush=True)
