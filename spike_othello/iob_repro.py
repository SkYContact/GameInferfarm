# iob_repro.py — 复现件：网格图 CUDA io-binding 设备驻留输入（模拟 C++ 农场路径）
# python 普通 run() 已验证通过（子代理）；此脚本区分"图×驱动"vs"C++ 绑定 bug"
import numpy as np
import onnxruntime as ort

M = r"D:\inferfarm\models\othello_pop_grid.fb1024.onnx"
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
s = ort.InferenceSession(M, so, providers=[("CUDAExecutionProvider", {"device_id": "0"})])

S, P = 1024, 128
rng = np.random.default_rng(1)
own = rng.standard_normal((S, 64), dtype=np.float32)
opp = rng.standard_normal((S, 64), dtype=np.float32)
pop = (rng.standard_normal((P, 41280), dtype=np.float32) * 0.05).astype(np.float32)
mid = rng.integers(0, P, size=S, dtype=np.int64)

def to_dev(a):
    return ort.OrtValue.ortvalue_from_numpy(a, "cuda", 0)

io = s.io_binding()
for name, arr in (("own", own), ("opp", opp), ("pop", pop), ("mid", mid)):
    io.bind_ortvalue_input(name, to_dev(arr))
out = to_dev(np.zeros((S, 64), dtype=np.float32))
io.bind_ortvalue_output("policy", out)
print("绑定完成，首跑……")
s.run_with_iobinding(io)
y1 = out.numpy()
print("首跑 OK，policy[0,:3] =", y1[0, :3])
s.run_with_iobinding(io)
print("复跑 OK，逐位同 =", np.array_equal(y1, out.numpy()))
