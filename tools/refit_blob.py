# -*- coding: utf-8 -*-
"""refit_blob.py — es_pilot 候选权重 → RW1 refit blob 导出器（refit v1，2026-09-21）

用法：
  导出 blob（候选或 parent）：
    python refit_blob.py --pt <model.pt> --engine <refit.engine> --out <blob.rw1>
                        [--qdq <QDQ.onnx>] [--check-parent]
  python 侧 B1/B2 门（同 blob 两次 refit 的逐位确定性）：
    python refit_blob.py --gate-b1b2 --blob <blob.rw1> --engine <refit.engine>
                        [--rows <numeric_rows.npz>] [--fb 64]

RW1 格式（小端）：
  magic "RW1\\0"(4B) | u32 version=1 | u32 n_entries
  entry: u16 name_len | name(utf8) | u8 dtype | u32 numel | 原始字节(C 序)
  dtype: 0=int8, 1=f16, 2=f32, 3=int64（3 为本导出器扩展：引擎有 5 个
  INT64 prototype 常量，契约三码表不够用；C++ 读端需支持）。

名字权威 = Refitter.get_all_weights()（=manifest.json 的 137 项）；dtype 权威 =
Refitter.get_weights_prototype(name)。分类规则（逐名精确一次；TRT 融合依赖闭包
实测见 probe7/8/9）：
  1) tmp_weight_N                → 跳过 + warning。实测 7 个全 numel=1 标量
     （1×int64 + 6×f32），onnx 图内不存在、get_named_weights 返回空（引擎现值
     不可读）——TRT 构建期折叠产物，非 θ 量（θ 无标量参数），保持引擎值不动。
  2) *_quantized                 → int8 权重：θ 转置后对称 per-channel 量化
     （scale 用 parent QDQ initializer，候选间恒定）。
  3) 其余 *_scale（配对 *_quantized 在引擎名内）→ f32 权重 scale，抄 parent。
  4) m.blks.N.dyt.alpha 与 /m/dyt*/Constant_output_0 → 跳过 + warning（保持
     引擎 parent 值）。dyt.alpha 或 dyt Constant 与任何权重/尺度类条目同 set
     会拖出 6 个 "(Unnamed Layer* N) [Constant]"——TRT 报 cannot be refitted
     却又 missing needed Weights，死锁（probe7/8/11 实测）。代价：候选 θ 对
     dyt.alpha 的扰动不生效；要解需 REFIT_INDIVIDUAL 重建引擎。
  5) 全部激活 *_scale（含 /m/dyt*/Tanh_output_0_scale）→ 必须写（parent 恒定
     值）：set 权重 scale 后 TRT 融合闭包要求同 blob 的激活 scale 一并重供
     （probe7/8/9 实测：不写则 get_missing 拖出它们，refit 失败；不带
     dyt.alpha 时 Tanh scale 单独重供无毒）。契约原文"激活 _scale 一律不写"
     被 TRT 该语义推翻，此处为强制偏差。
  7) 名字在 onnx Constant 节点输出表（/m/Constant_*、onnx::Expand_*…）
     → 图常量，抄 Constant 属性值。
  8) m.<key>                     → θ 常量（emb.weight/dyt.alpha/fc*.bias/ls.w/
     trunk_norm.weight…），直接 fp 值。

量化公式（与 onnxruntime.quantization MinMax 对称 per-channel 逐位一致，
B1 前置实验：fresh2 parent 全部 20 个量化权重 mismatch=0）：
  q = clip(np.round(w_t / scale), -127, 127).astype(int8)
  * w_t = θ 权重转置（Gemm→MatMul 改写=权重转置存），scale 沿 axis=1 广播
    （per Linear 输出通道）。
  * np.round=银行家舍入（round-half-to-even）。实测 ORT 用法与 np.round 逐位
    同；floor(x+0.5) 在 blks.{1,2,4}.fc* 上 mismatch 1~2、trunk.0 上 mismatch 5，
    away 舍入 mismatch 更大——都不采用。

onnx::MatMul_NNN_quantized → θ 键映射在运行时解析：initializer →
DequantizeLinear → MatMul 节点名（如 /m/cand_mlp/cand_mlp.0/MatMul）→
模块路径 → θ 键，再用形状（θ.T == quantized dims 且 scale.len == θ.shape[0]）
校验。实测：331=cand_mlp.0.weight 332=cand_mlp.2.weight 335=score.0.weight
336=score.2.weight 339=decl_proj.weight 340=decl_head.weight。
"""
import argparse
import hashlib
import os
import struct
import sys
import time

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")  # 生产同款：TF32=0（须在
# torch/tensorrt 载入前设好，否则 Myelin 拒建 context：build 0 vs execution -1）

import numpy as np

DEFAULT_ENGINE = r"D:/ygo_data/trt_lab/engines/fresh2.fb64.int8_refit.engine"
DEFAULT_QDQ = r"D:/ygo_data/trt_lab/engines/fresh2.fb64.int8.onnx"
DEFAULT_ROWS = r"D:/ygo_data/trt_lab/data/numeric_rows.npz"
PT_STATE_KEY = "state"  # model_final.pt: {vocab, meta_vocab, config, state}

MAGIC = b"RW1\0"
VERSION = 1
DT_INT8, DT_F16, DT_F32, DT_INT64 = 0, 1, 2, 3
RW1_NP = {DT_INT8: np.int8, DT_F16: np.float16, DT_F32: np.float32,
          DT_INT64: np.int64}
NP_RW1 = {np.dtype(np.int8): DT_INT8, np.dtype(np.float16): DT_F16,
          np.dtype(np.float32): DT_F32, np.dtype(np.int64): DT_INT64}


def log(msg):
    print(msg, flush=True)


def _bootstrap_torch_cuda():
    """torch lib DLL 目录前置 + 立 CUDA 上下文（载 engine 的前提，抄
    probe_refit.py 手法）。"""
    import torch
    _tl = os.path.join(os.path.dirname(torch.__file__), "lib")
    os.add_dll_directory(_tl)
    os.environ["PATH"] = _tl + os.pathsep + os.environ["PATH"]
    torch.zeros(1, device="cuda")
    return torch


def _sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- θ / onnx 源
def load_theta(pt_path):
    """model_final.pt → {key: f32 ndarray}。顶层是 {vocab, meta_vocab, config,
    state}，真 state_dict 在 top['state']（55 键）。"""
    torch = _bootstrap_torch_cuda()
    top = torch.load(pt_path, map_location="cpu", weights_only=True)
    sd = None
    if isinstance(top, dict) and PT_STATE_KEY in top and all(
            torch.is_tensor(v) for v in top[PT_STATE_KEY].values()):
        sd = top[PT_STATE_KEY]
    else:  # 回退：递归找"全是张量且 >10 键"的 dict
        def hunt(d):
            nonlocal sd
            if isinstance(d, dict):
                if len(d) > 10 and all(torch.is_tensor(v) for v in d.values()):
                    if sd is None:
                        sd = d
                    return
                for v in d.values():
                    hunt(v)
        hunt(top)
    if sd is None:
        raise RuntimeError("%s 里找不到 state_dict" % pt_path)
    out = {}
    for k, v in sd.items():
        a = v.detach().cpu().numpy()
        if a.dtype != np.float32:
            raise RuntimeError("θ %s dtype=%s 非 f32（避免静默改bits）"
                               % (k, a.dtype))
        out[k] = a
    return out


def load_qdq_sources(qdq_path):
    """QDQ onnx → (initializers 名→ndarray, Constant 节点输出名→ndarray,
    输入名→消费节点表)。"""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(qdq_path)
    inits = {}
    for it in m.graph.initializer:
        inits[it.name] = numpy_helper.to_array(it)
    consts = {}
    for nd in m.graph.node:
        if nd.op_type == "Constant":
            for a in nd.attribute:
                if a.name == "value":
                    consts[nd.output[0]] = numpy_helper.to_array(a.t)
    consumers = {}
    for nd in m.graph.node:
        for i_ in nd.input:
            consumers.setdefault(i_, []).append(nd)
    return inits, consts, consumers


# ---------------------------------------------------------------- 量化核心
def quantize(w_fp32, scale_per_channel):
    """对称 per-channel int8 量化（与 ORT MinMax QDQ 权重量化逐位一致）。

    w_fp32: f32 ndarray，已是引擎存向（Linear 权重转置后 [K, N]）；
    scale_per_channel: f32 [N]（per 输出通道）；
    返回 int8 同形。q = clip(np.round(w/s), -127, 127)。
    """
    if w_fp32.dtype != np.float32 or scale_per_channel.dtype != np.float32:
        raise TypeError("quantize 只吃 f32")
    div = w_fp32 / scale_per_channel.reshape(1, -1)
    return np.clip(np.round(div), -127, 127).astype(np.int8)


def _matmul_head_map(quant_names, inits, consumers, theta):
    """onnx::MatMul_NNN_quantized → (θ 键, scale 名, 溯源节点名)。
    链路：initializer → DequantizeLinear → MatMul/Gemm 节点名 → 模块路径；
    θ 键用形状校验（θ.T == quantized dims 且 scale.len == θ.shape[0]）。"""
    out = {}
    for n in quant_names:
        if not (n.startswith("onnx::MatMul_") and n.endswith("_quantized")):
            continue
        sname = n[: -len("_quantized")] + "_scale"
        node_name = None
        for dq in consumers.get(n, []):
            if dq.op_type != "DequantizeLinear":
                continue
            for mm in consumers.get(dq.output[0], []):
                if mm.op_type in ("MatMul", "Gemm"):
                    node_name = mm.name
        if node_name is None:
            raise RuntimeError("%s 追不到 DQ→MatMul 链" % n)
        parts = node_name.strip("/").split("/")     # [m, cand_mlp, cand_mlp.0, MatMul]
        mid = 1 if parts[0] == "m" else 0           # 头部网络在 m 域内与否都兼容
        mod_path = parts[mid:-1]                    # 去掉前导域与结尾算子
        cands = [".".join(mod_path) + ".weight",
                 mod_path[-1] + ".weight"]
        qdims = tuple(inits[n].shape)
        hit = None
        for k in cands:
            if k in theta and theta[k].T.shape == qdims \
                    and inits[sname].size == theta[k].shape[0]:
                hit = k
                break
        if hit is None:
            raise RuntimeError("%s (节点 %s) 配不上 θ 键 %s" % (n, node_name, cands))
        out[n] = (hit, sname, node_name)
    return out


# ---------------------------------------------------------------- 映射表
def build_mapping(engine_names, inits, consts, consumers, theta):
    """引擎名逐名分类。返回 (entries, skipped_act, skipped_dyt, skipped_tmp,
    table_lines)。entries: list of (name, ndarray)，ndarray dtype 已是写入 dtype。"""
    mm_heads = _matmul_head_map(engine_names, inits, consumers, theta)
    quant_partner = set(mm_heads) | {
        n[: -len("_quantized")] + "_quantized"
        for n in engine_names if n.endswith("_quantized")}

    def is_dyt_const(n):
        return (n.startswith("/m/dyt") and n.endswith("/Constant_output_0")
                and n in consts)

    entries, skipped_act, skipped_dyt, skipped_tmp, lines = [], [], [], [], []

    def emit(name, arr, note):
        entries.append((name, np.ascontiguousarray(arr)))
        lines.append("  %-40s %s" % (name, note))

    for n in engine_names:
        if n.startswith("tmp_weight"):
            skipped_tmp.append(n)
            continue
        if n.endswith("_quantized"):
            if n in mm_heads:
                key, sname, node = mm_heads[n]
                w_t = np.ascontiguousarray(theta[key].T)
                q = quantize(w_t, inits[sname].astype(np.float32))
                emit(n, q, "↔ %-22s 转置 int8 axis1 + %s  ←%s"
                     % (key, sname, node))
            else:
                if not (n.startswith("m.") and n.endswith(".weight_t_quantized")):
                    raise RuntimeError("陌生量化名 %s" % n)
                key = n[len("m."):-len(".weight_t_quantized")] + ".weight"
                sname = n[: -len("_quantized")] + "_scale"
                w_t = np.ascontiguousarray(theta[key].T)
                q = quantize(w_t, inits[sname].astype(np.float32))
                emit(n, q, "↔ %-22s 转置 int8 axis1 + %s"
                     % (key, sname))
            continue
        if n.endswith("_scale"):
            if (n[: -len("_scale")] + "_quantized") in quant_partner:
                emit(n, inits[n].astype(np.float32), "= parent 权重 scale（恒定）")
            else:
                if n not in inits:
                    raise RuntimeError("激活 scale %s 不在 onnx initializer" % n)
                emit(n, inits[n].astype(np.float32),
                     "= parent 激活 scale（refit 闭包依赖，恒定）")
            continue
        if n in consts:
            if is_dyt_const(n):
                skipped_dyt.append(n)
                continue
            emit(n, consts[n], "= 图常量（onnx Constant 节点值）%s%s"
                 % (str(consts[n].shape), consts[n].dtype))
            continue
        if n.startswith("m."):
            key = n[len("m."):]
            if key.endswith("dyt.alpha"):
                # probe11：dyt.alpha 与 quant/wscale/act 任何一组同 set 都会拖出
                # 6 个不可 refit 的 unnamed FoldedScale → refit 死锁；放弃换心，
                # 候选对 dyt.alpha 的扰动无效（需 REFIT_INDIVIDUAL 重建引擎）。
                skipped_dyt.append(n)
                continue
            if key not in theta:
                raise RuntimeError("引擎常量 %s 在 θ 里无键 %s" % (n, key))
            if n in inits and inits[n].shape != theta[key].shape:
                raise RuntimeError("θ %s 形状 %s ≠ onnx %s"
                                   % (key, theta[key].shape, inits[n].shape))
            emit(n, theta[key], "↔ %-22s θ 常量直抄" % key)
            continue
        raise RuntimeError("无法分类的引擎名 %s" % n)
    return entries, skipped_act, skipped_dyt, skipped_tmp, lines


def check_parent_bitwise(entries, inits):
    """--check-parent：全部 int8 量化权重 + 有 initializer 参照的条目逐位
    对照 parent（θ 必须就是烤引擎那份）。按展平值对照（0-d 标量被
    ascontiguousarray 提升成 (1,) 属正常，TRT set 只看 numel）。返回
    (n_checked, [坏名])。"""
    bad = []
    checked = 0
    for name, arr in entries:
        if name not in inits:
            continue
        ref = inits[name]
        checked += 1
        if arr.dtype != ref.dtype or arr.size != ref.size or \
                not np.array_equal(arr.reshape(-1), ref.reshape(-1)):
            mism = None
            if arr.size == ref.size and arr.dtype == ref.dtype:
                mism = int((arr.reshape(-1) != ref.reshape(-1)).sum())
            bad.append((name, mism if mism is not None else
                        "dtype/size %s%s vs %s%s" % (arr.dtype, arr.shape,
                                                     ref.dtype, ref.shape)))
    return checked, bad


# ---------------------------------------------------------------- RW1 编解码
def write_rw1(path, entries):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(entries)))
        for name, arr in entries:
            raw = arr.tobytes(order="C")
            nb = name.encode("utf-8")
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<B", NP_RW1[np.dtype(arr.dtype)]))
            f.write(struct.pack("<I", arr.size))
            f.write(raw)
    return os.path.getsize(path)


def read_rw1(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != MAGIC:
        raise RuntimeError("bad magic")
    ver, n = struct.unpack_from("<II", data, 4)
    if ver != VERSION:
        raise RuntimeError("bad version %d" % ver)
    off, out = 12, []
    for _ in range(n):
        (nl,) = struct.unpack_from("<H", data, off); off += 2
        name = data[off:off + nl].decode("utf-8"); off += nl
        (dt,) = struct.unpack_from("<B", data, off); off += 1
        (numel,) = struct.unpack_from("<I", data, off); off += 4
        arr = np.frombuffer(data, dtype=RW1_NP[dt], count=numel, offset=off)
        off += numel * RW1_NP[dt]().nbytes
        out.append((name, arr))
    if off != len(data):
        raise RuntimeError("blob 有尾料：%d/%d" % (off, len(data)))
    return out


# ---------------------------------------------------------------- 导出命令
def cmd_export(a):
    torch = _bootstrap_torch_cuda()  # noqa: F841（DLL/CUDA 前置）
    import tensorrt as trt
    t0 = time.time()
    theta = load_theta(a.pt)
    inits, consts, consumers = load_qdq_sources(a.qdq)
    log("[export] θ %d 键 / onnx initializer %d / Constant 节点 %d (%.1fs)"
        % (len(theta), len(inits), len(consts), time.time() - t0))

    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(
        open(a.engine, "rb").read())
    if not engine.refittable:
        raise RuntimeError("引擎不带 REFIT 位")
    ref = trt.Refitter(engine, logger)
    engine_names = list(ref.get_all_weights())          # 名字权威
    log("[export] 引擎 refit 名 %d 个" % len(engine_names))

    # prototype dtype 权威（无此 API 时按 w384 探针经验兜底）
    TRT2NP = {trt.DataType.FLOAT: np.float32, trt.DataType.HALF: np.float16,
              trt.DataType.INT8: np.int8, trt.DataType.INT32: np.int32,
              trt.DataType.INT64: np.int64, trt.DataType.BOOL: np.bool_}
    proto_np = {}
    if hasattr(ref, "get_weights_prototype"):
        for n in engine_names:
            proto_np[n] = TRT2NP[ref.get_weights_prototype(n).dtype]
    else:
        for n in engine_names:
            proto_np[n] = (np.int8 if n.endswith("_quantized")
                           else np.float32)
    del ref, engine

    entries, skipped_act, skipped_dyt, skipped_tmp, lines = build_mapping(
        engine_names, inits, consts, consumers, theta)

    log("[export] 映射表（%d 项写入 / %d dyt 折锁跳过 / %d tmp_weight 跳过）:"
        % (len(entries), len(skipped_dyt), len(skipped_tmp)))
    for l in lines:
        log(l)
    if skipped_dyt:
        log("[export][WARN] dyt 死锁家族 %d 名跳过（dyt.alpha×6 或 dyt "
            "Constant×6 与任何权重/尺度类条目同 set 即拖出 6 个不可 refit 的 "
            "unnamed FoldedScale → refit 死锁，probe7/8/11 实测；代价：候选 θ "
            "对 dyt.alpha 的扰动不生效，需 REFIT_INDIVIDUAL 重建引擎才能解）:"
            " %s" % (len(skipped_dyt), " ".join(sorted(skipped_dyt))))
    if skipped_tmp:
        log("[export][WARN] tmp_weight_N 跳过（实测 numel=1 标量、onnx 图内不"
            "存在、引擎现值不可读 → 推定 TRT 构建期折叠常量、非 θ 量，引擎保"
            "持原值）: %s" % " ".join(skipped_tmp))

    # dtype/覆盖 校验
    ent_names = set(n for n, _ in entries)
    for n, arr in entries:
        if proto_np[n] != arr.dtype:
            raise RuntimeError("%s prototype %s ≠ 写入 %s"
                               % (n, proto_np[n], arr.dtype))
        if arr.size == 0:
            raise RuntimeError("%s 空张量" % n)
    assert len(ent_names) == len(entries)

    if a.check_parent:
        checked, bad = check_parent_bitwise(entries, inits)
        log("[export] --check-parent: %d 条逐位对照 parent onnx, 坏 %d"
            % (checked, len(bad)))
        for name, info in bad:
            log("   [MISMATCH] %s %s" % (name, info))
        if bad:
            raise RuntimeError("parent 逐位复现失败（见上）")

    size = write_rw1(a.out, entries)
    log("[export] 写出 %s：%.2f MB（%d 条目）"
        % (a.out, size / 1e6, len(entries)))


# ---------------------------------------------------------------- B1/B2 门
def cmd_gate(a):
    torch = _bootstrap_torch_cuda()
    import tensorrt as trt
    TRT2TORCH = {trt.DataType.FLOAT: torch.float32, trt.DataType.INT64: torch.int64,
                 trt.DataType.BOOL: torch.bool, trt.DataType.INT32: torch.int32,
                 trt.DataType.HALF: torch.float16}

    entries = read_rw1(a.blob)
    log("[gate] blob %d 条目，%s" % (len(entries), a.blob))

    logger = trt.Logger(trt.Logger.WARNING)
    t0 = time.time()
    engine = trt.Runtime(logger).deserialize_cuda_engine(
        open(a.engine, "rb").read())
    log("[gate] deserialize %.2fs refittable=%s"
        % (time.time() - t0, engine.refittable))
    ref0 = trt.Refitter(engine, logger)
    known = set(ref0.get_all_weights())
    del ref0
    for n, _ in entries:
        if n not in known:
            raise RuntimeError("blob 名 %s 不在引擎 refit 名单" % n)
    log("[gate] 名单校验 OK（%d/%d）" % (len(entries), len(known)))

    rows = dict(np.load(a.rows))
    outs_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                  if engine.get_tensor_mode(engine.get_tensor_name(i))
                  == trt.TensorIOMode.OUTPUT]
    log("[gate] 输出: %s" % outs_names)

    def run_batch():
        """新 context + 新设备缓冲，跑 rows 前 fb 行，回传输出。"""
        ctx = engine.create_execution_context()
        dev = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dt = engine.get_tensor_dtype(name)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                arr = np.ascontiguousarray(rows[name][:a.fb])
                t = torch.from_numpy(arr).cuda()
            else:
                shape = tuple(engine.get_tensor_shape(name))
                t = torch.zeros(shape, dtype=TRT2TORCH[dt], device="cuda")
            dev[name] = t
            ctx.set_tensor_address(name, t.data_ptr())
        stream = torch.cuda.Stream()
        assert ctx.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()
        got = {n: dev[n].cpu().numpy().copy() for n in outs_names}
        del ctx, dev
        return got

    def refit_blob(tag):
        ref = trt.Refitter(engine, logger)
        keep = []
        for n, arr in entries:
            a2 = np.array(arr)  # 拷贝独立缓冲（frombuffer 只读视图保险）
            keep.append(a2)
            if not ref.set_named_weights(n, trt.Weights(a2)):
                raise RuntimeError("[%s] set_named_weights %s 失败" % (tag, n))
        miss = ref.get_missing()
        if miss and miss[0]:
            raise RuntimeError("[%s] get_missing 非空: %s" % (tag, miss[:5]))
        t0 = time.time()
        assert ref.refit_cuda_engine(), "[%s] refit_cuda_engine 失败" % tag
        log("[gate] %s refit %d 条 %.2fs" % (tag, len(entries), time.time() - t0))
        del ref, keep

    pre = run_batch()
    refit_blob("第1次")
    post1 = run_batch()
    refit_blob("第2次")
    post2 = run_batch()

    log("[gate] ---- B1: refit 前 vs 第1次 refit 后（同权重换心不动行为） ----")
    b1_ok = True
    for n in outs_names:
        same = pre[n].shape == post1[n].shape and np.array_equal(pre[n], post1[n])
        b1_ok &= bool(same)
        log("   %-12s pre=%s post1=%s -> %s"
            % (n, _sha(pre[n]), _sha(post1[n]),
               "逐位相等" if same else "!!不等!!"))
    log("[gate] ---- B2: 第1次 vs 第2次 refit 后（确定性） ----")
    b2_ok = True
    for n in outs_names:
        same = np.array_equal(post1[n], post2[n])
        b2_ok &= bool(same)
        log("   %-12s post1=%s post2=%s -> %s"
            % (n, _sha(post1[n]), _sha(post2[n]),
               "逐位相等" if same else "!!不等!!"))

    if a.save_prefix:
        os.makedirs(a.save_prefix, exist_ok=True)
        np.savez(os.path.join(a.save_prefix, "b1b2_outputs.npz"),
                 **{"pre_" + n: pre[n] for n in outs_names},
                 **{"post1_" + n: post1[n] for n in outs_names},
                 **{"post2_" + n: post2[n] for n in outs_names})
        log("[gate] 输出存档 %s" % os.path.join(a.save_prefix, "b1b2_outputs.npz"))

    log("[gate] B1=%s B2=%s" % ("PASS" if b1_ok else "FAIL",
                                "PASS" if b2_ok else "FAIL"))
    if not (b1_ok and b2_ok):
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="es_pilot 候选权重 → RW1 refit "
                                             "blob 导出器（refit v1）")
    ap.add_argument("--pt", help="torch 模型（候选 θ 或 parent）")
    ap.add_argument("--engine", default=DEFAULT_ENGINE, help="REFIT 引擎")
    ap.add_argument("--qdq", default=DEFAULT_QDQ, help="parent QDQ onnx")
    ap.add_argument("--out", help="输出 blob.rw1")
    ap.add_argument("--check-parent", action="store_true",
                    help="导出时对 parent onnx 逐位复核（仅 parent θ 用）")
    ap.add_argument("--gate-b1b2", action="store_true",
                    help="跑 B1/B2 门：blob 两次 refit 逐位对照")
    ap.add_argument("--blob", help="[gate] blob.rw1 路径")
    ap.add_argument("--rows", default=DEFAULT_ROWS, help="[gate] 行 npz")
    ap.add_argument("--fb", type=int, default=64, help="[gate] 批大小")
    ap.add_argument("--save-prefix", default=r"D:/ygo_data/refit_v1",
                    help="[gate] 输出存档目录（空串跳过）")
    a = ap.parse_args()
    if a.gate_b1b2:
        if not a.blob:
            ap.error("--gate-b1b2 需要 --blob")
        cmd_gate(a)
        return
    if not (a.pt and a.out):
        ap.error("导出模式需要 --pt 与 --out")
    cmd_export(a)


if __name__ == "__main__":
    main()
