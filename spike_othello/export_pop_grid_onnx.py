# -*- coding: utf-8 -*-
"""尖刺第 3.6 步（负载 A / 路由 v2.2"网格图+死路由块"）：导出 + 对拍。

接口与 v1/v2.0/v2.1 完全一致：own[S,64]/opp[S,64]/pop[P=128,41280]/mid[S] int64
-> policy[S,64]，S=1024 钉批。

## v2.2 要防的病（农场实测活性非确定性，根因已定位）
农场的批可部分填充（n<1024）：行 n..1023 是陈旧行，槽位留着上一批旧 mid（仍是
合法个体号）→ 参与 CumSum->j->ScatterND 后 (a) 同个体行数可超 8 → idx 溢出到下
一个体块；(b) 与新鲜行格子碰撞 → ScatterND 重复索引原子写 → 新鲜行被陈旧行覆
盖（值不定，同配置重跑指纹不同）。农场因此把批尾 mid 毒化为 -1（int64 全 0xFF）；
另有指纹探针图案 mid=(e*7+seed*13)%14969 可远超 P。

## v2.2 改法（图内全部标准 ONNX 算子）
1. 活值钳制：mid_live = Max(Min(mid, P-1), 0)（int64 走 Min/Max，Clip 不支持整型）
   ——保证 GatherElements 取列索引合法 ∈[0,P)。
2. 计数用原始 mid：Equal(mid[:,None], arange(P)[None,:] [1,128] 常量)->Cast->
   CumSum(axis0)。死行(-1)与越界行(>=P)不落任何列 → 新鲜行计数只含新鲜行
   （毒化保证），j_raw 对任意行 ∈[-1,7]。
3. 死路由块：mid<0 的行=死行，路由到专属第 P+1 个体块 idx_dead=P*8+(j_raw%8)
   （Mod 为 Python 语义，-1%8=7）。**越界行(mid>=P)同样进死块**——若按字面钳制
   后当活行路由，其 j 取自 P-1 列纯新鲜计数：无先行新鲜行时 j=-1 → idx=(P-1)*8-1
   反渗进 P-2 块；有 c 个先行新鲜行时 j=c-1 与第 c 个新鲜行撞格——均破坏"活行
   输出与无死行版逐位同"（任务验收条件），故越界行必须与死行同块。死行之间允许
   碰撞（输出不被收割；且死块零权重 → 死行输出恒 0，碰撞后仍确定）；新鲜行永不
   入死块。活行 idx = mid_live*8 + j_raw。
4. 权重面补一行：网格 [P,8] -> [P+1,8]；每层 W=Slice(pop,...) 转 [P,i,o] 后 Concat
   零行 [1,i,o]，b 偏置 Concat 零行 [1,1,o]（死块算垃圾=0 不影响活块，行独立）。
5. 输出 Gather 用同一 idx（含死路由）不变。

对拍（全 CPU EP，pop=[128,41280]，0 号个体=bc_t3k 真权重 + 127 随机 theta）：
  A. 钉批图 S=1024 ×4 轮（全活行，回归口径）：① ONNX vs torch 网格 ② vs torch v1
     路由 ③ vs 金标准 ④ vs ONNX v1 fb1024 对照 ⑤ 对照自检；
  B. 散射-还原专项：动态 S 孪生图 × S<1024 随机占用（空格子/跳号）；
  C. 死行鲁棒性（新增）：mid 含 -1 毒化行与 999999 越界行（含 1000 死行档），
     活行输出必须与无死行版**逐位同**，死行输出可任意（实测恒 0），复跑稳定。

用法：C:/Users/41601/Miniconda3/envs/q35/python.exe export_pop_grid_onnx.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUN40 = r"C:\word\水文工厂\runs\40_engine_battle"
sys.path.insert(0, RUN40)

from model import (LAYERS, PARAM_SLICES, D_TOTAL,  # noqa: E402
                   forward_one)
from export_pop_onnx import (PopRouteMLP, build_pop,  # noqa: E402
                             export_pop as export_pop_v1)

THETA_NOTE = "pop[0]=bc_t3k theta_final.pt, 1..127=init_theta(9000+i)"
MODELS_DIR = r"D:\inferfarm\models"
S_PIN = 1024   # 批形钉死（与 fb1024 对照组同 S）
P_CAP = 128    # 种群容量钉死（活块数）
J = 8          # 每个体每批至多 8 行（8 局各至多 1 个在飞决策）
G_CAP = P_CAP + 1  # 网格个体面 = P 活块 + 1 死块
MID_POISON = -1      # 农场批尾毒化值（int64 全 0xFF）
MID_OOR = 999999     # 探针图案类越界值

assert D_TOTAL == 41280 and sum(o * i + o for o, i in LAYERS) == D_TOTAL
assert S_PIN == P_CAP * J  # 全活行时占用被算术强制为"全满"


class PopGridMLP(torch.nn.Module):
    """网格路由 v2.2：own/opp [S,64] + pop [P,D] + mid [S](int64) -> policy[S,64]。
    行 s 散射到网格格位 (mid[s], j[s])；死/越界行路由到第 P+1 死块（零权重 → 输出
    恒 0），活格 idx 唯一（结构不变量：每个体每批 ≤J 行，毒化保证计数只含新鲜行）。
    pop 原地切片当 bmm 批权重（零 Gather），同一 idx Gather 还原行序。"""

    def __init__(self, p_cap=P_CAP):
        super().__init__()
        self.register_buffer("_cols",
                             torch.arange(p_cap, dtype=torch.int64).unsqueeze(0),
                             persistent=False)              # [1,P] 列号常量
        for k, (o, i) in enumerate(LAYERS):                # 死块零权重/零偏置
            self.register_buffer("_wdead%d" % k, torch.zeros(1, i, o),
                                 persistent=False)          # [1,i,o]
            self.register_buffer("_bdead%d" % k, torch.zeros(1, 1, o),
                                 persistent=False)          # [1,1,o]

    def forward(self, own, opp, pop, mid):
        P = pop.shape[0]
        G = P + 1                                           # 活块 P + 死块 1
        assert P == self._cols.shape[1]
        x = torch.cat([own, opp], dim=1)                    # [S,128]
        hi = torch.tensor(P - 1, dtype=torch.int64)
        lo = torch.tensor(0, dtype=torch.int64)
        mid_live = torch.maximum(torch.minimum(mid, hi), lo)  # 钳制 [0,P)（取列安全）
        oh = (mid.unsqueeze(1) == self._cols).to(torch.int64)  # 原始 mid 计数
        cnt = torch.cumsum(oh, dim=0)                       # [S,P]（死/越界不落列）
        j_raw = torch.gather(cnt, 1, mid_live.unsqueeze(1)).squeeze(1) - 1
        assert int(j_raw.max()) < J, "结构不变量被破坏：个体批内行数>J"
        dead = (mid < 0) | (mid > P - 1)                    # 毒化(-1)+越界(>=P)
        idx = torch.where(dead,                             # 活格唯一；死格允许碰撞
                          P * J + j_raw % J,
                          mid_live * J + j_raw)             # [S] int64
        grid = torch.zeros(G * J, x.shape[1], dtype=x.dtype)
        grid[idx] = x                                       # ONNX=ScatterND
        h = grid.view(G, J, -1)                             # [P+1,J,128]
        last = len(LAYERS) - 1
        for k, ((ms, vs), (o, i)) in enumerate(zip(PARAM_SLICES, LAYERS)):
            w = torch.cat([pop[:, ms[0]:ms[1]].view(P, o, i).transpose(1, 2),
                           getattr(self, "_wdead%d" % k)], dim=0)  # [P+1,i,o]
            b = torch.cat([pop[:, vs[0]:vs[1]].view(P, 1, o),
                           getattr(self, "_bdead%d" % k)], dim=0)  # [P+1,1,o]
            h = torch.bmm(h, w) + b                         # 死块=0@W+0=0
            if k < last:
                h = torch.relu(h)
        y = h.reshape(G * J, -1)                            # [P*J+J,64]
        return torch.index_select(y, 0, idx)                # 行序还原（同一 idx）


# 标准算子白名单（v2.2 实测形态 + 常规配套；OneHot 已剔除——CUDA EP 坑）
_STANDARD_OPS = {
    "Add", "Cast", "Concat", "Constant", "ConstantOfShape", "CumSum", "Equal",
    "Expand", "Gather", "GatherElements", "Greater", "Identity", "Less",
    "MatMul", "Max", "Min", "Mod", "Mul", "Or", "Relu", "Reshape", "ScatterND",
    "Shape", "Slice", "Squeeze", "Sub", "Transpose", "Unsqueeze", "Where",
}


def _has_const_cols(mm, p_cap):
    """Equal 右操作数的列号常量是否以 [1,P] int64 钉死（initializer 或 Constant）。"""
    import onnx
    want = [1, p_cap]
    for t in list(mm.graph.initializer):
        if list(t.dims) == want and t.data_type == onnx.TensorProto.INT64:
            return True
    for n in mm.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value" and list(a.t.dims) == want \
                        and a.t.data_type == onnx.TensorProto.INT64:
                    return True
    return False


def _export_grid(slots, out, dynamic=False):
    m = PopGridMLP().eval()
    own = torch.zeros(slots, 64)
    opp = torch.zeros(slots, 64)
    pop = torch.zeros(P_CAP, D_TOTAL)
    mid = torch.arange(slots, dtype=torch.int64) % P_CAP    # 全活行，满足不变量
    kw = {}
    if dynamic:
        kw["dynamic_axes"] = {"own": {0: "S"}, "opp": {0: "S"}, "mid": {0: "S"},
                              "policy": {0: "S"}}
    try:
        torch.onnx.export(m, (own, opp, pop, mid), out,
                          input_names=["own", "opp", "pop", "mid"],
                          output_names=["policy"], opset_version=17,
                          dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(m, (own, opp, pop, mid), out,
                          input_names=["own", "opp", "pop", "mid"],
                          output_names=["policy"], opset_version=17, **kw)
    import onnx
    mm = onnx.load(out)
    ins = {i.name: i for i in mm.graph.input}
    assert list(ins) == ["own", "opp", "pop", "mid"], list(ins)  # 输入名与次序
    elem = {n: i.type.tensor_type.elem_type for n, i in ins.items()}
    assert elem["mid"] == onnx.TensorProto.INT64, elem      # mid 必须 int64
    assert elem["own"] == elem["opp"] == elem["pop"] == onnx.TensorProto.FLOAT, elem
    if not dynamic:                                          # 钉批图才断言形状
        dims = {n: [d.dim_value for d in i.type.tensor_type.shape.dim]
                for n, i in ins.items()}
        assert dims["own"] == [slots, 64] and dims["opp"] == [slots, 64], dims
        assert dims["pop"] == [P_CAP, D_TOTAL], dims        # P 钉死=容量
        assert dims["mid"] == [slots], dims                 # S 钉死=批形
        odim = [d.dim_value for d in mm.graph.output[0].type.tensor_type.shape.dim]
        assert mm.graph.output[0].name == "policy" and odim == [slots, 64], odim
    ops = sorted({n.op_type for n in mm.graph.node})
    assert all(o in _STANDARD_OPS for o in ops), ops        # 禁自定义算子
    assert "OneHot" not in ops, "OneHot 已被 Equal+Cast 取代（CUDA EP 坑）"
    assert _has_const_cols(mm, P_CAP), "缺 [1,128] int64 列号常量（Equal 右操作数）"
    return ops


def export_pop_grid(slots, out):
    ops = _export_grid(slots, out)
    has = [o for o in ("ScatterND", "CumSum", "Equal", "Where", "Mod", "Min",
                       "Max", "Cast", "MatMul", "Gather") if o in ops]
    print("export %s (%.2f MB) S=%d P=%d+1死块 J=%d mid=INT64 无OneHot 关键算子=%s"
          % (out, os.path.getsize(out) / 1e6, slots, P_CAP, J, has))


def _cell_stats(mid_np):
    """占用统计（仅活行 0..P-1）+ 跳号个体数 + 死/越界行数。"""
    live = (mid_np >= 0) & (mid_np < P_CAP)
    cnt = np.bincount(mid_np[live], minlength=P_CAP)
    jumps = 0
    for p in np.flatnonzero(cnt):
        pos_ = np.flatnonzero(mid_np == p)
        if pos_[-1] - pos_[0] + 1 != len(pos_):
            jumps += 1
    return {"空格子": int((cnt == 0).sum()), "不足J行": int(((cnt > 0)
            & (cnt < J)).sum()), "满J行": int((cnt == J).sum()),
            "跳号个体": jumps, "死/越界行": int((~live).sum())}


def _sample_mid(slots, full, rng, n_dead=0, n_oor=0):
    """先按 0..J 行/个体生成合法活行占用（总量=slots-n_dead-n_oor），再追加毒化
    -1 与越界 999999 行，整体洗牌（同个体行不连续 -> j 跳号）。"""
    n_live = slots - n_dead - n_oor
    assert n_live >= 0
    if full:
        assert slots == P_CAP * J and n_dead == 0 and n_oor == 0
        cnt = np.full(P_CAP, J, dtype=np.int64)
    else:
        cnt = rng.integers(0, J + 1, size=P_CAP).astype(np.int64)
        while cnt.sum() != n_live:                    # 补齐/回退（保持 0<=c<=J）
            p = int(rng.integers(0, P_CAP))
            if cnt.sum() < n_live and cnt[p] < J:
                cnt[p] += 1
            elif cnt.sum() > n_live and cnt[p] > 0:
                cnt[p] -= 1
    assert (cnt <= J).all() and cnt.sum() == n_live
    mid = np.repeat(np.arange(P_CAP, dtype=np.int64), cnt).tolist()
    mid += [MID_POISON] * n_dead + [MID_OOR] * n_oor
    mid = np.array(mid, dtype=np.int64)[rng.permutation(slots)]
    hit0 = np.flatnonzero(mid == 0)
    if hit0.size and hit0[0] != 0:                   # 0 号=真权重个体放首行
        mid[0], mid[hit0[0]] = mid[hit0[0]], mid[0]
    return torch.from_numpy(mid)


def _check_invariant(mid):
    """散射安全性（torch 侧独立复核）：活行 j∈[0,J) 且活格 idx 唯一（ScatterND
    重复索引=未定义）；死/越界行全部落死块 [P*J,(P+1)*J)。返回 idx。"""
    mid_live = torch.maximum(torch.minimum(mid, torch.tensor(P_CAP - 1)),
                             torch.tensor(0))
    oh = (mid.unsqueeze(1) == torch.arange(P_CAP).unsqueeze(0)).to(torch.int64)
    j_raw = torch.gather(torch.cumsum(oh, 0), 1,
                         mid_live.unsqueeze(1)).squeeze(1) - 1
    dead = (mid < 0) | (mid > P_CAP - 1)
    live = ~dead
    idx = torch.where(dead, P_CAP * J + torch.remainder(j_raw, J),
                      mid_live * J + j_raw)
    assert int(j_raw[live].max()) < J and int(j_raw[live].min()) >= 0, \
        "活行 j 越界：结构不变量被破坏"
    li = idx[live]
    assert li.unique().numel() == int(live.sum()), "活格 idx 重复：散射-还原被破坏"
    if bool(dead.any()):
        di = idx[dead]
        assert bool(((di >= P_CAP * J) & (di < (P_CAP + 1) * J)).all()), \
            "死行未落死块"
    return idx


def _amax(logits):
    """平局取小格号（与 parity_check.py / v1 口径一致）。"""
    return logits.argmax(axis=1)


def _run_all(grid, route, pop, own, opp, mid):
    """三路结果：torch 网格 / torch v1 路由 / 逐行金标准 forward_one。"""
    x = np.concatenate([own, opp], axis=1)
    t_grid = grid(torch.from_numpy(own), torch.from_numpy(opp), pop, mid).numpy()
    t_route = route(torch.from_numpy(own), torch.from_numpy(opp), pop, mid).numpy()
    t_one = np.stack([forward_one(pop[m], torch.from_numpy(x[s:s + 1])).numpy()[0]
                      for s, m in enumerate(mid.tolist())])
    return t_grid, t_route, t_one


def parity_pinned(slots, grid_path, v1_path, rounds=4):
    """A. 交付图（S=1024 钉批，全活行回归口径）对拍：占用恒全满，行随机洗牌。"""
    import onnxruntime as ort
    pop = build_pop()
    sg = ort.InferenceSession(grid_path, providers=["CPUExecutionProvider"])
    sv = ort.InferenceSession(v1_path, providers=["CPUExecutionProvider"])
    grid, route = PopGridMLP().eval(), PopRouteMLP().eval()
    names = ["g1_onnx_grid_vs_torch_grid", "g2_torch_grid_vs_torch_v1route",
             "g3_torch_grid_vs_one", "g4_onnx_grid_vs_onnx_v1",
             "g5_onnx_v1_vs_torch_v1route"]
    agg = {n: [0, 0, 0.0] for n in names}            # [差异位数, 总位数, max|diff|]
    flip = {n: 0 for n in names}
    amax_gap_max, logit_absmax = 0.0, 0.0
    for r in range(rounds):
        rng = np.random.default_rng(3000 * slots + r)
        own = rng.random((slots, 64)).astype(np.float32)
        opp = rng.random((slots, 64)).astype(np.float32)
        mid = _sample_mid(slots, full=True, rng=rng)
        _check_invariant(mid)
        t_grid, t_route, t_one = _run_all(grid, route, pop, own, opp, mid)
        feed = {"own": own, "opp": opp, "pop": pop.numpy(),
                "mid": mid.numpy().astype(np.int64)}
        o_grid = sg.run(None, feed)[0]
        o_v1 = sv.run(None, feed)[0]
        pairs = [(names[0], o_grid, t_grid), (names[1], t_grid, t_route),
                 (names[2], t_grid, t_one), (names[3], o_grid, o_v1),
                 (names[4], o_v1, t_route)]
        logit_absmax = max(logit_absmax, float(np.abs(t_one).max()))
        for name, a, b in pairs:
            st = agg[name]
            st[0] += int((a != b).sum())
            st[1] += a.size
            st[2] = max(st[2], float(np.abs(a.astype(np.float64)
                                             - b.astype(np.float64)).max()))
            bad = _amax(a) != _amax(b)
            flip[name] += int(bad.sum())
            if bad.any() and name in (names[1], names[3]):
                srt = np.sort(b[bad], axis=1)
                amax_gap_max = max(amax_gap_max,
                                   float((srt[:, -1] - srt[:, -2]).max()))
        print("  round %d %s" % (r, _cell_stats(mid.numpy())))
    tag = os.path.basename(grid_path)
    print("[%s] A.钉批对拍(全活行) pop=(%d,%d)（%s），%d 轮"
          % (tag, P_CAP, D_TOTAL, THETA_NOTE, rounds))
    for label, name in [("① ONNX网格 vs torch网格", names[0]),
                        ("② torch网格 vs torch v1路由(同数学不同批形)", names[1]),
                        ("③ torch网格 vs 单模型forward_one(金标准)", names[2]),
                        ("④ ONNX网格 vs ONNX v1(fb1024对照)", names[3]),
                        ("⑤ ONNX v1对照 vs torch v1路由", names[4])]:
        st = agg[name]
        print("[%s] %s: 差异 %d/%d 位, max|diff| = %.3g, argmax 一致率 %.6f (%d/%d)"
              % (tag, label, st[0], st[1], st[2], 1.0 - flip[name] / (slots * rounds),
                 slots * rounds - flip[name], slots * rounds))
    print("[%s] |logit|max=%.3g，①相对差=%.2e，②④翻转行top1-top2间隙max=%.3g"
          % (tag, logit_absmax, agg[names[0]][2] / max(logit_absmax, 1e-9),
             amax_gap_max))
    return agg, flip


def parity_scatter(rounds_slots=(1023, 999, 512, 100, 37, 8, 1)):
    """B. 散射-还原专项：动态 S 孪生图（同结构仅 axis0 动态，跑完即删），
    S<P*J 才能出现真·空格子/部分个体不足 J 行；每档随机占用+洗牌（j 跳号）。"""
    import onnxruntime as ort
    tmp = os.path.join(HERE, "_grid_dyn_tmp.onnx")
    _export_grid(64, tmp, dynamic=True)
    sd = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    pop = build_pop()
    grid, route = PopGridMLP().eval(), PopRouteMLP().eval()
    best = {"onnx_vs_torch": [0, 0, 0.0], "torch_vs_v1route": [0, 0, 0.0],
            "torch_vs_one": [0, 0, 0.0], "onnx_vs_one": [0, 0, 0.0]}
    for S in rounds_slots:
        rng = np.random.default_rng(7000 + S)
        own = rng.random((S, 64)).astype(np.float32)
        opp = rng.random((S, 64)).astype(np.float32)
        mid = _sample_mid(S, full=False, rng=rng)
        _check_invariant(mid)
        t_grid, t_route, t_one = _run_all(grid, route, pop, own, opp, mid)
        o_dyn = sd.run(None, {"own": own, "opp": opp, "pop": pop.numpy(),
                              "mid": mid.numpy().astype(np.int64)})[0]
        for key, a, b in [("onnx_vs_torch", o_dyn, t_grid),
                          ("torch_vs_v1route", t_grid, t_route),
                          ("torch_vs_one", t_grid, t_one),
                          ("onnx_vs_one", o_dyn, t_one)]:
            st = best[key]
            st[0] += int((a != b).sum())
            st[1] += a.size
            st[2] = max(st[2], float(np.abs(a.astype(np.float64)
                                             - b.astype(np.float64)).max()))
            assert (_amax(a) == _amax(b)).all(), "散射-还原 argmax 翻转 S=%d" % S
        print("  S=%-4d %s" % (S, _cell_stats(mid.numpy())))
    os.remove(tmp)
    print("[scatter] B.散射-还原专项（动态S孪生图，S<%d 才有空格子）："
          "ONNX vs torch网格 max|diff|=%.3g；torch网格 vs v1路由=%.3g；"
          "torch网格 vs 金标准=%.3g；ONNX vs 金标准=%.3g；argmax 一致率 1.0 全档"
          % (P_CAP * J, best["onnx_vs_torch"][2], best["torch_vs_v1route"][2],
             best["torch_vs_one"][2], best["onnx_vs_one"][2]))
    return best


def parity_dead(slots, grid_path,
                rounds=((200, 50), (513, 3), (1000, 0), (0, 500))):
    """C. 死行鲁棒性（新增）：先造合法全活批取参考，再毒化 -1 行 + 撒 999999
    越界行（含 1000 死行档）。验收：活行输出与无死行版**逐位同**（torch/ONNX 双
    口径 0 位差异）；死/越界行输出可任意（实测恒 0：死块零权重）；同配置复跑
    （含新建 session）全稳定。"""
    import onnxruntime as ort
    pop = build_pop()
    sess = ort.InferenceSession(grid_path, providers=["CPUExecutionProvider"])
    grid = PopGridMLP().eval()
    for r, (n_dead, n_oor) in enumerate(rounds):
        rng = np.random.default_rng(9000 + 100 * r)
        own = rng.random((slots, 64)).astype(np.float32)
        opp = rng.random((slots, 64)).astype(np.float32)
        own_t, opp_t = torch.from_numpy(own), torch.from_numpy(opp)
        mid0 = _sample_mid(slots, full=False, rng=rng)     # 合法全活批（<=J 不变量）
        ref = grid(own_t, opp_t, pop, mid0).numpy()        # 无死行版参考（torch 网格）
        pos = rng.choice(slots, n_dead + n_oor, replace=False)
        mid = mid0.clone()
        mid[torch.from_numpy(pos[:n_dead])] = MID_POISON   # 毒化（int64 全 0xFF）
        mid[torch.from_numpy(pos[n_dead:])] = MID_OOR      # 越界探针值
        _check_invariant(mid)
        live = np.ones(slots, bool)
        live[pos] = False
        feed = {"own": own, "opp": opp, "pop": pop.numpy(), "mid": mid.numpy()}
        out_t = grid(own_t, opp_t, pop, mid).numpy()
        out_o = sess.run(None, feed)[0]
        bt = int((out_t[live] != ref[live]).sum())         # torch 活行逐位差
        bo = int((out_o[live] != ref[live]).sum())         # ONNX 活行逐位差
        dead_zero = bool((out_o[~live] == 0).all()) and bool((out_t[~live] == 0).all())
        reps = [sess.run(None, feed)[0] for _ in range(5)]  # 同 session 复跑 x5
        sess2 = ort.InferenceSession(grid_path, providers=["CPUExecutionProvider"])
        reps.append(sess2.run(None, feed)[0])              # 新建 session 复跑 x1
        stable = all(np.array_equal(o, out_o) for o in reps)
        print("  C%d 死行=%d 越界行=%d 活行=%d %s" %
              (r, n_dead, n_oor, int(live.sum()), _cell_stats(mid.numpy())))
        print("    活行差异 torch %d 位 / ONNX %d 位（应全 0），死行输出恒0=%s，"
              "复跑6次稳定=%s" % (bt, bo, dead_zero, stable))
        assert bt == 0 and bo == 0, "活行被死/越界行污染"
        assert dead_zero and stable
    print("[dead] C.死行鲁棒：全部档位活行输出与无死行版逐位同（torch/ONNX 双口径"
          " 0 位差异），死行输出恒 0（零权重死块），同配置复跑 6 次全稳定")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    grid_path = os.path.join(MODELS_DIR, "othello_pop_grid.fb1024.onnx")
    v1_path = os.path.join(MODELS_DIR, "othello_pop.fb1024.onnx")
    export_pop_grid(S_PIN, grid_path)     # v2.2 网格图（S=1024, P=128+1死块, J=8）
    export_pop_v1(S_PIN, v1_path)         # 对照组：v1 gather 图（不变，仅确认在位）
    parity_pinned(S_PIN, grid_path, v1_path)
    parity_scatter()
    parity_dead(S_PIN, grid_path)


if __name__ == "__main__":
    main()
