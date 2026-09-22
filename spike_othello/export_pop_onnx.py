# -*- coding: utf-8 -*-
"""尖刺第 3 步（负载 A / 候选设计 1"逐行权重路由"）：种群 ONNX 导出 + 对拍。
路由模块 forward_pop(own, opp, pop, mid)：pop [P,41280]=整个种群，mid [S] int64=
每行个体号；逐层 Gather(pop, mid) 切出每行权重，bmm 逐行前向（每行用自己那套
权重）→ policy [S,64]。权重是运行时输入而非常量 → 一次 build 全代热换。
产出 D:\\inferfarm\\models\\othello_pop.fb{128,16}.onnx（P=128 容量钉死，S=批形钉死），
内置对拍（全部 CPU EP，避免 CUDA 末位差噪音）：
  ① ONNX vs torch 路由 forward_pop（同算子同形，期望逐位同）
  ② ONNX vs torch 单模型 forward_one（mm 口径；bmm/mm kernel 不同，末位差预期）
  ③ torch 路由 forward_pop vs 单模型 forward_one（bmm vs mm 口径，REPORT 先例）
  附 torch 路由 vs model.forward_pop 全种群对角（ES 参考实现，baddbmm 口径）
用法：C:/Users/41601/Miniconda3/envs/q35/python.exe export_pop_onnx.py"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RUN40 = r"C:\word\水文工厂\runs\40_engine_battle"  # model.py / bc 检查点原仓库
sys.path.insert(0, RUN40)

from model import (LAYERS, PARAM_SLICES, D_TOTAL, init_theta,  # noqa: E402
                   forward_one, forward_pop as ref_forward_pop)

THETA_PT = os.path.join(RUN40, "bc_out", "bc_t3k", "theta_final.pt")
MODELS_DIR = r"D:\inferfarm\models"
P_CAP = 128  # 种群容量：两档模型同钉 P=128（assert 兜底）

assert D_TOTAL == 41280 and sum(o * i + o for o, i in LAYERS) == D_TOTAL


class PopRouteMLP(torch.nn.Module):
    """逐行权重路由：own/opp [S,64] + pop [P,D] + mid [S](int64) -> policy [S,64]。
    行 s 用第 mid[s] 个体的权重：index_select(pop,0,mid) 切行（ONNX=Gather）→
    逐层切片 reshape 出 [S,o,i] 权重，bmm 逐行算（每行各用各的权重）。"""

    def forward(self, own, opp, pop, mid):
        rows = torch.index_select(pop, 0, mid)                # [S, D]
        h = torch.cat([own, opp], dim=1).unsqueeze(1)         # [S, 1, 128]
        last = len(LAYERS) - 1
        for k, ((ms, vs), (o, i)) in enumerate(zip(PARAM_SLICES, LAYERS)):
            w = rows[:, ms[0]:ms[1]].reshape(-1, o, i).transpose(1, 2)  # [S, i, o]
            b = rows[:, vs[0]:vs[1]].reshape(-1, 1, o)                  # [S, 1, o]
            h = torch.bmm(h, w) + b                           # 逐行前向
            if k < last:
                h = torch.relu(h)
        return h.squeeze(1)                                   # [S, 64]


def export_pop(slots, out):
    m = PopRouteMLP().eval()
    own = torch.zeros(slots, 64)
    opp = torch.zeros(slots, 64)
    pop = torch.zeros(P_CAP, D_TOTAL)
    mid = torch.arange(slots, dtype=torch.int64) % P_CAP
    try:
        torch.onnx.export(m, (own, opp, pop, mid), out,
                          input_names=["own", "opp", "pop", "mid"],
                          output_names=["policy"], opset_version=17, dynamo=False)
    except TypeError:
        torch.onnx.export(m, (own, opp, pop, mid), out,
                          input_names=["own", "opp", "pop", "mid"],
                          output_names=["policy"], opset_version=17)
    import onnx
    mm = onnx.load(out)
    ins = {i.name: i for i in mm.graph.input}
    assert list(ins) == ["own", "opp", "pop", "mid"], list(ins)  # 输入名与次序
    dims = {n: [d.dim_value for d in i.type.tensor_type.shape.dim]
            for n, i in ins.items()}
    elem = {n: i.type.tensor_type.elem_type for n, i in ins.items()}
    assert dims["own"] == [slots, 64] and dims["opp"] == [slots, 64], dims
    assert dims["pop"] == [P_CAP, D_TOTAL], dims              # P 钉死=容量
    assert dims["mid"] == [slots], dims                       # S 钉死=批形
    assert elem["mid"] == onnx.TensorProto.INT64, elem        # mid 必须 int64
    odim = [d.dim_value for d in mm.graph.output[0].type.tensor_type.shape.dim]
    assert mm.graph.output[0].name == "policy" and odim == [slots, 64], odim
    print("export %s (%.2f MB) dims=%s mid=INT64" %
          (out, os.path.getsize(out) / 1e6, dims))


def build_pop():
    """0 号个体=bc_t3k 真终版权重，其余随机 init_theta —— 真检查点也过路由。"""
    pop = torch.empty(P_CAP, D_TOTAL)
    pop[0] = torch.load(THETA_PT, map_location="cpu")
    for i in range(1, P_CAP):
        pop[i] = init_theta(9000 + i)
    return pop


def _amax(logits):
    """平局取小格号（与 parity_check.py 口径一致）。"""
    return logits.argmax(axis=1)


def parity(slots, onnx_path, rounds=4):
    """三个对拍（+1 附检），全部 CPU EP。返回 {名: [差异位数, 总位数, max|diff|]}。"""
    import onnxruntime as ort
    pop = build_pop()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    route = PopRouteMLP()
    names = ["p1_onnx_vs_route", "p2_onnx_vs_one", "p3_route_vs_one",
             "p0_route_vs_refpop_diag"]
    agg = {n: [0, 0, 0.0] for n in names}
    amax_flip = 0          # ① ONNX vs 单模型 argmax 翻转数
    amax_gap_max = 0.0     # 翻转行的 top1-top2 间隙（应都是近平局）
    logit_absmax = 0.0     # 幅度量级（末位差的分母）
    for r in range(rounds):
        rng = np.random.default_rng(1000 * slots + r)
        own = rng.random((slots, 64)).astype(np.float32)
        opp = rng.random((slots, 64)).astype(np.float32)
        g = torch.Generator().manual_seed(2000 * slots + r)
        mid = torch.randint(0, P_CAP, (slots,), generator=g, dtype=torch.int64)
        if r == 0:
            mid[0] = 0  # 0 号=bc 终版权重
        x = np.concatenate([own, opp], axis=1)
        t_route = route(torch.from_numpy(own), torch.from_numpy(opp),
                        pop, mid).numpy()
        t_one = np.stack([forward_one(pop[m], torch.from_numpy(x[s:s + 1])).numpy()[0]
                          for s, m in enumerate(mid.tolist())])
        t_diag = ref_forward_pop(pop, torch.from_numpy(x))[mid, torch.arange(slots)].numpy()
        o = sess.run(None, {"own": own, "opp": opp, "pop": pop.numpy(),
                            "mid": mid.numpy().astype(np.int64)})[0]
        for name, a, b in [(names[0], o, t_route), (names[1], o, t_one),
                           (names[2], t_route, t_one), (names[3], t_route, t_diag)]:
            st = agg[name]
            st[0] += int((a != b).sum())
            st[1] += a.size
            st[2] = max(st[2], float(np.abs(a.astype(np.float64)
                                             - b.astype(np.float64)).max()))
        # 下游只吃 argmax：①的末位差是否翻转决策 + 翻转行是否近平局
        logit_absmax = max(logit_absmax, float(np.abs(t_one).max()))
        a1, a2 = _amax(o), _amax(t_one)
        bad = a1 != a2
        amax_flip += int(bad.sum())
        if bad.any():
            srt = np.sort(t_one[bad], axis=1)
            amax_gap_max = max(amax_gap_max,
                               float((srt[:, -1] - srt[:, -2]).max()))
    tag = os.path.basename(onnx_path)
    p1 = agg[names[0]]
    print("[%s] ①ONNX vs torch路由 max|diff| = %.3g %s"
          % (tag, p1[2], "(逐位同)" if p1[0] == 0 else "(有差异!)"))
    for label, name in [("②ONNX vs torch单模型(mm)", names[1]),
                        ("③torch路由(bmm) vs 单模型(mm)", names[2]),
                        ("附 torch路由 vs model.forward_pop对角(baddbmm)", names[3])]:
        st = agg[name]
        print("[%s] %s: 差异 %d/%d 位, max|diff| = %.3g"
              % (tag, label, st[0], st[1], st[2]))
    print("[%s] 口径 argmax: ①ONNX vs 单模型翻转 %d/%d 行, 翻转行top1-top2间隙max=%.3g"
          " (|logit|max=%.3g, 相对末位差=%.2e)"
          % (tag, amax_flip, slots * rounds, amax_gap_max, logit_absmax,
             p1[2] / max(logit_absmax, 1e-9)))
    return agg


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    export_pop(128, os.path.join(MODELS_DIR, "othello_pop.fb128.onnx"))
    export_pop(16, os.path.join(MODELS_DIR, "othello_pop.fb16.onnx"))
    parity(128, os.path.join(MODELS_DIR, "othello_pop.fb128.onnx"))
    parity(16, os.path.join(MODELS_DIR, "othello_pop.fb16.onnx"))


if __name__ == "__main__":
    main()
