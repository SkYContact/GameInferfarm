# -*- coding: utf-8 -*-
"""尖刺第 1 步：09 的 BC 检查点（flat 41,280 参数）→ inferfarm 钉批 ONNX。
输入 own[slots,64]/opp[slots,64] → policy[slots,64]；权重=bc_t3k 过拟合终版（真起点）。
用法：python export_bc_onnx.py   （在 runs/40_engine_battle 下跑，q35 环境）"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RUN40 = os.path.dirname(HERE)
sys.path.insert(0, RUN40)

from model import LAYERS, PARAM_SLICES  # noqa: E402


class FlatMLP(torch.nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.W = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()
        for (ms, vs), (o, i) in zip(PARAM_SLICES, LAYERS):
            self.W.append(torch.nn.Parameter(theta[ms[0]:ms[1]].reshape(o, i).clone()))
            self.b.append(torch.nn.Parameter(theta[vs[0]:vs[1]].clone()))

    def forward(self, own, opp):
        x = torch.cat([own, opp], dim=1)
        h = torch.relu(x @ self.W[0].T + self.b[0])
        h = torch.relu(h @ self.W[1].T + self.b[1])
        return h @ self.W[2].T + self.b[2]


def export(slots, out):
    m = FlatMLP(torch.load(os.path.join(RUN40, "bc_out", "bc_t3k", "theta_final.pt"),
                           map_location="cpu")).eval()
    own = torch.zeros(slots, 64)
    opp = torch.zeros(slots, 64)
    try:
        torch.onnx.export(m, (own, opp), out, input_names=["own", "opp"],
                          output_names=["policy"], opset_version=17, dynamo=False)
    except TypeError:
        torch.onnx.export(m, (own, opp), out, input_names=["own", "opp"],
                          output_names=["policy"], opset_version=17)
    import onnx
    mm = onnx.load(out)
    dims = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
            for i in mm.graph.input}
    assert all(v[0] == slots for v in dims.values()), dims
    print("export %s (%.2f MB) dims=%s" % (out, os.path.getsize(out) / 1e6, dims))


if __name__ == "__main__":
    export(16, r"D:\inferfarm\models\othello_bc_mlp.fb16.onnx")
    export(1, r"D:\inferfarm\models\othello_bc_mlp.fb1.onnx")
