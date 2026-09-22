# -*- coding: utf-8 -*-
# fig_bench.py —— 基准配图（谱系阶梯 + 胜率），matplotlib 微软雅黑朴素风
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# ---- 图 1：实现谱系吞吐阶梯（bench_full.log 2026-09-22，4 腿中位）----
labels = ["python 串行", "python 向量化\n(64环境同步)", "C++ 线程\n+逐次推理",
          "C++ fiber池\n+逐次推理", "C++ 推理农场\n(银行攒批)"]
gps = [19.4, 26.2, 200.7, 200.1, 815.5]
rel = [1.0, 1.3, 10.3, 10.3, 42.0]
colors = ["#8fa3b8", "#8fa3b8", "#5b7d99", "#5b7d99", "#c0504d"]

fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
bars = ax.bar(range(5), gps, color=colors, width=0.62, zorder=3)
ax.set_ylim(0, 950)
ax.set_xticks(range(5))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("对局吞吐（局/秒）", fontsize=10)
ax.set_title("同一五子棋负载、同一模型（6.9M 参数 CNN）：五种实现形态的吞吐",
             fontsize=11)
ax.grid(axis="y", ls="--", alpha=0.4, zorder=0)
for i, (b, g, r) in enumerate(zip(bars, gps, rel)):
    x = b.get_x() + b.get_width() / 2
    ax.text(x, g + 46, "%.1f 局/s" % g,
            ha="center", fontsize=10, fontweight="bold")
    ax.text(x, g + 14, "（%.1f×）" % r,
            ha="center", fontsize=8.5, color="#555555")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bench_ladder.png"), bbox_inches="tight")
plt.close(fig)

# ---- 图 2：模型价值（胜率，同一规则对手）----
fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=150)
cats = ["未训练参考模型\n(种子权重 MLP)", "行为克隆训练后\n(6.9M CNN，本基准模型)"]
vals = [0.0, 85.9]
bars = ax.bar(cats, vals, color=["#8fa3b8", "#c0504d"], width=0.45, zorder=3)
ax.set_ylim(0, 100)
ax.set_ylabel("对内置规则对手胜率（%）", fontsize=10)
ax.set_title("同一个农场，模型会不会下棋是另一回事：训练前 vs 训练后", fontsize=11)
ax.grid(axis="y", ls="--", alpha=0.4, zorder=0)
for b, v, txt in zip(bars, vals, ["0/16", "1100/1280（先手 87.2% / 后手 84.7%）"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 3, txt, ha="center", fontsize=9.5,
            fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bench_winrate.png"), bbox_inches="tight")
plt.close(fig)

print("figs ->", os.path.abspath(OUT))
