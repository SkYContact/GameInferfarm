# -*- coding: utf-8 -*-
# fig_bench.py —— 基准配图（8192 局新口径终版），matplotlib 微软雅黑朴素风
# 数据出处：bench_data/rerank_8192.log（谱系四档同口径重跑）+ bench_handoff.md
# （冠军 4 跑中位 2303 / 组数扫描 780-1376-1600-2303）+ 8192 局胜率 7012/8192
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# ---- 图 1：实现谱系吞吐阶梯（8192 局同口径，linear）----
labels = ["python 串行", "python 向量化\n(256环境同步)", "C++ 每链线程\n+逐次推理",
          "C++ 推理农场\n(6组银行攒批)"]
gps = [19.8, 25.9, 190.2, 2300]
rel = [1.0, 1.3, 9.6, 116.0]
colors = ["#8fa3b8", "#8fa3b8", "#5b7d99", "#c0504d"]

fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
bars = ax.bar(range(4), gps, color=colors, width=0.56, zorder=3)
ax.set_ylim(0, 2680)
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("对局吞吐（局/秒）", fontsize=10)
ax.set_title("同一五子棋负载、同一模型（6.85M 参数 CNN）：四种实现的吞吐",
             fontsize=11)
ax.grid(axis="y", ls="--", alpha=0.4, zorder=0)
for b, g, r in zip(bars, gps, rel):
    x = b.get_x() + b.get_width() / 2
    tag = ("%d 局/s" % g) if g >= 1000 else ("%.1f 局/s" % g)
    ax.text(x, g + 128, tag, ha="center", fontsize=10.5,
            fontweight="bold", color="#c0504d" if r > 50 else "#222222")
    ax.text(x, g + 38, "（%.1f×）" % r, ha="center", fontsize=8.5, color="#555555")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bench_ladder.png"), bbox_inches="tight")
plt.close(fig)

# ---- 图 2：同一张卡拆组数扩展（第二填充管线）----
groups = [1, 2, 4, 6]
g_gps = [780, 1376, 1600, 2303]
g_rel = [1.0, 1.77, 2.05, 2.95]

fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=150)
ax.plot(groups, g_gps, "-o", color="#c0504d", lw=2, ms=7, zorder=3)
ax.set_xticks(groups)
ax.set_xticklabels(["%d 组" % g for g in groups], fontsize=10)
ax.set_ylim(0, 2650)
ax.set_xlim(0.3, 7.5)   # 最右点标注右置溢出边框——右侧留 1.5 组宽边距
ax.set_xlabel("同一张 GPU 上拆分的设备组数", fontsize=10)
ax.set_ylabel("对局吞吐（局/秒）", fontsize=10)
ax.set_title("不需要第二张卡：同一张 GPU 拆成多组并行填充管线", fontsize=11)
ax.grid(axis="y", ls="--", alpha=0.4, zorder=0)
for gx, gy, gr in zip(groups, g_gps, g_rel):
    ax.annotate("%.1f×" % gr if gr > 1 else "1 组基线", (gx, gy),
                textcoords="offset points", xytext=(8, 10), fontsize=9.5,
                color="#555555")
    ax.annotate("%d" % gy, (gx, gy), textcoords="offset points", xytext=(8, -16),
                fontsize=10, fontweight="bold", color="#c0504d")
ax.text(0.03, 0.92, "8 组以上反降（调度台过载）", transform=ax.transAxes,
        fontsize=8.5, color="#888888")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bench_groups.png"), bbox_inches="tight")
plt.close(fig)

# ---- 图 3：模型价值（胜率，同一规则对手，8192 局）----
fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=150)
cats = ["未训练参考模型\n(种子权重 MLP)", "行为克隆训练后\n(6.85M CNN，本基准模型)"]
vals = [0.0, 85.6]
bars = ax.bar(cats, vals, color=["#8fa3b8", "#c0504d"], width=0.45, zorder=3)
ax.set_ylim(0, 100)
ax.set_ylabel("对内置规则对手胜率（%）", fontsize=10)
ax.set_title("同一个农场，模型会不会下棋是另一回事：训练前 vs 训练后", fontsize=11)
ax.grid(axis="y", ls="--", alpha=0.4, zorder=0)
for b, v, txt in zip(bars, vals,
                     ["0/16", "7012/8192（先手 88.4% / 后手 82.9%）"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 3, txt, ha="center", fontsize=9.5,
            fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bench_winrate.png"), bbox_inches="tight")
plt.close(fig)

print("figs ->", os.path.abspath(OUT))
