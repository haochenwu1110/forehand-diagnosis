"""论文结果图（新数据版，主方法 M0 + PCA90）：图 2–5。

    conda activate d2l
    export PAPERB_UNIFORM_TIER=1
    python paper_figures_v2.py --prep <prep_new> --layer2 <layer2_new> --layer3 <layer3_new_pca90> --out <paper_figs_v2>

输出（每张图 PDF 矢量 / PNG 300 dpi 预览 / TIF 600 dpi 投稿）：
  Fig2_group_differences  (a) 每个通道的 t 值曲线（组间差异簇着色）；(b–f) 五个组差异通道的组均值与范围
  Fig3_tradeoff           (a) 误报 vs 整段 2 倍检出；(b) 五个组水平差异在各参照模型下的排名
  Fig4_priority           每名新手 42 个单元的优先级（点阵，M0 + PCA90 现场重算）
  Fig5_examples           个体诊断示例（默认 N3 左膝屈曲、N5 躯干转动）
受试者编号：A1–A5 → S1–S5；B1、B4、B5、C1、C3 → S6–S10；D1–D5 → N1–N5（与论文表 1 一致）。
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("PAPERB_UNIFORM_TIER", "1")      # 新数据全部按 1 级

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

import analysis_common as AC  # noqa: E402
import layer2_eval as L2  # noqa: E402
import layer3 as L3  # noqa: E402
import localize as LZ  # noqa: E402

for _fp in ("/mnt/c/Windows/Fonts/arial.ttf", "/mnt/c/Windows/Fonts/arialbd.ttf", "/mnt/c/Windows/Fonts/ariali.ttf"):
    if Path(_fp).exists():
        font_manager.fontManager.addfont(_fp)
MM = 1 / 25.4
W = 174 * MM
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "axes.unicode_minus": True,
    "hatch.linewidth": 0.6,
})
INK, INK2, MUTED, GRID = "#1f1f1e", "#52514e", "#8a8984", "#e6e5e0"
SK, NV, GOLD, CC = "#2a5caa", "#e0622a", "#eda100", "#1a9e75"     # 熟练 / 新手 / 区间 / CCAE
CLUSTER, CLUSTER_WEAK, FILL = "#e0622a", "#b9b7b1", "#dcdad4"

SUBJ_ID = {"A1": "S1", "A2": "S2", "A3": "S3", "A4": "S4", "A5": "S5",
           "B1": "S6", "B4": "S7", "B5": "S8", "C1": "S9", "C3": "S10",
           "D1": "N1", "D2": "N2", "D3": "N3", "D4": "N4", "D5": "N5"}
CH_EN = {
    "pelvis_rot_imu": "Pelvis rotation", "thorax_rot_imu": "Thorax rotation",
    "separation_imu": "Thorax–pelvis separation", "trunk_forward_lean": "Trunk forward lean",
    "trunk_lateral_tilt": "Trunk lateral tilt", "shoulder_elev_R": "R shoulder elevation",
    "uarm_forward_R": "R upper-arm forward angle", "elbow_flex_R": "R elbow flexion",
    "shoulder_elev_L": "L shoulder elevation", "elbow_flex_L": "L elbow flexion",
    "hip_flex_L": "L hip flexion", "hip_flex_R": "R hip flexion", "knee_flex_L": "L knee flexion",
    "knee_flex_R": "R knee flexion",
}
# 行顺序按身体部位分组（图 1 小图、图 2、图 4 共用）
REGIONS = [("Trunk rotation", ["pelvis_rot_imu", "thorax_rot_imu", "separation_imu"]),
           ("Trunk posture", ["trunk_forward_lean", "trunk_lateral_tilt"]),
           ("Right arm", ["shoulder_elev_R", "uarm_forward_R", "elbow_flex_R"]),
           ("Left arm", ["shoulder_elev_L", "elbow_flex_L"]),
           ("Lower limbs", ["hip_flex_L", "hip_flex_R", "knee_flex_L", "knee_flex_R"])]
ROW_NAMES = [c for _, cs in REGIONS for c in cs]
PH_EN = ["Backswing", "Forward swing", "Follow-through"]
PH_SHORT = ["B", "F", "Fo"]
METHOD_EN = {"M0_mean": "Mean template", "M1_knn3": "kNN (k = 3)", "M2_pca80": "PCA80", "M2_pca90": "PCA90",
             "M2_pca95": "PCA95", "M2_pca98": "PCA98", "M3_ccae": "CCAE", "M3_ccae_noopt": "CCAE (no latent opt.)"}
METHODS = ("M0_mean", "M2_pca90")


def sid(s):
    return SUBJ_ID.get(str(s), str(s))


def order_channels(ch):
    return [ch.index(c) for c in ROW_NAMES if c in ch]


def clean(ax, grid_y=True):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if grid_y:
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def phase_axis(ax, sb, short=True):
    for b in sb[1:-1]:
        ax.axvline(b - 0.5, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)
    mids = [(sb[i] + sb[i + 1] - 1) / 2 for i in range(len(sb) - 1)]
    ax.set_xticks(mids, ["Backswing", "Forward", "Follow"] if short else PH_EN)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, sb[-1] - 0.5)


def panel_label(ax, s, x=-0.16, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  {name}: pdf / png / tif")


def fmt_p(p):
    return f"p = {p:.3f}".replace("0.", ".") if p >= 0.001 else "p < .001"


# ---------------------------------------------------------------- 共用计算
def group_clusters(d):
    ch = d["channels"]
    subs, M_all = AC.subject_means(d["X"], d["subj"])
    is_nv = np.array([d["group"][d["subj"] == s][0] in AC.NOVICE for g, s in zip(subs, subs)])
    LZ._IS_NV = is_nv
    M_sk, M_nv = M_all[~is_nv], M_all[is_nv]
    tcrit = stats.t.ppf(1 - LZ.ALPHA / 2, len(M_sk) + len(M_nv) - 2)
    T = LZ.t_curve(M_sk, M_nv)
    cl = []
    for c in range(len(ch)):
        null = LZ.channel_null(M_all, int(is_nv.sum()), c, tcrit)
        for a, b, sgn, mass in LZ.clusters(T[c], tcrit):
            sl_ = slice(a, b + 1)
            cl.append(dict(c=c, a=a, b=b, sgn=sgn, p=float(np.mean(null >= mass)),
                           diff=float(M_nv[:, c, sl_].mean() - M_sk[:, c, sl_].mean())))
    return dict(subs=subs, M_all=M_all, is_nv=is_nv, M_sk=M_sk, M_nv=M_nv, T=T, cl=cl, tcrit=tcrit)


def diagnosis(d, layer3_dir, p_min_arg):
    ch, sb = d["channels"], d["seg_bounds"]
    sl = L2.phase_slices(sb)
    p_min = p_min_arg
    abl = Path(layer3_dir) / "ablation.csv" if layer3_dir else None
    if abl is not None and abl.exists():
        t = pd.read_csv(abl)
        p_min = float(t.loc[t["config"] == "full", "P_MIN"].iloc[0])
    cfg = dict(name="full", methods=METHODS, use_tier=True, use_cons=True)
    refs = L3.build_references(d, METHODS, sl)
    X, subj, grp = d["X"], d["subj"], d["group"]
    nv = sorted(np.unique(subj[np.isin(grp, AC.NOVICE)]))
    pe = {s: L3.person_features(refs["full"].score(X[subj == s], sl), refs["base_novice"]) for s in nv}
    cells = {s: L3.cell_candidates(pe[s], ch, cfg) for s in nv}
    finds = {s: L3.build_findings(pe[s], ch, cfg, p_min, sl, sb) for s in nv}
    return dict(sl=sl, p_min=p_min, cfg=cfg, nv=nv, pe=pe, cells=cells, finds=finds)


# ---------------------------------------------------------------- 图 2
def fig2(d, G, out, p_cut=0.1):
    ch, sb = d["channels"], d["seg_bounds"]
    order = order_channels(ch)
    T, cl = G["T"], G["cl"]
    n = len(order)
    fig = plt.figure(figsize=(W, 190 * MM))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.3, 1, 1], hspace=0.5, wspace=0.38,
                          left=0.17, right=0.92, top=0.95, bottom=0.06)
    ax = fig.add_subplot(gs[0, :])
    tmax = float(np.abs(T).max())
    scale = 0.75 / tmax                             # 每行高 1，曲线最大占 0.75 行（簇所在行可略压到相邻行）
    x = np.arange(T.shape[1])
    ax.axvspan(sb[1] - 0.5, sb[2] - 0.5, color="#f3f5fa", lw=0, zorder=0)
    for b in sb[1:-1]:
        ax.axvline(b - 0.5, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    for i, c in enumerate(order):
        y0 = -i
        y = y0 + T[c] * scale
        ax.plot(x, y, color="#6d6b66", lw=0.6, zorder=3)
        ax.fill_between(x, y0, y, color=FILL, lw=0, zorder=2)
        for r in [r for r in cl if r["c"] == c]:
            xs = np.arange(r["a"], r["b"] + 1)
            strong = r["p"] < p_cut
            ax.fill_between(xs, y0, y[xs], color=CLUSTER if strong else CLUSTER_WEAK, lw=0, zorder=4)
            if strong:
                ax.plot(xs, y[xs], color="#a8410f", lw=0.7, zorder=5)
                right = r["b"] < T.shape[1] * 0.75
                xt = r["b"] + 2 if right else r["a"] - 2
                ax.text(xt, y0 - 0.02, f"{r['diff']:+.0f}°   {fmt_p(r['p'])}".replace("-", "−"),
                        ha="left" if right else "right", va="center", fontsize=6, color="#a8410f",
                        fontweight="bold", zorder=6,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
    ax.set_yticks([-i for i in range(n)], [CH_EN[ch[c]] for c in order])
    ax.tick_params(axis="y", length=0)
    k = 0
    for name, cs in REGIONS:
        m = len([c for c in cs if c in ch])
        if k > 0:
            ax.axhline(-k + 0.5, color=GRID, lw=0.6, zorder=1)
        k += m
    ax.set_ylim(-n + 0.2, 0.6)
    ax.set_xlim(-0.5, T.shape[1] - 0.5)
    ax.set_xticks([(sb[i] + sb[i + 1] - 1) / 2 for i in range(3)], PH_EN)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, colors=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # 比例尺：t = ±5
    xb = T.shape[1] + 2
    yb = -n + 1.5
    ax.plot([xb, xb], [yb - 5 * scale, yb + 5 * scale], color=INK2, lw=0.8, clip_on=False)
    for yy in (yb - 5 * scale, yb, yb + 5 * scale):
        ax.plot([xb - 1, xb + 1], [yy, yy], color=INK2, lw=0.8, clip_on=False)
    ax.text(xb + 2, yb + 5 * scale, "t = +5", fontsize=5.5, va="center", color=INK2, clip_on=False)
    ax.text(xb + 2, yb - 5 * scale, "t = −5", fontsize=5.5, va="center", color=INK2, clip_on=False)
    panel_label(ax, "a", x=-0.2, y=1.0)
    h = [Patch(fc=CLUSTER, label=f"Cluster, within-channel p < {p_cut:g}"),
         Patch(fc=CLUSTER_WEAK, label=f"Cluster, p ≥ {p_cut:g}"),
         Patch(fc=FILL, label="t (novice − skilled), participant means")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False)

    # (b–f) 五个组差异通道
    key = sorted([r for r in cl if r["p"] < p_cut], key=lambda r: r["p"])
    M_all, is_nv = G["M_all"], G["is_nv"]
    xx = np.arange(M_all.shape[2])
    for k, r in enumerate(key[:5]):
        a2 = fig.add_subplot(gs[1 + k // 3, k % 3])
        c = r["c"]
        for grp_mask, col in ((~is_nv, SK), (is_nv, NV)):
            M = M_all[grp_mask][:, c]
            a2.fill_between(xx, M.min(0), M.max(0), color=col, alpha=0.13, lw=0)
            a2.plot(xx, M.mean(0), color=col, lw=1.3)
        a2.axvspan(r["a"] - 0.5, r["b"] + 0.5, color=GOLD, alpha=0.22, lw=0)
        xs = np.arange(r["a"], r["b"] + 1)
        a2.fill_between(xs, M_all[~is_nv][:, c, xs].mean(0), M_all[is_nv][:, c, xs].mean(0),
                        facecolor="none", edgecolor="#a8410f", hatch="////", lw=0, alpha=0.7)
        a2.text(0.5 * (r["a"] + r["b"]), 1.0, f"{r['diff']:+.0f}°".replace("-", "−"),
                transform=a2.get_xaxis_transform(), ha="center", va="top", fontsize=8, fontweight="bold",
                color="#a8410f")
        phase_axis(a2, sb)
        a2.set_title(CH_EN[ch[c]], fontsize=7, pad=8)
        if k % 3 == 0:
            a2.set_ylabel("Angle (°)")
        clean(a2)
        panel_label(a2, "bcdef"[k], x=-0.3, y=1.08)
    lax = fig.add_subplot(gs[2, 2])
    lax.axis("off")
    lax.legend(handles=[Line2D([], [], color=SK, lw=1.3, label="Skilled: mean of 10 players"),
                        Patch(fc=SK, alpha=0.2, label="Skilled: range of player means"),
                        Line2D([], [], color=NV, lw=1.3, label="Novice: mean of 5 players"),
                        Patch(fc=NV, alpha=0.2, label="Novice: range of player means"),
                        Patch(fc=GOLD, alpha=0.35, label=f"Cluster (p < {p_cut:g})"),
                        Patch(fc="none", ec="#a8410f", hatch="////", lw=0, label="Gap between group means")],
               loc="center", frameon=False, handlelength=1.8)
    save(fig, out, "Fig2_group_differences")
    return key


# ---------------------------------------------------------------- 图 3
TARGETS = [("trunk_forward_lean/backswing", "Trunk forward lean · backswing", "o", "#2a5caa"),
           ("knee_flex_L/backswing", "L knee flexion · backswing", "s", "#8c564b"),
           ("shoulder_elev_L/backswing", "L shoulder elevation · backswing", "^", "#9467bd"),
           ("thorax_rot_imu/follow", "Thorax rotation · follow-through", "D", "#1f9e89"),
           ("pelvis_rot_imu/follow", "Pelvis rotation · follow-through", "v", "#6d6b66")]


def fig3(layer2_dir, out):
    summ = pd.read_csv(Path(layer2_dir) / "summary.csv", index_col=0)
    gc = pd.read_csv(Path(layer2_dir) / "group_consistency.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 74 * MM), gridspec_kw=dict(width_ratios=[1, 1.05]))
    fa, det = summ["mean_score"], summ["broad_p3@2"] * 100
    pca = ["M2_pca80", "M2_pca90", "M2_pca95", "M2_pca98"]
    blues = {"M2_pca80": "#9db7de", "M2_pca90": "#2a5caa", "M2_pca95": "#5f86c4", "M2_pca98": "#1b3c73"}
    a1.plot(fa[pca], det[pca], color="#c9d6ec", lw=2.0, zorder=1, solid_capstyle="round")
    off = {"M2_pca80": ((6, 3), "left"), "M2_pca90": ((-7, -3), "right"), "M2_pca95": ((6, 3), "left"),
           "M2_pca98": ((6, 2), "left")}
    for m in pca:
        main = m == "M2_pca90"
        a1.scatter(fa[m], det[m], s=48 if main else 26, color=blues[m], edgecolor="white", lw=0.8, zorder=3)
        o, ha = off[m]
        a1.annotate("PCA90 (main)" if main else METHOD_EN[m], (fa[m], det[m]), xytext=o,
                    textcoords="offset points", ha=ha, va="center", fontsize=6.5, color=blues[m],
                    fontweight="bold" if main else "normal")
    a1.scatter(fa["M0_mean"], det["M0_mean"], marker="s", s=26, color="#4a4844", zorder=3)
    a1.annotate("Mean template", (fa["M0_mean"], det["M0_mean"]), xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.5, color="#4a4844")
    a1.scatter(fa["M1_knn3"], det["M1_knn3"], marker="^", s=24, color="#9a9893", zorder=3)
    a1.annotate("kNN (k = 3)", (fa["M1_knn3"], det["M1_knn3"]), xytext=(5, 2), textcoords="offset points",
                ha="left", va="center", fontsize=6.5, color="#7a7873")
    a1.scatter(fa["M3_ccae"], det["M3_ccae"], marker="D", s=26, color=CC, zorder=4)
    a1.annotate("CCAE", (fa["M3_ccae"], det["M3_ccae"]), xytext=(0, -7), textcoords="offset points",
                ha="center", va="top", fontsize=6.5, color=CC, fontweight="bold")
    a1.scatter(fa["M3_ccae_noopt"], det["M3_ccae_noopt"], marker="D", s=26, facecolor="white", edgecolor=CC,
               lw=1.0, zorder=4)
    a1.annotate("", xy=(fa["M3_ccae"] + 0.008, det["M3_ccae"] - 0.25),
                xytext=(fa["M3_ccae_noopt"] - 0.008, det["M3_ccae_noopt"] + 0.1),
                arrowprops=dict(arrowstyle="->", color=CC, lw=0.8))
    a1.text(0.5 * (fa["M3_ccae"] + fa["M3_ccae_noopt"]), det["M3_ccae_noopt"] - 2.2, "latent optimisation",
            ha="center", va="top", fontsize=6, color=CC)
    a1.annotate("better", xy=(0.47, 76), xytext=(0.53, 71.5), fontsize=6, color=MUTED, ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
    a1.set_xlim(0.44, 0.95)
    a1.set_ylim(30, 78)
    a1.set_xlabel("False alarm: held-out skilled mean cell score (lower is better)")
    a1.set_ylabel("Injected 2-SD whole-phase deviation in top 3 (%)")
    clean(a1)
    a1.grid(axis="x", color=GRID, lw=0.5)
    panel_label(a1, "a", x=-0.17, y=1.02)

    meths = ["M0_mean", "M1_knn3", "M2_pca80", "M2_pca90", "M2_pca95", "M2_pca98", "M3_ccae"]
    xs = np.arange(len(meths))
    a2.axvspan(3.5, 5.5, color="#f1f0ec", lw=0, zorder=0)
    a2.text(4.5, 21.3, "group differences\nabsorbed as ‘style’", ha="center", va="center", fontsize=6,
            color=INK2, style="italic")
    for tg, lab, mk, col in TARGETS:
        q = gc[gc["target"] == tg].set_index("method")["rank"]
        a2.plot(xs, [q.get(m, np.nan) for m in meths], marker=mk, ms=3.6, lw=1.0, color=col, label=lab,
                markeredgecolor="white", markeredgewidth=0.5)
    a2.set_xticks(xs, ["Mean\ntemplate", "kNN\n(k = 3)", "PCA80", "PCA90", "PCA95", "PCA98", "CCAE"])
    for t in a2.get_xticklabels():
        if t.get_text() == "PCA90":
            t.set_fontweight("bold")
    a2.set_ylim(24, 0)
    a2.set_yticks([1, 5, 10, 15, 20])
    a2.set_ylabel("Rank among 42 cells (1 = top)")
    a2.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=5.8, handlelength=1.6)
    clean(a2)
    panel_label(a2, "b", x=-0.15, y=1.02)
    fig.tight_layout(w_pad=2.5)
    save(fig, out, "Fig3_tradeoff")


# ---------------------------------------------------------------- 图 4
def fig4(d, G, D, out):
    ch, sb = d["channels"], d["seg_bounds"]
    order = order_channels(ch)
    nv, p_min = D["nv"], D["p_min"]
    P = len(PH_SHORT)
    # 与组水平簇重叠的单元
    overlap = set()
    for r in [r for r in G["cl"] if r["p"] < 0.1]:
        for p in range(P):
            if r["a"] < sb[p + 1] and r["b"] >= sb[p]:
                overlap.add((r["c"], p))
    reported = {s: {(c["c"], c["p"]) for f in D["finds"][s] for c in f["cells"]} for s in nv}
    fig = plt.figure(figsize=(W, 112 * MM))
    ax = fig.add_axes([0.2, 0.06, 0.6, 0.84])
    gap = 1.2
    xpos = lambda k, p: k * (P + gap) + p  # noqa: E731
    pmax = max(r["priority"] for s in nv for r in D["cells"][s])
    size = lambda v: 6 + 170 * min(v, 3.2) / 3.2  # noqa: E731
    for i in range(len(order)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f7f6f3", lw=0, zorder=0)
    for k, s in enumerate(nv):
        for (c, p) in overlap:
            ax.add_patch(Rectangle((xpos(k, p) - 0.5, order.index(c) - 0.5), 1, 1, color=GOLD, alpha=0.25, lw=0,
                                   zorder=1))
        got = {(r["c"], r["p"]): r for r in D["cells"][s]}
        for i, c in enumerate(order):
            for p in range(P):
                r = got.get((c, p))
                x = xpos(k, p)
                if r is None:
                    ax.scatter(x, i, s=1.5, color="#bdbbb5", zorder=2)
                    continue
                v = r["priority"]
                if v >= p_min:
                    ax.scatter(x, i, s=size(v), color=NV, edgecolor="#8a3209", lw=0.6, zorder=3)
                    lab = f"{v:.2f}"
                    if size(v) > 70:
                        ax.text(x, i, lab, ha="center", va="center", fontsize=4.6, color="white",
                                fontweight="bold", zorder=4)
                    else:
                        ax.text(x, i + 0.42, lab, ha="center", va="top", fontsize=4.8, color="#8a3209", zorder=4)
                elif r["agreement"] == "both":
                    ax.scatter(x, i, s=size(v), color="#f2b393", lw=0, zorder=3)
                else:
                    ax.scatter(x, i, s=size(v), facecolor="none", edgecolor=MUTED, lw=0.6, zorder=3)
        x0, x1 = xpos(k, 0) - 0.4, xpos(k, P - 1) + 0.4
        ax.plot([x0, x1], [-1.05, -1.05], color=NV, lw=0.9, clip_on=False)
        ax.text((x0 + x1) / 2, -1.55, sid(s), ha="center", va="bottom", fontsize=8, fontweight="bold", color=NV,
                clip_on=False)
        for p in range(P):
            ax.text(xpos(k, p), -0.75, PH_SHORT[p], ha="center", va="bottom", fontsize=6, color=INK2, clip_on=False)
    ax.set_yticks(range(len(order)), [CH_EN[ch[c]] for c in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])
    ax.set_xlim(-0.7, xpos(len(nv) - 1, P - 1) + 0.7)
    ax.set_ylim(len(order) - 0.5, -0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    k = 0
    for name, cs in REGIONS:
        m = len([c for c in cs if c in ch])
        if k > 0:
            ax.axhline(k - 0.5, color=GRID, lw=0.6, zorder=1)
        ax.text(-0.31, k + (m - 1) / 2, name, transform=ax.get_yaxis_transform(), rotation=90, ha="center",
                va="center", fontsize=6, color=INK2)
        k += m
    # 图例
    lx = fig.add_axes([0.82, 0.2, 0.17, 0.66])
    lx.axis("off")
    lx.set_xlim(0, 1)
    lx.set_ylim(0, 1)
    lx.text(0.0, 0.98, "Priority P", fontsize=7, fontweight="bold", va="top")
    for j, v in enumerate([p_min, 1.5, 3.0]):
        y = 0.86 - j * 0.085
        lx.scatter(0.12, y, s=size(v), color=NV, edgecolor="#8a3209", lw=0.6)
        lx.text(0.32, y, f"{v:.2f}" if j == 0 else f"{v:.1f}", va="center", fontsize=6)
    items = [("rep", f"reported\n(P ≥ {p_min:.2f})"), ("both", "below threshold,\nboth references"),
             ("one", "below threshold,\none reference only"), ("ovl", "overlaps a group-\nlevel cluster")]
    for j, (kind, lab) in enumerate(items):
        y = 0.55 - j * 0.12
        if kind == "rep":
            lx.scatter(0.12, y, s=size(1.0), color=NV, edgecolor="#8a3209", lw=0.6)
        elif kind == "both":
            lx.scatter(0.12, y, s=size(1.0), color="#f2b393", lw=0)
        elif kind == "one":
            lx.scatter(0.12, y, s=size(1.0), facecolor="none", edgecolor=MUTED, lw=0.6)
        else:
            lx.add_patch(Rectangle((0.03, y - 0.035), 0.18, 0.07, color=GOLD, alpha=0.3, lw=0))
        lx.text(0.32, y, lab, va="center", fontsize=6)
    fig.text(0.5, 0.012, "B = backswing   F = forward swing   Fo = follow-through", ha="center", fontsize=6,
             color=INK2)
    save(fig, out, "Fig4_priority")
    return {sid(s): [(CH_EN[f["channel"]], [PH_SHORT[p] for p in f["phases"]], round(f["priority"], 2),
                      round(f["rep"]["resid"], 1)) for f in D["finds"][s]] for s in nv}


# ---------------------------------------------------------------- 图 5
def _to_t(pt, sb):
    ph, pct = pt
    i = {"backswing": 0, "forward": 1, "follow": 2}[ph]
    return sb[i] + pct / 100 * (sb[i + 1] - sb[i])


def fig5(d, D, out, cases=(("N3", "knee_flex_L"), ("N5", "thorax_rot_imu"))):
    ch, sb = d["channels"], d["seg_bounds"]
    X, subj, grp = d["X"], d["subj"], d["group"]
    _inv = {v: k for k, v in SUBJ_ID.items()}          # 公开数据用 N3，内部数据用 D3，两种都能用
    cases = tuple((s if s in set(subj) else _inv.get(s, s), c) for s, c in cases)
    sk = np.isin(grp, AC.SKILLED)
    _, M_sk = AC.subject_means(X[sk], subj[sk])
    fig, axes = plt.subplots(1, len(cases), figsize=(W, 66 * MM))
    x = np.arange(X.shape[2])
    for k, ((s, cname), ax) in enumerate(zip(cases, axes)):
        c = ch.index(cname)
        tmpl = M_sk[:, c].mean(0)
        ax.fill_between(x, M_sk[:, c].min(0), M_sk[:, c].max(0), color=SK, alpha=0.12, lw=0.6,
                        edgecolor="#b8c6e3")
        ax.plot(x, tmpl, color=SK, lw=1.3, ls=(0, (4, 2)))
        Xs = X[subj == s][:, c]
        mu, sd = Xs.mean(0), Xs.std(0)
        ax.fill_between(x, mu - sd, mu + sd, color=NV, alpha=0.2, lw=0)
        ax.plot(x, mu, color=NV, lw=1.5)
        rank, f = next((i + 1, f) for i, f in enumerate(D["finds"][s]) if f["channel"] == cname)
        a, b = _to_t(f["range"]["start"], sb), _to_t(f["range"]["end"], sb)
        ax.axvspan(a, b, color=GOLD, alpha=0.22, lw=0)
        p = f["rep"]["p"]
        xs = np.arange(sb[p], sb[p + 1])
        ax.fill_between(xs, tmpl[xs], mu[xs], facecolor="none", edgecolor="#a8410f", hatch="////", lw=0,
                        alpha=0.75)
        phase_axis(ax, sb)
        lo = min((M_sk[:, c].min(0)).min(), (mu - sd).min())
        hi = max((M_sk[:, c].max(0)).max(), (mu + sd).max())
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.3 * (hi - lo))
        mag = f"{f['rep']['resid']:+.0f}°".replace("-", "−")
        xt, ha = (0.5 * (sb[p] + sb[p + 1]), "center") if p < 2 else (sb[p + 1] - 1, "right")
        ax.text(xt, 0.97, mag, transform=ax.get_xaxis_transform(), ha=ha, va="top", fontsize=10,
                fontweight="bold", color="#a8410f")
        ax.text(xt, 0.85, f"mean gap, {PH_EN[p].lower()}", transform=ax.get_xaxis_transform(), ha=ha,
                va="top", fontsize=6, color="#a8410f")
        ax.set_title(f"{sid(s)} · {CH_EN[cname]}", loc="left", fontsize=7.5, fontweight="bold", pad=14)
        ax.text(0, 1.02, f"issue {rank} · {f['rep']['consistency']:.0%} of strokes", transform=ax.transAxes,
                fontsize=6, color=INK2, va="bottom")
        ax.set_ylabel("Angle (°)")
        clean(ax)
        panel_label(ax, "ab"[k], x=-0.14, y=1.1)
    h = [Patch(fc=SK, alpha=0.2, ec="#b8c6e3", label="Skilled range (player means)"),
         Line2D([], [], color=SK, lw=1.3, ls=(0, (4, 2)), label="Mean template"),
         Patch(fc=NV, alpha=0.3, label="Novice mean ± SD"),
         Patch(fc=GOLD, alpha=0.35, label="Reported interval (half-maximum)"),
         Patch(fc="none", ec="#a8410f", hatch="////", lw=0, label="Gap used for the degrees")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=3)
    save(fig, out, "Fig5_examples")


def main(a):
    out = Path(a.out)
    d, _ = AC.load(a.prep)
    print("组水平差异 …")
    G = group_clusters(d)
    print("图 2 …")
    key = fig2(d, G, out)
    print("    p < 0.1 的簇：" + "; ".join(f"{d['channels'][r['c']]} [{r['a']}–{r['b']}] {r['diff']:+.1f}° p={r['p']:.4f}"
                                    for r in key))
    if a.layer2:
        print("图 3 …")
        fig3(a.layer2, out)
    print(f"图 4 / 5：拟合 {METHODS} 并诊断 …")
    D = diagnosis(d, a.layer3, a.p_min)
    lists = fig4(d, G, D, out)
    print(f"    阈值 P_min = {D['p_min']:.2f}")
    for s, fs in lists.items():
        print(f"    {s}: " + "; ".join(f"{n} {'+'.join(ph)} {pr} ({deg:+.1f}°)" for n, ph, pr, deg in fs))
    fig5(d, D, out)
    print(f"完成：{out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--layer2", default=None)
    ap.add_argument("--layer3", default=None, help="读 ablation.csv 里的阈值（M0 + PCA90 那次）")
    ap.add_argument("--out", default="paper_figs_v2")
    ap.add_argument("--p_min", type=float, default=0.85)
    main(ap.parse_args())
