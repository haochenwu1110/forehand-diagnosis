"""论文结果图（英文，投稿用）：图 2–5。

    conda activate d2l
    python paper_figures.py --prep ../../prep_out_v31 --step1 ../../step1_out \
        --layer2 <加了 M3 那次第 2 层的输出目录> --layer3 ../../layer3_m3 --out ../../paper_figs

输出（每张图三种格式：TIFF 600 dpi 供投稿上传、PDF 矢量、PNG 预览）：
  Fig2_group_differences   组间差异：t 值热图 + 三个差异通道的每人均值曲线
  Fig3_tradeoff            参照模型权衡：误报 vs 整段 2SD 检出；三处上肢差异的排名
  Fig4_priority            每名新手 42 个单元的优先级（需要重新拟合 M0 + CCAE，GPU 上几分钟）
  Fig5_examples            个体诊断示例：N2 骨盆转角、N1 右肩抬高（新手均值 vs 熟练参照）

只想快速出前三张 / 不想重新拟合 CCAE：加 --skip_fig4。
【易错-90】图 4 的单元优先级必须和 layer3_m3 用同一套参照（M0_mean + M3_ccae），
  阈值从 layer3_m3/ablation.csv 读取，读不到时用 --p_min（默认 0.35）。
【易错-91】受试者编号：公开数据已经是 S1–S10、N1–N5，直接使用。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

import analysis_common as AC  # noqa: E402
import config as CFG  # noqa: E402
import localize as LZ  # noqa: E402

# ---------------------------------------------------------------- 样式
# WSL 里没有 Arial，直接借用 Windows 的字体文件（和 layer3 借中文字体同一个办法）
from matplotlib import font_manager  # noqa: E402
for _fp in ("/mnt/c/Windows/Fonts/arial.ttf", "/mnt/c/Windows/Fonts/arialbd.ttf",
            "/mnt/c/Windows/Fonts/ariali.ttf", "C:/Windows/Fonts/arial.ttf"):
    if Path(_fp).exists():
        font_manager.fontManager.addfont(_fp)
MM = 1 / 25.4
W_DOUBLE, W_SINGLE = 174 * MM, 84 * MM
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "axes.unicode_minus": True,
})
INK, INK2, MUTED, GRID = "#1f1f1e", "#52514e", "#8a8984", "#e6e5e0"
SK, NV = "#2a78d6", "#eb6834"                                     # skilled / novice
AGREE_COL = {"both": "#2a78d6", "M0_only": "#eb6834", "CCAE_only": "#1baf7a"}
AGREE_LAB = {"both": "Both references agree", "M0_only": "Mean template only (×0.5)",
             "CCAE_only": "CCAE only (×0.5)"}

SUBJ_ID = {}
CH_EN = {
    "pelvis_rot_imu": "Pelvis rotation", "thorax_rot_imu": "Thorax rotation",
    "separation_imu": "Thorax–pelvis separation", "shoulder_elev_R": "R shoulder elevation",
    "uarm_forward_R": "R upper-arm forward angle", "elbow_flex_R": "R elbow flexion",
    "trunk_forward_lean": "Trunk forward lean", "trunk_lateral_tilt": "Trunk lateral tilt",
    "hip_flex_L": "L hip flexion", "hip_flex_R": "R hip flexion", "knee_flex_L": "L knee flexion",
    "knee_flex_R": "R knee flexion", "shoulder_elev_L": "L shoulder elevation", "elbow_flex_L": "L elbow flexion",
}
PH_EN = ["Backswing", "Forward swing", "Follow-through"]
PH_KEY = {"backswing": 0, "forward": 1, "follow": 2}
METHOD_EN = {"M0_mean": "Mean template", "M1_knn1": "kNN (k = 1)", "M1_knn3": "kNN (k = 3)",
             "M1_knn5": "kNN (k = 5)", "M1_knn10": "kNN (k = 10)", "M2_pca80": "PCA80", "M2_pca90": "PCA90",
             "M2_pca95": "PCA95", "M2_pca98": "PCA98", "M3_ccae": "CCAE",
             "M3_ccae_noopt": "CCAE (no latent opt.)"}


def sid(s):
    return SUBJ_ID.get(str(s), str(s))


def ch_en(c):
    return CH_EN.get(c, c)


def clean(ax, grid_y=True):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if grid_y:
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def phase_axis(ax, sb, labels=True, short=False):
    for b in sb[1:-1]:
        ax.axvline(b - 0.5, color=MUTED, lw=0.6, ls="--", zorder=0)
    if labels:
        mids = [(sb[i] + sb[i + 1] - 1) / 2 for i in range(len(sb) - 1)]
        names = (["Back-\nswing", "Forward\nswing", "Follow-\nthrough"] if short
                 else ["Backswing", "Forward swing", "Follow-\nthrough"])
        ax.set_xticks(mids, names)
        ax.tick_params(axis="x", length=0, labelsize=6 if short else 6.5)


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  {name}: pdf / png / tif")


def order_channels(ch):
    return sorted(range(len(ch)), key=lambda i: (CFG.CHANNEL_TIER.get(ch[i], 3), i))


# ---------------------------------------------------------------- 图 2：组间差异
def fig2(d, out, p_cut=0.1):
    ch, sb = d["channels"], d["seg_bounds"]
    subs, M_all = AC.subject_means(d["X"], d["subj"])
    groups = [d["group"][d["subj"] == s][0] for s in subs]
    is_nv = np.array([g in AC.NOVICE for g in groups])
    M_sk, M_nv = M_all[~is_nv], M_all[is_nv]
    tcrit = stats.t.ppf(1 - LZ.ALPHA / 2, len(M_sk) + len(M_nv) - 2)
    T = LZ.t_curve(M_sk, M_nv)
    cl = []
    for c in range(len(ch)):
        null = LZ.channel_null(M_all, int(is_nv.sum()), c, tcrit)
        for a, b, sgn, mass in LZ.clusters(T[c], tcrit):
            cl.append(dict(c=c, a=a, b=b, sgn=sgn, p=float(np.mean(null >= mass))))
    keyc = [r for r in cl if r["p"] < p_cut]
    order = order_channels(ch)
    row_of = {c: i for i, c in enumerate(order)}

    fig = plt.figure(figsize=(W_DOUBLE, 140 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.42, wspace=0.32,
                          left=0.21, right=0.93, top=0.95, bottom=0.08)
    ax = fig.add_subplot(gs[0, :])
    cmap = LinearSegmentedColormap.from_list("div", [SK, "#f4f3ef", "#e34948"])
    lim = 6
    im = ax.imshow(np.clip(T[order], -lim, lim), aspect="auto", cmap=cmap, vmin=-lim, vmax=lim,
                   interpolation="nearest")
    for r in cl:
        strong = r["p"] < p_cut
        ax.add_patch(Rectangle((r["a"] - 0.5, row_of[r["c"]] - 0.5), r["b"] - r["a"] + 1, 1, fill=False,
                               ec=INK, lw=1.3 if strong else 0.5, ls="-" if strong else ":"))
    phase_axis(ax, sb)
    ax.set_yticks(range(len(ch)), [ch_en(ch[c]) for c in order])
    tiers = [CFG.CHANNEL_TIER.get(ch[c], 3) for c in order]
    for i in range(1, len(order)):
        if tiers[i] != tiers[i - 1]:
            ax.axhline(i - 0.5, color=INK2, lw=0.6)
    for t in (1, 2, 3):
        rows = [i for i, x in enumerate(tiers) if x == t]
        if rows:
            ax.text(1.005, np.mean(rows), f"Tier {t}", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=6.5, color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.07)
    cb.set_label("t (novice − skilled)")
    cb.ax.tick_params(labelsize=6)
    ax.set_title("a", loc="left", fontweight="bold", fontsize=9, x=-0.2)

    keych = []
    for r in sorted(keyc, key=lambda r: r["p"]):
        if r["c"] not in keych:
            keych.append(r["c"])
    x = np.arange(M_all.shape[2])
    for k, c in enumerate(keych[:3]):
        a2 = fig.add_subplot(gs[1, k])
        for m, v in zip(M_all, is_nv):
            a2.plot(x, m[c], color=NV if v else SK, lw=1.0 if v else 0.8, alpha=0.95 if v else 0.75,
                    ls="-" if v else "--")
        for r in keyc:
            if r["c"] == c:
                a2.axvspan(r["a"] - 0.5, r["b"] + 0.5, color="#eda100", alpha=0.18, lw=0)
        phase_axis(a2, sb, short=True)
        a2.set_title(ch_en(ch[c]), fontsize=7)
        a2.set_ylabel("Angle (°)" if k == 0 else "")
        clean(a2)
        a2.set_title("bcd"[k], loc="left", fontweight="bold", fontsize=9, x=-0.25)
    h = [plt.Line2D([], [], color=SK, ls="--", lw=0.8, label="Skilled (one line per participant)"),
         plt.Line2D([], [], color=NV, lw=1.0, label="Novice (one line per participant)"),
         Patch(fc="#eda100", alpha=0.3, label=f"Cluster, within-channel p < {p_cut:g}")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 0.0))
    save(fig, out, "Fig2_group_differences")
    return [dict(channel=ch[r["c"]], start=r["a"], end=r["b"], p=r["p"]) for r in keyc]


# ---------------------------------------------------------------- 图 3：权衡
# 标签位置（点的像素偏移, 对齐）：CCAE、PCA90、kNN 挤在一起，分别放上、下、右
LABEL_OFF = {"M3_ccae": ((0, 7), "center"), "M3_ccae_noopt": ((7, -2), "left"), "M2_pca90": ((-4, -10), "right"),
             "M1_knn3": ((5, -8), "left")}
def fig3(layer2_dir, out):
    summ = pd.read_csv(Path(layer2_dir) / "summary.csv", index_col=0)
    summ = summ[[not (m.startswith("M1_") and m != "M1_knn3") for m in summ.index]]   # 与表 4 一致，只画 k = 3
    gc_path = Path(layer2_dir) / "group_consistency.csv"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W_DOUBLE, 72 * MM), gridspec_kw=dict(width_ratios=[1.1, 1]))
    fa, det = summ["mean_score"], summ["broad_p3@2"] * 100
    pca = [m for m in ["M2_pca80", "M2_pca90", "M2_pca95", "M2_pca98"] if m in summ.index]
    a1.plot(fa[pca], det[pca], color=MUTED, lw=1.0, zorder=1)
    style = {"M0_mean": ("s", NV), "M3_ccae": ("D", "#1baf7a"), "M3_ccae_noopt": ("D", "white")}
    for m in summ.index:
        mk, col = style.get(m, ("o", SK) if m.startswith("M2") else ("^", "#6250d6"))
        ec = "#1baf7a" if m == "M3_ccae_noopt" else "white"
        a1.scatter(fa[m], det[m], s=26, marker=mk, color=col, edgecolor=ec, lw=0.8, zorder=3)
        off, ha = LABEL_OFF.get(m, ((4, 3), "left"))
        a1.annotate(METHOD_EN.get(m, m), (fa[m], det[m]), xytext=off, textcoords="offset points", fontsize=6, ha=ha)
    a1.set_xlim(fa.min() - 0.04, fa.max() + 0.12)
    a1.set_xlabel("False alarm: held-out skilled mean cell score (lower is better)")
    a1.set_ylabel("Top-3 hit rate, whole-phase 2 SD (%)")
    clean(a1)
    a1.grid(axis="x", color=GRID, lw=0.5)
    a1.set_title("a", loc="left", fontweight="bold", fontsize=9, x=-0.15)

    if gc_path.exists():
        gc = pd.read_csv(gc_path)
        targets = [("shoulder_elev_R/follow", "R shoulder elevation, follow-through", "o"),
                   ("uarm_forward_R/follow", "R upper-arm forward angle, follow-through", "s"),
                   ("uarm_forward_R/forward", "R upper-arm forward angle, forward swing", "^")]
        meths = [m for m in ["M0_mean", "M1_knn3", "M2_pca80", "M2_pca90", "M2_pca95", "M2_pca98", "M3_ccae"]
                 if m in set(gc["method"])]
        xs = np.arange(len(meths))
        cols = [SK, NV, "#6250d6"]
        for (tg, lab, mk), col in zip(targets, cols):
            q = gc[gc["target"] == tg].set_index("method")["rank"]
            if len(q):
                a2.plot(xs, [q.get(m, np.nan) for m in meths], marker=mk, ms=4, lw=0.9, color=col, label=lab)
        a2.set_xticks(xs, [METHOD_EN.get(m, m) for m in meths], rotation=35, ha="right", rotation_mode="anchor")
        a2.set_ylim(28, 0)
        a2.set_yticks([1, 5, 10, 15, 20])
        a2.set_ylabel("Rank among 42 cells (1 = top)")
        a2.legend(frameon=False, loc="lower left")
        clean(a2)
    a2.set_title("b", loc="left", fontweight="bold", fontsize=9, x=-0.15)
    fig.tight_layout()
    save(fig, out, "Fig3_tradeoff")


# ---------------------------------------------------------------- 图 4：单元优先级（需要拟合 M0 + CCAE）
def fig4(d, layer3_dir, step1_dir, out, p_min_arg, methods):
    import layer2_eval as L2
    import layer3 as L3
    ch, sb = d["channels"], d["seg_bounds"]
    sl = L2.phase_slices(sb)
    cfg = dict(name="full", methods=tuple(methods), use_tier=True, use_cons=True)
    p_min = p_min_arg
    abl = Path(layer3_dir) / "ablation.csv" if layer3_dir else None
    if abl is not None and abl.exists():
        t = pd.read_csv(abl)
        p_min = float(t.loc[t["config"] == "full", "P_MIN"].iloc[0])
    print(f"  图 4：拟合参照 {methods}（阈值 {p_min:.2f}）……")
    refs = L3.build_references(d, tuple(methods), sl)
    X, subj, grp = d["X"], d["subj"], d["group"]
    nv = sorted(np.unique(subj[np.isin(grp, AC.NOVICE)]), key=sid)
    targets = L3.load_group_targets(step1_dir, ch)
    order = order_channels(ch)
    P = len(L2.PHASES)
    alt = f"{L3.short_name(methods[-1])}_only"
    colors = {"both": AGREE_COL["both"], "M0_only": AGREE_COL["M0_only"], alt: AGREE_COL["CCAE_only"]}
    labels = {"both": AGREE_LAB["both"], "M0_only": AGREE_LAB["M0_only"],
              alt: f"{L3.short_name(methods[-1])} only (×0.5)"}
    fig, axes = plt.subplots(len(nv), 1, figsize=(W_DOUBLE, (32 * len(nv) + 30) * MM), sharex=True, squeeze=False)
    ymax = 0
    for ax, s in zip(axes[:, 0], nv):
        pe = L3.person_features(refs["full"].score(X[subj == s], sl), refs["base_novice"])
        for r in L3.cell_candidates(pe, ch, cfg):
            xpos = order.index(r["c"]) * (P + 1) + r["p"]
            ax.bar(xpos, r["priority"], width=0.85, color=colors.get(r["agreement"], MUTED),
                   alpha=1.0 if r["priority"] >= p_min else 0.3, hatch="////" if r["tier"] == 3 else None,
                   edgecolor="white", lw=0)
            ymax = max(ymax, r["priority"])
        ax.axhline(p_min, color=INK, lw=0.7, ls=(0, (3, 2)))
        for tg in targets:
            for p in tg["phases"]:
                ax.plot(order.index(ch.index(tg["channel"])) * (P + 1) + p, 0, marker="*", ms=6, color=INK,
                        clip_on=False, zorder=5)
        ax.set_ylabel(f"{sid(s)}\npriority")
        clean(ax)
    for ax in axes[:, 0]:
        ax.set_ylim(0, ymax * 1.08)
    axes[-1, 0].set_xticks([i * (P + 1) + 1 for i in range(len(order))],
                           [ch_en(ch[c]) for c in order], rotation=40, ha="right", rotation_mode="anchor")
    axes[-1, 0].set_xlim(-1, len(order) * (P + 1) - 1)
    h = [Patch(fc=v, label=labels[k]) for k, v in colors.items()]
    h += [Patch(fc="white", ec=INK2, hatch="////", lw=0.5, label="Tier 3 (×0.5)"),
          plt.Line2D([], [], color=INK, ls=(0, (3, 2)), lw=0.7, label=f"Threshold {p_min:.2f}"),
          plt.Line2D([], [], marker="*", color=INK, ls="", ms=6, label="Group-level difference")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "Each joint: three bars = backswing, forward swing, follow-through; pale bars are below "
             "threshold", ha="center", fontsize=6, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    save(fig, out, "Fig4_priority")


# ---------------------------------------------------------------- 图 5：个体诊断示例
def _interval(fd, sb):
    def to_t(pt):
        ph, pct = pt
        i = PH_KEY[ph]
        return sb[i] + pct / 100 * (sb[i + 1] - sb[i])
    return to_t(fd["range_pct"]["start"]), to_t(fd["range_pct"]["end"])


def fig5(d, layer3_dir, out, cases=(("N2", "pelvis_rot_imu"), ("N1", "shoulder_elev_R"))):
    ch, sb = d["channels"], d["seg_bounds"]
    X, subj, grp = d["X"], d["subj"], d["group"]
    sk = np.isin(grp, AC.SKILLED)
    sk_subs, M_sk = AC.subject_means(X[sk], subj[sk])
    fig, axes = plt.subplots(1, len(cases), figsize=(W_DOUBLE, 62 * MM), squeeze=False)
    x = np.arange(X.shape[2])
    for k, ((s, c_name), ax) in enumerate(zip(cases, axes[0])):
        c = ch.index(c_name)
        ax.fill_between(x, M_sk[:, c].min(0), M_sk[:, c].max(0), color=SK, alpha=0.15, lw=0,
                        label="Skilled range (participant means)")
        ax.plot(x, M_sk[:, c].mean(0), color=SK, lw=1.2, ls="--", label="Mean template")
        Xs = X[subj == s][:, c]
        ax.fill_between(x, Xs.mean(0) - Xs.std(0), Xs.mean(0) + Xs.std(0), color=NV, alpha=0.18, lw=0)
        ax.plot(x, Xs.mean(0), color=NV, lw=1.3, label=f"Novice mean ± SD")
        fd = None
        jf = Path(layer3_dir) / f"findings_{s}.json" if layer3_dir else None
        if jf is not None and jf.exists():
            js = json.loads(jf.read_text(encoding="utf-8"))
            fd = next((f for f in js["findings"] if f["channel"] == c_name), None)
        if fd:
            a, b = _interval(fd, sb)
            ax.axvspan(a, b, color="#eda100", alpha=0.2, lw=0, label="Reported interval (half-maximum)")
            ax.text(0.02, 0.96, f"{sid(s)}, issue {fd['rank']}: {fd['magnitude_deg']:+.0f}°".replace("-", "−") + ", "
                    f"{fd['consistency']:.0%} of strokes, tier {fd['tier']}", transform=ax.transAxes,
                    va="top", fontsize=6.5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.18 * (hi - lo))  # headroom for the annotation box
        phase_axis(ax, sb)
        ax.set_title(f"{sid(s)} — {ch_en(c_name)}", fontsize=7)
        ax.set_ylabel("Angle (°)")
        clean(ax)
        ax.set_title("ab"[k], loc="left", fontweight="bold", fontsize=9, x=-0.15)
    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "Fig5_examples")


def main(a):
    out = Path(a.out)
    d, _ = AC.load(a.prep)
    print("图 2 …")
    keyc = fig2(d, out)
    print("    p < 0.1 的簇：" + "; ".join(f"{k['channel']} [{k['start']}–{k['end']}] p={k['p']:.3f}" for k in keyc))
    if a.layer2:
        print("图 3 …")
        fig3(a.layer2, out)
    if not a.skip_fig4:
        print("图 4 …")
        fig4(d, a.layer3, a.step1, out, a.p_min, a.methods.split(","))
    print("图 5 …")
    fig5(d, a.layer3, out)
    print(f"完成：{out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--step1", default=None)
    ap.add_argument("--layer2", default=None)
    ap.add_argument("--layer3", default=None)
    ap.add_argument("--out", default="paper_figs")
    ap.add_argument("--p_min", type=float, default=0.35)
    ap.add_argument("--methods", default="M0_mean,M3_ccae")
    ap.add_argument("--skip_fig4", action="store_true")
    main(ap.parse_args())
