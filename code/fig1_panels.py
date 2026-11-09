"""图 1 用的真实数据小图：N1 一拍 x、残差 r、典型变异幅度 s_c、风格基线 B、平均单元分数 S、超额分数 E、优先级 P。

    conda activate d2l
    python fig1_panels.py --prep <prep_out_v31> --layer3 <layer3_m3> --out <fig1_panels>

输出（每个小图：透明背景 SVG / PDF 供拼图，PNG 预览，CSV 原始数值）：
  x_stroke      N1 一拍，14 个角度 × 125 点（每行按本行范围归一化，只看形状）
  r_residual    这一拍减去 CCAE 重建（度，对称色标）
  s_c           熟练者每个角度的典型变异幅度（度，14 × 1）
  B_baseline    熟练者留一基线（14 × 3）
  S_score       N1 的平均单元分数（14 × 3）
  E_excess      超额分数 E = S − B（14 × 3，负值显示为白色）
  P_priority    最终优先级（M0 + CCAE 合并，过阈值的格子加黑框）
  check_sheet   带坐标、色标的核对总图（只用来核对，不放进论文）
B、S、E 用同一个色标上限，从 S 到 E 颜色变浅就是"减掉风格"的效果。
行顺序与图 2、图 4 相同（按可信度分级）；金框 = 右肩抬高 × 随挥那一格。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import paper_figures as PF          # 复用样式、字体、通道名、行顺序
import analysis_common as AC
import layer2_eval as L2
import layer3 as L3

plt = PF.plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

MM = PF.MM
GOLD = "#eda100"
CMAP_SK = LinearSegmentedColormap.from_list("sk", ["#ffffff", "#c9dcf4", PF.SK, "#123f78"])
CMAP_NV = LinearSegmentedColormap.from_list("nv", ["#ffffff", "#fbd3bd", PF.NV, "#8a3209"])
# 残差的正负和组别无关，用中性的青–紫，避免和"蓝 = 熟练、橙 = 新手"混淆
CMAP_DIV = LinearSegmentedColormap.from_list("div", ["#1f7a7a", "#9fd0cc", "#ffffff", "#d7b5d8", "#7b3294"])
PH_SHORT = ["B", "F", "Fo"]


def build_novice_refs(d, methods, sl):
    """只拟合评估新手需要的参照：全部熟练者 + 每次留一人（共 6 次），比 layer3.build_references 快。"""
    X, subj, grp = d["X"], d["subj"], d["group"]
    sk = np.isin(grp, AC.SKILLED)
    sk_subs = AC.natsorted(np.unique(subj[sk]))

    def ref_without(excl):
        m = sk & ~np.isin(subj, list(excl))
        return L3.Reference(X[m], subj[m], methods)

    full = ref_without(())
    maps = []
    for g in sk_subs:
        sc = ref_without((g,)).score(X[subj == g], sl)
        maps.append({mn: sc[mn][0].mean(0) for mn in methods})
    base = {mn: np.mean([m_[mn] for m_ in maps], axis=0) for mn in methods}
    return full, base


def pick_stroke(Xs, c, sl_follow, mode):
    if mode != "auto":
        return int(mode)
    # 默认：右肩随挥段平均值最接近 N1 自己中位数的那一拍（典型拍，不挑极端）
    v = Xs[:, c, sl_follow].mean(1)
    return int(np.argmin(np.abs(v - np.median(v))))


def mini(name, arr, cmap, norm, out, size_mm, rows, hl=None, vlines=(), boxes=()):
    """无坐标轴的小图，透明背景。hl = (行, 列起, 列止) 金框；boxes = [(行, 列)] 黑框。"""
    fig = plt.figure(figsize=(size_mm[0] * MM, size_mm[1] * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    n_r, n_c = arr.shape
    for x in vlines:
        ax.axvline(x - 0.5, color="white", lw=0.8)
    if n_c <= 3:                                   # 单元格网格线
        for i in range(1, n_r):
            ax.axhline(i - 0.5, color="white", lw=0.4)
        for j in range(1, n_c):
            ax.axvline(j - 0.5, color="white", lw=0.4)
    for r, c in boxes:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=PF.INK, lw=1.0))
    if hl is not None:
        r, c0, c1 = hl
        ax.add_patch(Rectangle((c0 - 0.5, r - 0.5), c1 - c0, 1, fill=False, ec=GOLD, lw=1.4))
    ax.set_axis_off()
    for ext in ("svg", "pdf"):
        fig.savefig(out / f"{name}.{ext}", transparent=True)
    fig.savefig(out / f"{name}.png", dpi=600, transparent=True)
    plt.close(fig)
    pd.DataFrame(arr, index=rows).to_csv(out / f"{name}.csv", float_format="%.4f")
    print(f"  {name}: svg / pdf / png / csv")


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d, _ = AC.load(a.prep)
    ch, sb = d["channels"], d["seg_bounds"]
    sl = L2.phase_slices(sb)
    methods = a.methods.split(",")
    m_rec = methods[-1]                             # 画残差 / S / E 用的方法（默认 CCAE）
    p_min = a.p_min
    abl = Path(a.layer3) / "ablation.csv" if a.layer3 else None
    if abl is not None and abl.exists():
        t = pd.read_csv(abl)
        p_min = float(t.loc[t["config"] == "full", "P_MIN"].iloc[0])

    print(f"拟合参照 {methods}（6 次，GPU 上约几分钟）……")
    full, base = build_novice_refs(d, tuple(methods), sl)

    order = PF.order_channels(ch)
    rows = [PF.ch_en(ch[c]) for c in order]
    cols3 = PF.PH_EN
    X, subj = d["X"], d["subj"]
    Xs = X[subj == a.subject]
    c_hl = ch.index(a.hl_channel)
    r_hl = order.index(c_hl)
    p_hl = PF.PH_KEY[a.hl_phase]

    # ---- 单拍 x 与残差 r
    k = pick_stroke(Xs, c_hl, sl[p_hl], a.stroke)
    x1 = Xs[k:k + 1]
    xh = full.models[m_rec].reconstruct(x1)[0]
    x1 = x1[0]
    r = x1 - xh
    print(f"  {PF.sid(a.subject)} 共 {len(Xs)} 拍，选第 {k} 拍（0 起）；用 --stroke N 可换")
    lo, hi = x1.min(1, keepdims=True), x1.max(1, keepdims=True)
    xn = (x1 - lo) / np.maximum(hi - lo, 1e-6)
    hl_t = (r_hl, int(sb[p_hl]), int(sb[p_hl + 1]))
    vb = [int(v) for v in sb[1:-1]]
    mini("x_stroke", xn[order], CMAP_NV, Normalize(0, 1), out, (34, 24), rows, hl_t, vb)
    pd.DataFrame(x1[order], index=rows).to_csv(out / "x_stroke_degrees.csv", float_format="%.3f")
    pd.DataFrame(xh[order], index=rows).to_csv(out / "x_hat_reconstruction_degrees.csv", float_format="%.3f")
    rmax = float(np.percentile(np.abs(r), 99))
    mini("r_residual", r[order], CMAP_DIV, TwoSlopeNorm(0, -rmax, rmax), out, (34, 24), rows, hl_t, vb)

    # ---- s_c、B、S、E
    s_c = full.s[order][:, None]
    mini("s_c", s_c, CMAP_SK, Normalize(0, float(s_c.max())), out, (4, 24), rows)
    sc = full.score(Xs, sl)[m_rec]
    S = sc[0].mean(0)
    B = base[m_rec]
    E = S - B
    vmax = float(max(S.max(), B.max()))
    hl_c = (r_hl, p_hl, p_hl + 1)
    mini("B_baseline", B[order], CMAP_SK, Normalize(0, vmax), out, (10, 24), rows, hl_c)
    mini("S_score", S[order], CMAP_NV, Normalize(0, vmax), out, (10, 24), rows, hl_c)
    mini("E_excess", np.clip(E[order], 0, None), CMAP_NV, Normalize(0, vmax), out, (10, 24), rows, hl_c)
    pd.DataFrame(E[order], index=rows, columns=cols3).to_csv(out / "E_excess_signed.csv", float_format="%.4f")

    # ---- P：M0 + CCAE 合并后的优先级（与图 4、表 5 同一口径）
    cfg = dict(name="full", methods=tuple(methods), use_tier=True, use_cons=True)
    pe = L3.person_features(full.score(Xs, sl), base)
    P = np.zeros((len(ch), 3))
    for row in L3.cell_candidates(pe, ch, cfg):
        P[row["c"], row["p"]] = row["priority"]
    Po = P[order]
    above = [(i, j) for i in range(len(order)) for j in range(3) if Po[i, j] >= p_min]
    mini("P_priority", Po, CMAP_NV, Normalize(0, float(max(Po.max(), p_min))), out, (10, 24), rows, hl_c,
         boxes=above)

    # ---- 漏斗计数（图 1 上可以直接标）
    n_S = int((S > L2.TAU).sum())
    n_E = int((E > 0).sum())
    print(f"\n计数（{PF.sid(a.subject)}，{L3.short_name(m_rec)}）：42 格中 S > {L2.TAU:g} 的 {n_S} 格；"
          f"E > 0 的 {n_E} 格；P ≥ {p_min:.2f} 的 {len(above)} 格（合并相邻阶段后即问题清单条数）。\n"
          f"  注意 S > 1 和 E > 0 是两种口径，不是逐级筛选，图上建议只标 '42 cells → 问题条数'")
    print("过阈值的格子：" + "; ".join(f"{rows[i]} / {PH_SHORT[j]} = {Po[i, j]:.2f}" for i, j in above))

    # ---- 核对总图
    fig, axes = plt.subplots(1, 7, figsize=(PF.W_DOUBLE, 70 * MM),
                             gridspec_kw=dict(width_ratios=[6, 6, 0.6, 1.6, 1.6, 1.6, 1.6]))
    items = [("x (row-normalised)", xn[order], CMAP_NV, Normalize(0, 1)),
             ("r = x − x̂ (°)", r[order], CMAP_DIV, TwoSlopeNorm(0, -rmax, rmax)),
             ("s_c (°)", s_c, CMAP_SK, Normalize(0, float(s_c.max()))),
             ("B", B[order], CMAP_SK, Normalize(0, vmax)),
             ("S", S[order], CMAP_NV, Normalize(0, vmax)),
             ("E = S − B", np.clip(E[order], 0, None), CMAP_NV, Normalize(0, vmax)),
             ("P", Po, CMAP_NV, Normalize(0, float(max(Po.max(), p_min))))]
    for i, (ax, (tt, arr, cm, nm)) in enumerate(zip(axes, items)):
        im = ax.imshow(arr, cmap=cm, norm=nm, aspect="auto", interpolation="nearest")
        ax.set_title(tt, fontsize=6)
        ax.set_yticks(range(len(rows)), rows if i == 0 else [""] * len(rows), fontsize=5)
        if arr.shape[1] == 3:
            ax.set_xticks(range(3), PH_SHORT, fontsize=5)
            ax.add_patch(Rectangle((p_hl - 0.5, r_hl - 0.5), 1, 1, fill=False, ec=GOLD, lw=1.2))
        else:
            ax.set_xticks([])
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05)
        cb.ax.tick_params(labelsize=4.5)
    for i, j in above:
        axes[-1].add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=PF.INK, lw=1.0))
    fig.suptitle(f"{PF.sid(a.subject)}, stroke {k}; reference {L3.short_name(m_rec)}; P_min = {p_min:.2f}",
                 fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "check_sheet.png", dpi=300)
    plt.close(fig)
    print(f"  check_sheet.png（核对用）\n完成：{out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--layer3", default=None, help="读 ablation.csv 里的阈值")
    ap.add_argument("--out", default="fig1_panels")
    ap.add_argument("--subject", default="N1")
    ap.add_argument("--stroke", default="auto", help="第几拍（0 起）；auto = 右肩随挥最典型的一拍")
    ap.add_argument("--methods", default="M0_mean,M3_ccae")
    ap.add_argument("--p_min", type=float, default=0.35)
    ap.add_argument("--hl_channel", default="shoulder_elev_R")
    ap.add_argument("--hl_phase", default="follow", choices=["backswing", "forward", "follow"])
    main(ap.parse_args())
