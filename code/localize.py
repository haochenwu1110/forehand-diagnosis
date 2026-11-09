"""第 1 步：熟练者与新手的关节角差异定位（以人为单位，纯统计，不训练模型）。

    python localize.py --prep_dir <预处理输出目录> --out_dir step1_out

做法：
  1. 每人先把自己所有拍平均成一条曲线（每个关节 × 125 个归一化时间点）。
  2. 逐个 (关节, 时刻) 算两组之间的 t 值（以人为单位，熟练 5 人 vs 新手 3 人）。
  3. |t| 超过临界值的连续时刻合并成"差异簇"，簇质量 = 簇内 |t| 之和。
  4. 置换检验：把 8 个人重新分成"5 + 3"，共 C(8,3) = 56 种分法，每种分法算该关节的最大簇质量，
     得到"纯属偶然"的分布；真实簇比它大才算显著（每个关节内部已控制了 125 个时刻的多重比较）。
     跨关节再做 Holm 校正。
  5. 附加描述：簇内是否"3 名新手全部落在 5 名熟练者范围之外"（完全分离）。
  6. 稳健性：轮流去掉 1 名熟练者重做，看每个簇还在不在。
  7. 每个新手的个人报告。

注释约定同 baseline.py：【重要-n】【易错-n】，本文件从 20 开始编号。
"""
import argparse
from itertools import combinations
from math import comb
MAX_PERM = 10000
_IS_NV = None   # main() 里设置：真实分组（新手的位置）
from pathlib import Path

from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis_common as AC

MIN_LEN = 3          # 区间最短点数（归一化时间点）；更短的当作噪声直接丢掉
ALPHA = 0.05
COL = {"skilled": "#2a78d6", "novice": "#1baf7a", "below": "#2a78d6", "above": "#e34948"}


# ---------------------------------------------------------------- 核心算法
def outside_masks(M_sk, M_nv):
    """M_sk: [熟练人数, C, T]，M_nv: [新手人数, C, T]（每人平均曲线）。
    返回 below / above 两个 [C, T] 布尔数组，以及间隔大小 gap（度，>0 表示分开了多少）。

    【重要-20】判断规则："所有新手都在所有熟练者之外"。
    在两组没有差别的假设下，3 名新手恰好全部排在 5 名熟练者某一侧的概率是 2/56 ≈ 0.036（精确值，
    不依赖正态分布）。它比 t 检验保守，但每组只有 3–5 人时，这正是我们想要的。
    """
    sk_min, sk_max = M_sk.min(axis=0), M_sk.max(axis=0)
    nv_min, nv_max = M_nv.min(axis=0), M_nv.max(axis=0)
    below = nv_max < sk_min
    above = nv_min > sk_max
    gap = np.where(below, sk_min - nv_max, np.where(above, nv_min - sk_max, 0.0))
    return below, above, gap


def runs(mask_1d, min_len=MIN_LEN):
    """一维布尔数组里连续为 True 的区间，返回 [(start, end_inclusive), ...]。"""
    out, start = [], None
    for i, v in enumerate(np.append(mask_1d, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def find_intervals(M_sk, M_nv, channels, min_len=MIN_LEN):
    below, above, gap = outside_masks(M_sk, M_nv)
    rows = []
    for c, name in enumerate(channels):
        for direction, mask in (("novice_below", below[c]), ("novice_above", above[c])):
            for a, b in runs(mask, min_len):
                rows.append(dict(channel=name, c=c, start=a, end=b, length=b - a + 1, direction=direction,
                                 min_gap=gap[c, a:b + 1].min(), max_gap=gap[c, a:b + 1].max(),
                                 mean_diff_novice_minus_skilled=(M_nv[:, c, a:b + 1].mean()
                                                                 - M_sk[:, c, a:b + 1].mean())))
    return rows, below, above, gap


def t_curve(M_sk, M_nv):
    """两组（每人一条曲线）逐点的合并方差 t 值，形状 [C, T]。正值 = 新手高于熟练。"""
    a, b = M_nv, M_sk
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2))
    return (a.mean(0) - b.mean(0)) / (sp * np.sqrt(1 / na + 1 / nb) + 1e-9)


def clusters(t_1d, tcrit, min_len=1):
    """|t| > tcrit 的连续区间，返回 [(start, end, sign, mass)]。"""
    out = []
    for sgn in (1, -1):
        for a, b in runs(sgn * t_1d > tcrit, min_len):
            out.append((a, b, sgn, float(np.abs(t_1d[a:b + 1]).sum())))
    return out


def channel_null(M_all, n_novice, c, tcrit):
    """【重要-21】簇置换检验（cluster-based permutation，SPM 非参数版本的思路）。
    逐点检验 125 个时刻，偶然冒出"显著点"几乎是必然的，所以不看单点，看"连续簇的质量"：
    对 56 种分组方式各算一次该关节的最大簇质量，得到"纯属偶然时最大能有多大"的分布。
    【易错-20】56 种分法里包含真实分法本身（标准做法），所以 p 值最小只能到 1/56 ≈ 0.018。
    这是 8 个人的硬上限：跨 10 个关节做多重比较校正后，几乎不可能达到显著——
    所以组水平结论在论文里只能作为探索性结果，主结论要落在个人层面（见个人报告）。"""
    n = len(M_all)
    null = []
    # 人数多时（例如 15 + 15 人有 1.5 亿种分法）无法穷举，改为随机抽 MAX_PERM 种分法（固定种子，
    # 每个关节用同一组分法），并把真实分法放在第一个，p 值最小为 1/MAX_PERM
    if comb(n, n_novice) > MAX_PERM:
        rng = np.random.default_rng(0)
        assigns = [tuple(sorted(rng.choice(n, n_novice, replace=False))) for _ in range(MAX_PERM - 1)]
        assigns = [tuple(np.where(_IS_NV)[0])] + assigns if _IS_NV is not None else assigns
    else:
        assigns = combinations(range(n), n_novice)
    for nv_idx in assigns:
        sk_idx = [i for i in range(n) if i not in nv_idx]
        t = t_curve(M_all[sk_idx][:, [c]], M_all[list(nv_idx)][:, [c]])[0]
        cl = clusters(t, tcrit)
        null.append(max([m for *_, m in cl], default=0.0))
    return np.array(null)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def overlap(a1, b1, a2, b2):
    inter = max(0, min(b1, b2) - max(a1, a2) + 1)
    return inter / (b1 - a1 + 1)


# ---------------------------------------------------------------- 图
def fig_map(T, tcrit, intervals, channels, seg_bounds, out, title):
    from matplotlib.colors import LinearSegmentedColormap
    C, _ = T.shape
    cmap = LinearSegmentedColormap.from_list("bgr", [COL["below"], "#f0efec", COL["above"]])
    fig, ax = plt.subplots(figsize=(11, 0.38 * C + 1.6))
    im = ax.imshow(np.clip(T, -8, 8), aspect="auto", cmap=cmap, vmin=-8, vmax=8, interpolation="nearest")
    for r in intervals:
        ax.add_patch(plt.Rectangle((r["start"] - 0.5, r["c"] - 0.5), r["length"], 1, fill=False,
                                   ec="#1f1f1d", lw=1.8 if r.get("significant") else 0.6,
                                   ls="-" if r.get("significant") else ":"))
    for b in seg_bounds[1:-1]:
        ax.axvline(b - 0.5, color="#3d3d3a", lw=0.8)
    ax.set_yticks(range(C), channels, fontsize=8)
    ax.set_xlabel("normalized time (0-50 backswing, 50-100 forward swing, 100-125 follow-through)", fontsize=8)
    ax.set_title(title + f"\nblue: novice lower, red: novice higher; boxes: clusters |t|>{tcrit:.2f} "
                 "(solid = p<0.05 within joint)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="t (novice - skilled)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_curves(subs, groups, M_all, intervals, channels, seg_bounds, out):
    sig_ch = sorted({r["c"] for r in intervals if r.get("significant")})
    if not sig_ch:
        return
    n = len(sig_ch)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.0 * nrow), squeeze=False)
    x = np.arange(M_all.shape[2])
    for ax, c in zip(axes.flat, sig_ch):
        for s, g, m in zip(subs, groups, M_all):
            kind = "novice" if g in AC.NOVICE else "skilled"
            ax.plot(x, m[c], color=COL[kind], lw=1.2, alpha=0.85)
        for r in intervals:
            if r.get("significant") and r["c"] == c:
                ax.axvspan(r["start"] - 0.5, r["end"] + 0.5, color="#eda100", alpha=0.18, lw=0)
        for b in seg_bounds[1:-1]:
            ax.axvline(b, color="#8a8984", lw=0.6, ls="--")
        ax.set_title(channels[c], fontsize=9)
        ax.grid(color="#e6e5e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].plot([], [], color=COL["skilled"], label="skilled (each line = 1 subject)")
    axes.flat[0].plot([], [], color=COL["novice"], label="novice")
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Channels with significant skilled-novice differences (shaded = significant interval)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程
def main(prep_dir, out_dir, npz_name="angles_norm.npz", min_len=MIN_LEN):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d, meta = AC.load(prep_dir, npz_name)
    ch, sb = d["channels"], d["seg_bounds"]
    unit = "°/s" if "angspeed" in npz_name else "°"
    subs, M_all = AC.subject_means(d["X"], d["subj"])
    groups = [d["group"][d["subj"] == s][0] for s in subs]
    is_nv = np.array([g in AC.NOVICE for g in groups])
    global _IS_NV
    _IS_NV = is_nv
    sk_subs = [s for s, v in zip(subs, is_nv) if not v]
    nv_subs = [s for s, v in zip(subs, is_nv) if v]
    M_sk, M_nv = M_all[~is_nv], M_all[is_nv]
    R = [f"# 第 1 步：熟练者与新手的差异定位（输入 {npz_name}，单位 {unit}）", "",
         f"熟练组 {len(sk_subs)} 人：{', '.join(sk_subs)}；新手组 {len(nv_subs)} 人：{', '.join(nv_subs)}",
         f"使用的通道：{', '.join(ch)}" + (f"；自动去掉的已知问题通道：{', '.join(d['dropped'])}" if d["dropped"] else ""),
         ""]

    # 2–4 t 值、簇、置换检验
    tcrit = stats.t.ppf(1 - ALPHA / 2, len(M_sk) + len(M_nv) - 2)
    T = t_curve(M_sk, M_nv)
    below, above, gap = outside_masks(M_sk, M_nv)
    rows, ch_p = [], {}
    for c, name in enumerate(ch):
        null = channel_null(M_all, len(nv_subs), c, tcrit)
        for a, b, sgn, mass in clusters(T[c], tcrit):
            sl = slice(a, b + 1)
            sep = (below[c, sl] if sgn < 0 else above[c, sl])
            rows.append(dict(channel=name, c=c, start=a, end=b, length=b - a + 1,
                             direction="novice_above" if sgn > 0 else "novice_below", mass=mass,
                             p_perm=float(np.mean(null >= mass)),
                             mean_diff_novice_minus_skilled=M_nv[:, c, sl].mean() - M_sk[:, c, sl].mean(),
                             peak_diff=(M_nv[:, c, sl].mean(0) - M_sk[:, c, sl].mean(0))[
                                 np.argmax(np.abs(M_nv[:, c, sl].mean(0) - M_sk[:, c, sl].mean(0)))],
                             complete_separation_frac=float(sep.mean())))
    # 跨关节 Holm 校正：每个关节取其最小 p
    for r in rows:
        ch_p[r["channel"]] = min(ch_p.get(r["channel"], 1.0), r["p_perm"])
    names = list(ch_p)
    adj = dict(zip(names, holm([ch_p[n] for n in names]))) if names else {}
    for r in rows:
        r["p_holm_channel"] = adj[r["channel"]] if r["p_perm"] == ch_p[r["channel"]] else np.nan
        r["significant"] = r["p_perm"] < ALPHA          # 关节内校正（未跨关节校正）
        r["significant_holm"] = bool(r["p_holm_channel"] < ALPHA) if np.isfinite(r["p_holm_channel"]) else False
    _na = min(comb(len(M_all), len(nv_subs)), MAX_PERM)
    R.append(f"## 簇置换检验（每个关节 {_na} 种分组方式{'（随机抽取）' if _na == MAX_PERM else ''}；t 临界值 {tcrit:.2f}，df = {len(M_sk) + len(M_nv) - 2}）")
    R.append("p_perm：关节内校正（控制了 125 个时刻）；p_holm_channel：再跨关节做 Holm 校正。"
             f"p 最小为 1/{_na} = {1/_na:.4f}；人少时跨关节校正后几乎不可能显著，组水平结论请作为探索性结果。")
    R.append("")

    # 6 稳健性：轮流去掉 1 名熟练者
    for r in rows:
        r["robust_count"] = 0
    tcrit4 = stats.t.ppf(1 - ALPHA / 2, len(M_sk) - 1 + len(M_nv) - 2)
    for drop in range(len(M_sk)):
        keep = [i for i in range(len(M_sk)) if i != drop]
        Td = t_curve(M_sk[keep], M_nv)
        for r in rows:
            sgn = 1 if r["direction"] == "novice_above" else -1
            if any(q_sgn == sgn and overlap(r["start"], r["end"], qa, qb) >= 0.5
                   for qa, qb, q_sgn, _ in clusters(Td[r["c"]], tcrit4)):
                r["robust_count"] += 1
    for r in rows:
        r["robust"] = f"{r['robust_count']}/{len(M_sk)}"
        r["phase_start"] = f"{AC.phase_of(r['start'], sb)} {AC.pct_of_phase(r['start'], sb):.0f}%"
        r["phase_end"] = f"{AC.phase_of(r['end'], sb)} {AC.pct_of_phase(r['end'], sb):.0f}%"

    tab = pd.DataFrame(rows)
    if len(tab):
        tab = tab.sort_values("p_perm")
        cols = ["channel", "direction", "phase_start", "phase_end", "length", "mean_diff_novice_minus_skilled",
                "peak_diff", "complete_separation_frac", "p_perm", "p_holm_channel", "robust"]
        tab[cols].to_csv(out / "group_intervals.csv", index=False, float_format="%.3g")
        R.append("## 组间差异簇（按 p 排序；只列出长度 ≥ 1 的全部簇，p_perm < 0.05 的才算关节内显著）")
        R.append("mean_diff = 簇内新手均值 − 熟练均值；peak_diff = 簇内最大差值；complete_separation_frac = 簇内"
                 "'3 名新手全部在熟练者范围外'的时刻比例；robust = 去掉任一名熟练者后仍出现的次数")
        R.append(tab[cols].round(3).to_string(index=False))
    else:
        R.append("没有找到任何簇。")
    R.append("")

    # 6 个人报告：每个新手相对熟练组的偏差
    # 【重要-22】个人报告分两部分：
    #   (a) 组水平显著区间里，这个人偏了多少；
    #   (b) 这个人独有、但组水平不显著的问题：他自己超出熟练范围的区间（其他新手不一定有）。
    #   (b) 没有经过多重比较校正，只能作为"值得教练看一眼"的提示，论文里要这样表述。
    sk_min, sk_max, sk_mean = M_sk.min(0), M_sk.max(0), M_sk.mean(0)
    sk_sd = M_sk.std(0, ddof=1)
    pr = []
    def stroke_consistency(s, c, sl, sign):
        """该新手有多大比例的拍，在这段区间的平均值落在熟练者范围同一侧之外。
        【易错-21】个人报告只看平均曲线不够：平均值超出范围，可能只是几拍特别离谱拉出来的。
        这一栏越接近 1，说明是稳定的习惯。"""
        v = d["X"][d["subj"] == s][:, c, sl].mean(axis=1)
        lo_, hi_ = M_sk[:, c, sl].mean(axis=1).min(), M_sk[:, c, sl].mean(axis=1).max()
        return float(np.mean(v < lo_) if sign < 0 else np.mean(v > hi_))

    for s, m in zip(nv_subs, M_nv):
        for r in rows:
            if r.get("significant"):
                sl = slice(r["start"], r["end"] + 1)
                pr.append(dict(subject=s, kind="group_significant", channel=r["channel"],
                               phase_start=r["phase_start"], phase_end=r["phase_end"],
                               stroke_consistency=stroke_consistency(
                                   s, r["c"], sl, 1 if r["direction"] == "novice_above" else -1),
                               dev_from_skilled_mean=(m[r["c"], sl] - sk_mean[r["c"], sl]).mean(),
                               z_vs_skilled_spread=((m[r["c"], sl] - sk_mean[r["c"], sl]) /
                                                    np.maximum(sk_sd[r["c"], sl], 1e-6)).mean()))
        lo, hi = m < sk_min, m > sk_max
        for c, name in enumerate(ch):
            for direction, mask in (("below", lo[c]), ("above", hi[c])):
                for a, b in runs(mask, max(min_len, 8)):
                    sl = slice(a, b + 1)
                    pr.append(dict(subject=s, kind=f"individual_{direction}", channel=name,
                                   stroke_consistency=stroke_consistency(s, c, sl, -1 if direction == "below" else 1),
                                   phase_start=f"{AC.phase_of(a, sb)} {AC.pct_of_phase(a, sb):.0f}%",
                                   phase_end=f"{AC.phase_of(b, sb)} {AC.pct_of_phase(b, sb):.0f}%",
                                   dev_from_skilled_mean=(m[c, sl] - sk_mean[c, sl]).mean(),
                                   z_vs_skilled_spread=((m[c, sl] - sk_mean[c, sl]) /
                                                        np.maximum(sk_sd[c, sl], 1e-6)).mean()))
    prt = pd.DataFrame(pr)
    if len(prt):
        prt = prt.reindex(prt["dev_from_skilled_mean"].abs().sort_values(ascending=False).index)
        prt.to_csv(out / "novice_reports.csv", index=False, float_format="%.3g")
        R.append("## 新手个人报告（每人偏差最大的 5 处；group_significant = 组水平显著簇，individual = 本人平均曲线超出熟练范围）")
        for s in nv_subs:
            top = prt[prt.subject == s].head(5)
            R.append(f"{s}:")
            for t in top.itertuples():
                sign = "高于" if t.dev_from_skilled_mean > 0 else "低于"
                R.append(f"   {t.channel} [{t.phase_start} → {t.phase_end}] {sign}熟练均值 "
                         f"{abs(t.dev_from_skilled_mean):.0f}{unit}，{t.stroke_consistency:.0%} 的拍超出熟练范围（{t.kind}）")
    R.append("")

    fig_map(T, tcrit, rows, ch, sb, out / "fig_difference_map.png",
            f"Skilled (n={len(sk_subs)}) vs novice (n={len(nv_subs)}): t value per joint and time")
    fig_curves(subs, groups, M_all, rows, ch, sb, out / "fig_significant_channels.png")
    (out / "step1_report.txt").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))
    print(f"完成。输出目录：{out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_dir", required=True)
    ap.add_argument("--out_dir", default="step1_out")
    ap.add_argument("--npz", default="angles_norm.npz")
    ap.add_argument("--min_len", type=int, default=MIN_LEN)
    a = ap.parse_args()
    main(a.prep_dir, a.out_dir, a.npz, a.min_len)
