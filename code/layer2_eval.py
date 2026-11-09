"""第 2 层评测：比较 M0 平均模板、M1 kNN 检索、M2 PCA 投影（以后加 M3 自编码器）。

    python layer2_eval.py --prep_dir <prep_out_v31> --out_dir layer2_out --step1_dir step1_out

评测内容（全部以"人"为单位留一，熟练组 5 人）：
  A. 误报：用 4 名熟练者建参照，评估第 5 名熟练者。他本来没有错，偏差分数越低越好。
  B. 检出：在第 5 名熟练者的拍上人为注入已知偏差（某关节、某阶段、幅度 0.5–3 个标准差），
     看方法能否把它排到第一（定位准确率）、能否把它和没注入时区分开（检出 AUC）。
  C. 与组水平结果一致：用全部 5 名熟练者建参照评估新手，第 1 步找到的差异区间是否排在前面。
  D. 稳定性：每个新手奇数拍、偶数拍分别算，诊断图是否一致。
  E. 输出每个新手的结构化诊断表（第 3 层的输入）。

偏差分数的定义（所有方法共用，保证可比）：
  残差 R = 这一拍 − 参照曲线（度，有正负）
  单元分数 S(关节, 阶段) = 该阶段内 |R| 的平均 / 该关节在熟练者中的典型变异幅度
注释编号从 50 开始。
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis_common as AC
import reference as RF

PHASES = ["backswing", "forward", "follow"]
AMPS_SD = (0.5, 1.0, 2.0, 3.0)      # 注入偏差的幅度，单位 = 该关节的典型变异幅度
# 【重要-54】两种扰动形状：局部鼓包（占阶段 60%）和整段偏移（占阶段 100%，两端平滑）。
# PCA 之类的投影方法擅长抓"局部、不像任何熟练风格"的偏差，但可能把"整段偏移"当成风格吸收掉；
# 只测一种形状会偏袒某类方法，所以两种都测。
WIDTHS = (0.6, 1.0)
TAU = 1.0                            # "被判为偏差"的分数阈值（单元平均偏差超过 1 个典型变异幅度）
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def phase_slices(sb):
    return [slice(int(a), int(b)) for a, b in zip(sb[:-1], sb[1:])]


def cell_scores(X, Xh, s, sl):
    """S[n, C, P] 和带符号的平均残差 Rm[n, C, P]（度）。"""
    R = X - Xh
    S = np.stack([np.abs(R[:, :, q]).mean(2) for q in sl], axis=2) / s.reshape(1, -1, 1)
    Rm = np.stack([R[:, :, q].mean(2) for q in sl], axis=2)
    return S, Rm


def _window(w, width_frac):
    if width_frac >= 1.0:   # 整段偏移：Tukey 窗（中间平、两端各 20% 平滑过渡）
        x = np.linspace(0, 1, w)
        win = np.ones(w)
        edge = 0.2
        m = x < edge
        win[m] = 0.5 * (1 - np.cos(np.pi * x[m] / edge))
        m = x > 1 - edge
        win[m] = 0.5 * (1 - np.cos(np.pi * (1 - x[m]) / edge))
        return win
    return np.hanning(w + 2)[1:-1]


def inject_habit(Xs, c, ph, amp_deg, rng, width_frac=0.6):
    """【重要-53】"习惯型"扰动：同一个人所有拍都在同一位置、同一方向偏——这才是真实新手偏差的样子
    （第 1 步里新手的偏差在 80–100% 的拍里都出现）。用于评估"个人诊断"（多拍平均后定位）。"""
    Xp = Xs.copy()
    L = ph.stop - ph.start
    w = max(3, min(L, int(round(L * width_frac))))
    bump = _window(w, width_frac)
    st = rng.integers(ph.start, ph.stop - w + 1)
    Xp[:, c, st:st + w] += rng.choice([-1.0, 1.0]) * amp_deg * bump
    return Xp


def inject(Xs, c, ph, amp_deg, rng, width_frac=0.6):
    """【重要-50】合成扰动：在阶段 ph 内随机位置加一个汉宁窗形状的鼓包，随机正负号。
    鼓包宽度取该阶段的 60%，模拟"某个关节在某个阶段整体偏了"这种真实偏差的形态。"""
    Xp = Xs.copy()
    L = ph.stop - ph.start
    w = max(3, min(L, int(round(L * width_frac))))
    bump = _window(w, width_frac)
    starts = rng.integers(ph.start, ph.stop - w + 1, size=len(Xs))
    signs = rng.choice([-1.0, 1.0], size=len(Xs))
    for i in range(len(Xs)):
        Xp[i, c, starts[i]:starts[i] + w] += signs[i] * amp_deg * bump
    return Xp


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq


# ---------------------------------------------------------------- A + B：留一熟练者
def loso_skilled(d, methods, rng):
    X, subj, grp = d["X"], d["subj"], d["group"]
    sl = phase_slices(d["seg_bounds"])
    C, P = X.shape[1], len(sl)
    sk_subs = AC.natsorted(np.unique(subj[np.isin(grp, AC.SKILLED)]))
    base_seed = int(rng.integers(1_000_000_000))
    fa_rows, det_rows = [], []
    # 【重要-55】嵌套基线（v6.2 修正）：评估熟练者 h 时，h 必须完全不出现——
    # 基线 = 其余每名熟练者 g 在"去掉 h 和 g"的参照下的平均分数图，再对 g 平均。
    # v6.1 以前用的是"去掉 g"的参照，里面仍含 h 本人（轻微泄漏），与第 3 层 layer3.build_references 不一致。
    # 这样熟练者与新手经历同一流程：n 人参照 + n−1 人基线。
    nested = {}
    for held in sk_subs:
        others = [g for g in sk_subs if g != held]
        for g in others:
            ref = np.isin(grp, AC.SKILLED) & ~np.isin(subj, [held, g])
            s_ = RF.Scaler().fit(X[ref]).s.ravel()
            Xg = X[subj == g]
            for m in methods:
                m.fit(X[ref], subj[ref])
                nested.setdefault((m.name, held), []).append(cell_scores(Xg, m.reconstruct(Xg), s_, sl)[0].mean(0))
    nested = {k: np.mean(v, axis=0) for k, v in nested.items()}
    for held in sk_subs:
        ref = np.isin(grp, AC.SKILLED) & (subj != held)
        Xr, sr = X[ref], subj[ref]
        Xs = X[subj == held]
        # 【易错-50】典型变异幅度 s 也只能用参考组算，并且所有方法共用同一个 s，分数才可比
        s = RF.Scaler().fit(Xr).s.ravel()
        for m in methods:
            m.fit(Xr, sr)
            S0, _ = cell_scores(Xs, m.reconstruct(Xs), s, sl)
            fa_rows.append(dict(method=m.name, held_out=held, mean_score=S0.mean(),
                                flag_rate=(S0 > TAU).mean()))
            # 【易错-51】扣除的基线图只能来自"其他"熟练者的留一结果，不能用被评估者自己的，
            # 否则等于提前知道了他的风格偏差在哪（泄漏），定位准确率会虚高。
            b_other = nested[(m.name, held)]
            for c in range(C):
                for p in range(P):
                    for a_sd, wf in [(a, w_) for a in AMPS_SD for w_ in WIDTHS]:
                        # 【易错-52】v7：每个注入场景用固定种子，所有方法看到完全相同的注入数据（配对比较）。
                        # 以前所有方法共用一个随机数流，不同方法拿到的注入位置、正负号不同，比较里混入了随机噪声。
                        rng = np.random.default_rng([base_seed, sk_subs.index(held), c, p,
                                                     int(a_sd * 10), int(wf * 10)])
                        Xp = inject(Xs, c, sl[p], a_sd * s[c], rng, wf)
                        S1, _ = cell_scores(Xp, m.reconstruct(Xp), s, sl)
                        flat = (S1 - b_other).reshape(len(Xs), -1)
                        idx = c * P + p
                        rank = (flat > flat[:, [idx]]).sum(1) + 1
                        # 个人层面：习惯型扰动，多拍平均后排序
                        Xh_ = inject_habit(Xs, c, sl[p], a_sd * s[c], rng, wf)
                        Sh, _ = cell_scores(Xh_, m.reconstruct(Xh_), s, sl)
                        pm = (Sh.mean(0) - b_other).ravel()
                        rank_person = int((pm > pm[idx]).sum()) + 1
                        flat_raw = S1.reshape(len(Xs), -1)
                        rank_raw = (flat_raw > flat_raw[:, [idx]]).sum(1) + 1
                        det_rows.append(dict(method=m.name, held_out=held, channel=d["channels"][c],
                                             phase=PHASES[p], amp_sd=a_sd, width=wf, top1=np.mean(rank == 1),
                                             top1_raw=np.mean(rank_raw == 1),
                                             person_top1=float(rank_person == 1), person_top3=float(rank_person <= 3),
                                             top3=np.mean(rank <= 3),
                                             auc=auc(S1[:, c, p], S0[:, c, p]),
                                             flagged=np.mean(S1[:, c, p] > TAU)))
    return pd.DataFrame(fa_rows), pd.DataFrame(det_rows)


# ---------------------------------------------------------------- C + D + E：新手
def novices(d, methods, step1_dir, fa_maps):
    X, subj, grp, ch = d["X"], d["subj"], d["group"], d["channels"]
    sl = phase_slices(d["seg_bounds"])
    ref = np.isin(grp, AC.SKILLED)
    s = RF.Scaler().fit(X[ref]).s.ravel()
    nv_subs = sorted(np.unique(subj[np.isin(grp, AC.NOVICE)]))
    targets = []
    if step1_dir and (Path(step1_dir) / "group_intervals.csv").exists():
        gi = pd.read_csv(Path(step1_dir) / "group_intervals.csv")
        gi = gi[(gi["p_perm"] < 0.1) & (gi["channel"].isin(ch))]
        for r in gi.itertuples():
            ph_a, ph_b = r.phase_start.split()[0], r.phase_end.split()[0]
            for ph in {ph_a, ph_b}:
                targets.append((r.channel, ph))
    targets = sorted(set(targets))
    cons_rows, stab_rows, diag_rows, maps = [], [], [], {}
    for m in methods:
        m.fit(X[ref], subj[ref])
        excess = []
        for sname in nv_subs:
            Xs = X[subj == sname]
            S, Rm = cell_scores(Xs, m.reconstruct(Xs), s, sl)
            # 【重要-51】"超额分数" = 新手的分数 − 熟练者在留一评估时的平均分数（同一方法、同一单元）。
            # 熟练者自己在某些关节上也天然偏差大（风格差异），减掉这部分，剩下的才是新手特有的问题。
            ex = S.mean(0) - fa_maps[m.name]
            excess.append(ex)
            # D 稳定性：奇偶拍
            a, b = S[0::2].mean(0) - fa_maps[m.name], S[1::2].mean(0) - fa_maps[m.name]
            stab_rows.append(dict(method=m.name, subject=sname, split_half_r=np.corrcoef(a.ravel(), b.ravel())[0, 1]))
            # E 结构化诊断
            for c in range(len(ch)):
                for p in range(len(sl)):
                    sign = np.sign(Rm[:, c, p].mean())
                    diag_rows.append(dict(method=m.name, subject=sname, channel=ch[c], phase=PHASES[p],
                                          mean_residual_deg=Rm[:, c, p].mean(), mean_score=S[:, c, p].mean(),
                                          excess_score=ex[c, p],
                                          consistency=np.mean((S[:, c, p] > TAU) & (np.sign(Rm[:, c, p]) == sign)),
                                          tier=_tier(ch[c])))
        ex_mean = np.mean(excess, axis=0)
        maps[m.name] = ex_mean
        order = np.argsort(-ex_mean.ravel())
        for chn, ph in targets:
            idx = ch.index(chn) * len(sl) + PHASES.index(ph)
            cons_rows.append(dict(method=m.name, target=f"{chn}/{ph}",
                                  rank=int(np.where(order == idx)[0][0]) + 1, of=ex_mean.size))
    return pd.DataFrame(cons_rows), pd.DataFrame(stab_rows), pd.DataFrame(diag_rows), maps


def _tier(c):
    try:
        import config as CFG
        return CFG.CHANNEL_TIER.get(c, 3)
    except Exception:  # noqa
        return np.nan


# ---------------------------------------------------------------- 图
def fig_tradeoff(fa, det, out):
    """【重要-52】论文主图：横轴 = 误报（熟练者被判出的偏差），纵轴 = 检出能力。
    左上角最好：既不冤枉熟练者，又能找到真偏差。单看任何一个轴都会误导——
    M1 kNN@1 误报几乎为 0（因为它会贴着最像的那一拍走），但真偏差也会被它"贴"掉。"""
    g_fa = fa.groupby("method")["mean_score"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, a_sd in enumerate((1.0, 2.0)):
        ax = axes[i]
        g = det[det.amp_sd == a_sd].groupby("method")["person_top3"].mean()
        for j, mname in enumerate(g_fa.index):
            ax.scatter(g_fa[mname], g[mname], s=70, color=PALETTE[j % len(PALETTE)], zorder=3)
            ax.annotate(mname, (g_fa[mname], g[mname]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("false alarm: mean deviation score on held-out skilled (lower = better)", fontsize=8)
        ax.set_ylabel(f"person-level top-3 localization ({a_sd:g} SD habit, both shapes)", fontsize=8)
        ax.grid(color="#e6e5e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Layer 2 trade-off (leave-one-skilled-out): upper-left is better", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_tradeoff.png", dpi=130)
    plt.close(fig)


def fig_amp_curves(det, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for j, (mname, g) in enumerate(det.groupby("method")):
        gg = g.groupby("amp_sd")[["person_top1", "auc"]].mean()
        axes[0].plot(gg.index, gg["person_top1"], "o-", color=PALETTE[j % len(PALETTE)], lw=1.8, label=mname)
        axes[1].plot(gg.index, gg["auc"], "o-", color=PALETTE[j % len(PALETTE)], lw=1.8, label=mname)
    axes[0].set_ylabel("person-level top-1 localization")
    axes[1].set_ylabel("detection AUC (perturbed vs unperturbed)")
    for ax in axes:
        ax.set_xlabel("injected amplitude (channel SD units)")
        ax.grid(color="#e6e5e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_detection_curves.png", dpi=130)
    plt.close(fig)


def fig_novice_maps(maps, ch, out, show=("M0_mean", "M1_knn5", "M2_pca95")):
    from matplotlib.colors import LinearSegmentedColormap
    show = [m for m in show if m in maps]
    cmap = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#e34948"])
    fig, axes = plt.subplots(1, len(show), figsize=(3.6 * len(show), 0.33 * len(ch) + 1.6), squeeze=False)
    vmax = max(np.max(maps[m]) for m in show)
    for ax, mname in zip(axes[0], show):
        im = ax.imshow(np.clip(maps[mname], 0, None), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(3), ["back", "fwd", "follow"], fontsize=8)
        ax.set_yticks(range(len(ch)), ch if ax is axes[0, 0] else [], fontsize=7)
        ax.set_title(mname, fontsize=9)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="excess score (novice - skilled baseline)")
    fig.suptitle("Where novices deviate (mean of 3 novices)", fontsize=10)
    fig.savefig(out / "fig_novice_excess_maps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 主流程
def main(prep_dir, out_dir, step1_dir=None, seed=0, methods=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    d, _ = AC.load(prep_dir)
    methods = methods or RF.default_methods()
    R = ["# 第 2 层评测：参照检索方法对比", "",
         f"通道：{', '.join(d['channels'])}",
         f"方法：{', '.join(m.name for m in methods)}", ""]

    fa, det = loso_skilled(d, methods, rng)
    fa.to_csv(out / "false_alarm.csv", index=False, float_format="%.4g")
    det.to_csv(out / "detection.csv", index=False, float_format="%.4g")
    sl = phase_slices(d["seg_bounds"])

    # 熟练者在留一评估时的平均分数图（用于新手的"超额分数"）
    fa_maps = {}
    for m in methods:
        maps_ = []
        for held in AC.natsorted(np.unique(d["subj"][np.isin(d["group"], AC.SKILLED)])):
            ref = np.isin(d["group"], AC.SKILLED) & (d["subj"] != held)
            s = RF.Scaler().fit(d["X"][ref]).s.ravel()
            m.fit(d["X"][ref], d["subj"][ref])
            Xs = d["X"][d["subj"] == held]
            maps_.append(cell_scores(Xs, m.reconstruct(Xs), s, sl)[0].mean(0))
        fa_maps[m.name] = np.mean(maps_, axis=0)

    summ = fa.groupby("method")[["mean_score", "flag_rate"]].mean()
    for wf in WIDTHS:
        tag = "local" if wf < 1 else "broad"
        for a_sd in (1.0, 2.0, 3.0):
            g = det[(det.amp_sd == a_sd) & (det.width == wf)].groupby("method")
            summ[f"{tag}_p1@{a_sd:g}"] = g["person_top1"].mean()
            summ[f"{tag}_p3@{a_sd:g}"] = g["person_top3"].mean()
        summ[f"{tag}_auc@2"] = det[(det.amp_sd == 2.0) & (det.width == wf)].groupby("method")["auc"].mean()
    summ = summ.reindex([m.name for m in methods])
    summ.to_csv(out / "summary.csv", float_format="%.3f")
    R.append("## A + B：留一熟练者（误报越低越好；定位 top1 和检出 AUC 越高越好）")
    R.append(f"mean_score = 熟练者自己的平均偏差分数；flag_rate = 被判为偏差（分数 > {TAU}）的单元比例")
    R.append("local / broad = 局部鼓包 / 整段偏移；p1、p3 = 个人定位 top-1 / top-3（所有拍同一习惯，多拍平均后排序，对应实际诊断用法）；"
             "auc = 单拍检出 AUC；@ 后数字 = 注入幅度（关节典型变异幅度的倍数）")
    R.append("排序都按'超额分数'（扣除其他熟练者的天然风格偏差，与新手诊断同一口径）")
    R.append(f"随机猜中的 top-1 概率 = 1 / {d['X'].shape[1] * len(sl)} = {1 / (d['X'].shape[1] * len(sl)):.3f}")
    R.append(summ.round(3).to_string())
    R.append("")

    cons, stab, diag, maps = novices(d, methods, step1_dir, fa_maps)
    if len(cons):
        cons.to_csv(out / "group_consistency.csv", index=False)
        R.append("## C：第 1 步的组水平差异，在新手超额分数里排第几（越靠前越好）")
        R.append(cons.pivot(index="target", columns="method", values="rank")
                 .reindex(columns=[m.name for m in methods]).to_string())
        R.append(f"（共 {cons['of'].iloc[0]} 个单元）")
        R.append("")
    stab.to_csv(out / "stability.csv", index=False, float_format="%.3f")
    R.append("## D：新手奇偶拍诊断图的一致性（相关系数，越接近 1 越稳定）")
    R.append(stab.pivot(index="subject", columns="method", values="split_half_r")
             .reindex(columns=[m.name for m in methods]).round(2).to_string())
    R.append("")
    diag.to_csv(out / "novice_diagnosis_all_methods.csv", index=False, float_format="%.4g")
    R.append("## E：每个新手超额分数最高的 3 处（以 M2_pca95 为例；其他方法见 csv）")
    show = "M2_pca95" if "M2_pca95" in diag.method.values else methods[0].name
    for sname, g in diag[diag.method == show].groupby("subject"):
        R.append(f"{sname}:")
        for r in g.sort_values("excess_score", ascending=False).head(3).itertuples():
            R.append(f"   {r.channel}/{r.phase} {'高于' if r.mean_residual_deg > 0 else '低于'}参照 "
                     f"{abs(r.mean_residual_deg):.0f}°，超额分数 {r.excess_score:.2f}，一致性 {r.consistency:.0%}，可信度 {r.tier}")
    fig_tradeoff(fa, det, out)
    fig_amp_curves(det, out)
    fig_novice_maps(maps, d["channels"], out)
    (out / "layer2_report.txt").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))
    print(f"完成。输出目录：{out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_dir", required=True)
    ap.add_argument("--out_dir", default="layer2_out")
    ap.add_argument("--step1_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default=None,
                    help="逗号分隔，如 M0_mean,M2_pca90,M3_ccae；不给则用原来的 9 种")
    a = ap.parse_args()
    meths = [RF.make_method(n) for n in a.methods.split(",")] if a.methods else None
    main(a.prep_dir, a.out_dir, a.step1_dir, a.seed, meths)
