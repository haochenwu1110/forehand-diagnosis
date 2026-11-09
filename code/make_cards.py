"""Paper B 新数据：单人诊断卡（每人一页 A4）。

用法：python make_cards.py OUT_DIR [PREP_DIR] [LAYER3_DIR]
    默认 PREP_DIR = ../data（公开数据），LAYER3_DIR = ../results/diagnosis
输出：OUT_DIR/诊断卡_N1.png ... 以及 诊断卡_新手N1-N5.pdf、诊断卡_熟练者S1-S10.pdf
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

import layer3 as L3  # noqa: E402

_HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "cards")
_PREP = Path(sys.argv[2] if len(sys.argv) > 2 else _HERE.parent / "data")
NPZ = _PREP / "angles_norm.npz" if (_PREP / "angles_norm.npz").exists() else _PREP / "joint_angles_normalised.npz"
RES = Path(sys.argv[3] if len(sys.argv) > 3 else _HERE.parent / "results" / "diagnosis")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.unicode_minus": False, "pdf.fonttype": 3,
})
INK, INK2, MUTED, GRID = "#1f1f1e", "#52514e", "#8a8984", "#e6e5e0"
SK, NV, GOLD, ACC = "#2a5caa", "#e0622a", "#eda100", "#a8410f"
SK_BAND = "#d5e0f1"

ORDER = ["D1", "D2", "D3", "D4", "D5", "A1", "A2", "A3", "A4", "A5", "B1", "B4", "B5", "C1", "C3"]
SID = {"A1": "S1", "A2": "S2", "A3": "S3", "A4": "S4", "A5": "S5", "B1": "S6", "B4": "S7", "B5": "S8",
       "C1": "S9", "C3": "S10", "D1": "N1", "D2": "N2", "D3": "N3", "D4": "N4", "D5": "N5"}
GROUP = {s: ("高水平" if s[0] == "A" else "新手" if s[0] == "D" else "熟练") for s in SID}
GROUP_COL = {"高水平": "#1f3864", "熟练": SK, "新手": NV}
PH = {"backswing": 0, "forward": 1, "follow": 2}
PH_CN = ["引拍", "前挥", "随挥"]
CIRC = "①②③④⑤⑥"
GROUP_DIFF = ["trunk_forward_lean", "knee_flex_L", "shoulder_elev_L", "thorax_rot_imu", "pelvis_rot_imu"]

# ------------------------------------------------------------------ 数据
d = np.load(NPZ, allow_pickle=True)
ok = d["qc_pass"].astype(bool)
X, subj, ch = d["X"][ok], d["subject"][ok].astype(str), list(d["channels"])
INV = {v: k for k, v in SID.items()}                    # 公开数据用 S1/N1，内部流程用 A1/D1：统一转成内部编号
subj = np.array([INV.get(s, s) for s in subj])
sb = d["seg_bounds"].astype(int)
SKILLED = [s for s in SID if s[0] != "D"]
T = X.shape[2]
xx = np.arange(T)

nv = pd.read_csv(RES / "findings_all.csv", encoding="utf-8-sig")
nv = nv[nv["shown"] == True]  # noqa: E712
sk = pd.read_csv(RES / "skilled_loso_findings.csv", encoding="utf-8-sig")
sk["matches_group_difference"] = False
FD = pd.concat([sk, nv], ignore_index=True)
FD["subject"] = FD["subject"].astype(str).map(lambda s: INV.get(s, s))


def ref_for(s):
    """参照熟练者：新手用全部 10 人；熟练者留一（其余 9 人）。"""
    return [g for g in SKILLED if g != s]


def template(s):
    refs = ref_for(s)
    m = np.isin(subj, refs)
    mod = L3.make_method("M0_mean")
    mod.fit(X[m], subj[m])
    tpl = mod.reconstruct(X[subj == s]).mean(0)          # [C,T]
    pm = np.stack([X[subj == g].mean(0) for g in refs])   # 每名参照熟练者的平均曲线
    return tpl, pm


def idx_of(phase, pct):
    p = PH[phase]
    a, b = sb[p], sb[p + 1]
    return int(round(a + pct / 100 * (b - a - 1)))


# ------------------------------------------------------------------ 画图零件
def panel(ax, c, tpl, pm, me, sd, title, sub=None, rng=None, peak=None):
    ax.fill_between(xx, pm[:, c].min(0), pm[:, c].max(0), color=SK_BAND, lw=0, zorder=1)
    if rng is not None:
        ax.axvspan(rng[0] - 0.5, rng[1] + 0.5, color=GOLD, alpha=0.18, lw=0, zorder=0)
    ax.plot(xx, tpl[c], color=SK, lw=1.4, zorder=3)
    ax.fill_between(xx, me[c] - sd[c], me[c] + sd[c], color=NV, alpha=0.14, lw=0, zorder=2)
    ax.plot(xx, me[c], color=NV, lw=1.6, zorder=4)
    if peak is not None:
        ax.plot([peak, peak], [tpl[c, peak], me[c, peak]], color=ACC, lw=1.0, zorder=5)
        ax.scatter([peak, peak], [tpl[c, peak], me[c, peak]], s=10, color=ACC, zorder=6, lw=0)
    for b in sb[1:-1]:
        ax.axvline(b - 0.5, color=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=0)
    lo, hi = ax.get_ylim()
    pad = (hi - lo) * 0.16
    ax.set_ylim(lo, hi + pad)
    for k in range(3):
        ax.text((sb[k] + sb[k + 1] - 1) / 2, 0.985, PH_CN[k], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color=INK2)
    ax.set_xlim(0, T - 1)
    ax.set_xticks([0, sb[1] - 0.5, sb[2] - 0.5, T - 1])
    ax.set_xticklabels(["t1", "t2", "t3", "+0.35 s"], fontsize=6.5, color=INK2)
    ax.tick_params(axis="y", labelsize=6.5, colors=INK2)
    ax.set_ylabel("角度（°）", fontsize=7, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", color=INK, pad=14 if sub else 6)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=7, color=INK2, va="bottom")


def card(s):
    rows = FD[FD["subject"] == s].sort_values("rank")
    n = len(rows)
    grp, pid = GROUP[s], SID[s]
    is_nv = grp == "新手"
    tpl, pm = template(s)
    Xs = X[subj == s]
    me, sd = Xs.mean(0), Xs.std(0)

    fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
    W, H = 210, 297

    def fx(mm):
        return mm / W

    def fy(mm):          # 从上往下量
        return 1 - mm / H

    # ---------- 页眉
    fig.text(fx(14), fy(22), pid, fontsize=30, fontweight="bold", color=INK, va="baseline")
    chip_x = 14 + 9 * len(pid) + 4
    fig.patches.append(FancyBboxPatch((fx(chip_x), fy(21.5)), fx(16), 6.5 / H, boxstyle="round,pad=0,rounding_size=0.006",
                                      transform=fig.transFigure, fc=GROUP_COL[grp], ec="none"))
    fig.text(fx(chip_x + 8), fy(18.2), grp, fontsize=9, color="white", ha="center", va="center", fontweight="bold")
    fig.text(fx(196), fy(14), "个人技术诊断卡", fontsize=13, fontweight="bold", color=INK, ha="right", va="baseline")
    fig.text(fx(196), fy(20), "网球正手 · 全身惯性动作捕捉", fontsize=8, color=INK2, ha="right", va="baseline")
    ref_txt = ("参照：全部 10 名熟练者（S1–S10）" if is_nv else
               f"参照：其余 9 名熟练者（{pid} 本人不参与，留一评估）")
    fig.text(fx(14), fy(30), ref_txt + "　｜　方法：平均模板 + PCA90，扣除熟练者之间的风格差异　｜　阈值 P_min = 0.85",
             fontsize=7.5, color=INK2)
    fig.add_artist(Line2D([fx(14), fx(196)], [fy(33), fy(33)], color=GRID, lw=1.0, transform=fig.transFigure))

    # ---------- 结论框
    if n == 0:
        head = "未报出问题"
        body = "在这一参照下，本人的每个关节、每个阶段都没有超出熟练者之间的正常风格差异。"
    else:
        head = f"报出 {n} 条问题"
        if is_nv:
            m = int(rows["matches_group_difference"].sum())
            body = (f"其中 {m} 条与新手组整体差异一致（标 ★），其余是本人特有的问题。" if m else
                    "这些问题都不在新手组的整体差异里，是本人特有的问题。")
        else:
            body = "熟练者平均约 1 条，阈值按此校准。熟练者报出的问题可看作个人风格或误报，不一定需要纠正。"
    fig.patches.append(FancyBboxPatch((fx(14), fy(50)), fx(182), 13 / H, boxstyle="round,pad=0,rounding_size=0.008",
                                      transform=fig.transFigure, fc="#f6f5f1", ec="none"))
    fig.text(fx(19), fy(41.5), head, fontsize=13, fontweight="bold",
             color=(NV if (is_nv and n) else INK), va="center")
    fig.text(fx(19 + 5.2 * len(head) + 6), fy(41.5), body, fontsize=8, color=INK2, va="center")

    # ---------- 问题表
    y = 58
    if n:
        cols = [(14, "", "left"), (21, "部位", "left"), (58, "阶段", "left"), (84, "表现", "left"),
                (145, "偏差", "right"), (167, "出现比例", "right"), (196, "两种参照", "right")]
        for x0, lab, ha in cols:
            fig.text(fx(x0), fy(y), lab, fontsize=7, color=MUTED, ha=ha)
        y += 2.5
        fig.add_artist(Line2D([fx(14), fx(196)], [fy(y), fy(y)], color=GRID, lw=0.8, transform=fig.transFigure))
        for _, r in rows.iterrows():
            y += 6
            star = " ★" if r["matches_group_difference"] else ""
            vals = [CIRC[int(r["rank"]) - 1], r["joint_cn"] + star, r["phase_cn"], r["direction_text"],
                    f"{r['magnitude_deg']:+.0f}°", f"{r['consistency']:.0%}",
                    "都报出" if r["agreement"] == "both" else "仅平均模板"]
            for (x0, _, ha), v in zip(cols, vals):
                fig.text(fx(x0), fy(y), v, fontsize=8, color=INK, ha=ha,
                         fontweight="bold" if x0 in (14, 145) else "normal")
        y += 3
        fig.add_artist(Line2D([fx(14), fx(196)], [fy(y), fy(y)], color=GRID, lw=0.8, transform=fig.transFigure))
        y += 5
    else:
        fig.text(fx(14), fy(y + 1), "供参考：下面是新手组与熟练者差异最大的 5 个指标，本人的曲线与熟练者参照的对比。",
                 fontsize=8, color=INK2)
        y += 7

    # ---------- 曲线小图
    if n:
        items = []
        for _, r in rows.iterrows():
            c = ch.index(r["channel"])
            rg = json.loads(r["range_pct"])
            pk = json.loads(r["peak_pct"])
            rng = (idx_of(*rg["start"]), idx_of(*rg["end"]))
            rel = json.loads(r["related"]) if isinstance(r["related"], str) and r["related"].strip() else []
            sub = f"{r['phase_cn']}：{r['direction_text']}，约 {abs(r['magnitude_deg']):.0f}°"
            if rel:
                sub += "（躯干转动同时偏 {:.0f}°，已合并）".format(abs(rel[0]["magnitude_deg"]))
            items.append(dict(c=c, title=f"{CIRC[int(r['rank']) - 1]} {r['joint_cn']}", sub=sub,
                              rng=rng, peak=idx_of(*pk)))
    else:
        items = [dict(c=ch.index(c), title=L3.CHANNEL_CN[c], sub=None, rng=None, peak=None) for c in GROUP_DIFF]

    k = len(items)
    ncol = 2 if k > 1 else 1
    nrow = int(np.ceil(k / ncol))
    slot_legend = ncol == 2 and k % 2 == 1          # 空出的那一格放图例
    top, bottom = y + 6, (272 if slot_legend else 264)
    gap_y = 14
    ph = min(58, (bottom - top - gap_y * (nrow - 1)) / nrow)
    pw = (182 - 14 * (ncol - 1)) / ncol if ncol == 2 else 120
    for i, it in enumerate(items):
        r_, c_ = divmod(i, ncol)
        x0 = 14 + 6 + c_ * (pw + 14)
        y0 = top + r_ * (ph + gap_y)
        ax = fig.add_axes([fx(x0), fy(y0 + ph), fx(pw - 6), ph / H])
        panel(ax, it["c"], tpl, pm, me, sd, it["title"], it["sub"], it["rng"], it["peak"])

    # ---------- 图例与说明
    ly = 277
    handles = [Line2D([], [], color=NV, lw=1.6, label=f"{pid} 本人平均（阴影 ±1 SD）"),
               Line2D([], [], color=SK, lw=1.4, label="熟练者平均模板"),
               Patch(fc=SK_BAND, ec="none", label="熟练者之间的范围（各人平均曲线的最小–最大）")]
    if n:
        handles += [Patch(fc=GOLD, alpha=0.35, ec="none", label="报出问题的区间"),
                    Line2D([], [], color=ACC, lw=1.0, marker="o", ms=3, label="偏差最大处")]
    if slot_legend:
        lx = 14 + 6 + (pw + 14)
        ly0 = top + (nrow - 1) * (ph + gap_y) + 4
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(fx(lx), fy(ly0)), ncol=1, frameon=False,
                   fontsize=7.5, handlelength=2.2, labelspacing=1.0, title="图例", title_fontsize=8,
                   alignment="left")
    else:
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(fx(13), fy(ly + 4)), ncol=3, frameon=False,
                   fontsize=7, handlelength=2.0, columnspacing=1.6)
    note = ("偏差 = 本人 − 熟练者平均模板（代表阶段内的平均值）；出现比例 = 本人击球中同方向明显偏离的比例。"
            "时间按 t1 引拍开始、t2 拍头最靠后、t3 右肘速度峰值对齐。")
    fig.text(fx(14), fy(287), note, fontsize=6.5, color=MUTED)
    fig.text(fx(196), fy(291.5), f"原编号 {s}　仅限内部", fontsize=6.5, color=MUTED, ha="right")
    return fig


if __name__ == "__main__":
    groups = {"新手N1-N5": ORDER[:5], "熟练者S1-S10": ORDER[5:]}
    for name, subs in groups.items():
        with PdfPages(OUT / f"诊断卡_{name}.pdf") as pdf:
            for s in subs:
                f = card(s)
                pdf.savefig(f)
                f.savefig(OUT / f"诊断卡_{SID[s]}.png", dpi=150)
                plt.close(f)
                print("done", SID[s])
