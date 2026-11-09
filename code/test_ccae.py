"""M3（约束卷积自编码投影）快速自检：先跑这个，确认能用、看看耗时，再跑第 2、3 层。

    python test_ccae.py --prep ../../prep_out_v31

做三件事（GPU 约 1–2 分钟）：
  1. 5 名熟练者轮流留出，用其余 4 人训练，算留出者的平均单元分数（"没毛病的人"被判多少偏差，越低越好）；
  2. 用全部 5 名熟练者训练，算 3 名新手的平均单元分数；新手与熟练者的差距越大，区分越清楚；
  3. 报告可执行约束是否起作用：重建前后违反约束的拍的比例和违反量。
这只是粗看，正式比较以第 2、3 层的评测为准。
"""
import argparse
import time

import numpy as np

import analysis_common as AC
import layer2_eval as L2
import reference as RF

NAMES = ["M0_mean", "M2_pca90", "M3_ccae", "M3_ccae_nocons", "M3_ccae_noopt"]


def mean_score(m, s, Xs, sl):
    S, _ = L2.cell_scores(Xs, m.reconstruct(Xs), s, sl)
    return float(S.mean())


def main(prep):
    d, _ = AC.load(prep)
    X, subj, grp = d["X"], d["subj"], d["group"]
    sl = L2.phase_slices(d["seg_bounds"])
    sk = np.isin(grp, AC.SKILLED)
    sk_subs = sorted(np.unique(subj[sk]))
    nv_subs = sorted(np.unique(subj[np.isin(grp, AC.NOVICE)]))
    print(f"数据：{X.shape}，熟练 {sk_subs}，新手 {nv_subs}\n")

    res, timing, cons = {n: {} for n in NAMES}, {}, {}
    for held in sk_subs:                                     # 1. 熟练者留一
        ref = sk & (subj != held)
        s = RF.Scaler().fit(X[ref]).s.ravel()
        for n in NAMES:
            m = RF.make_method(n)
            t0 = time.time()
            m.fit(X[ref], subj[ref])
            t_fit = time.time() - t0
            t0 = time.time()
            res[n][held] = mean_score(m, s, X[subj == held], sl)
            timing.setdefault(n, []).append((t_fit, time.time() - t0))
            if getattr(m, "stats", None):
                cons.setdefault(n, []).append(dict(m.stats))
    s = RF.Scaler().fit(X[sk]).s.ravel()                     # 2. 新手
    nv = {n: {} for n in NAMES}
    for n in NAMES:
        m = RF.make_method(n)
        m.fit(X[sk], subj[sk])
        for v in nv_subs:
            nv[n][v] = mean_score(m, s, X[subj == v], sl)
            if getattr(m, "stats", None):
                cons.setdefault(n + "|novice", []).append(dict(m.stats))

    print(f"{'方法':16s} {'熟练者留一(均值)':>14s} {'新手(均值)':>10s} {'差距':>6s}   每人")
    for n in NAMES:
        a, b = np.mean(list(res[n].values())), np.mean(list(nv[n].values()))
        per = " ".join(f"{k} {v:.2f}" for k, v in {**res[n], **nv[n]}.items())
        print(f"{n:16s} {a:14.2f} {b:10.2f} {b - a:6.2f}   {per}")
    print("\n可执行约束（重建前 → 后：违反约束的拍的比例，平均违反量）")
    for key, lst in cons.items():
        ab = np.mean([x["active_before"] for x in lst]); aa = np.mean([x["active_after"] for x in lst])
        vb = np.mean([x["viol_before"] for x in lst]); va = np.mean([x["viol_after"] for x in lst])
        cl = np.mean([x["clipped"] for x in lst])
        who = "新手" if key.endswith("|novice") else "留出熟练者"
        print(f"  {key.split('|')[0]:16s} {who:6s} {ab:5.0%} → {aa:5.0%}   {vb:.1e} → {va:.1e}   潜变量触边界 {cl:.0%}")
    m3 = [m for m in (RF.make_method("M3_ccae"),)][0]
    print(f"\n设备：{m3.device}")
    tf, tr = np.mean([t[0] for t in timing["M3_ccae"]]), np.mean([t[1] for t in timing["M3_ccae"]])
    print(f"M3 训练一次 {tf:.1f}s，重建一人 {tr:.2f}s；估计耗时（只算 M3）："
          f"第 2 层约 {(31 * tf + 5 * 672 * tr) / 60:.0f} 分钟，第 3 层约 {(26 * tf + 1260 * tr) / 60:.0f} 分钟")
    print("判读：'差距'越大越好；M3 与 M3_ccae_nocons 应有差别，并且 M3 重建后违反约束的比例应接近 0%"
          "——否则说明约束没起作用。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    main(ap.parse_args().prep)
