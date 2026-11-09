"""第 3 层：结构化诊断。

把第 2 层的单元分数（14 通道 × 3 阶段）变成每个新手的"问题清单"：
哪个关节、哪个阶段（阶段内百分比）、往哪个方向、偏多少度、多少比例的拍都这样、可信度、两种方法是否一致。
输出的 JSON 是第 4 层（大模型转教练语言）的唯一输入，大模型只翻译，不判断。

用法：
    python layer3.py --prep prep_out_v31 --step1 step1_out --out layer3_out

只 import 已有文件（reference.py / layer2_eval.py / analysis_common.py / config.py），不改动它们。

标注索引（接着第 2 层的 54 往后编）：
    【重要-60】熟练者留一时的嵌套基线          【易错-60】基线里混进被评估者本人
    【重要-61】为什么合并 M0 和 PCA90           【易错-61】两个方法方向相反时不能算"一致"
    【重要-62】优先级公式与权重                 【易错-62】阈值只能用熟练者校准
    【重要-63】阈值校准规则                     【易错-63】校准时问题数不是单调的
    【重要-64】相邻阶段合并、分离角去重          【易错-64】度和"典型变异幅度"两种单位不能混
    【重要-65】方向文字必须和源码符号一致        【易错-65】注入幅度要乘 s 换成度
    【重要-66】验证与消融为什么要每种设置重新校准 【易错-66】奇偶拍稳定性要用同一个基线
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import analysis_common as AC  # noqa: E402
import config as CFG  # noqa: E402
import layer2_eval as L2  # noqa: E402
import reference as RF  # noqa: E402

PHASES = L2.PHASES                      # ["backswing", "forward", "follow"]
PHASE_CN = {"backswing": "引拍", "forward": "前挥", "follow": "随挥"}
TAU = L2.TAU                            # 单拍"偏差"阈值（1 个典型变异幅度），和第 2 层同一口径

# ---------------------------------------------------------------- 可调参数（全部写进报告）
DEFAULT_METHODS = ("M0_mean", "M2_pca90")
TIER_W = {1: 1.0, 2: 1.0, 3: 0.5}       # 可信度权重：1 级 IMU，2 级右臂（可与 IMU 核对），3 级纯骨架估计
AGREE_W_SINGLE = 0.5                    # 两个方法里只有一个标出时的权重
TARGET_FP = 1.0                         # 熟练者留一时，每人平均被报出的问题数上限
P_GRID = np.round(np.arange(0.01, 3.0001, 0.01), 2)   # 优先级阈值候选
INJ_AMPS = (1.0, 2.0, 3.0)              # 注入幅度（典型变异幅度的倍数）
INJ_WIDTHS = (0.6, 1.0)                 # 局部鼓包 / 整段偏移
HALF_PEAK = 0.5                         # 区间 = 残差超过峰值一半的连续段

CHANNEL_CN = {
    "pelvis_rot_imu": "骨盆转动", "thorax_rot_imu": "躯干转动", "separation_imu": "躯干-骨盆分离角",
    "shoulder_elev_R": "右肩抬高（上臂抬起）", "uarm_forward_R": "右上臂前后位置", "elbow_flex_R": "右肘屈曲",
    "trunk_forward_lean": "躯干前倾", "trunk_lateral_tilt": "躯干侧倾",
    "hip_flex_L": "左髋屈曲", "hip_flex_R": "右髋屈曲", "knee_flex_L": "左膝屈曲", "knee_flex_R": "右膝屈曲",
    "shoulder_elev_L": "左肩抬高（左臂抬起）", "elbow_flex_L": "左肘屈曲",
}

# 【重要-65】方向文字：(残差为负时, 残差为正时)。残差 = 本人 − 参照。
# 每一条都对照过源码里的角度定义（kinematics.compute_angles 的约定注释、imu_yaw 的符号约定、
# preprocess.py 里 separation_imu = thorax_rot_imu − pelvis_rot_imu）。
# 骨盆 / 躯干：imu_yaw 注释"引拍方向为负"，所以正 = 朝前挥方向转 = 更打开（更朝向球网）；
# 报告第 1 节会再用数据自动核对一次（熟练者 t2→t3 前挥时这个角应当增大）。
# 【易错】改动角度定义后，这张表必须同步改，否则大模型会把"抬得不够"说成"抬得太高"。
DIRECTION_TEXT = {
    "pelvis_rot_imu": ("骨盆更侧身、转得更少（更关闭）", "骨盆更早/更多地打开（更朝向球网）"),
    "thorax_rot_imu": ("躯干更侧身、转得更少（更关闭）", "躯干更早/更多地打开（更朝向球网）"),
    "separation_imu": ("躯干相对骨盆更落后（肩髋分离更大）", "躯干相对骨盆更超前（肩比髋先打开）"),
    "shoulder_elev_R": ("右上臂抬得更低", "右上臂抬得更高"),
    "uarm_forward_R": ("右上臂更靠后", "右上臂更靠前"),
    "elbow_flex_R": ("右肘更直", "右肘更弯"),
    "trunk_forward_lean": ("躯干更直立/后仰", "躯干更前倾"),
    "trunk_lateral_tilt": ("躯干更向左侧倾", "躯干更向右侧倾"),
    "hip_flex_L": ("左髋屈得更少", "左髋屈得更多"),
    "hip_flex_R": ("右髋屈得更少", "右髋屈得更多"),
    "knee_flex_L": ("左腿更直（屈膝不足）", "左膝屈得更多"),
    "knee_flex_R": ("右腿更直（屈膝不足）", "右膝屈得更多"),
    "shoulder_elev_L": ("左臂抬得更低", "左臂抬得更高"),
    "elbow_flex_L": ("左肘更直", "左肘更弯"),
}

# 躯干转动链：separation = thorax − pelvis，三者不独立；骨盆和躯干又随动作一起转（试点数据中一名新手的
# 骨盆、躯干"提前打开"分别占了第 1、4 条，教练看来是同一个毛病），所以三者在重叠阶段只列一条
ROT_GROUP = {"pelvis_rot_imu", "thorax_rot_imu", "separation_imu"}

# 消融设置。【重要-66】每种设置都用熟练者重新校准阈值，否则"去掉某项后问题数变多"
# 只是阈值没跟着变，不能说明那一项有用。
CONFIGS = [
    dict(name="full", methods=DEFAULT_METHODS, use_tier=True, use_cons=True, cn="完整"),
    dict(name="no_tier", methods=DEFAULT_METHODS, use_tier=False, use_cons=True, cn="去掉可信度权重"),
    dict(name="no_consistency", methods=DEFAULT_METHODS, use_tier=True, use_cons=False, cn="去掉一致性"),
    dict(name="M0_only", methods=("M0_mean",), use_tier=True, use_cons=True, cn="只用 M0"),
    dict(name="PCA90_only", methods=("M2_pca90",), use_tier=True, use_cons=True, cn="只用 PCA90"),
]
# 第二个参照方法的简称（v7：可用 --methods M0_mean,M3_ccae 把 PCA90 换成约束卷积自编码投影）
SHORT = {"M2_pca90": "PCA90", "M3_ccae": "CCAE"}


def short_name(mn):
    return SHORT.get(mn, mn)


def make_method(name):
    return RF.make_method(name)          # 含 M3_ccae（需要 PyTorch）


def tier_of(ch):
    return CFG.CHANNEL_TIER.get(ch, 3)


# ---------------------------------------------------------------- 参照与基线
class Reference:
    """用一组熟练者拟合好的参照：每个方法一个模型 + 共用的典型变异幅度 s。"""

    def __init__(self, X, subj, method_names):
        # 【易错-50 同款】s 只能用参照组算，所有方法共用同一个 s，分数才可比
        self.s = RF.Scaler().fit(X).s.ravel()
        self.models = {}
        for mn in method_names:
            m = make_method(mn)
            m.fit(X, subj)
            self.models[mn] = m

    def score(self, Xs, sl):
        """返回 {方法: (S[n,C,P] 分数, Rm[n,C,P] 带符号平均残差°, curve[C,T] 本人平均残差曲线°)}"""
        out = {}
        for mn, m in self.models.items():
            Xh = m.reconstruct(Xs)
            S, Rm = L2.cell_scores(Xs, Xh, self.s, sl)
            out[mn] = (S, Rm, (Xs - Xh).mean(0))
        return out


def build_references(d, method_names, sl):
    """拟合所有需要的参照，并算好基线图。

    【重要-60】基线 = "熟练者被当成新人评估时，天然会有多少偏差"。
      - 评估新手：参照 = 全部 5 名熟练者；基线 = 每名熟练者 g 在"其余 4 人"参照下的平均分数图，再对 g 平均。
      - 评估熟练者 h（误报、注入检出）：h 必须完全不存在于这个世界里——
        参照 = 其余 4 人；基线 = 其余每个 g 在"去掉 h 和 g 的 3 人"参照下的分数图，再对 g 平均。
      这样新手和熟练者经历的是同一个流程（n 人参照 + n−1 人基线），误报数才能和新手的问题数直接比。
    【易错-60】第 2 层（layer2_eval.loso_skilled 第 127 行）给熟练者 h 扣的基线来自"去掉 g"的参照，
      这个参照里还含有 h 本人，有轻微泄漏。第 3 层不复用那段代码，自己按上面的嵌套方式重算。
    """
    X, subj, grp = d["X"], d["subj"], d["group"]
    sk = np.isin(grp, AC.SKILLED)
    sk_subs = AC.natsorted(np.unique(subj[sk]))

    def ref_without(excl):
        mask = sk & ~np.isin(subj, list(excl))
        return Reference(X[mask], subj[mask], method_names)

    def mean_map(ref, g):
        sc = ref.score(X[subj == g], sl)
        return {mn: sc[mn][0].mean(0) for mn in method_names}

    full = ref_without(())
    loo = {h: ref_without((h,)) for h in sk_subs}
    base_novice = {mn: np.mean([mean_map(loo[g], g)[mn] for g in sk_subs], axis=0) for mn in method_names}
    base_skilled = {}
    for h in sk_subs:
        maps = [mean_map(ref_without((h, g)), g) for g in sk_subs if g != h]
        base_skilled[h] = {mn: np.mean([m_[mn] for m_ in maps], axis=0) for mn in method_names}
    return dict(full=full, loo=loo, base_novice=base_novice, base_skilled=base_skilled, sk_subs=sk_subs)


def person_features(scored, base):
    """单元特征。【易错-64】excess 的单位是"典型变异幅度"，resid / curve 的单位是度，两者不能混用：
    排序、阈值只用 excess；告诉人"偏多少度"只用 resid。"""
    pe = {}
    for mn, (S, Rm, curve) in scored.items():
        resid = Rm.mean(0)
        sign = np.sign(resid)
        cons = ((S > TAU) & (np.sign(Rm) == sign[None])).mean(0)
        pe[mn] = dict(excess=S.mean(0) - base[mn], resid=resid, cons=cons, curve=curve)
    pe["_n"] = len(next(iter(scored.values()))[0])
    return pe


# ---------------------------------------------------------------- 单元 → 问题清单
def cell_candidates(pe, channels, cfg):
    """每个 (通道, 阶段) 单元算优先级。返回 list[dict]，未被任何方法标出的单元不返回。

    【重要-61】M0（平均模板）回答"和典型熟练者哪里不一样"，PCA90 回答"超出熟练者全部动作范围的地方"。
      第 2 层评测里 M0 对组水平差异排名最好但误报高（43% 单元被判偏差），PCA90 误报低（22%）但会吸收
      一部分整段偏差。两个方法都认为"比熟练者天然偏差更大、方向一样"的单元，确定性最高。
    【重要-62】priority = excess × consistency × tier_w × agree_w
      excess      超额分数（取标出它的方法里较小的那个，保守）
      consistency 多大比例的拍都在同方向偏超过 1 个典型变异幅度（区分"习惯"和"偶尔几拍"）
      tier_w      3 级骨架通道 0.5：第 2 层原始诊断排名前列多是 3 级通道（躯干侧倾、髋屈曲），
                  这些通道没有 IMU 可核对，测量误差本身就可能造成偏差
      agree_w     只有一个方法标出 0.5
    """
    meths = list(cfg["methods"])
    C, P = pe[meths[0]]["excess"].shape
    rows = []
    for c in range(C):
        ch = channels[c]
        for p in range(P):
            flagged = [mn for mn in meths if pe[mn]["excess"][c, p] > 0]
            if not flagged:
                continue
            signs = {np.sign(pe[mn]["resid"][c, p]) for mn in flagged}
            # 【易错-61】两个方法都说"有偏差"但方向相反，说明这是参照选择造成的假象，不是"一致"
            if len(signs) > 1:
                continue
            if len(meths) == 1:
                agreement, aw = "single_method", 1.0
            elif len(flagged) == 2:
                agreement, aw = "both", 1.0
            else:
                agreement, aw = ("M0_only" if flagged[0] == "M0_mean"
                                 else f"{short_name(flagged[0])}_only"), AGREE_W_SINGLE
            src = "M0_mean" if "M0_mean" in flagged else flagged[0]
            exc = min(pe[mn]["excess"][c, p] for mn in flagged)
            cons = float(np.mean([pe[mn]["cons"][c, p] for mn in flagged]))
            tier = tier_of(ch)
            pri = exc * (cons if cfg["use_cons"] else 1.0) * (TIER_W[tier] if cfg["use_tier"] else 1.0) * aw
            rows.append(dict(c=c, p=p, channel=ch, agreement=agreement, src=src, excess=float(exc),
                             consistency=cons, tier=tier, priority=float(pri),
                             resid=float(pe[src]["resid"][c, p]),
                             direction=int(np.sign(pe[src]["resid"][c, p]) or 1)))
    return rows


def _range(curve_c, sl, phases, direction, sb):
    """在合并后的阶段内，找残差（按方向取正）峰值，以及超过峰值一半的连续区间。"""
    a, b = sl[phases[0]].start, sl[phases[-1]].stop
    v = direction * curve_c[a:b]
    k = int(np.argmax(v))
    thr = HALF_PEAK * v[k]
    lo, hi = k, k
    while lo > 0 and v[lo - 1] >= thr:
        lo -= 1
    while hi < len(v) - 1 and v[hi + 1] >= thr:
        hi += 1

    def where(t):
        ph = AC.phase_of(t, sb)
        return [ph, int(round(AC.pct_of_phase(t, sb)))]
    return dict(start=where(a + lo), end=where(a + hi), peak=where(a + k))


def build_findings(pe, channels, cfg, p_min, sl, sb):
    """阈值 → 相邻阶段合并 → 分离角去重 → 按优先级排序（不截断）。"""
    cells = [r for r in cell_candidates(pe, channels, cfg) if r["priority"] >= p_min]
    # 【重要-64】同一通道、相邻阶段、同方向 → 一条问题（例如上臂"前挥后半到随挥"都偏后，是同一个毛病）
    by_ch = {}
    for r in cells:
        by_ch.setdefault(r["channel"], []).append(r)
    findings = []
    for ch, rs in by_ch.items():
        rs = sorted(rs, key=lambda r: r["p"])
        groups = [[rs[0]]]
        for r in rs[1:]:
            last = groups[-1][-1]
            if r["p"] == last["p"] + 1 and r["direction"] == last["direction"]:
                groups[-1].append(r)
            else:
                groups.append([r])
        for g in groups:
            rep = max(g, key=lambda r: r["priority"])
            phases = [r["p"] for r in g]
            findings.append(dict(channel=ch, phases=phases, rep=rep, cells=g, priority=rep["priority"],
                                 range=_range(pe[rep["src"]]["curve"][rep["c"]], sl, phases, rep["direction"], sb),
                                 related=[]))
    findings.sort(key=lambda f: -f["priority"])
    # 【重要-64】骨盆转动、躯干转动、分离角三者不独立（separation = thorax − pelvis，骨盆和躯干一起转）：
    # 在重叠阶段同时上榜时，只保留优先级高的那条，其余放进 related，不单列
    # （否则同一个"提前打开"会占两三条名额，把其他问题挤出前 5 条）
    kept = []
    for f in findings:
        dup = None
        for k in kept:
            if (f["channel"] != k["channel"] and {f["channel"], k["channel"]} <= ROT_GROUP
                    and set(f["phases"]) & set(k["phases"])):
                dup = k
                break
        if dup is not None:
            dup["related"].append(dict(channel=f["channel"], phases=[PHASES[p] for p in f["phases"]],
                                       direction="+" if f["rep"]["direction"] > 0 else "-",
                                       magnitude_deg=round(f["rep"]["resid"], 1)))
        else:
            kept.append(f)
    return kept


def n_findings(pe, channels, cfg, p_min, sl, sb):
    return len(build_findings(pe, channels, cfg, p_min, sl, sb))


def calibrate(pe_skilled, pe_novice, channels, cfg, sl, sb):
    """【重要-63】阈值校准：只看熟练者。选最小的 P_MIN，使得 P_MIN 及以上的所有候选值下，
    熟练者留一时每人平均被报出的问题数都 ≤ TARGET_FP。
    【易错-62】不准看新手结果调阈值——新手的问题数只是顺带列出来，不参与选择。
    【易错-63】问题数随阈值不是严格单调的（阈值降低时，两条问题可能因为中间阶段也过线而合并成一条），
      所以不能"从小往大找到第一个满足的就停"，要保证它以上全部满足。"""
    rows = []
    for pm in P_GRID:
        rows.append(dict(config=cfg["name"], p_min=pm,
                         skilled_mean=np.mean([n_findings(pe, channels, cfg, pm, sl, sb) for pe in pe_skilled.values()]),
                         novice_mean=np.mean([n_findings(pe, channels, cfg, pm, sl, sb) for pe in pe_novice.values()])))
    tab = pd.DataFrame(rows)
    ok = (tab["skilled_mean"] <= TARGET_FP).values
    chosen = P_GRID[-1]
    for i in range(len(P_GRID) - 1, -1, -1):
        if not ok[i]:
            break
        chosen = P_GRID[i]
    return float(chosen), tab


# ---------------------------------------------------------------- 输出格式
def finding_to_dict(f, rank, group_targets):
    r = f["rep"]
    ch = f["channel"]
    phs = [PHASES[p] for p in f["phases"]]
    neg, pos = DIRECTION_TEXT.get(ch, ("低于参照", "高于参照"))
    return dict(
        rank=rank, channel=ch, joint_cn=CHANNEL_CN.get(ch, ch),
        phases=phs, phase_cn="→".join(PHASE_CN[p] for p in phs),
        range_pct=dict(start=f["range"]["start"], end=f["range"]["end"]),
        peak_pct=f["range"]["peak"],
        direction="+" if r["direction"] > 0 else "-",
        direction_text=pos if r["direction"] > 0 else neg,
        magnitude_deg=round(r["resid"], 1),
        magnitude_source=r["src"],
        consistency=round(r["consistency"], 2),
        tier=int(r["tier"]), agreement=r["agreement"],
        excess=round(r["excess"], 3), priority=round(r["priority"], 3),
        related=f["related"],
        matches_group_difference=bool(match_targets(f, group_targets)),
    )


def human_line(fd):
    a, b = fd["range_pct"]["start"], fd["range_pct"]["end"]
    rng = f"{PHASE_CN[a[0]]} {a[1]}% → {PHASE_CN[b[0]]} {b[1]}%"
    ag = fd["agreement"]
    agree = {"both": "两种方法一致", "M0_only": "仅平均模板", "single_method": "单方法"}.get(
        ag, f"仅 {ag.replace('_only', '')}")
    rel = f"；相关：{', '.join(x['channel'] for x in fd['related'])}" if fd["related"] else ""
    star = " ★组水平差异" if fd["matches_group_difference"] else ""
    return (f"{fd['rank']}. {fd['joint_cn']}｜{rng}｜{fd['direction_text']}，约 {abs(fd['magnitude_deg']):.0f}°"
            f"｜{fd['consistency']:.0%} 的拍如此｜可信度 {fd['tier']} 级｜{agree}｜优先级 {fd['priority']:.2f}{rel}{star}")


# ---------------------------------------------------------------- 第 1 步组水平差异
def load_group_targets(step1_dir, channels):
    if not step1_dir or not (Path(step1_dir) / "group_intervals.csv").exists():
        return []
    gi = pd.read_csv(Path(step1_dir) / "group_intervals.csv")
    gi = gi[(gi["p_perm"] < 0.1) & gi["channel"].isin(channels)]
    out = []
    for r in gi.itertuples():
        a, b = PHASES.index(r.phase_start.split()[0]), PHASES.index(r.phase_end.split()[0])
        out.append(dict(channel=r.channel, phases=list(range(a, b + 1)),
                        direction=-1 if "below" in r.direction else 1,
                        label=f"{r.channel} {r.phase_start}→{r.phase_end} ({'-' if 'below' in r.direction else '+'})"))
    return out


def match_targets(f, targets):
    return [t["label"] for t in targets
            if t["channel"] == f["channel"] and t["direction"] == f["rep"]["direction"]
            and set(t["phases"]) & set(f["phases"])]


# ---------------------------------------------------------------- 验证
def direction_check(d, sl):
    """【重要-65】用数据核对骨盆/躯干的正方向：前挥（t2→t3）时身体朝球网打开，
    如果"正 = 打开"成立，熟练者的平均曲线在前挥末应当比前挥初大。"""
    X, grp, ch = d["X"], d["group"], d["channels"]
    sk = np.isin(grp, AC.SKILLED)
    lines = []
    for c in ("pelvis_rot_imu", "thorax_rot_imu"):
        if c not in ch:
            lines.append(f"{c}: 数据里没有这个通道，跳过")
            continue
        m = X[sk][:, ch.index(c)].mean(0)
        a, b = sl[1].start, sl[1].stop - 1
        delta = m[b] - m[a]
        lines.append(f"{c}: 熟练者平均 前挥初 {m[a]:+.1f}° → 前挥末 {m[b]:+.1f}°（变化 {delta:+.1f}°）"
                     f" → {'符合 正=打开' if delta > 0 else '【不符合】正=打开，需要把 DIRECTION_TEXT 的两侧对调'}")
    return lines


def detection(d, refs, channels, sl, sb, p_mins, rng):
    """注入检出：在留出的熟练者 h 所有拍上注入习惯型偏差，跑完整第 3 层，看能否进前 3 条且方向正确。"""
    X, subj = d["X"], d["subj"]
    rows = []
    C, P = len(channels), len(sl)
    for h in refs["sk_subs"]:
        ref, base = refs["loo"][h], refs["base_skilled"][h]
        Xs = X[subj == h]
        for c in range(C):
            for p in range(P):
                for amp in INJ_AMPS:
                    for wf in INJ_WIDTHS:
                        # 【易错-65】注入幅度单位是"典型变异幅度"，必须乘参照组的 s 换成度
                        Xp = L2.inject_habit(Xs, c, sl[p], amp * ref.s[c], rng, wf)
                        sign = int(np.sign((Xp - Xs)[:, c].sum()))
                        pe = person_features(ref.score(Xp, sl), base)
                        for cfg in CONFIGS:
                            fs = build_findings(pe, channels, cfg, p_mins[cfg["name"]], sl, sb)
                            pos = [i for i, f in enumerate(fs)
                                   if f["channel"] == channels[c] and p in f["phases"]
                                   and f["rep"]["direction"] == sign]
                            rows.append(dict(config=cfg["name"], held_out=h, channel=channels[c], phase=PHASES[p],
                                             tier=tier_of(channels[c]), amp_sd=amp, width=wf,
                                             hit_top3=bool(pos and pos[0] < 3), hit_any=bool(pos)))
    return pd.DataFrame(rows)


def jaccard_top3(fa, fb):
    sa = {(f["channel"], p) for f in fa[:3] for p in f["phases"]}
    sb_ = {(f["channel"], p) for f in fb[:3] for p in f["phases"]}
    if not sa and not sb_:
        return np.nan
    return len(sa & sb_) / len(sa | sb_)


# 可选：对某一名新手做个案核对（试点数据用过；公开数据默认不做）
CASE_SUBJ = None
LYF_CHECKS = [
    (("pelvis_rot_imu", "separation_imu", "thorax_rot_imu"), "forward", "骨盆/躯干 前挥 提前打开"),
    (("uarm_forward_R",), "backswing", "上臂向前角 引拍"),
]


# ---------------------------------------------------------------- 图
def fig_findings(pe_novice, channels, cfg, p_min, targets, out):
    alt = f"{short_name(cfg['methods'][-1])}_only"
    colors = {"both": "#2a78d6", "M0_only": "#eb6834", alt: "#1baf7a"}
    labels = {"both": "两种方法一致", "M0_only": "仅平均模板", alt: f"仅 {short_name(cfg['methods'][-1])}"}
    # WSL 里的 Linux 没有中文字体，直接借用 Windows 的（/mnt/c/Windows/Fonts）
    from matplotlib import font_manager
    for fp in ("/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    C, P = len(channels), len(PHASES)
    subs = list(pe_novice)
    fig, axes = plt.subplots(len(subs), 1, figsize=(max(10, C * 0.9), 2.4 * len(subs)), sharex=True, squeeze=False)
    for ax, s in zip(axes[:, 0], subs):
        cells = cell_candidates(pe_novice[s], channels, cfg)
        for r in cells:
            x = r["c"] * P + r["p"]
            ax.bar(x, r["priority"], width=0.8, color=colors[r["agreement"]],
                   alpha=1.0 if r["priority"] >= p_min else 0.3)
        ax.axhline(p_min, color="#666666", lw=1, ls="--")
        for t in targets:
            for p in t["phases"]:
                ax.plot(channels.index(t["channel"]) * P + p, 0, marker="*", ms=10, color="#222222",
                        clip_on=False, zorder=5)
        for c in range(1, C):
            ax.axvline(c * P - 0.5, color="#dddddd", lw=0.6)
        ax.set_ylabel(s)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xticks([c * P + 1 for c in range(C)])
    axes[-1, 0].set_xticklabels([CHANNEL_CN.get(ch, ch) for ch in channels], rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in colors.values()]
    handles += [plt.Line2D([], [], color="#666666", ls="--"),
                plt.Line2D([], [], marker="*", color="#222222", ls="", ms=10)]
    fig.suptitle("每个新手的单元优先级（每个关节内 3 根柱 = 引拍 / 前挥 / 随挥；浅色 = 未过阈值）", fontsize=10)
    fig.legend(handles, list(labels.values()) + [f"阈值 {p_min:.2f}", "第 1 步组水平差异"],
               ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.965), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程
def main(prep_dir, step1_dir, out_dir, seed=0, methods=DEFAULT_METHODS, max_findings=5, skip_detection=False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    d, _ = AC.load(prep_dir)
    ch, sb = d["channels"], d["seg_bounds"]
    sl = L2.phase_slices(sb)
    methods = tuple(methods)
    for cfg in CONFIGS:
        if cfg["name"] in ("full", "no_tier", "no_consistency"):
            cfg["methods"] = methods
        elif cfg["name"] not in ("M0_only",):          # 第二个单方法设置跟随 --methods 的第二个方法
            alt = methods[-1]
            cfg.update(name=f"{short_name(alt)}_only", methods=(alt,), cn=f"只用 {short_name(alt)}")
    targets = load_group_targets(step1_dir, ch)

    # 消融里的单方法设置用 M0 和第二个方法，参照里一定要拟合这两个
    all_methods = tuple(dict.fromkeys(list(methods) + ["M0_mean"]))
    refs = build_references(d, all_methods, sl)
    X, subj, grp = d["X"], d["subj"], d["group"]
    nv_subs = sorted(np.unique(subj[np.isin(grp, AC.NOVICE)]))
    pe_sk = {h: person_features(refs["loo"][h].score(X[subj == h], sl), refs["base_skilled"][h])
             for h in refs["sk_subs"]}
    pe_nv = {s: person_features(refs["full"].score(X[subj == s], sl), refs["base_novice"]) for s in nv_subs}

    # 校准
    p_mins, cal_tabs = {}, []
    for cfg in CONFIGS:
        p_mins[cfg["name"]], tab = calibrate(pe_sk, pe_nv, ch, cfg, sl, sb)
        cal_tabs.append(tab)
    cal = pd.concat(cal_tabs)
    cal.to_csv(out / "calibration.csv", index=False, float_format="%.3f")

    full = CONFIGS[0]
    P0 = p_mins["full"]
    params = dict(P_MIN=P0, TAU=TAU, tier_weight={str(k): v for k, v in TIER_W.items()},
                  agreement_weight=dict(both=1.0, single=AGREE_W_SINGLE), methods=list(methods),
                  target_skilled_findings=TARGET_FP)

    # 新手诊断（完整设置）
    all_rows, nv_findings = [], {}
    for s in nv_subs:
        fs = build_findings(pe_nv[s], ch, full, P0, sl, sb)
        nv_findings[s] = fs
        fds = [finding_to_dict(f, i + 1, targets) for i, f in enumerate(fs)]
        js = dict(subject=s, n_strokes=int(pe_nv[s]["_n"]),
                  reference=f"{len(refs['sk_subs'])} 名熟练者（{', '.join(refs['sk_subs'])}）",
                  params=params, n_findings_total=len(fds), findings=fds[:max_findings])
        (out / f"findings_{s}.json").write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
        for fd in fds:
            all_rows.append(dict(subject=s, shown=fd["rank"] <= max_findings,
                                 **{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                                    for k, v in fd.items()}))
    pd.DataFrame(all_rows).to_csv(out / "findings_all.csv", index=False, encoding="utf-8-sig")

    sk_rows = []
    for h in refs["sk_subs"]:
        for i, f in enumerate(build_findings(pe_sk[h], ch, full, P0, sl, sb)):
            fd = finding_to_dict(f, i + 1, targets)
            sk_rows.append(dict(subject=h, **{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                                                   else v) for k, v in fd.items()}))
    pd.DataFrame(sk_rows).to_csv(out / "skilled_loso_findings.csv", index=False, encoding="utf-8-sig")

    # 验证
    det = detection(d, refs, ch, sl, sb, p_mins, rng) if not skip_detection else pd.DataFrame()
    if len(det):
        det.to_csv(out / "detection.csv", index=False)

    abl = []
    stab_rows = []
    for cfg in CONFIGS:
        pm = p_mins[cfg["name"]]
        fs_sk = {h: build_findings(pe_sk[h], ch, cfg, pm, sl, sb) for h in refs["sk_subs"]}
        fs_nv = {s: build_findings(pe_nv[s], ch, cfg, pm, sl, sb) for s in nv_subs}
        row = dict(config=cfg["name"], cn=cfg["cn"], P_MIN=pm,
                   skilled_mean_findings=np.mean([len(v) for v in fs_sk.values()]),
                   novice_mean_findings=np.mean([len(v) for v in fs_nv.values()]))
        if len(det):
            dc = det[det["config"] == cfg["name"]]
            for wf, tag in ((1.0, "broad"), (0.6, "local")):
                for amp in INJ_AMPS:
                    row[f"{tag}{amp:g}SD_top3"] = dc[(dc["width"] == wf) & (dc["amp_sd"] == amp)]["hit_top3"].mean()
            for t in (1, 2, 3):
                q = dc[(dc["width"] == 1.0) & (dc["amp_sd"] == 2.0) & (dc["tier"] == t)]
                row[f"broad2SD_top3_tier{t}"] = q["hit_top3"].mean() if len(q) else np.nan
        for t in targets:
            row[f"hit:{t['label']}"] = sum(bool(any(match_targets(f, [t]) for f in fs_nv[s][:max_findings]))
                                           for s in nv_subs)
        if CASE_SUBJ and CASE_SUBJ in fs_nv:
            for chans, ph, lab in LYF_CHECKS:
                if not any(c_ in ch for c_ in chans):
                    row[f"case:{lab}"] = "无此通道"
                    continue
                hit = [i + 1 for i, f in enumerate(fs_nv[CASE_SUBJ][:max_findings])
                       if f["channel"] in chans and PHASES.index(ph) in f["phases"]]
                row[f"case:{lab}"] = f"第 {hit[0]} 条" if hit else "未列出"
        # 【易错-66】奇偶拍用同一个参照和同一个基线，只换被评估的拍
        js = []
        for s in nv_subs:
            Xs = X[subj == s]
            fa = build_findings(person_features(refs["full"].score(Xs[0::2], sl), refs["base_novice"]), ch, cfg, pm, sl, sb)
            fb = build_findings(person_features(refs["full"].score(Xs[1::2], sl), refs["base_novice"]), ch, cfg, pm, sl, sb)
            jv = jaccard_top3(fa, fb)
            js.append(jv)
            stab_rows.append(dict(config=cfg["name"], subject=s, jaccard_top3=jv))
        row["stability_jaccard_mean"] = np.nanmean(js) if np.any(np.isfinite(js)) else np.nan
        abl.append(row)
    abl = pd.DataFrame(abl)
    abl.to_csv(out / "ablation.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    pd.DataFrame(stab_rows).to_csv(out / "stability.csv", index=False, float_format="%.3f")

    fig_findings(pe_nv, ch, full, P0, targets, out / "fig_findings.png")

    # 报告
    R = ["# 第 3 层：结构化诊断", "",
         f"通道（{len(ch)}）：{', '.join(ch)}",
         f"熟练者：{', '.join(refs['sk_subs'])}；新手：{', '.join(nv_subs)}",
         f"方法：{', '.join(methods)}；TAU = {TAU}；可信度权重 {TIER_W}；单方法权重 {AGREE_W_SINGLE}",
         f"阈值校准目标：熟练者留一时每人平均问题数 ≤ {TARGET_FP}", ""]
    R += ["## 1. 方向核对"]
    R += direction_check(d, sl)
    missing = [c for c in ch if c not in DIRECTION_TEXT]
    R.append(f"DIRECTION_TEXT 未覆盖的通道：{missing or '无'}")
    R.append("")
    R += ["## 2. 阈值校准（只用熟练者；新手列仅供参考，不参与选择）"]
    for cfg in CONFIGS:
        R.append(f"{cfg['cn']}（{cfg['name']}）：P_MIN = {p_mins[cfg['name']]:.2f}")
    sub = cal[cal["config"] == "full"]
    sub = sub[np.isclose(sub["p_min"] * 20, np.round(sub["p_min"] * 20))]   # 每 0.05 列一行
    nz = sub[(sub["skilled_mean"] > 0) | (sub["novice_mean"] > 0)]
    if len(nz):                                                              # 只列到问题数都归零为止
        sub = sub[sub["p_min"] <= min(nz["p_min"].max() + 0.051, 1.0)]   # 最多列到 1.0，完整见 calibration.csv
    R.append("完整设置的校准曲线：")
    R.append(sub[["p_min", "skilled_mean", "novice_mean"]].to_string(index=False, float_format="%.2f"))
    R.append("")
    R += [f"## 3. 新手问题清单（完整设置，每人最多列 {max_findings} 条）"]
    for s in nv_subs:
        fs = nv_findings[s]
        R.append(f"{s}（{int(pe_nv[s]['_n'])} 拍，共 {len(fs)} 条过阈值）：")
        if not fs:
            R.append("   （没有过阈值的问题）")
        for i, f in enumerate(fs[:max_findings]):
            R.append("   " + human_line(finding_to_dict(f, i + 1, targets)))
        # 不到 3 条时，列出没过阈值但最接近的几条，方便解释"为什么没报"（不进 JSON，不给第 4 层）
        if len(fs) < 3:
            # 阈值为 0 时合并范围会变大，所以按"通道 + 方向"去重，而不是按阶段
            shown = {(f["channel"], f["rep"]["direction"]) for f in fs}
            near = [f for f in build_findings(pe_nv[s], ch, full, 0.0, sl, sb)
                    if (f["channel"], f["rep"]["direction"]) not in shown][:3 - len(fs)]
            for f in near:
                R.append("   （未过阈值）" + human_line(finding_to_dict(f, 0, targets))[3:])
    R.append("")
    R += ["## 4. 验证（完整设置）"]
    a0 = abl[abl["config"] == "full"].iloc[0]
    R.append(f"4.1 误报：熟练者留一平均 {a0['skilled_mean_findings']:.2f} 条；新手平均 {a0['novice_mean_findings']:.2f} 条")
    R.append("    每名熟练者：" + ", ".join(f"{h} {len(build_findings(pe_sk[h], ch, full, P0, sl, sb))}"
                                        for h in refs["sk_subs"]))
    if len(det):
        dc = det[det["config"] == "full"]
        R.append("4.2 注入检出（习惯型，进前 3 条且方向正确的比例）：")
        R.append(dc.pivot_table(index="amp_sd", columns="width", values="hit_top3").rename(
            columns={0.6: "局部", 1.0: "整段"}).to_string(float_format="%.2f"))
        R.append("    按可信度分级（整段 2SD）：" + ", ".join(
            f"{t} 级 {dc[(dc['width'] == 1.0) & (dc['amp_sd'] == 2.0) & (dc['tier'] == t)]['hit_top3'].mean():.2f}"
            for t in (1, 2, 3) if ((dc["tier"] == t)).any()))
    else:
        R.append("4.2 注入检出：已跳过（--skip_detection）")
    R.append("4.3 第 1 步组水平差异出现在几名新手的清单里：")
    for t in targets:
        R.append(f"    {t['label']}：{int(a0['hit:' + t['label']])}/{len(nv_subs)}")
    if not targets:
        R.append("    （没有读到 group_intervals.csv）")
    if CASE_SUBJ:
        R.append(f"4.4 {CASE_SUBJ} 个案：")
        for _, _, lab in LYF_CHECKS:
            if f"case:{lab}" in a0:
                R.append(f"    {lab}：{a0['case:' + lab]}")
    R.append(f"4.5 奇偶拍稳定性（前 3 条的 Jaccard）：平均 {a0['stability_jaccard_mean']:.2f}")
    R.append("")
    R += ["## 5. 消融（每种设置都重新校准阈值）"]
    cols = ["cn", "P_MIN", "skilled_mean_findings", "novice_mean_findings"]
    cols += [c for c in abl.columns if c in ("broad2SD_top3", "local2SD_top3", "broad1SD_top3",
                                               "broad2SD_top3_tier1", "broad2SD_top3_tier2", "broad2SD_top3_tier3")]
    cols += [c for c in abl.columns if c.startswith("hit:") or c.startswith("case:")] + ["stability_jaccard_mean"]
    R.append(abl.set_index("cn")[cols[1:]].T.to_string(float_format=lambda v: f"{v:.2f}"))
    R.append("")
    R += ["## 6. 已有代码的问题", "layer2_eval.loso_skilled 第 127 行：熟练者 h 的基线来自'去掉 g'的参照，"
          "参照里仍含 h 本人（轻微泄漏，只影响第 2 层评测表的熟练者部分）。第 3 层已用嵌套留一处理，未改动第 2 层。"]
    (out / "layer3_report.txt").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--step1", default=None)
    ap.add_argument("--out", default="layer3_out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--max_findings", type=int, default=5)
    ap.add_argument("--skip_detection", action="store_true", help="跳过注入检出（调试用，约省 90% 时间）")
    a = ap.parse_args()
    main(a.prep, a.step1, a.out, a.seed, a.methods.split(","), a.max_findings, a.skip_detection)
