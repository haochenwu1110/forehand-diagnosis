"""把"21 关节坐标 CSV"（例如 A1-__Data.csv）转换成 Paper B 流程的输入（prep 目录）。

    conda activate d2l
    python convert_csv.py --in_dir <放 CSV 的文件夹> --groups groups.csv --fps 60 --out <prep_new>
    # 之后照常：
    PAPERB_UNIFORM_TIER=1 python localize.py --prep_dir <prep_new> --out_dir <step1_new>
    PAPERB_UNIFORM_TIER=1 python layer2_eval.py ...   /   layer3.py ...（参数和原来一样，只换 prep 目录）

输入格式（按 A1-__Data.csv 反推，见 README_新数据.md）：
  - 无表头，每行一帧，87 列：前 63 列 = 21 个关节 × (x, y, z) 米，y 朝上；后 24 列 = 8 组 × 3（含义未知，只记录峰值）
  - 21 个关节顺序（Xsens 式，去掉脚尖）：0 骨盆, 1–3 脊柱, 4 颈, 5–6 头, 7 右锁骨, 8 右肩, 9 右肘, 10 右腕,
    11 左锁骨, 12 左肩, 13 左肘, 14 左腕, 15 右髋, 16 右膝, 17 右踝, 18 左髋, 19 左膝, 20 左踝
  - 一个文件 = 一名受试者，多拍首尾相接（A1 是 21 拍 × 61 帧，每拍以击球附近为中心）
groups.csv：两列 subject,group；group 只能是 pro / amateur / novice（前两个算熟练组），填 exclude 则整人剔除。
受试者编号默认取文件名开头的字母+数字（A1-__Data.csv → A1）。
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import config as C
import kinematics as K

N_JOINTS = 21
JOINT_MAP = {
    "pelvis": 0, "spine1": 1, "spine2": 2, "spine3": 3, "neck": 4, "head": 6,
    "right_collar": 7, "right_shoulder": 8, "right_elbow": 9, "right_wrist": 10, "right_hand": 10,
    "left_collar": 11, "left_shoulder": 12, "left_elbow": 13, "left_wrist": 14, "left_hand": 14,
    "right_hip": 15, "right_knee": 16, "right_ankle": 17, "right_foot": 17,
    "left_hip": 18, "left_knee": 19, "left_ankle": 20, "left_foot": 20,
}


def split_strokes(F, stroke_len=None):
    """按骨盆位置的跳变切分首尾相接的多拍；给了 stroke_len 就按固定长度切。"""
    if stroke_len:
        n = len(F) // stroke_len
        return [(i * stroke_len, (i + 1) * stroke_len) for i in range(n)]
    pel = F[:, 0:3]
    step = np.linalg.norm(np.diff(pel, axis=0), axis=1)
    thr = max(0.10, 10 * np.median(step))
    cuts = np.where(step > thr)[0] + 1
    if len(cuts) >= 2:
        # 多数切点落在同一固定长度的整数倍上 → 认为每拍等长（小跳变也能切开）
        L = int(pd.Series(np.diff([0, *cuts])).mode().iloc[0])
        if L >= 20 and len(F) % L == 0 and np.mean(cuts % L == 0) >= 0.8:
            return [(i * L, (i + 1) * L) for i in range(len(F) // L)]
    bounds = [0, *cuts.tolist(), len(F)]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 20]


def lowpass(P, fs, fc):
    if fc is None or fc <= 0 or fc >= fs / 2:
        return P
    b, a = butter(C.FILTER_ORDER, fc / (fs / 2))
    return filtfilt(b, a, P, axis=0, padlen=min(3 * max(len(a), len(b)), len(P) - 1))


def detect_events(D, fs, t1_mode):
    """t3 = 右肘线速度峰值；t2 = 右腕沿挥拍方向最靠后（相对骨盆）；t1 = 窗口起点或动作起始。返回帧号（浮点）。"""
    n = len(D["pelvis"])
    ve = np.linalg.norm(np.gradient(D["right_elbow"], axis=0), axis=1)
    lo, hi = int(0.15 * n), int(0.9 * n)
    t3 = int(np.argmax(ve[lo:hi]) + lo)
    vw = np.gradient(D["right_wrist"], axis=0)[t3]
    dirn = K.unit(K.horizontal(vw))
    proj = np.sum((D["right_wrist"] - D["pelvis"]) * dirn, -1)
    t2 = int(np.argmin(proj[:t3])) if t3 > 0 else 0
    t1 = 0
    if t1_mode == "onset" and t2 > 3:
        # 与原流程同一口径：关节速度合成量越过 P20 + 0.2 × (P85 − P20)，从 t2 往前找最后一次低于阈值的帧
        joints = ("right_wrist", "right_elbow", "left_wrist", "right_knee", "left_knee", "neck")
        act = sum(np.linalg.norm(np.gradient(D[j] - D["pelvis"], axis=0), axis=1) for j in joints) * fs
        p20, p85 = np.percentile(act, [20, 85])
        thr = p20 + 0.2 * (p85 - p20)
        below = np.where(act[:t2] < thr)[0]
        t1 = int(below[-1]) if len(below) else 0
    return dict(t1=float(t1), t2=float(t2), t3=float(t3))


def normalise(series_dict, ev, fs, follow_s):
    """分段时间归一化：50 / 50 / 25 点（与 config.SEGMENTS 相同）。返回 (C, 125) 或 None。"""
    n = len(next(iter(series_dict.values())))
    tg = np.arange(n) / fs
    t = {k: v / fs for k, v in ev.items()}
    t_end = t["t3"] + follow_s
    if t_end > tg[-1] + 1e-9 or not (t["t1"] < t["t2"] < t["t3"]):
        return None, t_end * fs
    segs = [(t["t1"], t["t2"], 50), (t["t2"], t["t3"], 50), (t["t3"], t_end, 25)]
    tq = np.concatenate([np.linspace(a, b, m, endpoint=(i == 2)) for i, (a, b, m) in enumerate(segs)])
    return np.stack([np.interp(tq, tg, series_dict[c]) for c in C.CHANNELS]), t_end * fs


def stroke_angles(P, fs, t1_mode, mirror):
    D = {k: P[:, j] for k, j in JOINT_MAP.items()}
    if mirror:
        D = K.mirror_left_right(D)
    ev = detect_events(D, fs, t1_mode)
    A = K.compute_angles(D)
    i1 = int(ev["t1"])
    # 骨盆 / 躯干转动：骨架髋连线、肩连线的水平朝向，相对 t1，开合方向为正（与原流程 IMU 转角同号）
    A["pelvis_rot_imu"] = A["pelvis_yaw"] - A["pelvis_yaw"][i1]
    A["thorax_rot_imu"] = A["thorax_yaw"] - A["thorax_yaw"][i1]
    A["separation_imu"] = A["thorax_rot_imu"] - A["pelvis_rot_imu"]
    return A, ev, D


def check_frame(P_all, segs, hand):
    """坐标系检查：用右手系（前 = 上 × 右）时，每拍持拍手速度峰值处应当向前运动。只在每拍内部求导，
    避免把拍与拍之间的跳变当成速度。hand = 10（右腕）或 14（左腕，左手持拍时前向取反）。"""
    votes = []
    for s0, s1 in segs:
        P = P_all[s0:s1]
        lat = K.horizontal(P[:, 15] - P[:, 18])
        _, _, Fw = K.segment_frame(lat, np.broadcast_to(K.UP, lat.shape))
        v = np.gradient(P[:, hand], axis=0)
        k = int(np.argmax(np.linalg.norm(v, axis=1)))
        votes.append(np.sign(np.sum(v[k] * Fw[k])))
    return float(np.sign(np.sum(votes)))


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.in_dir).glob(a.pattern))
    if not files:
        raise SystemExit(f"{a.in_dir} 里没有找到 {a.pattern}")
    gmap = {}
    if a.groups:
        g = pd.read_csv(a.groups)
        gmap = dict(zip(g["subject"].astype(str), g["group"].astype(str)))
    rx = re.compile(a.subject_regex)
    fs = a.fps
    step_max = C.MAX_ANGLE_STEP_DEG * 60.0 / fs
    metas, Xs, keys, rep = [], [], [], []
    rep.append(f"# 新数据转换报告\n\n帧率 {fs:g} Hz（--fps，请核对数据说明）；t1 = {a.t1}；随挥 = t3 + {a.follow_s:g} s；"
               f"低通 {a.lowpass:g} Hz\n")
    for f in files:
        m = rx.match(f.stem)
        subj = m.group(1) if m else f.stem
        group = gmap.get(subj, a.default_group)
        if group is None:
            raise SystemExit(f"groups.csv 里没有 {subj}，请补上（或临时用 --default_group）")
        if group == "exclude":
            rep.append(f"- {f.name} → {subj}：groups.csv 标为 exclude，整人剔除")
            continue
        F = np.loadtxt(f, delimiter=",", ndmin=2)
        if F.shape[1] < 63:
            raise SystemExit(f"{f.name}: 只有 {F.shape[1]} 列，至少需要 63 列（21 关节 × 3）")
        segs = split_strokes(F, a.stroke_len)
        lens = sorted({b - s for s, b in segs})
        P_all = F[:, :63].reshape(-1, N_JOINTS, 3).copy()
        # 持拍手：每拍内右腕 / 左腕峰值速度谁大（拍内求导）
        pk = lambda j: np.median([np.linalg.norm(np.diff(P_all[s0:s1, j], axis=0), axis=1).max() for s0, s1 in segs])
        mirror = pk(14) > pk(10)
        # 左手持拍时镜像会交换左右，前向判断要反过来
        flip = check_frame(P_all, segs, 14 if mirror else 10) * (-1 if mirror else 1) < 0
        if flip:                                    # 左手系：z 取反
            P_all[:, :, 2] *= -1
        rep.append(f"- {f.name} → {subj}（{group}）：{len(F)} 帧，切出 {len(segs)} 拍（每拍帧数 {lens}）"
                   + ("；坐标为左手系，已把 z 取反" if flip else "") + ("；判定为左手持拍，已左右镜像" if mirror else ""))
        n_pass = 0
        for i, (s0, s1) in enumerate(segs):
            P = lowpass(P_all[s0:s1], fs, a.lowpass)
            A, ev, D = stroke_angles(P, fs, a.t1, mirror)
            Xn, t_end = normalise(A, ev, fs, a.follow_s)
            fails = []
            if Xn is None:
                fails.append("events_or_window" if t_end <= len(P) - 1 + 1e-9 else "follow_beyond_window")
            else:
                if not np.isfinite(Xn).all():
                    fails.append("nan")
                for c, (lo, hi) in C.PLAUSIBLE_RANGES.items():
                    if c in C.CHANNELS:
                        v = Xn[C.CHANNELS.index(c)]
                        if (v < lo).any() or (v > hi).any():
                            fails.append(f"range_{c}")
                raw = np.stack([A[c] for c in C.CHANNELS])
                if (np.abs(np.diff(raw, axis=1)) > step_max).any():
                    fails.append("angle_jump")
            key = f"{subj}_{i}"
            meta = dict(key=key, subject=subj, session=f.stem, group=group, stroke=i, n_frames=s1 - s0,
                        t1=ev["t1"], t2=ev["t2"], t3=ev["t3"], follow_end=t_end,
                        dur_backswing=(ev["t2"] - ev["t1"]) / fs, dur_forward=(ev["t3"] - ev["t2"]) / fs,
                        t1_at_window_start=ev["t1"] == 0, mirrored=bool(mirror), z_flipped=bool(flip),
                        qc_hard=";".join(fails), qc_pass=not fails)
            if F.shape[1] >= 87:
                E = F[s0:s1, 63:87].reshape(-1, 8, 3)
                for k in range(8):
                    meta[f"extra{k}_peak"] = float(np.linalg.norm(E[:, k], axis=1).max())
            metas.append(meta)
            if Xn is not None:
                Xs.append(Xn)
                keys.append(key)
            n_pass += not fails
        rep.append(f"  通过质量控制 {n_pass}/{len(segs)} 拍")
    meta = pd.DataFrame(metas)
    for col in ("dur_backswing", "dur_forward"):
        med = meta.groupby("subject")[col].transform("median")
        mad = meta.groupby("subject")[col].transform(lambda x: np.median(np.abs(x - np.median(x)))) * 1.4826
        z = (meta[col] - med) / mad.replace(0, np.nan)
        bad = z.abs() > C.PHASE_OUTLIER_Z
        meta.loc[bad, "qc_hard"] = (meta.loc[bad, "qc_hard"] + f";{col}_outlier").str.strip(";")
        meta.loc[bad, "qc_pass"] = False
    meta.to_csv(out / "strokes_meta.csv", index=False, float_format="%.5g")
    if not Xs:
        raise SystemExit("没有任何一拍能完成时间归一化，检查 --fps 和 --follow_s（报告见 conversion_report.md）")
    mk = meta.set_index("key").loc[keys]
    np.savez_compressed(
        out / "angles_norm.npz", X=np.stack(Xs), keys=np.array(keys), channels=np.array(C.CHANNELS),
        subject=mk["subject"].to_numpy(), session=mk["session"].to_numpy(), group=mk["group"].to_numpy(),
        qc_pass=mk["qc_pass"].to_numpy(bool), seg_bounds=np.cumsum([0, 50, 50, 25]),
        seg_names=np.array(["t1->t2", "t2->t3", "t3->t3+%g" % a.follow_s]))

    ok = meta[meta["qc_pass"]]
    rep += ["", "## 汇总", "",
            ok.groupby(["group", "subject"]).size().rename("通过拍数").to_frame().to_string(), "",
            f"未通过原因：{meta.loc[~meta['qc_pass'], 'qc_hard'].str.split(';').explode().value_counts().to_dict()}",
            f"t1 落在窗口起点的拍：{int(meta['t1_at_window_start'].sum())}/{len(meta)}（窗口没包含完整引拍时，"
            "引拍段只是'窗口起点到 t2'，各拍口径一致，但论文里要写明）",
            f"事件帧号中位数：t1 {meta['t1'].median():.0f}，t2 {meta['t2'].median():.0f}，t3 {meta['t3'].median():.0f}，"
            f"随挥终点 {meta['follow_end'].median():.0f}",
            f"前挥时长中位数 {meta['dur_forward'].median() * 1000:.0f} ms（原数据约 300 ms；差得很多说明 --fps 可能不对）",
            "", "## 与原流程的区别（论文里要写）",
            "- 14 个通道名沿用原流程（pelvis_rot_imu 等），但这里全部由骨架坐标算出，骨盆 / 躯干转动来自髋连线 / 肩连线",
            "- 全部关节来自同一套系统，可信度分级不再成立：后续脚本请加环境变量 PAPERB_UNIFORM_TIER=1（全部按 1 级）",
            "- 后 24 列含义未知，只把每组 3 列的模长峰值写进 strokes_meta.csv（extra0_peak … extra7_peak）"]
    (out / "conversion_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    print(f"\n完成：{out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--groups", default=None, help="CSV：subject,group（pro / amateur / novice）")
    ap.add_argument("--default_group", default=None, help="没写进 groups.csv 的人用这个组（只用于试跑）")
    ap.add_argument("--subject_regex", default=r"^([A-Za-z]+\d+)")
    ap.add_argument("--fps", type=float, default=60.0, help="采样频率，请按数据说明填写")
    ap.add_argument("--stroke_len", type=int, default=None, help="每拍固定帧数；不填则按骨盆跳变自动切分")
    ap.add_argument("--t1", choices=["start", "onset"], default="start", help="引拍起点：窗口起点 / 动作起始")
    ap.add_argument("--follow_s", type=float, default=0.4, help="随挥长度（秒），原流程 0.4")
    ap.add_argument("--lowpass", type=float, default=C.LOWPASS_HZ)
    ap.add_argument("--out", default="prep_new")
    main(ap.parse_args())
