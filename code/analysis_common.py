"""第 1、2 步共用的读数据工具。"""
from pathlib import Path

import numpy as np
import pandas as pd

SKILLED = ("pro", "amateur")
NOVICE = ("novice",)
# v3.1 之前的预处理结果里有这几个已知有问题的通道（见 README v3.1 更新），读入时自动去掉
KNOWN_BAD = ("pelvis_yaw", "thorax_yaw", "x_factor", "shoulder_horiz_R")


# 公开数据包（data/）使用论文里的组名和文件名；这里映射回代码内部的组名，两种目录都能直接读
PUBLIC_GROUPS = {"high-level": "pro", "proficient": "amateur", "novice": "novice"}
PUBLIC_NPZ = "joint_angles_normalised.npz"


def load(prep_dir, npz_name="angles_norm.npz", drop=KNOWN_BAD):
    f = Path(prep_dir) / npz_name
    if not f.exists() and (Path(prep_dir) / PUBLIC_NPZ).exists():
        f = Path(prep_dir) / PUBLIC_NPZ
    z = np.load(f, allow_pickle=True)
    ok = z["qc_pass"].astype(bool)
    ch = list(z["channels"])
    keep = [i for i, c in enumerate(ch) if c not in drop]
    group = np.array([PUBLIC_GROUPS.get(g, g) for g in z["group"]])
    d = dict(X=z["X"][ok][:, keep].astype(float), subj=z["subject"][ok], group=group[ok],
             keys=z["keys"][ok], channels=[ch[i] for i in keep], seg_bounds=z["seg_bounds"],
             dropped=[c for c in ch if c in drop])
    meta = pd.read_csv(Path(prep_dir) / "strokes_meta.csv")
    meta = meta[meta["qc_pass"] == True].set_index("key")  # noqa: E712
    if "group" in meta:
        meta["group"] = meta["group"].map(lambda g: PUBLIC_GROUPS.get(g, g))
    return d, meta


def natsorted(xs):
    """受试者编号按"字母 + 数字"自然排序（S2 在 S10 前）。内部编号 A1…D5 下与普通排序相同；
    公开编号 S1…S10 下保证与内部编号同序，注入检测的随机种子因此与论文一致。"""
    import re
    return sorted(xs, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))])


def subject_means(X, subj):
    subs = sorted(np.unique(subj))
    return subs, np.stack([X[subj == s].mean(axis=0) for s in subs])


def phase_of(t, seg_bounds, names=("backswing", "forward", "follow")):
    for i in range(len(seg_bounds) - 1):
        if seg_bounds[i] <= t < seg_bounds[i + 1]:
            return names[i]
    return names[-1]


def pct_of_phase(t, seg_bounds):
    """归一化时间点 -> '前挥 40%' 这种可读描述。"""
    for i in range(len(seg_bounds) - 1):
        a, b = seg_bounds[i], seg_bounds[i + 1]
        if a <= t < b or (i == len(seg_bounds) - 2 and t == b):
            return 100.0 * (t - a) / (b - a)
    return np.nan
