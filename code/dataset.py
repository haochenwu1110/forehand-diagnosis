"""PyTorch Dataset：读 angles_norm.npz，按受试者划分训练 / 测试。

用法示例：
    from dataset import load_npz, loso_splits, StrokeDataset
    d = load_npz("prep_out/angles_norm.npz")
    for test_subj, tr_idx, te_idx in loso_splits(d, train_groups=["pro"]):
        train_ds = StrokeDataset(d, tr_idx)                 # 归一化参数只用训练集算
        test_ds = StrokeDataset(d, te_idx, stats=train_ds.stats)
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def load_npz(path, only_qc_pass=True):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    if only_qc_pass:
        m = d["qc_pass"].astype(bool)
        for k in ("X", "keys", "subject", "session", "group", "qc_pass"):
            d[k] = d[k][m]
    return d


def loso_splits(d, train_groups=("pro",)):
    """留一受试者交叉验证。
    训练集：train_groups 里除测试者以外的所有人。
    测试集：被留出的那个人（任何组）。不在 train_groups 里的人每次都只当测试。"""
    subjects = np.unique(d["subject"])
    for s in subjects:
        te = np.flatnonzero(d["subject"] == s)
        tr = np.flatnonzero(np.isin(d["group"], train_groups) & (d["subject"] != s))
        if len(tr):
            yield s, tr, te


class StrokeDataset(Dataset):
    """每个样本：x (通道, 时间) float32，已按通道 z 标准化；附带受试者和组别。
    stats=None 时用本数据集计算均值/标准差（只应在训练集上这样做）。"""

    def __init__(self, d, idx, stats=None):
        X = d["X"][idx].astype(np.float32)
        if stats is None:
            mu = X.mean(axis=(0, 2), keepdims=True)[0]
            sd = X.std(axis=(0, 2), keepdims=True)[0] + 1e-6
            stats = (mu, sd)
        self.stats = stats
        self.X = torch.from_numpy((X - stats[0]) / stats[1])
        self.subject = d["subject"][idx]
        self.group = d["group"][idx]
        self.keys = d["keys"][idx]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]

    def to_degrees(self, x):
        """把模型输出（标准化空间）换回角度。"""
        mu, sd = (torch.as_tensor(s) for s in self.stats)
        return x * sd + mu
