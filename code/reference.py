"""第 2 层：参照检索。给每一拍动作，找一条"按熟练方式打应该是什么样"的参照曲线 x̂。

所有方法共用同一个接口：
    m = Method(...)
    m.fit(X_ref, subj_ref)          # 只用熟练者（参考组）的拍
    X_hat = m.reconstruct(X)        # 对任意拍给出参照曲线，形状同 X：[n, C, T]

    M0 MeanTemplate   熟练者的"平均人"（传统做法）
    M1 KNNRetrieval   从熟练者的拍里检索最像的 k 拍取平均（对应 RAG 的检索）
    M2 PCAProjection  投影到熟练动作的主成分空间，系数限制在熟练者见过的范围内
    M3 自编码器       等学完 D2L 第 6 章再写，接口相同（见文件末尾的占位类）

注释编号从 40 开始。
"""
import numpy as np


class Scaler:
    """【重要-40】按通道标准化：每个关节除以它在熟练者里的典型变异幅度。
    不做这一步，kNN 的距离和 PCA 的方差会被变化幅度大的关节（比如肩抬高，动辄几十度）主导，
    幅度小但同样重要的关节（比如躯干侧倾）几乎不起作用。
    【易错-40】缩放系数只能用参考组（训练集）算，评估时原样套用。"""

    def fit(self, X):
        self.mu = X.mean(axis=0)                                   # [C, T]
        self.s = (X - self.mu).std(axis=(0, 2)).reshape(-1, 1)     # [C, 1]
        self.s = np.maximum(self.s, 1e-6)
        return self

    def fwd(self, X):
        return (X - self.mu) / self.s

    def inv(self, Z):
        return Z * self.s + self.mu


class MeanTemplate:
    """M0：按人平衡的熟练者平均曲线。所有拍得到同一条参照。"""
    name = "M0_mean"

    def fit(self, X, subj):
        self.ref = np.stack([X[subj == s].mean(0) for s in np.unique(subj)]).mean(0)
        return self

    def reconstruct(self, X):
        return np.broadcast_to(self.ref, X.shape).copy()


class KNNRetrieval:
    """M1：在标准化后的空间里，从熟练者的拍中找欧氏距离最近的 k 拍，取平均。
    【重要-41】它回答"熟练者里打法和你最像的那几拍长什么样"，所以风格差异不会被算成错误。
    【易错-41】k 太小（=1）：参照会带上那一拍自己的噪声；k 太大：退化成平均模板（M0）。
    评测里会比较 k = 1, 3, 5, 10，就像 RAG-RUSS 比较检索 1 个和 2 个例子。
    【易错-42】评估熟练者时，参考组里绝不能有他自己的拍，否则最近邻就是他自己（数据泄漏），
    这由评测脚本的留一受试者保证。"""

    def __init__(self, k=5):
        self.k = k
        self.name = f"M1_knn{k}"

    def fit(self, X, subj):
        self.sc = Scaler().fit(X)
        self.Xr = X
        self.Zr = self.sc.fwd(X).reshape(len(X), -1)
        return self

    def neighbors(self, X):
        Z = self.sc.fwd(X).reshape(len(X), -1)
        d2 = (Z ** 2).sum(1)[:, None] + (self.Zr ** 2).sum(1)[None, :] - 2 * Z @ self.Zr.T
        return np.argsort(d2, axis=1)[:, :self.k]

    def reconstruct(self, X):
        nb = self.neighbors(X)
        return self.Xr[nb].mean(axis=1)


class PCAProjection:
    """M2：熟练动作的主成分空间。一拍动作 → 主成分系数 → 截断到熟练者见过的范围 → 重建。
    【重要-42】截断这一步是关键。PCA 的主成分张成的是一个"线性空间"，如果不限制系数，
    它能把任何动作（包括新手的错误）都用这些主成分拼出来，残差就看不出问题了。
    把系数限制在熟练者系数的 ±clip 个标准差以内，意思是"只允许熟练者见过的那种程度的风格变化"，
    超出部分留在残差里，就是偏差。
    【易错-43】主成分数用"累计解释方差达到 var_keep"自动决定。熟练者只有约 230 拍，
    主成分取太多会把噪声也学进来，相当于 D2L 4.4 讲的过拟合。"""

    def __init__(self, var_keep=0.90, clip=2.5):
        self.var_keep, self.clip = var_keep, clip
        self.name = f"M2_pca{int(var_keep * 100)}"

    def fit(self, X, subj):
        self.sc = Scaler().fit(X)
        Z = self.sc.fwd(X).reshape(len(X), -1)
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        ratio = np.cumsum(S ** 2) / np.sum(S ** 2)
        self.n_comp = int(np.searchsorted(ratio, self.var_keep) + 1)
        self.V = Vt[:self.n_comp].T                     # [p, k]
        scores = Z @ self.V
        self.score_sd = scores.std(axis=0) + 1e-9
        self.shape = X.shape[1:]
        return self

    def reconstruct(self, X):
        Z = self.sc.fwd(X).reshape(len(X), -1)
        a = Z @ self.V
        a = np.clip(a, -self.clip * self.score_sd, self.clip * self.score_sd)
        Zh = (a @ self.V.T).reshape((len(X),) + self.shape)
        return self.sc.inv(Zh)


class AutoencoderProjection:
    """M3 已实现，见 reference_ccae.py（ConstrainedConvAE，名称 M3_ccae）。
    放在单独文件里，是为了只用 M0–M2 时不必安装 PyTorch。用 make_method("M3_ccae") 创建。"""
    name = "M3_autoencoder"

    def fit(self, X, subj):
        raise NotImplementedError("请用 make_method('M3_ccae')，实现见 reference_ccae.py")


def make_method(name):
    """按名称创建参照方法：M0_mean、M1_knn{k}、M2_pca{方差百分比}、M3_ccae。"""
    if name == "M0_mean":
        return MeanTemplate()
    if name.startswith("M1_knn"):
        return KNNRetrieval(int(name[6:]))
    if name.startswith("M2_pca"):
        return PCAProjection(int(name[6:]) / 100.0)
    if name.startswith("M3_ccae"):
        from reference_ccae import ConstrainedConvAE      # 需要 PyTorch
        kw = {}
        if name.endswith("_noopt"):                       # 消融：只截断、不做约束投影
            kw["n_opt"] = 0
        if name.endswith("_nocons"):                      # 消融：做潜空间投影，但不加可执行约束
            kw["lam"] = 0.0
        m = ConstrainedConvAE(**kw)
        m.name = name
        return m
    raise ValueError(f"未知方法 {name}")


def default_methods():
    return [MeanTemplate(), KNNRetrieval(1), KNNRetrieval(3), KNNRetrieval(5), KNNRetrieval(10),
            PCAProjection(0.80), PCAProjection(0.90), PCAProjection(0.95), PCAProjection(0.98)]
