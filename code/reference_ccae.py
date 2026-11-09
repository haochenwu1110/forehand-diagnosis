"""M3：约束卷积自编码投影（Constrained Convolutional Auto-Encoder projection, CCAE）。

和 M2（PCA 投影）回答同一个问题——"在熟练者的动作空间里，离这一拍最近、又不超出熟练者风格范围的动作长什么样"，
区别有两点：
  1. 动作空间是非线性的：用一维卷积自编码器沿时间轴学习熟练动作（D2L 第 6 章卷积 + 第 14 章自编码器思想），
     PCA 只能表示"几条固定曲线的线性组合"。
  2. 重建时加"可执行条件"：重建出的参照动作必须满足
         θ_min,c ≤ θ_c(t) ≤ θ_max,c        （关节活动范围）
         |dθ_c/dt| ≤ q̇_max,c               （关节角速度上限）
     违反就在损失里罚，求解的是
         z* = argmin_z ‖x − D(z)‖²_S + λ·V(D(z))，  s.t. |z_k| ≤ clip·σ_k
     其中 ‖·‖_S 为按典型变异幅度标准化后的平方误差，V 为约束违反量，σ_k 为熟练者潜变量的标准差。
     这对应"在个人能力限制下给出可执行的参照动作"的思路。

【重要-80】这一版的 θ_min/θ_max/q̇_max 由参照组（熟练者）估计，是"熟练动作的可执行范围"；
  接口 set_limits() 预留了个人上限：将来有了单关节最大发力测试的数据，可以按人传入，参照就变成"这个人能做到的熟练动作"。
  不要用被评估者自己击球数据里的最大值当他的上限——那是受技术限制的，会把要诊断的错误当成"能力不足"吸收掉。
【重要-81】潜变量截断与 M2 的系数截断同理：不截断，自编码器也能把新手的错误重建出来，残差就看不出问题。
【易错-80】缩放系数、潜变量标准差、约束范围都只能用参照组算（留一时不含被评估者）。
【易错-81】训练集只有约 180–230 拍，网络要小（约 5 万参数），并用权重衰减；潜变量维数默认 8。
  训练有随机性，固定 seed 才可复现。

接口与 reference.py 中其他方法一致：fit(X_ref, subj_ref)、reconstruct(X)。
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # noqa
    raise ImportError("M3 需要 PyTorch：在 d2l 环境里运行（conda activate d2l）") from e

import reference as RF


class _Net(nn.Module):
    """Conv1d 编码器（14 通道 × 125 点 → 潜变量）+ 转置卷积解码器。"""

    def __init__(self, C, T, latent, width=32):
        super().__init__()
        self.T = T
        self.enc = nn.Sequential(
            nn.Conv1d(C, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU(),     # T → ⌈T/2⌉
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU(),     # → ⌈T/4⌉
        )
        with torch.no_grad():
            self.Tq = self.enc(torch.zeros(1, C, T)).shape[-1]
        self.width = width
        self.to_z = nn.Linear(width * self.Tq, latent)
        self.from_z = nn.Linear(latent, width * self.Tq)
        self.dec = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose1d(width, width, 4, stride=2, padding=1), nn.GELU(),   # ×2
            nn.ConvTranspose1d(width, width, 4, stride=2, padding=1), nn.GELU(),   # ×2
            nn.Conv1d(width, C, 5, padding=2),
        )

    def encode(self, x):
        return self.to_z(self.enc(x).flatten(1))

    def decode(self, z):
        h = self.from_z(z).view(len(z), self.width, self.Tq)
        return self.dec(h)[..., :self.T]


class ConstrainedConvAE:
    name = "M3_ccae"

    def __init__(self, latent=8, epochs=400, lr=3e-3, weight_decay=1e-4, clip=2.5, lam=100.0,
                 n_opt=60, opt_lr=0.05, margin_rom=0.5, margin_vel=1.2, seed=0, device=None):
        self.latent, self.epochs, self.lr, self.wd = latent, epochs, lr, weight_decay
        self.clip, self.lam, self.n_opt, self.opt_lr = clip, lam, n_opt, opt_lr
        self.margin_rom, self.margin_vel, self.seed = margin_rom, margin_vel, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stats = {}

    # ------------------------------------------------------------ 训练
    def fit(self, X, subj):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.sc = RF.Scaler().fit(X)                       # 【易错-80】只用参照组
        Z = torch.tensor(self.sc.fwd(X), dtype=torch.float32, device=self.device)
        n, C, T = Z.shape
        self.net = _Net(C, T, self.latent).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        # 按人平衡：每人的拍权重 = 1 / 该人拍数，避免拍多的人主导"熟练动作空间"
        _, inv, cnt = np.unique(subj, return_inverse=True, return_counts=True)
        w = torch.tensor(1.0 / cnt[inv], dtype=torch.float32, device=self.device)
        w = w / w.sum()
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            rec = self.net.decode(self.net.encode(Z))
            loss = (w * ((rec - Z) ** 2).mean(dim=(1, 2))).sum()
            loss.backward()
            opt.step()
            sched.step()
        self.net.eval()
        for p in self.net.parameters():            # 训练完冻结权重：重建时只优化潜变量
            p.requires_grad_(False)
        with torch.no_grad():
            zr = self.net.encode(Z)
            self.z_sd = zr.std(0) + 1e-6
            self.z_mu = zr.mean(0)
            self.stats["train_mse"] = float(((self.net.decode(zr) - Z) ** 2).mean())
        # 可执行条件（度）：关节活动范围 + 角速度上限（按归一化时间的相邻点差）
        s = self.sc.s.ravel()
        self.rom_lo = X.min(axis=(0, 2)) - self.margin_rom * s
        self.rom_hi = X.max(axis=(0, 2)) + self.margin_rom * s
        self.vel_max = np.abs(np.diff(X, axis=2)).max(axis=(0, 2)) * self.margin_vel
        return self

    def set_limits(self, rom_lo=None, rom_hi=None, vel_max=None):
        """【重要-80】预留：按人传入单关节测试得到的活动范围和角速度上限（单位：度、度/归一化点）。"""
        if rom_lo is not None:
            self.rom_lo = np.asarray(rom_lo, float)
        if rom_hi is not None:
            self.rom_hi = np.asarray(rom_hi, float)
        if vel_max is not None:
            self.vel_max = np.asarray(vel_max, float)

    # ------------------------------------------------------------ 约束
    def _prepare_limits(self):
        t = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float32, device=self.device)  # noqa: E731
        self._s, self._mu = t(self.sc.s), t(self.sc.mu)                              # [C,1]、[C,T]
        self._lo, self._hi = t(self.rom_lo)[:, None], t(self.rom_hi)[:, None]        # [C,1]
        self._vm = t(self.vel_max)[:, None]

    def _violation(self, Zhat):
        """约束违反量 V，按典型变异幅度标准化（与重建误差同一量纲）。返回每拍一个数。"""
        s, mu, lo, hi, vm = self._s, self._mu, self._lo, self._hi, self._vm
        th = Zhat * s + mu                                                            # 度
        v_rom = (torch.relu(th - hi) ** 2 + torch.relu(lo - th) ** 2) / s ** 2
        v_vel = torch.relu(th.diff(dim=-1).abs() - vm) ** 2 / s ** 2
        return v_rom.mean(dim=(1, 2)) + v_vel.mean(dim=(1, 2))

    # ------------------------------------------------------------ 重建（约束潜空间投影）
    def reconstruct(self, X):
        self._prepare_limits()
        Z = torch.tensor(self.sc.fwd(X), dtype=torch.float32, device=self.device)
        box = self.clip * self.z_sd
        with torch.no_grad():
            z0 = self.net.encode(Z)
            z0 = torch.max(torch.min(z0, self.z_mu + box), self.z_mu - box)           # 【重要-81】截断
            v_before = self._violation(self.net.decode(z0))
        # 【易错-82】λ 的量级：V 是对全部 14×125 个点取平均，少数几个点越界时 V 只有 1e-3 量级，
        # λ = 1 时约束几乎不起作用（v7 自检中加不加约束结果完全相同）。λ = 100 才能让越界点被压回范围内。
        z = z0.clone().requires_grad_(True)
        if self.n_opt > 0:
            opt = torch.optim.Adam([z], lr=self.opt_lr)
            for _ in range(self.n_opt):
                opt.zero_grad()
                rec = self.net.decode(z)
                loss = ((rec - Z) ** 2).mean(dim=(1, 2)) + self.lam * self._violation(rec)
                loss.sum().backward()
                opt.step()
                with torch.no_grad():                                                  # 投影回截断盒
                    z.copy_(torch.max(torch.min(z, self.z_mu + box), self.z_mu - box))
        with torch.no_grad():
            rec = self.net.decode(z)
            v_after = self._violation(rec)
            tol = 1e-6
            self.stats.update(active_before=float((v_before > tol).float().mean()),
                              active_after=float((v_after > tol).float().mean()),
                              viol_before=float(v_before.mean()), viol_after=float(v_after.mean()),
                              clipped=float(((z - self.z_mu).abs() >= box - 1e-6).any(1).float().mean()))
        return self.sc.inv(rec.cpu().numpy().astype(float))
