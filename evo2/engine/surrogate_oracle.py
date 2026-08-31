# -*- coding: utf-8 -*-
"""SurrogateOracle —— 用 ProteinGym 训练出的跨蛋白 DL 代理（DeepSet）封装成
b7 可评估的 oracle。对任意蛋白（此处为 PprI 53 位点）做零样本适应度预测。

模型权重: A:/claudework/out/surrogate_dl/model.pt（train_surrogate.py 产出）
说明: 输入为 PprI WT 序列 254 残基 + 53 设计位点，输出 z-score 适应度预测。
      DeepSet 结构（位点窗口 one-hot + WT/mut AA）与蛋白无关，可跨蛋白泛化。
"""
import os
import numpy as np
import torch
import torch.nn as nn

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AAI = {a: i for i, a in enumerate(AA20)}
WIN = 2
X_DIM = 1 + 20 + 20 + 20 * (2 * WIN + 1)

MODEL_PT = "A:/claudework/out/surrogate_dl/model.pt"
PPRI_WT_FASTA = "A:/claudework/ppri_evo/inputs/wt_254.fasta"
SITES_NPY = "A:/claudework/ppri_evo/inputs/ppri_sites_53.npy"  # 不存在则用 priors 重建


def read_fasta(fp):
    seqs, name = {}, None
    for line in open(fp, encoding="utf-8"):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = ""
        elif name:
            seqs[name] += line
    return seqs


class DeepSet(nn.Module):
    """与 train_surrogate.py 完全一致的架构（state_dict 兼容）。"""

    def __init__(self, x_dim=X_DIM, h=128, hidden=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(x_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(2 * h, hidden), nn.ReLU(),
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, site_idx, rows, site_win):
        B, K, _ = rows.shape
        valid = rows[:, :, 0] >= 0
        posf = (rows[:, :, 0].clamp(min=0).float()) / max(len(site_idx) - 1, 1)
        posf = posf * valid.float()
        win = site_win[rows[:, :, 0].clamp(min=0)]                # [B,K,2W+1,20]
        waa = torch.zeros(B, K, 20, device=rows.device)
        maa = torch.zeros(B, K, 20, device=rows.device)
        waa.scatter_(2, rows[:, :, 1].clamp(min=0).unsqueeze(-1), 1.0)
        maa.scatter_(2, rows[:, :, 2].clamp(min=0).unsqueeze(-1), 1.0)
        x = torch.cat([posf.unsqueeze(-1), waa, maa,
                       win.flatten(2) * valid.unsqueeze(-1)], dim=-1)
        e = self.enc(x)
        e = e * valid.unsqueeze(-1)
        pooled = torch.cat([e.sum(1) / valid.sum(1, keepdim=True).clamp(min=1),
                            e.max(1).values], dim=-1)
        return self.head(pooled).squeeze(-1)


def build_ppri_surrogate(model_pt=MODEL_PT, wt_fasta=PPRI_WT_FASTA, max_mut=12,
                         sites=None, wt_idx=None):
    """构造 SurrogateOracle：53 位点 / WT 序列 / 模型权重全从本机资产加载。"""
    wt_seq = list(read_fasta(wt_fasta).values())[0]

    if sites is None or wt_idx is None:
        # 从 priors 重建 53 位点（与 build_ppri_additive 同源）
        import sys
        HERE = os.path.dirname(os.path.abspath(__file__))
        for p in (os.path.join(HERE, "..", "benchmarks"),
                  "A:/claudework/ppri_evo"):
            p = os.path.abspath(p)
            if p not in sys.path:
                sys.path.insert(0, p)
        import b5_ppri_wave2 as b5
        pri, cls, wt_aa = b5.load_ppri_priors()
        sites = sorted(pri.keys())
        wt_idx = np.array([AAI[wt_seq[s]] for s in sites])

    L = len(sites)
    assert L == len(wt_idx) == len(set(sites))
    # 窗口特征：每个位点 ±2 残基 one-hot
    wins = []
    for p in sites:
        w = np.zeros((2 * WIN + 1, 20), dtype=np.float32)
        for d, q in enumerate(range(p - WIN, p + WIN + 1)):
            if 0 <= q < len(wt_seq) and wt_seq[q] in AAI:
                w[d, AAI[wt_seq[q]]] = 1
        wins.append(w)
    site_win = torch.from_numpy(np.stack(wins))       # [L, 2W+1, 20]

    model = DeepSet()
    model.load_state_dict(torch.load(model_pt, map_location="cpu"))
    model.eval()
    return SurrogateOracle(model, np.arange(L), site_win, wt_idx,
                           max_mut=max_mut, sites=sites)


class SurrogateOracle:
    """b7 兼容 oracle：evaluate(genos) -> z-score fitness 预测。"""

    name = "surrogate_ppri"
    # 由 b7 工厂设置
    max_mut = 12

    def __init__(self, model, site_pos, site_win, wt_idx, max_mut=12,
                 seed=0, budget=None, sites=None):
        self.model = model
        self.site_pos = np.asarray(site_pos)
        self.site_win = site_win
        self.wt_idx = np.asarray(wt_idx)
        self.L = len(wt_idx)
        self.max_mut = int(max_mut)
        self.n_evals = 0
        self.K = self.max_mut + 1   # rows 最大突变数（含 pad 余量）
        self.sites = np.asarray(sites) if sites is not None else self.site_pos
        self._wt_f = None

    def evaluate(self, genos):
        genos = np.asarray(genos)
        single = genos.ndim == 1
        if single:
            genos = genos[None, :]
        N, L = genos.shape
        rows = np.full((N, self.K, 3), -1, dtype=np.int64)
        for i in range(N):
            diff = np.where(genos[i] != self.wt_idx)[0]
            for t, j in enumerate(diff[: self.K]):
                rows[i, t] = (j, self.wt_idx[j], genos[i, j])
        with torch.no_grad():
            pred = self.model(self.site_pos, torch.from_numpy(rows), self.site_win)
        self.n_evals += N
        out = pred.numpy()
        return out[0] if single else out

    def n_mutations(self, geno):
        geno = np.atleast_2d(geno)
        return (geno != self.wt_idx).sum(1)

    def enforce_max_mut(self, geno, rng):
        """超出 max_mut 的个体：随机把多余位点回退为 WT。"""
        geno = np.atleast_2d(geno).copy()
        if not self.max_mut:
            return geno
        n_mut = self.n_mutations(geno)
        for n in np.flatnonzero(n_mut > self.max_mut):
            ms = np.flatnonzero(geno[n] != self.wt_idx)
            drop = rng.choice(ms, size=int(n_mut[n] - self.max_mut), replace=False)
            geno[n, drop] = self.wt_idx[drop]
        return geno

    def known_optimum(self):
        return None  # 代理景观无解析最优，用 greedy 参考

    @property
    def wt_f(self):
        if self._wt_f is None:
            self._wt_f = float(self.evaluate(self.wt_idx[None, :])[0])
        return self._wt_f
