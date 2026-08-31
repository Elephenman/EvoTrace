# -*- coding: utf-8 -*-
"""PprI SurrogateOracle v3：ESM3 特征版 DL 代理 → b7 优化器可直接用。

依赖: surrogate_dl_v3/model.pt + pca_components/mean.npy + ppri_wt.npz（ESM3 emb）
接口对齐 surrogate_oracle.py: .L .max_mut .wt_idx .wt_f .evaluate(genos)
"""
import os
import numpy as np
import torch
import torch.nn as nn

AA = "ACDEFGHIKLMNPQRSTVWY"
AAI = {a: i for i, a in enumerate(AA)}
WIN = 2

def _load_ppri_sites():
    """从 priors.csv 读取 53 个可突变位点（seq_idx，0-based 序列索引）。"""
    import csv
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "ppri_evo", "results", "priors.csv")
    if not os.path.exists(path):
        path = "A:/claudework/ppri_evo/results/priors.csv"
    rows = list(csv.DictReader(open(path)))
    return sorted({int(r["seq_idx"]) for r in rows})


PPRI_SITES = _load_ppri_sites()


class DeepSetV3(nn.Module):
    def __init__(self, pca_dim=64, h=128, hidden=256):
        super().__init__()
        xd = 1 + 20 + 20 + pca_dim * (2 * WIN + 1)
        self.enc = nn.Sequential(nn.Linear(xd, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * h, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, site_idx, rows, site_ctx):
        B, K, _ = rows.shape
        valid = rows[:, :, 0] >= 0
        posf = rows[:, :, 0].clamp(min=0).float() / max(len(site_idx) - 1, 1)
        posf = posf * valid.float()
        ctx = site_ctx[rows[:, :, 0].clamp(min=0)]
        waa = torch.zeros(B, K, 20, device=rows.device)
        maa = torch.zeros(B, K, 20, device=rows.device)
        waa.scatter_(2, rows[:, :, 1].clamp(min=0).unsqueeze(-1), 1.0)
        maa.scatter_(2, rows[:, :, 2].clamp(min=0).unsqueeze(-1), 1.0)
        x = torch.cat([posf.unsqueeze(-1), waa, maa, ctx.flatten(2) * valid.unsqueeze(-1)], dim=-1)
        e = self.enc(x) * valid.unsqueeze(-1)
        pooled = torch.cat([e.sum(1) / valid.sum(1, keepdim=True).clamp(min=1),
                            e.max(1).values], dim=-1)
        return self.head(pooled).squeeze(-1)


class PprISurrogateV3:
    """PprI 53 位点 ESM3 特征代理（零样本，直接用 DL 模型预测 z-score fitness）。"""

    def __init__(self, model_dir="A:/claudework/out/surrogate_dl_v3",
                 emb_file="A:/claudework/out/esm3_embeddings/ppri_wt.npz",
                 wt_seq_file="A:/claudework/ppri_evo/inputs/wt_254.fasta"):
        wt_seq = open(wt_seq_file).read().splitlines()[1].strip()
        self.wt_seq = wt_seq
        self.L = len(PPRI_SITES)
        self.max_mut = 12
        self.sites = np.array(PPRI_SITES, dtype=np.int64)  # 全局残基号 1-based
        # ESM3 embedding → PCA
        emb = np.load(emb_file)["emb"].astype(np.float32)  # (254, 1536)
        pca_mean = np.load(os.path.join(model_dir, "pca_mean.npy"))
        pca_comp = np.load(os.path.join(model_dir, "pca_components.npy"))
        pca_dim = pca_comp.shape[0]
        self.pca_dim = pca_dim
        # site_ctx: [S, 2W+1, pca_dim] —— sites 是 seq_idx（0-based），直接索引序列
        S = len(self.sites)
        ctx = np.zeros((S, 2 * WIN + 1, pca_dim), dtype=np.float32)
        for j, idx in enumerate(self.sites):  # idx = 序列 0-based
            if not (0 <= idx < len(wt_seq)):
                continue
            for dd, q in enumerate(range(idx - WIN, idx + WIN + 1)):
                if 0 <= q < len(wt_seq):
                    v = (emb[q] - pca_mean) @ pca_comp.T
                    ctx[j, dd] = v
        self.site_ctx = torch.from_numpy(ctx)
        self.site_idx = torch.arange(S, dtype=torch.long)
        # WT 基因型（sites 是 seq_idx）
        self.wt_idx = np.array([AAI[wt_seq[s]] for s in self.sites], dtype=np.int64)
        # 模型
        self.dev = torch.device("cpu")
        self.model = DeepSetV3(pca_dim=pca_dim)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"),
                                              map_location="cpu"))
        self.model.eval()
        # WT fitness（预测）
        self.wt_f = float(self.evaluate(self.wt_idx[None, :])[0])
        # ref fitness（全 19 替换最大预测，用于 norm）
        self.ref_f = float(self._ref_fitness())

    def _ref_fitness(self):
        best = -np.inf
        B = 1007
        for base in range(0, self.L * 19, B):
            rows_idx = np.arange(base, min(base + B, self.L * 19))
            site_i = rows_idx // 19
            aa_i = rows_idx % 19
            genos = np.tile(self.wt_idx, (len(rows_idx), 1))
            genos[rows_idx - base, site_i] = aa_i
            f = self.evaluate(genos)
            best = max(best, float(f.max()))
        return best

    def evaluate(self, genos):
        """genos: (N, L) AA 索引（0..19）→ fitness z-score。"""
        genos = np.asarray(genos, dtype=np.int64)
        N, L = genos.shape
        assert L == self.L, (L, self.L)
        rows = np.full((N, 1, 3), -1, dtype=np.int64)
        for i in range(N):
            m = (genos[i] != self.wt_idx)
            if not m.any():
                rows[i, 0] = (-1, 0, 0)
                continue
            # 单突变样本（K=1；多突变截断到 1，与训练一致 94% 单点）
            j = np.where(m)[0][0]
            rows[i, 0] = (int(j), int(self.wt_idx[j]), int(genos[i, j]))
        t_rows = torch.from_numpy(rows)
        with torch.no_grad():
            pred = self.model(self.site_idx, t_rows, self.site_ctx)
        return pred.numpy()

    def evaluate_multi(self, genos):
        """多突变完整评估（K≤8），供优化器真实使用。"""
        genos = np.asarray(genos, dtype=np.int64)
        N, L = genos.shape
        rows = np.full((N, 8, 3), -1, dtype=np.int64)
        for i in range(N):
            m = np.where(genos[i] != self.wt_idx)[0]
            for t, j in enumerate(m[:8]):
                rows[i, t] = (int(j), int(self.wt_idx[j]), int(genos[i, j]))
        t_rows = torch.from_numpy(rows)
        with torch.no_grad():
            pred = self.model(self.site_idx, t_rows, self.site_ctx)
        return pred.numpy()
