#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoTrace v2 — 对标基线策略（同一 oracle、同一预算下的进化策略对比）。

1. RandomMutagenesis   经典易错 PCR：随机基因型批次，无模型。
2. RidgeAL             ML-guided dEvo 基线（Wu et al. 2019 PNAS, Arnold lab 家族）：
                       one-hot(site,aa)+突变量特征，ridge 对偶形式（n 标签 × n 核矩阵，
                       O(n³) 而非 O(D³)），bootstrap 不确定度 + UCB 批次。
EvoTrace v2 / v1 的 campaign 行为在 benchmarks 中直接由 kernel+reflux 组合定义。
"""
import numpy as np

from .seqtools import AA20, AA2IDX

AA_IDX = {a: i for i, a in enumerate(AA20)}


class RandomMutagenesis:
    """每轮提出 batch 个随机（≤n_mut_max 突变）基因型。"""

    def __init__(self, sites, rng, n_mut_max=4, proposal="uniform"):
        self.sites = np.asarray(sites, int)
        self.rng = rng
        self.n_mut_max = n_mut_max
        self.proposal = proposal

    def propose(self, batch, prior_table=None, wt_idx=None):
        out = []
        tries = 0
        while len(out) < batch and tries < batch * 20:
            tries += 1
            k = int(self.rng.integers(1, self.n_mut_max + 1))
            js = self.rng.choice(len(self.sites), size=min(k, len(self.sites)), replace=False)
            muts = []
            for j in js:
                if self.proposal == "prior" and prior_table is not None:
                    p = prior_table[j] / prior_table[j].sum()
                    aa = int(self.rng.choice(20, p=p))
                else:
                    aa = int(self.rng.integers(0, 20))
                if aa != wt_idx[j]:
                    muts.append((int(self.sites[j]), AA20[aa]))
            if muts:
                out.append(muts)
        return out


class RidgeAL:
    """对偶 ridge + bootstrap UCB 的主动学习基线。

    特征：one-hot(site, aa) + 突变计数。对偶解 alpha = (K + λI)^{-1} y，
    预测 pred(c) = ȳ + c·X^T alpha。不确定度 = bootstrap alpha 的预测 std。
    """

    def __init__(self, sites, wt_idx, rng, lam=5.0, n_boot=8, feat_fn=None):
        self.sites = np.asarray(sites, int)
        self.wt_idx = np.asarray(wt_idx, int)
        self.rng = rng
        self.lam = lam
        self.n_boot = n_boot
        self.feat_fn = feat_fn  # 可选：附加标量特征（如先验分——学信任权重）
        self.X, self.y, self.X_muts = [], [], []
        self._K = None

    # ---- 特征
    def _feat(self, muts):
        v = np.zeros(len(self.sites) * 20 + 1 + (1 if self.feat_fn else 0))
        for i, a in muts:
            j = int(np.searchsorted(self.sites, i))
            v[j * 20 + AA_IDX[a]] = 1.0
        v[len(self.sites) * 20] = len(muts)
        if self.feat_fn:
            v[-1] = float(self.feat_fn(muts))
        return v

    def observe(self, muts, y):
        self.X.append(self._feat(muts))
        self.X_muts.append(tuple(sorted((int(i), a) for i, a in muts)))
        self.y.append(float(y))
        self._K = None

    # ---- 核矩阵
    def _kernel_matrix(self, Xa=None):
        X = np.array(self.X)
        if Xa is None:
            if self._K is None:
                self._K = X @ X.T
            return self._K, X
        return Xa @ X.T, X  # [n_cand, n_lab]

    def _fit_alpha(self, K, y, idx):
        Ks = K[np.ix_(idx, idx)]
        ys = y[idx] - np.mean(y[idx])
        A = Ks + self.lam * np.eye(len(idx))
        return np.linalg.solve(A, ys), np.mean(y[idx])

    def predict(self, cand_muts):
        """返回 (pred, unc)。"""
        if len(self.y) < 5:
            n = len(cand_muts)
            return np.zeros(n), np.ones(n)
        C = np.array([self._feat(m) for m in cand_muts])
        Kc, X = self._kernel_matrix(C)   # [n_cand, n_lab]
        K, _ = self._kernel_matrix()
        y = np.array(self.y)
        alpha, ybar = self._fit_alpha(K, y, np.arange(len(y)))
        pred = ybar + Kc @ alpha
        boots = np.empty((self.n_boot, len(cand_muts)))
        for b in range(self.n_boot):
            idx = self.rng.choice(len(y), size=len(y), replace=True)
            ab, yb = self._fit_alpha(K, y, idx)
            boots[b] = yb + Kc @ ab
        unc = boots.std(axis=0) + 1e-6
        return pred, unc

    def propose(self, batch, candidate_pool):
        """candidate_pool: list of muts（benchmark 由实测文库构造）。UCB 选 batch。"""
        cand = [m for m in candidate_pool]
        if not cand:
            return []
        pred, unc = self.predict(cand)
        score = pred + 2.0 * unc
        order = np.argsort(-score)
        picked, seen = [], set()
        for i in order:
            key = tuple(sorted((int(a), b) for a, b in cand[i]))
            if key in seen:
                continue
            seen.add(key)
            picked.append(cand[i])
            if len(picked) >= batch:
                break
        return picked
