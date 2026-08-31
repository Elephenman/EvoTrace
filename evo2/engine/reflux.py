#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoTrace v2 — 贝叶斯标签回流（机制 M4）+ 全局上位性层。

v1 wave-1 机制（ppri_evo/engine/evolve_w1.py）的泛化：
  prior'(i,a) ∝ prior(i,a)^(1-α) × post(i,a)^α
  post(i,a) = (evidence(i,a) + k0·prior(i,a)) / (k0 + E_tot)
v2 泛化点：
  1. 证据不再只数"赢家命中"，而是对每个带标签基因型的 (i,a) 组合按
     sigmoid((y - y_ref)/scale) 加权——标签质量连续化，非只有赢家才算证据；
  2. 全局上位性层 g(·)：加性先验分 → 观测适应度的单调映射（PavaIsotonic 实现，
     无 sklearn 依赖），由已获标签拟合——把 Sarkisyan/Otwinowski 反复证明的
     全局上位性显式建模进引擎；
  3. 稳定性修正（M1）独立于本模块：prior'' ∝ prior' × exp(−ddG/τ)，见 funnel。
"""
import numpy as np

from .seqtools import AA20, AA2IDX


def isotonic_fit(x, y, increasing=True):
    """PAVA 等长回归：把 (x,y) 散点拟合成单调阶梯。返回向量化 callable。"""
    order = np.argsort(x)
    xs = np.asarray(x, float)[order]
    ys = np.asarray(y, float)[order]
    blocks = []  # [sum_y, count, end_x]
    for xi, yi in zip(xs, ys):
        blocks.append([yi, 1.0, xi])
        while len(blocks) > 1:
            m2 = blocks[-1][0] / blocks[-1][1]
            m1 = blocks[-2][0] / blocks[-2][1]
            if (m1 > m2) if increasing else (m1 < m2):
                b2 = blocks.pop()
                b1 = blocks.pop()
                blocks.append([b1[0] + b2[0], b1[1] + b2[1], b2[2]])
            else:
                break
    end_xs = np.array([b[2] for b in blocks])
    vals = np.array([b[0] / b[1] for b in blocks])

    def g(q):
        q = np.atleast_1d(np.asarray(q, float))
        k = np.searchsorted(end_xs, q, side="left")
        k = np.clip(k, 0, len(vals) - 1)
        return vals[k]
    return g


class GlobalEpistasis:
    """加性潜变量 -> 观测适应度的单调映射（从标签在线拟合）。"""

    def __init__(self):
        self.g = None
        self.n = 0

    def update(self, latent, observed):
        if len(observed) >= 8:
            self.g = isotonic_fit(latent, observed, increasing=True)
            self.n = len(observed)

    def __call__(self, latent):
        if self.g is None:
            return latent
        return self.g(latent)


def reflux_prior(prior_table, wt_idx, labeled_muts, labeled_y, alpha=0.35, k0=2.0,
                 y_ref=None, scale=None, floor=1e-3):
    """贝叶斯标签回流（M4）。

    prior_table: [L, 20] 归一化先验；wt_idx: [L]；
    labeled_muts: list of dict{(site_idx_j, aa_char): 1}（0-based j）；
    labeled_y: 观测适应度（已与参考 y_ref 比较形成证据权重）。
    证据权重 w = sigmoid((y - y_ref)/scale) ∈ (0,1)。
    """
    L, n_aa = prior_table.shape
    ys = np.asarray(labeled_y, float)
    y_ref = float(np.median(ys)) if y_ref is None else float(y_ref)
    scale = float(np.std(ys) + 1e-9) if scale is None else float(scale)
    evidence = np.zeros((L, n_aa))
    for muts, y in zip(labeled_muts, ys):
        w = 1.0 / (1.0 + np.exp(-(y - y_ref) / scale))
        for (j, aa) in muts:
            evidence[j, AA2IDX[aa]] += w
    E_tot = evidence.sum()
    post = (evidence + k0 * prior_table) / (k0 + E_tot)
    new = (prior_table ** (1 - alpha)) * (post ** alpha)
    new /= new.sum(axis=1, keepdims=True)
    return np.maximum(new, floor) / np.maximum(new, floor).sum(axis=1, keepdims=True)
