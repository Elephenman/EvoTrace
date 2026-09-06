# -*- coding: utf-8 -*-
"""Potts/DCA 二阶耦合模型 + 岛屿 WF 驱动（提案_DCA二阶耦合先验.md §4 落点, 2026-09-06）。

组成:
  - fit_plm: 伪似然 DCA——逐位点 L2 正则 softmax 回归（one-hot 全谱特征,
    排除自身位点, Adam 全批次）, 返回全谱权重 [K, K*Q+1, Q]（含场 h）;
  - PottsModel: 对称化 + 稀疏化耦合（|J|≤eps 的对剔除）, 能量
    E(x) = Σ h_i(x_i) + Σ_(i<j) J_ij(x_i, x_j), 条件分布 P(x_i | x_rest);
  - island_wf: 岛屿 Wright-Fisher——每代各岛独立选择（softmax(E/T)）+
    耦合条件分布变异（逐个体 Poisson(λ) 个位点按条件重抽）+ 岛间迁移
    （每岛替换 ⌊m·Ne⌋ 个个体为随机供体岛拷贝）。

诚实边界: plmDCA 均场近似过估耦合, 靠 λ 正则 + eps 稀疏化兜底; 能量即适应度
是"Boltzmann 景观"假设, 不是 DMS 实测——L3 回放验证层只关心分布形状再现。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["fit_plm", "PottsModel", "island_wf"]


# ---------------------------------------------------------------- 伪似然 DCA
def fit_plm(mat21: np.ndarray, Q: int = 21, epochs: int = 150, lam: float = 0.01,
            lr: float = 0.05, seed: int = 0, verbose: bool = False) -> np.ndarray:
    """逐位点 softmax 回归。mat21: [N,K] int8（0..Q-1, gap=Q-1）。
    返回 Wfull [K, K*Q+1, Q]: 前行块 = 特征权重（自身位点块恒零）, 末行 = 场 h_i。"""
    rng = np.random.default_rng(seed)
    N, K = mat21.shape
    D = K * Q
    F = np.zeros((N, D), dtype=np.float32)
    F[np.arange(N)[:, None], np.arange(K)[None, :] * Q + mat21] = 1.0
    Wfull = np.zeros((K, D + 1, Q), dtype=np.float32)
    t0 = time.time()
    for i in range(K):
        others = np.concatenate([np.arange(0, i * Q), np.arange((i + 1) * Q, D)])
        Xi = np.hstack([F[:, others], np.ones((N, 1), dtype=np.float32)])
        Yi = np.zeros((N, Q), dtype=np.float32)
        Yi[np.arange(N), mat21[:, i]] = 1.0
        W = np.zeros((Xi.shape[1], Q), dtype=np.float32)
        m1 = np.zeros_like(W); m2 = np.zeros_like(W)
        for ep in range(epochs):
            lr_t = lr * max(0.05, 1.0 - ep / epochs)     # 线性衰减: 过参数化下
            Z = Xi @ W                                   # 恒定 lr 的 Adam 会随机
            Z -= Z.max(axis=1, keepdims=True)            # 游走, 耦合尺度失真
            P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
            G = Xi.T @ (P - Yi) / N + lam * W
            m1 = 0.9 * m1 + 0.1 * G
            m2 = 0.999 * m2 + 0.001 * G * G
            # 标准 Adam: 分子分母各自偏差修正（缺 m2 修正在近最优区会振荡）
            W -= lr_t * ((m1 / (1 - 0.9 ** (ep + 1))) /
                         (np.sqrt(m2 / (1 - 0.999 ** (ep + 1))) + 1e-8))
        Wfull[i, others, :] = W[:-1, :]
        Wfull[i, D, :] = W[-1, :]
        if verbose and ((i + 1) % 12 == 0 or i == K - 1):
            print(f"    fit site {i+1}/{K}（{time.time()-t0:.0f}s）")
    return Wfull


# ---------------------------------------------------------------- Potts 模型
class PottsModel:
    """对称化稀疏 Potts: E(x) = Σ h_i(x_i) + Σ J_ij(x_i,x_j)（|J|≤eps 对剔除）。"""

    def __init__(self, Wfull: np.ndarray, eps: float = 0.03, Q: int = 21):
        K, D1, _ = Wfull.shape
        self.K, self.Q, self.eps = K, Q, float(eps)
        self.h = Wfull[:, D1 - 1, :].copy()              # [K,Q]
        self.J: Dict[Tuple[int, int], np.ndarray] = {}   # (i,j), i<j → [Q,Q]
        self._adj: List[List[Tuple[int, np.ndarray]]] = [[] for _ in range(K)]
        for i in range(K):
            for j in range(i + 1, K):
                a = Wfull[i, j * Q:(j + 1) * Q, :]       # W_i 对 x_j 的权重
                b = Wfull[j, i * Q:(i + 1) * Q, :]       # W_j 对 x_i 的权重
                M = 0.5 * (a + b.T)                      # 对称化
                if np.abs(M).max() > eps:
                    M = M.astype(np.float32)
                    self.J[(i, j)] = M
                    self._adj[i].append((j, M))
                    self._adj[j].append((i, M.T.copy()))

    # ---- 能量: states [N,K] int8 → [N] float32
    def energy(self, states: np.ndarray) -> np.ndarray:
        st = np.asarray(states, dtype=np.int64)
        e = self.h[np.arange(self.K)[None, :], st].sum(axis=1)
        for (i, j), M in self.J.items():
            e = e + M[st[:, i], st[:, j]]
        return e.astype(np.float32)

    # ---- 条件分布: rows 状态 [N,K], site → P(x_site|rest) [N,Q]
    def conditional(self, states: np.ndarray, site: int) -> np.ndarray:
        st = np.asarray(states, dtype=np.int64)
        logits = np.broadcast_to(self.h[site][None, :], (len(st), self.Q)).copy()
        for j, M in self._adj[site]:
            logits += M[st[:, j], :]                     # [N,Q] gather
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        return P / P.sum(axis=1, keepdims=True)

    # ---- Gibbs 平衡采样（诊断用, b10 口径）
    def gibbs(self, init: np.ndarray, n_sweeps: int = 300, burn: int = 200,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        st = np.array(init, dtype=np.int64).copy()
        M, K = st.shape
        out = []
        for s in range(n_sweeps):
            for i in range(K):
                P = self.conditional(st, i)
                cum = np.cumsum(P, axis=1)
                u = rng.random((M, 1))
                st[:, i] = (u > cum).sum(axis=1).clip(0, self.Q - 1)
            if s >= burn:
                out.append(st.copy())
        return np.array(out, dtype=np.int8).reshape(-1, K)


# ---------------------------------------------------------------- 岛屿 WF
def island_wf(pm: PottsModel, pops0: np.ndarray, n_gen: int, T: float = 1.0,
              lam_mut: float = 0.3, m_mig: float = 0.0,
              rng: Optional[np.random.Generator] = None,
              observer=None) -> np.ndarray:
    """岛屿 Wright-Fisher, 耦合提议。

    pops0: [n_pop, Ne, K] int8 初态（b9 口径: 祖先群体起步, 非 WT 种子）。
    每代: 岛内 softmax(E/T) 抽亲本 → 拷贝 → 每个体 Poisson(lam_mut) 个位点
    按条件分布重抽（同个体多位点顺序条件化）→ 迁移（每岛替换 ⌊m·Ne⌋ 个体
    为随机供体岛个体拷贝）。返回终态 pops [n_pop, Ne, K]。
    observer(gen, pops, stats) 可选。
    """
    rng = rng or np.random.default_rng(0)
    pops = np.array(pops0, dtype=np.int64)
    n_pop, Ne, K = pops.shape
    for g in range(n_gen):
        gen_stats = []
        new = []
        for p in range(n_pop):
            x = pops[p]
            e = pm.energy(x)
            w = np.exp((e - e.max()) / T)
            w /= w.sum()
            par = rng.choice(Ne, size=Ne, p=w)
            children = x[par].copy()
            # 耦合变异: 按位点分批算条件分布
            k = rng.poisson(lam_mut, size=Ne)
            reqs = [(n, s) for n in np.flatnonzero(k > 0)
                    for s in rng.choice(K, size=int(k[n]), replace=False)]
            if reqs:
                by_site: Dict[int, List[int]] = {}
                for n, s in reqs:
                    by_site.setdefault(int(s), []).append(int(n))
                for s, idxs in by_site.items():
                    ctx = children[np.array(idxs)]
                    P = pm.conditional(ctx, s)
                    cum = np.cumsum(P, axis=1)
                    u = rng.random((len(idxs), 1))
                    draw = (u > cum).sum(axis=1).clip(0, pm.Q - 1)
                    children[np.array(idxs), s] = draw
            new.append(children)
            gen_stats.append(dict(pop=p, gen=g, best=round(float(e.max()), 3),
                                  mean=round(float(e.mean()), 3)))
        pops = np.stack(new)
        # 岛间迁移
        n_swap = int(m_mig * Ne)
        if n_swap > 0 and n_pop > 1:
            for p in range(n_pop):
                donors = rng.integers(0, n_pop)
                if donors == p:
                    continue
                slots = rng.choice(Ne, size=n_swap, replace=False)
                picks = rng.choice(Ne, size=n_swap, replace=False)
                pops[p][slots] = pops[donors][picks]
        if observer is not None:
            observer(g, pops, gen_stats)
    return pops.astype(np.int8)
