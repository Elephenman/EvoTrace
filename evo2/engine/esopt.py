#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoTrace v2 — ES 分支：进化策略优化器（OpenAI-ES + CEM），纯 numpy。

设计（对应"方案二：ES 直接优化序列"）：
  * 参数化: 类别分布 θ[L,20]（logits）——离散序列空间的分布型 ES。
  * OpenAI-ES: Gumbel 抗噪声对（g 与 −g）采样候选，rank-based fitness
    shaping（centered ranks），梯度 g_θ = Σ u_i·(onehot_i − p)/P。
  * CEM: 采样 batch → 取 elite 分数 → p ← (1−α)p + α·elite 均值 onehot，
    带 p_floor 防塌缩。
  * max_mut 约束: 采样后经 oracle.enforce_max_mut 归一（分布对应性略受
    干扰，属离散 ES 的标准近似）。
  * 先验提议: proposal_table（WF 同款 Q[L,20]）作为 θ/p 的初始化。

诚实性说明：两者均在 oracle 上按评估预算计费（每候选 1 eval），
不在线调用 Boltz-2（硬规则 §9.1）。
"""
import numpy as np

from .wfopt import BaseOptimizer, register


def _prior_table(oracle, cfg, temp_key="proposal_temp", default_temp=1.0):
    """WF 同款先验提议表：Q[L,20] → 温度加权后归一；无先验则均匀。"""
    Q = cfg.get("proposal_table")
    temp = float(cfg.get(temp_key, default_temp))
    if Q is not None:
        w = np.maximum(np.asarray(Q, float), 1e-6) ** temp
        w = w / w.sum(axis=1, keepdims=True)
    else:
        w = np.full((oracle.L, 20), 1 / 20)
    return w


def _centered_ranks(fits):
    """Wierstra fitness shaping: ranks → [-0.5, 0.5]。"""
    order = np.argsort(np.argsort(fits))
    return order / max(len(fits) - 1, 1) - 0.5


def _sample_gumbel(rng, shape):
    return rng.gumbel(size=shape)


@register
class OpenAIESOptimizer(BaseOptimizer):
    """类别分布上的 OpenAI-ES（Gumbel 抗噪声对 + rank shaping）。"""

    name = "es"

    def __init__(self, oracle, seed=0, budget=4000, cfg=None):
        super().__init__(oracle, seed, budget, cfg)
        c = self.cfg
        self.pop = int(c.get("pop", 64))
        self.lr = float(c.get("lr", 0.3))
        self.n_mut_max = int(c.get("n_mut_max", oracle.max_mut or 12))
        w = _prior_table(oracle, c)
        self.theta = np.log(np.maximum(w, 1e-9))       # logits 初始化=先验
        self._p = self._softmax(self.theta)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _sample(self, n, noise=None):
        """Gumbel-max 采样 n 个候选；noise 给定时返回对称对。"""
        p = self._p
        logp = np.log(np.maximum(p, 1e-12))
        if noise is None:
            g = _sample_gumbel(self.rng, (n,) + logp.shape)
            return np.argmax(logp[None] + g, axis=2)
        half = n // 2
        g = _sample_gumbel(self.rng, (half,) + logp.shape)
        cand = np.argmax(logp[None] + np.concatenate([g, -g], axis=0), axis=2)
        return cand[:n]

    def run(self):
        o = self.oracle
        while not self._done():
            n = min(self.pop, self.budget - o.n_evals)
            geno = self._sample(n)
            geno = o.enforce_max_mut(geno, self.rng)
            fits = o.evaluate(geno)
            self._observe(geno, fits)
            u = _centered_ranks(np.asarray(fits, float))
            # g_θ[j,a] = Σ_i u_i (onehot_i[j,a] − p[j,a]) / n
            onehot = np.zeros((n, o.L, 20))
            onehot[np.arange(n)[:, None], np.arange(o.L)[None, :], geno] = 1.0
            grad = (u[:, None, None] * (onehot - self._p[None])).sum(axis=0) / n
            self.theta += self.lr * grad
            self._p = self._softmax(self.theta)
        return dict(best_f=self.best_f, best_geno=self.best_g,
                    trace=self.trace, n_evals=o.n_evals)


@register
class CEMOptimizer(BaseOptimizer):
    """交叉熵方法：elite 加权的类别分布更新。"""

    name = "cem"

    def __init__(self, oracle, seed=0, budget=4000, cfg=None):
        super().__init__(oracle, seed, budget, cfg)
        c = self.cfg
        self.pop = int(c.get("pop", 100))
        self.elite_frac = float(c.get("elite_frac", 0.2))
        self.alpha = float(c.get("alpha", 0.7))
        self.p_floor = float(c.get("p_floor", 1e-3))
        self.n_mut_max = int(c.get("n_mut_max", oracle.max_mut or 12))
        w = _prior_table(oracle, c)
        self.p = w / w.sum(axis=1, keepdims=True)

    def run(self):
        o = self.oracle
        L = o.L
        while not self._done():
            n = min(self.pop, self.budget - o.n_evals)
            cw = np.cumsum(self.p, axis=1)
            cw[:, -1] = 1.0
            u = self.rng.random((n, 1))
            geno = (cw[None, :, :] < u[:, None, :]).sum(axis=2).clip(0, 19)
            geno = o.enforce_max_mut(geno, self.rng)
            fits = o.evaluate(geno)
            self._observe(geno, fits)
            k = max(1, int(self.elite_frac * n))
            elite = geno[np.argsort(fits)[-k:]]
            onehot = np.zeros((k, L, 20))
            onehot[np.arange(k)[:, None], np.arange(L)[None, :], elite] = 1.0
            mean = onehot.mean(axis=0)
            self.p = (1 - self.alpha) * self.p + self.alpha * mean
            self.p = np.maximum(self.p, self.p_floor)
            self.p /= self.p.sum(axis=1, keepdims=True)
        return dict(best_f=self.best_f, best_geno=self.best_g,
                    trace=self.trace, n_evals=o.n_evals)
