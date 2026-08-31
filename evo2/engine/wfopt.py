#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoTrace v2 — 优化器协议 + Wright-Fisher 优化器（oracle 预算约束版）。

b7 三方对比协议：每个优化器实现 ``run(budget)``，预算 = oracle.evaluate 调用数。
分支注册：rl-policy 分支提供 DQN/BC，es-optimizer 分支提供 OpenAI-ES/CEM，
b7 harness 通过 ``OPTIMIZERS`` 字典动态发现。
"""
import numpy as np

from .oracle import Oracle

OPTIMIZERS = {}   # name -> class（跨分支动态注册）


def register(cls):
    OPTIMIZERS[cls.name] = cls
    return cls


class BaseOptimizer:
    name = "base"

    def __init__(self, oracle: Oracle, seed=0, budget=4000, cfg=None):
        self.oracle = oracle
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.budget = int(budget)
        self.cfg = cfg or {}
        self.trace = []           # [(n_evals, best_so_far)]
        self.best_f = -np.inf
        self.best_g = None

    def _observe(self, geno, fits):
        """统一记账：预算由 oracle.n_evals 维护，这里维护 best 与 trace。"""
        i = int(np.argmax(fits))
        if fits[i] > self.best_f:
            self.best_f = float(fits[i])
            self.best_g = geno[i].copy()
        self.trace.append((self.oracle.n_evals, self.best_f))
        return fits

    def _done(self):
        return self.oracle.n_evals >= self.budget

    def run(self):
        raise NotImplementedError


@register
class WFOptimizer(BaseOptimizer):
    """Wright-Fisher：softmax 选择 + Poisson(λ) 突变 + WF 多项抽样。

    proposal='prior' 时用每位点贡献表 Q[L,20]（softmax 温度 proposal_temp）
    加权提议（EvoTrace 的先验提议算子）；'uniform' 为均匀提议。
    预算 = Ne × n_gen 次 oracle 评估（与真实定向进化"每代测全部群体"对应）。
    """

    name = "wf"

    def __init__(self, oracle, seed=0, budget=4000, cfg=None):
        super().__init__(oracle, seed, budget, cfg)
        c = self.cfg
        self.lam = float(c.get("lambda", 0.6))
        self.T = float(c.get("T", 0.6))
        self.proposal_temp = float(c.get("proposal_temp", 2.0))
        self.n_mut_max = int(c.get("n_mut_max", oracle.max_mut or 12))
        Q = c.get("proposal_table")
        if Q is not None:
            w = np.maximum(np.asarray(Q, float), 1e-6) ** self.proposal_temp
            w = w / w.sum(axis=1, keepdims=True)
        else:
            w = np.full((oracle.L, 20), 1 / 20)
        self.prop = w
        self.cw = np.cumsum(w, axis=1)
        self.cw[:, -1] = 1.0

    def _mutate(self, geno):
        N, L = geno.shape
        k = self.rng.poisson(self.lam, size=N)
        max_k = int(k.max()) if N else 0
        for step in range(max_k):
            act = np.flatnonzero(k > step)
            if len(act) == 0:
                break
            js = self.rng.integers(0, L, size=len(act))
            u = self.rng.random((len(act), 1))
            aa = (self.cw[js] < u).sum(axis=1).clip(0, 19)
            geno[act, js] = aa
        return self.oracle.enforce_max_mut(geno, self.rng)

    def run(self):
        o = self.oracle
        # 预算分配：Ne × n_gen ≈ budget
        Ne = int(self.cfg.get("Ne", 200))
        n_gen = max(1, self.budget // Ne)
        geno = np.tile(o.wt_idx[None, :], (Ne, 1))
        for g in range(n_gen):
            if self._done():
                break
            fits = self._observe(geno, o.evaluate(geno))
            ws = np.exp((fits - fits.max()) / self.T)
            ws /= ws.sum()
            parents = self.rng.choice(Ne, size=Ne, p=ws)
            geno = self._mutate(geno[parents].copy())
        if not self.trace or self.trace[-1][0] < min(o.n_evals, self.budget):
            fits = o.evaluate(geno)
            self._observe(geno, fits)
        return dict(best_f=self.best_f, best_geno=self.best_g,
                    trace=self.trace, n_evals=o.n_evals)
