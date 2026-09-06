# -*- coding: utf-8 -*-
"""potts 模块测试: fit_plm / PottsModel / island_wf（提案门验收, 2026-09-06）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from potts import PottsModel, fit_plm, island_wf  # engine 目录式

Q = 21


def _synthetic_coupled(N=800, K=4, seed=7):
    """两耦合位点的合成 MSA: x1 以 0.9 概率复制 x0（强正耦合）,
    其余位点独立。返回 mat21 [N,K]。"""
    rng = np.random.default_rng(seed)
    mat = rng.integers(0, 8, size=(N, K))          # 独立背景
    mat[:, 1] = mat[:, 0]                          # 先全部复制
    flip = rng.random(N) < 0.1
    mat[flip, 1] = (mat[flip, 0] + 1 + rng.integers(0, 6, flip.sum())) % 8
    return mat.astype(np.int8)


def test_fit_plm_recovers_strongest_coupling():
    mat = _synthetic_coupled()
    W = fit_plm(mat, epochs=80, lam=0.01, seed=1)
    K = mat.shape[1]
    pm = PottsModel(W, eps=0.0, Q=Q)
    fro = np.zeros((K, K))
    for (i, j), M in pm.J.items():
        fro[i, j] = np.linalg.norm(M)
    top = np.unravel_index(np.argmax(fro), fro.shape)
    assert tuple(sorted(top)) == (0, 1), f"最强耦合应对 (0,1), 实得 {top}"


def test_potts_energy_and_conditional():
    rng = np.random.default_rng(2)
    K = 5
    W = rng.normal(0, 0.3, size=(K, K * Q + 1, Q)).astype(np.float32)
    pm = PottsModel(W, eps=0.0, Q=Q)
    st = rng.integers(0, Q, size=(20, K))
    e = pm.energy(st)
    assert e.shape == (20,) and np.isfinite(e).all()
    # 能量手算对账（全部对称对 + 场）
    manual = pm.h[np.arange(K), st[0]].sum()
    for (i, j), M in pm.J.items():
        manual = manual + M[st[0, i], st[0, j]]
    assert abs(e[0] - manual) < 1e-3
    P = pm.conditional(st, 2)
    assert P.shape == (20, Q)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-5)
    assert (P >= 0).all()


def test_potts_sparsification_drops_weak_pairs():
    rng = np.random.default_rng(3)
    K = 6
    W = (rng.normal(0, 0.05, size=(K, K * Q + 1, Q))).astype(np.float32)
    pm_all = PottsModel(W, eps=0.0, Q=Q)
    pm_sparse = PottsModel(W, eps=0.5, Q=Q)
    assert len(pm_sparse.J) < len(pm_all.J)
    st = rng.integers(0, Q, size=(10, K))
    e_all, e_sp = pm_all.energy(st), pm_sparse.energy(st)
    assert np.abs(e_all - e_sp).max() < 1.0       # 稀疏化只丢小项


def test_island_wf_shapes_and_determinism():
    rng = np.random.default_rng(4)
    K, n_pop, Ne = 4, 3, 40
    W = (rng.normal(0, 0.2, size=(K, K * Q + 1, Q))).astype(np.float32)
    pm = PottsModel(W, eps=0.1, Q=Q)
    pops0 = rng.integers(0, Q, size=(n_pop, Ne, K)).astype(np.int8)
    kw = dict(pm=pm, pops0=pops0, n_gen=5, T=1.0, lam_mut=0.3)
    a = island_wf(m_mig=0.0, rng=np.random.default_rng(11), **kw)
    b = island_wf(m_mig=0.0, rng=np.random.default_rng(11), **kw)
    assert a.shape == (n_pop, Ne, K)
    assert np.array_equal(a, b)                    # seed 确定性
    assert a.dtype == np.int8 and (a >= 0).all() and (a < Q).all()


def test_migration_homogenizes_islands():
    """m 高时岛间应当混合: 各岛主等位一致。"""
    rng = np.random.default_rng(5)
    K, n_pop, Ne = 3, 4, 60
    W = np.zeros((K, K * Q + 1, Q), dtype=np.float32)   # 中性景观
    pm = PottsModel(W, eps=0.0, Q=Q)
    # 起始: 各岛固定不同等位
    pops0 = np.zeros((n_pop, Ne, K), dtype=np.int8)
    for p in range(n_pop):
        pops0[p][:, 0] = 1 + p
    end = island_wf(pm, pops0, n_gen=30, T=1.0, lam_mut=0.05, m_mig=0.5,
                    rng=np.random.default_rng(6))
    modes = [int(np.bincount(end[p][:, 0], minlength=Q).argmax()) for p in range(n_pop)]
    assert len(set(modes)) == 1, f"高迁移后各岛主等位应趋同, 实得 {modes}"
