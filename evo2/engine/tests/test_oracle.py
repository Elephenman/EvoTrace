#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""oracle 单元测试：加性闭式最优 / NK K=0 退化 / GB1 解析一致性 / WT 恒等。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from engine.oracle import AdditiveOracle, NKOracle, GB1PairwiseOracle
from engine.seqtools import AA2IDX


def test_additive_closed_form():
    rng = np.random.default_rng(0)
    wt = rng.integers(0, 20, size=10)
    G = rng.normal(size=(10, 20))
    o = AdditiveOracle(G, wt, max_mut=5)
    f, g = o.known_optimum()
    # 验证闭式解：取增益 top-5
    gain = G[np.arange(10), G.argmax(1)] - G[np.arange(10), wt]
    top = np.argsort(-gain)[:5]
    g2 = wt.copy(); g2[top] = G.argmax(1)[top]
    assert abs(f - o.eval_one(g2)) < 1e-9
    assert abs(f - G[np.arange(10), g2].sum()) < 1e-9
    # WT 自身适应度 = G 的 WT 列之和
    assert abs(o.eval_one(wt) - G[np.arange(10), wt].sum()) < 1e-9
    print(f"[OK] additive 闭式最优 f={f:.3f} 与逐位点和一致")


def test_nk_k0_separable():
    rng = np.random.default_rng(1)
    wt = rng.integers(0, 20, size=8)
    o = NKOracle(wt, K=0, seed=3)
    f, g = o.known_optimum()
    # K=0 每位点独立 argmax
    assert (g == o.tables.argmax(axis=1)).all()
    # 随机枚举不应超过闭式最优
    rand = o.evaluate(rng.integers(0, 20, size=(2000, 8)))
    assert rand.max() <= f + 1e-6
    # K=1/2/3 应产生不同景观且评估确定性
    for k in (1, 2, 3):
        o2 = NKOracle(wt, K=k, seed=5)
        g1 = rng.integers(0, 20, size=(4, 8))
        a, b = o2.evaluate(g1), o2.evaluate(g1)
        assert np.allclose(a, b)
    print("[OK] NK K=0 可分离且闭式最优正确；K=1..3 评估确定")


def test_gb1_consistency():
    o = GB1PairwiseOracle(max_mut=4)
    assert o.L == 17, f"GB1 全覆盖位点应为 17, 实际 {o.L}"
    # WT 恒等：f(WT) = 0（log 域 LS/LE 对 WT 全为 0）
    assert abs(o.eval_one(o.wt_idx)) < 1e-9
    # 单突变适应度 = log(单突变 Fit)
    j = 0
    pos = o.pos_list[j]
    aa, fit = next(iter(o.singles[pos].items()))
    g = o.wt_idx.copy(); g[j] = AA2IDX[aa]
    assert abs(o.eval_one(g) - np.log(fit)) < 1e-6
    # 双突变 = log(s1) + log(s2) + log(eps)
    p1, a1 = pos, aa
    j2 = 1
    p2 = o.pos_list[j2]
    a2, f2 = next(iter(o.singles[p2].items()))
    g2 = g.copy(); g2[j2] = AA2IDX[a2]
    eps = o.pairs.get((p1, a1, p2, a2))
    # oracle 对非正 eps 下限 1e-6 后取 log（负上位 = 双突变体死亡）
    expect = np.log(max(fit, 1e-6)) + np.log(max(f2, 1e-6)) + \
        (np.log(max(eps, 1e-6)) if eps else 0.0)
    assert abs(o.eval_one(g2) - expect) < 1e-6, \
        f"双突变解析不一致: {o.eval_one(g2)} vs {expect}"
    print(f"[OK] GB1 L={o.L} WT=0、单突变、双突变 log-乘法解析一致")


def test_max_mut_enforced():
    rng = np.random.default_rng(2)
    wt = rng.integers(0, 20, size=20)
    o = NKOracle(wt, K=1, seed=4, max_mut=6)
    g = rng.integers(0, 20, size=(50, 20))
    g2 = o.enforce_max_mut(g.copy(), rng)
    assert (o.n_mutations(g2) <= 6).all()
    print("[OK] max_mut 约束生效（随机 50 个体全部 <= 6 突变）")


if __name__ == "__main__":
    test_additive_closed_form()
    test_nk_k0_separable()
    test_gb1_consistency()
    test_max_mut_enforced()
    print("\nALL ORACLE TESTS PASSED")
