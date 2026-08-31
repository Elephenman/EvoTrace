# -*- coding: utf-8 -*-
"""EvoTrace 工具冒烟测试：六优化器 × 内置 oracle 能实例化、能跑完一轮、输出合法。"""
import numpy as np
import pytest

from engine.oracle import AdditiveOracle
from engine.seqtools import AA20, AA2IDX


def _additive(L=8, seed=3, max_mut=4):
    rng = np.random.default_rng(seed)
    wt = "".join(rng.choice(list(AA20), L))
    wt_idx = np.array([AA2IDX[a] for a in wt])
    G = rng.normal(0, 1.0, (L, 20))
    return AdditiveOracle(G, wt_idx, max_mut=max_mut)


def _all_optimizers():
    import engine.wfopt as W
    opts = {"wf": W.WFOptimizer}
    try:
        import engine.rlpolicy as R
        opts["dqn"] = R.DQNOptimizer
        opts["bc"] = R.BCOptimizer
    except ImportError:
        pass
    try:
        import engine.rlpolicy_torch as RT
        opts["ppo"] = RT.PPOOptimizer
    except ImportError:
        pass
    try:
        import engine.esopt as E
        opts["es"] = E.OpenAIESOptimizer
        opts["cem"] = E.CEMOptimizer
    except ImportError:
        pass
    return opts


def test_six_optimizers_discovered():
    """工具宣称六内核，必须全部可发现。"""
    opts = _all_optimizers()
    assert set(opts) == {"wf", "ppo", "dqn", "bc", "cem", "es"}


def test_oracle_interface_conformity():
    """oracle 统一接口：L / max_mut / wt_idx / sites / evaluate / n_mutations / enforce_max_mut。"""
    orc = _additive(L=8, max_mut=4)
    assert orc.L == 8
    assert orc.max_mut == 4
    assert orc.wt_idx.shape == (8,)
    assert orc.sites.shape == (8,)
    g = np.tile(orc.wt_idx, (5, 1))
    f = orc.evaluate(g)
    assert f.shape == (5,)
    assert np.all(np.isfinite(f))
    assert orc.n_evals == 5
    # n_mutations / enforce_max_mut
    g2 = g.copy()
    g2[1, 0] = (g2[1, 0] + 1) % 20
    g2[1, 1] = (g2[1, 1] + 2) % 20
    g2[2, 2] = (g2[2, 2] + 3) % 20
    nm = orc.n_mutations(g2)
    assert nm.tolist() == [0, 2, 1, 0, 0]
    rng = np.random.default_rng(0)
    g3 = orc.enforce_max_mut(g2.copy(), rng)
    assert int((g3[1] != orc.wt_idx).sum()) <= 4


@pytest.mark.parametrize("oname", ["wf", "ppo", "dqn", "bc", "cem", "es"])
def test_each_optimizer_runs(oname):
    """每个内核在加性景观(L=8) 300 evals 内跑完一轮且输出合法。"""
    opts = _all_optimizers()
    orc = _additive()
    cfg = {"lr": 1e-3, "ent": 0.01} if oname == "ppo" else {}
    opt = opts[oname](orc, seed=0, budget=300, cfg=cfg)
    res = opt.run()
    assert "best_f" in res and "trace" in res and "n_evals" in res
    assert res["n_evals"] > 0
    assert np.isfinite(res["best_f"])
    assert len(res["trace"]) > 0
