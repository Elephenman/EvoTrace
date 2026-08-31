# -*- coding: utf-8 -*-
"""EvoTrace 工具正确性测试：收敛性、确定性、max_mut 强制、代理输出值域。

判据来源：b7 基准（三景观全量对比）中"WF 是最稳内核"的结论——
在可分离加性景观上 WF 必须接近闭式最优；确定性必须成立（同 seed 同结果）。
"""
import numpy as np
import pytest

from engine.oracle import AdditiveOracle, NKOracle
from engine.seqtools import AA20, AA2IDX

from test_tool_smoke import _additive


def test_additive_known_optimum_is_closed_form():
    """加性景观闭式最优：known_optimum 返回的基因型 fitness 应等于其评估值。"""
    orc = _additive(L=8, max_mut=4)
    f_opt, g_opt = orc.known_optimum()
    assert orc.eval_one(g_opt) == pytest.approx(f_opt)
    assert orc.has_known_optimum
    # 闭式解必须不超过 max_mut 突变
    assert int((g_opt != orc.wt_idx).sum()) <= 4


def test_wf_converges_on_additive():
    """WF 在可分离加性景观上应收敛到 ≥90% 最优（b7 基准中 WF 最稳）。

    预算校准（2026-08-31 实测）：L=8 加性景观 Ne=200 时
    budget=800(4代)→norm 0.831、2000(10代)→0.988、4000→1.000。
    测试取 2000 以覆盖"10 代内收敛"这一核心行为。
    """
    from engine.wfopt import WFOptimizer
    orc = _additive(L=8, max_mut=4)
    f_opt, _ = orc.known_optimum()
    f_wt = orc.eval_one(orc.wt_idx)
    opt = WFOptimizer(orc, seed=0, budget=2000, cfg={"Ne": 200})
    res = opt.run()
    norm = (res["best_f"] - f_wt) / max(f_opt - f_wt, 1e-9)
    assert norm >= 0.90, f"WF norm={norm:.3f} 未收敛（best={res['best_f']:.3f} opt={f_opt:.3f}）"


def test_determinism_same_seed_same_result():
    """同 seed 两次 run 必须给出相同 best_f（工具可复现性）。"""
    from engine.wfopt import WFOptimizer
    r1 = WFOptimizer(_additive(), seed=7, budget=300, cfg={}).run()
    r2 = WFOptimizer(_additive(), seed=7, budget=300, cfg={}).run()
    assert r1["best_f"] == r2["best_f"]
    assert r1["trace"] == r2["trace"]


def test_nk_oracle_interface():
    """NK 景观（上位性）可实例化、evaluate 合法。"""
    rng = np.random.default_rng(5)
    wt = "".join(rng.choice(list(AA20), 12))
    wt_idx = np.array([AA2IDX[a] for a in wt])
    orc = NKOracle(wt_idx, K=2, seed=1, max_mut=6)
    g = np.tile(orc.wt_idx, (4, 1))
    f = orc.evaluate(g)
    assert f.shape == (4,)
    assert np.all(np.isfinite(f))


def test_enforce_max_mut_reverts_excess():
    """超出 max_mut 的基因型被强制回退到 ≤max_mut 突变。"""
    orc = _additive(L=8, max_mut=3)
    rng = np.random.default_rng(0)
    g = orc.wt_idx.copy()
    pos = rng.choice(8, 6, replace=False)
    for p in pos:
        others = [a for a in range(20) if a != orc.wt_idx[p]]
        g[p] = rng.choice(others)
    assert int((g != orc.wt_idx).sum()) == 6
    ge = orc.enforce_max_mut(g[None, :].copy(), rng)
    assert int((ge[0] != orc.wt_idx).sum()) <= 3
    # 回退的位点必须恢复为 WT 残基（不引入新突变）
    reverted = (ge[0] != g) & (ge[0] != orc.wt_idx)
    assert int(reverted.sum()) == 0
