# -*- coding: utf-8 -*-
"""b7 基准管道回归测试。

说明：b7 的 REBUILD 注册在 main() 内（import 时为空 dict），
故本模块以"源码级注册断言 + 独立 oracle 构造"验证管道装配，
再以轻量景观复现"加性景观上 WF ≥ CEM"这一 b7 全量基准结论。

不加载 ESM3 权重（surrogate/dna_aware 为惰性注册），保证测试轻量。
"""
import inspect

import numpy as np

import b7_three_way as b7


def test_b7_registry_declared_in_main():
    """REBUILD 注册（main 内）必须声明基础景观与 DL 能力。"""
    src = inspect.getsource(b7.main)
    for needle in ("ppri_additive", "nk_k", "gb1_pairwise",
                   "ppri_cal", "surrogate_ppri", "surrogate_ppri_v3",
                   "dna_aware_ppri"):
        assert needle in src, f"main() 未注册 {needle}"
    assert "REBUILD" in src


def test_b7_discovers_six_optimizers():
    opts = b7.discover_optimizers()
    assert set(opts) == {"wf", "ppo", "dqn", "bc", "cem", "es"}


def test_additive_pipeline_contract():
    """oracle 管道契约：L / max_mut / evaluate / known_optimum 自洽。"""
    from engine.oracle import AdditiveOracle
    from engine.seqtools import AA20, AA2IDX
    rng = np.random.default_rng(3)
    wt = "".join(rng.choice(list(AA20), 8))
    wt_idx = np.array([AA2IDX[a] for a in wt])
    G = rng.normal(0, 1.0, (8, 20))
    orc = AdditiveOracle(G, wt_idx, max_mut=4)
    g = np.tile(orc.wt_idx, (3, 1))
    f = orc.evaluate(g)
    assert f.shape == (3,) and np.all(np.isfinite(f))
    f_opt, g_opt = orc.known_optimum()
    assert orc.eval_one(g_opt) == f_opt
    assert int((g_opt != orc.wt_idx).sum()) <= 4


def test_b7_light_regression_wf_over_cem_on_additive():
    """轻量回归：加性景观上 WF norm 应 ≥ CEM（b7 全量结论的冒烟版）。

    预算校准（2026-08-31 实测）：L=8 加性景观，
    budget=400(2代) 时 WF(0.328)<CEM(0.618) —— WF 需 ≥10 代起步，
    budget=2000 时 WF(0.988)>CEM(0.618)、4000 时 WF(1.000)>CEM(0.658)。
    测试取 2000 以覆盖 b7 结论的成立区间。
    """
    from engine.oracle import AdditiveOracle
    from engine.seqtools import AA20, AA2IDX
    from engine.wfopt import WFOptimizer
    from engine.esopt import CEMOptimizer

    def _mk():
        rng = np.random.default_rng(3)
        wt = "".join(rng.choice(list(AA20), 8))
        wt_idx = np.array([AA2IDX[a] for a in wt])
        return AdditiveOracle(rng.normal(0, 1.0, (8, 20)), wt_idx, max_mut=4)

    orc = _mk()
    f_opt, _ = orc.known_optimum()
    f_wt = orc.eval_one(orc.wt_idx)
    span = max(f_opt - f_wt, 1e-9)

    r_wf = WFOptimizer(orc, seed=0, budget=2000, cfg={}).run()
    norm_wf = (r_wf["best_f"] - f_wt) / span

    r_cem = CEMOptimizer(_mk(), seed=0, budget=2000, cfg={}).run()
    norm_cem = (r_cem["best_f"] - f_wt) / span

    assert norm_wf >= 0.5, f"WF norm={norm_wf:.3f}（工具退化）"
    assert norm_wf >= norm_cem, f"WF({norm_wf:.3f}) < CEM({norm_cem:.3f})（与 b7 基准矛盾）"
