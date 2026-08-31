# -*- coding: utf-8 -*-
"""DNA 条件门控景观（DnaAwareLandscape）专项测试。

判据来源：Boltz-2 去混杂扫描（job 225686, 22 变体 × 2 条件 × 8 models）实测标签：
  - gate v3 应与实测判别指标正相关（Spearman > 0.3，全量拟合 +0.858 / LOO +0.583）
  - 机制方向：M255I > M255M（放大器）、Y217Y > Y217R（R 为削弱因子）、F88K > F88F
  - gate v1 为历史对照，已知与实测反相关（−0.588）——防止回归到解析式
"""
import csv
import os

import numpy as np
import pytest

from ppri_dna_aware import DnaAwareLandscape, AA

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRIORS = "A:/claudework/ppri_evo/results/priors.csv"
WT_FASTA = "A:/claudework/ppri_evo/inputs/wt_254.fasta"
DECONF_CSV = os.path.join(ROOT, "results", "boltz_six", "deconf_stats.csv")
GATE_CAL_CSV = os.path.join(ROOT, "results", "boltz_six", "gate_calibration.csv")


class _Stub:
    """不加载 ESM3 权重的桩基：只提供接口，evaluate 返回 0（隔离测试 gate 本身）。"""
    max_mut = 12

    def __init__(self):
        rows = list(csv.DictReader(open(PRIORS)))
        self.sites = sorted({int(r["seq_idx"]) for r in rows})
        wt = open(WT_FASTA).read().splitlines()[1].strip()
        self.wt_idx = np.array([AA.index(wt[s]) for s in self.sites])
        self.L = len(self.sites)
        self.n_evals = 0

    def evaluate(self, genos):
        self.n_evals += len(genos)
        return np.zeros(len(genos))


def _land(gate_version="v3"):
    return DnaAwareLandscape(_Stub(), gate_version=gate_version)


def _set(geno, land, pdb, aa):
    g = geno.copy()
    g[land.seqidx_to_col[pdb - 22]] = AA.index(aa)
    return g


def test_gate_v3_is_default():
    assert _land().gate_version == "v3"


def test_gate_v3_mechanism_directions():
    """去混杂实测的机制方向必须编码进 gate v3。"""
    land = _land("v3")
    wt = land.wt_idx.copy()
    assert land._gate(_set(wt, land, 88, "K")) > land._gate(_set(wt, land, 88, "F"))
    assert land._gate(_set(wt, land, 255, "I")) > land._gate(_set(wt, land, 255, "M"))
    assert land._gate(_set(wt, land, 217, "Y")) > land._gate(_set(wt, land, 217, "R"))
    # 读头：K > W/Y > F（去混杂边际效应 K+0.195 > R+0.114 > W-0.045 > Y-0.052 > F-0.212）
    assert land._gate(_set(wt, land, 88, "W")) > land._gate(_set(wt, land, 88, "F"))


def test_gate_v3_correlates_with_boltz_deconf():
    """gate v3 与 22 变体实测判别指标正相关（全量拟合 +0.858，断言留裕度 >0.3）。"""
    from scipy.stats import spearmanr
    import pandas as pd
    df = pd.read_csv(DECONF_CSV)
    assert len(df) >= 20
    land = _land("v3")
    wt = land.wt_idx.copy()
    scores = []
    for _, r in df.iterrows():
        g = _set(_set(_set(wt, land, 88, r["F88"]), land, 255, r["M255"]), land, 217, r["Y217"])
        scores.append(land._gate(g))
    rho_diface, p_d = spearmanr(scores, df["d_iface"])
    rho_dual, p_dual = spearmanr(scores, df["dual_S1"])
    rho_act, p_act = spearmanr(scores, df["d_act"])
    assert rho_diface > 0.3, f"v3 vs d_iface rho={rho_diface:.3f}"
    assert rho_dual > 0.3, f"v3 vs dual_S1 rho={rho_dual:.3f}"
    assert rho_act > 0.3, f"v3 vs d_act rho={rho_act:.3f}"


def test_s13_c1_replica_is_top_deconf():
    """F88K_M255I_Y217Y（= s13_c1 复现）在 22 变体中 gate v3 应名列前茅。"""
    import pandas as pd
    df = pd.read_csv(DECONF_CSV)
    land = _land("v3")
    wt = land.wt_idx.copy()
    scores = {}
    for _, r in df.iterrows():
        g = _set(_set(_set(wt, land, 88, r["F88"]), land, 255, r["M255"]), land, 217, r["Y217"])
        scores[r["variant"]] = land._gate(g)
    rank = sorted(scores, key=lambda k: scores[k], reverse=True)
    assert "F88K_M255I_Y217Y" in rank[:5], f"rank={rank[:6]}"
    # 且实测 composite（d_iface）同变体应同样靠前 —— gate 与实测一致
    df2 = df.set_index("variant")
    assert df2.loc["F88K_M255I_Y217Y", "d_iface"] > df2["d_iface"].median()


def test_gate_v1_anticorrelation_is_regression_guard():
    """v1 解析式已知反相关（-0.588）——保留为对照，断言仍负（防回归）。"""
    from scipy.stats import spearmanr
    import pandas as pd
    df = pd.read_csv(GATE_CAL_CSV).dropna(subset=["gate_v1", "dual_S1"])
    rho, _ = spearmanr(df["gate_v1"], df["dual_S1"])
    assert rho < 0, f"v1 反相关被破坏（rho={rho:.3f}）——若修复了解析式请更新本测试"


def test_dna_aware_evaluate_composes_base_and_gate():
    """组合景观：evaluate = w_base*z + w_gate*gate；默认 w_base=w_gate=0.5。"""
    land = _land("v3")
    assert land.w_base == 0.5 and land.w_gate == 0.5
    g = np.tile(land.wt_idx, (3, 1))
    f = land.evaluate(g)
    assert f.shape == (3,)
    assert np.all(np.isfinite(f))
    # 桩基 z=0 → 输出应恰为 w_gate*gate
    assert np.allclose(f, 0.5 * np.array([land._gate(x) for x in g]))
    # 搜索内核兼容接口
    assert land.n_mutations(g).tolist() == [0, 0, 0]
    rng = np.random.default_rng(1)
    g2 = land.enforce_max_mut(g.copy(), rng)
    assert g2.shape == g.shape
