# -*- coding: utf-8 -*-
"""CombinedScorer 回归测试 —— 用 n100 概念验证的真实数字锁定行为。

证据: evo2/results/dl_v7sep/combine_result.csv（verify_combine_v4_dl.py 产出）
关键事实:
  - min(rank(DL_act), rank(v4_sep)) vs 真 sep rho = +0.630, 全组最高;
  - RD_POS: DL_act 排名第 1（假特异, 真 sep 仅 0.01）→ 组合分被压到 <0.4;
  - HQL2: 同类假特异（DL_act 1.358）→ 组合分 <0.25;
  - s13_c1（已入选 MD 的真候选）组合分进 top-2。
"""
import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from combined_scorer import (CandidateScore, CombinedScorer, agreement_rho,
                             combine_rule, percentile_rank)

CSV = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "dl_v7sep", "combine_result.csv"))


def load_rows():
    with open(CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_percentile_rank_matches_verified_columns():
    rows = load_rows()
    r = percentile_rank([float(x["dl_act"]) for x in rows])
    expected = np.array([float(x["rank_dlact"]) for x in rows])
    assert np.allclose(r, expected, atol=1e-9)


def test_combine_rules_registry():
    a = b = np.array([0.1, 0.5, 0.9])
    assert np.allclose(combine_rule("min", a, b), a)
    assert np.allclose(combine_rule("mean", a, b), a)
    assert np.allclose(combine_rule("product", a, b), a * a)
    with pytest.raises(KeyError):
        combine_rule("nope", a, b)


def test_min_rule_presses_false_specific():
    """单高被惩罚: 结合强但不特异的候选不得进前排。"""
    rows = load_rows()
    strength = {x["system"]: float(x["dl_act"]) for x in rows}
    spec = {x["system"]: float(x["v4_sep"]) for x in rows}
    scorer = CombinedScorer(strength.get, spec.get)
    ranked = scorer.score(list(strength))
    got = {c.system_id: c for c in ranked}
    # RD_POS / HQL2: DL_act 排名第 1/第 3, 但真 sep 0.01/0.13 → 被压后
    assert got["RD_POS"].combined < 0.40
    assert got["HQL2"].combined < 0.25
    # s13_c1: 真候选（真 sep 0.50）保持前排
    top2 = {c.system_id for c in ranked[:2]}
    assert "s13_c1" in top2
    # 排序确为降序
    vals = [c.combined for c in ranked]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_combination_beats_single_signal_on_gold():
    """组合 rho ≥ 0.60 且不低于任一单用信号（真 sep 为金标准）。"""
    rows = load_rows()
    strength = {x["system"]: float(x["dl_act"]) for x in rows}
    spec = {x["system"]: float(x["v4_sep"]) for x in rows}
    gold = {x["system"]: float(x["sep_true"]) for x in rows}
    ranked = CombinedScorer(strength.get, spec.get).score(list(strength))
    rho_comb = agreement_rho(ranked, gold)

    r_a = percentile_rank(list(strength.values()))
    r_b = percentile_rank(list(spec.values()))
    keys = list(strength.keys())
    rho_a = float(np.corrcoef(  # spearman 对 rank 后即 pearson
        r_a, percentile_rank([gold[k] for k in keys]))[0, 1])
    rho_b = float(np.corrcoef(
        r_b, percentile_rank([gold[k] for k in keys]))[0, 1])
    assert rho_comb >= 0.60
    assert rho_comb >= max(rho_a, rho_b)


def test_skipped_and_bad_ids():
    strength = {"a": 1.0, "b": 0.5}
    spec = {"a": 0.2}  # b 缺特异性 → 跳过
    scorer = CombinedScorer(strength.get, spec.get)
    ranked = scorer.score(["a", "b", "c"])
    assert [c.system_id for c in ranked] == ["a"]
    assert scorer.skipped == ["b", "c"]
    assert CombinedScorer(strength.get, {}.get).score(["a", "b"]) == []


def test_synthetic_two_dimensional_tradeoff():
    """机制级小考: 一个维度的冠军若在另一维度垫底, 不得夺冠。"""
    strength = {"perfect": 10.0, "balanced": 5.0, "weak": 1.0}
    spec = {"perfect": 0.0, "balanced": 5.0, "weak": 10.0}
    top = CombinedScorer(strength.get, spec.get).score(strength)[0]
    assert top.system_id == "balanced"
    assert isinstance(top, CandidateScore)
