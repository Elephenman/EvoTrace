# -*- coding: utf-8 -*-
"""CombinedScorer —— 强度×特异性双代理互补打分器。

机制提案: evotrace_restore/提案_互补双代理打分器.md（2026-09-05 评审通过）
证据基线: evo2/results/dl_v7sep/combine_result.csv（n100, 30 系统）
  min(rank(DL_act), rank(v4_sep)) vs 真 sep rho=+0.630（全组最高, 超 v4 单用 +0.606）；
  反例 RD_POS: DL_act 排名第 1 但真 sep 仅 0.01, 被组合正确压后。

分工定型（2026-09-05 用户拍板）: DL(A1) 管结合强度 act/dual, v4(GBM) 管特异性 sep。
组合规则: 默认 min-rank —— 两维都高才推荐, 单高被惩罚。

设计约束: 本模块是纯函数式组合器, 不含任何模型/IO——强度与特异性代理以可调用对象
注入（`score(system_id) -> float` 协议），战役脚本与主线漏斗共用同一实现。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

__all__ = ["CandidateScore", "CombinedScorer", "percentile_rank", "combine_rule"]

# 组合规则注册表: 输入两个同长 percentile-rank 数组, 输出组合分。
# 概念验证结论（combine_result.csv）: min 最优, 加权和/乘积均低于 min。
COMBINE_RULES = {
    "min": lambda a, b: np.minimum(a, b),
    "mean": lambda a, b: 0.5 * (a + b),
    "product": lambda a, b: a * b,
}


def percentile_rank(values: Sequence[float]) -> np.ndarray:
    """平均秩百分位（升序）: 最大值 → 1.0, 最小值 → 接近 0。

    与验证脚本 verify_combine_v4_dl.py 的 rank() 口径一致（rankdata/n）,
    保证回归测试可直接对上 combine_result.csv 的 rank_* 列。
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    return rankdata(v, method="average") / v.size


def combine_rule(name: str, rank_a: np.ndarray, rank_b: np.ndarray) -> np.ndarray:
    if name not in COMBINE_RULES:
        raise KeyError(f"未知组合规则 {name!r}, 可选: {sorted(COMBINE_RULES)}")
    return COMBINE_RULES[name](np.asarray(rank_a, float), np.asarray(rank_b, float))


@dataclass
class CandidateScore:
    """单个候选（系统/变体）的双维评分与组合分。"""

    system_id: str
    strength: float          # 结合强度维（例: DL A1 的 act(S1) 预测）
    specificity: float       # 特异性维（例: v4 GBM 的 sep 预测）
    combined: float          # 组合分（默认 min-rank, 双高才高）

    def __str__(self) -> str:  # 便于战役脚本直接打印
        return (f"{self.system_id}: strength={self.strength:+.3f} "
                f"specificity={self.specificity:+.3f} combined={self.combined:.3f}")


class CombinedScorer:
    """双代理互补打分器。

    Parameters
    ----------
    strength_scorer, specificity_scorer:
        可调用对象, 满足 `score(system_id) -> float` 协议; 对无预测的候选返回 None
        或抛 KeyError（两种风格都支持, 由本类统一跳过并计入 self.skipped）。
        任何未来的代理模型实现同一协议即可换装, 无需改本类。
    rule:
        组合规则名, 默认 "min"（概念验证最优）。"mean"/"product" 备查。
    """

    def __init__(self,
                 strength_scorer: Callable[[str], float],
                 specificity_scorer: Callable[[str], float],
                 rule: str = "min"):
        self.strength_scorer = strength_scorer
        self.specificity_scorer = specificity_scorer
        self.rule = rule

    def score(self, system_ids: Iterable[str]) -> List[CandidateScore]:
        """对一批候选打分, 按组合分降序返回。

        单代理打不出分的候选（KeyError）被跳过并计入 self.skipped,
        保证一个坏代理不拖垮整批提名。
        """
        ids: List[str] = list(system_ids)
        self.skipped: List[str] = []
        strength: Dict[str, float] = {}
        specificity: Dict[str, float] = {}
        for sid in ids:
            try:
                a = self.strength_scorer(sid)
                b = self.specificity_scorer(sid)
            except KeyError:
                self.skipped.append(sid)
                continue
            if a is None or b is None:  # 代理可用 .get 风格返回 None 表示无预测
                self.skipped.append(sid)
                continue
            strength[sid] = float(a)
            specificity[sid] = float(b)
        if not strength:
            return []
        ordered = list(strength.keys())
        rk_a = percentile_rank([strength[s] for s in ordered])
        rk_b = percentile_rank([specificity[s] for s in ordered])
        comb = combine_rule(self.rule, rk_a, rk_b)
        out = [CandidateScore(s, strength[s], specificity[s], float(c))
               for s, c in zip(ordered, comb)]
        return sorted(out, key=lambda x: x.combined, reverse=True)

    def score_to_dict(self, system_ids: Iterable[str]) -> Dict[str, CandidateScore]:
        return {c.system_id: c for c in self.score(system_ids)}


def agreement_rho(scores: Sequence[CandidateScore],
                  gold: Dict[str, float]) -> float:
    """组合分 vs 金标准的 Spearman 相关（评测/监控用, 非打分路径）。"""
    pairs = [(c.combined, gold[c.system_id])
             for c in scores if c.system_id in gold and np.isfinite(gold[c.system_id])]
    if len(pairs) < 3:
        raise ValueError(f"可用配对不足: {len(pairs)}")
    pred, truth = zip(*pairs)
    return float(spearmanr(pred, truth).statistic)
