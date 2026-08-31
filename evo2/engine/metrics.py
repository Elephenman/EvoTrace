#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoTrace v2 — 指标库：Spearman、标签经济、固定概率理论、平行进化检验。"""
import numpy as np


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 3:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def kimura_fixation_prob(s, Ne, n_rep=20000, rng=None):
    """单倍体 Wright-Fisher 固定概率经验值 + Kimura 解析值 (1-e^{-2s})/(1-e^{-2Ns})。"""
    rng = rng or np.random.default_rng(0)
    emp = 0
    for _ in range(n_rep):
        k = 1  # 初始 1 个体
        for _g in range(int(40 * Ne)):
            # 二项抽样：P(每个后代取突变体) = k/Ne * (1+s) / (1 + s*k/Ne) 近似
            p = k * (1 + s) / (Ne + s * k)
            k = int(rng.binomial(Ne, min(p, 1.0)))
            if k == 0:
                break
            if k == Ne:
                emp += 1
                break
    emp /= n_rep
    theo = (1 - np.exp(-2 * s)) / (1 - np.exp(-2 * Ne * s))
    return emp, float(theo)


def parallelism_test(final_site_counts, n_pop, n_sites, n_aa=20):
    """平行进化检验：各平行种群独立固定（或命中）位点的重合度 vs 多项随机。

    final_site_counts: [n_pop] list of Counter（各群体终态的突变 (site, aa) 集）。
    返回观测重合对比例 vs 随机期望的 z 值。
    """
    obs_pairs, tot_pairs = 0, 0
    for i in range(n_pop):
        for j in range(i + 1, n_pop):
            si, sj = set(final_site_counts[i]), set(final_site_counts[j])
            obs_pairs += len(si & sj)
            tot_pairs += 1
    rate_obs = obs_pairs / max(tot_pairs, 1)
    # 期望：位点被任一群命中的概率近似 p = mean_count / (n_sites*n_aa)
    total_hits = sum(len(c) for c in final_site_counts)
    p = total_hits / max(n_pop * n_sites * n_aa, 1)
    exp_rate = p * n_pop  # 近似期望重合对比例
    var = exp_rate * (1 - p) / max(tot_pairs, 1)
    z = (rate_obs - exp_rate) / np.sqrt(max(var, 1e-12))
    return rate_obs, exp_rate, float(z)


def label_economy_curve(history_by_budget, truth_top):
    """budget -> best_y 曲线。history_by_budget: {B: best_y}。"""
    bs = sorted(history_by_budget)
    return bs, [history_by_budget[b] for b in bs]


def diversity_hamming(genos):
    """基因型集合的平均两两 Hamming 距离。"""
    G = np.asarray(genos)
    n = len(G)
    if n < 2:
        return 0.0
    tot, cnt = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += int((G[i] != G[j]).sum())
            cnt += 1
    return tot / cnt
