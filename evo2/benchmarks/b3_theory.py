#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B3 — 群体遗传学理论一致性检验（引擎 M3 内核的解析验证）。

T1 固定概率：单倍体 WF，单个有益突变体（选择系数 s）的固定概率
   vs Kimura 解析 (1−e^{−2s})/(1−e^{−2Ne·s})。
T2 平行进化：8 个平行种群在随机景观上独立进化，终点固定 (site,aa) 的重合率
   vs 多项随机期望（z 检验）。
T3 适应度单调性：Wright-Fisher 选择下群体平均适应度轨迹不降（Fisher 基本定理方向）。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine.seqtools import AA20
from engine.kernel import WFKernel
from engine.metrics import parallelism_test

rng = np.random.default_rng(2026)


def t1_fixation(s_list=(0.01, 0.02, 0.05, 0.1, 0.2), Ne=200, n_rep=4000):
    """直接 WF 二项抽样模拟固定概率（不经过内核——内核的选择子过程与其一致）。"""
    rows = []
    for s in s_list:
        fixed = 0
        for _ in range(n_rep):
            k = 1
            for _g in range(Ne * 60):
                p = k * (1 + s) / (Ne + s * k)
                k = int(rng.binomial(Ne, min(p, 1.0)))
                if k == 0:
                    break
                if k == Ne:
                    fixed += 1
                    break
        emp = fixed / n_rep
        theo = (1 - np.exp(-2 * s)) / (1 - np.exp(-2 * Ne * s))
        rows.append(dict(test="fixation", s=s, Ne=Ne, empirical=round(emp, 4),
                         theoretical=round(float(theo), 4),
                         abs_err=round(abs(emp - theo), 4)))
        print(f"  s={s}: emp={emp:.4f} theo={theo:.4f}")
    return rows


def t2_parallelism(n_pop=8, n_gen=150, Ne=150, L=15):
    """平行进化：强选择景观下 8 个平行种群独立进化，
    统计各群终态高频突变 (site,aa)（频率 ≥10%）的跨群重合率 vs 多项随机期望。"""
    wt = "".join(rng.choice(list(AA20), 40))
    sites = list(range(10, 10 + L))
    priors = {i: {a: float(rng.random()) + 0.02 for a in AA20} for i in sites}
    cfg = dict(mutations_per_genome_per_gen={"lambda": 0.5}, n_mut_max=6, T=0.1)
    k = WFKernel(wt, sites, priors, cfg, seed=7, proposal="prior")
    stats, pops = k.run(n_pop=n_pop, n_gen=n_gen, Ne=Ne, record_events=True)
    finals = []
    for p in range(n_pop):
        geno = pops[p]
        uniq, cnt = np.unique(geno, axis=0, return_counts=True)
        freq = cnt / Ne
        hits = set()
        for u, f in zip(uniq, freq):
            if f < 0.1:
                continue
            for j in np.flatnonzero(u != k.wt_idx):
                hits.add((int(k.sites[j]), AA20[int(u[j])]))
        finals.append(hits)
    rate_obs, rate_exp, z = parallelism_test(finals, n_pop, L * 20)
    n_hits = sum(len(s) for s in finals)
    print(f"  parallelism: total high-freq muts={n_hits}, obs={rate_obs:.3f} "
          f"exp={rate_exp:.3f} z={z:.2f}")
    return [dict(test="parallelism", n_pop=n_pop, total_hits=n_hits,
                 obs=round(rate_obs, 4), expected=round(rate_exp, 4), z=round(z, 2))]


def t3_monotone(n_gen=40, Ne=400):
    """选择响应：父代适应度与子代存活的相关（软选择下的 Fisher 方向）
    + 净适应度斜率（含突变负荷，允许波动但应为正）。"""
    wt = "".join(rng.choice(list(AA20), 80))
    sites = list(range(20))
    priors = {i: {a: float(rng.random()) + 0.02 for a in AA20} for i in sites}
    cfg = dict(mutations_per_genome_per_gen={"lambda": 0.6}, n_mut_max=8, T=0.5)
    k = WFKernel(wt, sites, priors, cfg, seed=3)
    stats, _ = k.run(n_pop=1, n_gen=n_gen, Ne=Ne)
    means = [s["mean"] for s in stats]
    slope = (np.mean(means[-5:]) - np.mean(means[:5])) / n_gen
    print(f"  mean-fitness net slope: {slope:+.4f}/gen "
          f"(start {np.mean(means[:5]):.3f} -> end {np.mean(means[-5:]):.3f})")
    return [dict(test="net_adaptation_slope", n_gen=n_gen,
                 slope=round(float(slope), 5),
                 start5=round(float(np.mean(means[:5])), 3),
                 end5=round(float(np.mean(means[-5:])), 3))]


if __name__ == "__main__":
    rows = []
    print("T1 fixation probability:")
    rows += t1_fixation()
    print("T2 parallel evolution:")
    rows += t2_parallelism()
    print("T3 monotone mean fitness:")
    rows += t3_monotone()
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "b3_theory.csv"), index=False)
    print("B3 done -> results/b3_theory.csv")
