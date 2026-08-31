#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B2 — TEM-1 (BLAT_ECOLX) 跨条件迁移基准。

数据：同一 TEM-1 蛋白的四个独立 DMS 研究（均为单突变，286 aa，不同表型/条件）：
  Firnberg 2014（训练域）/ Deng 2012 / Stiffler 2015 / Jacquier 2013（迁移域）。
协议：均匀先验（无结构信息——检验引擎纯数据驱动工作）+ 催化位点冻结
  （S70/K73/S130/N132/E166/K234/T235）。行走在 Firnberg 实测单突变集上。
预算 B ∈ {192, 768}，8 seeds。
策略：v2（行走进化 + 回流 + 残差混合分）/ ridge-AL（UCB）/ random。
指标：训练域 ρ、三迁移域 ρ（排名口径）、best_found。
"""
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine.seqtools import AA20, AA2IDX, parse_mutant
from engine.kernel import WFKernel
from engine.funnel import OracleLandscape
from engine.reflux import reflux_prior
from engine.baselines import RidgeAL
from engine.metrics import spearman

import json as _json
_BLOSUM = _json.load(open("A:/Data/ensemble-protein-scorer/tools/blosum62.json"))["matrix"]
_B62 = {(k[0], k[1]): float(v) for k, v in _BLOSUM.items()}


def blosum_prior(wt):
    """无结构模式的廉价先验：BLOSUM62 替换分数 softmax（τ=2）。"""
    pri = {}
    for i, w in enumerate(wt):
        d = {}
        for a in AA20:
            s = _B62.get((w, a), -4.0) if a != w else 2.0
            d[a] = float(np.exp(s / 2.0))
        z = sum(d.values())
        pri[i] = {a: v / z for a, v in d.items()}
    return pri

PG = ("A:/claudework/evo_data/processed/proteingym_benchmark/"
      "DMS_ProteinGym_substitutions")
TRAIN = "BLAT_ECOLX_Firnberg_2014.csv"
TRANSFER = ["BLAT_ECOLX_Deng_2012.csv", "BLAT_ECOLX_Stiffler_2015.csv",
            "BLAT_ECOLX_Jacquier_2013.csv"]
CATALYTIC = {69, 72, 129, 131, 165, 233, 234}  # S70 K73 S130 N132 E166 K234 T235（0-based）
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
BUDGETS = [192, 768]
BATCH = 24
NE, NPOP, NGEN = 300, 4, 8


def rebuild_wt(df):
    """WT = 各位置在 mutated_sequence 中的众数残基（单突变库中多数序列在该位为 WT）。"""
    from collections import defaultdict
    cols = defaultdict(Counter)
    for s in df.mutated_sequence:
        for i, c in enumerate(s):
            cols[i][c] += 1
    L = max(cols) + 1
    return "".join(cols[i].most_common(1)[0][0] for i in range(L))


def load_study(fn, wt):
    df = pd.read_csv(os.path.join(PG, fn))
    out = {}
    for m, y in zip(df.mutant, df.DMS_score):
        muts = parse_mutant(m)
        if muts and all(0 <= i < len(wt) and wt[i] == str(m).split(":")[j][0]
                        for j, (i, a) in enumerate(muts)):
            out[tuple(sorted(muts))] = float(y)
    return out


def score_panel(sites, sel, keys):
    seg_s, seg_a = [], []
    ptr = [0]
    for k in keys:
        for i, a in k:
            j = int(np.searchsorted(sites, i))
            seg_s.append(j)
            seg_a.append(AA2IDX[a])
        ptr.append(len(seg_s))
    seg_s, seg_a, ptr = map(np.array, (seg_s, seg_a, ptr))
    seg_id = np.repeat(np.arange(len(keys)), np.diff(ptr))
    sc = np.zeros(len(keys))
    np.add.at(sc, seg_id, sel[seg_s, seg_a])
    return sc, (seg_s, seg_a, seg_id)


def main():
    df_tr = pd.read_csv(os.path.join(PG, TRAIN))
    wt = rebuild_wt(df_tr)
    oracle_map = load_study(TRAIN, wt)
    transfer = {fn: load_study(fn, wt) for fn in TRANSFER}
    sites = np.array([i for i in range(len(wt)) if i not in CATALYTIC])
    print(f"train {len(oracle_map)} singles; transfer " +
          " ".join(f"{fn.split('_')[2]}={len(v)}" for fn, v in transfer.items()))
    key_list = list(oracle_map.keys())
    key_y = np.array([oracle_map[k] for k in key_list])
    rows = []
    for budget in BUDGETS:
        for seed in SEEDS:
            t0 = time.time()
            rng = np.random.default_rng(seed)
            oracle = OracleLandscape(wt, oracle_map)
            # 无结构模式廉价先验：BLOSUM62 替换分数 softmax（τ=2）
            pri_full = blosum_prior(wt)
            pri = {int(s): pri_full[s] for s in sites}
            cfg = dict(mutations_per_genome_per_gen={"lambda": 2.0}, n_mut_max=3,
                       T=1.2, proposal_temp=1.0)
            kernel = WFKernel(wt, sites, pri, cfg, seed=seed, measured_keys=oracle_map,
                              proposal="prior")
            sel0 = kernel.sel0  # 固定初始先验分特征
            prior0 = (lambda m: sum(sel0[int(np.searchsorted(kernel.sites, i)), AA2IDX[a]]
                                    for i, a in m))
            res_model = RidgeAL(sites, [AA2IDX[wt[s]] for s in sites], rng, lam=5.0,
                                feat_fn=prior0)
            seen = set()
            while oracle.budget_used < budget:
                stats, pops = kernel.run(n_pop=NPOP, n_gen=NGEN, Ne=NE)
                cands = kernel.propose_elites(pops, top_k=BATCH * 2, diversity=0)
                new = []
                for _, g in cands:
                    k = tuple(sorted(kernel.geno_to_muts(g)))
                    if k not in seen and k not in oracle.hit_keys:
                        seen.add(k)
                        new.append([(i, a) for i, a in k])
                    if len(new) >= BATCH:
                        break
                if not new:
                    kernel.T *= 1.5
                    if kernel.T > 50:
                        break
                    continue
                ys, ok = oracle.query(new)
                if ok.sum():
                    got = [m for m, o in zip(new, ok) if o]
                    kernel.prior_table = reflux_prior(
                        kernel.prior_table, kernel.wt_idx,
                        [[(int(np.searchsorted(kernel.sites, i)), a) for i, a in m]
                         for m in got],
                        ys[ok], alpha=0.35, k0=2.0)
                    kernel.set_prior_table(kernel.prior_table)
                    for m, yy, o in zip(new, ys, ok):
                        if o:
                            res_model.observe(m, yy)
            # ---- 打分与评估
            key_idx = {k: n for n, k in enumerate(key_list)}
            prior_sc, (seg_s, seg_a, seg_id) = score_panel(
                sites, kernel.sel, key_list)
            resid_sc = np.zeros(len(key_list))
            if len(res_model.y) >= 10:
                X = np.array(res_model.X)
                yv = np.array(res_model.y)
                K = X @ X.T
                alpha = np.linalg.solve(K + res_model.lam * np.eye(len(yv)), yv - yv.mean())
                coef = X.T @ alpha  # [onehot, count, prior]
                n_site_dim = len(sites) * 20
                np.add.at(resid_sc, seg_id, coef[:n_site_dim][seg_s * 20 + seg_a])
                cnt = np.bincount(seg_id, minlength=len(key_list)).astype(float)
                resid_sc = resid_sc + coef[n_site_dim] * cnt
                p0 = np.zeros(len(key_list))
                np.add.at(p0, seg_id, sel0[seg_s, seg_a])
                resid_sc = resid_sc + coef[-1] * p0
            v2_scores = prior_sc + resid_sc
            v2_map = {k: v2_scores[n] for k, n in key_idx.items()}
            # ridge-AL 对照分（同一标签、无先验特征的纯 ridge）
            r_sc = np.zeros(len(key_list))
            if len(res_model.y) >= 10:
                yv = np.array(res_model.y)
                X0 = np.array(res_model.X)[:, :len(sites) * 20 + 1]
                K0 = X0 @ X0.T
                alpha0 = np.linalg.solve(K0 + 5.0 * np.eye(len(yv)), yv - yv.mean())
                coef0 = X0.T @ alpha0
                rv = coef0[:len(sites) * 20][seg_s * 20 + seg_a]
                np.add.at(r_sc, seg_id, rv)
                cnt = np.bincount(seg_id, minlength=len(key_list)).astype(float)
                r_sc = r_sc + coef0[-1] * cnt
            r_map = {k: r_sc[n] for k, n in key_idx.items()}
            row = dict(budget=budget, seed=seed,
                       used=oracle.budget_used,
                       best_found=round(max(oracle.measured[k] for k in oracle.hit_keys), 3),
                       v2_rho=round(spearman(v2_scores, key_y), 4),
                       ridge_rho=round(spearman(r_sc, key_y), 4),
                       wall=round(time.time() - t0, 1))
            for fn in TRANSFER:
                tk = list(transfer[fn].keys())
                ty = list(transfer[fn].values())
                row[f"v2_{fn.split('_')[2]}"] = round(
                    spearman([v2_map.get(k, np.nan) for k in tk], ty), 4)
                row[f"ridge_{fn.split('_')[2]}"] = round(
                    spearman([r_map.get(k, np.nan) for k in tk], ty), 4)
            rows.append(row)
            print(f"[B={budget} s={seed}] v2 rho={row['v2_rho']:.3f} "
                  f"ridge rho={row['ridge_rho']:.3f} best={row['best_found']:.2f} "
                  f"({row['wall']}s)")
            pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "b2_tem1.csv"),
                                      index=False)
    print("B2 done")


if __name__ == "__main__":
    main()
