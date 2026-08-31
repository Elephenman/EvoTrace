#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B1a — avGFP 完整实测景观上的标签经济性基准。

数据：ProteinGym 版 Sarkisyan 2016（51,714 变体，log10 亮度）。
先验：1EMA 结构推导（chromophore_edge/core / buried / surface），零 DMS 信息泄漏。
协议（文库约束）：所有策略共享同一 oracle 与同一候选库（结构可变位点集内的实测
基因型）。v2/v1 的进化提议 = 实测图上的一步编辑行走（library-constrained
adaptive walk）；随机基线 = 文库的均匀/先验加权筛选。预算 = 实际标签数。
策略：v2（行走进化 + 回流 α=0.35 + 多样性批次）/ v1（行走进化、静态先验、无学习）
      / random（文库均匀筛选）/ random-prior（先验加权筛选）
      / ridge-AL（Wu 2019 家族：对偶 ridge + bootstrap UCB，候选 = 文库抽样 +
      已测最优的 1-编辑邻域）。
指标：best_found（已测最大 y）、pred_top16（模型分 top-16 的真值均值，全面板）、
      Spearman(模型分, 真值) @ 5000 子样。
运行：python benchmarks/b1a_avgfp.py [--quick]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine.seqtools import AA20, AA2IDX, parse_mutant
from engine.priors import load_priors_csv
from engine.kernel import WFKernel
from engine.funnel import OracleLandscape
from engine.reflux import reflux_prior
from engine.baselines import RidgeAL
from engine.metrics import spearman

DMS = ("A:/claudework/evo_data/processed/proteingym_benchmark/"
       "DMS_ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv")
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
BUDGET_TARGETS = [96, 192, 384]
BATCH = 24
NE, NPOP, NGEN = 300, 4, 8
STRATS = ["v2", "v2-prior", "v1", "random", "random-prior", "ridge-AL"]


def load_oracle():
    df = pd.read_csv(DMS)
    wt = json.load(open(os.path.join(HERE, "data_avgfp_wt.json")))["wt"]
    pri, wt_aa, cls = load_priors_csv(os.path.join(HERE, "data", "priors_avgfp.csv"))
    oracle_map, dropped = {}, 0
    for m, y in zip(df.mutant, df.DMS_score):
        muts = parse_mutant(m)
        ok = bool(muts)
        for tok in str(m).split(":"):
            if len(tok) >= 4:
                i = int(tok[1:-1]) - 1
                if i not in pri or wt[i] != tok[0]:
                    ok = False
                    break
        if not ok:
            dropped += 1
            continue
        oracle_map[tuple(sorted((int(i), a) for i, a in muts))] = float(y)
    return wt, pri, cls, oracle_map, dropped


class PanelScorer:
    """加性模型分 -> 评测面板。"""

    def __init__(self, wt, pri, panel_keys):
        self.keys = panel_keys
        self.site_lookup = {int(s): j for j, s in enumerate(sorted(pri.keys()))}
        self.site_j, self.aa_i, self.ptr = [], [], [0]
        for key in panel_keys:
            for i, a in key:
                self.site_j.append(self.site_lookup[i])
                self.aa_i.append(AA2IDX[a])
            self.ptr.append(len(self.site_j))
        self.site_j = np.array(self.site_j, int)
        self.aa_i = np.array(self.aa_i, int)
        self.ptr = np.array(self.ptr, int)
        self.seg = np.repeat(np.arange(len(panel_keys)), np.diff(self.ptr))
        self.counts = np.diff(self.ptr).astype(float)

    def score_kernel(self, kernel):
        vals = kernel.sel[self.site_j, self.aa_i]
        out = np.zeros(len(self.keys))
        np.add.at(out, self.seg, vals)
        return out

    def score_ridge(self, beta_full):
        """beta_full: [n_sites*20+1]，末位 = 突变计数系数。"""
        vals = beta_full[self.site_j * 20 + self.aa_i]
        out = np.zeros(len(self.keys))
        np.add.at(out, self.seg, vals)
        return out + beta_full[-1] * self.counts


def evaluate(scorer, oracle, panel, scores, rng):
    ys_hit = [oracle.measured[k] for k in oracle.hit_keys]
    best_found = max(ys_hit) if ys_hit else np.nan
    if np.all(np.isnan(scores)):
        idx16 = rng.choice(len(panel), size=16, replace=False)
    else:
        idx16 = np.argsort(-np.nan_to_num(scores, nan=-1e9))[:16]
    top16 = [oracle.measured[panel[i]] for i in idx16 if panel[i] in oracle.measured]
    sub = rng.choice(len(panel), size=min(5000, len(panel)), replace=False)
    sp = spearman(scores[sub], [oracle.measured[panel[i]] for i in sub])
    return best_found, float(np.mean(top16)), sp


def prior_score_of(key, sel_of_site):
    return sum(sel_of_site(i, a) for i, a in key)


def run_strategy(name, wt, pri, oracle_map, panel_keys, scorer, seed, budget_target):
    rng = np.random.default_rng(seed)
    sites = sorted(pri.keys())
    wt_idx = [AA2IDX[wt[s]] for s in sites]
    cfg = dict(mutations_per_genome_per_gen={"lambda": 2.0}, n_mut_max=10, T=0.6,
               proposal_temp=2.0)
    oracle = OracleLandscape(wt, oracle_map)
    history = []
    t0 = time.time()
    scores = np.full(len(panel_keys), np.nan)
    key_list = list(oracle_map.keys())
    key_y = np.array([oracle_map[k] for k in key_list])

    if name in ("v2", "v1", "v2-prior"):
        if name.startswith("v2"):
            cfg["T"] = 1.2  # 探索温度：防种群过早收敛
        kernel = WFKernel(wt, sites, pri, cfg, seed=seed, measured_keys=oracle_map,
                          proposal="prior")
        sel0 = kernel.sel0  # 固定初始先验分（先验信任度特征的基准）
        max_rounds = 30
        # 先验作特征：混合模型 = onehot + count + prior0，模型学先验信任权重
        use_prior_feat = (name == "v2")  # v2-prior 消融：不含先验特征
        prior0 = (lambda m: sum(sel0[int(np.searchsorted(kernel.sites, i)), AA2IDX[a]]
                                for i, a in m)) if use_prior_feat else None
        res_model = RidgeAL(sites, wt_idx, rng, lam=5.0, feat_fn=prior0) \
            if name.startswith("v2") else None
        # 探索配比：每轮 = 精英行走提议 + 文库探索（均匀随机，无偏覆盖）
        expl_quota = BATCH // 2 if name.startswith("v2") else 0
        w_explore = np.ones(len(key_list))
        w_explore = w_explore / w_explore.sum()
        while oracle.budget_used < budget_target and len(history) < max_rounds:
            stats, pops = kernel.run(n_pop=NPOP, n_gen=NGEN, Ne=NE)
            cands = kernel.propose_elites(pops, top_k=BATCH * 2, diversity=6)
            muts_batch = [kernel.geno_to_muts(g) for _, g in cands]
            # 去重：只查未测过的（湿实验对应"不重复测同变体"），批量吃满预算
            seen = set(oracle.hit_keys)
            new = []
            for m in muts_batch:
                k = tuple(sorted((int(i), a) for i, a in m))
                if k not in seen:
                    seen.add(k)
                    new.append(m)
                if len(new) >= BATCH - expl_quota:
                    break
            if expl_quota:
                idx = rng.choice(len(key_list), size=expl_quota * 3, p=w_explore, replace=False)
                for i in idx:
                    k = key_list[i]
                    if k not in seen:
                        seen.add(k)
                        new.append([(i2, a) for i2, a in k])
                    if len(new) >= BATCH:
                        break
            cap = max(1, min(BATCH, budget_target - oracle.budget_used))
            new = new[:cap]
            if not new:
                kernel.T *= 1.5  # 提议枯竭 -> 升温探索
                continue
            ys, ok = oracle.query(new)
            if name.startswith("v2") and ok.sum():
                for m, yy, o in zip(new, ys, ok):
                    if o:
                        res_model.observe(m, yy)  # 标签直接拟合（先验作为特征）
                got = [m for m, o in zip(new, ok) if o]
                kernel.prior_table = reflux_prior(
                    kernel.prior_table, kernel.wt_idx,
                    [[(int(np.searchsorted(kernel.sites, i)), a) for i, a in m] for m in got],
                    ys[ok], alpha=0.35, k0=2.0)
                kernel.set_prior_table(kernel.prior_table)
            history.append(dict(round=len(history), budget=oracle.budget_used,
                                best_label=float(ys[ok].max()) if ok.sum() else np.nan))
        scores = scorer.score_kernel(kernel)
        if name.startswith("v2") and len(res_model.y) >= 10:
            # 混合模型面板打分：onehot + count + (v2: prior0) 特征
            X = np.array(res_model.X)
            yv = np.array(res_model.y)
            K = X @ X.T
            alpha = np.linalg.solve(K + res_model.lam * np.eye(len(yv)), yv - yv.mean())
            coef = X.T @ alpha
            onehot_part = np.zeros(len(panel_keys))
            np.add.at(onehot_part, scorer.seg, coef[scorer.site_j * 20 + scorer.aa_i])
            count_coef = coef[len(sites) * 20]
            scores = onehot_part + count_coef * scorer.counts
            if use_prior_feat:
                p0 = np.zeros(len(panel_keys))
                np.add.at(p0, scorer.seg, sel0[scorer.site_j, scorer.aa_i])
                scores = scores + coef[-1] * p0
    elif name in ("random", "random-prior"):
        # 文库筛选基线：每轮均匀/先验加权抽 batch 个实测基因型送检
        p_tab = np.array([[pri[s].get(a, 1e-3) for a in AA20] for s in sites])
        site_lookup = {s: j for j, s in enumerate(sites)}
        w = []
        for key in key_list:
            if name == "random":
                w.append(1.0)
            else:
                w.append(float(np.prod([p_tab[site_lookup[i], AA2IDX[a]] for i, a in key])))
        w = np.asarray(w)
        w = w / w.sum()
        while oracle.budget_used < budget_target:
            idx = rng.choice(len(key_list), size=BATCH, p=w)
            batch = [key_list[i] for i in idx]
            ys, ok = oracle.query(batch)
            history.append(dict(round=len(history), budget=oracle.budget_used,
                                best_label=float(ys[ok].max()) if ok.sum() else np.nan))
    elif name == "ridge-AL":
        model = RidgeAL(sites, wt_idx, rng, lam=5.0)
        site_lookup = {s: j for j, s in enumerate(sites)}
        pool_sites = [s for s in sites if len(pri[s]) > 0]
        n_dim = len(sites) * 20 + 1
        beta_full = np.zeros(n_dim)
        while oracle.budget_used < budget_target:
            if len(model.y) < BATCH:
                idx = rng.choice(len(key_list), size=BATCH)
                batch = [ [ (i, a) for i, a in key_list[i] ] for i in idx ]
            else:
                # 候选池 = 随机文库抽样 + 已测 top-32 的 1-编辑邻域（限实测集内）
                y_arr = np.array(model.y)
                top_keys = [model.X_muts[i] for i in np.argsort(-y_arr)[:32]]
                already = set(model.X_muts)
                pool = set()
                for k in top_keys:
                    ks = set(k)
                    for s in pool_sites:
                        cur = [(i, a) for i, a in k if i != s]
                        wt_aa = wt[site_lookup[s]]
                        for a in pri[s]:
                            if a == wt_aa:
                                nk = tuple(cur)
                            else:
                                nk = tuple(sorted(cur + [(s, a)]))
                            if nk in oracle_map and nk not in already:
                                pool.add(nk)
                rnd_idx = rng.choice(len(key_list), size=400, replace=False)
                pool.update(key_list[i] for i in rnd_idx)
                batch = [[(i, a) for i, a in k] for k in pool]
                batch = model.propose(BATCH, candidate_pool=batch)
            ys, ok = oracle.query(batch)
            for m, yy, o in zip(batch, ys, ok):
                if o:
                    model.observe(m, yy)
            if len(model.y) >= 10:
                X = np.array(model.X)
                yv = np.array(model.y)
                K = X @ X.T
                ybar = yv.mean()
                alpha = np.linalg.solve(K + model.lam * np.eye(len(yv)), yv - ybar)
                coef = X.T @ alpha
                beta_full = np.zeros(n_dim)
                beta_full[:len(coef)] = coef
                scores = scorer.score_ridge(beta_full)
            history.append(dict(round=len(history), budget=oracle.budget_used,
                                best_label=float(max(model.y)) if model.y else np.nan))
    else:
        raise ValueError(name)

    best_found, pred_top16, sp = evaluate(scorer, oracle, panel_keys, scores, rng)
    return dict(strategy=name, seed=seed, budget_target=budget_target,
                budget_used=oracle.budget_used, best_found=round(best_found, 4),
                pred_top16=round(pred_top16, 4) if pred_top16 == pred_top16 else np.nan,
                spearman_5k=round(sp, 4) if sp == sp else np.nan,
                violations=oracle.violations, wall_s=round(time.time() - t0, 1),
                history=json.dumps(history))




def main(quick=False):
    wt, pri, cls, oracle_map, dropped = load_oracle()
    panel_keys = list(oracle_map.keys())
    scorer = PanelScorer(wt, pri, panel_keys)
    print(f"oracle: {len(oracle_map)} genotypes (dropped {dropped}); panel={len(panel_keys)}")
    yvals = np.array(list(oracle_map.values()))
    print(f"y: min {yvals.min():.3f} max {yvals.max():.3f} mean {yvals.mean():.3f}")
    seeds = SEEDS[:2] if quick else SEEDS
    budgets = [96, 192] if quick else BUDGET_TARGETS
    out = os.path.join(ROOT, "results", "b1a_results.csv")
    rows = []
    for name in STRATS:
        for B in budgets:
            for seed in seeds:
                r = run_strategy(name, wt, pri, oracle_map, panel_keys, scorer, seed, B)
                rows.append(r)
                print(f"[{name} B={B} s={seed}] used={r['budget_used']} "
                      f"best={r['best_found']:.3f} top16={r['pred_top16']} "
                      f"rho={r['spearman_5k']} viol={r['violations']} ({r['wall_s']}s)")
                pd.DataFrame(rows).to_csv(out, index=False)
    print("B1a done:", len(rows), "runs ->", out)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
