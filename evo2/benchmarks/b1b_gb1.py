#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B1b — GB1 四位点完整景观基准（FLIP two_vs_rest 官方协议对齐）。

数据：Wu et al. 2016 eLife，V39/D40/G41/V54 全组合 149,361 变体（Fitness 0-8.76）。
协议：FLIP two_vs_rest——官方训练侧 = WT + 39/40 位单/双突变（424 条）；测试侧 =
  3-4 突变（8,309 条）。AL 引擎用同预算 B=424（另跑 192/1536 预算曲线）。
景观完整实测 → 自由模式 WF（无行走约束），突变空间 = 4 个文库位点。
先验：1IGD 结构类别（buried/surface）→ 化学先验，零 DMS 泄漏。
v2 打分：混合模型 = one-hot + 突变数 + 先验分特征（模型学先验信任权重），
  对偶 ridge 直接拟合 y；先验仅作为特征与提议偏置参与。
策略：v2 / v1 / random / ridge-AL；ESM3 零样本另行合并（集群打分）。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine.seqtools import AA20, AA2IDX
from engine.kernel import WFKernel
from engine.funnel import OracleLandscape
from engine.reflux import reflux_prior
from engine.baselines import RidgeAL
from engine.metrics import spearman
from engine.priors import build_priors

GB1_CSV = "A:/claudework/evo_data/raw/flip/splits/gb1/four_mutations_full_data.csv"
GB1_WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTELEVLFQ"
SITES = [38, 39, 40, 53]  # 0-based: V39 D40 G41 V54
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
BUDGETS = [192, 424, 1536]
BATCH = 24
NE, NPOP, NGEN = 300, 4, 12


def load_gb1():
    df = pd.read_csv(GB1_CSV, low_memory=False)
    oracle_map = {}
    for v, y in zip(df.Variants, df.Fitness):
        muts = tuple(sorted((SITES[i], c) for i, c in enumerate(v)
                            if c != "VDGV"[i]))
        oracle_map[muts] = float(y)
    train_keys = set()
    tr = df[df.two_vs_rest == "train"]
    for v in tr.Variants:
        train_keys.add(tuple(sorted((SITES[i], c) for i, c in enumerate(v)
                                    if c != "VDGV"[i])))
    test_keys = [tuple(sorted((SITES[i], c) for i, c in enumerate(v)
                              if c != "VDGV"[i]))
                 for v in df[df.two_vs_rest == "test"].Variants]
    return oracle_map, train_keys, test_keys


def build_gb1_priors():
    rows, meta = build_priors(os.path.join(HERE, "data", "1igd.pdb"), GB1_WT,
                              seq_offset=0, protein_chain="A",
                              buried_density=70.0, out_dir=None)
    pri = {}
    for r in rows:
        pri.setdefault(r["seq_idx"], {})[r["aa"]] = r["prior"]
    return pri, meta


class PanelScorer:
    """面板（onehot / count / prior0）三个分量的分段求和打分器。"""

    def __init__(self, keys, sel0, sites):
        self.keys = keys
        self.n = len(keys)
        self.site_j, self.aa_i, self.p0_flat = [], [], []
        ptr = [0]
        for k in keys:
            for i, a in k:
                j = sites.index(i)
                self.site_j.append(j)
                self.aa_i.append(AA2IDX[a])
                self.p0_flat.append(sel0[j, AA2IDX[a]] if sel0 is not None else 0.0)
            ptr.append(len(self.site_j))
        self.site_j = np.array(self.site_j, int)
        self.aa_i = np.array(self.aa_i, int)
        self.p0_flat = np.array(self.p0_flat, float)
        self.ptr = np.array(ptr, int)
        self.seg = np.repeat(np.arange(self.n), np.diff(self.ptr))
        self.counts = np.diff(self.ptr).astype(float)
        self.p0 = np.zeros(self.n)
        np.add.at(self.p0, self.seg, self.p0_flat)  # 先验分按基因型聚合

    def kernel_scores(self, kernel):
        out = np.zeros(self.n)
        np.add.at(out, self.seg, kernel.sel[self.site_j, self.aa_i])
        return out

    def hybrid_scores(self, coef, n_sites):
        """coef: [n_sites*20 (onehot), 1 (count), 1 (prior)]。"""
        out = np.zeros(self.n)
        np.add.at(out, self.seg, coef[self.site_j * 20 + self.aa_i])
        return out + coef[n_sites * 20] * self.counts + coef[-1] * self.p0


def run_strategy(name, pri, oracle_map, train_keys, test_keys, seed, budget):
    rng = np.random.default_rng(seed)
    sites = list(SITES)
    pri = {s: pri[s] for s in SITES}
    wt_idx = [AA2IDX[GB1_WT[s]] for s in sites]
    cfg = dict(mutations_per_genome_per_gen={"lambda": 1.0}, n_mut_max=4, T=0.8,
               proposal_temp=2.0)
    oracle = OracleLandscape(GB1_WT, oracle_map)
    key_list = list(oracle_map.keys())
    t0 = time.time()
    n_dim = len(SITES) * 20 + 2  # onehot + count + prior
    scores_test = np.full(len(test_keys), np.nan)
    scores_train = np.full(len(train_keys), np.nan)

    def fit_and_score(res_model, test_sc, train_sc, sel0):
        X = np.array(res_model.X)
        yv = np.array(res_model.y)
        K = X @ X.T
        alpha = np.linalg.solve(K + res_model.lam * np.eye(len(yv)), yv - yv.mean())
        coef = X.T @ alpha
        coef = np.concatenate([coef, [0.0]])[:n_dim]
        coef[n_dim - 2] = coef[len(SITES) * 20]
        return test_sc.hybrid_scores(coef, len(SITES)), \
            train_sc.hybrid_scores(coef, len(SITES))

    if name in ("v2", "v1"):
        kernel = WFKernel(GB1_WT, sites, pri, cfg, seed=seed, proposal="prior")
        sel0 = kernel.sel0
        prior0 = (lambda m: sum(sel0[int(np.searchsorted(kernel.sites, i)), AA2IDX[a]]
                                for i, a in m))
        res_model = RidgeAL(SITES, wt_idx, rng, lam=5.0, feat_fn=prior0) \
            if name == "v2" else None
        test_sc = PanelScorer(test_keys, sel0, SITES)
        train_sc = PanelScorer(sorted(train_keys), sel0, SITES)
        best_key, best_y = (), -np.inf
        _dbg_rounds = 0
        while oracle.budget_used < budget:
            _dbg_rounds += 1
            if _dbg_rounds > 400:
                break
            # 冠军奠基：每轮从当前最优已测基因型出发（真实 dEvo 的自适应行走）
            founder = wt_idx.copy()
            for i, a in best_key:
                founder[SITES.index(i)] = AA2IDX[a]
            stats, pops = kernel.run(n_pop=NPOP, n_gen=NGEN, Ne=NE,
                                     founder=np.array(founder, dtype=np.int8))
            # 大精英池按适应度排序取新基因型（收敛种群下快速吃满预算）
            cands = kernel.propose_elites(pops, top_k=600,
                                          diversity=4 if name == "v2" else 0)
            batch = [tuple(sorted(kernel.geno_to_muts(g))) for _, g in cands]
            seen = set(oracle.hit_keys)
            elite_new = []
            for k in batch:
                if k not in seen:
                    seen.add(k)
                    elite_new.append([(i, a) for i, a in k])
                if len(elite_new) >= BATCH // 2:
                    break
            # 实测补齐（无偏探索）：与精英提议混批后再截断——精英可能落在
            # 景观未测空洞（数据集 93% 覆盖），补齐保证预算推进
            fill_new = []
            idx = rng.choice(len(key_list), size=min(BATCH * 4, len(key_list)),
                             replace=False)
            for i in idx:
                k = key_list[i]
                if k not in seen:
                    seen.add(k)
                    fill_new.append([(i2, a) for i2, a in k])
                if len(fill_new) >= BATCH // 2:
                    break
            new = (elite_new + fill_new)[:max(1, min(BATCH, budget - oracle.budget_used))]
            if not new:
                kernel.T *= 1.5
                if kernel.T > 30:
                    break
                continue
            ys, ok = oracle.query(new)
            for m, yy, o in zip(new, ys, ok):
                if o and yy > best_y:
                    best_y, best_key = yy, tuple(sorted((int(i), a) for i, a in m))
            if name == "v2" and ok.sum():
                for m, yy, o in zip(new, ys, ok):
                    if o:
                        res_model.observe(m, yy)
                got = [m for m, o in zip(new, ok) if o]
                kernel.prior_table = reflux_prior(
                    kernel.prior_table, kernel.wt_idx,
                    [[(int(np.searchsorted(kernel.sites, i)), a) for i, a in m]
                     for m in got],
                    ys[ok], alpha=0.35, k0=2.0)
                kernel.set_prior_table(kernel.prior_table)
        scores_test = test_sc.kernel_scores(kernel)
        scores_train = train_sc.kernel_scores(kernel)
        if name == "v2" and len(res_model.y) >= 10:
            scores_test, scores_train = fit_and_score(res_model, test_sc,
                                                      train_sc, sel0)
    elif name == "random":
        while oracle.budget_used < budget:
            idx = rng.integers(0, len(key_list), size=BATCH)
            oracle.query([[(i, a) for i, a in key_list[i]] for i in idx])
    elif name == "ridge-AL":
        model = RidgeAL(SITES, wt_idx, rng, lam=5.0)
        test_sc = PanelScorer(test_keys, np.zeros((len(SITES), 20)), SITES)
        train_sc = PanelScorer(sorted(train_keys), np.zeros((len(SITES), 20)), SITES)
        while oracle.budget_used < budget:
            if len(model.y) < BATCH:
                idx = rng.choice(len(key_list), size=BATCH)
                batch = [[(i, a) for i, a in key_list[i]] for i in idx]
            else:
                idx = rng.choice(len(key_list), size=600, replace=False)
                batch = model.propose(BATCH, candidate_pool=
                                      [[(i, a) for i, a in key_list[i]] for i in idx])
            ys, ok = oracle.query(batch)
            for m, yy, o in zip(batch, ys, ok):
                if o:
                    model.observe(m, yy)
            if len(model.y) >= 10:
                scores_test, scores_train = fit_and_score(model, test_sc,
                                                          train_sc, None)
    else:
        raise ValueError(name)

    test_y = np.array([oracle_map[k] for k in test_keys])
    sp_test = spearman(scores_test, test_y)
    sp_train = spearman(scores_train, [oracle_map[k] for k in sorted(train_keys)])
    idx16 = np.argsort(-np.nan_to_num(scores_test, nan=-1e9))[:16]
    top16 = float(np.mean([oracle_map[test_keys[i]] for i in idx16]))
    return dict(strategy=name, seed=seed, budget_target=budget,
                budget_used=oracle.budget_used,
                spearman_test=round(sp_test, 4),
                spearman_train=round(sp_train, 4) if sp_train == sp_train else np.nan,
                top16_test=round(top16, 3),
                wall_s=round(time.time() - t0, 1))


def main():
    oracle_map, train_keys, test_keys = load_gb1()
    pri, meta = build_gb1_priors()
    print(f"oracle {len(oracle_map)}; train {len(train_keys)}; test {len(test_keys)}")
    print("site classes:", {s: meta[s]["dominant_class"] if s in meta else "?"
                            for s in SITES})
    rows = []
    for budget in BUDGETS:
        for seed in SEEDS:
            for name in ["v2", "v1", "random", "ridge-AL"]:
                r = run_strategy(name, pri, oracle_map, train_keys, test_keys,
                                 seed, budget)
                rows.append(r)
                print(f"[{name} B={budget} s={seed}] used={r['budget_used']} "
                      f"rho_test={r['spearman_test']:.3f} "
                      f"rho_train={r['spearman_train']} top16={r['top16_test']} "
                      f"({r['wall_s']}s)", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "b1b_gb1.csv"),
                                      index=False)
    print("B1b done")


if __name__ == "__main__":
    main()
