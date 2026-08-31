#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B7 — 三方优化器对比（Wright-Fisher vs RL vs ES），共享 oracle 套件。

核心设计（v1，2026-08-31）：
  * 同一 oracle 接口 + 同一评估预算（默认 4000 次 evaluate）+ 5 seeds。
  * oracle 套件：
      ppri_additive — PprI 加性（闭式最优已知 → 平凡性对照）
      nk_k0/k1/k2/k3 — NK 可控上位性梯度（K=0 可分离）
      gb1_pairwise  — GB1 实测配对上位性（Olson 2014，真实数据）
      ppri_cal      — PprI 加性 + GB1 上位性幅度校准（L=53，合成但校准）
  * 优化器动态发现：engine.wfopt 必有；rlpolicy / esopt 视分支而定。
  * 非闭式景观用贪心坐标上升（多重启、大预算）给出参考最优。

用法：
  python b7_three_way.py [--budget 4000] [--seeds 5] [--oracles all] [--tag dev]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine.oracle import (Oracle, AdditiveOracle, NKOracle, GB1PairwiseOracle,
                           PprICalibratedOracle, build_ppri_additive)
from engine.seqtools import AA20, AA2IDX
import engine.wfopt  # noqa: F401  注册 wf


# ----------------------------------------------------------------------
def greedy_reference(oracle, budget=200000, restarts=6, seed=0):
    """贪心参考最优：多重启（随机位点顺序）+ 单点扫描 + 换位（swap）阶段。

    换位阶段解决"按索引顺序贪心占满 max_mut 名额"的陷阱：
    候选 = 把一个已突变位点回退 WT + 把一个未突变位点换成其当前最优 AA。
    r>0 的重启用随机基因型起点 —— 处理符号上位性景观（如 GB1：
    单突变均不优于 WT，从 WT 出发贪心第一步即被困）。
    """
    rng = np.random.default_rng(seed)
    L = oracle.L
    max_mut = oracle.max_mut or 12
    wt_f = float(oracle.evaluate(oracle.wt_idx[None, :])[0])
    best_f, best_g = -np.inf, None

    def try_accept(g, f):
        nonlocal best_f, best_g
        if f > best_f:
            best_f, best_g = f, g.copy()

    for r in range(restarts):
        if oracle.n_evals > budget:
            break
        if r == 0:
            g = oracle.wt_idx.copy()
        else:
            g = rng.integers(0, 20, size=L)
            g = oracle.enforce_max_mut(g[None, :], rng)[0]
        f = float(oracle.evaluate(g[None, :])[0])
        try_accept(g, f)
        for sweep in range(24):
            improved = False
            order = rng.permutation(L) if r > 0 else np.arange(L)
            # ---- 单点扫描 ----
            for j in order:
                cand = np.tile(g, (20, 1))
                cand[:, j] = np.arange(20)
                fits = oracle.evaluate(cand)
                nm = (cand != oracle.wt_idx[None, :]).sum(axis=1)
                fits = np.where(nm > max_mut, -np.inf, fits)
                i = int(np.argmax(fits))
                if fits[i] > f + 1e-9:
                    f, g, improved = float(fits[i]), cand[i], True
            # ---- 换位阶段 ----
            mut = np.flatnonzero(g != oracle.wt_idx)
            unm = np.flatnonzero(g == oracle.wt_idx)
            if len(mut) and len(unm):
                for m in mut:
                    cands = np.tile(g, (len(unm), 1))
                    cands[:, m] = oracle.wt_idx[m]
                    # 逐 u 尝试 20 种 AA：批量展开
                    big = np.repeat(cands, 20, axis=0)
                    aa_block = np.tile(np.arange(20), len(unm))
                    u_block = np.repeat(unm, 20)
                    big[np.arange(len(big)), u_block] = aa_block
                    nm = (big != oracle.wt_idx[None, :]).sum(axis=1)
                    fits = oracle.evaluate(big)
                    fits = np.where(nm > max_mut, -np.inf, fits)
                    i = int(np.argmax(fits))
                    if fits[i] > f + 1e-9:
                        f, g, improved = float(fits[i]), big[i], True
            if not improved:
                break
        try_accept(g, f)
    return dict(wt_f=wt_f, ref_f=best_f, n_evals=oracle.n_evals)


# ----------------------------------------------------------------------
def discover_optimizers():
    import engine.wfopt as W
    opts = {"wf": W.WFOptimizer}
    try:
        import engine.rlpolicy as R  # noqa
        opts["dqn"] = R.DQNOptimizer
        opts["bc"] = R.BCOptimizer
    except ImportError:
        pass
    try:
        import engine.rlpolicy_torch as RT  # noqa
        opts["ppo"] = RT.PPOOptimizer
    except ImportError:
        pass
    try:
        import engine.esopt as E  # noqa
        opts["es"] = E.OpenAIESOptimizer
        opts["cem"] = E.CEMOptimizer
    except ImportError:
        pass
    return opts


def make_optimizer(name, cls, oracle, seed, budget, cfg=None):
    return cls(oracle, seed=seed, budget=budget, cfg=cfg or {})


# WF 的先验提议表（仅 PprI 系景观有先验；其余均匀提议）
PROPOSALS = {}


def run_all(oracles, optimizers, budget, seeds, outdir, tag):
    rows, traces = [], {}
    for oname, orc in oracles.items():
        print(f"\n=== oracle {oname} (L={orc.L}, max_mut={orc.max_mut}) ===")
        t0 = time.time()
        ref = greedy_reference(orc)
        print(f"  greedy ref: wt_f={ref['wt_f']:.3f} ref_f={ref['ref_f']:.3f} "
              f"({orc.n_evals} evals, {time.time()-t0:.0f}s)")
        for oname2, cls in optimizers.items():
            fs = []
            for seed in range(seeds):
                # oracle 重建以清零预算（每个 run 独立实例，保证预算隔离）
                orc2 = REBUILD[oname]()
                cfg = {}
                if oname2 == "wf" and oname in PROPOSALS:
                    cfg["proposal_table"] = PROPOSALS[oname]
                if oname2 == "dqn":
                    # train_every 2 → 6：numpy DQN 每步构造 [1060,42] 特征数组，
                    # 4000 evals×5 seed 在 8GB 机器上碎片累积爆内存
                    cfg["train_every"] = 6
                if oname2 == "ppo":
                    # PPO 超参扫描结论（2026-08-31）：lr=1e-3 主导，代理景观 +6%；
                    # ent=0.01 在该预算下不敏感。数学景观 lr=3e-4 略优（0.322 vs 0.304）
                    cfg["lr"] = 1e-3
                    cfg["ent"] = 0.01
                opt = cls(orc2, seed=seed, budget=budget, cfg=cfg)
                t0 = time.time()
                res = opt.run()
                import gc
                gc.collect()          # 释放 numpy 碎片，防跨 seed 累积
                fs.append(res["best_f"])
                traces[f"{oname}|{oname2}|{seed}"] = res["trace"]
                print(f"  {oname2:>4} seed={seed}: best={res['best_f']:.3f} "
                      f"({res['n_evals']} evals, {time.time()-t0:.1f}s)")
            rows.append(dict(oracle=oname, optimizer=oname2, seeds=seeds,
                             budget=budget,
                             best_mean=round(float(np.mean(fs)), 4),
                             best_std=round(float(np.std(fs)), 4),
                             best_max=round(float(max(fs)), 4),
                             wt_f=round(ref["wt_f"], 4),
                             ref_f=round(ref["ref_f"], 4),
                             norm=round(float(np.mean(fs) - ref["wt_f"]) /
                                        max(ref["ref_f"] - ref["wt_f"], 1e-9), 4)))
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(outdir, f"b7_{tag}_results.csv"), index=False)
    with open(os.path.join(outdir, f"b7_{tag}_traces.json"), "w") as f:
        json.dump(traces, f)
    return df


REBUILD = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--oracles", default="all")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--optimizers", default=None,
                    help="逗号分隔优化器子集（如 ppo,wf），默认全部")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(ROOT, "results")
    os.makedirs(outdir, exist_ok=True)
    which = None if args.oracles == "all" else args.oracles.split(",")

    # ---- 构建可重建的 oracle 工厂（每个 run 独立实例，保证预算隔离） ----
    global REBUILD, _ppri_G_global
    orc0, G, _wt, _sites = build_ppri_additive()
    _ppri_G_global = G
    wt53 = orc0.wt_idx
    gb1_proto = GB1PairwiseOracle(max_mut=4) if (
        which is None or "gb1_pairwise" in which or "ppri_cal" in which) else None

    REBUILD["ppri_additive"] = lambda: AdditiveOracle(
        G, wt53, max_mut=12, sites=orc0.sites, name="ppri_additive")
    PROPOSALS["ppri_additive"] = G
    for k in (0, 1, 2, 3):
        REBUILD[f"nk_k{k}"] = (lambda k=k: NKOracle(wt53, K=k, seed=7 + k))
    if gb1_proto is not None:
        REBUILD["gb1_pairwise"] = lambda: GB1PairwiseOracle(max_mut=4)
        REBUILD["ppri_cal"] = lambda: PprICalibratedOracle(
            G, wt53, gb1_proto, seed=11)
        PROPOSALS["ppri_cal"] = G

    # 跨蛋白 DL 代理（ProteinGym 训练，PprI 零样本）—— 本机资产，需模型文件
    try:
        from engine.surrogate_oracle import build_ppri_surrogate
        REBUILD["surrogate_ppri"] = lambda: build_ppri_surrogate()
        print("[ok] surrogate_ppri (ProteinGym DL 代理 v1/one-hot) 可用")
    except Exception as e:  # noqa
        print(f"[warn] surrogate_ppri 不可用: {e}")

    # ESM3 特征版代理（v3）：esm3-sm-open-v1 per-residue emb → PCA → DeepSetV3。
    # 资产在 evo2/esm3/（train_surrogate_v3.py + ppri_surrogate_v3.py + surrogate_dl_v3/）
    try:
        esm3_dir = os.path.join(ROOT, "esm3")
        if esm3_dir not in sys.path:
            sys.path.insert(0, esm3_dir)
        from ppri_surrogate_v3 import PprISurrogateV3

        class SurrogateV3Oracle:
            name = "surrogate_ppri_v3"
            max_mut = 12

            def __init__(self):
                self._o = PprISurrogateV3()
                self.L = self._o.L
                self.wt_idx = self._o.wt_idx
                self.wt_f = self._o.wt_f
                self.ref_f = self._o.ref_f
                self.n_evals = 0
                self.sites = self._o.sites

            def evaluate(self, genos):
                self.n_evals += len(genos)
                return self._o.evaluate_multi(np.asarray(genos, dtype=np.int64))

            def n_mutations(self, geno):
                geno = np.atleast_2d(geno)
                return (geno != self.wt_idx).sum(1)

            def enforce_max_mut(self, geno, rng):
                geno = np.atleast_2d(geno).copy()
                if not self.max_mut:
                    return geno
                n_mut = self.n_mutations(geno)
                for n in np.flatnonzero(n_mut > self.max_mut):
                    ms = np.flatnonzero(geno[n] != self.wt_idx)
                    drop = rng.choice(ms, size=int(n_mut[n] - self.max_mut), replace=False)
                    geno[n, drop] = self.wt_idx[drop]
                return geno

        REBUILD["surrogate_ppri_v3"] = lambda: SurrogateV3Oracle()
        print("[ok] surrogate_ppri_v3 (ESM3 特征代理, held-out 0.407) 可用")
    except Exception as e:  # noqa
        print(f"[warn] surrogate_ppri_v3 不可用: {e}")
    keys = list(REBUILD) if which is None else [k for k in which if k in REBUILD]
    oracles = {k: REBUILD[k]() for k in keys}

    optimizers = discover_optimizers()
    if args.optimizers:
        optimizers = {k: v for k, v in optimizers.items() if k in args.optimizers.split(",")}
    print(f"优化器: {list(optimizers)}  oracle: {keys}  budget={args.budget} seeds={args.seeds}")

    df = run_all(oracles, optimizers, args.budget, args.seeds, outdir, args.tag)
    print("\n=== 汇总 ===")
    print(df.to_string(index=False))
    json.dump(json.loads(df.to_json(orient="records")),
              open(os.path.join(outdir, f"b7_{args.tag}_summary.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\nB7 done -> {outdir}")


if __name__ == "__main__":
    main()
