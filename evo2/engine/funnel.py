#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoTrace v2 — 基准用景观抽象与 campaign 循环（机制 M4/M5 的骨架）。

OracleLandscape：ProteinGym DMS 实测值作为"昂贵层真值"。两个诚实原则：
  1. 标签预算制：只有预算内的查询才返回实测值（模拟湿实验通量）；
  2. 文库约束：提议基因型若不在实测集内，视为"库外"，不消耗预算但记违例
     （对应湿实验中饱和突变库的可合成范围）。

run_campaign：EvoTrace v2 的核心循环——
  廉价核进化 n_gen 代 → propose_elites（多样性批次）→ 预算内查询 oracle →
  贝叶斯回流 prior → （可选）全局上位性层更新 → 重复 R 轮 → 输出最终推荐与日志。
"""
import numpy as np

from .kernel import WFKernel
from .reflux import GlobalEpistasis, reflux_prior
from .seqtools import AA20


def _latent_of(kernel, muts):
    """加性潜变量：当前 sel 下基因型的模型分。"""
    return float(sum(kernel.sel[int(np.searchsorted(kernel.sites, i)), AA20.index(a)]
                     for i, a in muts))


class OracleLandscape:
    """DMS 实测景观：genotype(0-based muts tuple) -> 观测值。"""

    def __init__(self, wt_seq, measured, unknown_value=np.nan):
        """measured: {frozen muts tuple: y}。"""
        self.wt = wt_seq
        self.measured = measured
        self.unknown = unknown_value
        self.n_queries = 0
        self.violations = 0
        self.hit_keys = set()

    def query(self, muts_list):
        """muts_list: list of [(site0, aa), ...]。返回 (y array, measured mask)。"""
        ys, ok = [], []
        for muts in muts_list:
            key = tuple(sorted((int(i), a) for i, a in muts))
            if key in self.measured:
                ys.append(self.measured[key])
                ok.append(True)
                self.n_queries += 1
                self.hit_keys.add(key)
            else:
                ys.append(self.unknown)
                ok.append(False)
                self.violations += 1
        return np.asarray(ys, float), np.asarray(ok, bool)

    @property
    def budget_used(self):
        return self.n_queries


def run_campaign(kernel, oracle, rounds=4, n_gen=12, Ne=400, n_pop=4,
                 batch=24, reflux_alpha=0.35, reflux_k0=2.0, use_epistasis=True,
                 log_prefix="", budget_cap=None):
    """EvoTrace v2 campaign 循环。返回 (final_recs, history)。

    final_recs: [(fitness_proxy, muts, y_or_nan)] 按代理分排序的最终推荐。
    """
    gep = GlobalEpistasis()
    history = []
    labeled_muts_all, labeled_y_all, labeled_latent_all = [], [], []
    for r in range(rounds):
        stats, pops = kernel.run(n_pop=n_pop, n_gen=n_gen, Ne=Ne)
        for s in stats:
            s["round"] = r
        history.extend(stats)
        cands = kernel.propose_elites(pops, top_k=max(batch // 3, 4), diversity=8)
        muts_batch = [kernel.geno_to_muts(g) for _, g in cands]
        ys, ok = oracle.query(muts_batch)
        # 只用已测的标签回流（预算内）
        got = [m for m, o in zip(muts_batch, ok) if o]
        got_y = ys[ok]
        if len(got):
            # 潜变量统一用当前 sel 重算（回流会改 sel，跨轮必须一致口径）
            latent = [_latent_of(kernel, m) for m in labeled_muts_all + got]
            if use_epistasis:
                gep.update(latent, labeled_y_all + list(got_y))
            labeled_muts_all += [tuple(sorted((int(i), a) for i, a in m)) for m in got]
            labeled_y_all += list(got_y)
            labeled_latent_all = latent
            kernel.prior_table = reflux_prior(
                kernel.prior_table, kernel.wt_idx,
                [[(int(np.searchsorted(kernel.sites, i)), a) for i, a in m] for m in got],
                got_y, alpha=reflux_alpha, k0=reflux_k0)
            kernel.set_prior_table(kernel.prior_table)
        history.append(dict(round=r, event="reflux", n_new_labels=int(ok.sum()),
                            budget_used=oracle.budget_used,
                            best_y=float(np.nanmax(ys)) if len(ys) else np.nan))
        if budget_cap is not None and oracle.budget_used >= budget_cap:
            break
    # 最终推荐池：全部已获标签 (潜变量=加性模型分, muts, 实测 y)。
    # 排序口径由 benchmark 决定：best_found 按 y，pred_topK 按模型分（潜变量或 g(潜)）。
    recs = []
    seen = set()
    for m, la, y in zip(labeled_muts_all, labeled_latent_all, labeled_y_all):
        if m in seen:
            continue
        seen.add(m)
        recs.append((float(la), m, float(y)))
    return recs, history, gep
