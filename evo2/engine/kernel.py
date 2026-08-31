#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoTrace v2 — 向量化 Wright-Fisher 进化内核（机制 M2/M3/M5）。

两种突变提议模式：
  1. 自由模式（measured_keys=None）：Poisson(λ) 突变 + 先验加权 AA 提议 +
     锚位掩码（M2）。适用：GB1 完整景观（4 位点全测）、PprI（先验定义突变空间）。
  2. 文库行走模式（measured_keys 给定）：每代每体 Poisson(λ) 次"一步编辑"尝试，
     候选 = 当前基因型在某位点上换成该位点实测等位库中的一个 AA（含回 WT），
     仅当新基因型在实测集合内才接受（library-constrained adaptive walk，
     DMS-replay 文献标准协议）。适用：avGFP 稀疏局部景观。
两种模式的选择/漂变完全一致：softmax(fitness/T) + Wright-Fisher 多项抽样。
"""
import numpy as np

from .seqtools import AA20, AA2IDX


class WFKernel:
    """Wright-Fisher 内核。sites: 可变位点 0-based 列表；priors: {site: {aa: p}}。"""

    def __init__(self, wt_seq, sites, priors, config, seed=0,
                 anchor_sites=None, measured_keys=None, proposal="prior",
                 evoprior=None, alpha=0.5):
        self.wt = wt_seq
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.sites = np.asarray(sorted(sites), dtype=int)
        self.L = len(self.sites)
        self.n_aa = 20
        self.proposal_mode = proposal
        # ---- EvoPrior 融合（v3 草案）：p_final = α·p_chem + (1−α)·p_evo ----
        # evoprior: [L_full, 20] 进化先验（None 表示不融合，退化为 v2 纯化学先验）
        self.evoprior = evoprior
        self.alpha = float(alpha)
        self.wt_idx = np.array([AA2IDX[wt_seq[i]] for i in self.sites], dtype=int)
        self.prior_table = self._build_prior_table(priors)
        self._refresh_sel()
        # 锚位掩码（M2）
        self.mask = np.ones((self.L, self.n_aa), dtype=bool)
        if anchor_sites:
            for site, allowed in anchor_sites.items():
                j = int(np.searchsorted(self.sites, site))
                if j < self.L and self.sites[j] == site:
                    m = np.zeros(self.n_aa, dtype=bool)
                    for aa in allowed:
                        if aa in AA2IDX:
                            m[AA2IDX[aa]] = True
                    if m.sum() > 0:
                        self.mask[j] = m
        # 稳定性（M3）
        self.ddg = np.zeros((self.L, self.n_aa), dtype=float)
        self.w_stab = float(config.get("w_stab", 0.0))
        self.tau = float(config.get("tau_stab", 2.0))
        # 动力学
        self.lam = float(config["mutations_per_genome_per_gen"]["lambda"])
        self.n_mut_max = int(config.get("n_mut_max", 12))
        self.proposal_temp = float(config.get("proposal_temp", 2.0))
        self.T = float(config.get("T", 0.6))
        self._gen = 0
        self.events = []
        # ---- 文库行走模式（measured_keys: set of tuple((site_global, aa_char),...))
        self.measured = None
        if measured_keys is not None:
            self._init_measured(measured_keys)

    # ------------------------------------------------------------------
    def _build_prior_table(self, priors):
        tab = np.full((self.L, self.n_aa), 0.002)
        for j, site in enumerate(self.sites):
            for aa, p in priors.get(int(site), {}).items():
                if aa in AA2IDX:
                    tab[j, AA2IDX[aa]] = max(float(p), 1e-3)
            if tab[j, self.wt_idx[j]] < 1e-3:
                tab[j, self.wt_idx[j]] = 1e-3
        tab /= tab.sum(axis=1, keepdims=True)
        # ---- EvoPrior 融合（v3 草案）：仅在调用方显式传入 evoprior 时生效 ----
        if self.evoprior is not None:
            evo = np.asarray(self.evoprior, float)[np.asarray(self.sites)]
            evo = np.clip(evo, 1e-6, None)
            evo /= evo.sum(axis=1, keepdims=True)
            fused = self.alpha * tab + (1.0 - self.alpha) * evo
            fused = np.clip(fused, 1e-6, None)
            tab = fused / fused.sum(axis=1, keepdims=True)
        return tab

    def _refresh_sel(self):
        self.sel = (np.log(np.maximum(self.prior_table, 1e-4))
                    - np.log(np.maximum(self.prior_table[np.arange(self.L), self.wt_idx], 1e-4))[:, None])
        if not hasattr(self, "sel0"):
            self.sel0 = self.sel.copy()  # 初始先验选择系数（先验信任度特征基准）

    def set_prior_table(self, prior_table):
        """回流更新（M4）：先验表 -> 选择系数重算。"""
        self.prior_table = prior_table
        self._refresh_sel()

    def set_ddg(self, ddg_table, tau=None, w_stab=None):
        self.ddg = ddg_table
        if tau is not None:
            self.tau = float(tau)
        if w_stab is not None:
            self.w_stab = float(w_stab)

    # ------------------------------------------------------------------
    def _init_measured(self, measured_keys):
        site_pos = {int(s): j for j, s in enumerate(self.sites)}
        keys, pool = set(), {j: set() for j in range(self.L)}
        for key in measured_keys:
            loc = []
            for i, a in key:
                j = site_pos.get(int(i))
                if j is None:
                    loc = None
                    break
                loc.append((j, AA2IDX[a]))
                pool[j].add(AA2IDX[a])
            if loc:
                keys.add(frozenset(loc))
        self.measured = keys
        self.pool = {j: sorted(aa for aa in pool[j] if aa != self.wt_idx[j])
                     for j in range(self.L)}
        self.pool_sites = [j for j in range(self.L) if self.pool[j]]

    def _proposal_weights(self):
        """[L, 20] 提议权重（先验^temp 或均匀），行走模式的位点等位库掩码。"""
        if self.proposal_mode == "prior":
            W = np.maximum(self.prior_table, 1e-6) ** self.proposal_temp
        else:
            W = np.ones((self.L, self.n_aa))
        M = np.zeros((self.L, self.n_aa), dtype=bool)
        for j in self.pool_sites:
            M[j, self.pool[j]] = True
            M[j, self.wt_idx[j]] = True  # 回 WT 亦是合法编辑
        return W * M

    # ------------------------------------------------------------------
    def _fitness(self, geno):
        f = self.sel[np.arange(self.L), geno.astype(int)].sum(axis=1)
        if self.w_stab > 0:
            f -= self.w_stab * self.ddg[np.arange(self.L), geno.astype(int)].sum(axis=1) / self.tau
        return f

    # ------------------------------------------------------------------
    def _draw_mutations(self, geno):
        """自由模式：Poisson(λ) 突变 + 先验加权提议 + 锚位掩码。"""
        N = geno.shape[0]
        k = self.rng.poisson(self.lam, size=N)
        max_k = int(k.max()) if len(k) else 0
        w = np.exp(self.sel * self.proposal_temp) * self.mask
        w /= w.sum(axis=1, keepdims=True)
        cw = np.cumsum(w, axis=1)
        cw[:, -1] = 1.0
        for step in range(max_k):
            active = k > step
            n_act = int(active.sum())
            if n_act == 0:
                break
            idx_rows = np.flatnonzero(active)
            site_j = self.rng.integers(0, self.L, size=n_act)
            u = self.rng.random((n_act, 1))
            aa = (cw[site_j] < u).sum(axis=1).clip(0, self.n_aa - 1)
            geno[idx_rows, site_j] = aa
        return geno

    def _draw_walk(self, geno):
        """文库行走模式：一步编辑 + 实测集合接受-拒绝。"""
        N = geno.shape[0]
        k_att = self.rng.poisson(self.lam, size=N)
        W = self._proposal_weights()
        # 每位点选项 = [wt] + pool[j]，权重归一
        opts_w = np.zeros((self.L, 21))  # 第 0 列 = wt，其余 = pool 顺序
        opts_aa = np.zeros((self.L, 21), dtype=int)
        for j in self.pool_sites:
            aas = [int(self.wt_idx[j])] + list(self.pool[j])
            opts_aa[j, :len(aas)] = aas
            opts_w[j, :len(aas)] = W[j, aas]
            s = opts_w[j, :len(aas)].sum()
            opts_w[j] = 0
            opts_w[j, :len(aas)] = (W[j, aas] / s if s > 0 else np.ones(len(aas)) / len(aas))
        pool_arr = np.array(self.pool_sites, int)
        for n in np.flatnonzero(k_att > 0):
            cur = geno[n]
            key = frozenset((int(j), int(cur[j])) for j in np.flatnonzero(cur != self.wt_idx))
            n_mut = len(key)
            for _ in range(k_att[n]):
                if n_mut >= self.n_mut_max:
                    break
                j = int(pool_arr[self.rng.integers(len(pool_arr))])
                aas = opts_aa[j]
                ws = opts_w[j]
                nn = int((ws > 0).sum())
                if nn == 0:
                    continue
                pick = self.rng.choice(nn, p=ws[:nn] / ws[:nn].sum())
                a2 = int(aas[pick])
                cur_aa = int(cur[j])
                if a2 == cur_aa:
                    continue
                new_key = set(key)
                if cur_aa != int(self.wt_idx[j]):
                    new_key.discard((j, cur_aa))
                if a2 != int(self.wt_idx[j]):
                    new_key.add((j, a2))
                new_key = frozenset(new_key)
                if new_key in self.measured:
                    key = new_key
                    cur[j] = a2
                    n_mut = len(key)
            geno[n] = cur
        return geno

    def _enforce_load(self, geno):
        """突变负荷上限：超出者随机回退多余位点（保留骨架，非整行重置）。"""
        n_mut = (geno != self.wt_idx[None, :]).sum(axis=1)
        excess = n_mut - self.n_mut_max
        for n in np.flatnonzero(excess > 0):
            mut_sites = np.flatnonzero(geno[n] != self.wt_idx)
            drop = self.rng.choice(mut_sites, size=int(excess[n]), replace=False)
            geno[n, drop] = self.wt_idx[drop]
        return geno

    # ------------------------------------------------------------------
    def run(self, n_pop=4, n_gen=15, Ne=500, founder=None, record_events=False):
        stats = []
        if founder is None:
            founder = self.wt_idx[None, :]
        founder = np.atleast_2d(np.asarray(founder, dtype=np.int8))
        pops = np.tile(founder[None, :, :], (n_pop, Ne, 1))
        walk = self.measured is not None
        for p in range(n_pop):
            geno = pops[p]
            prev_fixed = None
            for g in range(self._gen, self._gen + n_gen):
                fits = self._fitness(geno)
                mx = fits.max()
                ws = np.exp((fits - mx) / self.T)
                ws /= ws.sum()
                parents = self.rng.choice(Ne, size=Ne, p=ws)
                children = geno[parents].copy()
                children = self._draw_walk(children) if walk else self._draw_mutations(children)
                if not walk:
                    children = self._enforce_load(children)
                geno = children
                fits_new = self._fitness(geno)
                uniq = np.unique(geno, axis=0)
                row = dict(pop=p, gen=g, best=round(float(fits_new.max()), 4),
                           mean=round(float(fits_new.mean()), 4),
                           unique=int(len(uniq)))
                stats.append(row)
                if record_events:
                    fixed_sites = ((geno == geno[0:1]).all(axis=0)) & (geno[0] != self.wt_idx)
                    new_fixed = [(int(self.sites[j]), AA20[int(geno[0, j])])
                                 for j in np.flatnonzero(fixed_sites)]
                    if prev_fixed is not None and len(new_fixed) > len(prev_fixed):
                        self.events.append((g, p, "fixation", new_fixed[len(prev_fixed):]))
                    prev_fixed = new_fixed
            pops[p] = geno
        self._gen += n_gen
        return stats, pops

    # ------------------------------------------------------------------
    def propose_elites(self, pops, top_k=6, diversity=8, n_return=None):
        """按适应度降序返回去重基因型；diversity>0 时施加最小 Hamming 筛选（向量化）。"""
        all_g = pops.reshape(-1, self.L)
        fits = self._fitness(all_g)
        uniq, first = np.unique(all_g, axis=0, return_index=True)
        uf = fits[first]
        order = np.argsort(-uf)
        uniq, uf = uniq[order], uf[order]
        cap = min(n_return if n_return is not None else top_k * 3, len(uniq))
        chosen, chosen_f = [], []
        for g, f in zip(uniq, uf):
            if diversity > 1 and chosen:
                d = (np.array(chosen) != g).sum(axis=1).min()
                if d < diversity:
                    continue
            chosen.append(g)
            chosen_f.append(f)
            if len(chosen) >= cap:
                break
        return [(float(f), g.copy()) for g, f in zip(chosen, chosen_f)]

    def geno_to_muts(self, geno):
        diff = np.flatnonzero(geno != self.wt_idx)
        return [(int(self.sites[j]), AA20[int(geno[j])]) for j in diff]
