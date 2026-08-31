#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoTrace v2 — 统一适应度 oracle（b7 三方优化器对比的共享基座）。

设计原则
  1. 所有优化器（WF / RL / ES）只通过 ``evaluate`` 访问景观，预算以
     ``oracle.n_evals`` 计数，保证三方在同一评估预算下对比。
  2. 基因型统一为 ``geno[N, L]``（int，每列一个可变位点的 AA 索引 0..19），
     WT 列由 ``wt_idx`` 给出；"突变成 WT" 等价于未突变。
  3. 景观分四类：
     - AdditiveOracle      — 加性（位点可分离），有闭式最优（平凡性对照）。
     - NKOracle            — 经典 NK 模型，K 控制上位性强度（K=0 可分离）。
     - GB1PairwiseOracle   — Olson 2014 GB1 实测配对上位性（真实数据）。
     - PprICalibratedOracle— PprI 加性 + GB1 实测上位性幅度校准（L=53）。
"""
import os
import re

import numpy as np

from .seqtools import AA20, AA2IDX


class Oracle:
    """oracle 基类。子类实现 _eval(geno)->[N]，并提供 wt_idx / max_mut / name。"""

    name = "base"
    has_known_optimum = False

    def __init__(self, wt_idx, max_mut=12, sites=None):
        self.wt_idx = np.asarray(wt_idx, dtype=int)
        self.L = len(self.wt_idx)
        self.max_mut = int(max_mut) if max_mut else None
        self.sites = np.asarray(sites) if sites is not None else np.arange(self.L)
        self.n_evals = 0

    # ------------------------------------------------------------------
    def _eval(self, geno):
        raise NotImplementedError

    def evaluate(self, geno):
        geno = np.atleast_2d(np.asarray(geno, dtype=int))
        self.n_evals += geno.shape[0]
        return self._eval(geno)

    def eval_one(self, geno):
        return float(self.evaluate(geno[None, :])[0])

    # ------------------------------------------------------------------
    def n_mutations(self, geno):
        return (np.asarray(geno) != self.wt_idx[None, :]).sum(axis=1)

    def enforce_max_mut(self, geno, rng):
        """超出 max_mut 的个体：随机把多余位点回退为 WT。"""
        if not self.max_mut:
            return geno
        n_mut = self.n_mutations(geno)
        for n in np.flatnonzero(n_mut > self.max_mut):
            ms = np.flatnonzero(geno[n] != self.wt_idx)
            drop = rng.choice(ms, size=int(n_mut[n] - self.max_mut), replace=False)
            geno[n, drop] = self.wt_idx[drop]
        return geno

    def known_optimum(self):
        """返回 (fitness, geno) 或 None。仅可分离景观能给出闭式最优。"""
        return None


# ======================================================================
class AdditiveOracle(Oracle):
    """f(geno) = Σ_j G[j, geno_j]。G[L,20] 每位点贡献表（可含稳定性惩罚）。

    全局最优（≤max_mut 突变）= 相对 WT 增益最大的前 max_mut 个 (site,aa)，
    闭式可解 —— 这是"加性景观平凡"的对照。
    """

    def __init__(self, G, wt_idx, max_mut=12, sites=None, name="additive"):
        super().__init__(wt_idx, max_mut=max_mut, sites=sites)
        self.G = np.asarray(G, dtype=float)
        self.name = name
        self.has_known_optimum = True

    def _eval(self, geno):
        return self.G[np.arange(self.L)[None, :], geno].sum(axis=1)

    def known_optimum(self):
        best_aa = self.G.argmax(axis=1)
        gain = self.G[np.arange(self.L), best_aa] - self.G[np.arange(self.L), self.wt_idx]
        order = np.argsort(-gain)
        g = self.wt_idx.copy()
        take = order[: self.max_mut] if self.max_mut else order
        g[take] = best_aa[take]
        return float(self.eval_one(g)), g


# ======================================================================
class NKOracle(Oracle):
    """经典 NK 景观：位点 j 的贡献依赖自身 AA 与 K 个环形邻居。

    K=0 时退化为可分离景观（有闭式最优）；K 越大越崎岖。
    贡献表 per-site [20^(K+1)] float32，K<=3 内存约 35MB/L=53。
    """

    def __init__(self, wt_idx, K=2, seed=0, max_mut=12, name=None):
        super().__init__(wt_idx, max_mut=max_mut)
        self.K = int(K)
        self.name = name or f"nk_k{K}"
        rng = np.random.default_rng(seed)
        self.powers = (20 ** np.arange(self.K + 1)).astype(np.int64)  # 邻居混合进制
        self.tables = rng.random((self.L, 20 ** (self.K + 1))).astype(np.float32)
        # 邻居：环形向后 K 个（标准 NK 用相邻位点）
        self.nbrs = np.array([[(j + d) % self.L for d in range(1, self.K + 1)]
                              for j in range(self.L)], dtype=int) if K else None
        self.has_known_optimum = (K == 0)

    def _eval(self, geno):
        f = np.zeros(geno.shape[0], dtype=np.float64)
        for j in range(self.L):
            if self.K == 0:
                idx = geno[:, j]
            else:
                ctx = geno[:, [j] + list(self.nbrs[j])]          # [N, K+1]
                idx = ctx @ self.powers
            f += self.tables[j, idx]
        return f / self.L

    def known_optimum(self):
        if self.K != 0:
            return None
        g = self.tables.argmax(axis=1).astype(int)
        return float(self.eval_one(g)), g


# ======================================================================
class GB1PairwiseOracle(Oracle):
    """GB1 实测配对上位性 oracle（Olson 2014, MutLandscapes.txt）。

    仅保留"19 个非 WT 突变全部有单突变 Fit 实测"的位点（17 个），
    乘法模型：f = Π 单突变 Fit × Π 配对 eps，eps = pair_fit/(s1*s2)，
    NA 配对取 eps=1。真实数据、真实上位性；无闭式最优。
    """

    MUT_PATH = "A:/claudework/evo_data/raw/flip/gb1/MutLandscapes.txt"

    def __init__(self, max_mut=4, mut_path=None):
        self.mut_path = mut_path or self.MUT_PATH
        singles, pairs, pos2j, wt_idx = self._parse(self.mut_path)
        self.singles = singles            # dict pos -> {aa: fit}
        self.pairs = pairs                # dict (p1,a1,p2,a2) -> eps（已除以单突变积）
        self.pos2j = pos2j
        self.pos_list = sorted(pos2j)
        super().__init__(wt_idx, max_mut=max_mut, sites=np.array(self.pos_list))
        self.name = "gb1_pairwise"
        # 预计算对数表
        self.LS = np.full((self.L, 20), 0.0)          # log 单突变比（WT 列 = 0）
        for p, j in pos2j.items():
            for aa, fit in singles[p].items():
                self.LS[j, AA2IDX[aa]] = np.log(max(fit, 1e-6))
        self.LE = np.zeros((self.L, self.L, 20, 20))  # log eps（缺省 0）
        for (p1, a1, p2, a2), eps in pairs.items():
            j1, j2 = pos2j[p1], pos2j[p2]
            self.LE[j1, j2, AA2IDX[a1], AA2IDX[a2]] = np.log(max(eps, 1e-6))
            self.LE[j2, j1, AA2IDX[a2], AA2IDX[a1]] = np.log(max(eps, 1e-6))
        # 配对项掩码：只有两位点都突变才计入
        tri = np.triu_indices(self.L, 1)

    def _parse(self, path):
        import pandas as pd
        d = pd.read_csv(path, sep="\t")
        bgs = d["Background"].astype(str)
        pat = bgs.str.extract(r"^([A-Z])(\d+)([A-Z])$")
        pat.columns = ["wt", "pos", "mut"]
        valid = pat["pos"].notna()
        d = d.loc[valid].copy()
        for c in pat.columns:
            d[c] = pat[c].values
        d["pos"] = d["pos"].astype(int)
        cnt = d.groupby("pos")["mut"].nunique()
        keep = sorted(int(p) for p in cnt[cnt == 19].index)
        d = d[d["pos"].isin(keep)]
        wt_of = {int(r.pos): r.wt for r in d.itertuples()}
        singles = {}
        for r in d.itertuples():
            singles.setdefault(int(r.pos), {})[r.mut] = float(r.Fit)
        pairs = {}
        pair_cols = [c for c in d.columns if c not in ("Background", "Fit", "wt", "pos", "mut")]
        for r in d.itertuples():
            p1, a1, f1 = int(r.pos), r.mut, float(r.Fit)
            for c in pair_cols:
                m = re.match(r"^([A-Z])(\d+)([A-Z])$", c)
                if not m:
                    continue
                a2, p2 = m.group(3), int(m.group(2))
                v = getattr(r, c)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                if p2 not in singles or a2 not in singles[p2]:
                    continue
                denom = f1 * singles[p2][a2]
                if denom <= 0:
                    continue
                eps = float(v) / denom
                key = (p1, a1, p2, a2) if (p1, p2) < (p2, p1) else (p2, a2, p1, a1)
                pairs.setdefault(key, eps)
        pos2j = {p: j for j, p in enumerate(keep)}
        wt_idx = np.array([AA2IDX[wt_of[p]] for p in keep], dtype=int)
        return singles, pairs, pos2j, wt_idx

    def _eval(self, geno):
        N = geno.shape[0]
        logf = self.LS[np.arange(self.L)[None, :], geno].sum(axis=1)
        jj = np.arange(self.L)
        for i1 in range(self.L):
            for i2 in range(i1 + 1, self.L):
                m1 = geno[:, i1] != self.wt_idx[i1]
                m2 = geno[:, i2] != self.wt_idx[i2]
                act = np.flatnonzero(m1 & m2)
                if len(act) == 0:
                    continue
                logf[act] += self.LE[i1, i2, geno[act, i1], geno[act, i2]]
        return logf


# ======================================================================
class PprICalibratedOracle(Oracle):
    """PprI 加性景观 + GB1 实测上位性幅度校准的配对耦合（L=53）。

    背景：PprI 真实 Boltz 标签仅 36 条独立序列且与加性代理零相关
    （corr(add_fit, sep) = -0.11），不足以支撑经验 oracle；故配对耦合的
    分布（稀疏度 + 幅度）从 GB1 实测 eps 经验分布抽样，加性项保留 PprI
    的 prior×FoldX 景观。属于"合成但校准"的 oracle，须如实标注。
    """

    def __init__(self, G, wt_idx, gb1_oracle, seed=0, max_mut=12,
                 eps_scale=1.0, name="ppri_cal"):
        super().__init__(wt_idx, max_mut=max_mut)
        self.name = name
        self.G = np.asarray(G, dtype=float)
        rng = np.random.default_rng(seed)
        # GB1 经验 log-eps 池（非零项）
        le = gb1_oracle.LE
        tri_i, tri_j = np.triu_indices(gb1_oracle.L, 1)
        pool = []
        for i1, i2 in zip(tri_i, tri_j):
            v = le[i1, i2][np.triu_indices(20, 1)]
            pool.append(v[v != 0])
        pool = np.concatenate(pool) * float(eps_scale)
        # PprI 配对耦合：按 GB1 非零密度稀疏抽样
        self.LE = np.zeros((self.L, self.L, 20, 20))
        nz_frac = len(pool) / (gb1_oracle.L * (gb1_oracle.L - 1) / 2 * 190)
        for i1 in range(self.L):
            for i2 in range(i1 + 1, self.L):
                if rng.random() > nz_frac:
                    continue
                vals = rng.choice(pool, size=190, replace=True)
                k = 0
                for a1 in range(20):
                    for a2 in range(a1 + 1, 20):
                        self.LE[i1, i2, a1, a2] = vals[k]
                        self.LE[i2, i1, a2, a1] = vals[k]
                        k += 1

    def _eval(self, geno):
        f = self.G[np.arange(self.L)[None, :], geno].sum(axis=1)
        for i1 in range(self.L):
            for i2 in range(i1 + 1, self.L):
                m1 = geno[:, i1] != self.wt_idx[i1]
                m2 = geno[:, i2] != self.wt_idx[i2]
                act = np.flatnonzero(m1 & m2)
                if len(act):
                    f[act] += self.LE[i1, i2, geno[act, i1], geno[act, i2]]
        return f


# ======================================================================
def build_ppri_additive():
    """重建 B5 的 PprI 加性景观（prior×reflux×FoldX），返回 (oracle, G, wt_seq)。"""
    import sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    for p in (ROOT, os.path.join(ROOT, "benchmarks"), "A:/claudework/ppri_evo"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import b5_ppri_wave2 as b5  # noqa: 沿用 B5 的景观定义
    from engine.seqtools import read_fasta

    PPRI = "A:/claudework/ppri_evo"
    PDB_OFFSET = 22
    TAU, W_STAB = 2.0, 0.4
    wt = list(read_fasta(os.path.join(PPRI, "inputs", "wt_254.fasta")).values())[0]
    pri, cls, wt_aa = b5.load_ppri_priors()
    sites = sorted(pri.keys())
    site_pos = {s: j for j, s in enumerate(sites)}
    L = len(sites)
    pri_table = np.array([[pri[s].get(a, 1e-3) for a in AA20] for s in sites])
    pri_table /= pri_table.sum(1, keepdims=True)
    wt_idx = np.array([AA2IDX[wt[s]] for s in sites])
    ev = {}
    for name, muts_s, sep, w in b5.WINNERS:
        for pdb, aa in b5.parse_muts_pdb(muts_s):
            if pdb - PDB_OFFSET in site_pos:
                ev.setdefault((site_pos[pdb - PDB_OFFSET], aa), []).append(sep * w)
    ys = np.array([np.mean(v) for v in ev.values()])
    y_ref, scale = float(np.median(ys)), float(ys.std() + 1e-9)
    from engine.reflux import reflux_prior
    pri3 = reflux_prior(pri_table, wt_idx, [[k] for k in ev.keys()],
                        [np.mean(v) for v in ev.values()],
                        alpha=0.35, k0=2.0, y_ref=y_ref, scale=scale)
    ddg, _ = b5.build_ddg_tables(wt, sites, cls, wt_aa)
    pri4 = pri3 * np.exp(-ddg / TAU)
    pri4 /= pri4.sum(1, keepdims=True)
    logp = np.log(np.maximum(pri4, 1e-12))
    G = (logp - logp[np.arange(L), wt_idx][:, None]) - W_STAB * ddg / TAU
    orc = AdditiveOracle(G, wt_idx, max_mut=12, sites=np.array(sites),
                         name="ppri_additive")
    return orc, G, wt, sites
