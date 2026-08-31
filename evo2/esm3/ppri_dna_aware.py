# -*- coding: utf-8 -*-
"""DNA 条件门控景观（DnaAwareLandscape）—— EvoTrace PprI 工具缺失能力的补丁。

背景（v1.1 的硬伤）:
    surrogate_ppri_v3 是零样本、与 DNA 序列无关的通用适应度 z-score 预测器。
    对它做任何"朝靶标优化"都会把设计突变洗回 WT（见 evolve_six_wetlab 结果），
    因为景观里根本没有 DNA 维度。

修法:
    组合景观 fitness = w_base * z(ESM3 稳定性/表达) + w_gate * gate(靶标匹配)。
    gate 由 PprI 已知机制编码（数据驱动，源自 priors.csv 的 dominant_class）:
      - 读头(F88 系): 对靶标 readhead 碱基(G)的堆叠/氢键偏好 − 对非靶(T)偏好 = 特异性
      - 双锁(R253/Y217/M255): 正电/极性残基保留或增强 DNA 接触
      - 锚点(R85/R207/R267): 必须为 R 以捕获 ssDNA 磷酸骨架，改掉则惩罚
    这样景观具备 DNA 维度，演化会保留/强化靶标匹配突变，而非洗回 WT。

接口:
    兼容搜索内核 .L .wt_idx .sites .evaluate(genos) .max_mut .enforce_max_mut .n_mutations
    权重 w_base/w_gate 与靶标序列可配，后续用 Boltz 判读校准。
"""
import os
import csv
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AROMATIC = {4, 19, 18}   # F Y W   —— π-堆叠偏好 G
POSITIVE = {8, 14}       # K R     —— 盐桥锚定磷酸骨架 / 双锁
POLAR = {2, 3, 11, 13, 15, 16, 6}  # D E N Q S T H
NEGATIVE = {2, 3}        # D E     —— 排斥磷酸

# PprI 机制热点（pdb 残基号；seq_idx = pdb - 22，已与 priors.csv 校验）
READHEAD_PDB = 88          # F88 系读头，识别靶标 readhead 碱基
LOCK_PDB = (253, 217, 255) # R253 / Y217 / M255 双锁（接触 DNA nt23）
ANCHOR_PDB = (85, 207, 267)  # Patch1 锚点（R85/R207/R267 捕获骨架）


def _match_base(aa_idx, base):
    """残基对 DNA 碱基的化学匹配度（读头特异性项用）。"""
    if aa_idx in AROMATIC:
        return 1.0 if base == 'G' else 0.3
    if aa_idx in POLAR:
        return 0.4 if base == 'G' else 0.15
    if aa_idx in NEGATIVE:
        return -0.3
    return 0.0


def _lock_score(aa_idx):
    """双锁残基：正电/极性保留或增强接触 (+1)，负电排斥 (-1)，其他 0。"""
    if aa_idx in POSITIVE or aa_idx in POLAR:
        return 1.0
    if aa_idx in NEGATIVE:
        return -1.0
    return 0.0


class DnaAwareLandscape:
    """组合景观：ESM3 稳定性 z + 机制门控的靶标匹配项。"""

    def __init__(self, base_oracle,
                 dna_on="TCATGAGCAGTTTTTTGTTTTTTT",
                 dna_off="TTGCTATTTTTTATTGCTTTGAGT",
                 target_read_base='G', off_read_base='T',
                 w_base=0.5, w_gate=0.5):
        self.base = base_oracle
        self.L = base_oracle.L
        self.wt_idx = base_oracle.wt_idx
        self.sites = base_oracle.sites          # (53,) seq_idx 0-based
        self.dna_on = dna_on
        self.dna_off = dna_off
        self.target_read_base = target_read_base
        self.off_read_base = off_read_base
        self.w_base = w_base
        self.w_gate = w_gate
        self.max_mut = getattr(base_oracle, 'max_mut', 12)

        # seq_idx -> 列索引
        self.seqidx_to_col = {int(s): i for i, s in enumerate(self.sites)}
        # 热点列（pdb - 22 映射，缺失即冻结）
        self._read_col = self.seqidx_to_col.get(READHEAD_PDB - 22)
        self._lock_cols = [c for c in (self.seqidx_to_col.get(p - 22)
                                       for p in LOCK_PDB) if c is not None]
        self._anchor_cols = [c for c in (self.seqidx_to_col.get(p - 22)
                                         for p in ANCHOR_PDB) if c is not None]

    # ---- 机制门控 ----
    def _gate(self, geno):
        geno = np.asarray(geno, dtype=np.int64)
        # 读头特异性：对靶标 G 的偏好 − 对非靶 T 的偏好
        read = 0.0
        if self._read_col is not None:
            aa = int(geno[self._read_col])
            read = (_match_base(aa, self.target_read_base)
                    - _match_base(aa, self.off_read_base))
        # 双锁：正电/极性保留
        lock = (np.mean([_lock_score(int(geno[c])) for c in self._lock_cols])
                if self._lock_cols else 0.0)
        # 锚点：必须为 R，否则惩罚
        anchor = (np.mean([0.0 if int(geno[c]) == 14 else -1.0
                           for c in self._anchor_cols])
                  if self._anchor_cols else 0.0)
        return 0.4 * read + 0.3 * lock + 0.3 * anchor

    # ---- 组合适应度 ----
    def evaluate(self, genos):
        genos = np.asarray(genos, dtype=np.int64)
        z = np.asarray(self.base.evaluate(genos), dtype=np.float64)
        gates = np.array([self._gate(g) for g in genos])
        return self.w_base * z + self.w_gate * gates

    # ---- 搜索内核兼容接口 ----
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


# 供 b7 注册的工厂（默认 24nt 靶标）
def make_dna_aware(base_oracle=None, **kw):
    if base_oracle is None:
        from ppri_surrogate_v3 import PprISurrogateV3
        base_oracle = PprISurrogateV3()
    return DnaAwareLandscape(base_oracle, **kw)
