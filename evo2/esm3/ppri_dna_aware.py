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


# ---- Boltz-2 标签校准的经验偏好表（2026-08-31, 240 models, 六候选×24nt 靶/非靶）----
# 实测结论（见 results/gate_calibration.csv）:
#   读头 F88 : K(双锁 .85/.45/.35) >> F=WT(.20) >> R(.00, 反向设计 F88R 摧毁读头几何)
#   Y217     : Y 保留芳香为佳；R 与反向设计共现且双锁为 0
#   M255     : M(天然) > I(s13_c1, .35) >> K(.00)
#   锚点     : 降低 Patch1 正电荷(R85G/R207K) 显著提升判别 Δiface — 与 v1 的"必须为 R"相反
# ⚠ 混杂警告: n=6 设计且位点突变共现(F88R+Y217R+M255K 总是一起出现),
#   下表为 in-sample 拟合(Spearman v2 vs dual_S1 = +0.824), 泛化性待去混杂扫描验证。
READ_PREF = {'K': 1.00, 'W': 0.50, 'Y': 0.55, 'F': 0.30, 'Q': 0.20, 'A': -0.20, 'R': -1.00}
Y217_PREF = {'Y': 1.00, 'F': 0.50, 'W': 0.50, 'A': -0.20, 'K': -0.40, 'R': -0.60}
M255_PREF = {'M': 1.00, 'I': 0.60, 'L': 0.50, 'A': -0.20, 'R': -0.80, 'K': -1.00}
ANCHOR_CHARGE = {'R': 1.0, 'K': 0.70, 'H': 0.30}


class DnaAwareLandscape:
    """组合景观：ESM3 稳定性 z + 机制门控的靶标匹配项。

    gate_version:
        "v1" —— 解析式启发（芳香读头 / 正电双锁 / 锚点必 R）。
                已由 Boltz 证伪: Spearman vs 实测双锁率 = −0.588（反相关），仅作对照保留。
        "v2" —— Boltz-2 标签校准（默认）。Spearman vs 双锁率 +0.824 / vs Δiface +0.928。
    """

    def __init__(self, base_oracle,
                 dna_on="TCATGAGCAGTTTTTTGTTTTTTT",
                 dna_off="TTGCTATTTTTTATTGCTTTGAGT",
                 target_read_base='G', off_read_base='T',
                 w_base=0.5, w_gate=0.5, gate_version="v2"):
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
        self.gate_version = gate_version
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
        """按 gate_version 分派。v2 为 Boltz 校准版本（默认）。"""
        if self.gate_version == "v1":
            return self._gate_v1(geno)
        return self._gate_v2(geno)

    def _gate_v1(self, geno):
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

    def _gate_v2(self, geno):
        """Boltz-2 校准门控:

            gate = 0.40*读头(88) + 0.20*Y217 + 0.20*M255 − 0.20*锚点平均电荷

        末项为"非特异磷酸锚定惩罚"——降低 Patch1 正电荷可显著减少非靶结合
        （实测 s13_c1 R85G / TrackF_r1 R207K 的 Δiface 分别为 +10.5 / +12.9，
          均远高于 WT 的 +3.5）。
        """
        geno = np.asarray(geno, dtype=np.int64)

        def aa_at(pdb):
            c = self.seqidx_to_col.get(pdb - 22)
            return AA[int(geno[c])] if c is not None else None

        read = READ_PREF.get(aa_at(READHEAD_PDB), 0.0)
        y217 = Y217_PREF.get(aa_at(217), 0.0)
        m255 = M255_PREF.get(aa_at(255), 0.0)
        charge = sum(ANCHOR_CHARGE.get(aa_at(p), 0.0) for p in ANCHOR_PDB) / float(len(ANCHOR_PDB))
        return 0.40 * read + 0.20 * y217 + 0.20 * m255 - 0.20 * charge

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
