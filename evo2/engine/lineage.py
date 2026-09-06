# -*- coding: utf-8 -*-
"""谱系 / 事件时间线 / 叙事报告（v1 方案 §1.1 用户故事 2/4，路线图 R3 起步）。

LineageRecorder 经 events.run_with_events 的 observer 挂入，逐代记录：
  - 基因型首次出现代（first_appearance）与逐代频率
  - 突变固定事件（等位在种群内频率 ≥ fix_thresh）
  - 唯一基因型计数（缓存命中率代理: hit_rate = 1 - unique/total，v1 §3.2 缓存论证）

谱系重建（v0，无需 kernel 记录父指针）: 每个新基因型的父 = 此前出现过的、
与它编辑距离为 1 的基因型中频率最高者（单步编辑祖先，与突变提议算子一致）。
输出: 边表 CSV（from_generation,to_generation,child_hash,parent_hash）、
时间线 JSON、Newick（主干谱系）、Markdown 叙事报告（含三情景年轴换算，v1 §3.5）。
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # 包式导入（from engine.lineage）与独立导入（engine 目录在 sys.path）皆可
    from .seqtools import AA20
except ImportError:  # pragma: no cover
    from seqtools import AA20

__all__ = ["LineageRecorder"]


def _gkey(geno_row: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(geno_row, dtype=np.int8).tobytes()).hexdigest()[:16]


class LineageRecorder:
    """observer(gen, pops, stats) 协议的谱系记录器。

    Parameters
    ----------
    sites: 可变位点全局编号（kernel.sites），用于把 int8 基因型译回 (site, aa) 突变。
    """

    def __init__(self, sites, fix_thresh: float = 0.9):
        self.sites = list(map(int, sites))
        self.fix_thresh = fix_thresh
        self._seen: Dict[str, int] = {}              # hash → 首次出现代
        self._freq: Dict[int, Counter] = {}          # gen → {hash: count}
        self._genos: Dict[str, np.ndarray] = {}      # hash → int8 行（重建谱系用）
        self._wt: Optional[np.ndarray] = None
        self.total_inds = 0
        self.total_unique_events = 0
        self.fixations: List[dict] = []              # {gen, pop, muts}
        self._prev_alleles: Dict[Tuple[int, int], Dict[int, set]] = {}

    # ---------------------------------------------------------------- observer
    def observe(self, gen: int, pops: np.ndarray, stats) -> None:
        n_pop, Ne, L = pops.shape
        if self._wt is None:
            self._wt = None  # 由调用方 set_wt 提供; 无则固定判定退化为一致性判据
        cnt: Counter = Counter()
        for p in range(n_pop):
            pop = pops[p]
            keys = [_gkey(pop[i]) for i in range(Ne)]
            cnt.update(keys)
            self._record_fixations(gen, p, pop)
            for i in range(Ne):
                k = keys[i]
                if k not in self._seen:
                    self._seen[k] = gen
                    self._genos[k] = pop[i].copy()
                    self.total_unique_events += 1
        self._freq[gen] = cnt
        self.total_inds += n_pop * Ne

    def set_wt(self, wt_idx: np.ndarray) -> None:
        self._wt = np.asarray(wt_idx, dtype=np.int8)

    def _record_fixations(self, gen: int, pop_id: int, pop: np.ndarray) -> None:
        for j, site in enumerate(self.sites):
            col = pop[:, j]
            vals, counts = np.unique(col, return_counts=True)
            for v, c in zip(vals, counts):
                if c / len(col) >= self.fix_thresh and self._wt is not None \
                        and int(v) != int(self._wt[j]):
                    key = (pop_id, site, int(v))
                    if key not in self._prev_alleles or gen not in self._prev_alleles[key]:
                        self.fixations.append(dict(
                            gen=gen, pop=pop_id, site=site,
                            aa=AA20[int(v)], freq=round(float(c) / len(col), 3)))
                        self._prev_alleles.setdefault(key, set()).add(gen)

    # ---------------------------------------------------------------- 派生量
    @property
    def cache_hit_rate(self) -> float:
        """重复评估占比（v1 §3.2: 命中缓存的世代步比例）。"""
        if self.total_inds == 0:
            return 0.0
        return 1.0 - self.total_unique_events / self.total_inds

    def unique_curve(self) -> List[Tuple[int, int]]:
        return sorted((g, sum(c.values())) for g, c in self._freq.items())

    def build_lineage(self) -> List[Tuple[int, str, Optional[str]]]:
        """边表: (首次出现代, child_hash, parent_hash)。父 = 此前出现的编辑距离 1
        基因型中（同代频率最高者）；无单步祖先（如迁移注入）则 parent=None。"""
        order = sorted(self._seen.items(), key=lambda kv: kv[1])
        known: List[Tuple[str, np.ndarray, int]] = []   # (hash, geno, first_gen)
        edges = []
        for h, g in order:
            geno = self._genos[h]
            parent = None
            best = -1
            for h2, g2, g2_gen in known:
                if g2_gen >= g:
                    continue
                d = int((g2 != geno).sum())
                if d == 1:
                    freq = self._freq.get(g2_gen, Counter()).get(h2, 0)
                    if freq > best:
                        best, parent = freq, h2
            edges.append((g, h, parent))
            known.append((h, geno, g))
        return edges

    # ---------------------------------------------------------------- 导出
    def to_newick(self, top_k: int = 12) -> str:
        """主干谱系的简化 Newick（按首次出现代排列的链式树, 叶标 = 突变串）。"""
        edges = self.build_lineage()
        edges.sort(key=lambda e: e[0])
        label = {}
        for g, h, parent in edges[:top_k]:
            muts = self.muts_of(h)
            label[h] = "_".join(f"{s}{a}" for s, a in muts) or "WT"
        seq = ["("]
        items = [label[h] for g, h, p in edges[:top_k] if h in label]
        return "(" + ",".join(items) + ");"

    def muts_of(self, h: str) -> List[Tuple[int, str]]:
        if self._wt is None or h not in self._genos:
            return []
        diff = np.flatnonzero(self._genos[h] != self._wt)
        return [(self.sites[int(j)], AA20[int(self._genos[h][j])]) for j in diff]

    def timeline(self) -> List[dict]:
        return sorted(self.fixations, key=lambda d: d["gen"])

    def narrative_report(self, run_meta: Optional[dict] = None,
                         generations_per_year: float = 300.0) -> str:
        """Markdown 叙事报告（v1 §1.1 用户故事 4）：固定事件、平行性、年轴换算。"""
        lines = ["# 进化轨迹叙事报告", ""]
        if run_meta:
            lines += [f"- {k}: {v}" for k, v in run_meta.items()]
        lines += ["",
                  f"- 总个体评估: {self.total_inds}, 唯一基因型: {self.total_unique_events}, "
                  f"缓存命中率(v1 §3.2 口径): {self.cache_hit_rate:.4f}",
                  ""]
        tl = self.timeline()
        lines += [f"## 突变固定事件时间线（频率 ≥ {self.fix_thresh}）", ""]
        for d in tl[:60]:
            t_year = d["gen"] / generations_per_year
            lines.append(f"- 代 {d['gen']}（≈ {t_year:.1f} 年）: 群体 {d['pop']} "
                         f"位点 {d['site']} 固定 {d['aa']}（频率 {d['freq']}）")
        # 平行进化: 不同群体独立固定同一 (site, aa)
        par = Counter((d["site"], d["aa"]) for d in tl)
        repeated = {k: v for k, v in par.items() if v > 1}
        lines += ["", "## 平行进化（同一突变在 ≥2 个独立群体固定）", ""]
        if repeated:
            for (site, aa), v in sorted(repeated.items(), key=lambda kv: -kv[1]):
                lines.append(f"- 位点 {site} → {aa}: {v} 个群体独立固定")
        else:
            lines.append("- 无（本时段内未观察到跨群体重复固定）")
        lines += ["",
                  "## 年轴三情景换算（v1 §3.5）", "",
                  "| 假设 | 代/年 | 总年代数 |",
                  "|---|---|---|"]
        max_gen = max((d["gen"] for d in tl), default=0)
        for gpy, name in ((6.6, "寡营养(LTEE)"), (100, "土壤中位"), (300, "富营养")):
            lines.append(f"| {name} | {gpy} | {max_gen / gpy:.1f} |")
        return "\n".join(lines) + "\n"

    def save(self, out_prefix: str) -> Dict[str, str]:
        """落盘三件套: <prefix>_edges.csv / _timeline.json / _report.md。"""
        edges = self.build_lineage()
        with open(out_prefix + "_edges.csv", "w", encoding="utf-8") as f:
            f.write("first_gen,child,parent\n")
            for g, h, p in edges:
                f.write(f"{g},{h},{p or ''}\n")
        with open(out_prefix + "_timeline.json", "w", encoding="utf-8") as f:
            json.dump(dict(fixations=self.fixations,
                           total_inds=self.total_inds,
                           unique=self.total_unique_events,
                           cache_hit_rate=self.cache_hit_rate), f, indent=1)
        with open(out_prefix + "_report.md", "w", encoding="utf-8") as f:
            f.write(self.narrative_report())
        return {k: out_prefix + s for k, s in
                (("edges", "_edges.csv"), ("timeline", "_timeline.json"),
                 ("report", "_report.md"))}
