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
        self._total_freq: Counter = Counter()        # hash → 累计个体数（父选择用）
        self._genos: Dict[str, np.ndarray] = {}      # hash → int8 行（重建谱系用）
        self._wt: Optional[np.ndarray] = None
        self.total_inds = 0
        self.total_unique_events = 0
        self.fixations: List[dict] = []              # {gen, pop, site, aa, freq}
        self._fixed_state: Dict[Tuple[int, int], Optional[int]] = {}
        self.EXIT_THRESH = 0.5                       # 固定态丢失迟滞线

    # ---------------------------------------------------------------- observer
    def observe(self, gen: int, pops: np.ndarray, stats=None) -> None:
        n_pop, Ne, L = pops.shape
        cnt: Counter = Counter()
        for p in range(n_pop):
            self._record_fixations(gen, p, pops[p])
            uniq, counts = np.unique(pops[p], axis=0, return_counts=True)
            keys = [_gkey(r) for r in uniq]
            cnt.update(dict(zip(keys, map(int, counts))))
            for k, row in zip(keys, uniq):
                if k not in self._seen:
                    self._seen[k] = gen
                    self._genos[k] = row.copy()
                    self.total_unique_events += 1
            self.total_inds += Ne
        self._freq[gen] = cnt
        self._total_freq.update(cnt)

    def set_wt(self, wt_idx: np.ndarray) -> None:
        self._wt = np.asarray(wt_idx, dtype=np.int8)

    def _record_fixations(self, gen: int, pop_id: int, pop: np.ndarray) -> None:
        """带迟滞的固定事件记录（防阈值抖动反复计数）:
        进入固定态需 freq ≥ fix_thresh; 退出需跌回 < exit(0.5);
        固定态下换成别的等位 = 替换固定, 也算新事件。"""
        n = len(pop)
        for j, site in enumerate(self.sites):
            col = pop[:, j]
            vals, counts = np.unique(col, return_counts=True)
            freq = dict(zip(map(int, vals), (c / n for c in counts)))
            key = (pop_id, site)
            cur = self._fixed_state.get(key)
            if self._wt is not None and int(self._wt[j]) in freq \
                    and freq[int(self._wt[j])] >= self.fix_thresh:
                cand = None                       # WT 重新占优 = 退回野生型
            else:
                cand = max((a for a, f in freq.items()
                            if f >= self.fix_thresh and a != int(self._wt[j])),
                           key=lambda a: freq[a], default=None)
            if cand is not None:
                if cand != cur:
                    self.fixations.append(dict(
                        gen=gen, pop=pop_id, site=site, aa=AA20[cand],
                        freq=round(freq[cand], 3)))
                    self._fixed_state[key] = cand
            elif cur is not None and freq.get(cur, 0.0) < self.EXIT_THRESH:
                self._fixed_state[key] = None     # 丢失（多态), 允许将来再记
            # 其余情况（未固定且原本未固定 / 固定中但未跌破退出线）不动

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
        """边表: (首次出现代, child_hash, parent_hash)。

        父 = 已知基因型中与 child 编辑距离 1 的邻居（对每个位点枚举 19 个替换,
        共 L×19 次哈希查表——O(N×L×19), 万级基因型可跑）中累计频率最高者;
        无单步祖先（如迁移注入/WT 起点）则 parent=None。
        """
        index: Dict[Tuple[int, ...], str] = {}
        edges = []
        for h, g in sorted(self._seen.items(), key=lambda kv: kv[1]):
            geno = self._genos[h]
            t = tuple(int(x) for x in geno)
            parent, best = None, -1
            for j in range(len(t)):
                for aa in range(20):
                    if aa == t[j]:
                        continue
                    h2 = index.get(t[:j] + (aa,) + t[j + 1:])
                    if h2 is not None:
                        f = self._total_freq.get(h2, 0)
                        if f > best:
                            best, parent = f, h2
            edges.append((g, h, parent))
            index[t] = h
        return edges

    # ---------------------------------------------------------------- 导出
    def _tree_maps(self) -> Tuple[Dict[str, Optional[str]], Dict[Optional[str], List[str]]]:
        """边表 → (child→parent, parent→children) 映射（每次导出时构建, 万级边表毫秒级）。"""
        parent_of: Dict[str, Optional[str]] = {}
        children_of: Dict[Optional[str], List[str]] = {}
        for _, h, p in self.build_lineage():
            parent_of[h] = p
            children_of.setdefault(p, []).append(h)
        return parent_of, children_of

    def _keep_set(self, top_k: int,
                  parent_of: Dict[str, Optional[str]]) -> set:
        """保留集: 按累计频率取 top_k 基因型, 再补齐其全部祖先到根——
        保证导出的子树对每个入选叶都连通（"主干谱系"的正确定义）。"""
        ranked = sorted(self._total_freq.items(), key=lambda kv: -kv[1])
        keep: set = set()
        for h, _ in ranked[:max(top_k, 0)]:
            if h not in self._seen:
                continue
            keep.add(h)
            cur = parent_of.get(h)
            while cur is not None and cur not in keep:
                keep.add(cur)
                cur = parent_of.get(cur)
        return keep

    def _label_of(self, h: str) -> str:
        muts = self.muts_of(h)
        return "_".join(f"{s}{a}" for s, a in muts) or "WT"

    def to_newick(self, top_k: int = 12) -> str:
        """主干谱系 Newick（按累计频率取 top_k 叶 + 全部祖先, 真树结构非平铺链）。

        叶/内节点标 = 突变串（内部节点标符合 Newick 规范）; 分支长度 =
        子节点与父节点首次出现代之差（根分支 = 自身首现代）。
        """
        parent_of, children_of = self._tree_maps()
        keep = self._keep_set(top_k, parent_of)

        def rec(h: str, is_root: bool) -> str:
            kids = sorted((c for c in children_of.get(h, []) if c in keep),
                          key=lambda c: self._seen.get(c, 0))
            if kids:
                base = self._seen[h]
                body = "(" + ",".join(
                    rec(c, False) + f":{max(self._seen[c] - base, 0)}" for c in kids) + ")"
            else:
                body = ""
            return body + self._label_of(h)

        roots = sorted((c for c in children_of.get(None, []) if c in keep),
                       key=lambda c: self._seen.get(c, 0))
        inner = ",".join(rec(r, True) + f":{self._seen[r]}" for r in roots)
        return "(" + inner + ");"

    def to_ascii_tree(self, top_k: int = 12) -> str:
        """主干谱系 ASCII 渲染（同一保留集, 缩进树; 节点注 首现代/累计频率）。"""
        parent_of, children_of = self._tree_maps()
        keep = self._keep_set(top_k, parent_of)
        lines: List[str] = []

        def rec(h: Optional[str], prefix: str) -> None:
            kids = sorted((c for c in children_of.get(h, []) if c in keep),
                          key=lambda c: self._seen.get(c, 0))
            for i, c in enumerate(kids):
                last = (i == len(kids) - 1)
                conn = "`- " if last else "|- "
                lines.append(f"{prefix}{conn}{self._label_of(c)} "
                             f"[gen {self._seen[c]}, n {self._total_freq.get(c, 0)}]")
                rec(c, prefix + ("   " if last else "|  "))

        roots = sorted((c for c in children_of.get(None, []) if c in keep),
                       key=lambda c: self._seen.get(c, 0))
        rec(None, "")
        return "\n".join(lines)

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

    def save(self, out_prefix: str, tree_top_k: int = 12) -> Dict[str, str]:
        """落盘五件套: <prefix>_edges.csv / _timeline.json / _report.md /
        _tree.newick / _tree.txt（ASCII 树）。"""
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
        with open(out_prefix + "_tree.newick", "w", encoding="utf-8") as f:
            f.write(self.to_newick(tree_top_k) + "\n")
        with open(out_prefix + "_tree.txt", "w", encoding="utf-8") as f:
            f.write(self.to_ascii_tree(tree_top_k) + "\n")
        return {k: out_prefix + s for k, s in
                (("edges", "_edges.csv"), ("timeline", "_timeline.json"),
                 ("report", "_report.md"), ("newick", "_tree.newick"),
                 ("ascii_tree", "_tree.txt"))}
