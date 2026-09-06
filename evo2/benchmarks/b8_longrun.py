#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B8 — 长时程进化演示: GB1 完整景观 × 10⁴ 代 × 事件脚本（路线图 R2 出口）。

对照 v1 方案 §3.2/§3.5 的可行性核心论证, 三个问题各给一个实测数:
  1. 嵌入缓存离线构建: 全景观 160k 基因型的适应度表经 FitnessCache 预计算
     （昂贵层调用只发生一次）, 冷/温查询对比 → 命中率与墙钟;
  2. 运行期重复率: 4×10⁷ 个体-代 中唯一基因型占比（lineage 口径）→ v1 预期 >99%;
  3. 事件脚本下的轨迹: 瓶颈/迁移/温度台阶 + 平行 4 群 → 固定事件时间线、
     谱系三件套、三情景年轴换算。

诚实边界: 昂贵层这里是 GB1 DMS 实测景观查表（160k 全已知）, 不是 PLM/结构 Ensemble
——缓存机制与事件动力学是真实演练, "昂贵模型"是被查表替身的。这是演示, 不是预测。

运行: python evo2/benchmarks/b8_longrun.py  （单节点, 预计数分钟）
输出: evo2/results/b8_longrun/{summary.json, stats.csv, run1_edges.csv,
      run1_timeline.json, run1_report.md}
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE)))

from b1b_gb1 import GB1_WT, SITES, load_gb1  # noqa: E402
from engine.kernel import WFKernel  # noqa: E402
from engine.events import Event, EventScript, run_with_events  # noqa: E402
from engine.lineage import LineageRecorder  # noqa: E402
from engine.fitcache import FitnessCache  # noqa: E402
from engine.seqtools import AA20, AA2IDX  # noqa: E402

OUT = os.path.join(ROOT, "evo2", "results", "b8_longrun")
SEED = 20260906
N_POP, NE, N_GEN = 4, 1000, 10000
WT_AAS = "VDGV"  # SITES 上 GB1 WT 残基（与 b1b 同序）

CFG = {"mutations_per_genome_per_gen": {"lambda": 0.4}, "T": 0.6, "n_mut_max": 12}


# ---------------------------------------------------------------- 景观 → 表
def build_table():
    """GB1 全景观适应度表 [20,20,20,20]（轴序 = SITES 升序; 0 号 AA = WT 残基）。

    实测缺失的组合（~7%）用单突变加性和回填——回填规则写进 summary, 不冒充实测。
    """
    oracle, train_keys, _ = load_gb1()
    wt_idx = tuple(AA2IDX[c] for c in WT_AAS)

    def key_to_idx(key):
        idx = list(wt_idx)
        pos = {s: i for i, s in enumerate(SITES)}
        for site, aa in key:
            idx[pos[site]] = AA2IDX[aa]
        return tuple(idx)

    singles = {}  # (site_pos, aa_idx) → y
    for key, y in oracle.items():
        if len(key) == 1:
            (site, aa), = key
            singles[(SITES.index(site), AA2IDX[aa])] = y
    y_wt = oracle.get((), 1.0)

    table = np.full((20, 20, 20, 20), np.nan, dtype=np.float64)
    measured = 0
    for key, y in oracle.items():
        table[key_to_idx(key)] = y
        measured += 1
    # 加性回填: f = y_wt + Σ(single - y_wt)
    filled = 0
    it = np.ndindex(20, 20, 20, 20)
    for idx in it:
        if np.isnan(table[idx]):
            f = y_wt
            for i, a in enumerate(idx):
                if a != wt_idx[i]:
                    f += singles.get((i, a), 0.0) - y_wt
            table[idx] = f
            filled += 1
    return table, wt_idx, measured, filled, y_wt


def priors_from_singles(table, wt_idx, y_wt, T=0.8):
    """提议先验 = 单突变效应 softmax（比均匀提议更贴近真实 mutational target）。"""
    pri = {}
    for i, site in enumerate(SITES):
        d = {}
        for a in range(20):
            if a == wt_idx[i]:
                continue
            idx = list(wt_idx)
            idx[i] = a
            d[AA20[a]] = float(np.exp((table[tuple(idx)] - y_wt) / T))
        s = sum(d.values())
        pri[int(site)] = {k: max(v / (s + 1e-12), 1e-3) for k, v in d.items()}
    return pri


class OracleTableKernel(WFKernel):
    """适应度 = 预计算景观表查表（运行期零模型调用, 即缓存全命中路径）。"""

    def __init__(self, wt_seq, sites, pri, table, cfg, seed):
        super().__init__(wt_seq, sites, pri, cfg, seed=seed)
        self.table = table

    def _fitness(self, geno):
        i = geno.astype(int)
        return self.table[i[:, 0], i[:, 1], i[:, 2], i[:, 3]]


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    table, wt_idx, measured, filled, y_wt = build_table()
    t_table = time.time() - t0
    print(f"[1] 景观表: measured={measured} additive_filled={filled} ({t_table:.1f}s)")

    # ---- 缓存离线构建（昂贵层替身: 查表也走 cache, 演示冷/温差）----
    cache = FitnessCache(os.path.join(OUT, "landscape_cache.db"),
                         model_version="gb1_dms_v1")
    known = cache.load_all()
    t0 = time.time()
    new_items = []
    for idx in np.ndindex(20, 20, 20, 20):
        muts = tuple((SITES[i], AA20[a]) for i, a in enumerate(idx) if a != wt_idx[i])
        k = ";".join(f"{p}{a}" for p, a in sorted(muts))
        if k not in known and muts:
            new_items.append((muts, {"fitness": float(table[idx])}))
    cache.put_many(new_items)
    t_cold = time.time() - t0
    cache.misses = 0                        # 命中率只统计运行/温查询期（冷构建单独报告）
    cache.hits = 0
    t0 = time.time()
    warm_miss = 0
    for idx in list(np.ndindex(20, 20, 20, 20))[::160]:  # 温查询抽样 1000
        muts = tuple((SITES[i], AA20[a]) for i, a in enumerate(idx) if a != wt_idx[i])
        if muts and cache.get(muts) is None:
            warm_miss += 1
    t_warm = time.time() - t0
    cache.hits = 1000 - warm_miss
    print(f"[2] 缓存: 新写 {len(new_items)} 条 {t_cold:.1f}s | 温查询 1000 条 "
          f"{t_warm*1000:.0f}ms, miss={warm_miss}, hit_rate={cache.hit_rate:.4f}")

    # ---- 长时程运行 ----
    pri = priors_from_singles(table, wt_idx, y_wt)
    kernel = OracleTableKernel(GB1_WT, SITES, pri, table, CFG, seed=SEED)
    script = EventScript([
        Event("bottleneck", 3000, 50, factor=0.05),
        Event("migration", 6000, 10, donor=None, fraction=0.3),
        Event("temperature_shift", 7000, N_GEN - 7000, to_T=1.2),
    ])
    rec = LineageRecorder(kernel.sites, fix_thresh=0.9)
    rec.set_wt(kernel.wt_idx)
    t0 = time.time()
    stats, pops, elog = run_with_events(kernel, script, n_pop=N_POP, n_gen=N_GEN,
                                        Ne=NE, observer=rec.observe)
    t_run = time.time() - t0
    print(f"[3] 运行: {N_GEN} 代 × {N_POP} 群 × Ne={NE} = {N_POP*NE*N_GEN:.1e} 个体-代 "
          f"{t_run:.0f}s | 唯一基因型 {rec.total_unique_events} "
          f"| 重复率 {rec.cache_hit_rate:.4f} | 固定事件 {len(rec.fixations)}")

    # ---- 落盘 ----
    df = pd.DataFrame(stats)
    df.to_csv(os.path.join(OUT, "stats.csv"), index=False)
    files = rec.save(os.path.join(OUT, "run1"))
    best_f, best_g = -1e9, None
    for p in range(N_POP):
        fits = kernel._fitness(pops[p])
        j = int(fits.argmax())
        if fits[j] > best_f:
            best_f, best_g = float(fits[j]), pops[p][j].copy()
    best_muts = [(int(SITES[j2]), AA20[int(best_g[j2])])
                 for j2 in np.flatnonzero(best_g != kernel.wt_idx)]
    from collections import Counter as _C
    tl = rec.timeline()
    par = _C((d["site"], d["aa"]) for d in tl)
    parallel_repeats = sum(1 for v in par.values() if v > 1)
    summary = dict(
        seed=SEED, n_pop=N_POP, Ne=NE, n_gen=N_GEN, T=CFG["T"],
        events=[dict(type=e.type, t_gen=e.t_gen, duration_g=e.duration_g,
                     **{k: v for k, v in e.params.items()}) for e in script.events],
        landscape=dict(measured=measured, additive_filled=filled, y_wt=y_wt),
        cache=dict(new_entries=len(new_items), cold_build_s=round(t_cold, 2),
                   warm_query_1000_ms=round(t_warm * 1000, 1),
                   warm_hit_rate=round(cache.hit_rate, 4)),
        run=dict(wall_s=round(t_run, 1),
                 total_inds=rec.total_inds,
                 unique_genotypes=rec.total_unique_events,
                 repetition_rate=round(rec.cache_hit_rate, 6),
                 fixation_events=len(rec.fixations),
                 parallel_repeats=parallel_repeats),
        best_final=dict(fitness=round(best_f, 4),
                        muts=[f"{s}{a}" for s, a in best_muts]),
        replay=dict(seed=SEED, config=CFG, sites=SITES, wt_aas=WT_AAS,
                    note="seed+config+sites 唯一决定轨迹; 景观表由 DMS CSV 版本决定"),
        files=files)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f"[4] 输出: {OUT}")
    print(json.dumps({k: summary[k] for k in ("cache", "run", "best_final")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
