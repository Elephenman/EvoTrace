# -*- coding: utf-8 -*-
"""验证 DnaAwareLandscape 让工具能朝靶标进化（而非洗回 WT）。

对比:
    OLD = 纯 surrogate_ppri_v3（DNA 无关）→ 演化趋向洗回 WT / 随机偏离
    NEW = DnaAwareLandscape（z + 机制门控）→ 演化保留/强化靶标匹配机制

指标: 终末种群中 锚点保留率 / 双锁正电率 / 读头芳香率 / 平均 z,gate,combo
"""
import os
import csv
import numpy as np
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "esm3"))

from ppri_surrogate_v3 import PprISurrogateV3, AA
from ppri_dna_aware import DnaAwareLandscape

AAI = {a: i for i, a in enumerate(AA)}
POSITIVE = {8, 14}       # K R
POLAR = {2, 3, 11, 13, 15, 16, 6}
AROMATIC = {4, 19, 18}
# 机制热点 pdb -> seq_idx 列
MECH = {"readhead": (88,), "lock": (253, 217, 255), "anchor": (85, 207, 267)}


def seqidx_to_pdb(seq_idx):
    return seq_idx + 22


def decode(geno, sites, wt_idx):
    out = []
    for col, (s, w) in enumerate(zip(sites, wt_idx)):
        m = geno[col]
        if m != w:
            out.append(f"{seqidx_to_pdb(int(s))}{AA[w]}{AA[m]}")
    return ";".join(out) or "(WT)"


def mutate(wt, L, rng, max_mut):
    c = wt.copy()
    k = int(rng.integers(1, max_mut + 1))
    sites = rng.choice(L, size=k, replace=False)
    for s in sites:
        c[s] = int(rng.integers(0, 20))
    return c


def evolve(landscape, n_gen=80, pop=40, max_mut=8, seed=0):
    rng = np.random.default_rng(seed)
    wt = landscape.wt_idx.copy()
    pop_g = [wt.copy()] + [mutate(wt, landscape.L, rng, max_mut)
                           for _ in range(pop - 1)]
    for g in range(n_gen):
        cand = []
        for p in pop_g:
            for _ in range(pop):
                cand.append(mutate(p, landscape.L, rng, max_mut))
        cand = np.array(cand, dtype=np.int64)
        f = landscape.evaluate(cand)
        top = np.argsort(f)[-pop:]
        pop_g = [cand[t] for t in top]
    return np.array(pop_g, dtype=np.int64), f[top]


def stats(pop_g, landscape):
    """终末种群机制保留统计。"""
    sites = landscape.sites
    seqidx_to_col = {int(s): i for i, s in enumerate(sites)}
    anchor_cols = [seqidx_to_col[p - 22] for p in MECH["anchor"]
                   if (p - 22) in seqidx_to_col]
    lock_cols = [seqidx_to_col[p - 22] for p in MECH["lock"]
                 if (p - 22) in seqidx_to_col]
    read_col = seqidx_to_col.get(MECH["readhead"][0] - 22)
    n = len(pop_g)
    anchor_keep = np.mean([all(int(g[c]) == 14 for c in anchor_cols)
                           for g in pop_g]) if anchor_cols else 0.0
    lock_pos = np.mean([np.mean([1 if (int(g[c]) in POSITIVE or int(g[c]) in POLAR)
                                 else 0 for c in lock_cols]) for g in pop_g]) if lock_cols else 0.0
    read_aro = np.mean([1 if (read_col is not None and int(g[read_col]) in AROMATIC)
                        else 0 for g in pop_g])
    return {"anchor_keep": anchor_keep, "lock_pos": lock_pos, "read_aro": read_aro}


def main():
    base = PprISurrogateV3()
    old = base                       # 纯 z
    new = DnaAwareLandscape(base)    # z + gate

    print(f"[init] base L={base.L} wt_f={base.wt_f:.3f}  "
          f"read_col={new._read_col} lock_cols={new._lock_cols} anchor_cols={new._anchor_cols}")

    old_pop, old_f = evolve(old, seed=1)
    new_pop, new_f = evolve(new, seed=1)

    old_top = old_pop[np.argmax(old_f)]
    new_top = new_pop[np.argmax(new_f)]

    s_old = stats(old_pop, old)
    s_new = stats(new_pop, new)

    # 新景观 top 个体的 gate 分解
    new_gate = new._gate(new_top)

    print("\n=== TOP 个体 ===")
    print(f"OLD top: {decode(old_top, base.sites, base.wt_idx)}  z={float(old_f.max()):.3f}")
    print(f"NEW top: {decode(new_top, base.sites, base.wt_idx)}  "
          f"z={new.evaluate(new_top[None,:])[0]-new_gate:.3f} gate={new_gate:+.3f} "
          f"combo={float(new_f.max()):.3f}")

    print("\n=== 终末种群机制保留率（OLD vs NEW）===")
    print(f"锚点(R85/R207/R267)保留率 : {s_old['anchor_keep']:.2f} -> {s_new['anchor_keep']:.2f}")
    print(f"双锁(R253/Y217/M255)正电率: {s_old['lock_pos']:.2f} -> {s_new['lock_pos']:.2f}")
    print(f"读头(F88)芳香率           : {s_old['read_aro']:.2f} -> {s_new['read_aro']:.2f}")

    # 落盘
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "evolve_target_aware.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["landscape", "top_mut", "z", "gate", "combo",
                    "anchor_keep", "lock_pos", "read_aro"])
        w.writerow(["OLD_pure_v3", decode(old_top, base.sites, base.wt_idx),
                    float(old_f.max()), 0.0, float(old_f.max()),
                    s_old['anchor_keep'], s_old['lock_pos'], s_old['read_aro']])
        w.writerow(["NEW_dna_aware", decode(new_top, base.sites, base.wt_idx),
                    new.evaluate(new_top[None, :])[0] - new_gate, new_gate,
                    float(new_f.max()),
                    s_new['anchor_keep'], s_new['lock_pos'], s_new['read_aro']])
    print("\n[saved] results/evolve_target_aware.csv")


if __name__ == "__main__":
    main()
