# -*- coding: utf-8 -*-
"""六候选「保机制精修」—— dna_aware_ppri(gate v3) 的首个真实场景端到端使用。

任务（TOOL_FIX_TARGET_AWARE.md §6 承诺的增量轮次）:
    在保留六候选全部设计突变的前提下，允许新增 ≤3 个突变，
    用 DnaAwareLandscape(v3 gate) 搜索比原候选更优的 v2 变体；
    OLD(纯 v3 z) 同协议对照 —— 验证工具新增的 DNA 维度改变精修选择。

协议要点（与 evolve_six_wetlab 的"约束式打磨"的区别）:
    1. 冻结 = 候选的全部设计突变（机制 + 骨架），一个不洗 —— 上轮"打磨"把
       HQL2 七个骨架突变洗回 WT 的教训。
    2. 新增只能落在景观内当前为 WT 的位点；锚点 R85/R207/R267 硬禁止新增改动
       （R85A/R207A/R267A 三突变实验上完全丧失 ssDNA 结合，且 Boltz 看不见锚点
       破坏 —— 硬约束来自实验证据，非拟合）。
    3. z 用 evaluate_multi(K≤32) 忠实打分（原 evaluate 只取首突变，对 6-19 突变
       候选是失真排序）。
    4. gate v3 只覆盖 F88/M255/Y217；组合景观 combo = 0.5*z + 0.5*gate。

附带 gate 泛化小考:
    v3 校准集 = 22 变体（s13_c1 背景）。六候选除 s13_c1 自身外均不在校准集 →
    gate v1/v3 对六候选 Boltz 标签（d_act/d_dual/d_iface）的 Spearman 是一次
    跨背景 held-out 检验（n=6，样本极小，只作方向性证据）。
"""
import json
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
ESM3 = os.path.join(ROOT, "esm3")
if ESM3 not in sys.path:
    sys.path.insert(0, ESM3)

from ppri_surrogate_v3 import PprISurrogateV3, PPRI_SITES, AA, AAI  # noqa: E402
from ppri_dna_aware import DnaAwareLandscape  # noqa: E402

SITES = np.array(PPRI_SITES, dtype=np.int64)           # seq_idx (0-based)
PDB2J = {int(s) + 22: j for j, s in enumerate(SITES)}  # pdb -> 景观列
ANCHOR_PDB = (85, 207, 267)
HOTSPOT_PDB = (88, 217, 255)


class SurrV3(PprISurrogateV3):
    """evaluate_multi 放宽到 K≤32，使 6-19 突变深变体可忠实打分。"""
    def evaluate_multi(self, genos):
        genos = np.asarray(genos, dtype=np.int64)
        N, L = genos.shape
        K = min(L, 32)
        rows = np.full((N, K, 3), -1, dtype=np.int64)
        for i in range(N):
            m = np.where(genos[i] != self.wt_idx)[0]
            for t, j in enumerate(m[:K]):
                rows[i, t] = (int(j), int(self.wt_idx[j]), int(genos[i, j]))
        with torch.no_grad():
            return self.model(self.site_idx, torch.from_numpy(rows), self.site_ctx).numpy()


class MultiEvalAdapter:
    """把 .evaluate() 路由到 evaluate_multi —— DnaAwareLandscape.evaluate 内部
    调 base.evaluate（K=1 截断版），对深变体候选失真，这里统一改为多突变忠实版。"""
    def __init__(self, surr):
        self._s = surr

    def __getattr__(self, k):
        return getattr(self._s, k)

    def evaluate(self, genos):
        return self._s.evaluate_multi(genos)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float((rx * ry).mean())


def build_geno(mutations, wt_idx):
    geno = wt_idx.copy()
    landscape_muts, non_landscape = [], []
    for m in mutations:
        if m["pdb"] in PDB2J:
            j = PDB2J[m["pdb"]]
            geno[j] = AAI[m["mut"]]
            landscape_muts.append((j, m["pdb"], m["wt"], m["mut"]))
        else:
            non_landscape.append((m["pdb"], m["wt"], m["mut"]))
    return geno, landscape_muts, non_landscape


def full_mut_list(landscape_muts, non_landscape, geno):
    out = [(pdb, wt, AA[int(geno[j])]) for (j, pdb, wt, mut) in landscape_muts]
    out += [(pdb, wt, mut) for (pdb, wt, mut) in non_landscape]
    return sorted(out, key=lambda x: x[0])


def mut_str(geno, landscape_muts, non_landscape, wt_idx):
    """全部突变 = 基因型中所有非 WT 列（设计 + 新增）+ 景观外设计突变。"""
    out = [(int(SITES[j]) + 22, AA[int(wt_idx[j])], AA[int(geno[j])])
           for j in range(len(SITES)) if geno[j] != wt_idx[j]]
    out += [(pdb, wt, mut) for (pdb, wt, mut) in non_landscape]
    return ";".join(f"{p}{w}{a}" for p, w, a in sorted(out, key=lambda x: x[0]))


def gate_probe(dna, geno):
    """单基因型 gate v3 / v1 分值。"""
    return dna._gate_v3(geno), dna._gate_v1(geno)


def refine(landscape_eval, gate_fn, seed_geno, wt_idx, frozen_cols, allowed_cols,
           max_add=3):
    """贪心 add-only 精修：只允许在 allowed_cols 里当前为 WT 的位点上新增突变。

    landscape_eval(genos) -> combo 适应度；gate_fn(geno) -> gate 分值（仅记录用）。
    返回 (best_geno, added_cols, trace)。
    """
    cur = seed_geno.copy()
    cur_f = float(landscape_eval(cur[None, :])[0])
    added = []
    trace = [(0, cur_f, gate_fn(cur))]
    while len(added) < max_add:
        cand_cols, cand_aa = [], []
        for j in allowed_cols:
            if cur[j] != wt_idx[j]:
                continue
            for a in range(20):
                if a == wt_idx[j]:
                    continue
                cand_cols.append(j)
                cand_aa.append(a)
        if not cand_cols:
            break
        block = np.tile(cur, (len(cand_cols), 1))
        block[np.arange(len(cand_cols)), cand_cols] = cand_aa
        fits = landscape_eval(block)
        i = int(np.argmax(fits))
        if float(fits[i]) <= cur_f + 1e-9:
            break
        cur = block[i].copy()
        cur_f = float(fits[i])
        added.append((int(cand_cols[i]), int(cand_aa[i])))
        trace.append((len(added), cur_f, gate_fn(cur)))
    return cur, added, trace


def mech_report(name, geno, wt_idx):
    def aa_at(pdb):
        j = PDB2J.get(pdb)
        return AA[int(geno[j])] if j is not None else "-"
    f88, y217, m255 = aa_at(88), aa_at(217), aa_at(255)
    anchors = "".join(aa_at(p) for p in ANCHOR_PDB)
    return {"variant": name, "F88": f88, "Y217": y217, "M255": m255, "anchors": anchors}


def main():
    surr = SurrV3()
    wt_idx = surr.wt_idx.copy()
    adapter = MultiEvalAdapter(surr)
    dna = DnaAwareLandscape(adapter)          # gate v3 默认, w=0.5/0.5
    print(f"[ok] surrogate v3 loaded: L={surr.L} wt_f={surr.wt_f:.3f} ref_f={surr.ref_f:.3f}")
    print(f"[ok] DnaAwareLandscape gate v3: read_col={dna._read_col} "
          f"lock_cols={dna._lock_cols} anchor_cols={dna._anchor_cols}")

    cand = json.load(open("A:/Data/设计蛋白/PprI_ssDNA_design/wetlab/order_plan_data.json"))["variants"]

    # ---------- ① gate 泛化小考（六候选 vs Boltz 标签） ----------
    import csv as _csv
    stats = {}
    with open(os.path.join(ROOT, "results", "boltz_six", "boltz_six_stats.csv")) as fh:
        for r in _csv.DictReader(fh):
            stats[r["cand"]] = r
    names6 = [n.replace("PprI_", "") for n in cand]
    g3, g1, d_act, d_dual, d_if = [], [], [], [], []
    print("\n=== gate 泛化小考（校准集=22 变体 s13_c1 背景；六候选为准 held-out, n=6）===")
    for n in names6:
        geno, lm, nl = build_geno(cand["PprI_" + n]["mutations"], wt_idx)
        a3, a1 = gate_probe(dna, geno)
        g3.append(a3); g1.append(a1)
        d_act.append(float(stats[n]["d_act"]))
        d_dual.append(float(stats[n]["d_dual"]))
        d_if.append(float(stats[n]["d_iface"]))
        print(f"  {n:<12} gate_v3={a3:+.3f}  gate_v1={a1:+.3f}  "
              f"d_act={float(stats[n]['d_act']):+5.2f}  "
              f"d_dual={float(stats[n]['d_dual']):5.2f}  "
              f"d_iface={float(stats[n]['d_iface']):+6.2f}")
    for label, vec in (("d_act", d_act), ("d_dual", d_dual), ("d_iface", d_if)):
        print(f"  Spearman v3 vs {label:<7} = {spearman(g3, vec):+.3f}   "
              f"v1 vs {label:<7} = {spearman(g1, vec):+.3f}")

    # ---------- ② 保机制精修（P0/P1/正向对照 三个候选 × OLD/NEW） ----------
    targets = ["PprI_s13_c1", "PprI_TrackF_r1", "PprI_HQL2"]
    rows = []
    print("\n=== 保机制精修（冻结全部设计突变；新增≤3；锚点硬保护）===")
    for name in targets:
        geno0, lm, nl = build_geno(cand[name]["mutations"], wt_idx)
        design_cols = {j for (j, _, _, _) in lm}
        frozen = design_cols | {PDB2J[p] for p in ANCHOR_PDB if p in PDB2J}
        allowed = [j for j in range(len(SITES)) if j not in frozen]
        seed_z = float(surr.evaluate_multi(geno0[None, :])[0])
        seed_g3, _ = gate_probe(dna, geno0)
        seed_n = int((geno0 != wt_idx).sum())
        print(f"\n[{name}] seed: n_mut={seed_n}(+{len(nl)} 景观外) z={seed_z:.3f} "
              f"gate_v3={seed_g3:+.3f}")
        for tag, leval, gfn in (("NEW_dna_aware", dna.evaluate, dna._gate_v3),
                                ("OLD_pure_v3", surr.evaluate_multi, lambda g: 0.0)):
            best, added, trace = refine(leval, gfn, geno0, wt_idx, frozen, allowed, max_add=3)
            bz = float(surr.evaluate_multi(best[None, :])[0])
            bg3, _ = gate_probe(dna, best)
            added_str = ";".join(f"{int(SITES[j])+22}{AA[int(wt_idx[j])]}{AA[a]}"
                                 for j, a in added) or "-"
            mr = mech_report(name, best, wt_idx)
            print(f"  {tag:<15} +{len(added)} mut [{added_str}]  "
                  f"z={bz:+.3f} gate_v3={bg3:+.3f} combo={trace[-1][1]:+.3f}  "
                  f"F88={mr['F88']} Y217={mr['Y217']} M255={mr['M255']} "
                  f"anchors={mr['anchors']}")
            rows.append(dict(variant=name, landscape=tag, n_mut_seed=seed_n,
                             n_added=len(added), added=added_str,
                             z_seed=round(seed_z, 3), z_ref=round(bz, 3),
                             dz=round(bz - seed_z, 3),
                             gate_seed=round(seed_g3, 3), gate_ref=round(bg3, 3),
                             combo_ref=round(trace[-1][1], 3),
                             muts=mut_str(best, lm, nl, wt_idx)))

    # ---------- ③ 落盘 ----------
    out_csv = os.path.join(ROOT, "results", "refine_six_wetlab.csv")
    with open(out_csv, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[saved] {out_csv}")
    return rows


if __name__ == "__main__":
    main()
