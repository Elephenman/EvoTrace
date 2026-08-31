#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 EvoTrace v1.1 的 ESM3 代理景观，对六大湿实验候选做「约束式局部打磨」。

方法学边界（务必如实披露）：
  * v1.1 的 PprI 景观 = ESM3 零样本、与 DNA 序列无关 的通用蛋白适应度 z-score 预测器。
  * 因此它不能「直接吃 24nt 靶标」；24nt 靶标（pos17=G / pos23=T 读头）由
    **硬冻结读头/锚点/双锁/补丁突变**来承载设计意图，景观只打磨「其他/骨架」壳层。
  * evaluate_multi 原版只取前 8 突变 → 这里放宽为 K<=32，使 6-19 突变深变体可忠实打分。

搜索：WF 式 greedy 单点扫描 + 换位（swap）阶段，种子=候选基因型，硬冻结特异性位点，
只在 editable（其他/骨架）位点内寻优。输出每候选最优打磨版 + 工具 z-score 增量。
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

# ---- 放开 K 上限，使深变体可忠实打分 ----
class SurrV3(PprISurrogateV3):
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
            pred = self.model(self.site_idx, torch.from_numpy(rows), self.site_ctx)
        return pred.numpy()


SITES = np.array(PPRI_SITES, dtype=np.int64)          # seq_idx (0-based)
PDB2J = {int(s) + 22: j for j, s in enumerate(SITES)}  # pdb -> site index

# ---- 候选定义（来自 wetlab/order_plan_data.json） ----
CAND = json.load(open("A:/Data/设计蛋白/PprI_ssDNA_design/wetlab/order_plan_data.json"))["variants"]

# 模块 -> 是否硬冻结（特异性/锚点/读头/补丁/口袋 = 冻结；其他/骨架 = 可打磨）
FREEZE_MODULES = {
    "核心1 (G 读取: F88/Y170/N127/S167)",
    "核心1 (G 读取: F88/Y170/N127/S167); T-run 芳香堆叠 (Y149/Y196)",
    "核心2 (T23 锁: R253/Y217/M255/P256)",
    "核心2 (T23 锁: R253/Y217/M255/P256); T-run 芳香堆叠 (Y149/Y196)",
    "Patch1 (R85/R207/R267 非特异磷酸锚)",
    "Patch3 (正电静电面)",
    "G10 口袋设计区 (PDB181-206)",
}

# 锚点（b5 硬约束，景观外也保持）：88/135/171 限定芳香族
ANCHOR_PDB = {88, 135, 171}
ANCHOR_FAM = set("FYWMILV")


def build_geno(mutations):
    """把候选突变映射到 53 位点基因型；返回 (geno, landscape_muts, non_landscape, editable_j)。"""
    geno = SITES_wt_idx.copy()
    landscape_muts = []   # (j, pdb, wt, mut, module)
    non_landscape = []    # (pdb, wt, mut) 落在 53 位点外，必须保留
    for m in mutations:
        pdb = m["pdb"]
        if pdb in PDB2J:
            j = PDB2J[pdb]
            geno[j] = AAI[m["mut"]]
            landscape_muts.append((j, pdb, m["wt"], m["mut"], m["module"]))
        else:
            non_landscape.append((pdb, m["wt"], m["mut"]))
    editable_j = [j for (j, pdb, wt, mut, mod) in landscape_muts
                  if mod not in FREEZE_MODULES]
    return geno, landscape_muts, non_landscape, editable_j


def full_mut_list(landscape_muts, non_landscape, geno):
    out = []
    for (j, pdb, wt, mut, mod) in landscape_muts:
        out.append((pdb, wt, AA[geno[j]]))
    for (pdb, wt, mut) in non_landscape:
        out.append((pdb, wt, mut))
    out.sort(key=lambda x: x[0])
    return out


def greedy_swap(surr, geno, editable_j, seed=0):
    """WF 式局部精修：单点扫描 + 换位，只在 editable_j 内。返回 (best_geno, best_f)。"""
    rng = np.random.default_rng(seed)
    f = float(surr.evaluate_multi(geno[None, :])[0])
    best_g, best_f = geno.copy(), f
    for sweep in range(20):
        improved = False
        # 单点扫描
        for j in editable_j:
            cand = np.tile(best_g, (20, 1))
            cand[:, j] = np.arange(20)
            fits = surr.evaluate_multi(cand)
            i = int(np.argmax(fits))
            if fits[i] > best_f + 1e-9:
                best_g, best_f, improved = cand[i].copy(), float(fits[i]), True
        # 换位：回退一个 editable + 改另一个 editable
        mut = [j for j in editable_j if best_g[j] != SITES_wt_idx[j]]
        unm = [j for j in editable_j if best_g[j] == SITES_wt_idx[j]]
        if mut and unm:
            big = np.repeat(best_g[None, :], len(unm) * 20, axis=0)
            uu = np.repeat(unm, 20)
            aa = np.tile(np.arange(20), len(unm))
            big[np.arange(len(big)), uu] = aa
            # 同时把某个已突变 editable 回退 WT
            for m in mut:
                bb = big.copy()
                bb[:, m] = SITES_wt_idx[m]
                fits = surr.evaluate_multi(bb)
                i = int(np.argmax(fits))
                if fits[i] > best_f + 1e-9:
                    best_g, best_f, improved = bb[i].copy(), float(fits[i]), True
        if not improved:
            break
    return best_g, best_f


def main():
    global SITES_wt_idx
    surr = SurrV3()
    SITES_wt_idx = surr.wt_idx.copy()
    print(f"[ok] surrogate v3 loaded: L={surr.L}, wt_f={surr.wt_f:.3f}, ref_f={surr.ref_f:.3f}")
    print(f"[note] landscape is DNA-independent zero-shot; 24nt target carried by frozen readout sites")

    rows = []
    for name, v in CAND.items():
        geno, lm, nl, editable = build_geno(v["mutations"])
        seed_f = float(surr.evaluate_multi(geno[None, :])[0])
        out_muts0 = full_mut_list(lm, nl, geno)
        if not editable:
            print(f"  {name}: 设计已全固定/无壳层位点 (seed_f={seed_f:.3f}) -> 保持原样")
            rows.append(dict(variant=name, role=v["role"], n_mut=len(out_muts0),
                             seed_f=round(seed_f, 3), polish_f=round(seed_f, 3),
                             delta=0.0, polished="NO", muts=";".join(
                                 f"{p}{a}" for p, w, a in out_muts0)))
            continue
        best_g, best_f = greedy_swap(surr, geno, editable, seed=20260831)
        out_muts1 = full_mut_list(lm, nl, best_g)
        delta = best_f - seed_f
        polished = "YES" if delta > 0.01 else "NO"
        print(f"  {name}: seed_f={seed_f:.3f} -> polish_f={best_f:.3f} "
              f"delta={delta:+.3f} ({'polished' if polished=='YES' else 'no gain'})")
        rows.append(dict(variant=name, role=v["role"], n_mut=len(out_muts1),
                         seed_f=round(seed_f, 3), polish_f=round(best_f, 3),
                         delta=round(delta, 3), polished=polished,
                         muts=";".join(f"{p}{a}" for p, w, a in out_muts1)))

    df = pd.DataFrame(rows) if (pd := __import__("pandas")) else None
    out = os.path.join(ROOT, "results", "evolve_six_wetlab.csv")
    import csv as _csv
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["variant", "role", "n_mut", "seed_f",
                                           "polish_f", "delta", "polished", "muts"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n-> {out}")
    # 排序（打磨后 z-score 降序）
    rows.sort(key=lambda r: r["polish_f"], reverse=True)
    print("\n=== 工具打磨后排序（z-score 降序，DNA 无关通用适应度） ===")
    for r in rows:
        print(f"  {r['variant']:<16} polish_f={r['polish_f']:>7.3f} "
              f"delta={r['delta']:>+6.3f}  n_mut={r['n_mut']}  {r['polished']}")


if __name__ == "__main__":
    main()
