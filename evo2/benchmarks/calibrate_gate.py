# -*- coding: utf-8 -*-
"""用 Boltz-2 判读标签（240 models）校准 DNA-aware gate。

对比:
  gate_v1 (解析式启发，此前版本)  —— 读头芳香偏好 / 双锁正电即加分 / 锚点必须为 R
  gate_v2 (Boltz 标签校准)        —— 读头 K>F>>R / Y217 芳香 / M255 M>I>>K / 锚点去电荷提升判别
评估: Spearman 相关 vs Boltz dual_S1（靶标双锁率）与 Δiface（结合判别）
"""
import csv
import json
import os
import sys

sys.path.insert(0, r"A:/claudework/evo2/esm3")
from ppri_dna_aware import DnaAwareLandscape  # noqa

AA = "ACDEFGHIKLMNPQRSTVWY"

PLAN = r"A:/Data/设计蛋白/PprI_ssDNA_design/wetlab/order_plan_data.json"
STATS = r"A:/claudework/out/boltz_six/boltz_six_stats.csv"
OUT = r"A:/claudework/out/boltz_six/gate_calibration.csv"

# ---- Boltz 实测标签 ----
stats = {r["cand"]: {k: float(v) for k, v in r.items() if k != "cand"}
         for r in csv.DictReader(open(STATS))}

# ---- 候选基因型（53 位点空间）----
plan = json.load(open(PLAN, encoding="utf-8"))
variants = plan["variants"]

# 需要一个 base oracle 来提供 L/sites/wt_idx；用轻量 stub（不做 ESM3 推理）
class Stub:
    L = 53
    def __init__(self, sites, wt_idx):
        self.sites = sites
        self.wt_idx = wt_idx
    def evaluate(self, g):
        return [0.0] * len(g)

priors = list(csv.DictReader(open(r"A:/claudework/ppri_evo/results/priors.csv")))
sites = [int(r["seq_idx"]) for r in priors]
WT = open(r"A:/claudework/ppri_evo/inputs/wt_254.fasta").read().splitlines()[1].strip()
wt_idx = [AA.index(WT[s]) for s in sites]          # sites 为 0-based seq_idx
seqidx_to_col = {s: i for i, s in enumerate(sites)}

stub = Stub(sites, wt_idx)
land = DnaAwareLandscape(stub)   # gate_v1

# ---- gate_v2: Boltz 校准的经验偏好表 ----
READ_PREF = {'K': 1.00, 'W': 0.50, 'Y': 0.55, 'F': 0.30, 'Q': 0.20,
             'A': -0.20, 'R': -1.00}
Y217_PREF = {'Y': 1.00, 'F': 0.50, 'W': 0.50, 'A': -0.20, 'K': -0.40, 'R': -0.60}
M255_PREF = {'M': 1.00, 'I': 0.60, 'L': 0.50, 'A': -0.20, 'R': -0.80, 'K': -1.00}
CHARGE = {'R': 1.0, 'K': 0.70, 'H': 0.30}

def gate_v2(geno):
    col = seqidx_to_col
    def aa_at(pdb):
        c = col.get(pdb - 22)
        return AA[int(geno[c])] if c is not None else None
    read = READ_PREF.get(aa_at(88), 0.0)
    y217 = Y217_PREF.get(aa_at(217), 0.0)
    m255 = M255_PREF.get(aa_at(255), 0.0)
    anchor_charge = sum(CHARGE.get(aa_at(p), 0.0) for p in (85, 207, 267)) / 3.0
    return 0.40 * read + 0.20 * y217 + 0.20 * m255 - 0.20 * anchor_charge

rows = []
for vname, e in variants.items():
    cand = vname.replace("PprI_", "")
    geno = list(wt_idx)
    outside = []
    for m in e["mutations"]:
        c = seqidx_to_col.get(m["pdb"] - 22)
        if c is None:
            outside.append(f"{m['wt']}{m['pdb']}{m['mut']}")
            continue
        assert AA[geno[c]] == m["wt"]
        geno[c] = AA.index(m["mut"])
    g1 = land._gate(geno) if hasattr(land, "_gate") else float("nan")
    g2 = gate_v2(geno)
    st = stats[cand]
    rows.append(dict(cand=cand, n_mut=len(e["mutations"]),
                     gate_v1=round(g1, 4), gate_v2=round(g2, 4),
                     dual_S1=st["dual_S1"], d_iface=st["d_iface"],
                     d_act=st["d_act"], act_S1=st["act_S1"],
                     outside53=";".join(outside) or "-"))

def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")

print(f"{'cand':<11} {'n':>3} | {'gate_v1':>8} {'gate_v2':>8} | "
      f"{'dual_S1':>8} {'Δiface':>8} {'Δact':>7} {'act_S1':>7} | 53外位点")
print("-" * 95)
for r in sorted(rows, key=lambda r: -r["gate_v2"]):
    print(f"{r['cand']:<11} {r['n_mut']:>3} | {r['gate_v1']:>8.3f} {r['gate_v2']:>8.3f} | "
          f"{r['dual_S1']:>8.2f} {r['d_iface']:>+8.1f} {r['d_act']:>+7.2f} {r['act_S1']:>7.2f} | {r['outside53']}")

dual = [r["dual_S1"] for r in rows]
dif = [r["d_iface"] for r in rows]
g1 = [r["gate_v1"] for r in rows]
g2 = [r["gate_v2"] for r in rows]
print("\n=== Spearman 相关（n=6）===")
print(f"gate_v1 vs dual_S1 : rho = {spearman(g1, dual):+.3f}")
print(f"gate_v1 vs Δiface  : rho = {spearman(g1, dif):+.3f}")
print(f"gate_v2 vs dual_S1 : rho = {spearman(g2, dual):+.3f}")
print(f"gate_v2 vs Δiface  : rho = {spearman(g2, dif):+.3f}")

with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nWROTE {OUT}")
