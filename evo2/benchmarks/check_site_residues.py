# -*- coding: utf-8 -*-
"""由 order_plan_data.json 突变表推导各候选在机制位点的实际残基。"""
import json

PLAN = r"A:/Data/设计蛋白/PprI_ssDNA_design/wetlab/order_plan_data.json"
WT = open(r"A:/claudework/ppri_evo/inputs/wt_254.fasta").read().splitlines()[1].strip()

# seq(1-based) = PDB - 21 ; 位点 -> PDB
SITES = [("F88读头", 88), ("H92", 92), ("E93", 93), ("H96", 96), ("E123", 123),
         ("R85锚", 85), ("R207锚", 207), ("R267锚", 267),
         ("Y217锁", 217), ("R253锁", 253), ("M255锁", 255)]

data = json.load(open(PLAN, encoding="utf-8"))
variants = data["variants"]

rows = []
for name, e in variants.items():
    s = list(WT)
    mods = {}
    for m in e["mutations"]:
        pdb, wt, mut = m["pdb"], m["wt"], m["mut"]
        si = pdb - 21
        assert s[si - 1] == wt, f"{name} PDB{pdb}: fasta={s[si-1]} != {wt}"
        s[si - 1] = mut
        mods.setdefault(m.get("module", "?"), []).append(f"{wt}{pdb}{mut}")
    seq = "".join(s)
    rows.append((name.replace("PprI_", ""), len(e["mutations"]), seq, e.get("desc", ""), mods))

hdr = " ".join(f"{n:<8}" for n, _ in SITES)
print(f"{'cand':<12} {'n':>3} | {hdr}")
print("-" * 16 + "|" + "-" * (9 * len(SITES)))
for name, n, seq, desc, mods in rows:
    cells = []
    for _, pdb in SITES:
        si = pdb - 21
        cells.append(f"{WT[si-1]}{pdb}{seq[si-1]:<5}")
    print(f"{name:<12} {n:>3} | " + " ".join(cells))

print("\n=== 各位点突变模块归属（F88/Y217/R253/M255 关键）===")
for name, n, seq, desc, mods in rows:
    keep = []
    for k, v in mods.items():
        if any(x in k for x in ("核心1", "核心2", "双锁", "Patch1")):
            keep.append(f"{k}: {','.join(v)}")
    print(f"\n[{name}] n={n} — {desc}")
    for k in keep:
        print("   ", k)

with open(r"A:/claudework/out/boltz_six/site_residues.csv", "w", newline="", encoding="utf-8") as fh:
    import csv
    w = csv.writer(fh)
    w.writerow(["cand", "n_mut"] + [f"{n}(PDB{p})" for n, p in SITES])
    for name, n, seq, desc, mods in rows:
        w.writerow([name, n] + [f"{WT[p-22]}{seq[p-22]}" for _, p in SITES])
print("\nWROTE site_residues.csv")
