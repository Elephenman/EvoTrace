#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""去混杂扫描：固定 s13_c1 背景，单/双位点扫描三个机制位点。

目的：v2 gate 在 n=6 设计上 in-sample 拟合，且 F88R+Y217R+M255K 总是共现 → 混杂。
      本扫描在固定背景下分离各位点效应，得到可泛化的 gate 校准数据。

设计：
  背景 = s13_c1 (P0 首选)
  扫描: F88(PDB88) × M255(PDB255)，Y217 固定 Y        -> 5 × 4 = 20 变体
        Y217(PDB217) ∈ {F,R}，F88=K, M255=M           -> 2 变体
  共 22 变体 × {S1_G17 靶, OFF_T_G17 非靶} = 44 预测
"""
import json
import os

WT = open("A:/claudework/ppri_evo/inputs/wt_254.fasta").read().splitlines()[1].strip()
assert len(WT) == 254
PLAN = json.load(open("A:/Data/设计蛋白/PprI_ssDNA_design/wetlab/order_plan_data.json",
                      encoding="utf-8"))["variants"]
DNA = {"S1_G17": "TCATGAGCAGTTTTTTGTTTTTTT",
       "OFF_T_G17": "TTGCTATTTTTTATTGCTTTGAGT"}
OUT = "A:/claudework/out/boltz_yamls_deconf"
os.makedirs(OUT, exist_ok=True)

BASE = PLAN["PprI_s13_c1"]["mutations"]


def apply_muts(seq, mutations):
    s = list(seq)
    for m in mutations:
        i = m["pdb"] - 22          # PDB -> 0-based seq_idx (实证: 1-based seq = PDB-21)
        assert s[i] == m["wt"], f"WT mismatch pdb{m['pdb']}: have {s[i]} want {m['wt']}"
        s[i] = m["mut"]
    return "".join(s)


def build(overrides):
    """overrides: {pdb: (wt_aa, mut_aa)} 覆盖 s13_c1 背景上的指定位点。"""
    muts = []
    seen = set()
    for m in BASE:                       # 背景突变
        if m["pdb"] in overrides:
            continue
        muts.append(dict(m))
        seen.add(m["pdb"])
    for pdb, (wt, mut) in sorted(overrides.items()):
        muts.append({"pdb": pdb, "wt": wt, "mut": mut, "module": "deconf_scan"})
    return muts


# s13_c1 背景在这三个位点的当前残基
base_state = {m["pdb"]: m["mut"] for m in BASE if m["pdb"] in (88, 217, 255)}
wt_state = {88: "F", 217: "Y", 255: "M"}
print("s13_c1 背景机制位点:", base_state)

variants = {}
# 主因子: F88 × M255 (Y217 = Y)
for f in "FKRWY":
    for mm in "MIKA":
        ov = {88: ("F", f), 255: ("M", mm), 217: ("Y", "Y")}
        variants[f"d_F88{f}_M255{mm}_Y217Y"] = ov
# Y217 单独扫描 (F88=K, M255=M)
for y in "FR":
    ov = {88: ("F", "K"), 255: ("M", "M"), 217: ("Y", y)}
    variants[f"d_F88K_M255M_Y217{y}"] = ov

n = 0
for vname, ov in variants.items():
    muts = build(ov)
    prot = apply_muts(WT, muts)
    assert len(prot) == 254
    # 校验位点确实生效
    for pdb, (wt, mut) in ov.items():
        assert prot[pdb - 22] == mut, (vname, pdb, prot[pdb - 22], mut)
    for dname, dna in DNA.items():
        y = (f"version: 1\nsequences:\n"
             f"  - protein:\n      id: A\n      sequence: {prot}\n"
             f"  - dna:\n      id: B\n      sequence: {dna}\n"
             f"  - ligand:\n      id: C\n      ccd: MN\n")
        with open(os.path.join(OUT, f"{vname}_{dname}.yaml"), "w") as f:
            f.write(y)
        n += 1

# 记录变体 -> 位点状态映射（供后续判读聚合）
meta = {v: {"F88": ov[88][1], "M255": ov[255][1], "Y217": ov[217][1]} for v, ov in variants.items()}
with open(os.path.join(OUT, "_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print(f"generated {n} yamls ({len(variants)} variants) -> {OUT}")
print("F88:", sorted({m['F88'] for m in meta.values()}),
      "M255:", sorted({m['M255'] for m in meta.values()}),
      "Y217:", sorted({m['Y217'] for m in meta.values()}))
