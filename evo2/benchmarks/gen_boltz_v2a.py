#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 V2-A 候选（TrackF_r1+M255I / HQL2+M255I）× {S1_G17 靶, OFF_T_G17 非靶} 的 Boltz-2 yaml。

V2-A = 亲本原始设计突变（order_plan_data.json，不做任何工具精修）
       + 单点 M255I（去混杂实测的双锁放大器，refine_six_wetlab DNA-aware 自主命中）。
注意：不加 R82V/R265V/R220V 等 z 驱动壳层突变 —— 那是 V2-B，需另行验证。
"""
import json
import os

WT = open("A:/claudework/ppri_evo/inputs/wt_254.fasta").read().splitlines()[1].strip()
assert len(WT) == 254, len(WT)
CAND = json.load(open("A:/Data/设计蛋白/PprI_ssDNA_design/wetlab/order_plan_data.json"))["variants"]
DNA = {"S1_G17": "TCATGAGCAGTTTTTTGTTTTTTT",
       "OFF_T_G17": "TTGCTATTTTTTATTGCTTTGAGT"}
OUT = "A:/claudework/out/boltz_yamls_v2a"
os.makedirs(OUT, exist_ok=True)

M255I = {"pdb": 255, "wt": "M", "mut": "I", "module": "核心2 (T23 锁: R253/Y217/M255/P256)"}

V2A = {}
for parent in ("PprI_TrackF_r1", "PprI_HQL2"):
    muts = [dict(m) for m in CAND[parent]["mutations"]]
    assert not any(m["pdb"] == 255 for m in muts), f"{parent} 已有 255 位突变，不能直接加 M255I"
    muts.append(dict(M255I))
    V2A[f"{parent}_M255I"] = muts


def apply_muts(seq, mutations):
    s = list(seq)
    for m in mutations:
        i = m["pdb"] - 22          # PDB -> seq_idx
        assert s[i] == m["wt"], f"WT mismatch pdb{m['pdb']}: have {s[i]} want {m['wt']}"
        s[i] = m["mut"]
    return "".join(s)


n = 0
for cname, muts in V2A.items():
    prot = apply_muts(WT, muts)
    assert len(prot) == 254, (cname, len(prot))
    print(f"[{cname}] n_mut={len(muts)}  M255={'I' if prot[255-22] == 'I' else prot[255-22]}")
    for dname, dna in DNA.items():
        yaml = (f"version: 1\nsequences:\n"
                f"  - protein:\n      id: A\n      sequence: {prot}\n"
                f"  - dna:\n      id: B\n      sequence: {dna}\n"
                f"  - ligand:\n      id: C\n      ccd: MN\n")
        with open(os.path.join(OUT, f"{cname}_{dname}.yaml"), "w") as f:
            f.write(yaml)
        n += 1
print(f"generated {n} yamls -> {OUT}")
for fn in sorted(os.listdir(OUT)):
    print(" ", fn)
