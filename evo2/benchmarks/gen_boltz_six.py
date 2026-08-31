#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成六候选 × {24nt S1_G17 靶, 24nt OFF_T_G17 非靶} 的 Boltz-2 验证 yaml。
使用 order_plan_data.json 的原始设计突变（不做工具「打磨」—— 那是零样本回退假象）。
"""
import json
import os

WT = open("A:/claudework/ppri_evo/inputs/wt_254.fasta").read().splitlines()[1].strip()
assert len(WT) == 254, len(WT)
CAND = json.load(open("A:/Data/设计蛋白/PprI_ssDNA_design/wetlab/order_plan_data.json"))["variants"]
DNA = {"S1_G17": "TCATGAGCAGTTTTTTGTTTTTTT",
       "OFF_T_G17": "TTGCTATTTTTTATTGCTTTGAGT"}
OUT = "A:/claudework/out/boltz_yamls_six"
os.makedirs(OUT, exist_ok=True)


def apply_muts(seq, mutations):
    s = list(seq)
    for m in mutations:
        i = m["pdb"] - 22          # PDB -> seq_idx
        assert s[i] == m["wt"], f"WT mismatch pdb{m['pdb']}: have {s[i]} want {m['wt']}"
        s[i] = m["mut"]
    return "".join(s)


n = 0
for cname, v in CAND.items():
    prot = apply_muts(WT, v["mutations"])
    assert len(prot) == 254, (cname, len(prot))
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
