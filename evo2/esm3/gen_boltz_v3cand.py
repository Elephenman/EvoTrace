# -*- coding: utf-8 -*-
"""生成 Boltz-2 验证 yaml：新候选 × {S1_G17 靶, OFF_T_G17 非靶}。
序列: WT / K216R / Y170F / Q120V / K216R+Y170F+Q120V / F88Y_Y170F / R85A_R207A_R267A(负对照)
"""
import os
AA = "ACDEFGHIKLMNPQRSTVWY"
wt = open("A:/claudework/ppri_evo/inputs/wt_254.fasta").read().splitlines()[1].strip()
OUT = "A:/claudework/out/boltz_yamls_v3cand"
os.makedirs(OUT, exist_ok=True)

def mut(seq, pdb, aa):
    """PDB 残基号 → seq_idx = PDB-22。"""
    i = pdb - 22
    assert seq[i] != aa and seq[i] in AA, (pdb, seq[i], aa)
    return seq[:i] + aa + seq[i+1:]

CANDS = {
    "WT": wt,
    "K216R": mut(wt, 216, "R"),
    "Y170F": mut(wt, 170, "F"),
    "Q120V": mut(wt, 120, "V"),
    "K216R_Y170F_Q120V": mut(mut(mut(wt, 216, "R"), 170, "F"), 120, "V"),
    "F88Y_Y170F": mut(mut(wt, 88, "Y"), 170, "F"),
    "R85A_R207A_R267A": mut(mut(mut(wt, 85, "A"), 207, "A"), 267, "A"),
}
DNA = {"S1_G17": "TCATGAGCAGTTTTTTGTTTTTTT",
       "OFF_T_G17": "TTGCTATTTTTTATTGCTTTGAGT"}

# 验证位点映射
for pdb in [216, 170, 120, 88, 85, 207, 267]:
    i = pdb - 22
    print(f"  PDB{pdb} -> seq{i} = {wt[i]}")

n = 0
for cname, prot in CANDS.items():
    assert len(prot) == len(wt), cname
    for dname, dna in DNA.items():
        yaml = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {prot}
  - dna:
      id: B
      sequence: {dna}
  - ligand:
      id: C
      ccd: MN
"""
        with open(os.path.join(OUT, f"{cname}_{dname}.yaml"), "w") as f:
            f.write(yaml)
        n += 1
print(f"generated {n} yamls -> {OUT}")
