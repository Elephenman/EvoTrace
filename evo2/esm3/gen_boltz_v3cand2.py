# -*- coding: utf-8 -*-
"""Boltz 第二轮组合验证 yaml：K216R+Q120V+F88Y_Y170F（推荐组合）+ K216R+Q120V（双最优单点）× {S1, OFF}。"""
import os
AA = "ACDEFGHIKLMNPQRSTVWY"
wt = open("A:/claudework/ppri_evo/inputs/wt_254.fasta").read().splitlines()[1].strip()
OUT = "A:/claudework/out/boltz_yamls_v3cand2"
os.makedirs(OUT, exist_ok=True)

def mut(seq, pdb, aa):
    i = pdb - 22
    assert seq[i] != aa and seq[i] in AA, (pdb, seq[i], aa)
    return seq[:i] + aa + seq[i+1:]

CANDS = {
    "K216R_Q120V": mut(mut(wt, 216, "R"), 120, "V"),
    "K216R_Q120V_F88Y_Y170F": mut(mut(mut(mut(wt, 216, "R"), 120, "V"), 88, "Y"), 170, "F"),
}
DNA = {"S1_G17": "TCATGAGCAGTTTTTTGTTTTTTT",
       "OFF_T_G17": "TTGCTATTTTTTATTGCTTTGAGT"}
n = 0
for cname, prot in CANDS.items():
    assert len(prot) == len(wt), cname
    for dname, dna in DNA.items():
        with open(os.path.join(OUT, f"{cname}_{dname}.yaml"), "w") as f:
            f.write(f"""version: 1
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
""")
        n += 1
print(f"generated {n} yamls -> {OUT}")
# 验证位点
for pdb, aa in [(216,"R"),(120,"V"),(88,"Y"),(170,"F")]:
    print(f"  PDB{pdb}->seq{pdb-22}={wt[pdb-22]} -> {aa}")
