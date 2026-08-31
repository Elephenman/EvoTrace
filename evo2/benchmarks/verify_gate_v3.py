# -*- coding: utf-8 -*-
"""验证 gate v3 已写入模块、默认启用，并在去混杂集上对照 v1/v2/v3。"""
import csv, sys, math
sys.path.insert(0, r"A:/claudework/evo2/esm3")
from ppri_dna_aware import DnaAwareLandscape, AA

S = list(csv.DictReader(open(r"A:/claudework/out/boltz_six/deconf_stats.csv", encoding="utf-8")))
for s in S:
    for k in ("dual_S1", "d_iface", "d_act", "z_iface", "z_act"):
        s[k] = float(s[k])

PRIORS = list(csv.DictReader(open(r"A:/claudework/ppri_evo/results/priors.csv", encoding="utf-8")))
SITES = [int(r["seq_idx"]) for r in PRIORS]
WT = open(r"A:/claudework/ppri_evo/inputs/wt_254.fasta").read().splitlines()[1].strip()
WT_IDX = [AA.index(WT[s]) for s in SITES]
COL = {s: i for i, s in enumerate(SITES)}
PDB2COL = {88: COL.get(66), 217: COL.get(195), 255: COL.get(233),
           85: COL.get(63), 207: COL.get(186), 267: COL.get(245)}
BG_MUT = {88: 'K', 217: 'Y', 255: 'I', 85: 'G', 207: 'K', 267: 'R'}


class Stub:
    L = 53
    def __init__(s, si, wi):
        s.sites, s.wt_idx = si, wi
    def evaluate(s, g):
        return [0.0] * len(g)


stub = Stub(SITES, WT_IDX)
lands = {v: DnaAwareLandscape(stub, gate_version=v) for v in ("v1", "v2", "v3")}
print("默认 gate_version =", DnaAwareLandscape(stub).gate_version)
print()

# s13_c1 背景 + 扫描位点
genos = {}
for s in S:
    g = list(WT_IDX)
    for pdb, c in PDB2COL.items():
        if c is None:
            continue
        aa = {'88': s["F88"], '255': s["M255"], '217': s["Y217"]}.get(str(pdb), BG_MUT[pdb])
        g[c] = AA.index(aa)
    genos[s["variant"]] = g

def rank_norm(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(o):
        r[i] = k / (len(v) - 1)
    return r


def spear(a, b):
    ra, rb = rank_norm(list(a)), rank_norm(list(b))
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    n = sum((ra[i] - ma) * (rb[i] - mb) for i in range(len(ra)))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((x - mb) ** 2 for x in rb))
    return n / (da * db) if da > 0 and db > 0 else 0.0


lock = [s["dual_S1"] for s in S]
disc = [s["z_act"] for s in S]
ifac = [s["z_iface"] for s in S]
rl, rd, ri = rank_norm(lock), rank_norm(disc), rank_norm(ifac)
comp = [0.40 * rl[i] + 0.30 * rd[i] + 0.30 * ri[i] for i in range(len(S))]

print(f"{'gate':<8}{'vs composite':>14}{'vs dual_S1':>13}{'vs Δact':>10}{'vs Δiface':>11}")
for v in ("v1", "v2", "v3"):
    g = [lands[v]._gate(genos[s["variant"]]) for s in S]
    print(f"{v:<8}{spear(g, comp):>14.3f}{spear(g, lock):>13.3f}"
          f"{spear(g, disc):>10.3f}{spear(g, ifac):>11.3f}")
print()

# v3 排名 top / bottom
sc = [(s["variant"], lands["v3"]._gate(genos[s["variant"]]), comp[i])
      for i, s in enumerate(S)]
sc.sort(key=lambda x: -x[1])
print("v3 gate 排名（前 6 / 后 3）— 对照实测 composite：")
for name, gv, cp in sc[:6] + sc[-3:]:
    m = [x for x in S if x["variant"] == name][0]
    print(f"  {name:<20} gate={gv:+.3f}  composite={cp:.3f}  "
          f"(dual={m['dual_S1']:.2f} Δiface={m['d_iface']:+.1f} Δact={m['d_act']:+.2f})")
print()
print("s13_c1 (=F88K_M255I_Y217Y) gate v3 =",
      f"{lands['v3']._gate(genos['F88K_M255I_Y217Y']):+.3f}")
