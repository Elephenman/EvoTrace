# -*- coding: utf-8 -*-
"""nom2 全量 80 系统 x 100 模型 接触指纹判读。

口径与 n100 战役完全一致：复用 ppri_evo_boltz_n100/scan_n100.py 的
parse_cif / Grid / min_dist / HEXXH / READHEAD / LOCK_* / ANCHORS / E123 / CUT。
输出 schema 与 contact_fingerprint_n100.csv 完全相同（20 列）。
"""
import sys, glob, os, csv, json
sys.path.insert(0, "/eaas/default/groups/public_cluster/home/u22607007/ppri_evo_boltz_n100")
from scan_n100 import (parse_cif, Grid, min_dist, HEXXH, READHEAD, LOCK_R253,
                       LOCK_Y217, LOCK_M255, ANCHORS, E123, CUT)
import math

BASE_G = "/home/u22607007/ppri_evo_boltz_nom2/out_nom2_full/boltz_results_*/predictions/*"
OUT = "/home/u22607007/ppri_evo_boltz_nom2/contact_fingerprint_nom2.csv"

rows = []
dirs = sorted(glob.glob(BASE_G))
print("targets:", len(dirs), flush=True)
for d in dirs:
    name = os.path.basename(d)
    cifs = sorted(glob.glob(os.path.join(d, "*.cif")))
    for cif in cifs:
        model = os.path.basename(cif).rsplit("_model_", 1)[-1].replace(".cif", "")
        conf = {}
        cj = glob.glob(os.path.join(d, f"confidence_*_model_{model}.json"))
        if cj:
            try:
                conf = json.load(open(cj[0]))
            except Exception:
                conf = {}
        prot, dna = parse_cif(cif)
        if not prot or not dna:
            continue
        grid = Grid(dna)
        all_dna = [(nt, x, y, z) for nt, at in dna.items() for (x, y, z) in at]
        hx = [a for s in HEXXH for a in prot.get(s, [])]
        d_act = min_dist(hx, all_dna) if hx else 99.0

        def sd(resi, nt):
            if resi not in prot or nt not in dna:
                return 99.0
            return min_dist(prot[resi], [(nt, x, y, z) for (x, y, z) in dna[nt]])

        def sany(resi):
            return min_dist(prot.get(resi, []), all_dna) if resi in prot else 99.0

        iface_res, covered = set(), set()
        for resi, atoms in prot.items():
            for ax, ay, az in atoms:
                for nt, bx, by, bz in grid.near(ax, ay, az):
                    if (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2 <= CUT * CUT:
                        iface_res.add(resi)
                        covered.add(nt)
                        break
        d_read17 = sd(READHEAD, 17)
        d_253_23 = sd(LOCK_R253, 23)
        rows.append(dict(
            pred=name, model=model,
            conf=round(conf.get("confidence_score", float("nan")), 4),
            iptm=round(conf.get("iptm", float("nan")), 4),
            ligand_iptm=round(conf.get("ligand_iptm", float("nan")), 4),
            act=int(d_act <= CUT), d_act=round(d_act, 2),
            lockA=int(d_read17 <= CUT), d_read17=round(d_read17, 2),
            lockB=int(d_253_23 <= CUT), d_253_23=round(d_253_23, 2),
            dual=int(d_read17 <= CUT and d_253_23 <= CUT),
            d_y217_23=round(sd(LOCK_Y217, 23), 2), d_m255_23=round(sd(LOCK_M255, 23), 2),
            d_r85=round(sany(ANCHORS[0]), 2), d_r207=round(sany(ANCHORS[1]), 2),
            d_r267=round(sany(ANCHORS[2]), 2), d_e123=round(sany(E123), 2),
            iface=len(iface_res), ntcov=len(covered)))
    print("[ok]", name, len(cifs), flush=True)

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("WROTE", OUT, "rows=", len(rows), flush=True)
