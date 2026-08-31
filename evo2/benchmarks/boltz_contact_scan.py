#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Boltz 六候选 x 24nt 靶标 接触指纹判读（集群端纯 Python 执行）。

编号基准（实证）：chain A = 蛋白 254aa 连续 1..254；chain B = DNA 24nt 1..24。
  seq = PDB - 21。HEXXH motif 实测位于 seq 71-75 (H71 E72 I73 S74 H75)。

机制位点（seq, 1-based）：
  HEXXH  : 71,72,73,74,75          催化激活（接触 DNA = 激活）
  读头   : 67  (= PDB F88, 设计中可为 K/Y/L/W/Q)
  双锁   : 232 (= PDB R253), 196 (= Y217), 234 (= M255)
  锚点   : 64  (= R85), 186 (= R207), 246 (= R267)
  E123   : 102 (= PDB E123, 三齿配位)

判读定义（沿用铁律，改为数据驱动）：
  act     = HEXXH(71-75) 任一原子距任一 DNA 原子 <= CUT
  lockA   = res67 距 nt17 <= CUT   (读头锁定 dG17)
  lockB   = res232 距 nt23 <= CUT  (R253 锁定 nt23)
  dual    = lockA 且 lockB
  iface   = 蛋白中距 DNA <= CUT 的残基数（结合界面规模）
  ntcov   = 被蛋白接触的 DNA 碱基数
输出：逐 model CSV，供聚合判别 (S1 - OFF)。
"""
import csv
import glob
import json
import math
import os
import sys

BASE = "/home/u22607007/ppri_evo_boltz_six/out_six/boltz_results_yamls/predictions"
OUT = "/home/u22607007/ppri_evo_boltz_six/contact_fingerprint.csv"
CUT = 5.0

# mmCIF atom_site columns (0-based after split)
C_ATOM, C_COMP, C_SEQ, C_ASYM, C_X = 3, 5, 6, 9, 10

HEXXH = (71, 72, 73, 74, 75)
READHEAD = 67
LOCK_R253, LOCK_Y217, LOCK_M255 = 232, 196, 234
ANCHORS = (64, 186, 246)
E123 = 102
SITES = sorted(set(HEXXH) | {READHEAD, LOCK_R253, LOCK_Y217, LOCK_M255, E123} | set(ANCHORS))


def parse_cif(path):
    """返回 (prot, dna): prot[resi] -> [(x,y,z)...], dna[nt] -> [(x,y,z)...]"""
    prot, dna = {}, {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            p = line.split()
            if len(p) < 14 or p[C_SEQ] == "?":
                continue
            try:
                resi = int(p[C_SEQ])
            except ValueError:
                continue
            xyz = (float(p[C_X]), float(p[C_X + 1]), float(p[C_X + 2]))
            comp = p[C_COMP]
            if comp in ("DA", "DC", "DG", "DT", "DU"):
                dna.setdefault(resi, []).append(xyz)
            else:
                prot.setdefault(resi, []).append(xyz)
    return prot, dna


class Grid:
    """DNA 原子空间网格，加速邻近查询。"""

    def __init__(self, dna, cell=5.0):
        self.cell = cell
        self.buckets = {}
        for nt, atoms in dna.items():
            for x, y, z in atoms:
                k = (int(math.floor(x / cell)), int(math.floor(y / cell)), int(math.floor(z / cell)))
                self.buckets.setdefault(k, []).append((nt, x, y, z))

    def near(self, x, y, z):
        c = self.cell
        i, j, k = int(math.floor(x / c)), int(math.floor(y / c)), int(math.floor(z / c))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    for item in self.buckets.get((i + di, j + dj, k + dk), ()):
                        yield item


def min_dist(atoms, query):
    """atoms: [(x,y,z)...] 到 query: 可迭代 (nt,x,y,z) 的最小距离。"""
    best = 1e9
    for ax, ay, az in atoms:
        for _nt, bx, by, bz in query:
            d = (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
            if d < best:
                best = d
    return math.sqrt(best)


def main():
    dirs = sorted(d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d)))
    rows = []
    for name in dirs:
        d = os.path.join(BASE, name)
        cifs = sorted(glob.glob(os.path.join(d, "*.cif")))
        for cif in cifs:
            model = os.path.basename(cif).rsplit("_model_", 1)[-1].replace(".cif", "")
            conf = {}
            cj = os.path.join(d, f"confidence_{name}_model_{model}.json")
            if os.path.exists(cj):
                try:
                    conf = json.load(open(cj))
                except Exception:
                    conf = {}
            prot, dna = parse_cif(cif)
            if not prot or not dna:
                continue
            grid = Grid(dna)
            all_dna = [(nt, x, y, z) for nt, at in dna.items() for (x, y, z) in at]
            # HEXXH 激活
            hexxh_atoms = [a for s in HEXXH for a in prot.get(s, [])]
            d_act = min_dist(hexxh_atoms, all_dna) if hexxh_atoms else 99.0
            # 位点-碱基特异距离
            def sd(resi, nt):
                if resi not in prot or nt not in dna:
                    return 99.0
                return min_dist(prot[resi], [(nt, x, y, z) for (x, y, z) in dna[nt]])
            d_read17 = sd(READHEAD, 17)
            d_253_23 = sd(LOCK_R253, 23)
            d_217_23 = sd(LOCK_Y217, 23)
            d_255_23 = sd(LOCK_M255, 23)
            # 位点到任意 DNA
            def sany(resi):
                return min_dist(prot.get(resi, []), all_dna) if resi in prot else 99.0
            # 界面规模 + 覆盖碱基数
            iface_res, covered = set(), set()
            for resi, atoms in prot.items():
                hit = False
                for ax, ay, az in atoms:
                    for nt, bx, by, bz in grid.near(ax, ay, az):
                        if (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2 <= CUT * CUT:
                            hit = True
                            covered.add(nt)
                if hit:
                    iface_res.add(resi)
            rows.append(dict(
                pred=name,
                cand=name.replace("PprI_", "").replace("_S1_G17", "").replace("_OFF_T_G17", ""),
                cond="S1" if "S1_G17" in name else ("OFF" if "OFF_T_G17" in name else "?"),
                model=model,
                conf=round(conf.get("confidence_score", float("nan")), 4),
                iptm=round(conf.get("iptm", float("nan")), 4),
                ligand_iptm=round(conf.get("ligand_iptm", float("nan")), 4),
                act=int(d_act <= CUT), d_act=round(d_act, 2),
                lockA=int(d_read17 <= CUT), d_read17=round(d_read17, 2),
                lockB=int(d_253_23 <= CUT), d_253_23=round(d_253_23, 2),
                dual=int(d_read17 <= CUT and d_253_23 <= CUT),
                d_y217_23=round(d_217_23, 2), d_m255_23=round(d_255_23, 2),
                d_r85=round(sany(64), 2), d_r207=round(sany(186), 2), d_r267=round(sany(246), 2),
                d_e123=round(sany(E123), 2),
                iface=len(iface_res), ntcov=len(covered),
            ))
        print(f"[ok] {name}: {len(cifs)} models", flush=True)

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWROTE {OUT} rows={len(rows)}")


if __name__ == "__main__":
    main()
