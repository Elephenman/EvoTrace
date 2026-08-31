#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3cand Boltz 判读：新候选验证（out_v3cand）。

判读铁律（沿用）：
  activation = HEXXH(seq71-75) 任一原子距 DNA ≤5Å 的模型比例
  duallock   = K67-G17≤5 且 R232-T23≤5 的模型比例
  判别 = act_S1 - act_OFF
用法: python analyze_v3cand_boltz.py [out_dir]
"""
import csv
import glob
import json
import math
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "A:/claudework/out/boltz_v3cand_out"
HEXXH = set(range(71, 76))
K67, R232 = 67, 232
G17, T23 = 17, 23


def parse_cif(path):
    chains = {}
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")):
            p = line.split()
            if len(p) < 13 or not p[6].isdigit():
                continue
            r = chains.setdefault(p[9], {}).setdefault(int(p[6]), {})
            r[p[3]] = (float(p[10]), float(p[11]), float(p[12]))
    return chains


def min_dist(resA, resB):
    return min(math.dist(a, b) for a in resA.values() for b in resB.values())


def model_metrics(path):
    ch = parse_cif(path)
    prot, dna = ch.get("A", {}), ch.get("B", {})
    if not prot or not dna:
        return None
    dna_atoms = [v for r in dna.values() for v in r.values()]
    hexxh = [v for s in HEXXH if s in prot for v in prot[s].values()]
    act = min(math.dist(a, b) for a in hexxh for b in dna_atoms) <= 5.0 if hexxh else False
    lock = False
    if K67 in prot and G17 in dna and R232 in prot and T23 in dna:
        lock = min_dist(prot[K67], dna[G17]) <= 5.0 and min_dist(prot[R232], dna[T23]) <= 5.0
    return dict(act=act, lock=lock)


def main():
    cands = {}
    for cif in sorted(glob.glob(os.path.join(OUT, "**", "*.cif"), recursive=True)):
        # 路径格式: .../predictions/<name>/.../sample_*.cif
        parts = cif.replace("\\", "/").split("/")
        try:
            name = parts[parts.index("predictions") + 1]
        except ValueError:
            continue
        m = model_metrics(cif)
        if m:
            cands.setdefault(name, []).append(m)
    if not cands:
        print("no cif found under", OUT)
        return
    print(f"{'candidate':<28} {'n':>3} {'act_S1':>7} {'act_OFF':>7} {'sep':>6} {'lockS1':>7}")
    rows = []
    for name in sorted(cands):
        ms = cands[name]
        n = len(ms)
        act = sum(1 for m in ms if m["act"]) / n
        lock = sum(1 for m in ms if m["lock"]) / n
        tag = "S1" if "S1" in name else ("OFF" if "OFF" in name else "?")
        rows.append((name, n, act, lock, tag))
    # 按候选聚合 S1/OFF
    agg = {}
    for name, n, act, lock, tag in rows:
        base = name.replace("_S1_G17", "").replace("_OFF_T_G17", "")
        agg.setdefault(base, {})[tag] = (act, lock, n)
    print("\n=== 按候选聚合（判别 = act_S1 − act_OFF）===")
    print(f"{'candidate':<26} {'act_S1':>7} {'act_OFF':>7} {'sep':>7} {'lock_S1':>7}")
    for base in sorted(agg):
        d = agg[base]
        s1 = d.get("S1", (0, 0, 0))
        off = d.get("OFF", (0, 0, 0))
        print(f"{base:<26} {s1[0]:>7.3f} {off[0]:>7.3f} {s1[0]-off[0]:>7.3f} {s1[1]:>7.3f}")


if __name__ == "__main__":
    main()
