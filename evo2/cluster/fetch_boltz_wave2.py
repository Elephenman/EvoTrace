#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CHPC 上计算 wave-2 Boltz 判决并取回 CSV。

用法: CHPC_PASS='...' python fetch_boltz_wave2.py
在集群登录节点用系统 python3 跑 metrics（纯 math/glob），避免传 540 个 CIF。
"""
import os
import sys

import paramiko

HERE = os.path.dirname(os.path.abspath(__file__))
HOST, PORT, USER = "10.202.94.52", 20009, "u22607007"

REMOTE_SCRIPT = r'''
import csv, glob, math, os
from collections import defaultdict

SEEDS = ["out_wave2_s1", "out_wave2_s2", "out_wave2_s3"]
CONDS = ["S1_G17", "OFF_T_G17", "GCA"]
HEXXH = set(range(71, 76))
K67, R232, G17, T23 = 67, 232, 17, 23

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) ** 0.5

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

def metrics(path):
    ch = parse_cif(path)
    prot, dna = ch.get("A", {}), ch.get("B", {})
    if not prot or not dna:
        return None
    ha = [v for s in HEXXH if s in prot for v in prot[s].values()]
    da = [v for r in dna.values() for v in r.values()]
    hmin = min((dist(a, b) for a in ha for b in da), default=999)
    d17 = min((dist(a, b) for a in prot[K67].values() for b in dna[G17].values())) if (K67 in prot and G17 in dna) else 999
    d23 = min((dist(a, b) for a in prot[R232].values() for b in dna[T23].values())) if (R232 in prot and T23 in dna) else 999
    return dict(act=hmin <= 5.0, dual=(d17 <= 5 and d23 <= 5))

def rate(models, key):
    return round(100 * sum(1 for m in models if m[key]) / len(models), 1) if models else None

agg = defaultdict(lambda: defaultdict(list))
per_seed = defaultdict(lambda: defaultdict(dict))
for sdir in SEEDS:
    seed = sdir[-1]
    for cond in CONDS:
        for d in glob.glob(os.path.join(os.path.expanduser("~/evo2_boltz"), sdir,
                                        "boltz_results_wave2_yamls", "predictions", f"*_{cond}")):
            name = os.path.basename(d)[: -len(cond) - 1]
            models = [m for m in (metrics(p) for p in glob.glob(os.path.join(d, "*.cif"))) if m]
            agg[name][cond].extend(models)
            per_seed[name][cond][seed] = models

rows = []
for name in sorted(agg):
    by = agg[name]
    a1, ao, ag = rate(by["S1_G17"], "act"), rate(by["OFF_T_G17"], "act"), rate(by["GCA"], "act")
    dl1 = rate(by["S1_G17"], "dual")
    def seed_std(cond):
        rates = [rate(per_seed[name][cond].get(s, []), "act") for s in "123"]
        rates = [r for r in rates if r is not None]
        if len(rates) < 2:
            return None
        mu = sum(rates) / len(rates)
        return round((sum((r - mu) ** 2 for r in rates) / (len(rates) - 1)) ** 0.5, 1)
    rows.append(dict(elite=name,
                     n=len(by["S1_G17"]) + len(by["OFF_T_G17"]) + len(by["GCA"]),
                     act_s1=a1, act_s1_std=seed_std("S1_G17"),
                     act_off=ao, act_off_std=seed_std("OFF_T_G17"),
                     act_gca=ag, act_gca_std=seed_std("GCA"),
                     sep=(round(a1 - ao, 1) if None not in (a1, ao) else None),
                     gca_delta=(round(a1 - ag, 1) if None not in (a1, ag) else None),
                     dual_s1=dl1))
with open(os.path.expanduser("~/evo2_boltz/wave2_verdict.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("WROTE", len(rows), "rows")
'''


def run(c, cmd, timeout=600):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


def main():
    pw = os.environ.get("CHPC_PASS")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=pw, timeout=25,
              look_for_keys=False, allow_agent=False)
    b64 = __import__("base64").b64encode(REMOTE_SCRIPT.encode()).decode()
    run(c, "cat > /tmp/w2_metrics.b64 << 'EOB'\n" + b64 + "\nEOB")
    run(c, "base64 -d /tmp/w2_metrics.b64 > ~/evo2_boltz/wave2_metrics.py")
    o, e = run(c, "python3 ~/evo2_boltz/wave2_metrics.py", timeout=900)
    print(o.strip(), e.strip()[:300])
    sftp = c.open_sftp()
    try:
        sftp.get("/public/home/{}/evo2_boltz/wave2_verdict.csv".format(USER),
                 os.path.join(HERE, "..", "results", "b5_wave2_verdict.csv"))
        print("[fetched] b5_wave2_verdict.csv")
    except Exception as ex:
        print("[sftp failed, fallback base64]", ex)
        o, e = run(c, "base64 ~/evo2_boltz/wave2_verdict.csv")
        import base64
        data = base64.b64decode(o)
        open(os.path.join(HERE, "..", "results", "b5_wave2_verdict.csv"), "wb").write(data)
        print("[fetched via base64]")
    c.close()


if __name__ == "__main__":
    main()
