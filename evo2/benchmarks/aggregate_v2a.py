# -*- coding: utf-8 -*-
"""V2-A 判读聚合：TrackF_r1+M255I / HQL2+M255I vs 亲本（job 225685）配对比较。

口径与 verdict_stats.py 完全一致（Wilson 95% CI + Δ 的 SEM 传播 + |z|>2 显著性），
另加 亲本→V2-A 的两比例 z 检验（S1 激活率 / S1 双锁率 是否因 M255I 改变）。
"""
import csv
import math
from collections import defaultdict

SRC = r"A:/claudework/out/boltz_v2a_contact_fingerprint.csv"
PARENT_STATS = r"A:/claudework/evo2/results/boltz_six/boltz_six_stats.csv"
OUT = r"A:/claudework/evo2/results/boltz_v2a/v2a_vs_parent_stats.csv"

rows = list(csv.DictReader(open(SRC)))
byc = defaultdict(list)
for r in rows:
    byc[(r["cand"], r["cond"])].append(r)


def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def prop_z(k1, n1, k2, n2):
    """两比例 z 检验（亲本 k2/n2 → 子代 k1/n1）。"""
    if min(n1, n2) == 0:
        return 0.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se > 0 else 0.0


summary = {}
for (c, cond), rs in byc.items():
    n = len(rs)
    k_act = sum(int(r["act"]) for r in rs)
    k_dual = sum(int(r["dual"]) for r in rs)
    a_lo, a_hi = wilson(k_act, n)
    d_lo, d_hi = wilson(k_dual, n)
    if_m, if_sem = stats([float(r["iface"]) for r in rs])
    nt_m, nt_sem = stats([float(r["ntcov"]) for r in rs])
    li_m, _ = stats([float(r["ligand_iptm"]) for r in rs])
    dm, _ = stats([float(r["d_m255_23"]) for r in rs])
    summary[(c, cond)] = dict(n=n, act=k_act / n, act_k=k_act, act_lo=a_lo, act_hi=a_hi,
                              dual=k_dual / n, dual_k=k_dual, dual_lo=d_lo, dual_hi=d_hi,
                              iface=if_m, iface_sem=if_sem, ntcov=nt_m, ntcov_sem=nt_sem,
                              ligiptm=li_m, d_m255_23=dm)

# 亲本判读（job 225685 同参数）
parent = {r["cand"]: r for r in csv.DictReader(open(PARENT_STATS))}
PAIRS = [("TrackF_r1_M255I", "TrackF_r1"), ("HQL2_M255I", "HQL2")]

print("=" * 118)
print("V2-A（M255I）vs 亲本 —— S1 条件逐指标（Wilson 95% CI + 两比例 z 检验）")
print("=" * 118)
for child, par in PAIRS:
    s, o = summary[(child, "S1")], summary[(child, "OFF")]
    ps, po = float(parent[par]["act_S1"]), float(parent[par]["act_OFF"])
    pd_, pdual = float(parent[par]["dual_S1"]), 0.0
    p_if_s, p_if_o = float(parent[par]["iface_S1"]), float(parent[par]["iface_OFF"])
    print(f"\n[{child} vs {par}]")
    print(f"  act(S1)  : {par}={ps:.2f} -> {s['act']:.2f} [{s['act_lo']:.2f},{s['act_hi']:.2f}]"
          f"   Δact(S1−OFF): {float(parent[par]['d_act']):+.2f} -> {s['act'] - o['act']:+.2f}")
    print(f"  dual(S1) : {par}={pd_:.2f} -> {s['dual']:.2f} [{s['dual_lo']:.2f},{s['dual_hi']:.2f}]"
          f"   dual(OFF): {par}={pdual:.2f} -> {o['dual']:.2f}")
    print(f"  iface    : S1 {p_if_s:.1f} -> {s['iface']:.1f}±{s['iface_sem']:.1f} | "
          f"OFF {p_if_o:.1f} -> {o['iface']:.1f}±{o['iface_sem']:.1f} | "
          f"Δiface {float(parent[par]['d_iface']):+.1f} -> {s['iface'] - o['iface']:+.1f}")
    print(f"  d_m255_23(S1) 均值: {s['d_m255_23']:.2f} Å  ligIPTM(S1): {s['ligiptm']:.4f}")

print("\n" + "=" * 118)
print("两比例 z 检验（M255I 效应，|z|>2 显著）")
print("=" * 118)
# 亲本 20 models/条件，从 fingerprint 重取 k 数：用六候选原始 fingerprint
six = defaultdict(list)
for r in csv.DictReader(open(r"A:/claudework/out/boltz_six/contact_fingerprint.csv")):
    six[(r["cand"], r["cond"])].append(r)
for child, par in PAIRS:
    for metric in ("act", "dual"):
        rs_par = six[(par, "S1")]
        k2, n2 = sum(int(r[metric]) for r in rs_par), len(rs_par)
        rs_ch = byc[(child, "S1")]
        k1, n1 = sum(int(r[metric]) for r in rs_ch), len(rs_ch)
        z = prop_z(k1, n1, k2, n2)
        flag = " *" if abs(z) > 2 else ""
        print(f"  {par} -> {child}  {metric}(S1): {k2}/{n2}={k2/n2:.2f} -> "
              f"{k1}/{n1}={k1/n1:.2f}  z={z:+.2f}{flag}")

# 落盘
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
final = []
for child, par in PAIRS:
    s, o = summary[(child, "S1")], summary[(child, "OFF")]
    p = parent[par]
    d_if = s["iface"] - o["iface"]
    d_if_sem = math.hypot(s["iface_sem"], o["iface_sem"])
    final.append(dict(
        cand=child, parent=par,
        act_S1=round(s["act"], 3), act_S1_lo=round(s["act_lo"], 3), act_S1_hi=round(s["act_hi"], 3),
        act_OFF=round(o["act"], 3), d_act=round(s["act"] - o["act"], 3),
        parent_d_act=round(float(p["d_act"]), 3),
        dual_S1=round(s["dual"], 3), dual_S1_lo=round(s["dual_lo"], 3), dual_S1_hi=round(s["dual_hi"], 3),
        parent_dual_S1=round(float(p["dual_S1"]), 3),
        iface_S1=round(s["iface"], 1), iface_OFF=round(o["iface"], 1),
        d_iface=round(d_if, 1), d_iface_sem=round(d_if_sem, 1),
        z_iface=round(d_if / d_if_sem if d_if_sem else 0, 2),
        parent_d_iface=round(float(p["d_iface"]), 1),
        ntcov_S1=round(s["ntcov"], 1), ntcov_OFF=round(o["ntcov"], 1),
        d_m255_23_S1=round(s["d_m255_23"], 2),
        ligiptm_S1=round(s["ligiptm"], 4),
    ))
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(final[0].keys()))
    w.writeheader()
    w.writerows(final)
print(f"\nWROTE {OUT}")
