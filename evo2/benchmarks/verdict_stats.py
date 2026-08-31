# -*- coding: utf-8 -*-
"""逐候选统计（20 models）：均值 ± SEM + Wilson 区间，检验判别是否显著。"""
import csv
import math
from collections import defaultdict

SRC = r"A:/claudework/out/boltz_six/contact_fingerprint.csv"
rows = list(csv.DictReader(open(SRC)))

NUM = ["act", "dual", "lockA", "lockB", "iface", "ntcov", "d_act", "d_read17",
       "d_253_23", "ligand_iptm", "conf"]
byc = defaultdict(list)
for r in rows:
    byc[(r["cand"], r["cond"])].append(r)

def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd, sd / math.sqrt(n), n

# Wilson interval for a rate
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

cands = sorted({c for c, _ in byc})
print(f"{'cand':<11} {'cond':<4} | {'act(rate)':>16} | {'dual(rate)':>16} | "
      f"{'iface':>13} | {'ntcov':>12} | {'ligIPTM':>13}")
print("-" * 110)
summary = {}
for c in cands:
    for cond in ("S1", "OFF"):
        rs = byc[(c, cond)]
        n = len(rs)
        k_act = sum(int(r["act"]) for r in rs)
        k_dual = sum(int(r["dual"]) for r in rs)
        a_lo, a_hi = wilson(k_act, n)
        d_lo, d_hi = wilson(k_dual, n)
        if_m, if_sd, if_sem, _ = stats([float(r["iface"]) for r in rs])
        nt_m, _, nt_sem, _ = stats([float(r["ntcov"]) for r in rs])
        li_m, _, li_sem, _ = stats([float(r["ligand_iptm"]) for r in rs])
        print(f"{c:<11} {cond:<4} | {k_act/n:>5.2f} [{a_lo:.2f},{a_hi:.2f}] | "
              f"{k_dual/n:>5.2f} [{d_lo:.2f},{d_hi:.2f}] | "
              f"{if_m:>6.1f}±{if_sem:<5.1f} | {nt_m:>5.1f}±{nt_sem:<5.1f} | {li_m:.4f}±{li_sem:.4f}")
        summary[(c, cond)] = dict(n=n, act=k_act / n, act_lo=a_lo, act_hi=a_hi,
                                  dual=k_dual / n, dual_lo=d_lo, dual_hi=d_hi,
                                  iface=if_m, iface_sem=if_sem,
                                  ntcov=nt_m, ntcov_sem=nt_sem,
                                  ligiptm=li_m)
    print()

print("=== 判别 Δ = S1 − OFF（含 SEM 传播）===")
print(f"{'cand':<11} | {'Δact':>14} | {'Δdual':>14} | {'Δiface':>16} | {'Δntcov':>14} | 显著?")
print("-" * 95)
final = []
for c in cands:
    s, o = summary[(c, "S1")], summary[(c, "OFF")]
    d_act = s["act"] - o["act"]
    d_dual = s["dual"] - o["dual"]
    d_if = s["iface"] - o["iface"]
    d_if_sem = math.hypot(s["iface_sem"], o["iface_sem"])
    d_nt = s["ntcov"] - o["ntcov"]
    d_nt_sem = math.hypot(s["ntcov_sem"], o["ntcov_sem"])
    # 判据: Δiface 的 |z| > 2 视为显著
    z_if = d_if / d_if_sem if d_if_sem > 0 else 0.0
    z_nt = d_nt / d_nt_sem if d_nt_sem > 0 else 0.0
    sig = []
    if abs(z_if) > 2:
        sig.append(f"iface(z={z_if:+.1f})")
    if abs(z_nt) > 2:
        sig.append(f"ntcov(z={z_nt:+.1f})")
    if s["act_lo"] > o["act_hi"]:
        sig.append("act(CI不重叠)")
    elif o["act_lo"] > s["act_hi"]:
        sig.append("act(反向显著)")
    print(f"{c:<11} | {d_act:>+6.2f}        | {d_dual:>+6.2f}        | "
          f"{d_if:>+6.1f}±{d_if_sem:<5.1f}  | {d_nt:>+5.1f}±{d_nt_sem:<5.1f} | {', '.join(sig) or '不显著'}")
    final.append(dict(cand=c, d_act=d_act, d_dual=d_dual, d_iface=d_if, d_iface_sem=d_if_sem,
                      z_iface=z_if, d_ntcov=d_nt, z_ntcov=z_nt,
                      act_S1=s["act"], act_OFF=o["act"], dual_S1=s["dual"],
                      iface_S1=s["iface"], iface_OFF=o["iface"],
                      ntcov_S1=s["ntcov"], ntcov_OFF=o["ntcov"]))

with open(r"A:/claudework/out/boltz_six/boltz_six_stats.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(final[0].keys()))
    w.writeheader()
    for r in final:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print("\nWROTE boltz_six_stats.csv")
