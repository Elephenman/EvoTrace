# -*- coding: utf-8 -*-
"""聚合 Boltz 六候选 x 24nt 接触指纹 -> 靶标特异性排序。"""
import csv
import os
from collections import defaultdict

SRC = r"A:/claudework/out/boltz_six/contact_fingerprint.csv"
OUT = r"A:/claudework/out/boltz_six/boltz_six_verdict.csv"

FIELDS_FLOAT = ["conf", "iptm", "ligand_iptm", "act", "d_act", "lockA", "d_read17",
                "lockB", "d_253_23", "dual", "d_r85", "d_r207", "d_r267", "d_e123",
                "iface", "ntcov"]

rows = list(csv.DictReader(open(SRC)))
agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    key = (r["cand"], r["cond"])
    for f in FIELDS_FLOAT:
        try:
            agg[key][f].append(float(r[f]))
        except ValueError:
            pass

def m(v):
    return sum(v) / len(v) if v else float("nan")

cands = sorted({c for c, _ in agg})
rec = {}
for c in cands:
    rec[c] = {}
    for cond in ("S1", "OFF"):
        d = agg.get((c, cond), {})
        rec[c][cond] = {f: m(d.get(f, [])) for f in FIELDS_FLOAT}

print(f"{'cand':<12} {'n':>3} | {'actS1':>6} {'actOFF':>6} {'Δact':>6} | "
      f"{'dualS1':>6} {'dualOFF':>6} {'Δdual':>6} | {'ipTMS1':>7} {'ipTMOFF':>7} {'ΔipTM':>6} | "
      f"{'ifS1':>5} {'ifOFF':>5} {'Δif':>5} | {'ntS1':>5} {'ntOFF':>5} {'Δnt':>5}")
print("-" * 130)
out_rows = []
for c in cands:
    s, o = rec[c]["S1"], rec[c]["OFF"]
    def d_(k):
        return s[k] - o[k]
    print(f"{c:<12} {len(agg[(c,'S1')]['act']):>3} | "
          f"{s['act']:>6.2f} {o['act']:>6.2f} {d_('act'):>+6.2f} | "
          f"{s['dual']:>6.2f} {o['dual']:>6.2f} {d_('dual'):>+6.2f} | "
          f"{s['ligand_iptm']:>7.3f} {o['ligand_iptm']:>7.3f} {d_('ligand_iptm'):>+6.3f} | "
          f"{s['iface']:>5.1f} {o['iface']:>5.1f} {d_('iface'):>+5.1f} | "
          f"{s['ntcov']:>5.1f} {o['ntcov']:>5.1f} {d_('ntcov'):>+5.1f}")
    row = dict(cand=c,
               act_S1=round(s["act"], 3), act_OFF=round(o["act"], 3), sep_act=round(d_("act"), 3),
               dual_S1=round(s["dual"], 3), dual_OFF=round(o["dual"], 3), sep_dual=round(d_("dual"), 3),
               lockA_S1=round(s["lockA"], 3), lockA_OFF=round(o["lockA"], 3),
               lockB_S1=round(s["lockB"], 3), lockB_OFF=round(o["lockB"], 3),
               ligiptm_S1=round(s["ligand_iptm"], 4), ligiptm_OFF=round(o["ligand_iptm"], 4),
               sep_ligiptm=round(d_("ligand_iptm"), 4),
               conf_S1=round(s["conf"], 4), conf_OFF=round(o["conf"], 4),
               iface_S1=round(s["iface"], 1), iface_OFF=round(o["iface"], 1), sep_iface=round(d_("iface"), 1),
               ntcov_S1=round(s["ntcov"], 1), ntcov_OFF=round(o["ntcov"], 1), sep_ntcov=round(d_("ntcov"), 1),
               d_act_S1=round(s["d_act"], 2), d_act_OFF=round(o["d_act"], 2),
               d_read17_S1=round(s["d_read17"], 2), d_read17_OFF=round(o["d_read17"], 2),
               d_253_23_S1=round(s["d_253_23"], 2), d_253_23_OFF=round(o["d_253_23"], 2),
               )
    out_rows.append(row)

# 综合排序：主看 Δact（靶标特异激活判别），辅以 act_S1（绝对靶标激活）与 Δdual
def score(r):
    return 0.5 * r["sep_act"] + 0.3 * r["act_S1"] + 0.2 * r["sep_dual"]
out_rows.sort(key=score, reverse=True)
for i, r in enumerate(out_rows, 1):
    r["rank"] = i
    r["combo_score"] = round(score(r), 4)

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["rank", "cand", "combo_score"] + [k for k in out_rows[0] if k not in ("rank", "cand", "combo_score")])
    w.writeheader()
    w.writerows(out_rows)
print(f"\n=== 综合排序 (0.5*Δact + 0.3*act_S1 + 0.2*Δdual) ===")
for r in out_rows:
    print(f"{r['rank']}. {r['cand']:<12} combo={r['combo_score']:>6.3f}  "
          f"Δact={r['sep_act']:+.2f}  act_S1={r['act_S1']:.2f}  Δdual={r['sep_dual']:+.2f}  "
          f"Δnt={r['sep_ntcov']:+.1f}")
print(f"\nWROTE {OUT}")
