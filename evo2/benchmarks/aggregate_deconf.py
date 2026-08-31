# -*- coding: utf-8 -*-
"""去混杂扫描聚合：单点效应 + 交互 + 内部一致性检验

背景固定 = s13_c1 (19 突变)，仅扫 F88 ∈ {F,K,R,W,Y} × M255 ∈ {A,I,K,M}，Y217 固定 Y；
外加 Y217 ∈ {F,R}(F88=K/M255=M) 两个。
输出：deconf_stats.csv (每变体 × 条件) + 打印主效应表。
"""
import csv, re, math, os

FP = r"A:/claudework/out/boltz_six/deconf_fingerprint.csv"
OUT_DIR = r"A:/claudework/out/boltz_six"

rows = list(csv.DictReader(open(FP, encoding="utf-8")))
for r in rows:
    for k in ("conf", "iptm", "ligand_iptm", "act", "dual", "iface"):
        r[k] = float(r[k])

# ---- 分组: variant -> cond -> list ----
G = {}
for r in rows:
    m = re.match(r"d_(F88\w)_(M255\w)_(Y217\w)_(S1_G17|OFF_T_G17)", r["pred"])
    if not m:
        continue
    v = (m.group(1), m.group(2), m.group(3))
    cond = "S1" if m.group(4) == "S1_G17" else "OFF"
    G.setdefault(v, {}).setdefault(cond, []).append(r)


def mean(x):
    return sum(x) / len(x)


def sem(x):
    if len(x) < 2:
        return 0.0
    m = mean(x)
    return math.sqrt(sum((a - m) ** 2 for a in x) / (len(x) - 1)) / math.sqrt(len(x))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


stats = []
for v in sorted(G):
    s1, off = G[v].get("S1", []), G[v].get("OFF", [])
    if not s1 or not off:
        continue
    if_s1, if_off = [r["iface"] for r in s1], [r["iface"] for r in off]
    ac_s1, ac_off = [r["act"] for r in s1], [r["act"] for r in off]
    du_s1, du_off = [r["dual"] for r in s1], [r["dual"] for r in off]

    d_if = mean(if_s1) - mean(if_off)
    # Δiface 的 SEM（两独立组）
    se_if = math.sqrt(sem(if_s1) ** 2 + sem(if_off) ** 2)
    d_ac = mean(ac_s1) - mean(ac_off)
    se_ac = math.sqrt(sem(ac_s1) ** 2 + sem(ac_off) ** 2)

    k1, n1 = int(sum(du_s1)), len(du_s1)
    k0, n0 = int(sum(du_off)), len(du_off)
    lo1, hi1 = wilson(k1, n1)
    lo0, hi0 = wilson(k0, n0)

    stats.append(dict(
        variant="_".join(v),
        F88=v[0][-1], M255=v[1][-1], Y217=v[2][-1],
        n_s1=n1, n_off=n0,
        dual_S1=mean(du_s1), dual_OFF=mean(du_off),
        dual_lo=lo1, dual_hi=hi1,
        iface_S1=mean(if_s1), iface_OFF=mean(if_off),
        d_iface=d_if, d_iface_se=se_if,
        z_iface=(d_if / se_if) if se_if > 1e-9 else 0.0,
        act_S1=mean(ac_s1), act_OFF=mean(ac_off),
        d_act=d_ac, d_act_se=se_ac,
        z_act=(d_ac / se_ac) if se_ac > 1e-9 else 0.0,
        conf=mean([r["conf"] for r in s1 + off]),
        iptm=mean([r["iptm"] for r in s1 + off]),
    ))

os.makedirs(OUT_DIR, exist_ok=True)
op = os.path.join(OUT_DIR, "deconf_stats.csv")
with open(op, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
    w.writeheader()
    for s in stats:
        w.writerow(s)
print("WROTE", op, "rows", len(stats))
print()

# ---------- 打印：全表 ----------
print("== 每变体（背景 = s13_c1，8 samples/条件）==")
print(f"{'variant':<26}{'dual_S1':>8}{'dual_OFF':>9}{'iface_S1':>9}{'iface_OFF':>10}{'Δiface':>8}{'z':>7}{'actS1':>7}{'actOFF':>7}{'Δact':>7}{'z':>7}")
for s in sorted(stats, key=lambda x: -x["d_iface"]):
    print(f"{s['variant']:<26}{s['dual_S1']:>8.2f}{s['dual_OFF']:>9.2f}"
          f"{s['iface_S1']:>9.1f}{s['iface_OFF']:>10.1f}{s['d_iface']:>8.1f}{s['z_iface']:>7.1f}"
          f"{s['act_S1']:>7.2f}{s['act_OFF']:>7.2f}{s['d_act']:>7.2f}{s['z_act']:>7.1f}")
print()

# ---------- 主效应：F88（边际化 M255，Y217=Y）----------
print("== 主效应 F88（Y217=Y，边际化 M255，n=4 变体×8）==")
print(f"{'F88':<5}{'dual_S1':>9}{'Δiface':>9}{'Δact':>8}")
for aa in "FKRWY":
    sub = [s for s in stats if s["F88"] == aa and s["Y217"] == "Y"]
    if not sub:
        continue
    print(f"{aa:<5}{mean([s['dual_S1'] for s in sub]):>9.2f}"
          f"{mean([s['d_iface'] for s in sub]):>9.1f}"
          f"{mean([s['d_act'] for s in sub]):>8.2f}")
print()

print("== 主效应 M255（Y217=Y，边际化 F88）==")
print(f"{'M255':<6}{'dual_S1':>9}{'Δiface':>9}{'Δact':>8}")
for aa in "AIKM":
    sub = [s for s in stats if s["M255"] == aa and s["Y217"] == "Y"]
    if not sub:
        continue
    print(f"{aa:<6}{mean([s['dual_S1'] for s in sub]):>9.2f}"
          f"{mean([s['d_iface'] for s in sub]):>9.1f}"
          f"{mean([s['d_act'] for s in sub]):>8.2f}")
print()

# ---------- 交互：F88 × M255 的 Δiface 网格 ----------
print("== 交互网格 Δiface（行=F88，列=M255，Y217=Y）==")
print(f"{'':<6}" + "".join(f"{a:>8}" for a in "AIKM"))
for aa in "FKRWY":
    line = f"{aa:<6}"
    for bb in "AIKM":
        sub = [s for s in stats if s["F88"] == aa and s["M255"] == bb and s["Y217"] == "Y"]
        line += f"{sub[0]['d_iface']:>8.1f}" if sub else f"{'-':>8}"
    print(line)
print()

# ---------- Y217 效应（F88=K, M255=M）----------
print("== Y217 效应（F88=K, M255=M）==")
for aa in "YFR":
    sub = [s for s in stats if s["F88"] == "K" and s["M255"] == "M" and s["Y217"] == aa]
    if sub:
        s = sub[0]
        print(f"  Y217{aa}: dual_S1={s['dual_S1']:.2f} Δiface={s['d_iface']:+.1f} (z={s['z_iface']:.1f}) Δact={s['d_act']:+.2f}")
print()

# ---------- 内部一致性：本扫描的 s13_c1 复现 vs job 225685 ----------
rep = [s for s in stats if s["F88"] == "K" and s["M255"] == "I" and s["Y217"] == "Y"]
if rep:
    s = rep[0]
    print("== 内部一致性：d_F88K_M255I_Y217Y 即 s13_c1 复现 ==")
    print(f"  本扫描(225686, 8 samples): dual_S1={s['dual_S1']:.2f} Δiface={s['d_iface']:+.1f} Δact={s['d_act']:+.2f}")
    print(f"  原判读(225685, 20 samples): dual_S1=0.35 Δiface=+10.5 Δact=+0.45")
