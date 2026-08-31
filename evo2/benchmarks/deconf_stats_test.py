# -*- coding: utf-8 -*-
"""去混杂结果的显著性检验：
1) dual_S1 两两比较（s13_c1 复现 vs 其他 F88 变体）
2) 修正上一轮混杂结论：F88R 是否真"摧毁双锁"？真正的摧毁因子是谁？
"""
import csv, math

S = list(csv.DictReader(open(r"A:/claudework/out/boltz_six/deconf_stats.csv", encoding="utf-8")))
for s in S:
    for k in ("dual_S1", "dual_OFF", "d_iface", "d_act", "z_iface", "z_act"):
        s[k] = float(s[k])
    s["n_s1"] = int(s["n_s1"])

by = {s["variant"]: s for s in S}


def ztest_prop(k1, n1, k2, n2):
    """两比例 z 检验（合并方差）"""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se < 1e-12:
        return 0.0, 1.0
    z = (p1 - p2) / se
    pv = math.erfc(abs(z) / math.sqrt(2))
    return z, pv


print("=" * 78)
print("1) dual_OFF 全表检验 — 双锁是否靶标专有？")
print("=" * 78)
tot_off = sum(round(s["dual_OFF"] * s["n_s1"]) for s in S)
n_off = sum(s["n_s1"] for s in S)
tot_s1 = sum(round(s["dual_S1"] * s["n_s1"]) for s in S)
n_s1 = sum(s["n_s1"] for s in S)
z, p = ztest_prop(tot_s1, n_s1, tot_off, n_off)
print(f"  dual_S1 = {tot_s1}/{n_s1} = {tot_s1/n_s1:.3f}")
print(f"  dual_OFF= {tot_off}/{n_off} = {tot_off/n_off:.3f}")
print(f"  z={z:.2f}  p={p:.2e}  → {'双锁严格靶标专有' if p < 0.01 else '不显著'}")
print()

print("=" * 78)
print("2) 上一轮混杂结论复核：F88R 是否摧毁双锁？（本次背景固定 s13_c1）")
print("=" * 78)
print("  上一轮(job225685) 观察：RDP2/RD_POS (F88R + Y217R + M255K) dual=0.00")
print("  → 归因为 F88R。本次固定背景单点扫描：")
for v in ("F88R_M255I_Y217Y", "F88R_M255A_Y217Y", "F88R_M255K_Y217Y", "F88R_M255M_Y217Y"):
    s = by.get(v)
    if s:
        print(f"    {v:<20} dual_S1={s['dual_S1']:.2f}  Δiface={s['d_iface']:+.1f}  Δact={s['d_act']:+.2f}")
print()
print("  → F88R 单独存在时 dual_S1 = 0.12~0.38 (非 0)，Δiface 主效应 +9.5 (全场最高)")
print()

print("=" * 78)
print("3) 真正的摧毁因子：Y217")
print("=" * 78)
print("  固定 F88=K, M255=M：")
for aa in ("Y", "F", "R"):
    s = by.get(f"F88K_M255M_Y217{aa}")
    if s:
        print(f"    Y217{aa}: dual_S1={s['dual_S1']:.2f}  Δiface={s['d_iface']:+.1f}  Δact={s['d_act']:+.2f}")
kY = round(by["F88K_M255M_Y217Y"]["dual_S1"] * 8)
kR = round(by["F88K_M255M_Y217R"]["dual_S1"] * 8)
z, p = ztest_prop(kY, 8, kR, 8)
print(f"  Y217Y vs Y217R: z={z:.2f} p={p:.3f} {'显著' if p<0.05 else '(n=8 功效不足，趋势一致)'}")
print()

print("=" * 78)
print("4) M255 主效应复核（上一轮归因 M255K 为破坏因子）")
print("=" * 78)
for aa in ("A", "I", "K", "M"):
    sub = [s for s in S if s["M255"] == aa and s["Y217"] == "Y"]
    m_if = sum(x["d_iface"] for x in sub) / len(sub)
    m_du = sum(x["dual_S1"] for x in sub) / len(sub)
    print(f"    M255{aa}: Δiface(边际)={m_if:+6.1f}   dual_S1(边际)={m_du:.2f}")
print("  → M255M(天然) Δiface 最差 (-0.9)；M255K 居中 (+3.5)，非上一轮所判的'破坏因子'")
print()

print("=" * 78)
print("5) s13_c1 (=F88K_M255I_Y217Y) 是否仍最优？关键两两比较")
print("=" * 78)
ref = by["F88K_M255I_Y217Y"]
print(f"  基准 s13_c1: dual_S1={ref['dual_S1']:.2f} Δiface={ref['d_iface']:+.1f} Δact={ref['d_act']:+.2f}(z={ref['z_act']:.1f})")
print()
print(f"  {'变体':<22}{'dual_S1':>8}{'Δdual z':>9}{'p':>8}{'Δiface':>9}{'Δact':>7}  判语")
for v in sorted(S, key=lambda x: -x["dual_S1"]):
    if v["variant"] == "F88K_M255I_Y217Y":
        continue
    k1 = round(ref["dual_S1"] * 8)
    k2 = round(v["dual_S1"] * 8)
    z, p = ztest_prop(k1, 8, k2, 8)
    verdict = []
    if v["dual_S1"] < ref["dual_S1"] and p < 0.10:
        verdict.append("双锁弱于 s13_c1")
    if v["d_iface"] > ref["d_iface"] + 2:
        verdict.append("界面判别更强")
    if v["d_act"] < ref["d_act"] - 0.2:
        verdict.append("激活判别弱")
    print(f"  {v['variant']:<22}{v['dual_S1']:>8.2f}{z:>9.2f}{p:>8.3f}"
          f"{v['d_iface']:>9.1f}{v['d_act']:>7.2f}  {'; '.join(verdict) or '—'}")
