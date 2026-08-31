# -*- coding: utf-8 -*-
"""gate v3 校准 —— 基于去混杂扫描 (job 225686, 22 变体 × 2 条件 × 8 models)

相比 v2 的改进:
  v2 校准集 = n=6 真实候选，位点突变共现混杂（F88R 总与 Y217R+M255K 同现）
            → 把 Y217R 的破坏效应错记到 F88R 头上。
  v3 校准集 = n=22，背景固定为 s13_c1，仅单点/双点扰动 → 无混杂。

标签合成 (每变体):
  lock  = dual_S1            （靶标专有双锁率；dual_OFF 全表 0/176）
  disc  = z_act              （激活判别显著性）
  iface = z_iface            （界面判别显著性）
  composite = 0.40*rn(lock) + 0.30*rn(disc) + 0.30*rn(iface)   （rn = 秩归一到 [0,1]）

拟合: 加性模型 g = w_F*T_F88[aa] + w_M*T_M255[aa] + w_Y*T_Y217[aa] + b
验证: 留一交叉验证 (LOO) 的 Spearman；并与 v1/v2 对照。
"""
import csv, itertools, math
import numpy as np

STATS = r"A:/claudework/out/boltz_six/deconf_stats.csv"
S = list(csv.DictReader(open(STATS, encoding="utf-8")))
for s in S:
    for k in ("dual_S1", "d_iface", "d_act", "z_iface", "z_act"):
        s[k] = float(s[k])

# ---------- 合成标签 ----------
def rank_norm(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    for rank, i in enumerate(order):
        r[i] = rank / (len(vals) - 1)
    return r

lock = [s["dual_S1"] for s in S]
disc = [s["z_act"] for s in S]
ifac = [s["z_iface"] for s in S]
rl, rd, ri = rank_norm(lock), rank_norm(disc), rank_norm(ifac)
Y = np.array([0.40 * rl[i] + 0.30 * rd[i] + 0.30 * ri[i] for i in range(len(S))])

VARIANTS = [(s["F88"], s["M255"], s["Y217"]) for s in S]

# ---------- 位点水平表（边际均值，Y217 用相对 Y 的偏差）----------
def build_tables(idx):
    """idx: 参与拟合的样本下标"""
    sub = [S[i] for i in idx]
    Yi = np.array([Y[i] for i in idx])

    TF, TM, TY = {}, {}, {}
    for aa in set(s["F88"] for s in sub):
        m = [Y[j] for j, s in enumerate(sub) if s["F88"] == aa and s["Y217"] == "Y"]
        TF[aa] = sum(m) / len(m) if m else 0.0
    for aa in set(s["M255"] for s in sub):
        m = [Y[j] for j, s in enumerate(sub) if s["M255"] == aa and s["Y217"] == "Y"]
        TM[aa] = sum(m) / len(m) if m else 0.0
    # Y217: 仅 F88=K/M255=M 三点，用相对 Y 的偏差
    base = [Y[j] for j, s in enumerate(sub) if s["F88"] == "K" and s["M255"] == "M" and s["Y217"] == "Y"]
    b0 = sum(base) / len(base) if base else 0.0
    for aa in ("Y", "F", "R"):
        m = [Y[j] for j, s in enumerate(sub) if s["F88"] == "K" and s["M255"] == "M" and s["Y217"] == aa]
        TY[aa] = (sum(m) / len(m) - b0) if m else 0.0
    # 中心化（减均值）避免与截距共线
    for T in (TF, TM, TY):
        mu = sum(T.values()) / len(T)
        for k in T:
            T[k] -= mu
    return TF, TM, TY


def design(sub, TF, TM, TY):
    X = []
    for s in sub:
        X.append([TF.get(s["F88"], 0.0), TM.get(s["M255"], 0.0), TY.get(s["Y217"], 0.0), 1.0])
    return np.array(X)


def fit(idx):
    sub = [S[i] for i in idx]
    Yi = np.array([Y[i] for i in idx])
    TF, TM, TY = build_tables(idx)
    X = design(sub, TF, TM, TY)
    w, *_ = np.linalg.lstsq(X, Yi, rcond=None)
    return TF, TM, TY, w


def predict(v, TF, TM, TY, w):
    f, m, y = v
    return w[0] * TF.get(f, 0.0) + w[1] * TM.get(m, 0.0) + w[2] * TY.get(y, 0.0) + w[3]


# ---------- 留一交叉验证 ----------
pred_loo = np.zeros(len(S))
for h in range(len(S)):
    idx = [i for i in range(len(S)) if i != h]
    TF, TM, TY, w = fit(idx)
    pred_loo[h] = predict(VARIANTS[h], TF, TM, TY, w)

# ---------- 全量拟合（产出最终表）----------
TF, TM, TY, w = fit(list(range(len(S))))
pred_full = np.array([predict(v, TF, TM, TY, w) for v in VARIANTS])


def spearman(a, b):
    ra, rb = rank_norm(list(a)), rank_norm(list(b))
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(len(ra)))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((x - mb) ** 2 for x in rb))
    return num / (da * db) if da > 0 and db > 0 else 0.0


# ---------- v1 / v2 对照（在去混杂集上打分）----------
import sys
sys.path.insert(0, r"A:/claudework/evo2/esm3")
from ppri_dna_aware import DnaAwareLandscape, AA as _AA

PRIORS = list(csv.DictReader(open(r"A:/claudework/ppri_evo/results/priors.csv", encoding="utf-8")))
SITES = [int(r["seq_idx"]) for r in PRIORS]
WT = open(r"A:/claudework/ppri_evo/inputs/wt_254.fasta").read().splitlines()[1].strip()
WT_IDX = [_AA.index(WT[s]) for s in SITES]
COL = {s: i for i, s in enumerate(SITES)}
PDB2COL = {88: COL.get(66), 217: COL.get(195), 255: COL.get(233),
           85: COL.get(63), 207: COL.get(186), 267: COL.get(245)}
BG_MUT = {88: 'K', 217: 'Y', 255: 'I', 85: 'G', 207: 'K', 267: 'R'}  # s13_c1 背景


class Stub:
    L = 53
    def __init__(self, sites, wt_idx):
        self.sites, self.wt_idx = sites, wt_idx
    def evaluate(self, g):
        return [0.0] * len(g)


stub = Stub(SITES, WT_IDX)
lands = {v: DnaAwareLandscape(stub, gate_version=v) for v in ("v1", "v2")}

v1s, v2s = [], []
for s in S:
    g = list(WT_IDX)
    for pdb, c in PDB2COL.items():
        if c is None:
            continue
        aa = {'88': s["F88"], '255': s["M255"], '217': s["Y217"]}.get(str(pdb), BG_MUT[pdb])
        g[c] = _AA.index(aa)
    v1s.append(lands["v1"]._gate(g))
    v2s.append(lands["v2"]._gate(g))

print("=" * 80)
print("gate 校准对照 —— 评价集 = 去混杂扫描 (n=22, 背景固定 s13_c1)")
print("=" * 80)
print(f"{'gate':<22}{'rho vs composite':>20}{'rho vs dual_S1':>18}{'rho vs Δact':>15}")
rows = [("v1 解析式(已证伪)", v1s), ("v2 Boltz n=6(混杂)", v2s),
        ("v3 去混杂 全量拟合", pred_full), ("v3 去混杂 LOO-CV", pred_loo)]
for name, p in rows:
    print(f"{name:<22}{spearman(p, Y):>20.3f}{spearman(p, lock):>18.3f}{spearman(p, disc):>15.3f}")
print()
print(f"权重 w = [F88 {w[0]:+.3f} | M255 {w[1]:+.3f} | Y217 {w[2]:+.3f} | b {w[3]:+.3f}]")
print()

print("=" * 80)
print("v3 位点偏好表（去混杂实测边际效应，已中心化）")
print("=" * 80)
print("  F88 :", "  ".join(f"{k}:{v:+.3f}" for k, v in sorted(TF.items(), key=lambda x: -x[1])))
print("  M255:", "  ".join(f"{k}:{v:+.3f}" for k, v in sorted(TM.items(), key=lambda x: -x[1])))
print("  Y217:", "  ".join(f"{k}:{v:+.3f}" for k, v in sorted(TY.items(), key=lambda x: -x[1])))
print()

print("=" * 80)
print("每变体：v3(LOO) 预测 vs 实测 composite")
print("=" * 80)
print(f"{'variant':<22}{'dual':>6}{'Δiface':>8}{'Δact':>7}{'composite':>10}{'v3_LOO':>9}")
order = sorted(range(len(S)), key=lambda i: -Y[i])
for i in order:
    s = S[i]
    print(f"{s['variant']:<22}{s['dual_S1']:>6.2f}{s['d_iface']:>8.1f}{s['d_act']:>7.2f}"
          f"{Y[i]:>10.3f}{pred_loo[i]:>9.3f}")

# 导出 v3 表供模块写入
with open(r"A:/claudework/out/boltz_six/gate_v3_tables.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["site", "aa", "value"])
    for site, T in (("F88", TF), ("M255", TM), ("Y217", TY)):
        for aa, v in sorted(T.items(), key=lambda x: -x[1]):
            wr.writerow([site, aa, f"{v:+.4f}"])
    wr.writerow(["w", "F88", f"{w[0]:+.4f}"])
    wr.writerow(["w", "M255", f"{w[1]:+.4f}"])
    wr.writerow(["w", "Y217", f"{w[2]:+.4f}"])
    wr.writerow(["w", "bias", f"{w[3]:+.4f}"])
print()
print("WROTE gate_v3_tables.csv")
