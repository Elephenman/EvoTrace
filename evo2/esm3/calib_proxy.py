# -*- coding: utf-8 -*-
"""代理校准层 v1：用 Boltz-2 状态标签校准 ESM3 代理（工具主动学习闭环）。

数据源（全为 Boltz-2 状态分布标签，HEXXH 接触口径）：
  1. reference_labels.csv  — 旧战役 8 系统（WT/HQL2/TrackF_r1/RDP2/RD_POS/skB_c1/s13_c1/s13_c2）
  2. wave0_boltz_verdict.csv + wave0_seqs.fa — 24 精英
  3. 本轮 v3cand 两轮 Boltz（9 候选 × S1/OFF，本地判读）

流程: 序列 → v3 代理 fitness → Spearman(代理, 标签) → Isotonic 校准(fitness→act_s1)
输出: calib_proxy_report.csv + calib_isotonic.npz + 相关性/校准质量报告
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "A:/claudework/out")
from ppri_surrogate_v3 import PprISurrogateV3

RES = "A:/claudework/ppri_evo/results"
OUT = "A:/claudework/out/b7_full"

# ---- 1) reference 8 系统 ----
ref_lab = pd.read_csv(os.path.join(RES, "reference_labels.csv"))
ref_seq = json.load(open(os.path.join(RES, "reference_sequences.json")))
# ---- 2) wave0 24 精英 ----
w0_ver = pd.read_csv(os.path.join(RES, "wave0_boltz_verdict.csv"))
w0_fa = {}
for line in open(os.path.join(RES, "wave0_seqs.fa")):
    line = line.strip()
    if line.startswith(">"):
        cur = line[1:].split()[0]
    else:
        w0_fa[cur] = w0_fa.get(cur, "") + line
# eN → elite_N_pX
elite_map = {}
for k in w0_fa:
    if k.startswith("elite_"):
        num = int(k.split("_")[1])
        elite_map[f"e{num}"] = k
# ---- 3) v3cand 两轮（序列从 CANDS 重建） ----
AA = "ACDEFGHIKLMNPQRSTVWY"
wt = open("A:/claudework/ppri_evo/inputs/wt_254.fasta").read().splitlines()[1].strip()
def mut(seq, pdb, aa):
    i = pdb - 22
    return seq[:i] + aa + seq[i+1:]
vc_seqs = {
    "WT": wt,
    "K216R": mut(wt, 216, "R"),
    "Q120V": mut(wt, 120, "V"),
    "Y170F": mut(wt, 170, "F"),
    "F88Y_Y170F": mut(mut(wt, 88, "Y"), 170, "F"),
    "K216R_Y170F_Q120V": mut(mut(mut(wt, 216, "R"), 170, "F"), 120, "V"),
    "R85A_R207A_R267A": mut(mut(mut(wt, 85, "A"), 207, "A"), 267, "A"),
    "K216R_Q120V": mut(mut(wt, 216, "R"), 120, "V"),
    "K216R_Q120V_F88Y_Y170F": mut(mut(mut(mut(wt, 216, "R"), 120, "V"), 88, "Y"), 170, "F"),
}
# v3cand 判读（本地重算，避免手写）
sys.path.insert(0, "A:/claudework/out")
from analyze_v3cand_boltz import parse_cif, model_metrics
import glob
vc_lab = {}
for base_dir in ["A:/claudework/out/boltz_v3cand_out", "A:/claudework/out/boltz_v3cand2_out"]:
    for cif in glob.glob(os.path.join(base_dir, "**", "*.cif"), recursive=True):
        parts = cif.replace("\\", "/").split("/")
        if "predictions" not in parts:
            continue
        name = parts[parts.index("predictions") + 1]
        m = model_metrics(cif)
        if m:
            vc_lab.setdefault(name, []).append(m)
vc_agg = {}
for name, ms in vc_lab.items():
    n = len(ms)
    act = sum(1 for m in ms if m["act"]) / n
    lock = sum(1 for m in ms if m["lock"]) / n
    base = name.replace("_S1_G17", "").replace("_OFF_T_G17", "")
    tag = "S1" if "_S1_" in name else "OFF"
    vc_agg.setdefault(base, {})[tag] = act

# ---- 组装标签集 ----
rows = []
# reference
for _, r in ref_lab.iterrows():
    nm = r["system"]
    if nm in ref_seq and not pd.isna(r["act_s1_pct"]):
        rows.append(dict(name=nm, seq=ref_seq[nm], act_s1=r["act_s1_pct"] / 100.0,
                         act_off=r["act_off_pct"] / 100.0 if not pd.isna(r["act_off_pct"]) else np.nan,
                         src="reference"))
# wave0
for _, r in w0_ver.iterrows():
    fa_key = elite_map.get(r["elite"])
    if fa_key and fa_key in w0_fa:
        rows.append(dict(name=r["elite"], seq=w0_fa[fa_key], act_s1=r["act_s1"] / 100.0,
                         act_off=r["act_off"] / 100.0, src="wave0"))
# v3cand
for base, d in vc_agg.items():
    if base in vc_seqs and "S1" in d:
        rows.append(dict(name=base, seq=vc_seqs[base], act_s1=d["S1"],
                         act_off=d.get("OFF", np.nan), src="v3cand"))

df = pd.DataFrame(rows).drop_duplicates(subset="name")
print(f"[dataset] {len(df)} 系统（reference {sum(df.src=='reference')} + wave0 {sum(df.src=='wave0')} + v3cand {sum(df.src=='v3cand')}）")

# ---- v3 代理打分 ----
orc = PprISurrogateV3()
fits = []
for _, r in df.iterrows():
    seq = r["seq"]
    assert len(seq) == len(wt), r["name"]
    # 序列 → 基因型（53 位点 AA 索引）
    g = np.array([AA.index(seq[s]) if seq[s] in AA else orc.wt_idx[j]
                  for j, s in enumerate(orc.sites)], dtype=np.int64)
    fits.append(float(orc.evaluate_multi(g[None, :])[0]))
df["proxy_fit"] = fits
df["sep"] = df["act_s1"] - df["act_off"]

# ---- 相关性 ----
print("\n=== 代理 fitness vs Boltz 标签 (Spearman) ===")
for col in ["act_s1", "act_off", "sep"]:
    valid = df[col].notna()
    if valid.sum() >= 6:
        rho = spearmanr(df.loc[valid, "proxy_fit"], df.loc[valid, col]).statistic
        print(f"  proxy_fit vs {col:>8}: rho={rho:.3f} (n={valid.sum()})")
# 标签间一致性
if df["act_off"].notna().sum() >= 6:
    rho = spearmanr(df["act_s1"], df["act_off"]).statistic
    print(f"  act_s1 vs act_off: rho={rho:.3f}（标签自身）")

# ---- Isotonic 校准（代理 → act_s1），LOO CV ----
y = df["act_s1"].to_numpy()
x = df["proxy_fit"].to_numpy()
n = len(y)
pred_cv = np.full(n, np.nan)
for i in range(n):
    mask = np.ones(n, bool); mask[i] = False
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x[mask], y[mask])
    pred_cv[i] = iso.predict([x[i]])[0]
base_mse = float(np.mean((y - y.mean()) ** 2))
cal_mse = float(np.mean((y - pred_cv) ** 2))
rho_cal = spearmanr(pred_cv, y).statistic
print(f"\n=== Isotonic 校准（LOO CV, n={n}）===")
print(f"  基线 MSE（均值预测）: {base_mse:.4f}")
print(f"  校准后 MSE:           {cal_mse:.4f}  ({'改善' if cal_mse < base_mse else '无改善'})")
print(f"  校准后 Spearman:      {rho_cal:.3f}")

# 全量拟合保存
iso_full = IsotonicRegression(out_of_bounds="clip")
iso_full.fit(x, y)
_x = getattr(iso_full, "X_thresholds_", getattr(iso_full, "X_", x))
_y = getattr(iso_full, "y_thresholds_", getattr(iso_full, "y_", y))
np.savez(os.path.join(OUT, "calib_isotonic.npz"),
         x=_x, y=_y, base_mse=base_mse, cal_mse=cal_mse)
df.to_csv(os.path.join(OUT, "calib_proxy_dataset.csv"), index=False)
print(f"\n保存: calib_isotonic.npz + calib_proxy_dataset.csv -> {OUT}")
print("\n=== 数据集明细 ===")
print(df[["name", "src", "proxy_fit", "act_s1", "act_off", "sep"]].round(3).to_string(index=False))
