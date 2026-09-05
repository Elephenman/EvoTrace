# -*- coding: utf-8 -*-
"""A1 全量(ep24) 后处理：算 V1 act/dual/dcat + V1b sep，对比 v4/v7 基准。"""
import sys, os
sys.path.insert(0, "A:/claudework/evo2/benchmarks")
import numpy as np
import ablate_v7_sep as A
import train_dl_v5 as M

d = np.load(f"{A.OUT}/loo_A1_ep24.npz", allow_pickle=True)
pred = d["pred"]
# 用与 run_ablation 完全相同的 cells 聚合（load_all）
rows, cells, toks, globs = A.load_all()
print(f"pred {pred.shape}, cells {len(cells)}")
# run_ablation 里折顺序 = sorted(cells.system.unique())，pred 按 cells 原始行序填写(te_masks)
# cells 行序 = groupby([system,condition]) 排序。核对 pred 第0列非空即可。

y = cells.to_numpy()
from scipy.stats import spearmanr
def sr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return spearmanr(a[ok], b[ok]).statistic if ok.sum() >= 4 else np.nan

for k, c in enumerate(["y_act", "y_dual", "y_dcat"]):
    r = sr(pred[:, k], cells[c].to_numpy())
    print(f"  V1 {c}: A1 = {r:+.3f}")

cd = cells.assign(p_act=pred[:, 0])
P = []
for s, g in cd.groupby("system"):
    if {"S1_G17", "OFF_T_G17"} <= set(g.condition):
        t = g.set_index("condition")
        P.append((t.loc["S1_G17", "y_act"] - t.loc["OFF_T_G17", "y_act"],
                  t.loc["S1_G17", "p_act"] - t.loc["OFF_T_G17", "p_act"]))
P = np.array(P)
r_sep = sr(P[:, 1], P[:, 0]); sign = float((np.sign(P[:, 1]) == np.sign(P[:, 0])).mean())
big = np.argsort(-np.abs(P[:, 0]))[:10]
r_big = sr(P[big, 1], P[big, 0])
print(f"  V1b sep: A1 = {r_sep:+.3f} sign {sign:.2f} top10 {r_big:+.3f}")
print()
print("对比基准: v4@n100 act .739/dual .747/sep .330 符号 .92 ; v7 DL act .770/dual .838/sep −.124")
