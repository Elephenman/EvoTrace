# -*- coding: utf-8 -*-
"""v4 模型消融/敏感性检查（防泄漏与归因）。

A 剔除条件特征（cond/nt10/17/23）—— sep 能力应崩塌, 证明条件维是来源
B 剔除代理 z 特征 —— 检验 ESM3 代理的边际贡献
C V2 扩展: 全局训练（含 old/six/v3cand/v2a）→ 预测 deconf-22 composite（外部数据反哺?）
D V1b 分层: 按 |sep_true| 分桶看排序质量集中在哪
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_sep_model_v4 import (CAT_FEATS, NUM_FEATS, build_design, load_cells,
                                load_sequences, loo_system, make_gbm, make_ridge,
                                spear)

OUT = "A:/claudework/evo2/results/sep_model_v4"
D0 = pd.read_csv(f"{OUT}/cells.csv")   # 已含全部特征与标签
print(f"[data] {len(D0)} cells, {D0.system.nunique()} systems")


def enc(d, target, cat_feats, num_feats):
    d = d.dropna(subset=[target]).copy()
    X = pd.get_dummies(d[num_feats + cat_feats], columns=cat_feats, drop_first=True)
    return d, X, d[target].to_numpy(float)


def run_variant(name, cat_over, num_over, target="y_act"):
    d, X, y = enc(D0, target, cat_over, num_over)
    sysm = d.system.to_numpy()
    pg = loo_system(d, X.to_numpy(float), y, sysm, make_gbm)
    return d, pg, spear(pg, y)


# ---- A/B: 条件特征与 z 的贡献 ----
full_cat, full_num = CAT_FEATS, NUM_FEATS
noc_cat = [c for c in full_cat if c not in ("cond", "nt17", "nt23", "nt10")]
noz_num = [f for f in full_num if f != "z"]

print("\n== A/B 消融（GBM, 按系统 LOO）==")
for tgt in ("y_act", "y_dual"):
    _, _, r_full = run_variant("full", full_cat, full_num, tgt)
    _, _, r_nocond = run_variant("no-cond", noc_cat, full_num, tgt)
    _, _, r_noz = run_variant("no-z", full_cat, noz_num, tgt)
    print(f"{tgt:8s}: full {r_full:+.3f} | 去条件特征 {r_nocond:+.3f} | 去z {r_noz:+.3f}")

# sep 的条件消融
def sep_eval(cat_over, num_over):
    d, X, y = enc(D0, "y_act", cat_over, num_over)
    sysm = d.system.to_numpy()
    pg = loo_system(d, X.to_numpy(float), y, sysm, make_gbm)
    d = d.assign(pred=pg)
    P = []
    for s, g in d.groupby("system"):
        if {"S1_G17", "OFF_T_G17"} <= set(g.condition):
            t = g.set_index("condition")
            P.append((t.loc["S1_G17", "y_act"] - t.loc["OFF_T_G17", "y_act"],
                      t.loc["S1_G17", "pred"] - t.loc["OFF_T_G17", "pred"]))
    P = np.array(P)
    return P, spear(P[:, 1], P[:, 0])

P_full, r_full = sep_eval(full_cat, full_num)
P_nc, r_nc = sep_eval(noc_cat, full_num)
sign_full = (np.sign(P_full[:, 1]) == np.sign(P_full[:, 0])).mean()
print(f"\nsep: full rho {r_full:+.3f} (符号准确 {sign_full:.2f}) | 去条件特征 rho {r_nc:+.3f}")

# ---- D: sep 分层 ----
order = np.argsort(-np.abs(P_full[:, 0]))
print("\n== D sep 按 |真值| 分层（LOO 预测）==")
for k, tag in [(10, "top10 |sep|"), (20, "top20"), (len(P_full), "全部")]:
    idx = order[:k]
    rho = spearmanr(P_full[idx, 1], P_full[idx, 0]).statistic
    print(f"{tag:12s}: Spearman {rho:+.3f}")

# ---- C: 全局训练 → deconf-22 composite 反哺 ----
ds = pd.read_csv("A:/claudework/out/boltz_six/deconf_stats.csv")

def rank_norm(v):
    o = np.argsort(np.argsort(v))
    return o / (len(v) - 1)

comp = 0.40 * rank_norm(ds.dual_S1.to_numpy()) + 0.30 * rank_norm(ds.z_act.to_numpy()) \
    + 0.30 * rank_norm(ds.z_iface.to_numpy())

# 用 y_dual（composite 主成分 0.40 权重）作代理目标做 transfer 检验
variants = ds.variant.tolist()
pred_global = np.full(len(variants), np.nan)
for i, v in enumerate(variants):
    tr = D0[~((D0.system == v))]
    tr = tr[tr.y_dual.notna()]
    Xtr = pd.get_dummies(tr[NUM_FEATS + CAT_FEATS], columns=CAT_FEATS, drop_first=True)
    m = make_gbm().fit(Xtr.to_numpy(float), tr.y_dual.to_numpy(float))
    te = D0[(D0.system == v) & (D0.condition == "S1_G17")]
    Xte = pd.get_dummies(te[NUM_FEATS + CAT_FEATS], columns=CAT_FEATS, drop_first=True)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    pred_global[i] = m.predict(Xte.to_numpy(float))[0]
rho_g = spear(pred_global, comp)
rho_own = spear(ds.dual_S1.to_numpy(), comp)  # 自身 dual 对 composite（上限参照）
print(f"\n== C 外部数据反哺 deconf composite ==")
print(f"全局训练(y_dual) 预测 22 变体 vs composite: {rho_g:+.3f}"
      f"（参照: 实测 dual_S1 自身 {rho_own:+.3f}）")
np.save(f"{OUT}/ablation_summary.npy", np.array([r_full, r_nc, rho_g]))
print(f"\n[out] {OUT}/ablation_summary.npy")
