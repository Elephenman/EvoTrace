# -*- coding: utf-8 -*-
"""v4 + DL(A1) 互补组合验证 —— 证明"DL 判结合强度 × v4 判特异性"组合优于单用任一。

互补前提（已由 n100 数据证实）: DL 预测的 act(S1) 与真 sep 相关仅 -0.082（正交），
即"强结合的候选未必特异"。故真正的高价值候选须两个维度都满足。

本脚本: 在 v4 与 DL 共同覆盖的 n100 系统上,
  1) 用 v4 GBM LOO 给逐系统 sep_pred（特异性维度, v4 强项）
  2) 用 DL A1 LOO 给逐系统 act(S1) pred（结合强度维度, DL 强项）
  3) 比较单指标 vs 互补组合(加权 rank 或双门) 对"真 sep"与"真 高价值(结合×特异)"的命中。
"""
import sys, os
sys.path.insert(0, "A:/claudework/evo2/benchmarks")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import ablate_v7_sep as A
import train_dl_v5 as M

RES = "A:/claudework/evo2/results"

# ---- 1) DL A1 LOO 逐系统 act(S1)/act(OFF)/sep ----
d = np.load(f"{A.OUT}/loo_A1_ep24.npz", allow_pickle=True)
_, cells, _, _ = A.load_all()
cells2 = cells.copy(); cells2["p_act"] = d["pred"][:, 0]
dl = []
for s, g in cells2.groupby("system"):
    if {"S1_G17", "OFF_T_G17"} <= set(g.condition):
        t = g.set_index("condition")
        dl.append(dict(system=s, dl_act=t.loc["S1_G17", "p_act"],
                       dl_act_off=t.loc["OFF_T_G17", "p_act"]))
dl = pd.DataFrame(dl)

# ---- 2) v4 GBM LOO 逐系统 sep（用 n100 merged 特征表, 复用 v4 验证协议）----
merged = pd.read_csv(f"{RES}/boltz_n100/cells_n100_merged.csv")
from train_sep_model_v4 import encode, make_gbm, loo_system
# 训练 y_act 的 LOO, 求每系统 sep_pred
dd, X, y = encode(merged, "y_act")
sysm = dd.system.to_numpy()
pg = loo_system(dd, X.to_numpy(float), y, sysm, make_gbm)
dd = dd.assign(pred_act=pg)
v4seps = []
for s, g in dd.groupby("system"):
    if {"S1_G17", "OFF_T_G17"} <= set(g.condition):
        t = g.set_index("condition")
        v4seps.append(dict(system=s,
                           sep_true=t.loc["S1_G17", "y_act"] - t.loc["OFF_T_G17", "y_act"],
                           v4_sep=t.loc["S1_G17", "pred_act"] - t.loc["OFF_T_G17", "pred_act"],
                           v4_act=t.loc["S1_G17", "pred_act"]))
v4 = pd.DataFrame(v4seps)

# ---- 3) 对齐共同系统 ----
m = dl.merge(v4, on="system", how="inner")
print(f"v4系统{len(v4)} × DL系统{len(dl)} → 共同 {len(m)} 个")
assert len(m) >= 15, "共同系统太少, 无法统计"

# ---- 4) 指标对比 ----
def rank(v):
    from scipy.stats import rankdata
    return rankdata(v) / len(v)

def sr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return spearmanr(a[ok], b[ok]).statistic

# 真"高价值" = 结合强(act_true S1) 与 特异(sep_true) 双高 → 用 min(rank_act, rank_sep) 作真值? 简化: 真 sep 为主判据
# 我们用"真 sep"作金标准(特异性是用户核心), 同时报告与"真 act"的平衡。
print("\n===== 各信号 vs 真 sep (金标准: 特异性) =====")
sig = {
    "DL_act(S1)": m.dl_act.to_numpy(),
    "DL_sep": m.dl_act.to_numpy() - m.dl_act_off.to_numpy(),
    "v4_act(S1)": m.v4_act.to_numpy(),
    "v4_sep": m.v4_sep.to_numpy(),
}
for k, v in sig.items():
    print(f"  {k:12s} vs 真sep rho = {sr(v, m.sep_true):+.3f}")
# 互补组合: DL_act 与 v4_sep 联合
comb_min = np.minimum(rank(m.dl_act), rank(m.v4_sep))  # 双高(min) = 既结合又特异
comb_add = 0.5 * rank(m.dl_act) + 0.5 * rank(m.v4_sep)  # 加权和
comb_mult = rank(m.dl_act) * rank(m.v4_sep)             # 乘积(双高惩罚单高)
print("\n===== 互补组合 vs 真 sep =====")
print(f"  min(dl_act, v4_sep) rho = {sr(comb_min, m.sep_true):+.3f}")
print(f"  0.5·dl_act+0.5·v4_sep rho = {sr(comb_add, m.sep_true):+.3f}")
print(f"  dl_act × v4_sep        rho = {sr(comb_mult, m.sep_true):+.3f}")

# ---- 5) 能否识别'假特异'(结合强但sep低的HQL2类) ----
print("\n===== 候选定位（sep_pred 排序）=====")
m["rank_v4sep"] = rank(m.v4_sep); m["rank_dlact"] = rank(m.dl_act)
m["comb"] = np.minimum(m.rank_v4sep, m.rank_dlact)
m["is_key"] = m.system.str.contains("s13|TrackF_r1$|HQL2$|RD_POS|TrackF_F88", na=False)
view = m.sort_values("comb", ascending=False)[["system", "sep_true", "v4_sep", "dl_act", "comb"]].head(15)
print(view.to_string())
m.to_csv(f"{A.OUT}/combine_result.csv", index=False)
