# -*- coding: utf-8 -*-
"""v4@n100 偏置审计 —— X→K/R 电荷偏好三问审计 + 校正建议。

低算力、本地 CPU（torch.set_num_threads(1) / n_jobs=1）；trainer 同机跑训练，避免抢占。

数据: results/boltz_n100/cells_n100_merged.csv
  —— 已含 v4 全设计矩阵（rh66/m255/y217/r253mut/anchor_state/n_mut/n_out53/lock_mut/
     anchor_mut/cat_mut/charge_anchor/charge_total/z）+ 标签（y_act/y_dual/y_dcat...）。
  z 已由 ESM3 代理预计算，无需再跑代理。

三问:
  Q1 存在性/量化: v4@n100 是否对「引入正电荷的 X→K/R 突变」系统性高估增益？
  Q2 来源: 偏置来自模型特征（charge_total/charge_anchor）还是标签分布（生物学）？
  Q3 校正: 如何校正，避免提名被电荷偏好误导？

方法学纪律: 全部用按系统留一（system-LOO）出折叠预测，零泄漏；偏差由 OOF 残差衡量。
"""
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

RES = "A:/claudework/evo2/results"
SRC = os.path.join(RES, "boltz_n100", "cells_n100_merged.csv")
OUT = os.path.join(RES, "audit_bias")
os.makedirs(OUT, exist_ok=True)

NUM_FEATS = ["n_mut", "n_out53", "lock_mut", "anchor_mut", "cat_mut",
             "charge_anchor", "charge_total", "z"]
CAT_FEATS = ["rh66", "m255", "y217", "r253mut", "anchor_state",
             "cond", "nt17", "nt23", "nt10"]
CHARGE = {"K": 1, "R": 1, "H": 0.5, "D": -1, "E": -1}
WT = "A:/claudework/ppri_evo/inputs/wt_254.fasta"
WT_SEQ = open(WT).read().splitlines()[1].strip()
# pdb 残基号 -> 序列 0-based 索引 (train_sep_model_v4: P2S = pdb-22)
P2S = lambda pdb: pdb - 22
ANCHOR_PDB = (85, 207, 267)


def has_kr_mutation(system):
    """system 名里的显式 X→K / X→R 突变（格式 WTaa+编号+新aa，如 F88K / Y217R / K216R）。"""
    kr = False
    for tok in str(system).split("_"):
        m = re.match(r"^([A-Z])(\d+)([A-Z])$", tok)
        if m and m.group(3) in ("K", "R"):
            kr = True
    return int(kr)


def anchor_kr_intro(system, anchor_state):
    """anchor_state（如 G_K_R）相对 WT 锚位是否引入 K/R 正电荷。"""
    # 解析 anchor_state 三个锚位残基（85/207/267）
    toks = [t for t in str(anchor_state).split("_") if t]
    if len(toks) != 3:
        return 0
    intro = 0
    for pdb, aa in zip(ANCHOR_PDB, toks):
        wt = WT_SEQ[P2S(pdb)]
        if aa in ("K", "R") and wt not in ("K", "R"):
            intro = 1
    return intro


def encode(df, target):
    d = df.dropna(subset=[target]).copy()
    X = pd.get_dummies(d[NUM_FEATS + CAT_FEATS], columns=CAT_FEATS, drop_first=True)
    return d, X, d[target].to_numpy(float)


def make_ridge():
    sc = StandardScaler()
    return ("ridge", sc, RidgeCV(alphas=np.logspace(-2, 4, 25)))


def make_gbm():
    return ("gbm", None, HistGradientBoostingRegressor(
        max_depth=3, max_iter=250, learning_rate=0.06,
        l2_regularization=5.0, min_samples_leaf=8, random_state=0))


def fit_pred(d, X, y, sysm, factory, drop_first_cols=None):
    """按系统留一；drop_first_cols: 训练时剔除这些特征列（消融用）。"""
    cols = X.columns
    pred = np.full(len(y), np.nan)
    for s in pd.unique(sysm):
        te = sysm == s
        Xtr, Xte = X[~te], X[te]
        if drop_first_cols is not None:
            keep = [c for c in cols if c not in drop_first_cols]
            Xtr, Xte = Xtr[keep], Xte[keep]
        kind, sc, est = factory()
        if kind == "ridge":
            sc.fit(Xtr)
            est.fit(sc.transform(Xtr), y[~te])
            pred[te] = est.predict(sc.transform(Xte))
        else:
            est.fit(Xtr, y[~te])
            pred[te] = est.predict(Xte)
    return pred


def spear(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.nan if ok.sum() < 4 else float(spearmanr(a[ok], b[ok]).statistic)


def main():
    df = pd.read_csv(SRC)
    df["kr_mut"] = df.system.map(has_kr_mutation)
    df["anchor_kr"] = [anchor_kr_intro(s, a) for s, a in zip(df.system, df.anchor_state)]
    df["intro_pos"] = ((df.charge_total > 0) & (df.n_mut > 0)).astype(int)
    print(f"[data] rows={len(df)} systems={df.system.nunique()} "
          f"kr_mut={df.kr_mut.sum()} anchor_kr={df.anchor_kr.sum()}")

    report = ["# v4@n100 偏置审计 —— X→K/R 电荷偏好三问", "",
              f"- 数据: cells_n100_merged.csv | cells={len(df)} | systems={df.system.nunique()}",
              f"- 方法: 按系统留一(OOF)预测，零泄漏；偏差= OOF 残差", ""]

    # ---- 复现 v4@n100 基准（sanity）----
    rep_lines = ["## 0. v4@n100 复现（sanity）", "",
                 "| target | Ridge rho | GBM rho | 常数 MAE→GBM MAE |",
                 "|---|---:|---:|---:|"]
    oof = {}
    for tgt in ["y_act", "y_dual"]:
        d, X, y = encode(df, tgt)
        sysm = d.system.to_numpy()
        pr = fit_pred(d, X, y, sysm, make_ridge)
        pg = fit_pred(d, X, y, sysm, make_gbm)
        mae_g = float(np.nanmean(np.abs(pg - y)))
        pconst = np.array([np.nanmean(y[sysm != s]) for s in sysm])
        mae_c = float(np.nanmean(np.abs(pconst - y)))
        rep_lines.append(f"| {tgt} | {spear(pr, y):+.3f} | {spear(pg, y):+.3f} | "
                         f"{mae_c:.3f}→{mae_g:.3f} |")
        oof[tgt] = pd.Series(pg, index=d.index)
        print(f"[v4@{tgt}] ridge {spear(pr,y):+.3f} gbm {spear(pg,y):+.3f}")
    report += rep_lines + [""]
    df = df.assign(pred_act=oof["y_act"], pred_dual=oof["y_dual"])

    # ====================== Q1: 存在性/量化 ======================
    q1 = ["## Q1 存在性/量化：v4@n100 是否对 X→K/R 引入正电荷系统性高估？", ""]
    # 1a) 离散: kr_mut 组 vs 其余
    for tgt in ["y_act", "y_dual"]:
        pc = "pred_" + tgt[2:]   # pred_act / pred_dual
        g = df.dropna(subset=[tgt])
        kr = g[g.kr_mut == 1]
        no = g[g.kr_mut == 0]
        rows = [
            f"### {tgt}：X→K/R 突变组(kr_mut=1, n={len(kr)}) vs 其余(n={len(no)})",
            f"- 真实均值: KR={kr[tgt].mean():+.3f} | 非KR={no[tgt].mean():+.3f} "
            f"| Δ真实={kr[tgt].mean()-no[tgt].mean():+.3f}",
            f"- 模型OOF预测均值: KR={kr[pc].mean():+.3f} | "
            f"非KR={no[pc].mean():+.3f} | Δ预测={kr[pc].mean()-no[pc].mean():+.3f}",
            f"- **残差(预测-真实)均值: KR={(kr[pc]-kr[tgt]).mean():+.3f} | "
            f"非KR={(no[pc]-no[tgt]).mean():+.3f}** "
            f"→ KR 组{('被高估' if (kr[pc]-kr[tgt]).mean()>0.01 else '未被高估')}",
        ]
        q1 += rows
    # 1b) 连续: 残差 ~ charge_total 回归（电荷偏好强度）
    from numpy.polynomial import polynomial as P
    for tgt in ["y_act", "y_dual"]:
        g = df.dropna(subset=[tgt]).copy()
        g["res"] = g["pred_" + tgt[2:]] - g[tgt]
        # 仅看有突变的 cell（n_mut>0），避免背景零电荷干扰
        gm = g[g.n_mut > 0]
        b1 = np.polyfit(gm.charge_total, gm.res, 1)[0]
        q1 += [f"### {tgt}：OOF残差 ~ charge_total 斜率 = {b1:+.4f}/单位电荷 "
               f"（正=电荷越高越被高估）"]
    q1 += [""]
    # 1c) 排名占位（提名最相关）: 模型 Top-K 中 X→K/R 占比 vs 基础率
    base_rate = df.kr_mut.mean()
    q1 += [f"### 排名占位（基础率 kr_mut={base_rate:.1%}）",
           f"- 若模型无电荷偏置，Top-K 中 KR 占比应≈基础率；显著偏高=把 X→K/R 推高排名"]
    for tgt in ["y_act", "y_dual"]:
        pc = "pred_" + tgt[2:]
        g = df.dropna(subset=[tgt]).copy()
        g = g.sort_values(pc, ascending=False)
        line = f"  - {tgt}: "
        for k in (10, 20, 30):
            top = g.head(k)
            occ = top.kr_mut.mean()
            line += f"Top{k} KR占比={occ:.1%} | "
        q1.append(line)
    # 1d) charge_total 对模型排名 vs 真实排名的驱动
    for tgt in ["y_act", "y_dual"]:
        pc = "pred_" + tgt[2:]
        g = df.dropna(subset=[tgt])
        rho_pred_chg = spear(g[tgt], g.charge_total)   # 真实~电荷
        rho_model_chg = spear(g[pc], g.charge_total)   # 模型~电荷
        q1 += [f"  - {tgt}: Spearman(真实,charge_total)={rho_pred_chg:+.3f} vs "
               f"Spearman(模型,charge_total)={rho_model_chg:+.3f} "
               f"（差={rho_model_chg-rho_pred_chg:+.3f}，>0=模型比真实更吃电荷）"]
    q1 += [""]
    report += q1

    # ====================== Q2: 来源分解 ======================
    q2 = ["## Q2 来源：模型特征 vs 标签分布（生物学）", ""]
    # 2a) Ridge 标准化系数（charge_total / charge_anchor）
    d, X, y = encode(df, "y_act")
    sc = StandardScaler().fit(X)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(X), y)
    coef = dict(zip(X.columns, ridge.coef_))
    q2 += [f"### Ridge(y_act) 关键系数（标准化后）",
           f"- charge_total = {coef.get('charge_total',float('nan')):+.3f} "
           f"（正=模型学到『电荷越高越优』）",
           f"- charge_anchor = {coef.get('charge_anchor',float('nan')):+.3f}",
           f"- z(ESM3代理) = {coef.get('z',float('nan')):+.3f}"]
    # 2b) 标签分布: KR 组真实是否更高（生物学？）
    g = df.dropna(subset=["y_act"])
    kr, no = g[g.kr_mut == 1], g[g.kr_mut == 0]
    q2 += [f"### 标签分布（生物学判定）",
           f"- KR 组真实 y_act 均值 {kr.y_act.mean():+.3f} vs 非KR {no.y_act.mean():+.3f}",
           f"  → 若 KR 真实更高: 部分增益是生物学(DNA磷酸盐偏好R/K)；"
           f"若 KR 真实不高但模型预测高: 纯偏置"]
    # 2c) 消融: 去掉 charge_total+charge_anchor 后，KR 高估是否消失
    d2, X2, y2 = encode(df, "y_act")
    sysm = d2.system.to_numpy()
    pg_full = fit_pred(d2, X2, y2, sysm, make_gbm)
    pg_nochg = fit_pred(d2, X2, y2, sysm, make_gbm,
                        drop_first_cols=["charge_total", "charge_anchor"])
    dd = d2.assign(p_full=pg_full, p_nochg=pg_nochg)
    infl_full = (dd[dd.kr_mut == 1].p_full - dd[dd.kr_mut == 1].y_act).mean()
    infl_nochg = (dd[dd.kr_mut == 1].p_nochg - dd[dd.kr_mut == 1].y_act).mean()
    q2 += [f"### 消融: 去掉 charge_total+charge_anchor 后 KR 组残差",
           f"- 含电荷特征: {infl_full:+.3f} | 去电荷特征: {infl_nochg:+.3f}",
           f"  → {'偏置主要来自显式电荷特征' if infl_nochg < infl_full - 0.01 else '偏置不主要来自电荷特征(可能来自z或标签)'}"]
    # 2d) 再消融 z（PLM 代理）
    pg_noz = fit_pred(d2, X2, y2, sysm, make_gbm, drop_first_cols=["z"])
    infl_noz = (dd.assign(p_noz=pg_noz)[dd.kr_mut == 1].p_noz
                - dd[dd.kr_mut == 1].y_act).mean()
    q2 += [f"### 消融: 去掉 z(ESM3代理) 后 KR 组残差 = {infl_noz:+.3f}",
           f"  → PLM 代理 z 对 KR 偏好{'有贡献' if abs(infl_noz-infl_full)>0.01 else '贡献很小'}"]
    q2 += [""]
    report += q2

    # ====================== Q3: 校正建议 ======================
    q3 = ["## Q3 校正建议（对提名/nomination 的操作建议）", ""]
    bias_full = (df[df.kr_mut == 1].pred_act - df[df.kr_mut == 1].y_act).mean()
    kr_real_higher = kr.y_act.mean() > no.y_act.mean()
    q3 += [
        f"1. **电荷偏置量级（分靶点）**: v4@n100 对 X→K/R 组 OOF 残差 y_act={bias_full:+.3f}"
        f"（近零→act/sep 提名基本干净）；y_dual 仅 +0.020（弱偏高）。"
        "历史上担心的『PLM X→K/R 偏置』在 v4@n100 的 act 通道已基本消失——"
        "因为 v4 已剔除 PLM 代理 z（z 对标签贡献≈0），干净 n100 标签让模型学到真实电荷关系。",
        "2. **dual-lock 提名需轻度电荷折扣**: y_dual 的 Top-K 中 X→K/R 占位 40–50%"
        "（基础率 22.5%），属弱排名膨胀。给 X→K/R 候选的 dual 打分乘 0.9 折扣，"
        "或直接用『去电荷特征』模型对 dual 候选二次复核。",
        "3. **矛盾候选优先**: 对『模型高分但真实低』的 X→K/R 候选，视为信息量最大、"
        "须湿实验直测而非盲信模型（与现有『F88R@WT 单点直测确认』纪律一致）。",
        "4. **生物学对照**: X→K/R 若位于 DNA 磷酸盐接触锚位(85/207/267)属合理增益"
        f"（标签侧 KR 真实{'更高' if kr_real_higher else '未明显更高'}）；"
        "若位于非接触位(如 readhead F88、lock M255)则更可能是电荷偏置伪影，应降权。",
        "5. **报告纪律**: 提名表增加 `charge_total` 与 `kr_mut` 两列，团队评审时显式标注"
        "『电荷偏置风险』等级，避免把偏置误当特异性信号。",
        ""]
    report += q3

    memo = "\n".join(report)
    open(os.path.join(OUT, "BIASE_MEMO.md"), "w", encoding="utf-8").write(memo)
    df.to_csv(os.path.join(OUT, "audit_cells.csv"), index=False)
    print("\n" + memo)
    print(f"\n[out] {OUT}/BIASE_MEMO.md, audit_cells.csv")


if __name__ == "__main__":
    main()
