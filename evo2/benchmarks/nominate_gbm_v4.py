# -*- coding: utf-8 -*-
"""组合空间提名（p2 ProMEP 式）—— 用 v4 GBM 集成（当前最优模型）对设计空间打分提名。

池: 背景 ∈ {s13_c1, TrackF_r1, WT} × 热点 ∈ {F88, M255, Y217} × 20 AA × 2 条件
输出: combo_nomination_gbm.csv + md（top-12 UCB 提名）
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_sep_model_v4 import (AA, CAT_FEATS, NUM_FEATS, P2S, WT_SEQ,
                                genotype_features, load_sequences, make_gbm)
from sklearn.ensemble import HistGradientBoostingRegressor

OUT = "A:/claudework/evo2/results/sep_model_v4"
HOTSPOTS = {88: "F88", 255: "M255", 217: "Y217"}
# 词汇表限制: 只提名训练数据中实际出现过的 AA（词汇表外哑变量全零 → 退化预测）
VOCAB = {88: set("FKRWY"), 255: set("IAKM"), 217: set("YFR")}
COND_SEQ = {"S1_G17": "TCATGAGCAGTTTTTTGTTTTTTT",
            "OFF_T_G17": "TTGCTATTTTTTATTGCTTTGAGT"}


def main():
    D0 = pd.read_csv(f"{OUT}/cells.csv")
    seqs = load_sequences()
    # z 缓存（v4 的 zcache 语义: 系统 z）
    zvals = {s: D0.loc[D0.system == s, "z"].iloc[0] for s in D0.system.unique() if
             (D0.system == s).any()}

    def zc(name, seq):
        return zvals.get(name, 0.0)

    # 训练集（全数据, act 有标签的行）
    tr = D0.dropna(subset=["y_act"]).copy()
    Xtr = pd.get_dummies(tr[NUM_FEATS + CAT_FEATS], columns=CAT_FEATS, drop_first=True)
    ytr = tr.y_act.to_numpy(float)

    # 候选池
    tested = set(D0.system.unique())
    cands = []
    for bname in ("s13_c1", "TrackF_r1", "WT"):
        base = seqs[bname]
        for pdb, hname in HOTSPOTS.items():
            for aa in sorted(VOCAB[pdb]):
                if aa == base[P2S(pdb)]:
                    continue
                name = f"{hname}{aa}@{bname}"
                if name in tested:
                    continue
                s = list(base)
                s[P2S(pdb)] = aa
                seq = "".join(s)
                for cond in COND_SEQ:
                    g = genotype_features(name, seq, lambda n, s_: zc(n, s_))
                    nt10, nt17, nt23 = COND_SEQ[cond][9], COND_SEQ[cond][16], COND_SEQ[cond][22]
                    g.update(cond=cond, nt10=nt10, nt17=nt17, nt23=nt23)
                    cands.append(dict(name=name, background=bname, hotspot=hname,
                                      mut_aa=aa, condition=cond, **g))
    C = pd.DataFrame(cands)
    Xc = pd.get_dummies(C[NUM_FEATS + CAT_FEATS], columns=CAT_FEATS, drop_first=True) \
        .reindex(columns=Xtr.columns, fill_value=0)
    print(f"[pool] {len(C)//2} 候选 × 2 条件; 设计矩阵 {Xc.shape}", flush=True)

    # 8 seeds GBM 集成
    preds = []
    for seed in range(8):
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.06,
                                          l2_regularization=5.0, min_samples_leaf=8,
                                          random_state=seed)
        m.fit(Xtr.to_numpy(float), ytr)
        preds.append(m.predict(Xc.to_numpy(float)))
    P = np.mean(preds, 0)
    S = np.std(preds, 0)
    C["act_pred"] = P
    C["act_σ"] = S
    df = C.pivot_table(index=["name", "background", "hotspot", "mut_aa"],
                       columns="condition", values=["act_pred", "act_σ"]).reset_index()
    df.columns = ["name", "background", "hotspot", "mut_aa",
                  "act_off", "act_s1", "sig_off", "sig_s1"]
    df["sep"] = df.act_s1 - df.act_off
    df["sep_σ"] = np.sqrt(df.sig_s1 ** 2 + df.sig_off ** 2)
    df["UCB"] = df.sep + 0.5 * df.sep_σ
    df = df.sort_values("UCB", ascending=False)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/combo_nomination_gbm.csv", index=False)
    top = df.head(12)
    lines = ["# 组合空间提名 top-12（v4 GBM 8-seed 集成; UCB = sep + 0.5σ; p2 ProMEP 式）", "",
             f"池: 3 背景 × 3 热点 × 20 AA × 2 条件 = {len(df)} 候选（排除已实测系统）", "",
             "| 候选 | 背景 | 位点 | 突变 | act(S1) | sep | UCB |",
             "|---|---|---|---|---:|---:|---:|"]
    for _, r in top.iterrows():
        lines.append(f"| {r['name']} | {r['background']} | {r['hotspot']} | {r['mut_aa']} "
                     f"| {r.act_s1:.3f} | {r.sep:+.3f} | {r.UCB:+.3f} |")
    lines += ["", "## 校验: 已实测最优（s13_c1 内部参考）",
              f"- s13_c1 真实 Δact = +0.45（六候选判读）; 模型内插应≈此值"]
    txt = "\n".join(lines)
    open(f"{OUT}/combo_nomination_gbm.md", "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
