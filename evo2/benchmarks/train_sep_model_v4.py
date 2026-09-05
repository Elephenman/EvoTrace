# -*- coding: utf-8 -*-
"""统一 Boltz 标注 → 条件特异性模型 v4（用好全部已标注训练集）。

背景:
  gate v3 只用了 22 行去混杂变体级均值（3 位点查找表）；calib_proxy 只用了 40 系统
  做 1 维 isotonic。实际上手头的已标注数据远多于此:
    - 旧战役 per_model_metrics.csv: 3240 逐模型行 × 3 条件（S1/OFF/GCA）× 36 系统
    - evo2 六候选/去混杂/V2-A 指纹: 675 逐模型行 × 2 条件
    - v3cand 校准集: 9 系统 act 标签
  本脚本把这些合并为 统一单元表（系统 × DNA 条件 → 连续标签），训练
  条件感知模型（基因型 + DNA 条件 → 激活/双锁），并与三个基线对比:
    B0 常数;  B1 代理 z（现状: 条件无关）;  B2 gate v3 表（条件无关, 仅 3 位点）
  验证:
    V1 按系统 LOO —— 预测未见系统的 cell 标签 & 靶/非靶分离 sep
    V2 deconf-22 composite LOO —— 与 gate v3 的 +0.583 同口径对比
    V3 V2-A 留出上位性检验 —— 训练集不含 M255I 增量系统, 预测其 sep 增量方向
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
# NOTE: scipy / sklearn 延迟导入（仅在拟合/打分函数内使用），使本模块顶层轻量化，
# 避免被 train_dl_v5 间接导入时把重型库带进 v7 的并行 worker 进程。

RES = "A:/claudework/evo2/results"
OUT = os.path.join(RES, "sep_model_v4")
os.makedirs(OUT, exist_ok=True)

WT_SEQ = open("A:/claudework/ppri_evo/inputs/wt_254.fasta").read().splitlines()[1].strip()
AA = "ACDEFGHIKLMNPQRSTVWY"
P2S = lambda pdb: pdb - 22          # pdb 残基号 -> 序列 0-based 索引
COND_SEQ = {
    "S1_G17": "TCATGAGCAGTTTTTTGTTTTTTT",
    "OFF_T_G17": "TTGCTATTTTTTATTGCTTTGAGT",
    "GCA": "TCATGAGCATTTTTTTGTTTTTTT",
}
COND_BASE = {c: (s[9], s[16], s[22]) for c, s in COND_SEQ.items()}  # nt10/17/23
CLUSTER = {  # pdb -> 簇
    88: "readhead", 253: "lock", 217: "lock", 255: "lock",
    85: "anchor", 207: "anchor", 267: "anchor",
    92: "cat", 93: "cat", 96: "cat", 123: "cat",
}
CHARGE = {"K": 1, "R": 1, "H": 0.5, "D": -1, "E": -1}


def mut_of(seq):
    return [(i, a) for i, a in enumerate(seq) if a != WT_SEQ[i]]


def apply_pdb(seq, pdb, aa):
    i = P2S(pdb)
    return seq[:i] + aa + seq[i + 1:]


# ---------------------------------------------------------------- 序列源
def load_sequences():
    seqs = {}
    p = "A:/claudework/ppri_evo/results/all_candidate_sequences.json"
    seqs.update(json.load(open(p)))
    p2 = "A:/claudework/ppri_evo/results/reference_sequences.json"
    for k, v in json.load(open(p2)).items():
        seqs.setdefault(k, v)
    # v3cand 9 系统（突变列表已知）
    for spec in ["K216R", "Q120V", "Y170F", "F88Y_Y170F", "K216R_Y170F_Q120V",
                 "R85A_R207A_R267A", "K216R_Q120V", "K216R_Q120V_F88Y_Y170F"]:
        s = WT_SEQ
        for pdb, aa in [(216, "R"), (120, "V"), (170, "F"), (88, "Y"),
                        (85, "A"), (207, "A"), (267, "A")]:
            if f"{pdb}{aa}" in spec.split("_"):
                s = apply_pdb(s, pdb, aa)
        seqs[spec] = s
    # 去混杂 22 变体: s13_c1 背景 × F88/M255/Y217 指定残基
    base = seqs["s13_c1"]
    import re
    for m88 in "FKRWY":
        for m255 in "AIKM":
            for y217 in "YFR":
                name = f"F88{m88}_M255{m255}_Y217{y217}"
                s = apply_pdb(apply_pdb(apply_pdb(base, 88, m88), 255, m255), 217, y217)
                seqs[name] = s
    # V2-A: 亲本 + M255I
    for par in ("TrackF_r1", "HQL2"):
        seqs[f"{par}_M255I"] = apply_pdb(seqs[par], 255, "I")
    return seqs


# ---------------------------------------------------------------- 标签源
def agg_cells(df, keys, cols, rename):
    g = df.groupby(keys)
    out = g.size().rename("n").to_frame()
    for c in cols:
        out[rename.get(c, c)] = g[c].mean()
    return out.reset_index()


def load_cells():
    cells = []
    # 1) 旧战役 3240 行
    pm = pd.read_csv("A:/claudework/ppri_evo/results/per_model_metrics.csv")
    old = agg_cells(pm, ["system", "condition"],
                    ["act", "dual", "d17", "d23", "hexxh"],
                    {"d17": "y_d17", "d23": "y_d23", "hexxh": "y_dcat",
                     "act": "y_act", "dual": "y_dual"})
    old["source"] = "old"
    cells.append(old)

    def load_fp(path, sys_col):
        df = pd.read_csv(path)
        df["cond"] = df["cond"].astype(str).map(
            lambda c: {"S1": "S1_G17", "OFF": "OFF_T_G17"}.get(c, c))
        if sys_col not in df.columns:  # deconf: 从 pred 名解析变体
            pat = df["pred"].str.replace(r"^(d_)?", "", regex=True)
            df[sys_col] = pat.str.replace(r"_(S1_G17|OFF_T_G17|GCA)$", "", regex=True)
        out = agg_cells(df, [sys_col, "cond"],
                        ["act", "dual", "d_act", "d_read17", "d_253_23", "iface", "ntcov"],
                        {"d_act": "y_dcat", "d_read17": "y_d17", "d_253_23": "y_d23",
                         "act": "y_act", "dual": "y_dual",
                         "iface": "y_iface", "ntcov": "y_ntcov"})
        out = out.rename(columns={sys_col: "system", "cond": "condition"})
        return out

    six = load_fp(f"{RES}/boltz_six/contact_fingerprint.csv", "cand"); six["source"] = "six"
    dc = load_fp(f"{RES}/boltz_six/deconf_fingerprint.csv", "variant"); dc["source"] = "deconf"
    v2a = load_fp(f"{RES}/boltz_v2a/contact_fingerprint.csv", "cand"); v2a["source"] = "v2a"
    cells += [six, dc, v2a]

    # 2) v3cand 校准集（仅 act）
    cal = pd.read_csv("A:/claudework/evo2/esm3/calib_proxy_dataset.csv")
    rows = []
    for _, r in cal[cal.src == "v3cand"].iterrows():
        rows.append(dict(system=r["name"], condition="S1_G17", source="v3cand", n=np.nan, y_act=r["act_s1"]))
        rows.append(dict(system=r["name"], condition="OFF_T_G17", source="v3cand", n=np.nan, y_act=r["act_off"]))
    cells.append(pd.DataFrame(rows))

    allc = pd.concat(cells, ignore_index=True)
    # 去重: 同 (system, condition) 优先新数据（n 更大）
    pri = {"deconf": 0, "v2a": 1, "six": 2, "old": 3, "v3cand": 4}
    allc["pri"] = allc["source"].map(pri)
    allc = allc.sort_values("pri").drop_duplicates(["system", "condition"], keep="first")
    return allc.drop(columns="pri")


# ---------------------------------------------------------------- 特征
CAT_FEATS = ["rh66", "m255", "y217", "r253mut", "anchor_state", "cond", "nt17", "nt23", "nt10"]
NUM_FEATS = ["n_mut", "n_out53", "lock_mut", "anchor_mut", "cat_mut",
             "charge_anchor", "charge_total", "z"]


def genotype_features(name, seq, zcache):
    muts = mut_of(seq)
    at = {i: a for i, a in muts}
    cl = lambda i: CLUSTER.get(i + 22, "other")
    anchor_idx = [P2S(p) for p in (85, 207, 267)]

    def charge_delta(indices=None):
        d = 0.0
        for i, a in muts:
            if indices is None or i in indices:
                d += CHARGE.get(a, 0.0) - CHARGE.get(WT_SEQ[i], 0.0)
        return d

    row = dict(
        rh66=at.get(66, WT_SEQ[66]),                       # F88
        m255=at.get(233, WT_SEQ[233]),                     # M255
        y217=at.get(195, WT_SEQ[195]),                     # Y217
        r253mut="R" if 231 not in at else at[231],         # R253
        anchor_state="_".join(at.get(i, WT_SEQ[i]) for i in anchor_idx),
        n_mut=len(muts),
        n_out53=sum(1 for i, _ in muts if cl(i) == "other"),
        lock_mut=sum(1 for i, _ in muts if cl(i) == "lock"),
        anchor_mut=sum(1 for i, _ in muts if cl(i) == "anchor"),
        cat_mut=sum(1 for i, _ in muts if cl(i) == "cat"),
        charge_anchor=charge_delta(set(anchor_idx)),
        charge_total=charge_delta(),
        z=zcache(name, seq),
    )
    return row


def build_design(cells, seqs, zcache):
    rows = []
    for _, c in cells.iterrows():
        name = c["system"]
        if name not in seqs:
            continue
        g = genotype_features(name, seqs[name], zcache)
        nt10, nt17, nt23 = COND_BASE[c["condition"]]
        g.update(system=name, condition=c["condition"], cond=c["condition"],
                 nt10=nt10, nt17=nt17, nt23=nt23,
                 source=c["source"], y_act=c.get("y_act", np.nan),
                 y_dual=c.get("y_dual", np.nan), y_dcat=c.get("y_dcat", np.nan),
                 y_d17=c.get("y_d17", np.nan), y_d23=c.get("y_d23", np.nan),
                 y_iface=c.get("y_iface", np.nan), y_ntcov=c.get("y_ntcov", np.nan))
        rows.append(g)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 模型/验证
def encode(df, target):
    d = df.dropna(subset=[target]).copy()
    X = pd.get_dummies(d[NUM_FEATS + CAT_FEATS], columns=CAT_FEATS, drop_first=True)
    return d, X, d[target].to_numpy(float)


def make_ridge():
    from sklearn.linear_model import RidgeCV
    return Pipeline_like(RidgeCV(alphas=np.logspace(-2, 4, 25)))


class Pipeline_like:
    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler().fit(X)
        self.est.fit(self.sc.transform(X), y)
        return self

    def predict(self, X):
        return self.est.predict(self.sc.transform(X))


def make_gbm():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_depth=3, max_iter=250,
                                         learning_rate=0.06, l2_regularization=5.0,
                                         min_samples_leaf=8, random_state=0)


def loo_system(d, X, y, systems, model_factory):
    """按系统留一: 每次留出该系统全部条件 cell。"""
    pred = np.full(len(y), np.nan)
    for s in pd.unique(systems):
        te = systems == s
        m = model_factory().fit(X[~te], y[~te])
        pred[te] = m.predict(X[te])
    return pred


def spear(a, b):
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return np.nan
    return spearmanr(a[ok], b[ok]).statistic


def main():
    seqs = load_sequences()
    cells = load_cells()
    sys.path.insert(0, "A:/claudework/evo2/esm3")
    try:
        from ppri_surrogate_v3 import PprISurrogateV3
        orc = PprISurrogateV3()
        zc = {}

        def zcache(name, seq):
            if name not in zc:
                g = np.array([AA.index(seq[s]) for s in orc.sites], dtype=np.int64)
                zc[name] = float(orc.evaluate_multi(g[None, :])[0])
            return zc[name]
    except Exception as e:  # 代理不可用则置 0
        print(f"[warn] surrogate z 不可用: {e}")
        zcache = lambda name, seq: 0.0

    D = build_design(cells, seqs, zcache)
    skipped = sorted(set(cells.system) - set(D.system))
    print(f"[data] cells={len(D)}  systems={D.system.nunique()}  "
          f"(skipped no-seq: {len(skipped)} {skipped[:8]})")
    print(f"[data] per source: {D.source.value_counts().to_dict()}")
    D.to_csv(f"{OUT}/cells.csv", index=False)

    report = ["# 条件特异性模型 v4 —— 统一 Boltz 标注实验", "",
              f"cells={len(D)}, systems={D.system.nunique()}, "
              f"skipped(no-seq)={len(skipped)}", ""]

    # ---------- V1: cell 级 LOO ----------
    report += ["## V1 按系统 LOO（cell 级标签）", "",
               "| target | n | Ridge rho | GBM rho | GBM MAE | 常数 MAE | z 基线 rho |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for target in ["y_act", "y_dual", "y_dcat"]:
        d, X, y = encode(D, target)
        if len(d) < 20:
            continue
        sysm = d.system.to_numpy()
        pr = loo_system(d, X.to_numpy(float), y, sysm, make_ridge)
        pg = loo_system(d, X.to_numpy(float), y, sysm, make_gbm)
        mae_g = float(np.nanmean(np.abs(pg - y)))
        # 常数基线: 留一均值
        pconst = np.array([np.nanmean(y[sysm != s]) for s in sysm])
        mae_c = float(np.nanmean(np.abs(pconst - y)))
        rho_z = spear(d.z.to_numpy(float), y)
        report.append(f"| {target} | {len(d)} | {spear(pr, y):+.3f} | "
                      f"{spear(pg, y):+.3f} | {mae_g:.3f} | {mae_c:.3f} | {rho_z:+.3f} |")
        print(f"[V1] {target}: ridge {spear(pr, y):+.3f}  gbm {spear(pg, y):+.3f}  "
              f"(z {rho_z:+.3f})")

    # ---------- V1b: sep（靶-非靶分离）----------
    # 用 GBM y_act 的 LOO 预测做 sep_pred; 基线: z / gate(条件无关) 只能给系统级排序
    d, X, y = encode(D, "y_act")
    sysm = d.system.to_numpy()
    pg = loo_system(d, X.to_numpy(float), y, sysm, make_gbm)
    d = d.assign(pred_act=pg)
    pairs = []
    for s, g in d.groupby("system"):
        if {"S1_G17", "OFF_T_G17"} <= set(g.condition):
            t = g.set_index("condition")
            pairs.append(dict(system=s, sep_true=t.loc["S1_G17", "y_act"] - t.loc["OFF_T_G17", "y_act"],
                              sep_pred=t.loc["S1_G17", "pred_act"] - t.loc["OFF_T_G17", "pred_act"],
                              z=g.z.iloc[0]))
    P = pd.DataFrame(pairs)
    rho_sep = spear(P.sep_pred.to_numpy(), P.sep_true.to_numpy())
    rho_z = spear(P.z.to_numpy(), P.sep_true.to_numpy())
    report += ["", "## V1b 靶/非靶分离 sep = act(S1) − act(OFF)（系统级, LOO 预测差）", "",
               f"- n={len(P)}  系统; GBM sep Spearman **{rho_sep:+.3f}**; "
               f"现状基线(代理 z 排序) {rho_z:+.3f}",
               f"- sep 符号正确率: {float((np.sign(P.sep_pred) == np.sign(P.sep_true)).mean()):.2f}"]
    P.to_csv(f"{OUT}/sep_loo.csv", index=False)

    # ---------- V2: deconf-22 composite, 与 gate v3 同口径 ----------
    ds = pd.read_csv("A:/claudework/out/boltz_six/deconf_stats.csv")

    def rank_norm(v):
        o = np.argsort(np.argsort(v))
        return o / (len(v) - 1)

    comp = 0.40 * rank_norm(ds.dual_S1.to_numpy()) + 0.30 * rank_norm(ds.z_act.to_numpy()) \
        + 0.30 * rank_norm(ds.z_iface.to_numpy())
    dd = D[(D.source == "deconf") & (D.condition == "S1_G17")].set_index("system") \
        .reindex(ds.variant.to_numpy())
    feat = pd.get_dummies(dd[NUM_FEATS + CAT_FEATS].reset_index(drop=True),
                          columns=CAT_FEATS, drop_first=True).to_numpy(float)
    yv = comp
    n = len(yv)
    pred_r = np.full(n, np.nan)
    pred_g = np.full(n, np.nan)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        pred_r[i] = make_ridge().fit(feat[m], yv[m]).predict(feat[[i]])[0]
        pred_g[i] = make_gbm().fit(feat[m], yv[m]).predict(feat[[i]])[0]
    report += ["", "## V2 deconf-22 composite LOO（gate v3 同口径, 其基准 +0.583）", "",
               f"- Ridge(全特征): {spear(pred_r, yv):+.3f}   GBM(全特征): {spear(pred_g, yv):+.3f}"]

    # ---------- V3: V2-A 留出上位性检验 ----------
    hold = ["TrackF_r1_M255I", "HQL2_M255I"]
    dh, Xh, yh = encode(D[~D.system.isin(hold)], "y_act")
    mh = make_gbm().fit(Xh.to_numpy(float), yh)
    Dv = D[D.system.isin(hold)]
    _, Xv, yv2 = encode(Dv, "y_act")
    Xv = Xv.reindex(columns=Xh.columns, fill_value=0)   # 对齐训练设计矩阵列
    pv = mh.predict(Xv.to_numpy(float))
    Dv = Dv.assign(pred=pv)
    lines = []
    for par in ("TrackF_r1", "HQL2"):
        g = Dv[Dv.system == f"{par}_M255I"].set_index("condition")
        tp = D[(D.system == par) & D.y_act.notna()].set_index("condition")
        d_true = (tp.loc["S1_G17", "y_act"] - tp.loc["OFF_T_G17", "y_act"]) \
            - 0  # 亲本真实 sep
        d_mut = g.loc["S1_G17", "y_act"] - g.loc["OFF_T_G17", "y_act"]
        d_pred = g.loc["S1_G17", "pred"] - g.loc["OFF_T_G17", "pred"]
        lines.append(f"{par}+M255I: 真实 sep {d_mut:+.2f} (亲本 {d_true:+.2f}, "
                     f"增量 {d_mut - d_true:+.2f}) | 模型预测 sep {d_pred:+.2f} "
                     f"(亲本 z 排序不可给增量)")
    report += ["", "## V3 V2-A 留出上位性检验（训练集不含这两个系统）", ""] + \
              ["- " + x for x in lines]

    txt = "\n".join(report)
    open(f"{OUT}/report.md", "w", encoding="utf-8").write(txt)
    print("\n" + txt)
    print(f"\n[out] {OUT}/report.md, cells.csv, sep_loo.csv")


if __name__ == "__main__":
    main()
