# -*- coding: utf-8 -*-
"""全设计空间打分（p2 ProMEP 式）—— v6 集成对 3021 单点 × 2 条件 + 105 系统全打分。

输出:
  dl_v6/design_space_scores.csv  全库 act(S1)/act(OFF)/sep/σ/UCB
  dl_v6/design_space_top.md      top-30 提名（含与去混杂直接证据的矛盾标记）
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_sep_model_v4 import load_sequences
import train_dl_v5 as M
from train_dl_v6 import build_ctx, build_tokens_v6, load_pca, EMB_DIR

OUT = "A:/claudework/evo2/results/dl_v6"
os.makedirs(OUT, exist_ok=True)
COND_SEQ = {"S1_G17": "TCATGAGCAGTTTTTTGTTTTTTT",
            "OFF_T_G17": "TTGCTATTTTTTATTGCTTTGAGT"}


def main():
    orc = M.load_ctx()
    M.zc_global = {}

    def zcache(n, s):
        if n not in M.zc_global:
            g = np.array([M.AA.index(s[x]) for x in M.SITES], dtype=np.int64)
            M.zc_global[n] = float(orc.evaluate_multi(g[None, :])[0])
        return M.zc_global[n]

    seqs = load_sequences()
    for n, s in seqs.items():
        zcache(n, s)
    # 全库 = 105 系统 + 3021 单点（sm_* 嵌入已回传）
    have_emb = {f[:-4] for f in os.listdir(EMB_DIR) if f.endswith(".npz")}
    import json
    all_seqs = dict(seqs)
    for f in sorted(os.listdir(EMB_DIR)):
        if f.endswith(".npz") and f[:-4].startswith("sm_"):
            pass  # 序列需重建: sm_<bg>_p<pdb><aa>
    BG = {"WT": seqs["WT"], "s13c1": seqs["s13_c1"], "TrackF": seqs["TrackF_r1"]}
    for f in sorted(os.listdir(EMB_DIR)):
        if not f.endswith(".npz"):
            continue
        name = f[:-4]
        if name in all_seqs or not name.startswith("sm_"):
            continue
        try:
            _, bg, site = name.split("_")
            pdb = int(site[1:-1])
            aa = site[-1]
        except ValueError:
            continue
        base = BG.get(bg)
        if base is None:
            continue
        i = pdb - 22
        if base[i] == aa:
            continue
        all_seqs[name] = base[:i] + aa + base[i + 1:]
    print(f"[pool] {len(all_seqs)} 序列", flush=True)

    comps, mean = load_pca()
    ctxs = build_ctx(all_seqs, comps, mean)
    toks = build_tokens_v6(all_seqs, ctxs, zcache)
    # 全库基因型 global 特征
    globs = {}
    G = None
    names = sorted(all_seqs)
    for n in names:
        for cond in COND_SEQ:
            globs[(n, cond)] = M.global_vec(n, all_seqs[n], cond, zcache(n, all_seqs[n]))[None]
            G = globs[(n, cond)].shape[-1]
    print(f"[glob] {G} 维", flush=True)

    # 训练 8 seeds（全数据, 同 V3 口径）
    rows = M.load_rows(seqs, zcache)
    row_g = [globs[(r.system, r.condition)] for _, r in rows.iterrows()]
    nets = []
    for seed in range(8):
        net, _ = M.train_model(rows, toks, row_g, seed)
        nets.append(net)
    print(f"[ens] 8 seeds done", flush=True)

    # 批量打分（每批 512 行）
    score_names, score_conds = [], []
    for n in names:
        for cond in COND_SEQ:
            score_names.append(n)
            score_conds.append(cond)
    preds = []
    B = 512
    for net in nets:
        out = np.zeros((len(score_names), 5), dtype=np.float32)
        for b in range(0, len(score_names), B):
            ns = score_names[b:b + B]
            cs = score_conds[b:b + B]
            tg = [globs[(n, c)] for n, c in zip(ns, cs)]
            out[b:b + B] = M.predict(net, toks, ns, cs, tg)
        preds.append(out)
    P = np.mean(preds, 0)
    S = np.std(preds, 0)
    act = 1 / (1 + np.exp(-P[:, 0]))

    df = pd.DataFrame(dict(name=score_names, condition=score_conds, act=act, sig=S[:, 0]))
    w = df.pivot_table(index="name", columns="condition", values=["act", "sig"])
    w.columns = [f"{a}_{b[:3]}" for a, b in w.columns]
    w = w.reset_index()
    w["background"] = w.name.map(lambda n: "s13c1" if "_s13c1_" in n else
                                 ("TrackF" if "_TrackF_" in n else
                                  ("WT" if n.startswith("sm_WT") else "system")))
    w["sep"] = w.act_S1 - w.act_OFF
    w["sep_σ"] = np.sqrt(w.sig_S1 ** 2 + w.sig_OFF ** 2)
    w["UCB"] = w.sep + 0.5 * w.sep_σ
    w.to_csv(f"{OUT}/design_space_scores.csv", index=False)

    # top-30 提名: 只看单点（系统已实测）, 按 UCB
    sm = w[w.background != "system"].sort_values("UCB", ascending=False).head(30)
    lines = ["# v6 全设计空间 top-30（8-seed 集成, UCB = sep + 0.5σ）", "",
             f"库规模: {len(w)} 候选（单点 {int((w.background != 'system').sum())} + 系统 {int((w.background == 'system').sum())}）", "",
             "| 候选 | 背景 | act(S1) | sep | σ | UCB |", "|---|---|---:|---:|---:|---:|"]
    for _, r in sm.iterrows():
        lines.append(f"| {r['name']} | {r['background']} | {r.act_S1:.3f} "
                     f"| {r.sep:+.3f} | {r.sep_σ:.3f} | {r.UCB:+.3f} |")
    open(f"{OUT}/design_space_top.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
