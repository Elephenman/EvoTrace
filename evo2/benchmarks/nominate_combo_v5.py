# -*- coding: utf-8 -*-
"""组合空间提名（p2 ProMEP 式）—— 用 v5 模型对设计空间全打分, 提名 top-k 给 Boltz 复核。

流程:
  1) 全数据训练 8 seeds 集成（同 V3 口径）
  2) 候选池: 背景 ∈ {s13_c1, TrackF_r1, WT} × 扰动位点 ∈ {F88, M255, Y217} × AA ∈ 20
     （3 背景 × 3 位点 × 20 AA = 180 候选 × 2 条件 = 360 打分单元）
  3) 每候选输出: act(S1), sep(S1−OFF), UCB = sep_mean + 0.5·sep_σ (p1 式获取)
  4) 提名 top-10（排除已实测系统）→ 湿实验/Boltz 复核清单
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_sep_model_v4 import AA, P2S, WT_SEQ, load_sequences
import train_dl_v5 as M

OUT = "A:/claudework/evo2/results/dl_v5"
HOTSPOTS = {88: "F88", 255: "M255", 217: "Y217"}


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
    rows = M.load_rows(seqs, M.zc_global)
    toks = M.build_tokens(seqs)
    globs = {}
    for _, r in rows.iterrows():
        globs.setdefault(r.system, {})
        if r.condition not in globs[r.system]:
            globs[r.system][r.condition] = M.global_vec(
                r.system, seqs[r.system], r.condition, M.zc_global[r.system])[None]
    row_globs = [globs[r.system][r.condition] for _, r in rows.iterrows()]

    # 8 seeds 全数据集成
    nets = []
    for seed in range(8):
        net, _ = M.train_model(rows, toks, row_globs, seed)
        nets.append(net)
    print(f"[ens] 8 seeds trained on {len(rows)} rows", flush=True)

    # 候选池构建
    tested = set(rows.system.unique())
    cands = []
    for bname in ("s13_c1", "TrackF_r1", "WT"):
        base = seqs[bname]
        for pdb, hname in HOTSPOTS.items():
            for aa in AA:
                if aa == base[P2S(pdb)]:
                    continue
                name = f"{hname}{aa}@{bname}"
                if name in tested:
                    continue
                s = list(base)
                s[P2S(pdb)] = aa
                s = "".join(s)
                # z 值沿系统级缓存（背景 z 近似——组合 z 需要重算, 此处用背景值）
                for cond in ("S1_G17", "OFF_T_G17"):
                    cands.append(dict(name=name, background=bname, hotspot=hname,
                                      mut_aa=aa, condition=cond, seq=s,
                                      z=M.zc_global.get(bname, 0.0)))
    print(f"[pool] {len(cands)//2} 候选 × 2 条件", flush=True)

    # 打分（逐候选构造 token/glob —— 复用模型接口）
    sys_list = [c["name"] for c in cands]
    # 临时把候选塞进 toks/globs
    new_seqs = {c["name"]: c["seq"] for c in cands}
    for n, s in new_seqs.items():
        zcache(n, s)
    toks_all = dict(toks)
    toks_all.update(M.build_tokens(new_seqs))
    globs_list = []
    for c in cands:
        globs_list.append(M.global_vec(c["name"], c["seq"], c["condition"], c["z"])[None])
    preds = []
    for net in nets:
        preds.append(M.predict(net, toks_all, sys_list,
                               [c["condition"] for c in cands], globs_list))
    P = np.mean(preds, 0)          # [N,5]
    S = np.std(preds, 0)
    df = pd.DataFrame(cands)
    df["act_pred"] = 1 / (1 + np.exp(-P[:, 0]))
    df["act_σ"] = S[:, 0]
    df = df.pivot_table(index=["name", "background", "hotspot", "mut_aa"],
                        columns="condition", values=["act_pred", "act_σ"]).reset_index()
    df.columns = ["name", "background", "hotspot", "mut_aa",
                  "act_off", "act_s1", "sig_off", "sig_s1"]
    df["sep"] = df.act_s1 - df.act_off
    df["sep_σ"] = np.sqrt(df.sig_s1 ** 2 + df.sig_off ** 2)
    df["UCB"] = df.sep + 0.5 * df.sep_σ
    df = df.sort_values("UCB", ascending=False)
    df.to_csv(f"{OUT}/combo_nomination.csv", index=False)

    top = df.head(12)
    lines = ["# 组合空间提名 top-12（p2 ProMEP 式; UCB = sep + 0.5σ, 8 seeds）", "",
             "| 候选 | 背景 | 位点 | 突变 | act(S1) | sep | UCB |",
             "|---|---|---|---|---:|---:|---:|"]
    for _, r in top.iterrows():
        lines.append(f"| {r['name']} | {r['background']} | {r['hotspot']} | {r['mut_aa']} "
                     f"| {r.act_s1:.3f} | {r.sep:+.3f} | {r.UCB:+.3f} |")
    txt = "\n".join(lines)
    open(f"{OUT}/combo_nomination.md", "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
