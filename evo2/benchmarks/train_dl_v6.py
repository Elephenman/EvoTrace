# -*- coding: utf-8 -*-
"""DL v6 —— 变体专属 ESM3 表征版（v5 架构, token 上下文从 WT 换成变体自身嵌入）。

差异（对 v5）: token 的 ESM3 上下文 = 该变体自身序列在位点 j ±2 窗口的嵌入经 v3 同款
PCA(64) 投影（组件复用 out/surrogate_dl_v3/pca_*.npy）——每变体表征专属, 不再共享 WT。
其余（双通路/多任务/验证口径）与 v5 完全一致, 直接对比。
前置: fetch_esm3_v6.py get 已回传 npz 到 out/esm3_embeddings_v6/。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_sep_model_v4 import load_sequences
import train_dl_v5 as M

EMB_DIR = "A:/claudework/out/esm3_embeddings_v6"
PCA_DIR = "A:/claudework/out/surrogate_dl_v3"
OUT = "A:/claudework/evo2/results/dl_v6"
os.makedirs(OUT, exist_ok=True)


def load_pca():
    comps = np.load(f"{PCA_DIR}/pca_components.npy")
    mean = np.load(f"{PCA_DIR}/pca_mean.npy")
    return comps.astype(np.float32), mean.astype(np.float32)


def build_ctx(seqs, comps, mean, win=2):
    """每变体 → [53, 5, 64] 上下文（变体专属）。"""
    ctxs = {}
    for name, seq in seqs.items():
        npz = os.path.join(EMB_DIR, name + ".npz")
        if not os.path.exists(npz):
            continue
        emb = np.load(npz)["emb"].astype(np.float32)      # [254, 1536]
        T = np.zeros((len(M.SITES), 2 * win + 1, comps.shape[0]), dtype=np.float32)
        for j, s in enumerate(M.SITES):
            q0 = max(0, int(s) - win)
            q1 = min(len(seq), int(s) + win + 1)
            block = emb[q0:q1]                             # [w,1536]
            proj = (block - mean) @ comps.T                # [w,64]
            T[j, : proj.shape[0]] = proj
        ctxs[name] = T
    return ctxs


def build_tokens_v6(seqs, ctxs, zcache_fn):
    """v5 的 token 布局, 但 ctx 换成变体专属（缺失变体回退 WT ctx = 零贡献标记）。"""
    toks = {}
    F = 20 + 20 + 1 + 1 + len(M.CLUSTER_LIST) + M.CTX.shape[1] * M.CTX.shape[2] + 1
    wt_name = "WT" if "WT" in ctxs else None
    for name, seq in seqs.items():
        muts = {i: a for i, a in enumerate(seq) if a != M.WT_SEQ[i]}
        c = ctxs.get(name)
        T = np.zeros((len(M.SITES), F), dtype=np.float32)
        for j, s in enumerate(M.SITES):
            wt = M.WT_SEQ[s]
            mu = muts.get(int(s), wt)
            cl = M.CLUSTER.get(int(s) + 22, "other")
            T[j, M.AA.index(wt)] = 1.0
            T[j, 20 + M.AA.index(mu)] = 1.0
            T[j, 40] = j / max(len(M.SITES) - 1, 1)
            T[j, 41] = 1.0 if mu != wt else 0.0
            T[j, 42 + M.CIDX[cl]] = 1.0
            b = 42 + len(M.CLUSTER_LIST)
            if c is not None:
                T[j, b:b + c.shape[1] * c.shape[2]] = c[j].ravel() * M.CTX_SCALE
            T[j, -1] = zcache_fn(name, seq)
        toks[name] = T
    return toks


def main():
    orc = M.load_ctx()
    M.zc_global = {}

    def zcache(name, seq):
        if name not in M.zc_global:
            g = np.array([M.AA.index(seq[s]) for s in M.SITES], dtype=np.int64)
            M.zc_global[name] = float(orc.evaluate_multi(g[None, :])[0])
        return M.zc_global[name]

    seqs = load_sequences()
    for n, s in seqs.items():
        zcache(n, s)
    comps, mean = load_pca()
    ctxs = build_ctx(seqs, comps, mean)
    print(f"[ctx] 变体专属嵌入覆盖 {len(ctxs)}/{len(seqs)} 系统", flush=True)
    if len(ctxs) < len(seqs):
        miss = sorted(set(seqs) - set(ctxs))
        print(f"[warn] 缺失(回退WT): {len(miss)} {miss[:6]}", flush=True)

    rows = M.load_rows(seqs, zcache)
    toks = build_tokens_v6(seqs, ctxs, zcache)
    globs = {}
    for _, r in rows.iterrows():
        globs.setdefault(r.system, {})
        if r.condition not in globs[r.system]:
            globs[r.system][r.condition] = M.global_vec(
                r.system, seqs[r.system], r.condition, M.zc_global[r.system])[None]
    row_globs = [globs[r.system][r.condition] for _, r in rows.iterrows()]
    print(f"[rows] {len(rows)}  [tok] {toks[rows.system.iloc[0]].shape}", flush=True)

    cells = rows.groupby(["system", "condition"]).agg(
        n=("act", "size"), y_act=("act", "mean"), y_dual=("dual", "mean"),
        y_dcat=("dcat", "mean")).reset_index()

    # ---- V1 LOO（断点续跑） ----
    def loo(epochs=36):
        pred = np.full((len(cells), 3), np.nan)
        ckpt = os.path.join(OUT, "loo_ckpt.npz")
        done = {}
        if os.path.exists(ckpt):
            done = dict(np.load(ckpt, allow_pickle=True))
            print(f"[ckpt] 恢复 {len(done)} 折", flush=True)
        import time
        t0 = time.time()
        for fi, s in enumerate(sorted(cells.system.unique())):
            if f"f{fi}" in done:
                pred[:, :] = done[f"f{fi}"]
                continue
            te = (cells.system == s).to_numpy()
            tr = rows[~rows.system.isin([s])].reset_index(drop=True)
            tr_g = [globs[r.system][r.condition] for _, r in tr.iterrows()]
            net, scalers = M.train_model(tr, toks, tr_g, 0, epochs=epochs)
            te_g = [globs[r.system][r.condition] for _, r in cells[te].iterrows()]
            pr = M.predict(net, toks, cells.system[te].tolist(),
                           cells.condition[te].tolist(), te_g)
            pred[te] = M.unnorm(pr, scalers)[:, :3]
            done[f"f{fi}"] = pred.copy()
            np.savez(ckpt, **done)
            if fi % 10 == 0 or fi == len(cells.system.unique()) - 1:
                print(f"[LOO] {fi+1}/{len(cells.system.unique())} ({time.time()-t0:.0f}s)", flush=True)
        return pred

    pred = loo()
    report = ["# DL v6 —— 变体专属 ESM3 表征版结果", "",
              f"ctx 覆盖 {len(ctxs)}/{len(seqs)}; rows={len(rows)}; cells={len(cells)}", "",
              "## V1 按系统 LOO（v5: +0.552/+0.682/+0.639; v4 GBM: +0.638/+0.718/+0.705）", ""]
    for k, c in enumerate(["y_act", "y_dual", "y_dcat"]):
        r = M.spear(pred[:, k], cells[c].to_numpy())
        print(f"[V1] {c}: {r:+.3f}", flush=True)
        report.append(f"- {c}: **{r:+.3f}**")

    cd = cells.assign(p_act=pred[:, 0])
    P = []
    for s, g in cd.groupby("system"):
        if {"S1_G17", "OFF_T_G17"} <= set(g.condition):
            t = g.set_index("condition")
            P.append((t.loc["S1_G17", "y_act"] - t.loc["OFF_T_G17", "y_act"],
                      t.loc["S1_G17", "p_act"] - t.loc["OFF_T_G17", "p_act"]))
    P = np.array(P)
    r_sep = M.spear(P[:, 1], P[:, 0])
    sign = float((np.sign(P[:, 1]) == np.sign(P[:, 0])).mean())
    big = np.argsort(-np.abs(P[:, 0]))[:10]
    r_big = M.spear(P[big, 1], P[big, 0])
    report += ["", f"## V1b sep（v5: −0.232/0.76/−0.151; v4: +0.150/0.83/+0.652）", "",
               f"- 全量 **{r_sep:+.3f}**, 符号 **{sign:.2f}**, top10 **{r_big:+.3f}**"]

    # ---- V3 V2-A 留出（8 seeds） ----
    hold = ["TrackF_r1_M255I", "HQL2_M255I"]
    tr = rows[~rows.system.isin(hold)].reset_index(drop=True)
    tr_g = [globs[r.system][r.condition] for _, r in tr.iterrows()]
    preds = []
    for seed in range(8):
        net, _ = M.train_model(tr, toks, tr_g, seed)
        hg = [globs[s][c] for s in hold for c in ("S1_G17", "OFF_T_G17")]
        preds.append(M.predict(net, toks, [s for s in hold for _ in (0, 1)],
                               ["S1_G17", "OFF_T_G17"] * 2, hg))
    pm8 = np.mean(preds, 0)
    report += ["", "## V3 V2-A 留出（真实 TrackF +0.25 / HQL2 −0.05）", ""]
    for i, s in enumerate(hold):
        sep = pm8[2 * i, 0] - pm8[2 * i + 1, 0]
        report.append(f"- {s}: 预测 sep **{sep:+.3f}**")
    ok = (pm8[0, 0] - pm8[1, 0] > 0) and (pm8[0, 0] - pm8[1, 0]) > (pm8[2, 0] - pm8[3, 0])
    report.append(f"- 排序判决: {'✓' if ok else '✗'}")
    txt = "\n".join(report)
    open(f"{OUT}/report.md", "w", encoding="utf-8").write(txt)
    print("\n" + txt)
    np.save(f"{OUT}/loo_pred.npy", pred)


if __name__ == "__main__":
    main()
