# -*- coding: utf-8 -*-
"""DL v7 —— n=100 干净标签 + 变体专属表征（v6 架构, 数据源换成重扫指纹）。

数据: contact_fingerprint_n100.csv 逐模型行（7400 行, 每 cell 截 40 → ~2900 行, 37 系统）
验证: 与 v4/v5/v6 同口径 —— 按系统 LOO + V1b sep。（V2-A 已由 n=100 数据直接裁决, 不再作 V3）
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

OUT = "A:/claudework/evo2/results/dl_v7"
os.makedirs(OUT, exist_ok=True)
FP = "A:/claudework/evo2/results/boltz_n100/contact_fingerprint_n100.csv"


def load_rows_n100(seqs, zcache, cap=40):
    df = pd.read_csv(FP).drop_duplicates(subset=["pred", "model"])
    def sv(p):
        for cond in ("S1_G17", "OFF_T_G17"):
            if p.endswith("_" + cond):
                return p[:-len(cond) - 1], cond
        return p, "?"
    vc = [sv(p) for p in df.pred]
    df["system"] = [v for v, c in vc]
    df["condition"] = [c for v, c in vc]
    # 剥 d_/w_/sm_ 前缀映射到序列表; sm_* 单点在此重构
    import re
    def sys_seq(v):
        for p in ("d_", "w_", "sm_"):
            if v.startswith(p):
                core = v[len(p):]
                break
        else:
            core = v
        if core in seqs:
            return core, seqs[core]
        m = re.match(r"(WT|TrackF)_F88([A-Z])$", core)
        if m:
            bg = "WT" if m.group(1) == "WT" else "TrackF_r1"
            base = seqs[bg]
            return core, base[:88 - 22] + m.group(2) + base[88 - 22 + 1:]
        return None, None
    mapped = df.system.map(lambda v: sys_seq(v)[0])
    df = df[mapped.notna()]
    df["system"] = mapped[mapped.notna()]
    df = df.groupby(["system", "condition"], group_keys=False).apply(
        lambda g: g.head(cap)).reset_index(drop=True)
    df = df.rename(columns={"d_act": "dcat", "d_read17": "d17", "d_253_23": "d23"})
    df["src"] = "n100"
    return df[["system", "condition", "model", "act", "dual", "dcat", "d17", "d23",
               "iface", "src"]]


def main():
    orc = M.load_ctx()
    M.zc_global = {}

    def zcache(name, seq):
        if name not in M.zc_global:
            g = np.array([M.AA.index(seq[s]) for s in M.SITES], dtype=np.int64)
            M.zc_global[name] = float(orc.evaluate_multi(g[None, :])[0])
        return M.zc_global[name]

    seqs = load_sequences()
    # 并入 n100 中的 7 个 sm 单点（嵌入已在 EMB_DIR: sm_WT_F88K 等）
    import re as _re
    for raw in pd.read_csv(FP).pred.unique():
        for cond in ("S1_G17", "OFF_T_G17"):
            if raw.endswith("_" + cond):
                raw = raw[: -len(cond) - 1]
        if raw.startswith("sm_"):
            core = raw[3:]
            m = _re.match(r"(WT|TrackF)_F88([A-Z])$", core)
            if m and core not in seqs:
                base = seqs["WT" if m.group(1) == "WT" else "TrackF_r1"]
                i = 88 - 22
                seqs[core] = base[:i] + m.group(2) + base[i + 1:]
    for n, s in seqs.items():
        zcache(n, s)
    comps, mean = load_pca()
    ctxs = build_ctx(seqs, comps, mean)
    rows = load_rows_n100(seqs, zcache)
    print(f"[rows] {len(rows)} 行, {rows.system.nunique()} 系统", flush=True)
    toks = build_tokens_v6(seqs, ctxs, zcache)
    globs = {}
    for _, r in rows.iterrows():
        globs.setdefault(r.system, {})
        if r.condition not in globs[r.system]:
            globs[r.system][r.condition] = M.global_vec(
                r.system, seqs[r.system], r.condition, M.zc_global[r.system])[None]
    row_globs = [globs[r.system][r.condition] for _, r in rows.iterrows()]

    cells = rows.groupby(["system", "condition"]).agg(
        n=("act", "size"), y_act=("act", "mean"), y_dual=("dual", "mean"),
        y_dcat=("dcat", "mean")).reset_index()
    print(f"[cells] {len(cells)}", flush=True)

    def loo(epochs=24):
        pred = np.full((len(cells), 3), np.nan)
        ckpt = os.path.join(OUT, "loo_ckpt.npz")
        done = {}
        if os.path.exists(ckpt):
            done = dict(np.load(ckpt, allow_pickle=True))
            print(f"[ckpt] 恢复 {len(done)} 折", flush=True)
        import time
        t0 = time.time()
        systems = sorted(cells.system.unique())
        for fi, s in enumerate(systems):
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
            if fi % 5 == 0 or fi == len(systems) - 1:
                print(f"[LOO] {fi+1}/{len(systems)} ({time.time()-t0:.0f}s)", flush=True)
        return pred

    pred = loo()
    report = ["# DL v7 —— n=100 干净标签 + 变体专属表征", "",
              f"rows={len(rows)}, cells={len(cells)}, systems={cells.system.nunique()}", "",
              "## V1 按系统 LOO（v6@旧标签: +0.576/+0.680/+0.650; v4@n100: +0.739/+0.747/—）", ""]
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
    sg = float((np.sign(P[:, 1]) == np.sign(P[:, 0])).mean())
    big = np.argsort(-np.abs(P[:, 0]))[:10]
    r_big = M.spear(P[big, 1], P[big, 0])
    report += ["", f"## V1b sep（v4@n100 直接sep: +0.549/0.97/+0.341）", "",
               f"- 全量 **{r_sep:+.3f}**, 符号 **{sg:.2f}**, top10 **{r_big:+.3f}**"]
    txt = "\n".join(report)
    open(f"{OUT}/report.md", "w", encoding="utf-8").write(txt)
    np.save(f"{OUT}/loo_pred.npy", pred)
    print("\n" + txt)
    print(f"[out] {OUT}/report.md", flush=True)


if __name__ == "__main__":
    main()
