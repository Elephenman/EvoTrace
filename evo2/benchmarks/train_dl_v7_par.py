# -*- coding: utf-8 -*-
"""DL v7 并行加速版 —— n=100 干净标签 + 变体专属表征（v6 架构, 数据源换成重扫指纹）。

与 train_dl_v7.py 同口径（数据/特征/LOO/指标完全一致），但把 37 折 LOO 并行跑以加速。

并行实现注意:
  本机页面文件极小（空闲 ~5.8GB），多进程 torch 会因每个子进程独立提交 torch 大块
  内存而撑爆页面文件（WinError 1455 / alloc 失败）。故改用**线程并行**: 同一进程内
  多线程，torch 算子释放 GIL 可真正并行计算，而只有一份 torch，内存开销与顺序版一致
  （已验证单进程 torch 训练可行）。

断点续跑:
  - 主 ckpt: results/dl_v7/loo_ckpt.npz（key=f0..f36，每折存该折真实 te 预测的 (n_te,3) 切片）
  - worker ckpt: results/dl_v7/loo_ckpt_w{i}.npz（每折独立落盘，崩溃也不丢）
  任意时刻重跑都会跳过已完成折。

指标（与 v4/v5/v6 同口径）:
  V1  按系统 LOO 的 act/dual/dcat Spearman
  V1b 同系统 S1_G17 vs OFF_T_G17 的 sep（真实 sep vs 预测 sep 的 Spearman / 符号一致率 / top10）
判决基线: v4@n100 基准 act/dual = +0.739/+0.747，sep = +0.330 / 符号 0.92
"""
import os
import sys
import time
import concurrent.futures as cf

# 限制每个训练线程的 BLAS/OpenMP 线程池，避免多线程并行时过度订阅 CPU。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# 仅主进程导入 M（torch 仅需一份）；worker 逻辑在 _wk 中直接用本进程全局量。
import train_dl_v5 as M

OUT = "A:/claudework/evo2/results/dl_v7"
os.makedirs(OUT, exist_ok=True)
FP = "A:/claudework/evo2/results/boltz_n100/contact_fingerprint_n100.csv"
EPOCHS = 24          # 与已完成的 f0-f6 保持一致
N_THREADS = 2        # 同进程内并行折数（受页面文件限制的稳妥取值）

# ---- 主进程全局（线程共享，免序列化）----
_ROWS = _CELLS = _TOKS = _GLOBS = None


def _wk(fi, s):
    """训练单个 LOO 折并返回 (fi, pred_te[n_te,3])。"""
    tr = _ROWS[~_ROWS.system.isin([s])].reset_index(drop=True)
    tr_g = [_GLOBS[r.system][r.condition] for _, r in tr.iterrows()]
    net, scalers = M.train_model(tr, _TOKS, tr_g, 0, epochs=EPOCHS)
    te = (_CELLS.system == s).to_numpy()
    te_sys = _CELLS.system[te].tolist()
    te_cond = _CELLS.condition[te].tolist()
    te_g = [_GLOBS[r.system][r.condition] for _, r in _CELLS[te].iterrows()]
    pr = M.predict(net, _TOKS, te_sys, te_cond, te_g)
    return fi, M.unnorm(pr, scalers)[:, :3]


# ------------------------------------------------------ 数据装载（与 v7 完全一致）
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
    global _ROWS, _CELLS, _TOKS, _GLOBS
    t0 = time.time()
    from train_sep_model_v4 import load_sequences, AA
    from train_dl_v6 import build_ctx, build_tokens_v6, load_pca, EMB_DIR
    orc = M.load_ctx()
    M.zc_global = {}

    def zcache(name, seq):
        if name not in M.zc_global:
            g = np.array([AA.index(seq[s]) for s in M.SITES], dtype=np.int64)
            M.zc_global[name] = float(orc.evaluate_multi(g[None, :])[0])
        return M.zc_global[name]

    seqs = load_sequences()
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
    cells = rows.groupby(["system", "condition"]).agg(
        n=("act", "size"), y_act=("act", "mean"), y_dual=("dual", "mean"),
        y_dcat=("dcat", "mean")).reset_index()
    print(f"[cells] {len(cells)}  系统数={cells.system.nunique()}", flush=True)

    _ROWS, _CELLS, _TOKS, _GLOBS = rows, cells, toks, globs

    systems = sorted(cells.system.unique())
    n_folds = len(systems)
    te_masks = [(cells.system == s).to_numpy() for s in systems]
    pred = np.full((len(cells), 3), np.nan)

    # ---- 恢复已完成折 ----
    done = {}
    main_ckpt = os.path.join(OUT, "loo_ckpt.npz")
    if os.path.exists(main_ckpt):
        d = np.load(main_ckpt, allow_pickle=True)
        for k in d.files:
            done[k] = d[k]
        print(f"[ckpt] 主 ckpt 恢复 {len(done)} 折", flush=True)

    recovered = 0
    for fi in range(n_folds):
        key = f"f{fi}"
        if key in done:
            p = np.asarray(done[key])
            # 兼容两种存储: 完整 (74,3) 矩阵 或 紧凑 te 切片
            if p.shape[0] == len(cells):
                pred[te_masks[fi]] = p[te_masks[fi]]
            else:
                pred[te_masks[fi]] = p
            recovered += 1
    print(f"[ckpt] 已恢复 {recovered} 折预测", flush=True)

    pending = [(fi, systems[fi]) for fi in range(n_folds)
               if not np.isfinite(pred[te_masks[fi]].sum())]
    still_pending = []
    for fi, s in pending:
        wk = os.path.join(OUT, f"loo_ckpt_w{fi}.npz")
        if os.path.exists(wk):
            p = np.load(wk)["p"]
            pred[te_masks[fi]] = p
            done[f"f{fi}"] = p
            np.savez(main_ckpt, **done)
            print(f"[wk] 复用 worker ckpt f{fi}", flush=True)
        else:
            still_pending.append((fi, s))
    print(f"[plan] 待训练折: {[f for f, _ in still_pending]}", flush=True)

    if still_pending:
        # 线程并行: 同进程一份 torch，GIL 在 torch 算子处释放可真正并行。
        wthreads = max(1, (os.cpu_count() or 4) // N_THREADS)
        import torch
        torch.set_num_threads(wthreads)
        left = len(still_pending)
        with cf.ThreadPoolExecutor(max_workers=N_THREADS) as ex:
            futs = [ex.submit(_wk, fi, s) for fi, s in still_pending]
            for fut in cf.as_completed(futs):
                fi, pte = fut.result()
                pred[te_masks[fi]] = pte
                done[f"f{fi}"] = pte
                np.savez(os.path.join(OUT, f"loo_ckpt_w{fi}.npz"), p=pte)
                np.savez(main_ckpt, **done)
                left -= 1
                print(f"[LOO] f{fi} 完成 ({time.time()-t0:.0f}s, 剩 {left})", flush=True)
    print(f"[LOO] 全部完成 {time.time()-t0:.0f}s", flush=True)
    np.save(f"{OUT}/loo_pred.npy", pred)

    # ---- V1 按系统 LOO ----
    report = ["# DL v7 并行版 —— n=100 干净标签 + 变体专属表征", "",
              f"rows={len(rows)}, cells={len(cells)}, systems={cells.system.nunique()}, "
              f"dev={M.DEV}, epochs={EPOCHS}, threads={N_THREADS}", "",
              "## V1 按系统 LOO（v4@n100 基准: act +0.739 / dual +0.747 / dcat —）", ""]
    v1 = {}
    for k, c in enumerate(["y_act", "y_dual", "y_dcat"]):
        r = M.spear(pred[:, k], cells[c].to_numpy())
        v1[c] = r
        print(f"[V1] {c}: {r:+.3f}", flush=True)
        report.append(f"- {c}: **{r:+.3f}**")

    # ---- V1b sep（修复空 P 的 IndexError）----
    cd = cells.assign(p_act=pred[:, 0])
    P = []
    for s, g in cd.groupby("system"):
        if {"S1_G17", "OFF_T_G17"} <= set(g.condition):
            t = g.set_index("condition")
            P.append((t.loc["S1_G17", "y_act"] - t.loc["OFF_T_G17", "y_act"],
                      t.loc["S1_G17", "p_act"] - t.loc["OFF_T_G17", "p_act"]))
    if P:
        P = np.array(P)
        r_sep = M.spear(P[:, 1], P[:, 0])
        sg = float((np.sign(P[:, 1]) == np.sign(P[:, 0])).mean())
        big = np.argsort(-np.abs(P[:, 0]))[:10]
        r_big = M.spear(P[big, 1], P[big, 0])
        nsys = len(P)
    else:
        r_sep = r_big = np.nan
        sg = 0.0
        nsys = 0
    report += ["", f"## V1b sep（v4@n100 基准: rho +0.330 / 符号 0.92）", "",
               f"- 系统数={nsys}, 全量 **{r_sep:+.3f}**, 符号 **{sg:.2f}**, top10 **{r_big:+.3f}**"]
    print(f"[V1b] sep rho={r_sep:+.3f} sign={sg:.2f} top10={r_big:+.3f} nsys={nsys}", flush=True)

    # ---- 判决 ----
    base_act, base_dual, base_sep, base_sign = 0.739, 0.747, 0.330, 0.92
    ok_act = v1["y_act"] >= base_act
    ok_dual = v1["y_dual"] >= base_dual
    ok_sep = (not np.isnan(r_sep)) and (r_sep >= base_sep) and (sg >= base_sign)
    verdict = "PASS" if (ok_act and ok_dual and ok_sep) else "REVIEW"
    report += ["", "## 对 v4@n100 基准的判决", "",
               f"- V1 act : v7 {v1['y_act']:+.3f}  vs 基准 {base_act:+.3f}  → {'✓' if ok_act else '✗'}",
               f"- V1 dual: v7 {v1['y_dual']:+.3f}  vs 基准 {base_dual:+.3f}  → {'✓' if ok_dual else '✗'}",
               f"- V1 sep : v7 rho {r_sep:+.3f}/符号 {sg:.2f}  vs 基准 {base_sep:+.3f}/{base_sign:.2f}  → {'✓' if ok_sep else '✗'}",
               f"- **判决: {verdict}**（V1 act/dual 达基准且 sep 不弱于基准则为 PASS）"]
    txt = "\n".join(report)
    open(f"{OUT}/report.md", "w", encoding="utf-8").write(txt)
    print("\n" + txt)
    print(f"[out] {OUT}/report.md", flush=True)


if __name__ == "__main__":
    main()
