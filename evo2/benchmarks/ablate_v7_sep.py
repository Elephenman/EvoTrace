# -*- coding: utf-8 -*-
"""DL v7 sep 失败诊断 —— A1 vs A2 快速 A/B（不破坏 v5/v6 主线）。

根因（已核实）: v7 的 sep = pred_act(S1_G17) − pred_act(OFF_T_G17)，但模型只被要求
学"基因型总体 act"，DNA 条件仅经 global_vec 的 3 碱基 one-hot(nt10/17/23) + cond_vec(24bp)
做 query cross-attn 进入 —— 网络从不被显式要求拉齐两条件之差 → sep 失效 (rho −0.124 vs v4 +0.330)。

两版改造（用户选 A1+A2 都做对比）:
  base = 原 v7 (M.train_model + M.CondNet)
  A1   = base + 显式 sep 对比目标: 每 epoch 末尾对训练系统做系统级前向,
         L_sep = MSE( pred_act(S1)_sys − pred_act(OFF)_sys , sep_true_sys )，直接给"学条件差"的梯度。
  A2   = 条件×基因型前置交叉: CondNet 变体把 DNA 24bp 编码广播拼到每个位点 token 的输入，
         transformer 能在每层看到"该位点 vs 整条 DNA"，而非仅 query 旁路。loss 同 base。

快速验证口径（方向判断，非最终判决）: 全部 37 系统 LOO 但 epochs 减半(12) + 单 seed，
只比 sep rho/符号（sep 是本次目标）。最终版若方向对再上全 epochs。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import torch.nn as nn
import train_dl_v5 as M
from train_dl_v6 import build_ctx, build_tokens_v6, load_pca
from train_dl_v7_par import load_rows_n100

RES = "A:/claudework/evo2/results"
OUT = os.path.join(RES, "dl_v7sep")
os.makedirs(OUT, exist_ok=True)
DEV = "cpu"
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
EPOCHS = int(os.environ.get("SEP_EPOCHS", "12"))     # 快速验证减半
SEED = 0


# ============================================================== A1: sep 辅助损失训练
def train_model_A1(rows, toks, globs, seed, epochs=36, bs=512, lr=1.5e-3,
                   lam_sep=1.0, verbose=False):
    """同 M.train_model，但加显式 sep 对比目标（同系统 S1 vs OFF 的 act 差逼近真实 sep）。"""
    # ---- 预聚合系统级 sep 真值（用训练集 cells 均值，天然与 y_act 同口径） ----
    sep_true = {}
    for s, g in rows.groupby("system"):
        s1 = g[g.condition == "S1_G17"]["act"].mean()
        off = g[g.condition == "OFF_T_G17"]["act"].mean()
        if np.isfinite(s1) and np.isfinite(off):
            sep_true[s] = float(s1 - off)
    syslist = sorted(sep_true)
    if not syslist:
        print("[A1] 无成对系统, 退回 base", flush=True)
        return M.train_model(rows, toks, globs, seed, epochs=epochs)

    torch.manual_seed(seed); np.random.seed(seed)
    F = toks[rows.system.iloc[0]].shape[-1]
    G = globs[0].shape[-1]
    net = M.CondNet(F, G).to(DEV)
    base = float(np.nanmean(rows["act"].to_numpy(np.float32)))
    with torch.no_grad():
        net.head[-1].bias[0] = np.log(max(base, 1e-3) / max(1 - base, 1e-3))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scalers = {}
    for c in ("dcat", "d17", "d23"):
        v = rows[c].to_numpy(np.float32); mu = float(np.nanmean(v)); sd = float(np.nanstd(v)) or 1.0
        scalers[c] = (mu, sd)
    y = {"act": rows["act"].to_numpy(np.float32), "dual": rows["dual"].to_numpy(np.float32)}
    for c in ("dcat", "d17", "d23"):
        mu, sd = scalers[c]; y[c] = (rows[c].to_numpy(np.float32) - mu) / sd
    idx = np.arange(len(rows))
    tok_np = np.stack([toks[s] for s in rows.system])
    glob_np = np.concatenate(globs, 0)
    dna_np = np.stack([M.cond_vec(c) for c in rows.condition])
    # 每个成对系统的 S1 代表行索引 & OFF 代表行索引（用于系统级 sep 前向）
    s1_idx, off_idx = {}, {}
    for si_, s in enumerate(syslist):
        s1_idx[s] = int(np.flatnonzero((rows.system == s) & (rows.condition == "S1_G17"))[0])
        off_idx[s] = int(np.flatnonzero((rows.system == s) & (rows.condition == "OFF_T_G17"))[0])
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx); tot = 0.0
        for b in range(0, len(idx), bs):
            bidx = idx[b:b + bs]
            tb = torch.tensor(tok_np[bidx], device=DEV); gb = torch.tensor(glob_np[bidx], device=DEV)
            db = torch.tensor(dna_np[bidx], device=DEV)
            pad = torch.zeros(len(bidx), 53, dtype=torch.bool, device=DEV)
            out = net(tb, db, gb, pad); loss = 0.0
            for k, c in enumerate(M.LABELS):
                v = torch.tensor(y[c][bidx], device=DEV); m = torch.isfinite(v)
                if not m.any(): continue
                if M.LW[c][0] == "bce":
                    loss = loss + M.LW[c][1] * nn.functional.binary_cross_entropy_with_logits(out[m, k], v[m])
                else:
                    loss = loss + M.LW[c][1] * nn.functional.mse_loss(out[m, k], v[m])
            if torch.is_tensor(loss):
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                tot += loss.item() * len(bidx)
        # ---- A1: 系统级 sep 对比 pass（需梯度, 不用 no_grad） ----
        if lam_sep > 0:
            net.eval()
            si = [s1_idx[s] for s in syslist]; oi = [off_idx[s] for s in syslist]
            tt = torch.tensor(tok_np[si], device=DEV, requires_grad=True)
            tg = torch.tensor(glob_np[si], device=DEV)
            td = torch.tensor(dna_np[si], device=DEV)
            to = torch.tensor(tok_np[oi], device=DEV, requires_grad=True)
            tg2 = torch.tensor(glob_np[oi], device=DEV)
            td2 = torch.tensor(dna_np[oi], device=DEV)
            pp = torch.zeros(len(si), 53, dtype=torch.bool, device=DEV)
            pred_s1 = net(tt, td, tg, pp)[:, 0]
            pred_off = net(to, td2, tg2, pp)[:, 0]
            sep_pred = pred_s1 - pred_off
            sep_t = torch.tensor([sep_true[s] for s in syslist], device=DEV)
            loss_sep = nn.functional.mse_loss(sep_pred, sep_t)
            net.train()
            opt.zero_grad(); loss_sep.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        else:
            loss_sep = torch.tensor(0.0)
        sched.step()
        if verbose and (ep % 4 == 0 or ep == epochs - 1):
            print(f"    ep{ep:02d} loss {tot/len(idx):.4f} sepL {loss_sep.item():.4f}", flush=True)
    return net, scalers


# ============================================================== A2: 条件前置交叉网络
class CondNet_A2(nn.Module):
    """DNA 24bp 编码广播拼到每个位点 token 输入 → transformer 每层可见条件。
    差异(对 CondNet): 去掉 query cross-attn 旁路, 改为 dna_emb 拼进每 token。
    """
    def __init__(self, f_dim, g_dim, d=32, nl=2, heads=4, dna_d=16, hid=96):
        super().__init__()
        self.dna = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, dna_d))
        self.inp = nn.Sequential(nn.LayerNorm(f_dim + dna_d), nn.Linear(f_dim + dna_d, d), nn.GELU())
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.05, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nl)
        self.gproj = nn.Sequential(nn.LayerNorm(g_dim), nn.Linear(g_dim, hid), nn.GELU())
        self.head = nn.Sequential(nn.Linear(d + hid, 64), nn.GELU(), nn.Linear(64, 5))

    def forward(self, tok, dna, glob, pad):
        dvec = self.dna(dna).mean(1, keepdim=True)          # [B,24,4]->[B,24,dna_d]->[B,1,dna_d]
        x = torch.cat([tok, dvec.expand(-1, tok.shape[1], -1)], dim=-1)
        x = self.enc(self.inp(x), src_key_padding_mask=pad)
        pooled = x.mean(1)
        return self.head(torch.cat([pooled, self.gproj(glob)], dim=1))


def train_model_A2(rows, toks, globs, seed, epochs=36, bs=512, lr=1.5e-3, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    F = toks[rows.system.iloc[0]].shape[-1]; G = globs[0].shape[-1]
    net = CondNet_A2(F, G).to(DEV)
    base = float(np.nanmean(rows["act"].to_numpy(np.float32)))
    with torch.no_grad():
        net.head[-1].bias[0] = np.log(max(base, 1e-3) / max(1 - base, 1e-3))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scalers = {}
    for c in ("dcat", "d17", "d23"):
        v = rows[c].to_numpy(np.float32); mu = float(np.nanmean(v)); sd = float(np.nanstd(v)) or 1.0
        scalers[c] = (mu, sd)
    y = {"act": rows["act"].to_numpy(np.float32), "dual": rows["dual"].to_numpy(np.float32)}
    for c in ("dcat", "d17", "d23"):
        mu, sd = scalers[c]; y[c] = (rows[c].to_numpy(np.float32) - mu) / sd
    idx = np.arange(len(rows))
    tok_np = np.stack([toks[s] for s in rows.system]); glob_np = np.concatenate(globs, 0)
    dna_np = np.stack([M.cond_vec(c) for c in rows.condition])
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx); tot = 0.0
        for b in range(0, len(idx), bs):
            bidx = idx[b:b + bs]
            tb = torch.tensor(tok_np[bidx], device=DEV); gb = torch.tensor(glob_np[bidx], device=DEV)
            db = torch.tensor(dna_np[bidx], device=DEV)
            pad = torch.zeros(len(bidx), 53, dtype=torch.bool, device=DEV)
            out = net(tb, db, gb, pad); loss = 0.0
            for k, c in enumerate(M.LABELS):
                v = torch.tensor(y[c][bidx], device=DEV); m = torch.isfinite(v)
                if not m.any(): continue
                if M.LW[c][0] == "bce":
                    loss = loss + M.LW[c][1] * nn.functional.binary_cross_entropy_with_logits(out[m, k], v[m])
                else:
                    loss = loss + M.LW[c][1] * nn.functional.mse_loss(out[m, k], v[m])
            if torch.is_tensor(loss):
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                tot += loss.item() * len(bidx)
        sched.step()
        if verbose and (ep % 4 == 0 or ep == epochs - 1):
            print(f"    ep{ep:02d} loss {tot/len(idx):.4f}", flush=True)
    return net, scalers


# ============================================================== 数据装载（v7 同源）
def load_all():
    from train_sep_model_v4 import load_sequences, AA
    orc = M.load_ctx()
    M.zc_global = {}
    def zcache(name, seq):
        if name not in M.zc_global:
            g = np.array([AA.index(seq[s]) for s in M.SITES], dtype=np.int64)
            M.zc_global[name] = float(orc.evaluate_multi(g[None, :])[0])
        return M.zc_global[name]
    seqs = load_sequences()
    import re as _re
    for raw in pd.read_csv(f"{RES}/boltz_n100/contact_fingerprint_n100.csv").pred.unique():
        for cond in ("S1_G17", "OFF_T_G17"):
            if raw.endswith("_" + cond): raw = raw[: -len(cond) - 1]
        if raw.startswith("sm_"):
            core = raw[3:]; m = _re.match(r"(WT|TrackF)_F88([A-Z])$", core)
            if m and core not in seqs:
                base = seqs["WT" if m.group(1) == "WT" else "TrackF_r1"]; i = 88 - 22
                seqs[core] = base[:i] + m.group(2) + base[i + 1:]
    for n, s in seqs.items(): zcache(n, s)
    comps, mean = load_pca(); ctxs = build_ctx(seqs, comps, mean)
    rows = load_rows_n100(seqs, zcache)
    toks = build_tokens_v6(seqs, ctxs, zcache)
    globs = {}
    for _, r in rows.iterrows():
        globs.setdefault(r.system, {})
        if r.condition not in globs[r.system]:
            globs[r.system][r.condition] = M.global_vec(r.system, seqs[r.system], r.condition, M.zc_global[r.system])[None]
    cells = rows.groupby(["system", "condition"]).agg(
        n=("act", "size"), y_act=("act", "mean"), y_dual=("dual", "mean"), y_dcat=("dcat", "mean")).reset_index()
    return rows, cells, toks, globs


def run_ablation(mode, rows, cells, toks, globs):
    """全系统 LOO，只评估 sep（本次目标）。折并行 2 worker（内存受限, 线程共享一份 torch）。"""
    import concurrent.futures as cf
    systems = sorted(cells.system.unique())
    n = len(systems)
    te_masks = [(cells.system == s).to_numpy() for s in systems]
    pred = np.full((len(cells), 3), np.nan)

    def _fold(fi_s):
        fi, s = fi_s
        te = te_masks[fi]
        tr = rows[~rows.system.isin([s])].reset_index(drop=True)
        tr_g = [globs[r.system][r.condition] for _, r in tr.iterrows()]
        if mode == "A1":
            net, scalers = train_model_A1(tr, toks, tr_g, SEED, epochs=EPOCHS)
        elif mode == "A2":
            net, scalers = train_model_A2(tr, toks, tr_g, SEED, epochs=EPOCHS)
        else:  # base
            net, scalers = M.train_model(tr, toks, tr_g, SEED, epochs=EPOCHS)
        te_g = [globs[r.system][r.condition] for _, r in cells[te].iterrows()]
        pr = M.predict(net, toks, cells.system[te].tolist(), cells.condition[te].tolist(), te_g)
        return fi, M.unnorm(pr, scalers)[:, :3]

    t0 = time.time()
    NT = int(os.environ.get("SEP_NTHREAD", "2"))
    if NT > 1:
        import threading
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // NT - 1))
        done = 0
        with cf.ThreadPoolExecutor(max_workers=NT) as ex:
            for fi, pte in ex.map(_fold, [(fi, s) for fi, s in enumerate(systems)]):
                pred[te_masks[fi]] = pte
                done += 1
                if done % 5 == 0 or done == n:
                    print(f"  [{mode}] LOO {done}/{n} ({time.time()-t0:.0f}s)", flush=True)
    else:
        for fi, s in enumerate(systems):
            _, pte = _fold((fi, s))
            pred[te_masks[fi]] = pte
            if (fi + 1) % 5 == 0 or fi == n - 1:
                print(f"  [{mode}] LOO {fi+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    cd = cells.assign(p_act=pred[:, 0])
    P = []
    for s, g in cd.groupby("system"):
        if {"S1_G17", "OFF_T_G17"} <= set(g.condition):
            t = g.set_index("condition")
            P.append((t.loc["S1_G17", "y_act"] - t.loc["OFF_T_G17", "y_act"],
                      t.loc["S1_G17", "p_act"] - t.loc["OFF_T_G17", "p_act"]))
    P = np.array(P)
    r_sep = M.spear(P[:, 1], P[:, 0]); sign = float((np.sign(P[:, 1]) == np.sign(P[:, 0])).mean())
    big = np.argsort(-np.abs(P[:, 0]))[:10]
    r_big = M.spear(P[big, 1], P[big, 0])
    print(f"  [{mode}] sep rho={r_sep:+.3f} sign={sign:.2f} top10={r_big:+.3f}  (v4@n100 基准 +0.330/0.92)", flush=True)
    np.savez(f"{OUT}/loo_{mode}_ep{EPOCHS}.npz", pred=pred, system=cells.system.to_numpy(), condition=cells.condition.to_numpy())
    return r_sep, sign, r_big


def main():
    modes = [m.strip() for m in os.environ.get("SEP_MODES", "base,A1,A2").split(",")]
    rows, cells, toks, globs = load_all()
    print(f"[data] rows={len(rows)} cells={len(cells)} systems={cells.system.nunique()} epochs={EPOCHS}", flush=True)
    lines = []
    for mode in modes:
        r_sep, sign, r_big = run_ablation(mode, rows, cells, toks, globs)
        lines.append(f"- {mode}: sep rho **{r_sep:+.3f}** sign **{sign:.2f}** top10 **{r_big:+.3f}**")
    txt = "\n".join(["# v7 sep A/B 快速诊断 (epochs=%d, seed=%d)" % (EPOCHS, SEED), "",
                     "- 基准: v7 原版 sep −0.124 / v4@n100 sep +0.330", ""] + lines)
    open(f"{OUT}/ablation_report_ep{EPOCHS}.md", "w", encoding="utf-8").write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()
