# -*- coding: utf-8 -*-
"""DL v5 —— 条件感知 transformer 代理（参考 EvoMax/ProMEP/Liu-lab 三篇配方）。

设计映射（三篇 → PprI）:
  p1 EvoMax   : 预训练 PLM 表征 + 小模型在稀疏标签上训练 + 集成方差做获取
  p2 ProMEP   : 组合空间全打分提名 top-k; 警惕 PLM 残基偏置
  p3 Liu lab  : 稳定性余量决定进化可达性（起点鲁棒性优先）
实现要点（v1 教训: 突变 one-hot 被 320 维 ESM3 上下文稀释 → 网络走常数捷径）:
  - 双通路: (a) 位点 token transformer（容量项, 上下文降权 0.25, 显式 is_mut 标记）
            (b) 全局特征直连通路（v4 已验证的 GBM 特征 + DNA 条件, 绕过注意力稀释）
  - 多任务头: act / dual (BCE) + dcat / d17 / d23 (标准化后 MSE), 逐模型行 = 标签噪声增广
  - 验证: 与 v4 完全同口径 —— 按系统 LOO + V1b sep + V3 V2-A 留出上位性
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# NOTE: train_sep_model_v4 现已改为顶层轻量化（scipy/sklearn 延迟导入），
# 故此处可安全 import；v7 并行 worker 也不会因此加载重型库。
from train_sep_model_v4 import (AA, CLUSTER, COND_SEQ, P2S, WT_SEQ,
                                genotype_features, load_sequences)

RES = "A:/claudework/evo2/results"
OUT = os.path.join(RES, "dl_v5")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLUSTER_LIST = ["readhead", "lock", "anchor", "cat", "other"]
CIDX = {c: i for i, c in enumerate(CLUSTER_LIST)}
CTX_SCALE = 0.25
LABELS = ["act", "dual", "dcat", "d17", "d23"]
LW = {"act": ("bce", 1.0), "dual": ("bce", 1.0), "dcat": ("mse", 0.5),
      "d17": ("mse", 0.25), "d23": ("mse", 0.25)}

SITES = None
CTX = None


def load_ctx():
    global SITES, CTX
    sys.path.insert(0, "A:/claudework/evo2/esm3")
    from ppri_surrogate_v3 import PprISurrogateV3
    orc = PprISurrogateV3()
    SITES = np.asarray(orc.sites, dtype=np.int64)
    CTX = np.asarray(getattr(orc, "site_ctx"), dtype=np.float32) * CTX_SCALE
    return orc


# ------------------------------------------------------ 特征
def load_rows(seqs, zcache, old_max_models=8):
    frames = []
    pm = pd.read_csv("A:/claudework/ppri_evo/results/per_model_metrics.csv")
    pm = pm.rename(columns={"hexxh": "dcat"})
    pm = pm.groupby(["system", "condition"], group_keys=False).apply(
        lambda g: g.head(old_max_models)).reset_index(drop=True)
    frames.append(pm[["system", "condition", "model", "act", "dual",
                      "dcat", "d17", "d23"]].assign(src="old"))

    def fp(path, sys_col=None, name_map=None):
        df = pd.read_csv(path)
        df["cond"] = df["cond"].map(lambda c: {"S1": "S1_G17", "OFF": "OFF_T_G17"}.get(c, c))
        if sys_col is None:
            pat = df["pred"].str.replace(r"^(d_)?", "", regex=True)
            df["system"] = pat.str.replace(r"_(S1_G17|OFF_T_G17)$", "", regex=True)
        else:
            df["system"] = df[sys_col]
        df = df.rename(columns={"d_act": "dcat", "d_read17": "d17", "d_253_23": "d23"})
        return df[["system", "cond", "model", "act", "dual", "dcat", "d17", "d23",
                   "iface"]].rename(columns={"cond": "condition"}).assign(src=name_map)

    frames.append(fp(f"{RES}/boltz_six/contact_fingerprint.csv", "cand", "six"))
    frames.append(fp(f"{RES}/boltz_six/deconf_fingerprint.csv", None, "deconf"))
    frames.append(fp(f"{RES}/boltz_v2a/contact_fingerprint.csv", "cand", "v2a"))
    cal = pd.read_csv("A:/claudework/evo2/esm3/calib_proxy_dataset.csv")
    v3 = cal[cal.src == "v3cand"]
    frames.append(pd.DataFrame({
        "system": list(v3.name) * 2,
        "condition": ["S1_G17"] * len(v3) + ["OFF_T_G17"] * len(v3),
        "model": 0, "act": list(v3.act_s1) + list(v3.act_off),
        "dual": np.nan, "dcat": np.nan, "d17": np.nan, "d23": np.nan,
        "iface": np.nan, "src": "v3cand"}))
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows.system.isin(seqs)].reset_index(drop=True)
    return rows


def build_tokens(seqs):
    """每系统 → [53, F] token。"""
    from train_sep_model_v4 import AA, CLUSTER, WT_SEQ
    toks = {}
    F = 20 + 20 + 1 + 1 + len(CLUSTER_LIST) + CTX.shape[1] * CTX.shape[2] + 1
    for name, seq in seqs.items():
        muts = {i: a for i, a in enumerate(seq) if a != WT_SEQ[i]}
        T = np.zeros((len(SITES), F), dtype=np.float32)
        for j, s in enumerate(SITES):
            wt = WT_SEQ[s]
            mu = muts.get(int(s), wt)
            cl = CLUSTER.get(int(s) + 22, "other")
            T[j, AA.index(wt)] = 1.0
            T[j, 20 + AA.index(mu)] = 1.0
            T[j, 40] = j / max(len(SITES) - 1, 1)
            T[j, 41] = 1.0 if mu != wt else 0.0           # 显式突变标记
            T[j, 42 + CIDX[cl]] = 1.0
            b = 42 + len(CLUSTER_LIST)
            T[j, b:b + CTX.shape[1] * CTX.shape[2]] = CTX[j].ravel()
            T[j, -1] = zc_global.get(name, 0.0)
        toks[name] = T
    return toks


def cond_vec(condition):
    s = COND_SEQ[condition]  # 模块级内联常量，worker 无需 import 重型模块
    x = np.zeros((24, 4), dtype=np.float32)
    for i, b in enumerate(s):
        x[i, "ACGT".index(b)] = 1.0
    return x


_ANCHORS = None


def _anchors():
    global _ANCHORS
    if _ANCHORS is None:
        from train_sep_model_v4 import P2S
        _ANCHORS = [P2S(p) for p in (85, 207, 267)]
    return _ANCHORS


def global_vec(name, seq, condition, zval):
    """v4 同款全局特征（直连通路）: 热点 AA one-hot + 计数/电荷数值 + 条件。"""
    from train_sep_model_v4 import genotype_features, AA, WT_SEQ
    g = genotype_features(name, seq, lambda n, s: zval)
    muts = {i: a for i, a in enumerate(seq) if a != WT_SEQ[i]}
    v = []
    for key in ("rh66", "m255", "y217", "r253mut"):
        v.extend(np.eye(20, dtype=np.float32)[AA.index(g[key])])
    for i in _anchors():                                   # anchor_state 3×20
        v.extend(np.eye(20, dtype=np.float32)[AA.index(muts.get(i, WT_SEQ[i]))])
    for key in ("n_mut", "n_out53", "lock_mut", "anchor_mut", "cat_mut",
                "charge_anchor", "charge_total"):
        v.append(np.float32(g[key]))
    v.append(np.float32(zval))
    nt10, nt17, nt23 = (COND_SEQ[condition][9], COND_SEQ[condition][16], COND_SEQ[condition][22])
    v.extend(np.eye(4, dtype=np.float32)["ACGT".index(nt10)])
    v.extend(np.eye(4, dtype=np.float32)["ACGT".index(nt17)])
    v.extend(np.eye(4, dtype=np.float32)["ACGT".index(nt23)])
    return np.array(v, dtype=np.float32)


# ------------------------------------------------------ 模型
class CondNet(nn.Module):
    def __init__(self, f_dim, g_dim, d=32, nl=2, heads=4, dna_d=32, hid=96):
        super().__init__()
        self.dna = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, dna_d))
        self.inp = nn.Sequential(nn.LayerNorm(f_dim), nn.Linear(f_dim, d), nn.GELU())
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.05,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nl)
        self.qproj = nn.Linear(dna_d, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.gproj = nn.Sequential(nn.LayerNorm(g_dim), nn.Linear(g_dim, hid), nn.GELU())
        self.head = nn.Sequential(nn.Linear(d + hid, 64), nn.GELU(), nn.Linear(64, 5))

    def forward(self, tok, dna, glob, pad):
        x = self.inp(tok)
        x = self.enc(x, src_key_padding_mask=pad)
        q = self.qproj(self.dna(dna).mean(1)).unsqueeze(1)
        att, _ = self.attn(q, x, x, key_padding_mask=pad)
        pooled = att.squeeze(1) + x.mean(1)
        return self.head(torch.cat([pooled, self.gproj(glob)], dim=1))


def train_model(rows, toks, globs, seed, epochs=36, bs=512, lr=1.5e-3, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    F = toks[rows.system.iloc[0]].shape[-1]
    G = globs[0].shape[-1]
    net = CondNet(F, G).to(DEV)
    # act 输出偏置初始化为基率 logit, 避免常数捷径
    base = float(np.nanmean(rows["act"].to_numpy(np.float32)))
    with torch.no_grad():
        net.head[-1].bias[0] = np.log(max(base, 1e-3) / max(1 - base, 1e-3))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scalers = {}
    for c in ("dcat", "d17", "d23"):
        v = rows[c].to_numpy(np.float32)
        mu = float(np.nanmean(v))
        sd = float(np.nanstd(v)) or 1.0
        scalers[c] = (mu, sd)
    y = {"act": rows["act"].to_numpy(np.float32), "dual": rows["dual"].to_numpy(np.float32)}
    for c in ("dcat", "d17", "d23"):
        mu, sd = scalers[c]
        y[c] = (rows[c].to_numpy(np.float32) - mu) / sd
    idx = np.arange(len(rows))
    tok_np = np.stack([toks[s] for s in rows.system])
    glob_np = np.concatenate(globs, 0)
    dna_np = np.stack([cond_vec(c) for c in rows.condition])
    losses = []
    for ep in range(epochs):
        net.train()
        np.random.shuffle(idx)
        tot = 0.0
        for b in range(0, len(idx), bs):
            bidx = idx[b:b + bs]
            tb = torch.tensor(tok_np[bidx], device=DEV)
            gb = torch.tensor(glob_np[bidx], device=DEV)
            db = torch.tensor(dna_np[bidx], device=DEV)
            pad = torch.zeros(len(bidx), 53, dtype=torch.bool, device=DEV)
            out = net(tb, db, gb, pad)
            loss = 0.0
            for k, c in enumerate(LABELS):
                v = torch.tensor(y[c][bidx], device=DEV)
                m = torch.isfinite(v)
                if not m.any():
                    continue
                if LW[c][0] == "bce":
                    loss = loss + LW[c][1] * nn.functional.binary_cross_entropy_with_logits(
                        out[m, k], v[m])
                else:
                    loss = loss + LW[c][1] * nn.functional.mse_loss(out[m, k], v[m])
            if not torch.is_tensor(loss):
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(bidx)
        sched.step()
        losses.append(tot / len(idx))
        if verbose and (ep % 6 == 0 or ep == epochs - 1):
            print(f"    ep{ep:02d} loss {losses[-1]:.4f}", flush=True)
    return net, scalers


def predict(net, toks, systems, conditions, globs):
    tok_np = np.stack([toks[s] for s in systems])
    glob_np = np.concatenate(globs, 0)
    dna_np = np.stack([cond_vec(c) for c in conditions])
    net.eval()
    with torch.no_grad():
        tb = torch.tensor(tok_np, device=DEV)
        gb = torch.tensor(glob_np, device=DEV)
        db = torch.tensor(dna_np, device=DEV)
        pad = torch.zeros(len(systems), 53, dtype=torch.bool, device=DEV)
        return net(tb, db, gb, pad).cpu().numpy()


def unnorm(pred, scalers):
    p = pred.copy()
    for k, c in enumerate(LABELS):
        if c in scalers:
            mu, sd = scalers[c]
            p[:, k] = p[:, k] * sd + mu
    return p


def spear(a, b):
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    return spearmanr(a[ok], b[ok]).statistic if ok.sum() >= 4 else np.nan


# ------------------------------------------------------ 主流程
def main():
    global zc_global
    from train_sep_model_v4 import load_sequences, AA
    orc = load_ctx()
    zc_global = {}

    def zcache(name, seq):
        if name not in zc_global:
            g = np.array([AA.index(seq[s]) for s in SITES], dtype=np.int64)
            zc_global[name] = float(orc.evaluate_multi(g[None, :])[0])
        return zc_global[name]

    seqs = load_sequences()
    for n, s in seqs.items():
        zcache(n, s)
    rows = load_rows(seqs, zcache)
    print(f"[rows] {len(rows)} 行, {rows.system.nunique()} 系统, "
          f"{rows.condition.value_counts().to_dict()}", flush=True)
    toks = build_tokens(seqs)
    globs = {}
    for _, r in rows.iterrows():
        globs.setdefault(r.system, {})
        if r.condition not in globs[r.system]:
            globs[r.system][r.condition] = global_vec(
                r.system, seqs[r.system], r.condition, zc_global[r.system])[None]
    row_globs = [globs[r.system][r.condition] for _, r in rows.iterrows()]
    G = row_globs[0].shape[-1]
    print(f"[feat] token {toks['WT'].shape}, global {G} 维", flush=True)

    cells = rows.groupby(["system", "condition"]).agg(
        n=("act", "size"), y_act=("act", "mean"), y_dual=("dual", "mean"),
        y_dcat=("dcat", "mean")).reset_index()
    print(f"[cells] {len(cells)}", flush=True)

    # ---------- V1 按系统 LOO（带断点续跑） ----------
    def loo(seeds=(0,), epochs=36):
        pred = np.full((len(cells), 3, len(seeds)), np.nan)
        ckpt_path = os.path.join(OUT, f"loo_ckpt_seed{seeds[0]}.npz")
        done = {}
        if os.path.exists(ckpt_path):
            z = np.load(ckpt_path, allow_pickle=True)
            done = {k: v for k, v in z.items()}
            print(f"[ckpt] 恢复 {len(done)} 折", flush=True)
        for si, seed in enumerate(seeds):
            for fi, s in enumerate(sorted(cells.system.unique())):
                key = f"f{fi}"
                if key in done:
                    pred[:, :, si] = done[key]
                    continue
                te = (cells.system == s).to_numpy()
                tr = rows[~rows.system.isin([s])].reset_index(drop=True)
                tr_globs = [globs[r.system][r.condition] for _, r in tr.iterrows()]
                net, scalers = train_model(tr, toks, tr_globs, seed, epochs=epochs)
                te_g = [globs[r.system][r.condition] for _, r in cells[te].iterrows()]
                pr = predict(net, toks, cells.system[te].tolist(),
                             cells.condition[te].tolist(), te_g)
                pred[te, :, si] = unnorm(pr, scalers)[:, :3]
                done[key] = pred[:, :, si].copy()
                np.savez(ckpt_path, **done)
                if fi % 10 == 0 or fi == len(cells.system.unique()) - 1:
                    print(f"[LOO] seed{seed} fold {fi+1}/{len(cells.system.unique())} "
                          f"({__import__('time').time()-t0:.0f}s)", flush=True)
        return pred.mean(2), pred.std(2).mean(0)

    t0 = __import__("time").time()
    pmean, pstd = loo()
    print(f"[LOO] {__import__('time').time()-t0:.0f}s", flush=True)
    report = ["# DL v5 条件感知 transformer 代理（三篇配方落地 v2）", "",
              f"rows={len(rows)}, cells={len(cells)}, systems={cells.system.nunique()}, dev={DEV}",
              "", "## V1 按系统 LOO（v4 GBM 基线: act +0.638 / dual +0.718 / dcat +0.705）", ""]
    for k, c in enumerate(["y_act", "y_dual", "y_dcat"]):
        r = spear(pmean[:, k], cells[c].to_numpy())
        print(f"[V1] {c}: DL rho {r:+.3f}", flush=True)
        report.append(f"- {c}: **{r:+.3f}**  (集成 σ={pstd[k]:.3f})")

    # ---------- V1b sep ----------
    cd = cells.assign(p_act=pmean[:, 0])
    P = []
    for s, g in cd.groupby("system"):
        if {"S1_G17", "OFF_T_G17"} <= set(g.condition):
            t = g.set_index("condition")
            P.append((t.loc["S1_G17", "y_act"] - t.loc["OFF_T_G17", "y_act"],
                      t.loc["S1_G17", "p_act"] - t.loc["OFF_T_G17", "p_act"]))
    P = np.array(P)
    r_sep = spear(P[:, 1], P[:, 0])
    sign = float((np.sign(P[:, 1]) == np.sign(P[:, 0])).mean())
    big = np.argsort(-np.abs(P[:, 0]))[:10]
    r_big = spear(P[big, 1], P[big, 0])
    report += ["", "## V1b sep（v4 基线: +0.150 / 符号 0.83 / top10 +0.652）", "",
               f"- 全量 **{r_sep:+.3f}**, 符号 **{sign:.2f}**, top10 **{r_big:+.3f}**"]

    # ---------- V3 V2-A 留出 ----------
    hold = ["TrackF_r1_M255I", "HQL2_M255I"]
    tr = rows[~rows.system.isin(hold)].reset_index(drop=True)
    tr_globs = [globs[r.system][r.condition] for _, r in tr.iterrows()]
    preds = []
    for seed in range(8):
        net, _ = train_model(tr, toks, tr_globs, seed)
        hg = [globs[s][c] for s in hold for c in ("S1_G17", "OFF_T_G17")]
        preds.append(predict(net, toks, [s for s in hold for _ in (0, 1)],
                             ["S1_G17", "OFF_T_G17"] * 2, hg))
    pm8 = np.mean(preds, 0)
    lines = ["", "## V3 V2-A 留出上位性（8 seeds; 真实 TrackF +0.25 / HQL2 −0.05）", ""]
    for i, s in enumerate(hold):
        sep = pm8[2 * i, 0] - pm8[2 * i + 1, 0]
        sd = float(np.std([p[2 * i, 0] - p[2 * i + 1, 0] for p in preds]))
        lines.append(f"- {s}: 预测 sep **{sep:+.3f}** (σ={sd:.3f})")
    ok = (pm8[0, 0] - pm8[1, 0] > 0) and (pm8[2, 0] - pm8[3, 0] < pm8[0, 0] - pm8[1, 0])
    lines.append(f"- 排序判决 (TrackF+M255I > HQL2+M255I): {'✓' if ok else '✗'}")
    report += lines
    print("\n".join(lines), flush=True)

    txt = "\n".join(report)
    open(f"{OUT}/report.md", "w", encoding="utf-8").write(txt)
    np.savez(f"{OUT}/loo_pred.npz", pmean=pmean, pstd=pstd,
             system=cells.system.to_numpy(), condition=cells.condition.to_numpy())
    print("\n" + txt)
    print(f"\n[out] {OUT}/report.md, loo_pred.npz", flush=True)


if __name__ == "__main__":
    main()
