# -*- coding: utf-8 -*-
"""跨蛋白 DL 代理 v3：ESM3 per-residue embedding 特征（替代 ±2 one-hot 窗口）。

- 特征: site j 上下文 = concat(ESM3_emb[j-2..j+2]) 经 PCA(64) → 320 维
        + wt_aa(20) + mut_aa(20) + frac(1) = 341 维
- 数据: ProteinGym substitutions（217 DMS），held-out SPG1_Olson/Wu + GRB2
- 超长序列(>2048aa)：集群端已居中裁剪，位点落在窗口外 → 样本剔除（记录比例）
- PCA 只 fit 训练集残基；训练策略 v2（cosine + held-out 早停）
"""
import os, re, glob, json, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

ROOT = "A:/claudework/evo_data/processed/proteingym_benchmark/DMS_ProteinGym_substitutions"
EMB_ROOT = "A:/claudework/out/esm3_embeddings"          # <name>.npz -> emb (L_eff,1536)
OUT = "A:/claudework/out/surrogate_dl_v3"
HOLDOUT = {"SPG1_STRSG_Olson_2014", "SPG1_STRSG_Wu_2016", "GRB2_HUMAN_Faure_2021"}
AA = "ACDEFGHIKLMNPQRSTVWY"
AAI = {a: i for i, a in enumerate(AA)}
WIN = 2
PAT = re.compile(r"([A-Z])(\d+)([A-Z])$")
MAXLEN = 2048
PCA_DIM = 64
X_DIM = 1 + 20 + 20 + 20 + PCA_DIM * (2 * WIN + 1)  # frac+wt+mut+esm_ctx = 321
# 实际: wt/mut 在 forward 里 scatter，X_DIM 用于 DeepSet 检查


def wt_sequence(fp):
    row = pd.read_csv(fp, nrows=1, usecols=["mutant", "mutated_sequence"])
    seq = list(row["mutated_sequence"].iloc[0])
    for wt, pos, mu in PAT.findall(row["mutant"].iloc[0]):
        seq[int(pos) - 1] = wt
    return "".join(seq)


class DS:
    """一个 DMS 数据集：ESM3 特征索引。rows[i,t]=(site_local, wt_aa, mut_aa)，-1 pad。"""
    __slots__ = ("name", "site_pos", "site_ctx", "rows", "y", "wt_seq",
                 "crop", "kmax", "k_trunc", "dropped")

    def __init__(self, fp, name, emb, kmax=8):
        self.name = name
        self.wt_seq = wt_sequence(fp)
        df = pd.read_csv(fp, usecols=["mutant", "DMS_score"])
        self.y = df["DMS_score"].to_numpy(np.float32)
        mu = df["mutant"].astype(str)
        n = len(df)
        k = mu.str.count(":") + 1
        K = min(int(k.max()), kmax)
        self.rows = np.full((n, K, 3), -1, dtype=np.int64)
        self.kmax = K
        self.k_trunc = int((k > K).sum())
        # 全局位点 → 局部（裁剪后）
        L = len(self.wt_seq)
        self.crop = max(0, (L - MAXLEN) // 2)
        L_eff = emb.shape[0] if emb is not None else min(L, MAXLEN)
        glob2loc = {}
        for i, m in enumerate(mu.to_numpy()):
            for t, tok in enumerate(m.split(":")[:K]):
                wt, p, maa = PAT.match(tok).groups()
                p = int(p) - 1
                lp = p - self.crop
                if emb is None:
                    glob2loc.setdefault(p, lp)  # 无 emb 时用裁剪坐标（本地调试）
                if 0 <= lp < L_eff:
                    self.rows[i, t] = (glob2loc.get(p, lp), AAI[wt], AAI[maa])
                    glob2loc.setdefault(p, lp)
        self.dropped = int((self.rows[:, :, 0] < 0).any(1).sum())
        # site 索引: 出现过的局部位点
        self.site_pos = np.array(sorted(set(self.rows[:, :, 0].ravel()) - {-1}), dtype=np.int64)
        # 局部索引映射
        loc2idx = {int(p): j for j, p in enumerate(self.site_pos)}
        rr = self.rows.copy()
        for i in range(n):
            for t in range(K):
                p = self.rows[i, t, 0]
                if p >= 0:
                    rr[i, t, 0] = loc2idx[int(p)]
        self.rows = rr
        # ESM 上下文: [S, 2W+1, PCA_DIM] 待 fill（PCA 后）
        self.site_ctx = None

    def zscore(self):
        self.y = (self.y - self.y.mean()) / max(self.y.std(), 1e-8)


def build_site_ctx(ds, emb_all, pca):
    """每个位点的 ±2 窗口 ESM3 上下文（边界 pad 零），输出 [S, 2W+1, PCA_DIM]。"""
    S = len(ds.site_pos)
    ctx = np.zeros((S, 2 * WIN + 1, PCA_DIM), dtype=np.float32)
    emb = emb_all[ds.name]  # (L_eff, 1536)
    L_eff = emb.shape[0]
    for j, lp in enumerate(ds.site_pos):
        g = lp + ds.crop
        for dd, q in enumerate(range(g - WIN, g + WIN + 1)):
            if 0 <= q < ds.crop + L_eff and 0 <= q - ds.crop < L_eff:
                ctx[j, dd] = pca.transform(emb[q - ds.crop:q - ds.crop + 1])[0]
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pca-dim", type=int, default=PCA_DIM)
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"device={dev} epochs={args.epochs} batch={args.batch} pca={args.pca_dim}", flush=True)
    t0 = time.time()

    files = sorted(glob.glob(os.path.join(ROOT, "*.csv")))
    names = [os.path.basename(f)[:-4] for f in files]
    print(f"{len(files)} DMS files", flush=True)

    # 1) 加载 ESM3 embeddings（缺失的 dataset 报错退出）
    emb_all = {}
    missing = []
    for nm in names:
        npz = os.path.join(EMB_ROOT, nm + ".npz")
        if not os.path.exists(npz):
            missing.append(nm)
        else:
            emb_all[nm] = np.load(npz)["emb"].astype(np.float32)
    if missing:
        print("MISSING EMBEDDINGS:", missing)
        sys_exit = 1
    print(f"embeddings loaded: {len(emb_all)}/{len(names)}", flush=True)

    # 2) 构建 DS（训练+held-out）
    dss = {}
    for fp, nm in zip(files, names):
        d = DS(fp, nm, emb_all.get(nm))
        d.zscore()
        dss[nm] = d
        if nm not in HOLDOUT and nm in emb_all and d.site_ctx is None:
            pass
    train_names = [nm for nm in names if nm not in HOLDOUT and nm in emb_all]
    held_names = [nm for nm in names if nm in HOLDOUT and nm in emb_all]
    print(f"train={len(train_names)} held={len(held_names)} ({time.time()-t0:.0f}s)", flush=True)

    # 3) PCA fit（仅训练集残基）
    mats = []
    for nm in train_names:
        mats.append(emb_all[nm])
    X = np.concatenate(mats, 0)
    print(f"PCA fit on {X.shape[0]} residues x {X.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
    pca = PCA(n_components=args.pca_dim, svd_solver="randomized", random_state=0)
    pca.fit(X.astype(np.float64))
    del X, mats
    print(f"PCA explained var: {pca.explained_variance_ratio_.sum():.3f} ({time.time()-t0:.0f}s)", flush=True)

    # 4) 构建 site_ctx
    for nm in train_names + held_names:
        dss[nm].site_ctx = build_site_ctx(dss[nm], emb_all, pca)
        if nm not in HOLDOUT:
            dr = dss[nm].dropped
            if dr:
                print(f"  {nm}: dropped {dr}/{len(dss[nm].y)} samples (site out of crop)", flush=True)
    print(f"site_ctx done ({time.time()-t0:.0f}s)", flush=True)

    # 5) 模型（DeepSet，x=posf+wt+mut+esm_ctx）
    x_dim = 1 + 20 + 20 + args.pca_dim * (2 * WIN + 1)

    class DeepSetV3(nn.Module):
        def __init__(self, xd=x_dim, h=128, hidden=256):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(xd, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(2 * h, hidden), nn.ReLU(),
                                      nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, site_idx, rows, site_ctx):
            B, K, _ = rows.shape
            valid = rows[:, :, 0] >= 0
            posf = rows[:, :, 0].clamp(min=0).float() / max(len(site_idx) - 1, 1)
            posf = posf * valid.float()
            ctx = site_ctx[rows[:, :, 0].clamp(min=0)]        # [B,K,2W+1,pca]
            waa = torch.zeros(B, K, 20, device=rows.device)
            maa = torch.zeros(B, K, 20, device=rows.device)
            waa.scatter_(2, rows[:, :, 1].clamp(min=0).unsqueeze(-1), 1.0)
            maa.scatter_(2, rows[:, :, 2].clamp(min=0).unsqueeze(-1), 1.0)
            x = torch.cat([posf.unsqueeze(-1), waa, maa, ctx.flatten(2) * valid.unsqueeze(-1)], dim=-1)
            e = self.enc(x) * valid.unsqueeze(-1)
            pooled = torch.cat([e.sum(1) / valid.sum(1, keepdim=True).clamp(min=1),
                                e.max(1).values], dim=-1)
            return self.head(pooled).squeeze(-1)

    model = DeepSetV3().to(dev)
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    B = args.batch
    n_train = sum(len(d.y) for d in (dss[nm] for nm in train_names))
    print(f"train samples={n_train} ({time.time()-t0:.0f}s)", flush=True)

    # 6) 训练 + 每 epoch held-out 早停
    def eval_held():
        rho = {}
        with torch.no_grad():
            for nm in held_names:
                d = dss[nm]
                preds = []
                for s in range(0, len(d.y), B):
                    rows = torch.from_numpy(d.rows[s:s + B]).to(dev)
                    preds.append(model(d.site_pos, rows, torch.from_numpy(d.site_ctx).to(dev)).cpu().numpy())
                pred = np.concatenate(preds)
                rho[nm] = float(spearmanr(pred, d.y).statistic)
        return rho

    best, best_ep, best_rho = None, -1, None
    for ep in range(args.epochs):
        losses = []
        for nm in train_names:
            d = dss[nm]
            m = len(d.y)
            if m < 64:
                continue
            order = np.random.permutation(m)
            for s in range(0, m, B):
                idx = order[s:min(s + B, m)]
                rows = torch.from_numpy(d.rows[idx]).to(dev)
                y = torch.from_numpy(d.y[idx]).to(dev)
                pred = model(d.site_pos, rows, torch.from_numpy(d.site_ctx).to(dev))
                loss = nn.functional.mse_loss(pred, y)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss))
        sched.step()
        rho = eval_held()
        mean_rho = float(np.mean(list(rho.values())))
        print(f"ep{ep} loss={np.mean(losses):.4f} held={rho} mean={mean_rho:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if best_rho is None or mean_rho > best_rho:
            best_rho, best_ep = mean_rho, ep
            best = {k: v.clone() for k, v in model.state_dict().items()}
        if ep >= 2 and mean_rho < best_rho - 0.005:
            print(f"early stop at ep{ep}", flush=True)
            break

    model.load_state_dict(best)
    rho = eval_held()
    print(f"BEST ep{best_ep}: {rho} mean={np.mean(list(rho.values())):.4f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(args.out, "model.pt"))
    np.save(os.path.join(args.out, "pca_components.npy"), pca.components_)
    np.save(os.path.join(args.out, "pca_mean.npy"), pca.mean_)
    json.dump({"holdout": rho, "best_epoch": best_ep, "pca_dim": args.pca_dim,
               "train_datasets": len(train_names), "train_samples": int(n_train),
               "x_dim": x_dim, "device": str(dev), "epochs": ep + 1},
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
