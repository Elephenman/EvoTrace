#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B10 — R4 后续: DCA 二阶耦合推断 + Potts 平衡采样（机制提案门验证实验）。

动机（b9 判决的必要条件检验）: b9 两家族组内/盲测双不过, A4/HUMAN 的回放平衡
距离被逐位点 PSSM 熵结构性封顶（模拟 ~0.42 vs 祖先侧实测 ~0.77）。假设: 二阶
耦合（Potts/DCA, v1 §3.2）是解除封顶、再现自然多样性的必要项。

协议:
  1. 复用 b9 家族装载/截断协议（共识距离中位数二分, ancestral 半 = 训练集）;
  2. 伪似然 DCA: 逐位点 L2 正则 softmax 回归（one-hot 全谱特征, 排除自身位点,
     Adam 全批次）——plmDCA 的标准 v0 实现;
  3. Potts 吉布斯采样: 用拟合条件分布 P(x_i | x_rest) 做系统扫描 Gibbs,
     从祖先群体起步, 烧入后收集平衡样本;
  4. 比对（同 b9 口径）: 两两 Hamming 距离 KS（组内/盲测/全 MSA）+ 位点熵
     Spearman。判决线: 组内 KS p>0.05 且模拟均值距离 ≥ 0.8× 祖先实测
     （突破 PSSM 熵上界）。

诚实边界: 本实验直接采样拟合模型的平衡分布（检验"目标分布"问题）, 不经过
WF 进化动力学; WF 内核的联合提议接入是过门后的下一件事。无温度调节 v0
（T=1 固定）。

运行: python evo2/benchmarks/b10_potts_coupling.py [family_substring]
输出: evo2/results/b10_potts/report_{tag}.md + summary_{tag}.json
"""
import io
import json
import os
import sys
import time
import zipfile

import numpy as np
from scipy.stats import ks_2samp, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE)))

from b9_natreplay import (MSA_ZIP, parse_a2m, encode_seqs,  # noqa: E402
                          variable_sites, pair_distances, site_entropy)

OUT = os.path.join(ROOT, "evo2", "results", "b10_potts")
SEED = 20260906
Q = 21                 # 20 AA + gap(=20); b9 的掩码 31 在入口重映射
FIT_N = 2000           # 伪似然拟合的训练子样（祖先半内随机抽）
EPOCHS = 150           # Adam 全批次轮数
LAM = 0.01             # L2 正则（plmDCA 标准量级）
LR = 0.05
M_CHAINS = 1000        # Gibbs 链数
BURN, COLLECT = 200, 100
PAIRS = 2000


def pick_by_substring(zf, substring=""):
    """按子串选家族（b10 宽松版: 只设下限 300 序列, 不设上限——A4 有 5 万+ 序列）。"""
    cands = sorted(n for n in zf.namelist()
                   if n.endswith((".a2m", ".a3m")) and substring.lower() in n.lower())
    for name in cands:
        raw = zf.read(name)
        if raw.count(b">") >= 300:
            return name, raw
    raise RuntimeError(f"无符合条件家族（候选 {len(cands)} 个）")


A2M_DIR = os.path.join(ROOT, "evo_data", "raw", "proteingym", "a2m")


def load_family(substring=""):
    """数据源自动选择: 有 a2m 目录用目录（集群轻量部署）, 否则回退 zip。"""
    if os.path.isdir(A2M_DIR):
        cands = sorted(f for f in os.listdir(A2M_DIR)
                       if f.endswith((".a2m", ".a3m")) and substring.lower() in f.lower())
        for f in cands:
            raw = open(os.path.join(A2M_DIR, f), "rb").read()
            if raw.count(b">") >= 300:
                return f, raw
        raise RuntimeError(f"无符合条件家族（候选 {len(cands)} 个, dir={A2M_DIR}）")
    return pick_by_substring(zipfile.ZipFile(MSA_ZIP), substring)


def onehot(mat21):
    """[N,K] int8 (0..20) → [N, K*Q] float32 one-hot。"""
    N, K = mat21.shape
    F = np.zeros((N, K * Q), dtype=np.float32)
    F[np.arange(N)[:, None], np.arange(K)[None, :] * Q + mat21] = 1.0
    return F


def fit_plm(mat21, rng):
    """伪似然 DCA: 逐位点 softmax 回归。返回 Wfull [K, K*Q+1, Q]
    （前 K*Q 行 = 特征权重, 末行 = 场 h_i; 自身位点块在返回前置零）。"""
    N, K = mat21.shape
    D = K * Q
    F = onehot(mat21)
    Wfull = np.zeros((K, D + 1, Q), dtype=np.float32)
    t0 = time.time()
    for i in range(K):
        others = np.concatenate([np.arange(0, i * Q), np.arange((i + 1) * Q, D)])
        Xi = np.hstack([F[:, others], np.ones((N, 1), dtype=np.float32)])
        Yi = np.zeros((N, Q), dtype=np.float32)
        Yi[np.arange(N), mat21[:, i]] = 1.0
        W = np.zeros((Xi.shape[1], Q), dtype=np.float32)
        m1 = np.zeros_like(W); m2 = np.zeros_like(W)
        for ep in range(EPOCHS):
            Z = Xi @ W
            Z -= Z.max(axis=1, keepdims=True)
            P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
            G = Xi.T @ (P - Yi) / N + LAM * W
            m1 = 0.9 * m1 + 0.1 * G
            m2 = 0.999 * m2 + 0.001 * G * G
            W -= LR * (m1 / (np.sqrt(m2) + 1e-8)) / (1 - 0.9 ** (ep + 1))
        # 写回全谱权重矩阵（排除自身位点块）
        Wfull[i, others, :] = W[:-1, :]
        Wfull[i, D, :] = W[-1, :]
        if (i + 1) % 12 == 0 or i == K - 1:
            print(f"    site {i+1}/{K}（{time.time()-t0:.0f}s）")
    return Wfull


def gibbs_sample(Wfull, init21, rng):
    """系统扫描 Gibbs。init21: [M,K] 起始态; 返回收集期样本 [M*COLLECT, K]。"""
    M, K = init21.shape
    D = K * Q
    F = onehot(init21)                       # [M, D]
    bias = np.ones((M, 1), dtype=np.float32)
    samples = []
    for s in range(BURN + COLLECT):
        for i in range(K):
            Fb = np.hstack([F, bias])
            logits = Fb @ Wfull[i]           # [M, Q]
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
            cum = np.cumsum(P, axis=1)
            u = rng.random((M, 1))
            new = (u > cum).sum(axis=1).clip(0, Q - 1).astype(np.int64)
            F[:, i * Q:(i + 1) * Q] = 0.0
            F[np.arange(M), i * Q + new] = 1.0
        if s >= BURN:
            samples.append(F.reshape(M, K, Q).argmax(axis=2).astype(np.int8))
    return np.array(samples)                 # [COLLECT, M, K]


def to_masked(mat21):
    """21 态（gap=20）→ b9 掩码口径（gap=31）。"""
    return np.where(mat21 == 20, 31, mat21).astype(np.int8)


def main(substring="", lam=LAM):
    global LAM
    LAM = float(lam)
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    zf = None
    name, raw = load_family(substring)
    query, seqs = parse_a2m(raw)
    sites = variable_sites(query, seqs)
    K = len(sites)
    wt_str = "".join(query[c] for c in sites)
    mat = encode_seqs(seqs, wt_str, sites)
    print(f"[1] 家族 {name}: {len(seqs)} 序列, {K} 位点")

    # ---- 截断（同 b9: 共识距离中位数二分）----
    consensus = np.zeros(K, dtype=np.int8)
    for j in range(K):
        col = mat[:, j]; col = col[col != 31]
        consensus[j] = np.bincount(col, minlength=32).argmax() if len(col) else 0
    ok = mat != 31
    dist = np.where(ok, mat != consensus[None, :], False).sum(1) / np.maximum(ok.sum(1), 1)
    anc_mask, der_mask = dist <= np.median(dist), dist > np.median(dist)
    anc, der = mat[anc_mask], mat[der_mask]
    print(f"[2] 截断: 训练(anc) {anc_mask.sum()} / 盲测(der) {der_mask.sum()}")

    # ---- 训练子样 → 21 态编码 ----
    sel = rng.choice(anc.shape[0], size=min(FIT_N, anc.shape[0]), replace=False)
    mat21 = np.where(anc[sel] == 31, 20, anc[sel]).astype(np.int8)
    der21 = np.where(der == 31, 20, der).astype(np.int8)
    all21 = np.where(mat == 31, 20, mat).astype(np.int8)

    print(f"[3] 伪似然 DCA 拟合（N={len(mat21)}, epochs={EPOCHS}, λ={LAM}）")
    Wfull = fit_plm(mat21, rng)

    print(f"[4] Gibbs 平衡采样（{M_CHAINS} 链 × 烧入{BURN}+收集{COLLECT} 扫描）")
    init = mat21[rng.integers(0, len(mat21), M_CHAINS)]
    samp = gibbs_sample(Wfull, init, rng).reshape(-1, K)
    sim = to_masked(samp[rng.integers(0, len(samp), 4000)])
    print(f"    样本 {samp.shape} → 评测子样 {sim.shape}（{time.time()-t0:.0f}s）")

    # ---- 比对（同 b9 口径, 距离均按掩码列归一）----
    d_sim = pair_distances(sim, PAIRS, rng)
    d_anc = pair_distances(anc, PAIRS, rng)          # 组内（掩码, 不回填——比 b9 的
    d_anc_fill = pair_distances(                     #   anc_fill 口径更保守）
        np.where(anc == 31, consensus[None, :], anc), PAIRS, rng)
    d_der = pair_distances(der, PAIRS, rng)
    d_all = pair_distances(mat, PAIRS, rng)
    ks_in = ks_2samp(d_sim, d_anc)
    ks_der = ks_2samp(d_sim, d_der)
    ks_all = ks_2samp(d_sim, d_all)
    rho = spearmanr(site_entropy(sim), site_entropy(der)).statistic

    # 判决: 组内过 + 突破 PSSM 熵封顶（b9 基线: A4 组内 D=0.780 / 模拟均值 0.42）
    cap_pass = d_sim.mean() >= 0.8 * d_anc.mean()
    ks_pass = ks_in.pvalue > 0.05
    verdict = "PASS" if (ks_pass and cap_pass) else "REVIEW"

    # 耦合强度 top 对（Frobenius, 报告用）
    D_ = K * Q
    J = Wfull[:, :D_, :].copy()
    fro = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            blk_i = J[i, j * Q:(j + 1) * Q, :]
            fro[i, j] = np.linalg.norm(blk_i)
    top = [(int(a), int(b), round(float(fro[a, b]), 2))
           for a, b in zip(*np.unravel_index(np.argsort(fro.ravel())[::-1][:8], fro.shape))
           if fro[a, b] > 0]

    report = f"""# B10 — DCA 二阶耦合 + Potts 平衡采样报告

- 家族: `{name}`（{len(seqs)} 序列, {K} 位点; 训练 = ancestral 半子样 {len(mat21)}）
- 方法: 伪似然 DCA（逐位点 L2 正则 softmax 回归, Adam {EPOCHS} 轮, λ={LAM}）
  → 条件分布 Gibbs 平衡采样（{M_CHAINS} 链, 烧入 {BURN} + 收集 {COLLECT} 扫描）
- 墙钟: {time.time()-t0:.0f}s | seed {SEED}

## 判决: **{verdict}**（组内 KS p>0.05 且 模拟均值距离 ≥ 0.8× 祖先实测）

| 检验 | 统计量 | b9 PSSM 基线 | 结果 |
|---|---|---|---|
| 组内距离 KS（ancestral） | D={ks_in.statistic:.3f}, p={ks_in.pvalue:.3f} | D=0.780（A4）/ 0.422（NEIME） | {'✓' if ks_pass else '✗'} |
| 盲测距离 KS（derived） | D={ks_der.statistic:.3f}, p={ks_der.pvalue:.3f} | D=0.9025（A4） | 参考 |
| 全 MSA 距离 KS | D={ks_all.statistic:.3f}, p={ks_all.pvalue:.3f} | — | 参考 |
| 熵 Spearman（vs derived） | rho={rho:+.3f} | −0.005（A4） | {'✓' if rho > 0 else '✗'} |

- 距离均值: 模拟 {d_sim.mean():.3f} vs 祖先实测 {d_anc.mean():.3f}
  （回填口径 {d_anc_fill.mean():.3f}）/ 盲测 {d_der.mean():.3f}
- **熵封顶检验**: b9 的 PSSM 基线把 A4 回放平衡距离钉死在 ~0.42
  （祖先实测 0.77）; 本实验模拟均值 {d_sim.mean():.3f}
  → {'已突破' if cap_pass else '仍受压'}（阈值 0.8× 实测 = {0.8*d_anc.mean():.3f}）
- 最强耦合对（Frobenius）: {top}

## 结论解读

- {'耦合感知的目标分布在组内多样性上' if ks_pass else '耦合先验仍未在组内多样性上'}
  {'通过了 KS 检验' if ks_pass else '未通过 KS 检验——见下方讨论'}。
- 本实验只回答"目标分布"问题（平衡分布形状）, 不涉及 WF 动力学;
  过门后的接入点: kernel 变异提议由逐位点 prior_table 升级为
  条件分布 P(x_i | x_rest)（联合提议）, 事件脚本/平行系接口不变。
- 诚实边界: 共识距离截断代理（非真实 ASR）; 无温度调节; plmDCA 均场近似
  的已知偏差（过估耦合）未做校正。
"""
    fam_tag = os.path.basename(name).split("_")[0]
    with open(os.path.join(OUT, f"report_{fam_tag}.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(OUT, f"summary_{fam_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(family=name, K=K, n_anc=int(anc_mask.sum()),
                       n_der=int(der_mask.sum()), fit_n=len(mat21),
                       epochs=EPOCHS, lam=LAM, chains=M_CHAINS,
                       d_sim=round(float(d_sim.mean()), 3),
                       d_anc=round(float(d_anc.mean()), 3),
                       d_der=round(float(d_der.mean()), 3),
                       ks_in=dict(D=float(ks_in.statistic), p=float(ks_in.pvalue)),
                       ks_der=dict(D=float(ks_der.statistic), p=float(ks_der.pvalue)),
                       ks_all=dict(D=float(ks_all.statistic), p=float(ks_all.pvalue)),
                       entropy_rho=float(rho), verdict=verdict,
                       top_couplings=top, seed=SEED,
                       wall_s=round(time.time() - t0, 1)), f, indent=1, ensure_ascii=False)
    print(f"[5] 判决 {verdict}: 组内 KS p={ks_in.pvalue:.3g}, "
          f"均值 {d_sim.mean():.3f} vs 实测 {d_anc.mean():.3f} → {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(a[0] if a else "", a[1] if len(a) > 1 else LAM))
