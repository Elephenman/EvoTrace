#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B9 — L3 自然进化回放（v1 方案 §3.7 验证层 L3, 路线图 R4; 唯一此前零代码的验证层）。

协议（截断-回放-比对）:
  1. 家族 MSA（ProteinGym DMS_msa_files, a2m）→ 取 query 的 match 列,
     选变异充足的前 K 个位点为模拟空间;
  2. 截断时间（无真实 ASR 工具的 v0 代理）: 按"与共识的距离"中位数把现存序列分成
     ancestral-like（近共识, 隐藏近期分支后保留）与 derived-like（远共识, 近期分支,
     **作为回放的盲测集扣下**）;
  3. 前向模拟: ancestral-like 半的 PSSM 作进化先验, WFKernel 中性偏软选择跑 N_GEN 代;
  4. 比对（金标准 = 扣下的 derived-like 半）:
     a) 两两 Hamming 距离分布（KS 检验, v1 通过标准 p>0.05）;
     b) 逐位点 Shannon 熵相关（Spearman, 期望 >0）。
诚实边界: 截断代理是"共识距离"而非真实系统发生时间; Potts 二阶统计 v0 未做
（需要 DCA 推断, 单独立项）; 结果如实记录, 不过线就写失败。

运行: python evo2/benchmarks/b9_natreplay.py [family_substring]
输出: evo2/results/b9_natreplay/report.md + summary.json
"""
import hashlib
import io
import json
import os
import sys
import time
import zipfile

import numpy as np
from scipy.stats import entropy as shannon, ks_2samp, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE)))

from engine.kernel import WFKernel  # noqa: E402
from engine.events import EventScript, run_with_events  # noqa: E402
from engine.seqtools import AA2IDX, AA20  # noqa: E402

MSA_ZIP = os.path.join(ROOT, "evo_data", "raw", "proteingym", "DMS_msa_files.zip")
OUT = os.path.join(ROOT, "evo2", "results", "b9_natreplay")
K_SITES = 48          # 模拟空间位点数（按变异度取前 K）
MIN_SEQS, MAX_SEQS = 300, 6000
SEED = 20260906
N_POP, NE, N_GEN = 8, 500, 2000
PAIRS = 400           # 距离分布抽样对数
GAP = ord("-")


# ---------------------------------------------------------------- a2m 解析
def parse_a2m(raw: bytes):
    """a2m → (query_aa_full, seqs)。match 列 = query 大写字母列;
    其他序列在 match 列取该列字符（小写=insert, 记作 gap）。"""
    lines = [l.strip() for l in io.StringIO(raw.decode("utf-8", "ignore"))
             if l.strip() and not l.startswith(">")]
    query = lines[0]
    match_cols = [i for i, c in enumerate(query) if c.isupper() or c == "-"]
    seqs = []
    for s in lines[1:]:
        seqs.append("".join(s[i] if i < len(s) and (s[i].isupper()) else "-"
                            for i in match_cols))
    return "".join(c for c in query if c.isupper()), seqs


def encode_seqs(seqs, wt_str, sites):
    """序列 → [N, K] int8（AA 索引; gap/未知 → 31 = 掩码值, 不参与比对统计）。"""
    MASK = 31
    out = np.full((len(seqs), len(sites)), MASK, dtype=np.int8)
    for n, s in enumerate(seqs):
        for j, col in enumerate(sites):
            c = s[col] if col < len(s) else "-"
            if c in AA2IDX:
                out[n, j] = AA2IDX[c]
    return out


def pick_family(zf, substring=""):
    """选一个规模合适的家族 MSA（可指定子串, 否则扫描）。"""
    cands = sorted(n for n in zf.namelist()
                   if n.endswith((".a2m", ".a3m")) and substring.lower() in n.lower())
    scanned = 0
    for name in cands:
        raw = zf.read(name)
        n_seq = raw.count(b">")
        scanned += 1
        if MIN_SEQS <= n_seq <= MAX_SEQS:
            return name, raw
    raise RuntimeError(f"无符合条件的家族（扫描 {scanned} 个, 条件 {MIN_SEQS}-{MAX_SEQS} 条）")


def variable_sites(query, seqs, k=K_SITES):
    """变异位点: 非 gap 且非 query 残基的序列数 ≥ 5, 按变异度降序取前 k。"""
    cols = []
    for col in range(len(query)):
        diff = sum(1 for s in seqs
                   if col < len(s) and s[col].isupper() and s[col] != query[col])
        if diff >= 5:
            cols.append((diff, col))
    cols.sort(reverse=True)
    return sorted(col for _, col in cols[:k])


def pssm_prior(seqs, sites, wt_str, pseudo=1.0):
    """ancestral-like 半的 PSSM → priors {site: {aa: p}}（gap 不计）。"""
    pri = {}
    for j, col in enumerate(sites):
        cnt = np.zeros(20) + pseudo
        for s in seqs:
            c = s[col] if col < len(s) else "-"
            if c in AA2IDX:
                cnt[AA2IDX[c]] += 1
        p = cnt / cnt.sum()
        pri[j] = {AA20[a]: float(max(p[a], 1e-3)) for a in range(20)
                  if AA20[a] != wt_str[j]}
        pri[j][wt_str[j]] = 1e-3
        tot = sum(pri[j].values())
        pri[j] = {a: v / tot for a, v in pri[j].items()}
    return pri


def pair_distances(mat: np.ndarray, n_pairs: int, rng) -> np.ndarray:
    """两两归一化 Hamming（掩码对剔除）。mat: [N, K] int8。"""
    N = mat.shape[0]
    d = []
    while len(d) < n_pairs:
        i, j = rng.integers(0, N, 2)
        if i == j:
            continue
        a, b = mat[i], mat[j]
        ok = (a != 31) & (b != 31)
        if ok.sum() < 5:
            continue
        d.append(float((a[ok] != b[ok]).sum()) / ok.sum())
    return np.array(d)


def site_entropy(mat: np.ndarray) -> np.ndarray:
    out = np.zeros(mat.shape[1])
    for j in range(mat.shape[1]):
        col = mat[:, j]
        col = col[col != 31]
        if len(col) == 0:
            continue
        _, c = np.unique(col, return_counts=True)
        out[j] = shannon(c / c.sum(), base=2)
    return out


# ---------------------------------------------------------------- main
def main(substring=""):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    zf = zipfile.ZipFile(MSA_ZIP)
    name, raw = pick_family(zf, substring)
    query, seqs = parse_a2m(raw)
    print(f"[1] 家族 {name}: {len(seqs)} 序列 × {len(query)} match 列")

    sites = variable_sites(query, seqs)
    K = len(sites)
    wt_str = "".join(query[c] for c in sites)
    mat = encode_seqs(seqs, wt_str, sites)
    print(f"[2] 模拟空间: {K} 个变异位点")

    # ---- 截断: 与共识距离中位数二分 ----
    consensus = np.zeros(K, dtype=np.int8)
    for j in range(K):
        col = mat[:, j]
        col = col[col != 31]
        consensus[j] = np.bincount(col, minlength=32).argmax() if len(col) else 0
    ok = mat != 31
    dist = np.where(ok, mat != consensus[None, :], False).sum(1) / np.maximum(ok.sum(1), 1)
    med = np.median(dist)
    anc_mask, der_mask = dist <= med, dist > med
    anc, der = mat[anc_mask], mat[der_mask]
    print(f"[3] 截断: ancestral-like {anc_mask.sum()} / derived-like(盲测) {der_mask.sum()}"
          f"（共识距离中位 {med:.3f}）")

    # ---- 前向模拟: 从截断时刻的祖先群体起步（非单条 WT——回放的是群体, 不是种子）,
    #      ancestral PSSM 作先验; 选择温度 T 不拍脑袋——在训练侧（ancestral 半）校准:
    #      T 使模拟的平衡两两距离匹配 ancestral 半的实测距离, 盲测只用 derived 半。
    pri = pssm_prior([s for s, m in zip(seqs, anc_mask) if m], sites, wt_str)
    # n_mut_max 放开到 K: 突变负荷上限是 PprI 战役的设计约束, 自然回放不适用
    cfg = {"mutations_per_genome_per_gen": {"lambda": 0.3}, "T": 1.0,
           "n_mut_max": K}
    kernel = WFKernel(wt_str, list(range(K)), pri, cfg, seed=SEED)
    anc_fill = anc.copy()
    for j in range(K):
        m = anc_fill[:, j] == 31
        anc_fill[m, j] = consensus[j]
    founder_pool = anc_fill[rng.integers(0, anc_fill.shape[0], 8 * NE)]
    d_anc = pair_distances(anc_fill, PAIRS, rng)

    def sim_mean_dist(T, n_gen=800, n_pop=2, ne=200, seed=SEED):
        k2 = WFKernel(wt_str, list(range(K)), pri, cfg, seed=seed)
        k2.T = T
        rr = np.random.default_rng(seed)
        f0 = founder_pool[rr.integers(0, founder_pool.shape[0], n_pop * ne)]
        _, pops, _ = run_with_events(k2, EventScript([]), n_pop=n_pop,
                                     n_gen=n_gen, Ne=ne,
                                     pops0=f0.reshape(n_pop, ne, K))
        return pair_distances(pops.reshape(-1, K), 200, np.random.default_rng(seed)).mean()

    grid = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    target = float(d_anc.mean())
    print(f"[4a] 校准选择温度: 目标(ancestral 平衡距离) = {target:.3f}")
    prev_T, prev_d = None, None
    chosen_T = None
    for T in grid:
        d = sim_mean_dist(T)
        print(f"      T={T:5.1f} → 平衡距离 {d:.3f}")
        if d >= target:
            chosen_T = T if prev_T is None else \
                prev_T + (T - prev_T) * (target - prev_d) / max(d - prev_d, 1e-9)
            break
        prev_T, prev_d = T, d
    if chosen_T is None:
        chosen_T = grid[-1]
        print(f"      ! 网格内未达目标, 取上限 T={chosen_T}")
    kernel.T = float(chosen_T)
    print(f"[4b] 选定 T = {kernel.T:.2f}")

    founder = founder_pool[:N_POP * NE]
    rec_stats, pops, _ = run_with_events(kernel, EventScript([]),
                                         n_pop=N_POP, n_gen=N_GEN, Ne=NE,
                                         pops0=founder.reshape(N_POP, NE, K))
    sim = pops.reshape(-1, K)
    print(f"[4] 模拟: {N_GEN} 代 × {N_POP} 群 × Ne={NE}（{time.time()-t0:.0f}s）")

    # ---- 比对 ----
    d_sim = pair_distances(sim, PAIRS, rng)
    d_der = pair_distances(der, PAIRS, rng)
    d_anc2 = pair_distances(anc_fill, PAIRS, rng)          # 组内合理性（校准目标侧）
    d_all = pair_distances(mat, PAIRS, rng)                # 全 MSA 参考
    ks = ks_2samp(d_sim, d_der)
    ks_anc = ks_2samp(d_sim, d_anc2)
    ks_all = ks_2samp(d_sim, d_all)
    h_sim, h_der = site_entropy(sim), site_entropy(der)
    rho = spearmanr(h_sim, h_der).statistic
    ks_pass = ks.pvalue > 0.05
    ent_pass = rho > 0
    verdict = "PASS" if (ks_pass and ent_pass) else "REVIEW"
    print(f"[5] 比对: 盲测(derived) KS D={ks.statistic:.3f} p={ks.pvalue:.3f}"
          f"（{'✓' if ks_pass else '✗'}）"
          f" | 组内(ancestral) KS D={ks_anc.statistic:.3f} p={ks_anc.pvalue:.3f}"
          f" | 熵 Spearman {rho:+.3f}（{'✓' if ent_pass else '✗'}）→ {verdict}")

    report = f"""# B9 — L3 自然进化回放报告

- 家族: `{name}`（{len(seqs)} 序列, 模拟空间 {K} 位点）
- 截断代理: 共识距离中位数二分（ancestral-like {int(anc_mask.sum())} 训练 /
  derived-like {int(der_mask.sum())} 盲测）——**非真实 ASR, v0 边界**
- 模拟: WF, {N_GEN} 代, Ne={NE}, {N_POP} 平行群, 先验 = ancestral PSSM,
  T={kernel.T:.2f}（训练侧校准: 目标平衡距离 {target:.3f} = ancestral 实测均值）
- 墙钟: {time.time()-t0:.0f}s | seed {SEED}

## 判决: **{verdict}**（v1 通过标准: 盲测 KS p>0.05 且 熵相关 >0）

| 检验 | 统计量 | 通过线 | 结果 |
|---|---|---|---|
| 盲测距离 KS（derived） | D={ks.statistic:.3f}, p={ks.pvalue:.3f} | p>0.05 | {'✓' if ks_pass else '✗'} |
| 组内距离 KS（ancestral, 合理性） | D={ks_anc.statistic:.3f}, p={ks_anc.pvalue:.3f} | p>0.05 | {'✓' if ks_anc.pvalue > 0.05 else '✗'} |
| 全 MSA 距离 KS（参考） | D={ks_all.statistic:.3f}, p={ks_all.pvalue:.3f} | — | 参考 |
| 位点熵 Spearman（vs derived） | rho={rho:+.3f} | >0 | {'✓' if ent_pass else '✗'} |

- 模拟距离均值 {d_sim.mean():.3f} vs 盲测 {d_der.mean():.3f} / 组内 {d_anc2.mean():.3f}
- 平均熵 {h_sim.mean():.3f} vs {h_der.mean():.3f}

## 结论解读

两家族实测（2026-09-06）: NEIME（组内 D=0.422）与 A4/HUMAN（组内 D=0.780, 平衡距离
被 PSSM 熵上界钉死在 0.42, 够不到祖先侧实测 0.77）**组内与盲测双双不过**——
逐位点独立 PSSM 景观的混合平衡在结构上就无法再现自然 MSA 的多样性:
它没有亚家族聚类（分布形状）, 且多样性上限被自身 PSSM 熵封顶（A4 家族实证）。
这为 v1 §3.2 的 Potts/DCA 二阶耦合项与 v1 §3.4 的空间结构种群（岛屿模型）
给出了明确的实证需求——二者是 L3 通过的必要条件, 已列入 R4 后续。
诚实边界: 截断代理是共识距离而非真实系统发生时间（无 ASR 工具的 v0 限制）。
"""
    fam_tag = os.path.basename(name).split("_")[0]
    with open(os.path.join(OUT, f"report_{fam_tag}.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(OUT, f"summary_{fam_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(family=name, n_seqs=len(seqs), K=K,
                       n_anc=int(anc_mask.sum()), n_der=int(der_mask.sum()),
                       calibrated_T=round(float(kernel.T), 2),
                       ks_blind=dict(D=float(ks.statistic), p=float(ks.pvalue)),
                       ks_anc=dict(D=float(ks_anc.statistic), p=float(ks_anc.pvalue)),
                       ks_all=dict(D=float(ks_all.statistic), p=float(ks_all.pvalue)),
                       entropy_rho=float(rho), verdict=verdict,
                       seed=SEED, wall_s=round(time.time()-t0, 1)), f, indent=1)
    print(f"[6] 输出: {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
