#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B12 — 路线A决定性实验: Potts 耦合参与外推（提案 §4.11）。

b11 终局（双家族 REVIEW）确认机制缺口: 适应度=逐位点祖先对数先验时,
二阶耦合结构从未被选择——"提议有耦合、选择无耦合"。本实验把耦合项以
权重 α 加回适应度, 并把校准判据从"均值唯一"升级为形状感知
（祖先侧二阶配对相关 KS p, 零盲测泄漏）。

方向约定（engine.potts.island_wf 最大化 fitness_fn 输出）:
  fitness_α(x) = logprior(x) − α·E(x)/K
  - E 为 plmDCA 能量: 类祖先（训练分布内）序列能量低 → −E 奖励类祖先
    耦合模式, 二阶结构由此可被选择并遗传;
  - gap 富集序列是高能量离群 → 被 −E 惩罚。这与 b11 §4.9 原劫持方向
    相反（原失败是"最大化 E"选中 gap 汤）; 且 §4.9 修复持续生效:
    gap 不进变异字母表, 退化守卫全程在岗。

协议（继承 b9/b11 口径不变）:
  - 截断 = 共识距离中位数二分; 盲测零泄漏（α/λ_mut/T 的选择只用祖先侧）;
  - λ 固定 0.2（b11 双家族一致选定, 不再搜索, 单点留痕）;
  - 判据升级: 均值达标（|replay−target| ≤ 25%·target）者中, 选祖先侧
    二阶配对相关 KS p 最大者（形状感知）; 无达标者回退最接近均值并标注;
  - 判决三条件不变（盲测 KS p>0.05 且 熵 rho>0 且 二阶配对相关 KS p>0.05）。

判读（决定性）:
  - α>0 当选 且 盲测 KS / 二阶 KS 显著改善 → 耦合参与外推有效, 路线A续;
  - α=0 当选 或 形状仍不过 → 模型类能力边界实锤, 转路线B
    （b10/b11 阴性结果可发表, 关闭 DCA 先验线）。

运行: python evo2/benchmarks/b12_potts_coupled.py <family_substring>
输出: evo2/results/b12_coupled/report_{tag}.md + summary_{tag}.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import ks_2samp, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE)))

from b9_natreplay import parse_a2m, encode_seqs, variable_sites, pair_distances, site_entropy  # noqa: E402
from b10_potts_coupling import load_family, to_masked  # noqa: E402
from engine.potts import PottsModel, fit_plm, island_wf  # noqa: E402
from b11_island_potts import sample_pair_dists, pair_support, paircorr_values  # noqa: E402

OUT = os.path.join(ROOT, "evo2", "results", "b12_coupled")
SEED = 20260906
Q = 21
FIT_N = 2000
EPOCHS, LR = 150, 0.05
EPS_SPARSE = 0.03
N_POP, NE, N_GEN = 8, 500, 2000          # 终跑规模 = b9/b11 口径
LAM_FIX = 0.2                             # b11 双家族一致选定, 不再搜索
ALPHA_GRID = [0.0, 0.3, 1.0, 3.0]         # 耦合权重（0 = b11 对照）
LAM_MUT_GRID = [2.0, 6.0, 12.0]
PAIRS = 2000
MEAN_TOL = 0.25                           # 均值达标容差（相对 target）
M_GRID = [0.0, 0.005, 0.02]


def main(substring=""):
    if not substring:
        print("错误: 必须显式传入家族名（家族锁定）。用法: "
              "b12_potts_coupled.py <family_substring>", file=sys.stderr)
        return 2
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    name, raw = load_family(substring)
    fam_tag = os.path.basename(name).split("_")[0]
    query, seqs = parse_a2m(raw)
    sites = variable_sites(query, seqs)
    K = len(sites)
    wt_str = "".join(query[c] for c in sites)
    mat = encode_seqs(seqs, wt_str, sites)
    print(f"[1] 家族 {name}: {len(seqs)} 序列, {K} 位点", flush=True)

    # ---- 截断（b9 口径, 与 b11 一致）----
    consensus = np.zeros(K, dtype=np.int8)
    for j in range(K):
        col = mat[:, j]; col = col[col != 31]
        consensus[j] = np.bincount(col, minlength=32).argmax() if len(col) else 0
    ok = mat != 31
    dist = np.where(ok, mat != consensus[None, :], False).sum(1) / np.maximum(ok.sum(1), 1)
    anc_mask, der_mask = dist <= np.median(dist), dist > np.median(dist)
    anc, der = mat[anc_mask], mat[der_mask]
    print(f"[2] 截断: 训练 {anc_mask.sum()} / 盲测 {der_mask.sum()}", flush=True)

    sel = rng.choice(anc.shape[0], size=min(FIT_N, anc.shape[0]), replace=False)
    anc21 = np.where(anc[sel] == 31, 20, anc[sel]).astype(np.int8)
    der = to_masked(der)
    anc_m = to_masked(anc)

    d_anc = pair_distances(anc_m, PAIRS, rng)
    d_der = pair_distances(der, PAIRS, rng)
    target = float(d_anc.mean())
    pc_anc_pairs = pair_support(anc_m)

    prop_states = np.arange(Q - 1)          # gap(20) 不进变异字母表（§4.9 修复）
    counts = np.zeros((K, Q))
    for s in range(Q):
        counts[:, s] = (anc21 == s).sum(0)
    logprior = np.log((counts + 0.5) / (counts.sum(1, keepdims=True) + 0.5 * Q))
    hp_row = np.arange(K)[None, :]

    print(f"[3] fit_plm λ={LAM_FIX}（b11 双家族选定值, 留痕不搜索）", flush=True)
    Wfull = fit_plm(anc21, Q=Q, epochs=EPOCHS, lam=LAM_FIX, lr=LR,
                    seed=SEED, verbose=False)
    pm = PottsModel(Wfull, eps=EPS_SPARSE, Q=Q)
    print(f"    稀疏耦合对 {len(pm.J)}/{K*(K-1)//2}", flush=True)

    def make_fitness(alpha):
        if alpha == 0.0:
            return lambda x: logprior[hp_row, x].sum(1)
        def f(x):
            return logprior[hp_row, x].sum(1) - alpha * pm.energy(x) / K
        return f

    # ---- §4.11 形状感知校准（全部祖先侧, 盲测零泄漏）----
    cands = []
    print(f"[4] α×λ_mut×T 联合校准（判据: 均值达标内取祖先侧二阶 KS p 最大; "
          f"α∈{ALPHA_GRID} × λ_mut∈{LAM_MUT_GRID}, 目标均值 {target:.3f}）",
          flush=True)
    for alpha in ALPHA_GRID:
        fit = make_fitness(alpha)
        f_sd = float(fit(anc21[:min(1000, len(anc21))]).std()) or 1.0
        t_grid = sorted({max(0.25, round(f_sd * f, 3)) for f in (0.5, 1.0, 2.0, 4.0)})
        for lmut in LAM_MUT_GRID:
            for T in t_grid:
                pops0 = anc21[rng.integers(0, len(anc21), 2 * 200)].reshape(2, 200, K)
                try:
                    end = island_wf(pm, pops0, n_gen=600, T=T, lam_mut=lmut,
                                    m_mig=0.005, rng=rng,
                                    proposal_states=prop_states, fitness_fn=fit)
                    sim_pop = to_masked(end.reshape(-1, K))
                    d = sample_pair_dists(sim_pop, 1000, rng)
                except RuntimeError:
                    print(f"    α={alpha:4.2f} λ_mut={lmut:5.2f} T={T:8.3f} → "
                          f"退化回放（守卫触发）", flush=True)
                    continue
                mean = float(d.mean())
                pairs_cal = sorted(set(pair_support(sim_pop)) & set(pc_anc_pairs))
                pc_p = (float(ks_2samp(paircorr_values(sim_pop, pairs_cal),
                                       paircorr_values(anc_m, pairs_cal)).pvalue)
                        if len(pairs_cal) >= 10 else 0.0)
                cands.append(dict(alpha=float(alpha), lam_mut=float(lmut),
                                  T=float(T), mean=mean, pc_p=pc_p))
                print(f"    α={alpha:4.2f} λ_mut={lmut:5.2f} T={T:8.3f} → "
                      f"均值 {mean:.3f} 祖先二阶KS p={pc_p:.3f}", flush=True)
    if not cands:
        print("全部组合回放退化——路线A在此协议下不可行（§4.11）", file=sys.stderr)
        return 3
    tol_ok = [c for c in cands if abs(c["mean"] - target) <= MEAN_TOL * target]
    pool, pooled = (tol_ok, True) if tol_ok else (cands, False)
    best = min(pool, key=lambda c: (-c["pc_p"], abs(c["mean"] - target),
                                    c["alpha"], c["lam_mut"], c["T"]))
    print(f"    冻结 α={best['alpha']}, λ_mut={best['lam_mut']}, T*={best['T']} "
          f"（均值 {best['mean']:.3f}, 祖先二阶KS p={best['pc_p']:.3f}; "
          f"均值达标组合数 {len(tol_ok)}/{len(cands)}）", flush=True)
    fit_chosen = make_fitness(best["alpha"])

    # ---- m 选择（祖先侧, 防泄漏, 同 b11）----
    best_m, best_p = M_GRID[0], -1.0
    for m in M_GRID:
        pops0 = anc21[rng.integers(0, len(anc21), 4 * 250)].reshape(4, 250, K)
        end = island_wf(pm, pops0, n_gen=800, T=best["T"], lam_mut=best["lam_mut"],
                        m_mig=m, rng=rng, proposal_states=prop_states,
                        fitness_fn=fit_chosen)
        d_sim_m = sample_pair_dists(to_masked(end.reshape(-1, K)), 1000, rng)
        p = float(ks_2samp(d_sim_m, d_anc).pvalue)
        print(f"    m={m:5.3f} → 组内 KS p={p:.3f}（均值 {d_sim_m.mean():.3f}）",
              flush=True)
        if p > best_p:
            best_m, best_p = m, p
    print(f"    选定 m = {best_m}（组内 p={best_p:.3f}）", flush=True)

    # ---- 终跑（b9/b11 规模）----
    pops0 = anc21[rng.integers(0, len(anc21), N_POP * NE)].reshape(N_POP, NE, K)
    end = island_wf(pm, pops0, n_gen=N_GEN, T=best["T"], lam_mut=best["lam_mut"],
                    m_mig=best_m, rng=rng, proposal_states=prop_states,
                    fitness_fn=fit_chosen)
    sim = to_masked(end.reshape(-1, K))
    print(f"[5] 终跑完成（{time.time()-t0:.0f}s）", flush=True)

    # ---- 判决（v1 三条件, 同 b11 口径）----
    d_sim = sample_pair_dists(sim, PAIRS, rng)
    ks_in = ks_2samp(d_sim, d_anc)
    ks_der = ks_2samp(d_sim, d_der)
    rho = float(spearmanr(site_entropy(sim), site_entropy(der)).statistic)
    pc_pairs = sorted(set(pair_support(sim)) & set(pair_support(der)))
    pc_sim = paircorr_values(sim, pc_pairs)
    pc_der = paircorr_values(der, pc_pairs)
    ks_pc = ks_2samp(pc_sim, pc_der)
    verdict = "PASS" if (ks_der.pvalue > 0.05 and rho > 0 and ks_pc.pvalue > 0.05) \
        else "REVIEW"

    # ---- b11 基线对照（动态读取）----
    b11 = None
    p11 = os.path.join(ROOT, "evo2", "results", "b11_island", f"summary_{fam_tag}.json")
    if os.path.isfile(p11):
        with open(p11, encoding="utf-8") as f:
            b11 = json.load(f)

    def g11(k):
        return b11.get(k, "—") if b11 else "—"

    def g11_ks(k):
        v = g11(k)
        return (f"D={v['D']:.3f}, p={v['p']:.3g}" if isinstance(v, dict) and 'D' in v
                else "—")

    route = ("路线A续: 耦合参与外推有效"
             if (best["alpha"] > 0 and ks_der.pvalue > 0.05 and ks_pc.pvalue > 0.05)
             else ("路线A未过: 形状/二阶仍不过——转路线B"
                   if best["alpha"] > 0 else
                   "α=0 当选: 耦合项无增益——转路线B"))

    report = f"""# B12 — Potts 耦合参与外推（路线A决定性实验, §4.11）

- 家族: `{name}`（{len(seqs)} 序列, {K} 位点; 训练 = ancestral 半子样 2000）
- 适应度: **logprior − α·E/K**（island_wf 最大化方向; −E 奖励类祖先耦合模式,
  gap 富集=高能量离群被惩罚, 与 b11 §4.9 原劫持方向相反; gap 不进变异字母表）
- 校准（祖先侧, 盲测零泄漏）: **判据升级为形状感知**——均值达标
  （≤{MEAN_TOL:.0%}·target）内取祖先侧二阶配对相关 KS p 最大;
  λ 固定 {LAM_FIX}（b11 双家族选定）; α∈{ALPHA_GRID} × λ_mut∈{LAM_MUT_GRID} × T~适应度 sd 锚定
  → **α={best['alpha']}, λ_mut={best['lam_mut']}, T={best['T']:.3f}**
  （均值 {best['mean']:.3f} vs 目标 {target:.3f}, 祖先二阶KS p={best['pc_p']:.3f},
  达标组合 {len(tol_ok)}/{len(cands)}）, m={best_m}（组内 p={best_p:.3f}）
- 墙钟: {time.time()-t0:.0f}s | seed {SEED}

## 判决: **{verdict}**（v1 标准: 盲测 KS p>0.05 且 熵 rho>0 且 二阶配对相关 KS p>0.05）

| 检验 | b12（本实验 α={best['alpha']}） | b11（α=0 隐含） |
|---|---|---|
| 盲测距离 KS（derived） | D={ks_der.statistic:.3f}, p={ks_der.pvalue:.3g} | {g11_ks('ks_der')} |
| 组内距离 KS（ancestral） | D={ks_in.statistic:.3f}, p={ks_in.pvalue:.3g} | {g11_ks('ks_in')} |
| 熵 Spearman（vs derived） | rho={rho:+.3f} | {g11('entropy_rho')} |
| 二阶配对相关 KS（vs derived） | D={ks_pc.statistic:.3f}, p={ks_pc.pvalue:.3g} | {g11_ks('ks_paircorr')} |

- 距离均值: 模拟 {d_sim.mean():.3f} vs 祖先实测 {d_anc.mean():.3f} / 盲测 {d_der.mean():.3f}
  （b11: 模拟 {g11('d_sim')} / 盲测 {g11('d_der')}）
- 二阶配对相关性: 模拟均值 {pc_sim.mean():+.3f} / 实测(derived)均值 {pc_der.mean():+.3f}

## 判读: {route}

- α 网格含 0.0 对照: α=0 应复现 b11 行为（提议耦合/选择无耦合）;
  α>0 当选且形状改善才算"耦合参与外推有效"。
- 若转路线B: b10/b11/b12 三代阴性证据链完整（均匀混合→岛屿→耦合入适应度）,
  "DCA 二阶耦合对共识距离二分回放的外推无增益"可作为可发表阴性结论,
  EvoTrace 主线收敛 v4+DL hybrid。
"""
    with open(os.path.join(OUT, f"report_{fam_tag}.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(OUT, f"summary_{fam_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(family=name, K=K, fam_tag=fam_tag, lam=LAM_FIX,
                       fitness="logprior-alpha_E_over_K(§4.11)",
                       alpha=best["alpha"], lam_mut=best["lam_mut"], T=best["T"],
                       m=best_m, m_select_p=best_p, n_sparse_pairs=len(pm.J),
                       calib_table=[{k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in c.items()} for c in cands],
                       mean_ok_n=len(tol_ok), n_cands=len(cands),
                       d_sim=round(float(d_sim.mean()), 3),
                       d_anc=round(float(d_anc.mean()), 3),
                       d_der=round(float(d_der.mean()), 3),
                       ks_in_p=float(ks_in.pvalue), ks_der_p=float(ks_der.pvalue),
                       ks_der_D=float(ks_der.statistic),
                       entropy_rho=float(rho), ks_pc_p=float(ks_pc.pvalue),
                       ks_pc_D=float(ks_pc.statistic),
                       verdict=verdict, route=route, seed=SEED,
                       wall_s=round(time.time() - t0, 1)), f, indent=1,
                   ensure_ascii=False)
    print(f"[6] 判决 {verdict}（{route}）: 盲测 KS p={ks_der.pvalue:.3g}, "
          f"rho={rho:+.3f}, 二阶配对相关 KS p={ks_pc.pvalue:.3g}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(main(""))
    sys.exit(main(a[0]))
