#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B11 — 岛屿模型 + Potts 耦合提议的 L3 回放（b9 协议复跑, 提案 §4 第 2/3 项）。

b9/b10 的缺口链条: 逐位点 PSSM → 多样性封顶（b9）; 二阶耦合 → 均值解锁但
分布形状不过（b10, 均匀混合无亚家族聚类）。本实验按 v1 §3.4 补第二个必要
条件——空间结构: n_pop 个岛独立漂变 + 低迁移率 m, 期望产生多峰分布。

协议（在 b9 基础上只换"引擎", 截断/盲测口径不变）:
  1. 家族装载/截断 = b9 共识距离中位数二分（ancestral 半 = 训练+校准,
     derived 半 = 盲测, 全程不碰）;
  2. λ 训练侧校准（祖先侧, 铁律: 只用 ancestral 半）: λ 网格
     [0.01, 0.05, 0.1, 0.2], 每个 λ 在祖先半上拟合后跑短回放
     （n_pop=2/ne=200/n_gen=600, 与 T 校准同尺度）, 选回放均值距离最接近
     祖先实测均值的 λ, 冻结后进入后续流程;
  3. 伪似然 DCA 拟合（engine.potts.fit_plm, v2 修正版 Adam）→ PottsModel;
  4. T 校准（祖先侧）: 岛屿 WF 短程跑, 选平衡均值距离 ≥ 祖先实测的最小 T;
  5. m 选择（祖先侧, 防泄漏）: m ∈ {0, 0.005, 0.02}, 取组内 KS p 最大者;
  6. 终跑: b9 同规模（2000 代 × 8 群 × Ne=500）, 盲测判决
     （v1 标准: 盲测 KS p>0.05 且 熵 Spearman >0 且 二阶配对相关 KS p>0.05）。

诚实边界: 能量即适应度是 Boltzmann 景观假设; plmDCA 过耦合靠 λ+eps 稀疏化
兜底; 截断代理仍是共识距离（非真实 ASR）; λ/T/m 校准尺度小于终跑尺度, 有
外推风险（详见报告 caveat）。

运行: python evo2/benchmarks/b11_island_potts.py <family_substring> [force_lam]
  <family_substring> 必须显式传入（家族锁定, P2-1）; 不传则报错退出。
  [force_lam] 可选: 给定则跳过 λ 网格校准直接强制使用该 λ（调试用）。
输出: evo2/results/b11_island/report_{tag}.md + summary_{tag}.json
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

OUT = os.path.join(ROOT, "evo2", "results", "b11_island")
SEED = 20260906
Q = 21
FIT_N = 2000
EPOCHS, LR = 150, 0.05
EPS_SPARSE = 0.03
N_POP, NE, N_GEN = 8, 500, 2000          # 终跑规模 = b9 口径
LAM_MUT = 0.3
PAIRS = 2000
# λ 训练侧校准网格（P1-1, 铁律: 只用祖先半 anc21）
LAM_GRID = [0.01, 0.05, 0.1, 0.2]
T_REF = 1.0                              # λ 校准参考选择温度（T 校准另做）
T_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
M_GRID = [0.0, 0.005, 0.02]


def run_mean_dist(pm, anc21, T, m, n_gen=600, n_pop=2, ne=200, seed=SEED):
    """短程岛屿 WF 的平衡均值距离（校准用）。"""
    rng = np.random.default_rng(seed)
    pops0 = anc21[rng.integers(0, len(anc21), n_pop * ne)].reshape(n_pop, ne, -1)
    end = island_wf(pm, pops0, n_gen=n_gen, T=T, lam_mut=LAM_MUT, m_mig=m, rng=rng)
    d = pair_distances(to_masked(end.reshape(-1, end.shape[-1])), 400,
                       np.random.default_rng(seed))
    return float(d.mean())


def pair_support(mat, min_obs=10):
    """通过 min_obs 门限的位点对列表（gap=31 掩码, 逐对剔除）。"""
    K = mat.shape[1]
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            ok = (mat[:, i] != 31) & (mat[:, j] != 31)
            if ok.sum() >= min_obs:
                pairs.append((i, j))
    return pairs


def paircorr_values(mat, pairs):
    """在给定共享位点对集合上算 Pearson r（P2-a: 两分布必须同对集合）。

    常数列（std≈0）对记 r=0 而非剔除——保证 sim/der 逐对可比、
    两分布样本量严格一致。用于 P1-2 二阶统计判决。"""
    out = []
    for i, j in pairs:
        a, b = mat[:, i], mat[:, j]
        ok = (a != 31) & (b != 31)
        av, bv = a[ok].astype(np.float64), b[ok].astype(np.float64)
        if av.std() < 1e-9 or bv.std() < 1e-9:
            out.append(0.0)
        else:
            out.append(float(np.corrcoef(av, bv)[0, 1]))
    return np.array(out, dtype=float) if out else np.array([0.0])


def read_baseline(fam_tag, subdir):
    """按家族标签动态读基线 summary（P2-1, 不再硬编码 A4 数值）。"""
    p = os.path.join(ROOT, "evo2", "results", subdir, f"summary_{fam_tag}.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main(substring="", force_lam=None):
    if not substring:
        print("错误: 必须显式传入家族名（家族锁定, P2-1）。用法: "
              "b11_island_potts.py <family_substring> [force_lam]", file=sys.stderr)
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

    # ---- 截断（b9 口径）----
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
    der = to_masked(der)                                   # 盲测侧回 b9 掩码口径
    anc_m = to_masked(anc)

    # 祖先/盲测实测距离（供 λ 校准与 T 校准共用的目标/对照）
    d_anc = pair_distances(anc_m, PAIRS, rng)
    d_der = pair_distances(der, PAIRS, rng)
    target = float(d_anc.mean())

    # ---- λ 训练侧校准（P1-1, 铁律: 只用祖先半 anc21）----
    if force_lam is not None:
        best_lam = float(force_lam)
        print(f"[3] λ 校准: 跳过网格, 强制 λ={best_lam}", flush=True)
        Wfull = fit_plm(anc21, Q=Q, epochs=EPOCHS, lam=best_lam, lr=LR,
                        seed=SEED, verbose=True)
        pm = PottsModel(Wfull, eps=EPS_SPARSE, Q=Q)
        lam_calib = [dict(lam=best_lam, replay_mean=round(
            run_mean_dist(pm, anc21, T_REF, m=0.005), 3))]
    else:
        print(f"[3] λ 校准（网格 {LAM_GRID}, 参考 T={T_REF}, 目标均值 {target:.3f}）",
              flush=True)
        cand = {}
        for lam in LAM_GRID:
            Wfull_l = fit_plm(anc21, Q=Q, epochs=EPOCHS, lam=lam, lr=LR,
                              seed=SEED, verbose=False)
            pm_l = PottsModel(Wfull_l, eps=EPS_SPARSE, Q=Q)
            d = run_mean_dist(pm_l, anc21, T_REF, m=0.005)
            cand[lam] = (pm_l, d)
            print(f"    λ={lam:5.2f} → 回放均值 {d:.3f}（err {abs(d-target):.3f}）",
                  flush=True)
        best_lam = min(LAM_GRID, key=lambda l: abs(cand[l][1] - target))
        pm = cand[best_lam][0]
        lam_calib = [dict(lam=float(l), replay_mean=round(cand[l][1], 3))
                     for l in LAM_GRID]
        print(f"    冻结 λ = {best_lam}（回放均值 {cand[best_lam][1]:.3f} ≈ 目标 "
              f"{target:.3f}）", flush=True)
    LAM_REG = best_lam
    print(f"    稀疏耦合对 {len(pm.J)}/{K*(K-1)//2}", flush=True)

    # ---- T 校准（祖先侧均值）----
    print(f"[4] T 校准: 目标均值 {target:.3f}", flush=True)
    chosen_T = T_GRID[-1]
    for T in T_GRID:
        d = run_mean_dist(pm, anc21, T, m=0.005)
        print(f"    T={T:5.2f} → 均值 {d:.3f}", flush=True)
        if d >= target:
            chosen_T = T
            break
    print(f"    选定 T = {chosen_T}", flush=True)

    # ---- m 选择（祖先侧形状 KS, 防盲测泄漏）----
    best_m, best_p = M_GRID[0], -1.0
    for m in M_GRID:
        end = island_wf(pm, anc21[rng.integers(0, len(anc21), 4 * 250)].reshape(4, 250, K),
                        n_gen=800, T=chosen_T, lam_mut=LAM_MUT, m_mig=m, rng=rng)
        d_sim = pair_distances(to_masked(end.reshape(-1, K)), 1000, rng)
        p = float(ks_2samp(d_sim, d_anc).pvalue)
        print(f"    m={m:5.3f} → 组内 KS p={p:.3f}（均值 {d_sim.mean():.3f}）", flush=True)
        if p > best_p:
            best_m, best_p = m, p
    print(f"    选定 m = {best_m}（组内 p={best_p:.3f}）", flush=True)

    # ---- 终跑（b9 规模）----
    pops0 = anc21[rng.integers(0, len(anc21), N_POP * NE)].reshape(N_POP, NE, K)
    end = island_wf(pm, pops0, n_gen=N_GEN, T=chosen_T, lam_mut=LAM_MUT,
                    m_mig=best_m, rng=rng)
    sim = to_masked(end.reshape(-1, K))
    print(f"[5] 终跑完成（{time.time()-t0:.0f}s）", flush=True)

    # ---- 比对（b9 口径 + P1-2 二阶统计）----
    d_sim = pair_distances(sim, PAIRS, rng)
    ks_in = ks_2samp(d_sim, d_anc)
    ks_der = ks_2samp(d_sim, d_der)
    rho = float(spearmanr(site_entropy(sim), site_entropy(der)).statistic)
    # P1-2 + P2-a: 盲测 derived 半上, 回放 vs 实测的成对位点相关性分布 KS
    # （共享位点对集合 = 两侧 support 交集, 保证两分布逐对可比、样本量一致）
    pc_pairs = sorted(set(pair_support(sim)) & set(pair_support(der)))
    pc_sim = paircorr_values(sim, pc_pairs)
    pc_der = paircorr_values(der, pc_pairs)
    ks_pc = ks_2samp(pc_sim, pc_der)
    verdict = "PASS" if (ks_der.pvalue > 0.05 and rho > 0 and ks_pc.pvalue > 0.05) \
        else "REVIEW"

    # ---- P2-1 动态基线（按家族标签匹配, 不硬编码 A4）----
    b9 = read_baseline(fam_tag, "b9_natreplay")
    b10 = read_baseline(fam_tag, "b10_potts")

    def f_b9_ks(k):
        return (f"D={b9[k]['D']:.3f}, p={b9[k]['p']:.3f}"
                if (b9 and k in b9) else "—（缺 b9 基线）")

    def f_b10_ks(k):
        return (f"D={b10[k]['D']:.3f}, p={b10[k]['p']:.3f}"
                if (b10 and k in b10) else "—（缺 b10 基线）")

    b9_rho = f"{b9['entropy_rho']:+.3f}" if (b9 and 'entropy_rho' in b9) else "—"
    b10_rho = f"{b10['entropy_rho']:+.3f}" if (b10 and 'entropy_rho' in b10) else "—"
    b10_dsim = f"{b10['d_sim']:.3f}" if (b10 and 'd_sim' in b10) else "—"

    report = f"""# B11 — 岛屿模型 + Potts 耦合提议回放报告

- 家族: `{name}`（{len(seqs)} 序列, {K} 位点; 训练 = ancestral 半子样 {len(anc21)}）
- 引擎: engine.potts（v2 Adam）+ 岛屿 WF（{N_POP} 群, 终跑 {N_GEN} 代 × Ne={NE}）
- 校准（全部在祖先侧, 盲测零泄漏）: T={chosen_T}（均值目标 {target:.3f}）,
  m={best_m}（组内 KS p={best_p:.3f}）; λ_mut={LAM_MUT}, 稀疏耦合对 {len(pm.J)}
- λ 训练侧校准（P1-1）: 冻结 λ={LAM_REG}（网格 {LAM_GRID}, 参考 T={T_REF},
  目标均值 {target:.3f}）
- 墙钟: {time.time()-t0:.0f}s | seed {SEED}

## 判决: **{verdict}**（v1 标准: 盲测 KS p>0.05 且 熵 rho>0 且 二阶配对相关 KS p>0.05）

| 检验 | 统计量 | b9 PSSM 基线 | b10 Potts 均匀混合 |
|---|---|---|---|
| 盲测距离 KS（derived） | D={ks_der.statistic:.3f}, p={ks_der.pvalue:.3f} | {f_b9_ks('ks_blind')} | {f_b10_ks('ks_der')} |
| 组内距离 KS（ancestral） | D={ks_in.statistic:.3f}, p={ks_in.pvalue:.3f} | {f_b9_ks('ks_anc')} | {f_b10_ks('ks_in')} |
| 熵 Spearman（vs derived） | rho={rho:+.3f} | {b9_rho} | {b10_rho} |
| 二阶配对相关 KS（vs derived, P1-2） | D={ks_pc.statistic:.3f}, p={ks_pc.pvalue:.3f} | — | —（b10 未做） |

- 距离均值: 模拟 {d_sim.mean():.3f} vs 祖先实测 {d_anc.mean():.3f} / 盲测 {d_der.mean():.3f}
  （b10 Potts 均匀混合均值 {b10_dsim}）
- 二阶配对相关性: 模拟均值 {pc_sim.mean():+.3f} / 实测(derived)均值 {pc_der.mean():+.3f}

## 结论解读

- 岛屿结构（m={best_m}）{"引入" if best_m > 0 else "未引入"}多峰性;
  {"盲测通过 L3 v1 标准（含二阶统计）" if verdict == "PASS" else "仍未过 v1 盲测标准——归因如实记录"}。
- 对照链: PSSM（封顶）→ Potts 均匀混合（均值解锁/形状不过）→ Potts+岛屿（本实验）。
- 选择机制 = Potts 能量 softmax(E/T) 替换 b9 的 PSSM 先验软选择（**模型替换, 非调参**,
  P2-4）; 本实验为过门验证, 独立脚本合法, 过门后才并入 WFKernel/run_with_events（分阶段, 非豁免）。

## 外推风险声明（P2-2 caveat）

λ/T/m 校准均在短程小尺度（n_pop=2/ne=200/n_gen=600–800）完成, 终跑尺度
（n_pop=8/ne=500/n_gen=2000）更大, 平衡分布存在尺度外推风险; 若终跑 KS 不过
则回炉重做校准, 过门结论须谨慎解读。
另: 迁移率 m 在 λ 校准与 T 校准期被钉死为 0.005, m 的网格选择是在 λ/T 已冻结
前提下进行的——三者未做联合搜索, 耦合选择的最优组合可能被序列化决策错过。
"""
    with open(os.path.join(OUT, f"report_{fam_tag}.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(OUT, f"summary_{fam_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(family=name, K=K, fam_tag=fam_tag, lam_reg=LAM_REG,
                       lam_calib=lam_calib, T=chosen_T, m=best_m,
                       m_select_p=best_p, n_sparse_pairs=len(pm.J),
                       d_sim=round(float(d_sim.mean()), 3),
                       d_anc=round(float(d_anc.mean()), 3),
                       d_der=round(float(d_der.mean()), 3),
                       ks_in=dict(D=float(ks_in.statistic), p=float(ks_in.pvalue)),
                       ks_der=dict(D=float(ks_der.statistic), p=float(ks_der.pvalue)),
                       entropy_rho=rho,
                       ks_paircorr=dict(D=float(ks_pc.statistic), p=float(ks_pc.pvalue)),
                       pc_sim_mean=round(float(pc_sim.mean()), 4),
                       pc_der_mean=round(float(pc_der.mean()), 4),
                       verdict=verdict, seed=SEED,
                       wall_s=round(time.time() - t0, 1)), f, indent=1, ensure_ascii=False)
    print(f"[6] 判决 {verdict}: 盲测 KS p={ks_der.pvalue:.3g}, rho={rho:+.3f}, "
          f"二阶配对相关 KS p={ks_pc.pvalue:.3g} → {OUT}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(main(""))
    sys.exit(main(a[0], a[1] if len(a) > 1 else None))
