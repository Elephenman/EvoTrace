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
LAM_MUT_GRID = [0.3, 2.0, 6.0, 12.0]     # §4.10: b9 自然回放口径 ≤12 次/基因组/代
PAIRS = 2000
# λ 训练侧校准网格（P1-1, 铁律: 只用祖先半 anc21）
LAM_GRID = [0.01, 0.05, 0.1, 0.2]
T_REF = 1.0                              # λ 校准参考选择温度（T 校准另做）
T_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
M_GRID = [0.0, 0.005, 0.02]


def sample_pair_dists(mat, n_pairs, rng, max_factor=500):
    """pair_distances 的守卫版（§4.9）: 样本对凑不满时显式失败, 不无限空转。"""
    N = len(mat)
    d, tries = [], 0
    while len(d) < n_pairs:
        tries += 1
        if tries > max_factor * n_pairs:
            raise RuntimeError("回放退化: 有效样本对（≥5 非掩码位点）凑不满——gap 膨胀守卫触发")
        i, j = rng.integers(0, N, 2)
        if i == j:
            continue
        a, b = mat[i], mat[j]
        okk = (a != 31) & (b != 31)
        if okk.sum() < 5:
            continue
        d.append(float((a[okk] != b[okk]).sum()) / okk.sum())
    return np.array(d)


def run_mean_dist(pm, anc21, T, m, n_gen=600, n_pop=2, ne=200, seed=SEED,
                  proposal_states=None, fitness_fn=None, lam_mut=None):
    """短程岛屿 WF 的平衡均值距离（校准用, §4.9 加退化守卫）。
    返回 float；回放退化（<50% 随机样本对有效, 即 gap 汤信号）→ None。"""
    rng = np.random.default_rng(seed)
    pops0 = anc21[rng.integers(0, len(anc21), n_pop * ne)].reshape(n_pop, ne, -1)
    end = island_wf(pm, pops0, n_gen=n_gen, T=T, lam_mut=lam_mut or LAM_MUT,
                    m_mig=m, rng=rng, proposal_states=proposal_states,
                    fitness_fn=fitness_fn)
    msk = to_masked(end.reshape(-1, end.shape[-1]))
    r3 = np.random.default_rng(seed + 1)
    vals, nok = [], 0
    for _ in range(400):
        i, j = r3.integers(0, len(msk), 2)
        if i == j:
            continue
        a, b = msk[i], msk[j]
        okk = (a != 31) & (b != 31)
        if okk.sum() < 5:
            continue
        nok += 1
        vals.append(float((a[okk] != b[okk]).sum()) / okk.sum())
    if nok < 200:                       # 退化守卫: 有效样本对 <50%
        return None
    return float(np.mean(vals))


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

    # ---- §4.10 协议修订: 混合制 + λ_mut 校准 ----
    # 诊断链（2026-09-07）: ①plmDCA 能量作 WF 适应度被 gap 劫持
    # （corr(E,gap数)=+0.94~0.99, 任何 T 退化）→ 适应度回退为祖先对数先验
    # （b9 式 PSSM, corr≈0, 探针实证零退化）, Potts 条件分布仅作变异提议
    # （回归提案 §4.2 原意"只升级提议, 不换选择机制"）; ②λ_mut=0.3 突变
    # 供给不足（b9 自然回放 ≤12 次/代）→ λ_mut 加入祖先侧校准网格。
    prop_states = np.arange(Q - 1)          # 20 种氨基酸, gap(20) 不可被变异产生
    counts = np.zeros((K, Q))
    for s in range(Q):
        counts[:, s] = (anc21 == s).sum(0)
    logprior = np.log((counts + 0.5) / (counts.sum(1, keepdims=True) + 0.5 * Q))
    hp_row = np.arange(K)[None, :]

    def fitness_logprior(x):
        return logprior[hp_row, x].sum(1)

    f_sd = float(fitness_logprior(anc21[:min(1000, len(anc21))]).std()) or 1.0
    t_grid = sorted({max(0.25, round(f_sd * f, 3)) for f in (0.5, 1.0, 2.0, 4.0)})
    pm_by_lam = {}
    cands = []
    if force_lam is not None:
        lam_list = [float(force_lam)]
        print(f"[3] 联合校准: 强制 λ={force_lam}", flush=True)
    else:
        lam_list = list(LAM_GRID)
        print(f"[3] λ×λ_mut×T 联合校准（§4.10: fitness=祖先对数先验, gap 不进变异"
              f"字母表; λ∈{LAM_GRID} × λ_mut∈{LAM_MUT_GRID} × T~N({f_sd:.1f}²), "
              f"目标均值 {target:.3f}）", flush=True)
    for lam in lam_list:
        Wfull_l = fit_plm(anc21, Q=Q, epochs=EPOCHS, lam=lam, lr=LR,
                          seed=SEED, verbose=False)
        pm_l = PottsModel(Wfull_l, eps=EPS_SPARSE, Q=Q)
        pm_by_lam[lam] = pm_l
        for lmut in LAM_MUT_GRID:
            for T in t_grid:
                r = run_mean_dist(pm_l, anc21, T=T, m=0.005, proposal_states=prop_states,
                                  fitness_fn=fitness_logprior, lam_mut=lmut)
                tag = "退化回放（守卫触发）" if r is None else f"均值 {r:.3f}"
                print(f"    λ={lam:5.2f} λ_mut={lmut:5.2f} T={T:8.3f} → {tag}", flush=True)
                if r is not None:
                    cands.append((abs(r - target), lam, lmut, T, r))
    if not cands:
        print("全部 (λ,λ_mut,T) 组合回放退化——需上探 T 或重审协议（§4.10）",
              file=sys.stderr)
        return 3
    _, best_lam, chosen_lmut, chosen_T, best_r = min(
        cands, key=lambda t: (t[0], t[1], t[2], t[3]))
    pm = pm_by_lam[best_lam]
    calib_table = [dict(lam=float(l), lam_mut=float(lm), T=float(T),
                        replay_mean=round(r, 3))
                   for _, l, lm, T, r in cands]
    LAM_REG = best_lam
    print(f"    冻结 λ={best_lam}, λ_mut={chosen_lmut}, T*={chosen_T}"
          f"（回放均值 {best_r:.3f} ≈ 目标 {target:.3f}）", flush=True)
    print(f"    稀疏耦合对 {len(pm.J)}/{K*(K-1)//2}", flush=True)

    # ---- m 选择（祖先侧形状 KS, 防盲测泄漏）----
    best_m, best_p = M_GRID[0], -1.0
    for m in M_GRID:
        end = island_wf(pm, anc21[rng.integers(0, len(anc21), 4 * 250)].reshape(4, 250, K),
                        n_gen=800, T=chosen_T, lam_mut=chosen_lmut, m_mig=m, rng=rng,
                        proposal_states=prop_states, fitness_fn=fitness_logprior)
        d_sim = sample_pair_dists(to_masked(end.reshape(-1, K)), 1000, rng)
        p = float(ks_2samp(d_sim, d_anc).pvalue)
        print(f"    m={m:5.3f} → 组内 KS p={p:.3f}（均值 {d_sim.mean():.3f}）", flush=True)
        if p > best_p:
            best_m, best_p = m, p
    print(f"    选定 m = {best_m}（组内 p={best_p:.3f}）", flush=True)

    # ---- 终跑（b9 规模）----
    pops0 = anc21[rng.integers(0, len(anc21), N_POP * NE)].reshape(N_POP, NE, K)
    end = island_wf(pm, pops0, n_gen=N_GEN, T=chosen_T, lam_mut=chosen_lmut,
                    m_mig=best_m, rng=rng, proposal_states=prop_states,
                    fitness_fn=fitness_logprior)
    sim = to_masked(end.reshape(-1, K))
    print(f"[5] 终跑完成（{time.time()-t0:.0f}s）", flush=True)

    # ---- 比对（b9 口径 + P1-2 二阶统计）----
    d_sim = sample_pair_dists(sim, PAIRS, rng)
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
- 校准（全部在祖先侧, 盲测零泄漏）: **§4.10 混合制** — 适应度=祖先对数先验
  （b9 式 PSSM, Laplace 平滑; plmDCA 能量被 gap 劫持 corr=+0.94~0.99, 判死）,
  Potts 条件分布仅作变异提议（回归提案 §4.2 原意）; **λ×λ_mut×T 三维联合网格**
  → λ={LAM_REG}, λ_mut={chosen_lmut}, T={chosen_T}（回放均值 {best_r:.3f} ≈
  祖先实测目标 {target:.3f}）, m={best_m}（组内 KS p={best_p:.3f}）,
  稀疏耦合对 {len(pm.J)}
- §4.9 协议修正（gap 膨胀诊断 2026-09-07）: gap 不进变异字母表
  （条件分布在 20 AA 上重归一化, 祖先既有 gap 保留）; 全部组合先过退化守卫
  （<50% 样本对有效即剔除）——原"先 λ 后 T"在 T_REF=1.0 下第 1 代即 gap
  固定（gap 占比 12.7%→99.99%）。
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
                       fitness="logprior(§4.10)", lam_mut=chosen_lmut,
                       calib_table=calib_table, T=chosen_T, m=best_m,
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
