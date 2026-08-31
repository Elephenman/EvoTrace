#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B5 — PprI wave-2 回溯：PprI 战役四教训（M1 稳定性修正 / M2 锚位 / M3 稳定性入
适应度 / M4 赢家回流）在真实蛋白上的联合实战。

输入（全部来自 ppri_evo 已有资产）：
  results/priors.csv          — 8SLN 几何化学先验（53 位点）
  results/foldx/scan/s*/      — 17 位点单突变 FoldX ΔΔG 实测
  engine/evolve_w1.py WINNERS — 确认赢家（e18 63.4 / e3 55 / w4 53.3 / e13 35×0.3）
机制：
  M4 回流: prior' ∝ prior^(1-α) × post^α（winner 伪计数，α=0.35, k0=2）
  M1 修正: prior'' ∝ prior' × exp(−ΔΔG/τ), τ=2；17 实测位点用实测值，
           未测位点类别外推：base_edge 且 WT 芳香/疏水 → 极性替换默认 +2
  M2 锚位: 88/135/171 提议限制在 {F,Y,W,M,I,L,V}
  M3 适应度: f = Σs + w_stab·(−ΣΔΔG/τ), w_stab=0.4（对应 envspec fold_stability 权重）
输出：results/b5_wave2_elites.csv（含 ddG 预测与 gate 判定）+ wave-1 对照红旗统计
      + top-3 Boltz 候选 yaml（results/boltz_yamls_wave2/）
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, "A:/claudework/ppri_evo")

from engine.seqtools import AA20, AA2IDX, read_fasta, apply_mutant, diff_seqs
from engine.kernel import WFKernel
from engine.reflux import reflux_prior

PPRI = "A:/claudework/ppri_evo"
PDB_OFFSET = 22
TAU = 2.0
W_STAB = 0.4
ALPHA = 0.35
ANCHOR_SITES = {88, 135, 171}  # PDB 编号
ANCHOR_FAMILY = set("FYWMILV")
AROM_HYDRO = set("FYWMILV")
POLAR = set("NQSTHDEKR")
# wave-1 确认赢家（name, muts PDB 编号, sep, weight）
WINNERS = [
    ("e18", "22R;82W;84R;88Y;128N;131N;164K;185E;186N;255H;256Q;265S", 63.4, 1.0),
    ("e3", "22K;84R;88Q;128S;131Q;135H;165S;171N;208S;210T;256N;265N", 55.0, 1.0),
    ("w4", "22R;82W;84R;88Y;128N;131N;164K;185E;186N;255H;256Q;265S", 53.3, 1.0),
    ("e13", "22R;82F;84K;88N;128S;135Q;165K;174N;208S;252L;256T;265N", 35.0, 0.3),
]
# 17 位点 FoldX 实测 ΔΔG（results/foldx/scan 解析结果，脚本内置快照）
FOLDX_MEASURED = {
    "128S": 2.34433, "131Q": 0.587358, "135H": 8.54886, "164K": 0.900931,
    "165S": 0.664391, "171N": 1.11591, "185E": 2.78078, "186N": 1.76292,
    "208S": 0.18865, "210T": 1.02896, "22K": 1.32604, "255H": 1.89662,
    "256N": 0.720514, "265N": -0.288119, "82W": 2.04042, "84R": 3.74251,
    "88Q": 2.66697,
}


def parse_muts_pdb(s):
    out = []
    for tok in s.split(";"):
        if tok:
            out.append((int("".join(c for c in tok if c.isdigit())), tok[-1]))
    return out


def load_ppri_priors():
    pri, cls, wt_aa = {}, {}, {}
    for r in csv_rows(os.path.join(PPRI, "results", "priors.csv")):
        j = int(r["seq_idx"])
        pri.setdefault(j, {})[r["aa"]] = max(float(r["prior"]), 1e-3)
        cls[j] = r["dominant_class"]
        wt_aa[j] = r["wt_aa"]
    return pri, cls, wt_aa


def csv_rows(path):
    import csv as _csv
    return list(_csv.DictReader(open(path)))


def build_ddg_tables(wt_seq, pri_sites, cls, wt_aa):
    """M1: ddG[L,20] — 实测优先，类别外推填充；返回 (ddg, src)。

    外推规则（战役报告 §4）：base_edge 且 WT 芳香/疏水 → 极性替换默认 +2 kcal/mol。
    """
    L = len(pri_sites)
    sites_sorted = sorted(pri_sites)
    ddg = np.zeros((L, 20))
    src = [["ext0"] * 20 for _ in range(L)]
    for j, site in enumerate(sites_sorted):
        for aa in AA20:
            key = f"{site + PDB_OFFSET}{aa}"  # FOLDX 实测键为 PDB 编号
            if key in FOLDX_MEASURED:
                ddg[j, AA2IDX[aa]] = FOLDX_MEASURED[key]
                src[j][AA2IDX[aa]] = "measured"
            elif (cls[site] == "base_edge" and wt_aa[site] in AROM_HYDRO
                  and aa in POLAR and aa != wt_aa[site]):
                ddg[j, AA2IDX[aa]] = 2.0
                src[j][AA2IDX[aa]] = "extrapolated"
    return ddg, src


def main():
    wt = list(read_fasta(os.path.join(PPRI, "inputs", "wt_254.fasta")).values())[0]
    pri, cls, wt_aa = load_ppri_priors()
    sites = sorted(pri.keys())
    site_pos = {s: j for j, s in enumerate(sites)}
    print(f"PprI: {len(sites)} mutable sites, WT {len(wt)} aa")

    # ---- M4 赢家回流（wave-1 机制）
    pri_table = np.array([[pri[s].get(a, 1e-3) for a in AA20] for s in sites])
    pri_table = pri_table / pri_table.sum(1, keepdims=True)
    wt_idx = np.array([AA2IDX[wt[s]] for s in sites])
    labeled_muts, labeled_y = [], []
    for name, muts_s, sep, w in WINNERS:
        for pdb, aa in parse_muts_pdb(muts_s):
            if pdb - PDB_OFFSET in site_pos:
                labeled_muts.append((site_pos[pdb - PDB_OFFSET], aa))
                labeled_y.append(sep * w)
    # winner 证据 → 回流（每个 (site,aa) 一次证据，权重 = sep*w / 中位）
    ev = {}
    for (j, aa), y in zip(labeled_muts, labeled_y):
        ev.setdefault((j, aa), []).append(y)
    ys_all = np.array([np.mean(v) for v in ev.values()])
    y_ref, scale = float(np.median(ys_all)), float(ys_all.std() + 1e-9)
    pri_table2 = pri_table.copy()
    for (j, aa), v in ev.items():
        wgt = 1.0 / (1.0 + np.exp(-(np.mean(v) - y_ref) / scale))
        # 累积到回流输入格式
    muts_list = [[k] for k in ev.keys()]
    muts_y = [np.mean(v) for v in ev.values()]
    pri_table3 = reflux_prior(pri_table, wt_idx, muts_list, muts_y,
                              alpha=ALPHA, k0=2.0, y_ref=y_ref, scale=scale)
    print(f"M4 reflux: {len(ev)} winner (site,aa) evidence pairs")

    # ---- M1 稳定性修正
    ddg, src = build_ddg_tables(wt, sites, cls, wt_aa)
    pri_table4 = pri_table3 * np.exp(-ddg / TAU)
    pri_table4 /= pri_table4.sum(1, keepdims=True)
    print(f"M1 stability correction: tau={TAU}, measured 17 sites + extrapolated defaults")

    # ---- M2 锚位约束
    anchors = {}
    for s in ANCHOR_SITES:
        j = s - PDB_OFFSET  # 锚位 key 用 seq_idx（与 kernel.sites 同一口径）
        if j in site_pos:
            anchors[j] = sorted(ANCHOR_FAMILY)
    print(f"M2 anchors: {sorted(ANCHOR_SITES)} -> family {sorted(ANCHOR_FAMILY)}")

    # ---- 内核（自由模式：先验定义突变空间）
    cfg = dict(mutations_per_genome_per_gen={"lambda": 0.6}, n_mut_max=12, T=0.6,
               proposal_temp=2.0, w_stab=W_STAB, tau_stab=TAU)
    kernel = WFKernel(wt, sites, pri, cfg, seed=20260830, anchor_sites=anchors,
                      proposal="prior")
    kernel.set_prior_table(pri_table4)
    kernel.set_ddg(ddg, tau=TAU, w_stab=W_STAB)
    stats, pops = kernel.run(n_pop=4, n_gen=15, Ne=500, record_events=True)
    print(f"kernel: last best={stats[-1]['best']:.2f} mean={stats[-1]['mean']:.2f} "
          f"unique={stats[-1]['unique']}")

    # ---- 精英 + M3 稳定性 gate
    cands = kernel.propose_elites(pops, top_k=24, diversity=6)
    rows = []
    for fit, g in cands:
        muts = kernel.geno_to_muts(g)
        ddg_pred = float(sum(ddg[site_pos[i], AA2IDX[a]] for i, a in muts))
        anchor_viol = [(i, a) for i, a in muts
                       if i in ANCHOR_SITES and a not in ANCHOR_FAMILY]
        cat_frozen = [i for i, a in muts if i in {92, 93, 94, 95, 96, 123}]
        rows.append(dict(
            fitness=round(float(fit), 3),
            muts=";".join(f"{i}{a}" for i, a in muts),
            n_mut=len(muts), ddg_pred=round(ddg_pred, 2),
            gate_pass=(ddg_pred <= 3.0) and not anchor_viol and not cat_frozen,
            anchor_violation=len(anchor_viol),
            fitness_stab=round(float(fit - W_STAB * ddg_pred / TAU), 3)))
    df = pd.DataFrame(rows).sort_values("fitness_stab", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    out = os.path.join(ROOT, "results", "b5_wave2_elites.csv")
    df.to_csv(out, index=False)
    print(f"elites: {len(df)} ({df.gate_pass.sum()} pass gate) -> {out}")

    # ---- wave-1 对照（无 M1/M2 时红旗率）
    kernel_w1 = WFKernel(wt, sites, pri, dict(cfg, w_stab=0.0), seed=20260830,
                         proposal="prior")
    pri_w1 = reflux_prior(pri_table, wt_idx, muts_list, muts_y,
                          alpha=ALPHA, k0=2.0, y_ref=y_ref, scale=scale)
    kernel_w1.set_prior_table(pri_w1)
    stats_w1, pops_w1 = kernel_w1.run(n_pop=4, n_gen=15, Ne=500)
    cands_w1 = kernel_w1.propose_elites(pops_w1, top_k=24, diversity=6)
    n_viol, n_ddg = 0, 0
    for fit, g in cands_w1:
        muts = kernel_w1.geno_to_muts(g)
        ddg_pred = float(sum(ddg[site_pos[i], AA2IDX[a]] for i, a in muts))
        if any(i in ANCHOR_SITES and a not in ANCHOR_FAMILY for i, a in muts):
            n_viol += 1
        if ddg_pred > 3.0:
            n_ddg += 1
    print(f"wave-1 对照（24 精英）: 锚位违规 {n_viol}, ddG>3 {n_ddg}")
    print(f"wave-2 (M1+M2): 锚位违规 0（构造保证）, ddG>3 已 gate 拦截")

    # ---- top-3 Boltz 候选
    env = json.load(open(os.path.join(PPRI, "config", "envspec.json")))
    conds = {"S1_G17": env["environment"]["target_dna"]["sequence"]}
    for c in env["environment"]["nontarget_panel"][:2]:
        conds[c["id"]] = c["sequence"]
    yd = os.path.join(ROOT, "results", "boltz_yamls_wave2")
    os.makedirs(yd, exist_ok=True)
    top3 = df[df.gate_pass].head(3)
    for _, r in top3.iterrows():
        muts = [(int("".join(c for c in tok if c.isdigit())), tok[-1])
                for tok in r.muts.split(";")]
        seq = apply_mutant(wt, muts)
        name = f"wave2_{r['rank']}"
        for cond, dna in conds.items():
            yaml = ("version: 1\nsequences:\n"
                    f"  - protein:\n      id: A\n      sequence: {seq}\n"
                    f"  - dna:\n      id: B\n      sequence: {dna}\n"
                    "  - ligand:\n      id: C\n      ccd: MN\n")
            with open(os.path.join(yd, f"{name}_{cond}.yaml"), "w") as f:
                f.write(yaml)
        print(f"candidate {name}: {r.muts} ddG_pred={r.ddg_pred}")
    json.dump(dict(stats_wave2=stats[-60:], stats_wave1=stats_w1[-60:],
                   n_anchor_viol_w1=n_viol, n_ddg_gt3_w1=n_ddg,
                   ddg_sources={"measured": sum(1 for r in src for c in r if c == 'measured'),
                                 "extrapolated": sum(1 for r in src for c in r if c == 'extrapolated')}),
              open(os.path.join(ROOT, "results", "b5_wave2_summary.json"), "w"),
              indent=1, default=str)
    print("B5 engine stage done")


if __name__ == "__main__":
    main()
