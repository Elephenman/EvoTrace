#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B4 — ProteinGym 廉价层地板校准（本地 CPU，无 PLM）。

评测集：pgym_sub（30 个分层抽样的 DMS 数据集，与集群 ESM3 打分同一批——可直接对比）。
三个打分器（全部加性、位点无关、单次前向零成本）：
  S1 blosum62    — BLOSUM62 加性（经典进化先验，零训练）
  S2 dms_matrix  — DMS 校准替换矩阵：从另外 10 个不相交 DMS 数据集用 ridge 拟合
                   20×20 全局替换分数（wt_aa → mut_aa），零样本迁移到评测集
  S3 esm3        — ESM3 open 1.4B 零样本（集群打分，另行合并：merge_esm3.py）
输出：results/b4_pgym_floor.csv（逐数据集 ρ）+ 汇总统计。
对标：ProteinGym 官方榜单参考带（results 对比在图表层完成）。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine.seqtools import AA20, AA2IDX, parse_mutant
from engine.metrics import spearman

PG = ("A:/claudework/evo_data/processed/proteingym_benchmark/"
      "DMS_ProteinGym_substitutions")
BLOSUM_JSON = "A:/Data/ensemble-protein-scorer/tools/blosum62.json"
N_TRAIN = 10


def load_dataset(fn):
    """返回 [(subs list, y)]，subs = [(wt_aa, mut_aa)]（从突变串直接取 WT 残基）。"""
    df = pd.read_csv(os.path.join(PG, fn))
    out = []
    for m, y in zip(df.mutant, df.DMS_score):
        subs = []
        for tok in str(m).split(":"):
            if len(tok) >= 4:
                subs.append((tok[0], tok[-1]))
        if subs:
            out.append((subs, float(y)))
    return out


def additive_score(subs, table):
    # 方向统一 (wt -> mut)；缺失键（如 BLOSUM 对角/未覆盖）回退 -4
    return sum(table.get((w, a), -4.0) for w, a in subs)


def fit_dms_matrix(train_sets):
    """全局 20×20 替换分数（+截距）ridge。特征 = onehot(wt,mut)。"""
    D = 400
    X, y = [], []
    for rows in train_sets:
        for subs, yy in rows:
            v = np.zeros(D + 1)
            for w, a in subs:
                if w in AA2IDX and a in AA2IDX:
                    v[AA2IDX[w] * 20 + AA2IDX[a]] += 1.0
            v[-1] = len(subs)
            X.append(v)
            y.append(yy)
    X = np.array(X)
    y = np.array(y)
    lam = 10.0 * max(len(y), 1) / 1000.0
    A = X.T @ X + lam * np.eye(D + 1)
    beta = np.linalg.solve(A, X.T @ (y - y.mean()))
    assert np.all(np.isfinite(beta)), "dms matrix fit not finite"
    table = {}
    for w in AA20:
        for a in AA20:
            table[(w, a)] = float(beta[AA2IDX[w] * 20 + AA2IDX[a]])
    return table, float(y.mean()), float(beta[-1])


def main():
    eval_files = sorted(os.listdir(os.path.join(ROOT, "cluster", "payload",
                                                "pgym_sub")))
    eval_ds = [ln.strip() for ln in open(os.path.join(
        ROOT, "cluster", "payload", "pgym_sub", "eval_datasets.txt"))] \
        if os.path.exists(os.path.join(ROOT, "cluster", "payload", "pgym_sub",
                                       "eval_datasets.txt")) else None
    # pgym_sub 的数据集清单 = build_payload 里 random.seed(42) 抽的 30 个
    all_files = sorted(os.listdir(PG))
    import random
    rng0 = random.Random(42)
    eval_ds = sorted(rng0.sample(all_files, 30))
    print(f"eval datasets: {len(eval_ds)}")
    # 训练集：不相交的 10 个（数据量适中的）
    rest = [f for f in all_files if f not in eval_ds]
    sizes = []
    for f in rest:
        n = sum(1 for _ in open(os.path.join(PG, f))) - 1
        sizes.append((n, f))
    sizes.sort()
    train_files = [f for n, f in sizes[len(sizes) // 3: 2 * len(sizes) // 3]]
    rng1 = random.Random(7)
    train_files = rng1.sample(train_files, N_TRAIN)
    print("train files:", train_files)

    train_sets = [load_dataset(f) for f in train_files]
    dms_table, dms_mu, dms_count_beta = fit_dms_matrix(train_sets)
    print("dms matrix fitted on", sum(len(s) for s in train_sets), "variants")
    blosum = json.load(open(BLOSUM_JSON))["matrix"]
    b62 = {}
    for k, v in blosum.items():
        b62[(k[0], k[1])] = float(v)

    rows = []
    for f in eval_ds:
        data = load_dataset(f)
        if len(data) < 50:
            continue
        ys = [y for _, y in data]
        s_b62 = [additive_score(m, b62) for m, _ in data]
        s_dms = [additive_score(m, dms_table) + dms_count_beta * len(m)
                 for m, _ in data]
        rows.append(dict(dataset=f[:-4], n=len(data),
                         rho_blosum=round(spearman(s_b62, ys), 4),
                         rho_dms_matrix=round(spearman(s_dms, ys), 4)))
        print(f"  {f[:-4]}: blosum={rows[-1]['rho_blosum']:.3f} "
              f"dms={rows[-1]['rho_dms_matrix']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "b4_pgym_floor.csv"), index=False)
    print("\nsummary:")
    print(df[["rho_blosum", "rho_dms_matrix"]].mean().round(4).to_string())
    print("median:", df[["rho_blosum", "rho_dms_matrix"]].median().round(4).to_dict())


if __name__ == "__main__":
    main()
