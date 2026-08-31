#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并 ESM3 零样本分数到各基准并计算对标指标。

输入：results/esm3/ 下的分片 CSV（seq_id, dataset, sum_logp, mean_logp）
输出：
  results/esm3_summary.csv — 各基准的 ESM3 Spearman 汇总
  合并逻辑：
   - avgfp: seq_id = avgfp_{行号} -> DMS 表 -> 面板（结构位点内）Spearman@5000 + top16
   - gb1:   seq_id = gb1_{Variants} -> two_vs_rest test 集 Spearman + top16
   - tem1:  Firnberg 训练域 Spearman
   - tem1_x: Deng/Stiffler/Jacquier 迁移域 Spearman
   - pgym:  30 评测集逐集 Spearman（与其他打分器同口径）
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine.seqtools import parse_mutant
from engine.metrics import spearman

PG = ("A:/claudework/evo_data/processed/proteingym_benchmark/"
      "DMS_ProteinGym_substitutions")
ESM3_DIR = os.path.join(ROOT, "results", "esm3")


def load_esm3(pattern):
    files = sorted(glob.glob(os.path.join(ESM3_DIR, pattern)))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.seq_id != "seq_id"]
    df["sum_logp"] = df.sum_logp.astype(float)
    return df.drop_duplicates("seq_id")


def main():
    out = {}
    # ---- avgfp
    esm3 = load_esm3("avgfp_*.csv")
    if esm3 is not None:
        df = pd.read_csv(os.path.join(
            PG, "GFP_AEQVI_Sarkisyan_2016.csv"))
        wt = json.load(open(os.path.join(ROOT, "benchmarks",
                                         "data_avgfp_wt.json")))["wt"]
        import csv as _csv
        pri_sites = set()
        for r in _csv.DictReader(open(os.path.join(
                ROOT, "benchmarks", "data", "priors_avgfp.csv"))):
            pri_sites.add(int(r["seq_idx"]))
        rows = []
        for i, (m, y, s) in enumerate(zip(df.mutant, df.DMS_score,
                                          df.mutated_sequence)):
            muts = parse_mutant(m)
            ok = bool(muts) and all(t in pri_sites for t, _ in muts)
            rows.append((f"avgfp_{i}", m, float(y), s, ok))
        d = pd.DataFrame(rows, columns=["seq_id", "mutant", "y", "seq", "ok"])
        d = d.merge(esm3[["seq_id", "sum_logp"]], on="seq_id", how="inner")
        panel = d[d.ok]
        rng = np.random.default_rng(0)
        sub = rng.choice(len(panel), size=min(5000, len(panel)), replace=False)
        out["avgfp_rho5k"] = round(spearman(panel.sum_logp.values[sub],
                                            panel.y.values[sub]), 4)
        top16 = panel.nlargest(16, "sum_logp")
        out["avgfp_top16"] = round(float(top16.y.mean()), 3)
        out["avgfp_n_scored"] = int(len(panel))
    # ---- gb1
    esm3 = load_esm3("gb1_*.csv")
    if esm3 is not None:
        df = pd.read_csv("A:/claudework/evo_data/raw/flip/splits/gb1/"
                         "four_mutations_full_data.csv", low_memory=False)
        test = df[df.two_vs_rest == "test"].copy()
        test["seq_id"] = "gb1_" + test.Variants
        m = test.merge(esm3[["seq_id", "sum_logp"]], on="seq_id", how="inner")
        out["gb1_rho_test"] = round(spearman(m.sum_logp.values, m.Fitness.values), 4)
        out["gb1_top16"] = round(float(m.nlargest(16, "sum_logp").Fitness.mean()), 3)
        out["gb1_n"] = int(len(m))
    # ---- tem1 (Firnberg)
    esm3 = load_esm3("tem1_*.csv")
    if esm3 is not None:
        df = pd.read_csv(os.path.join(PG, "BLAT_ECOLX_Firnberg_2014.csv"))
        ids = [f"tem1_{i}" for i in range(len(df))]
        d = pd.DataFrame({"seq_id": ids, "y": df.DMS_score})
        d = d.merge(esm3[["seq_id", "sum_logp"]], on="seq_id", how="inner")
        out["tem1_firnberg_rho"] = round(spearman(d.sum_logp.values, d.y.values), 4)
    # ---- tem1_x transfer domains
    esm3 = load_esm3("tem1x_*.csv")
    if esm3 is not None and len(esm3):
        for fn, tag in [("BLAT_ECOLX_Deng_2012.csv", "deng"),
                        ("BLAT_ECOLX_Stiffler_2015.csv", "stiffler"),
                        ("BLAT_ECOLX_Jacquier_2013.csv", "jacquier")]:
            df = pd.read_csv(os.path.join(PG, fn))
            ids = [f"{fn[:-4]}_{i}" for i in range(len(df))]
            d = pd.DataFrame({"seq_id": ids, "y": df.DMS_score})
            d = d.merge(esm3[["seq_id", "sum_logp"]], on="seq_id", how="inner")
            out[f"tem1_{tag}_rho"] = round(spearman(d.sum_logp.values,
                                                    d.y.values), 4)
    # ---- pgym 30
    esm3 = load_esm3("pgym_*.csv")
    if esm3 is not None and len(esm3):
        import random
        all_files = sorted(os.listdir(PG))
        eval_ds = sorted(random.Random(42).sample(all_files, 30))
        rows = []
        for f in eval_ds:
            ds = f[:-4]
            sub = esm3[esm3.dataset == ds]
            if len(sub) < 50:
                continue
            df = pd.read_csv(os.path.join(PG, f))
            if len(df) > 1000:
                df = df.sample(1000, random_state=0)  # 与 build_payload 同一抽样
            ids = [f"{ds}_{i}" for i in range(len(df))]
            d = pd.DataFrame({"seq_id": ids, "y": df.DMS_score.values})
            d = d.merge(sub[["seq_id", "sum_logp"]], on="seq_id", how="inner")
            rows.append(dict(dataset=ds, n=len(d),
                             rho_esm3=round(spearman(d.sum_logp.values,
                                                     d.y.values), 4)))
        rd = pd.DataFrame(rows)
        rd.to_csv(os.path.join(ROOT, "results", "b4_esm3_pgym.csv"), index=False)
        out["pgym30_esm3_mean_rho"] = round(float(rd.rho_esm3.mean()), 4)
        out["pgym30_esm3_median_rho"] = round(float(rd.rho_esm3.median()), 4)
        out["pgym30_n"] = int(len(rd))
    pd.DataFrame([out]).T.rename(columns={0: "value"}).to_csv(
        os.path.join(ROOT, "results", "esm3_summary.csv"))
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
