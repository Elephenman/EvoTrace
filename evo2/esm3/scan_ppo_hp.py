# -*- coding: utf-8 -*-
"""PPO 超参扫描（lr × ent）：additive（数学）+ surrogate v1 景观，找最优超参。

用法: python scan_ppo_hp.py [--seeds 2] [--budget 3000]
输出: ppo_hp_scan.csv（每配置 norm 表）+ 最佳配置
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, "A:/EvoTrace-cmp/evo2")
sys.path.insert(0, "A:/EvoTrace-cmp/evo2/benchmarks")

from engine.oracle import build_ppri_additive, AdditiveOracle
from engine.surrogate_oracle import build_ppri_surrogate
from engine.rlpolicy_torch import PPOOptimizer
import b7_three_way as b7

OUT = "A:/claudework/out/b7_full/ppo_hp_scan.csv"


def build_additive():
    orc0, *_ = build_ppri_additive()
    return orc0


def build_surr_v1():
    return build_ppri_surrogate()


GRID = [
    {"lr": 1e-4, "ent": 0.01}, {"lr": 3e-4, "ent": 0.01}, {"lr": 1e-3, "ent": 0.01},
    {"lr": 1e-4, "ent": 0.05}, {"lr": 3e-4, "ent": 0.05}, {"lr": 1e-3, "ent": 0.05},
]
LANDS = {"additive": build_additive, "surr_v1": build_surr_v1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--rollout", type=int, default=1024)
    args = ap.parse_args()

    rows = []
    t0 = time.time()
    for lname, bf in LANDS.items():
        ref = b7.greedy_reference(bf())
        print(f"\n=== {lname}: wt_f={ref['wt_f']:.3f} ref_f={ref['ref_f']:.3f} ===", flush=True)
        for cfg in GRID:
            fs = []
            for seed in range(args.seeds):
                orc = bf()
                opt = PPOOptimizer(orc, seed=seed, budget=args.budget,
                                   cfg={**cfg, "rollout": args.rollout})
                res = opt.run()
                norm = (res["best_f"] - ref["wt_f"]) / max(ref["ref_f"] - ref["wt_f"], 1e-9)
                fs.append(norm)
                print(f"  {lname} lr={cfg['lr']:.0e} ent={cfg['ent']} seed={seed}: "
                      f"best={res['best_f']:.3f} norm={norm:.3f} ({time.time()-t0:.0f}s)", flush=True)
            rows.append(dict(landscape=lname, lr=cfg["lr"], ent=cfg["ent"],
                             norm_mean=round(float(np.mean(fs)), 4),
                             norm_std=round(float(np.std(fs)), 4)))
            print(f"  -> {lname} lr={cfg['lr']:.0e} ent={cfg['ent']}: mean={np.mean(fs):.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("\n=== PPO 超参扫描结果 ===")
    print(df.to_string(index=False))
    # 最佳配置（两景观平均 norm 最高）
    agg = df.groupby(["lr", "ent"])["norm_mean"].mean().reset_index()
    best = agg.sort_values("norm_mean", ascending=False).iloc[0]
    print(f"\nBEST cfg: lr={best['lr']:.0e} ent={best['ent']} (mean norm={best['norm_mean']:.3f})")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
