#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B6 — EvoPrior 位点偏好预测器演示（规则驱动 vs 数据驱动对比）。

输入（均为本地路径，数据不入库）
------------------------------
  --seq        WT 序列 fasta（如 pprI_evo/inputs/wt_254.fasta）
  --msa        同源 MSA（a3m，如 pprI 的 uniref.a3m / bfd.mgnify30...a3m）
  --pdb        结构 PDB（生成化学先验 p_chem；如 8SLN.pdb）
  --priors_csv 已有 p_chem（priors.csv；与 --pdb 二选一）

可选
----
  --use_plm        用真实 ESM-2 嵌入（需 transformers+torch）；默认 placeholder
  --plm_model      facebook/esm2_t33_650M_UR50D
  --embed_cache    嵌入缓存 .npy（避免重复推理）
  --alpha          固定融合权重；默认做交叉验证学习
  --per_site       逐位点自适应 α（EVOPRIOR §6③）：每位点独立学 α_j，
                   盲点位（化学先验被手工规则盲化）α_j→0 更信 p_evo，
                   化学先验可靠的位点 α_j→1；避免全局单一 α 的妥协。
  --seq_offset     PDB 编号 = seq_idx + offset（PprI 8SLN 通常为 0 或 1）
  --chain         蛋白链（默认 A）
  --out           输出目录（默认 evo2/results/evoprior）

输出
----
  evoprior_priors.csv   每位点 20 维 p_chem / p_evo / p_final
  evoprior_report.txt   M135 式盲点对比 + "规则驱动 vs 数据驱动" 结论

运行
----
  python benchmarks/b6_evoprior_demo.py \
      --seq pprI_evo/inputs/wt_254.fasta \
      --msa pprI_evo/.../uniref.a3m \
      --pdb pprI_evo/inputs/8SLN.pdb --seq_offset 0
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))  # evo2/

from engine.seqtools import read_fasta, AA20  # noqa: E402
from engine.priors import build_priors, load_priors_csv  # noqa: E402
from engine.evoprior import (  # noqa: E402
    build_evoprior, fuse_tables, learn_alpha_cv,
)


def priors_to_table(priors_dict, L):
    """{seq_idx:{aa:p}} -> [L,20]（行归一，缺失位填均匀）。"""
    tab = np.ones((L, 20)) / 20.0
    for j, aamap in priors_dict.items():
        if 0 <= int(j) < L:
            row = np.array([aamap.get(a, 1e-3) for a in AA20], float)
            row = np.clip(row, 1e-3, None)
            tab[int(j)] = row / row.sum()
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--msa", required=True)
    ap.add_argument("--pdb", default=None)
    ap.add_argument("--priors_csv", default=None)
    ap.add_argument("--use_plm", action="store_true")
    ap.add_argument("--plm_model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--embed_cache", default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--per_site", action="store_true")
    ap.add_argument("--seq_offset", type=int, default=0)
    ap.add_argument("--chain", default="A")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seqs = read_fasta(args.seq)
    seq = next(iter(seqs.values()))
    L = len(seq)

    # ---- p_chem ----
    if args.priors_csv:
        pdict, _, _ = load_priors_csv(args.priors_csv)
        p_chem = priors_to_table(pdict, L)
        src = f"priors_csv:{args.priors_csv}"
    elif args.pdb:
        rows, _ = build_priors(
            args.pdb, seq, args.seq_offset, protein_chain=args.chain,
            ligand_resnames=("DA", "DT", "DG", "DC"),
        )
        pdict = {r["seq_idx"]: {r["aa"]: r["prior"] for r in rows
                                 if r["seq_idx"] == r["seq_idx"]}
                 for r in []}  # 占位，下面重排
        pdict = {}
        for r in rows:
            pdict.setdefault(r["seq_idx"], {})[r["aa"]] = r["prior"]
        p_chem = priors_to_table(pdict, L)
        src = f"pdb:{args.pdb}"
    else:
        raise SystemExit("必须提供 --pdb 或 --priors_csv 以生成化学先验 p_chem")

    # ---- p_evo ----
    print(f"[B6] 训练 EvoPrior（use_plm={args.use_plm}）...")
    evo = build_evoprior(
        seq, args.msa, use_plm=args.use_plm, plm_model=args.plm_model,
        embed_cache=args.embed_cache, verbose=True,
    )
    p_evo = evo["p_evo"]
    pssm = evo["pssm"]

    # ---- α ----
    if args.alpha is not None:
        alpha = args.alpha
        curve = None
        per_site = False
    elif args.per_site:
        alpha, curve = learn_alpha_cv(p_chem, p_evo, pssm, folds=5, per_site=True)
        per_site = True
    else:
        alpha, curve = learn_alpha_cv(p_chem, p_evo, pssm, folds=5)
        per_site = False
    p_final = fuse_tables(p_chem, p_evo, alpha)

    # ---- 输出 ----
    out = args.out or os.path.join(HERE, "..", "results", "evoprior")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "evoprior_priors.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_idx", "wt_aa"] + [f"chem_{a}" for a in AA20]
                   + [f"evo_{a}" for a in AA20] + [f"final_{a}" for a in AA20])
        for j in range(L):
            w.writerow([j, seq[j]] + list(p_chem[j].round(4)) + list(p_evo[j].round(4))
                       + list(p_final[j].round(4)))

    # ---- 报告（M135 式盲点 + 规则 vs 数据）----
    rep = os.path.join(out, "evoprior_report.txt")
    with open(rep, "w") as f:
        f.write("EvoPrior 报告（规则驱动 vs 数据驱动）\n")
        f.write(f"seq_len={L}  p_chem 来源={src}  use_plm={args.use_plm}\n")
        if per_site:
            f.write(f"融合 α=逐位点自适应 [L] 数组（min={alpha.min():.3f} "
                    f"max={alpha.max():.3f} mean={alpha.mean():.3f}）\n")
            # 导出 α 数组供 kernel 直接复用
            np.savetxt(os.path.join(out, "evoprior_alpha_per_site.csv"),
                       alpha.reshape(-1, 1), delimiter=",",
                       header="alpha_per_site", comments="")
        else:
            f.write(f"融合 α={alpha:.3f}\n")
        # 盲点诊断：化学先验与进化先验冲突最大的位点
        div = (p_chem * np.log((p_chem + 1e-8) / (p_evo + 1e-8))).sum(1)
        worst = int(np.argmax(div))
        f.write(f"最大分歧位点 #{worst} (WT={seq[worst]}):\n")
        top_chem = sorted(range(20), key=lambda k: -p_chem[worst, k])[:3]
        top_evo = sorted(range(20), key=lambda k: -p_evo[worst, k])[:3]
        f.write("  化学先验 top3: " + ", ".join(f"{AA20[k]}={p_chem[worst,k]:.2f}" for k in top_chem) + "\n")
        f.write("  进化先验 top3: " + ", ".join(f"{AA20[k]}={p_evo[worst,k]:.2f}" for k in top_evo) + "\n")
        f.write(f"  -> 若该位化学先验把高频进化残基压低，即'手工规则盲点'，"
                f"融合后已被数据驱动修正。\n")
        if curve:
            f.write("\nα 交叉验证曲线 (alpha, mean_KL):\n")
            for a, k in curve[::2]:
                f.write(f"  α={a:.2f}  KL={k:.4f}\n")
    print(f"[B6] 完成 → {csv_path}\n      {rep}")


if __name__ == "__main__":
    main()
