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
    ap.add_argument("--embed_npz", default=None,
                    help="集群预计算 PLM 嵌入 npz（cluster/esm3_embed_cluster.py 产物）")
    ap.add_argument("--embed_key", default=None, help="npz 内的序列 key（单条目时可省）")
    ap.add_argument("--pca_dim", type=int, default=32,
                    help="PCA 降维维度（ESM3 为 1536 维，必须降维）；0=不降维")
    ap.add_argument("--no_cv", action="store_true",
                    help="关闭留出位点交叉验证（关闭后 p_evo 来自全量拟合，有过拟合风险）")
    ap.add_argument("--p_evo_source", default="auto",
                    choices=("auto", "pssm", "mlp"),
                    help="p_evo 来源：auto=PSSM 为主+MLP 补低覆盖列（默认）；"
                         "pssm=纯 PSSM；mlp=纯 MLP 留出预测")
    ap.add_argument("--min_neff", type=float, default=30.0,
                    help="auto 模式下判定 MSA 覆盖不足的列深度阈值")
    ap.add_argument("--w_decay", type=float, default=1e-2,
                    help="MLP 权重衰减（单蛋白小样本下影响显著）")
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
    print(f"[B6] 训练 EvoPrior（embed_npz={args.embed_npz or '无'}，"
          f"use_plm={args.use_plm}）...")
    evo = build_evoprior(
        seq, args.msa, use_plm=args.use_plm, plm_model=args.plm_model,
        embed_cache=args.embed_cache,
        embed_npz=args.embed_npz, embed_key=args.embed_key,
        pca_dim=args.pca_dim, cv_eval=not args.no_cv,
        w_decay=args.w_decay, p_evo_source=args.p_evo_source,
        min_neff=args.min_neff, verbose=True,
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
        f.write(f"seq_len={L}  p_chem 来源={src}\n")
        f.write(f"嵌入来源={evo['embed_source']}  use_plm={args.use_plm}\n")
        if evo.get("pca"):
            p = evo["pca"]
            f.write(f"PCA: {p['in_dim']} → {p['dim']} 维"
                    f"（解释方差 {p['explained_var']:.3f}）\n")
        cv = evo.get("cv")
        if cv:
            f.write("\n[诚实性评估] 位点 %d 折留出交叉熵（越低越好）\n" % cv["folds"])
            f.write(f"  MLP（嵌入特征）        CE={cv['mlp_ce']:.4f}\n")
            f.write(f"  基线1 全局氨基酸组成   CE={cv['comp_ce']:.4f}\n")
            if "wtcond_ce" in cv:
                f.write(f"  基线2 WT 查表（无嵌入）CE={cv['wtcond_ce']:.4f}\n")
                f.write(f"  Δ(基线1−MLP) = {cv['delta_ce']:+.4f}   "
                        f"Δ(基线2−MLP) = {cv['delta_wt_ce']:+.4f}\n")
                f.write("  判读：MLP 优于基线1 说明嵌入携带位点信息；"
                        "优于基线2 才说明嵌入带来**超出 WT 身份**的增量。\n")
            f.write(f"  PSSM 平均熵（记忆下界）="
                    f"{-np.mean(np.sum(pssm*np.log(np.clip(pssm,1e-9,None)),axis=1)):.4f}\n")
        f.write(f"\np_evo 来源={evo.get('embed_source')} / mode={args.p_evo_source}  "
                f"MSA 列深度中位数={np.median(evo['neff']):.0f}  "
                f"低覆盖列={int(evo['low_cov'].sum())}/{L}\n")
        f.write("\n")
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
