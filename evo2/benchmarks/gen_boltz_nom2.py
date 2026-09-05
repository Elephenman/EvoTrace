# -*- coding: utf-8 -*-
"""gen_boltz_nom2.py — n100 下一轮 Boltz 提名批次生成器（本地）。

提名方向（MEMORY.md / 2026-09-04 对齐）:
  n100 重扫已推翻 V2-A、钉死 F88R@WT 为全场最优单点、F88×M255 组合在 s13_c1 背景全扫。
  下一轮空白 = F88×M255 组合在 *其它* 背景（WT / TrackF_r1）的全交叉——
  直接回答"F88R/M255A 有益互作是否背景特异"。
  提名纪律: 直接边际优先、模型补盲区; 矛盾候选=信息量最大点。

生成物:
  - yamls/{stem}_{COND}.yaml   每个变体 × 2 条件, 内嵌 msa: 集群绝对路径 (离线管线)
  - nom2_seqs.tsv              stem<TAB>protein_seq  (供集群登录节点 MMSeqs2 预取 a3m)
  - _nom2_meta.json            变体元数据

yaml 离线格式对齐 n100 (boltz221, 2.2.1): 内嵌 msa: 路径, 不再 --use_msa_server。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_sep_model_v4 import load_sequences

WT = open("A:/claudework/ppri_evo/inputs/wt_254.fasta").read().splitlines()[1].strip()
SEQS = load_sequences()
DNA = {"S1_G17": "TCATGAGCAGTTTTTTGTTTTTTT",
       "OFF_T_G17": "TTGCTATTTTTTATTGCTTTGAGT"}

LOCAL = "A:/claudework/evo2/results/boltz_nom2"
REMOTE = "/home/u22607007/ppri_evo_boltz_nom2"
REMOTE_MSA = f"{REMOTE}/msa_a3m"

# 提名集: 背景 × F88 × M255 (Y217 保持天然 Y, 隔离 F88×M255 主效应)
BGS = ("WT", "TrackF_r1")
F88 = "FKRWY"          # F K R W Y  (对齐 s13_c1 deconf 字母表)
M255 = "MIKA"          # M I K A
F88I, M255I = 88 - 22, 255 - 22   # = 66, 233


def build(bg, f, m):
    s = list(SEQS[bg])
    s[F88I] = f
    s[M255I] = m
    return "".join(s)


def yaml_text(prot, dna, msa_path):
    return (f"version: 1\nsequences:\n"
            f"  - protein:\n      id: A\n      sequence: {prot}\n"
            f"      msa: {msa_path}\n"
            f"  - dna:\n      id: B\n      sequence: {dna}\n"
            f"  - ligand:\n      id: C\n      ccd: MN\n")


def main():
    os.makedirs(os.path.join(LOCAL, "yamls"), exist_ok=True)
    variants = {}
    for bg in BGS:
        for f in F88:
            for m in M255:
                name = f"n2_{bg}_F88{f}_M255{m}"
                variants[name] = build(bg, f, m)
    # 校验长度
    for n, p in variants.items():
        assert len(p) == 254, (n, len(p))

    # 序列清单 (每个变体唯一蛋白序, 2 条件共用同一 a3m)
    with open(os.path.join(LOCAL, "nom2_seqs.tsv"), "w") as fh:
        for vname, prot in variants.items():
            fh.write(f"{vname}\t{prot}\n")
    # 元数据
    meta = {}
    n_yaml = 0
    for bg in BGS:
        base = WT if bg == "WT" else SEQS["TrackF_r1"]
        for f in F88:
            for m in M255:
                vname = f"n2_{bg}_F88{f}_M255{m}"
                prot = variants[vname]
                meta[vname] = {"background": bg, "F88": f, "M255": m,
                               "n_mut": sum(1 for a, b in zip(base, prot) if a != b)}
    for vname, prot in variants.items():
        for dname, dna in DNA.items():
            stem = f"{vname}_{dname}"
            msa_path = f"{REMOTE_MSA}/{vname}.a3m"
            with open(os.path.join(LOCAL, "yamls", f"{stem}.yaml"), "w") as fh:
                fh.write(yaml_text(prot, dna, msa_path))
            n_yaml += 1
    json.dump(meta, open(os.path.join(LOCAL, "_nom2_meta.json"), "w"), indent=1)
    print(f"[gen] {len(variants)} 变体 × 2 条件 = {n_yaml} yamls -> {LOCAL}/yamls")
    print(f"[gen] {len(variants)} 唯一蛋白序 -> nom2_seqs.tsv (MSA 预取清单)")
    # 简短提名清单表
    print(f"\n| 变体 | 背景 | F88 | M255 | 相对背景突变数 |")
    print("|---|---|---|---|---|")
    for v, p in variants.items():
        print(f"| {v} | {meta[v]['background']} | {meta[v]['F88']} | {meta[v]['M255']} | {meta[v]['n_mut']} |")


if __name__ == "__main__":
    main()
