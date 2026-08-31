#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 ESM3 零样本打分载荷（fasta 分片 + manifest）。

输出 cluster/payload/<bench>/：seq_<n>.fasta（单序列文件）+ manifest.csv
基准：
  avgfp_panel  — B1a 全评测面板（结构可变位点内的实测变体）
  tem1         — B2 Firnberg 2014 全部单突变
  gb1_test     — B1b FLIP two_vs_rest 测试子样
  pgym_sub     — B4 ProteinGym 分层子集（30 数据集 × ≤1000 变体）
"""
import json
import os
import random
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
PGYM_DIR = ("A:/claudework/evo_data/processed/proteingym_benchmark/"
            "DMS_ProteinGym_substitutions")
OUT = os.path.join(HERE, "payload")


def parse_mutant(mutant):
    out = []
    for tok in str(mutant).split(":"):
        if len(tok) >= 4:
            out.append((int(tok[1:-1]) - 1, tok[-1]))
    return out


def emit(name, items):
    """items: list of (seq_id, dataset, seq)。写单一多序列 fasta + 瘦 manifest。"""
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "all.fasta"), "w") as f:
        for sid, ds, seq in items:
            f.write(f">{sid}\n{seq}\n")
    man = [dict(seq_id=sid, dataset=ds) for sid, ds, _ in items]
    pd.DataFrame(man).to_csv(os.path.join(d, "manifest.csv"), index=False)
    print(f"{name}: {len(items)} seqs -> {d}")
    return d


def wt_of_df(df):
    """从单突变 token 投票重建 WT。"""
    from collections import Counter
    votes = Counter()
    for m in df.mutant:
        for tok in str(m).split(":"):
            if len(tok) >= 4:
                votes[(int(tok[1:-1]) - 1, tok[0])] += 1
    seq_len = max(p for p, _ in votes) + 1
    wt = ["X"] * seq_len
    for (p, aa), n in votes.items():
        if wt[p] == "X" or n > votes.get((p, wt[p]), 0):
            wt[p] = aa
    return "".join(wt)


def build_avgfp():
    df = pd.read_csv(os.path.join(PGYM_DIR, "GFP_AEQVI_Sarkisyan_2016.csv"))
    wt = json.load(open(os.path.join(HERE, "..", "benchmarks", "data_avgfp_wt.json")))["wt"]
    items = []
    for i, (m, s) in enumerate(zip(df.mutant, df.mutated_sequence)):
        muts = parse_mutant(m)
        if muts and len(s) == len(wt):
            items.append((f"avgfp_{i}", "avgfp", s))
    return emit("avgfp_panel", items)


def build_tem1():
    df = pd.read_csv(os.path.join(PGYM_DIR, "BLAT_ECOLX_Firnberg_2014.csv"))
    items = [(f"tem1_{i}", "tem1", s) for i, s in enumerate(df.mutated_sequence)]
    return emit("tem1", items)


def build_gb1():
    d = "A:/claudework/evo_data/raw/flip/splits/gb1"
    df = pd.read_csv(os.path.join(d, "four_mutations_full_data.csv"))
    test = df[df["two_vs_rest"] == "test"]
    random.seed(0)
    idx = random.sample(range(len(test)), min(5000, len(test)))
    sub = test.iloc[idx]
    items = [(f"gb1_{r.Variants}", "gb1", r.sequence) for r in sub.itertuples()]
    return emit("gb1_test", items)


def build_pgym_sub(n_datasets=30, per_cap=1000):
    files = sorted(os.listdir(PGYM_DIR))
    random.seed(42)
    picks = random.sample(files, n_datasets)
    items = []
    for fn in picks:
        df = pd.read_csv(os.path.join(PGYM_DIR, fn))
        if len(df) > per_cap:
            df = df.sample(per_cap, random_state=0)
        ds = fn[:-4]
        for i, s in enumerate(df.mutated_sequence):
            items.append((f"{ds}_{i}", ds, s))
    return emit("pgym_sub", items)



def build_tem1_x():
    items = []
    for fn in ["BLAT_ECOLX_Deng_2012.csv", "BLAT_ECOLX_Stiffler_2015.csv",
               "BLAT_ECOLX_Jacquier_2013.csv"]:
        df = pd.read_csv(os.path.join(PGYM_DIR, fn))
        ds = fn[:-4]
        for i, s in enumerate(df.mutated_sequence):
            items.append((f"{ds}_{i}", ds, s))
    return emit("tem1_x", items)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "test"):
        # 计时测试：20 条 avGFP
        df = pd.read_csv(os.path.join(PGYM_DIR, "GFP_AEQVI_Sarkisyan_2016.csv"))
        wt = json.load(open(os.path.join(HERE, "..", "benchmarks", "data_avgfp_wt.json")))["wt"]
        items = [(f"test_{i}", "test", s) for i, s in enumerate(df.mutated_sequence[:20])]
        emit("timetest", items)
    if which in ("all", "avgfp"):
        build_avgfp()
    if which in ("all", "tem1"):
        build_tem1()
    if which in ("all", "gb1"):
        build_gb1()
    if which in ("all", "pgym"):
        build_pgym_sub()
    if which in ("all", "tem1x"):
        build_tem1_x()
