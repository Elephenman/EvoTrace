#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集群端 ESM3 零样本打分（10.205.1.3 sugon 分区，esm3 env）。

用法: python esm3_score_batch.py --manifest manifest.csv --shard-id 0 --n-shards 8 --out-dir out
manifest.csv 列: seq_id,dataset,fasta_path  （fasta_path 为集群本地路径）
输出: out/scores_shard{ID}.csv  列: seq_id,dataset,n_mut,sum_logp,mean_logp
口径: 全序列上下文 log-likelihood（ESM3 logits 逐位点 logp 求和）——零样本适应度代理。
"""
import argparse
import csv
import os

import numpy as np
import torch


def read_fasta(path):
    seqs, name, buf = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        elif line:
            buf.append(line)
    if name:
        seqs[name] = "".join(buf)
    return seqs


def score_sequence(model, tk, seq, ESMProtein, LogitsConfig):
    t = model.encode(ESMProtein(sequence=seq))
    with torch.no_grad():
        out = model.logits(t, LogitsConfig(sequence=True))
    logits = out.logits.sequence[0].float()
    logp = torch.log_softmax(logits, dim=-1).cpu().numpy()
    toks = np.array(tk.encode(seq))
    return logp[np.arange(len(toks)), toks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from esm.pretrained import load_local_model
    from esm.sdk.api import ESMProtein, LogitsConfig

    from esm.sdk.api import ESMProtein, LogitsConfig
    model = load_local_model("esm3_sm_open_v1", device="cpu")
    tk = model.tokenizers.sequence
    model.eval()
    torch.set_num_threads(max(1, os.cpu_count() - 1))

    rows = list(csv.DictReader(open(args.manifest)))
    shard = rows[args.shard_id::args.n_shards]
    fasta_all = read_fasta(os.path.join(os.path.dirname(os.path.abspath(args.manifest)),
                                        "all.fasta"))
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"scores_shard{args.shard_id}.csv")
    done = set()
    if os.path.exists(out_path):
        done = {r["seq_id"] for r in csv.DictReader(open(out_path))}
        print(f"[resume] {len(done)} already scored")
    n = 0
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["seq_id", "dataset", "n_mut", "sum_logp", "mean_logp"])
            f.flush()
        for r in shard:
            if r["seq_id"] in done:
                continue
            seq = fasta_all.get(r["seq_id"])
            if seq is None:
                continue
            try:
                lp = score_sequence(model, tk, seq, ESMProtein, LogitsConfig)
            except Exception as e:
                print(f"[err] {r['seq_id']}: {e}", flush=True)
                continue
            w.writerow([r["seq_id"], r["dataset"], len(seq), round(float(lp.sum()), 4),
                        round(float(lp.mean()), 5)])
            n += 1
            if n % 20 == 0:
                f.flush()
                print(f"[shard {args.shard_id}] {n} scored ({r['seq_id']})", flush=True)
    print(f"[shard {args.shard_id}] DONE {n}")


if __name__ == "__main__":
    main()
