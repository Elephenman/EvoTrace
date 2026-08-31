#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集群端 ESM3 per-residue embedding 批量提取（在 10.205.1.3 esm3 env 内运行）。

用法: python esm3_embed_cluster.py --fasta seqs.fa --outdir out/
输出: out/<name>.npz  (emb: L×1536 float32，每残基一行，token[1:-1] 对齐残基)
- 逐条处理、已完成自动跳过（断点续跑）
- 超长序列（>2048aa）forward 失败时居中裁剪到 2048 重试
"""
import argparse
import os
import time

import numpy as np
import torch
from esm.pretrained import load_local_model
from esm.sdk.api import ESMProtein, LogitsConfig


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.time()
    model = load_local_model("esm3_sm_open_v1", device="cpu")
    model.eval()
    print(f"[esm3] model loaded {time.time()-t0:.0f}s", flush=True)

    # hook 最后一个 transformer block 输出
    h = {}

    def hook(mod, i, o):
        out = o[0] if isinstance(o, tuple) else o
        h["x"] = out.detach().float()
        return o

    model.transformer.blocks[-1].register_forward_hook(hook)

    seqs = read_fasta(args.fasta)
    print(f"[esm3] {len(seqs)} seqs", flush=True)

    done = skip = 0
    for name, seq in seqs.items():
        npz = os.path.join(args.outdir, name + ".npz")
        if os.path.exists(npz):
            skip += 1
            continue
        L = len(seq)
        t1 = time.time()
        try:
            emb = extract(model, h, seq)
        except Exception as e:
            if L > 2048:  # 兜底：居中裁剪
                print(f"  {name}: L={L} forward err ({type(e).__name__}: {str(e)[:60]}) -> crop 2048", flush=True)
                c = (L - 2048) // 2
                emb = extract(model, h, seq[c:c + 2048])
            else:
                print(f"  {name}: FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
                continue
        np.savez_compressed(npz, emb=emb.numpy())
        done += 1
        print(f"  {name}: L={L} emb={tuple(emb.shape)} {time.time()-t1:.0f}s (done={done} skip={skip})", flush=True)

    print(f"[esm3] DONE done={done} skip={skip} total={time.time()-t0:.0f}s", flush=True)


def extract(model, h, seq):
    """单条序列 → per-residue embedding (L, 1536)。"""
    h.clear()
    protein = ESMProtein(sequence=seq)
    t = model.encode(protein)
    with torch.no_grad():
        _ = model.logits(t, LogitsConfig(sequence=True))
    hid = h["x"][0]  # (Ltok, 1536); token0=<cls>, token-1=<eos>
    L = len(seq)
    assert hid.shape[0] >= L + 2, (hid.shape, L)
    return hid[1:1 + L]


if __name__ == "__main__":
    main()
