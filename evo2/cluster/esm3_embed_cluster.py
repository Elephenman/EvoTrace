#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集群端 ESM3 per-residue embedding 提取（在 10.205.1.3 的 esm3 env、CPU 下运行）。

用途：为 EvoPrior 提供真实蛋白质语言模型嵌入，替代本地的 one-hot 占位嵌入
       （受网络限制，本地无法下载 ESM-2 权重，改由集群预计算后回传 npz）。

用法:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate esm3
    python esm3_embed_cluster.py --fasta seqs.fa --out esm3_emb.npz [--layer -1]

输出:
    npz，每个 seq_id 一个 [L, D] float32 数组（已去除 BOS/EOS）。
      - layer=-1（默认）→ 最终层 embeddings（ESM3-sm-open-v1: D=1536）
      - layer=k>=0     → 第 k 层 hidden states
"""
import argparse
import os
import time

import numpy as np
import torch


def read_fasta(path):
    """返回 {seq_id: sequence} 有序字典。"""
    seqs = {}
    name, buf = None, []
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


def install_offline_data_root(model_dir):
    """让 esm 的 data_root() 直接指向本地权重目录，绕开 HuggingFace 缓存。

    集群上的 HF 缓存常因网络/传输中断而只留下 refs 而无 blobs 与 metadata，
    此时 snapshot_download() 会抛 LocalEntryNotFoundError。而权重本体
    （data/weights/esm3_sm_open_v1.pth）与 tokenizer 数据其实已在本地，
    故在导入 esm.pretrained 之前把 data_root 打补丁指向该目录。

    打补丁需在 `import esm.pretrained` **之前**执行：
      - esm/pretrained.py 用 `from esm.utils.constants.esm3 import data_root`
      - esm/tokenization/residue_tokenizer.py 用 `C.data_root(...)`
    """
    from pathlib import Path

    import esm.utils.constants.esm3 as c3

    p = Path(model_dir).expanduser().resolve()
    if not (p / "data" / "weights" / "esm3_sm_open_v1.pth").exists():
        raise FileNotFoundError(f"本地权重目录不完整: {p}")

    def _dr(model="esm3"):
        return p

    try:
        c3.C.data_root = staticmethod(_dr)       # residue_tokenizer 走 C.data_root
    except Exception:                            # pragma: no cover
        pass
    c3.data_root = _dr                           # pretrained 走 from-import
    return p


def strip_special(emb, L):
    """ESM3 token 序列含 BOS/EOS，按长度判断并剥离，返回 [L, D]。"""
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim == 3 and emb.shape[0] == 1:      # (1, L+2, D)
        emb = emb[0]
    if emb.shape[0] == L + 2:
        emb = emb[1:-1]
    elif emb.shape[0] == L + 1:
        emb = emb[:L]
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, help="输入 fasta（可多条）")
    ap.add_argument("--out", required=True, help="输出 npz 路径")
    ap.add_argument("--layer", type=int, default=-1,
                    help=">=0 取第 k 层 hidden states；-1 取最终 embeddings")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model_dir", default="~/models/esm3-sm-open-v1",
                    help="本地 ESM3 权重目录（data_root 打补丁目标）")
    ap.add_argument("--no_offline_patch", action="store_true",
                    help="禁用 data_root 打补丁，强制走 HuggingFace 缓存")
    args = ap.parse_args()

    if not args.no_offline_patch:
        p = install_offline_data_root(args.model_dir)
        print(f"[esm3] data_root -> {p}（离线直连本地权重）", flush=True)

    from esm.pretrained import load_local_model
    from esm.sdk.api import ESMProtein, LogitsConfig

    t0 = time.time()
    model = load_local_model("esm3_sm_open_v1", device=args.device)
    model.eval()
    print(f"[esm3] model loaded in {time.time() - t0:.1f}s", flush=True)

    seqs = read_fasta(args.fasta)
    print(f"[esm3] {len(seqs)} sequences to embed", flush=True)

    out = {}
    for i, (name, seq) in enumerate(seqs.items(), 1):
        t = time.time()
        protein = ESMProtein(sequence=seq)
        tok = model.encode(protein)
        use_hidden = args.layer >= 0
        with torch.no_grad():
            res = model.logits(
                tok,
                LogitsConfig(
                    sequence=True,
                    return_embeddings=(not use_hidden),
                    return_hidden_states=use_hidden,
                    ith_hidden_layer=args.layer,
                ),
            )
        raw = res.hidden_states if use_hidden else res.embeddings
        emb = strip_special(raw.float().numpy() if hasattr(raw, "float") else raw,
                            len(seq))
        if emb.shape[0] != len(seq):
            raise RuntimeError(f"{name}: emb {emb.shape} 与 L={len(seq)} 不匹配，"
                               f"请检查 BOS/EOS 处理")
        out[name] = emb.astype(np.float32)
        print(f"  [{i}/{len(seqs)}] {name}  L={len(seq)}  emb={emb.shape}  "
              f"{time.time() - t:.1f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"[done] saved {args.out}  keys={list(out)}  "
          f"total={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
