#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v6 表征升级: 大批量变体专属 ESM3 嵌入（sugon CPU 集群, 4 片数组作业）。

序列集:
  1) cells.csv 全部 63 系统（有序列者）
  2) 单点空间: 背景 {WT, s13_c1, TrackF_r1} × 53 可变位点 × 19 AA（跳过当前残基）
      = 3021 条 —— 覆盖任何未来单点/提名候选的表征需求
上传走 base64 分块; sbatch 数组 0-3, 每片 ~770 序列 ≈ 45 min。
"""
import base64
import json
import os
import sys

import numpy as np
import paramiko

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "A:/claudework/evo2/esm3")
from ppri_surrogate_v3 import PPRI_SITES

WT = open("A:/claudework/ppri_evo/inputs/wt_254.fasta").read().splitlines()[1].strip()
SEQS = json.load(open("A:/claudework/ppri_evo/results/all_candidate_sequences.json"))
REF = json.load(open("A:/claudework/ppri_evo/results/reference_sequences.json"))
for k, v in REF.items():
    SEQS.setdefault(k, v)
KEY = "A:/edge/文献/10.205.1.3_0826123315_rsa.txt"
REMOTE = "/public/home/u22607007/ppri_evo/esm3_embed_v6"
NSHARD = 4

recs = {}
# 1) 命名系统
cells = [ln.split(",")[0] for ln in open("A:/claudework/evo2/results/sep_model_v4/cells.csv").read().splitlines()[1:]]
for name in sorted(set(cells)):
    if name in SEQS:
        recs[name] = SEQS[name]
# 2) 单点空间
BG = {"WT": WT, "s13c1": SEQS["s13_c1"], "TrackF": SEQS["TrackF_r1"]}
for bg, base in BG.items():
    for idx in PPRI_SITES:                    # 0-based
        pdb = int(idx) + 22
        cur = base[int(idx)]
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            if aa == cur:
                continue
            s = base[:int(idx)] + aa + base[int(idx) + 1:]
            recs[f"sm_{bg}_p{pdb}{aa}"] = s

print(f"[seqs] {len(recs)} 条")
os.makedirs("A:/claudework/out/esm3_v6", exist_ok=True)
names = sorted(recs)
shards = [names[i::NSHARD] for i in range(NSHARD)]
for si, sh in enumerate(shards):
    with open(f"A:/claudework/out/esm3_v6/part{si}.fa", "w") as f:
        for n in sh:
            f.write(f">{n}\n{recs[n]}\n")

BATCH = """#!/bin/bash
#SBATCH --job-name=esm3v6
#SBATCH --partition=sugon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/public/home/u22607007/ppri_evo/esm3_embed_v6/logs/v6_%a.out
#SBATCH --error=/public/home/u22607007/ppri_evo/esm3_embed_v6/logs/v6_%a.err

mkdir -p ~/ppri_evo/esm3_embed_v6/logs ~/ppri_evo/esm3_embed_v6/out
source ~/pprI_work/activate_env.sh esm3
cd ~/models/esm3-sm-open-v1
python ~/ppri_evo/esm3_embed/esm3_embed_cluster.py \\
  --fasta ~/ppri_evo/esm3_embed_v6/part${SLURM_ARRAY_TASK_ID}.fa \\
  --outdir ~/ppri_evo/esm3_embed_v6/out
echo "===== DONE shard ${SLURM_ARRAY_TASK_ID} $(date) ====="
"""


def run(c, cmd, timeout=90):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


def up_b64_chunked(c, local_path, remote_path, chunk=60000):
    raw = open(local_path, "rb").read()
    b64 = base64.b64encode(raw).decode()
    run(c, f"rm -f {remote_path}")
    for i in range(0, len(b64), chunk):
        o, e = run(c, f"echo {b64[i:i+chunk]} >> {remote_path}.b64")
        if e.strip():
            print(f"  [warn] chunk {i}: {e[:80]}")
    o, e = run(c, f"base64 -d {remote_path}.b64 > {remote_path} && rm {remote_path}.b64 && wc -c {remote_path}")
    print(f"  [up] {remote_path}: {o.strip()}")


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("10.205.1.3", port=10022, username="u22607007",
              key_filename=KEY, timeout=25)
    run(c, f"rm -rf {REMOTE} && mkdir -p {REMOTE}/logs {REMOTE}/out")
    for si in range(NSHARD):
        up_b64_chunked(c, f"A:/claudework/out/esm3_v6/part{si}.fa", f"{REMOTE}/part{si}.fa")
    sb = os.path.join("A:/claudework/out/esm3_v6", "esm3v6.sbatch")
    open(sb, "w", newline="\n").write(BATCH)
    up_b64_chunked(c, sb, f"{REMOTE}/esm3v6.sbatch")
    run(c, f"sed -i 's/\\r$//' {REMOTE}/esm3v6.sbatch")
    o, e = run(c, f"cd {REMOTE} && sbatch esm3v6.sbatch")
    print("[sbatch]", o.strip(), e.strip()[:200])
    o, e = run(c, "squeue -u u22607007 -h -o '%.10i %.12j %.2t' | head")
    print("[queue]", o.strip())
    c.close()


if __name__ == "__main__":
    main()
