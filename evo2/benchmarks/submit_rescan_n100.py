#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wave-3 精度重扫: 37 变体 × 2 条件 × diffusion_samples=100 = 7400 模型。

目的（v4/v5 实验定位的 #1 瓶颈）: n=8/20 的抽样噪声 > 效应量（s13_c1 dual 复测 0.35 vs 0.75），
一切精度主张不可计算。n=100 把 Wilson CI 半宽压到 ±0.07 级别。

集合:
  A) deconf-22 重扫（s13_c1 背景 F88×M255 + Y217 单独）
  B) 命名系统 8 个: WT/HQL2/TrackF_r1/s13_c1/RDP2/RD_POS/TrackF_r1_M255I/HQL2_M255I
  C) 新单点: WT 背景 F88∈{K,Y,R,W}; TrackF 背景 F88∈{Y,R,W}
数组 0-3 分片（~19 预测/片 ≈ 1850 模型 ≈ 85 min/片, 4090 单卡）。
GPU 目前被外部作业占满（预计 9/7 释放）→ 本作业排队自动执行。
"""
import base64
import json
import os
import sys

import paramiko

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_sep_model_v4 import load_sequences

WT = open("A:/claudework/ppri_evo/inputs/wt_254.fasta").read().splitlines()[1].strip()
SEQS = load_sequences()
DNA = {"S1_G17": "TCATGAGCAGTTTTTTGTTTTTTT",
       "OFF_T_G17": "TTGCTATTTTTTATTGCTTTGAGT"}
LOCAL = "A:/claudework/out/boltz_yamls_n100"
NSHARD = 4
PW = "love1314520YYF"
REMOTE = "/home/u22607007/ppri_evo_boltz_n100"


def yaml_text(prot, dna):
    return (f"version: 1\nsequences:\n"
            f"  - protein:\n      id: A\n      sequence: {prot}\n"
            f"  - dna:\n      id: B\n      sequence: {dna}\n"
            f"  - ligand:\n      id: C\n      ccd: MN\n")


def diff_muts(name):
    seq = SEQS[name]
    return [(i + 22, WT[i], seq[i]) for i in range(len(WT)) if seq[i] != WT[i]]


def build_named(name, overrides=None):
    seq = list(SEQS[name])
    for pdb, wt, mut in (overrides or []):
        i = pdb - 22
        assert seq[i] == wt, (name, pdb, seq[i], wt)
        seq[i] = mut
    return "".join(seq)


def main_gen():
    os.makedirs(LOCAL, exist_ok=True)
    variants = {}                       # name -> protein sequence
    # A) deconf-22（复用 gen_boltz_deconf 的 build 逻辑, 直接用序列表）
    sys.path.insert(0, "A:/claudework/out")
    base_s13 = SEQS["s13_c1"]
    # s13_c1 在三机制的当前态
    cur = {88: base_s13[88 - 22], 255: base_s13[255 - 22], 217: base_s13[217 - 22]}
    for f in "FKRWY":
        for mm in "MIKA":
            s = list(base_s13)
            s[88 - 22] = f
            s[255 - 22] = mm
            s[217 - 22] = "Y"
            variants[f"d_F88{f}_M255{mm}_Y217Y"] = "".join(s)
    for y in "FR":
        s = list(base_s13)
        s[88 - 22] = "K"
        s[255 - 22] = "M"
        s[217 - 22] = y
        variants[f"d_F88K_M255M_Y217{y}"] = "".join(s)
    # B) 命名系统
    for n in ("WT", "HQL2", "TrackF_r1", "s13_c1", "RDP2", "RD_POS",
              "TrackF_r1_M255I", "HQL2_M255I"):
        variants[f"w_{n}"] = SEQS[n]
    # C) 新单点
    for aa in "KYRW":
        s = list(WT)
        s[88 - 22] = aa
        variants[f"sm_WT_F88{aa}"] = "".join(s)
    for aa in "YRW":
        s = list(SEQS["TrackF_r1"])
        s[88 - 22] = aa
        variants[f"sm_TrackF_F88{aa}"] = "".join(s)

    n = 0
    for vname, prot in variants.items():
        assert len(prot) == 254, vname
        for dname, dna in DNA.items():
            with open(os.path.join(LOCAL, f"{vname}_{dname}.yaml"), "w") as f:
                f.write(yaml_text(prot, dna))
            n += 1
    print(f"[gen] {len(variants)} 变体 × 2 = {n} yamls -> {LOCAL}")
    meta = {v: {"n_mut": sum(1 for a, b in zip(WT, p) if a != b)} for v, p in variants.items()}
    json.dump(meta, open(os.path.join(LOCAL, "_meta.json"), "w"), indent=1)


BATCH = """#!/bin/bash
#SBATCH --job-name=n100
#SBATCH --partition=4090
#SBATCH --comment=ls_lhz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --time=06:00:00
#SBATCH --output=/home/u22607007/ppri_evo_boltz_n100/logs/n100_%a.out
#SBATCH --error=/home/u22607007/ppri_evo_boltz_n100/logs/n100_%a.err

mkdir -p ~/ppri_evo_boltz_n100/logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate boltz
export NVIDIA_LIB=/opt/app/nvidia/570.195.03/lib
export LD_LIBRARY_PATH=$NVIDIA_LIB:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com

cd ~/ppri_evo_boltz_n100
echo "===== boltz n100 shard ${SLURM_ARRAY_TASK_ID} $(date) ====="
boltz predict shard_${SLURM_ARRAY_TASK_ID} \\
  --out_dir out_n100 \\
  --seed 1 \\
  --diffusion_samples 100 \\
  --recycling_steps 3 \\
  --sampling_steps 200 \\
  --use_msa_server \\
  --override \\
  --no_trifast \\
  --cache ~/.boltz 2>&1 | tail -30
echo "===== DONE shard ${SLURM_ARRAY_TASK_ID} $(date) ====="
"""


def run(c, cmd, timeout=90):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


def up_b64_chunked(c, local_path, remote_path, chunk=60000):
    b64 = base64.b64encode(open(local_path, "rb").read()).decode()
    run(c, f"rm -f {remote_path} {remote_path}.b64")
    for i in range(0, len(b64), chunk):
        o, e = run(c, f"echo {b64[i:i+chunk]} >> {remote_path}.b64")
        if e.strip():
            print(f"  [warn] chunk {i}: {e[:80]}")
    o, e = run(c, f"base64 -d {remote_path}.b64 > {remote_path} && rm {remote_path}.b64 && wc -c {remote_path}")
    print(f"  [up] {os.path.basename(remote_path)}: {o.split()[1] if o.split() else '?'} bytes")


def main():
    main_gen()
    names = sorted(f[:-5] for f in os.listdir(LOCAL) if f.endswith(".yaml"))
    print(f"[shard] {len(names)} predictions -> {NSHARD} shards ({len(names)//NSHARD}/shard)")
    # 本地预分片（每片一个子目录）
    for si in range(NSHARD):
        d = os.path.join(LOCAL, f"shard_{si}")
        os.makedirs(d, exist_ok=True)
        for nm in names[si::NSHARD]:
            os.replace(os.path.join(LOCAL, nm + ".yaml"), os.path.join(d, nm + ".yaml"))

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("10.202.94.52", port=20009, username="u22607007", password=PW,
              timeout=25, look_for_keys=False, allow_agent=False)
    run(c, f"rm -rf {REMOTE} && mkdir -p {REMOTE}/logs")
    for si in range(NSHARD):
        run(c, f"mkdir -p {REMOTE}/shard_{si}")
        d = os.path.join(LOCAL, f"shard_{si}")
        for fn in sorted(os.listdir(d)):
            up_b64_chunked(c, os.path.join(d, fn), f"{REMOTE}/shard_{si}/{fn}")
    sb = os.path.join(LOCAL, "n100.sbatch")
    open(sb, "w", newline="\n").write(BATCH)
    up_b64_chunked(c, sb, f"{REMOTE}/n100.sbatch")
    run(c, f"sed -i 's/\\r$//' {REMOTE}/n100.sbatch")
    o, e = run(c, f"cd {REMOTE} && sbatch n100.sbatch")
    print("[sbatch]", o.strip(), e.strip()[:200])
    o, e = run(c, "squeue -u u22607007 -h -o '%.12i %.12j %.2t %.8M' | head")
    print("[queue]", o.strip())
    c.close()


if __name__ == "__main__":
    main()
