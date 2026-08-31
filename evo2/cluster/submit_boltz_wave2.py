#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CHPC GPU（10.202.94.52:20009, 4090 分区）Boltz-2 wave-2 确认作业提交。

用法: CHPC_PASS='...' python submit_boltz_wave2.py
3 候选 × 3 条件 × 3 seed × 20 diffusion samples = 540 模型（与 wave-1 确认同规模）。
注意：该机 SFTP 报 ENOENT（ppri_evo/docs/README.md 踩坑 #2）→ 全部走 base64 通道。
"""
import base64
import os
import sys

import paramiko

HERE = os.path.dirname(os.path.abspath(__file__))
YD = os.path.join(HERE, "..", "results", "boltz_yamls_wave2")
HOST, PORT, USER = "10.202.94.52", 20009, "u22607007"

SBATCH = """#!/bin/bash
#SBATCH --job-name=boltz_w2
#SBATCH --partition=4090
#SBATCH --comment=ls_lhz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/wave2_%j.out
#SBATCH --error=logs/wave2_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate boltz
export NVIDIA_LIB=/opt/app/nvidia/570.195.03/lib
export LD_LIBRARY_PATH=$NVIDIA_LIB:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com

cd ~/evo2_boltz
for SEED in 1 2 3; do
  echo "===== boltz wave2 seed $SEED $(date) ====="
  boltz predict wave2_yamls \\
    --out_dir out_wave2_s$SEED \\
    --seed $SEED \\
    --diffusion_samples 20 \\
    --recycling_steps 3 \\
    --sampling_steps 200 \\
    --use_msa_server \\
    --override \\
    --no_trifast \\
    --cache ~/.boltz 2>&1 | tail -5
done
echo "===== DONE $(date) ====="
"""


def run(c, cmd, timeout=120):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


def put_file(c, content, remote_path):
    b64 = base64.b64encode(content.encode()).decode()
    run(c, "cat > /tmp/w2_up.b64 << 'EOB'\n" + b64 + "\nEOB")
    run(c, f"base64 -d /tmp/w2_up.b64 > {remote_path} && "
           f"sed -i 's/\r$//' {remote_path}")


def main():
    pw = os.environ.get("CHPC_PASS")
    if not pw:
        print("CHPC_PASS env required")
        sys.exit(1)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=pw, timeout=25,
              look_for_keys=False, allow_agent=False)
    run(c, "rm -rf ~/evo2_boltz/wave2_yamls && mkdir -p ~/evo2_boltz/wave2_yamls ~/evo2_boltz/logs")
    n = 0
    for fn in sorted(os.listdir(YD)):
        if fn.endswith(".yaml"):
            content = open(os.path.join(YD, fn), encoding="utf-8").read()
            put_file(c, content, f"~/evo2_boltz/wave2_yamls/{fn}")
            n += 1
    put_file(c, SBATCH, "~/evo2_boltz/wave2.sbatch")
    o, e = run(c, "ls ~/evo2_boltz/wave2_yamls | wc -l")
    print(f"[uploaded] {o.strip()} yamls")
    o, e = run(c, "cd ~/evo2_boltz && sbatch wave2.sbatch")
    print("[sbatch]", o.strip(), e.strip()[:200])
    c.close()


if __name__ == "__main__":
    main()
