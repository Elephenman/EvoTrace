#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提交去混杂扫描：22 变体 × 2 条件 = 44 预测，diffusion_samples=8（352 模型预测，约 15 min）。
全部文件走 base64 通道（CHPC SFTP put 报 ENOENT）。sbatch 严禁放在输入目录。
"""
import base64
import os
import paramiko

PW = os.environ.get("CHPC_PASS", "love1314520YYF")
LOCAL = "A:/claudework/out/boltz_yamls_deconf"
REMOTE = "/home/u22607007/ppri_evo_boltz_deconf"

BATCH = """#!/bin/bash
#SBATCH --job-name=deconf
#SBATCH --partition=4090
#SBATCH --comment=ls_lhz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/u22607007/ppri_evo_boltz_deconf/logs/dec_%j.out
#SBATCH --error=/home/u22607007/ppri_evo_boltz_deconf/logs/dec_%j.err

mkdir -p ~/ppri_evo_boltz_deconf/logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate boltz
export NVIDIA_LIB=/opt/app/nvidia/570.195.03/lib
export LD_LIBRARY_PATH=$NVIDIA_LIB:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com

cd ~/ppri_evo_boltz_deconf
echo "===== boltz deconf $(date) ====="
boltz predict yamls \\
  --out_dir out_deconf \\
  --seed 1 \\
  --diffusion_samples 8 \\
  --recycling_steps 3 \\
  --sampling_steps 200 \\
  --use_msa_server \\
  --override \\
  --no_trifast \\
  --cache ~/.boltz 2>&1 | tail -40
echo "===== DONE $(date) ====="
"""


def run(c, cmd, timeout=120):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


def up_b64(c, local_path, remote_path):
    b64 = base64.b64encode(open(local_path, "rb").read()).decode()
    o, e = run(c, f"echo {b64} | base64 -d > {remote_path}")
    if e.strip():
        print(f"  [warn] {remote_path}: {e[:120]}")


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("10.202.94.52", port=20009, username="u22607007", password=PW,
              timeout=25, look_for_keys=False, allow_agent=False)
    run(c, f"rm -rf {REMOTE} && mkdir -p {REMOTE}/yamls {REMOTE}/logs")
    local_sb = os.path.join(os.path.dirname(LOCAL), "deconf.sbatch")
    open(local_sb, "w").write(BATCH)
    n = 0
    for fn in sorted(os.listdir(LOCAL)):
        if fn.endswith(".yaml"):
            up_b64(c, os.path.join(LOCAL, fn), f"{REMOTE}/yamls/{fn}")
            n += 1
    up_b64(c, local_sb, f"{REMOTE}/deconf.sbatch")
    print(f"[up] {n} yamls + sbatch (base64)")
    o, e = run(c, f"ls {REMOTE}/yamls/*.yaml | wc -l")
    print("[verify yaml count]", o.strip(), e.strip()[:100])
    run(c, "sed -i 's/\\r$//' " + REMOTE + "/deconf.sbatch")
    o, e = run(c, f"cd {REMOTE} && sbatch deconf.sbatch")
    print("[sbatch]", o.strip(), e.strip()[:200])
    c.close()


if __name__ == "__main__":
    main()
