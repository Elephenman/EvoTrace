#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提交 V2-A 候选 Boltz-2 验证：TrackF_r1+M255I / HQL2+M255I × {S1, OFF} = 4 预测。
参数与六候选验证 (job 225685) 完全一致（seed 1 / 20 samples / recycling 3 / steps 200），
保证与亲本判读可直接配对比较。文件走 base64 通道（CHPC SFTP put 报 ENOENT）。
"""
import os
import base64
import paramiko

PW = os.environ.get("CHPC_PASS", "love1314520YYF")
LOCAL = "A:/claudework/out/boltz_yamls_v2a"
REMOTE = "/home/u22607007/ppri_evo_boltz_v2a"

BATCH = """#!/bin/bash
#SBATCH --job-name=v2a_boltz
#SBATCH --partition=4090
#SBATCH --comment=ls_lhz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/u22607007/ppri_evo_boltz_v2a/logs/v2a_%j.out
#SBATCH --error=/home/u22607007/ppri_evo_boltz_v2a/logs/v2a_%j.err

mkdir -p ~/ppri_evo_boltz_v2a/logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate boltz
export NVIDIA_LIB=/opt/app/nvidia/570.195.03/lib
export LD_LIBRARY_PATH=$NVIDIA_LIB:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com

cd ~/ppri_evo_boltz_v2a
echo "===== boltz v2a $(date) ====="
boltz predict yamls \\
  --out_dir out_v2a \\
  --seed 1 \\
  --diffusion_samples 20 \\
  --recycling_steps 3 \\
  --sampling_steps 200 \\
  --use_msa_server \\
  --override \\
  --no_trifast \\
  --cache ~/.boltz 2>&1 | tail -40
echo "===== DONE $(date) ====="
"""


def run(c, cmd, timeout=90):
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
    local_sb = os.path.join(os.path.dirname(LOCAL), "v2a.sbatch")
    open(local_sb, "w").write(BATCH)
    n = 0
    for fn in sorted(os.listdir(LOCAL)):
        if fn.endswith(".yaml"):
            up_b64(c, os.path.join(LOCAL, fn), f"{REMOTE}/yamls/{fn}")
            n += 1
    up_b64(c, local_sb, f"{REMOTE}/v2a.sbatch")
    print(f"[up] {n} yamls + sbatch (base64)")
    o, e = run(c, "ls " + REMOTE + "/yamls/ | wc -l")
    print("[verify yaml count]", o.strip(), e.strip()[:100])
    run(c, "sed -i 's/\\r$//' " + REMOTE + "/v2a.sbatch")
    o, e = run(c, f"cd {REMOTE} && sbatch v2a.sbatch")
    print("[sbatch]", o.strip(), e.strip()[:200])
    c.close()


if __name__ == "__main__":
    main()
