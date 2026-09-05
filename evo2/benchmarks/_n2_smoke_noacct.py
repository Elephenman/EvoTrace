#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Submit smoke WITHOUT --account to --partition=gpu. Capture sbatch response + queue.
This tests team-lead's hypothesis. Expected: rejected (ls_lhz not allowed on gpu)."""
import paramiko, time

PW = "love1314520YYF"
REMO = "/home/u22607007/ppri_evo_boltz_nom2"

SMOKE = """#!/bin/bash
#SBATCH --job-name=smoke2
#SBATCH --partition=gpu
#SBATCH --comment=ls_lhz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/u22607007/ppri_evo_boltz_nom2/logs/smoke2.out
#SBATCH --error=/home/u22607007/ppri_evo_boltz_nom2/logs/smoke2.err

mkdir -p ~/ppri_evo_boltz_nom2/logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate boltz221
export NVIDIA_LIB=/opt/app/nvidia/570.195.03/lib
export LD_LIBRARY_PATH=$NVIDIA_LIB:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com

cd ~/ppri_evo_boltz_nom2
echo "===== boltz nom2 SMOKE2 (no --account) $(date) ====="
boltz predict smoke \\
  --out_dir out_smoke2 \\
  --seed 1 \\
  --diffusion_samples 10 \\
  --recycling_steps 3 \\
  --sampling_steps 200 \\
  --override 2>&1 | tail -40
echo "===== SMOKE2 DONE $(date) ====="
"""

def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("10.202.94.52", port=20009, username="u22607007", password=PW,
              timeout=25, look_for_keys=False, allow_agent=False)
    return c

def run(c, cmd, timeout=120):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")

c = connect()
# write smoke2 sbatch
import base64
b64 = base64.b64encode(SMOKE.encode()).decode()
run(c, f"rm -f {REMO}/smoke2.sbatch {REMO}/smoke2.sbatch.b64")
for i in range(0, len(b64), 60000):
    run(c, f"echo {b64[i:i+60000]} >> {REMO}/smoke2.sbatch.b64")
run(c, f"base64 -d {REMO}/smoke2.sbatch.b64 > {REMO}/smoke2.sbatch && rm {REMO}/smoke2.sbatch.b64 && sed -i 's/\\r$//' {REMO}/smoke2.sbatch")
print("=== smoke2.sbatch written ===")
# submit
o, e = run(c, f"cd {REMO} && sbatch smoke2.sbatch", timeout=60)
print("SBATCH stdout:", o.strip())
print("SBATCH stderr:", e.strip())
# poll queue briefly
for t in range(6):
    oq, _ = run(c, "squeue -u u22607007 -h -o '%.12i %.12j %.2t %.8M %.10P'")
    print(f"-- t={t*15}s --\n{oq.strip()}")
    time.sleep(15)
c.close()
print("=== DONE ===")
