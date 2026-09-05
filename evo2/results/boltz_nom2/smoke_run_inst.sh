#!/bin/bash
# nom2 冒烟 — 实例单卡验证 boltz 通路 (1 yaml x n=10)
export PATH="/home/u22607007/miniconda3/envs/boltz221/bin:/usr/bin:/bin:/usr/local/bin"
export HF_ENDPOINT=https://hf-mirror.com
export BOLTZ_HOME=/home/u22607007/.boltz
cd /home/u22607007/ppri_evo_boltz_nom2
echo "===== NOM2 SMOKE start $(date) ====="
time boltz predict shard_0/n2_WT_F88R_M255A_OFF_T_G17.yaml \
  --out_dir out_smoke_inst \
  --seed 1 \
  --diffusion_samples 10 \
  --recycling_steps 3 \
  --sampling_steps 200 \
  --no_kernels \
  --override 2>&1 | tail -40
echo "===== SMOKE DONE $(date) rc=${PIPESTATUS[0]} ====="
