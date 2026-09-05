#!/bin/bash
# nom2 全量跑批 — 实例单卡串跑 80 yaml x n=100
# 用法: nohup bash run_nom2_full_inst.sh > nom2_full.log 2>&1 &
export PATH="/home/u22607007/miniconda3/envs/boltz221/bin:/usr/bin:/bin:/usr/local/bin"
export HF_ENDPOINT=https://hf-mirror.com
export BOLTZ_HOME=/home/u22607007/.boltz
cd /home/u22607007/ppri_evo_boltz_nom2
OUT=out_nom2_full
mkdir -p $OUT
T0=$(date +%s)
TOTAL=0
DONE=0
for shard in shard_0 shard_1 shard_2 shard_3 shard_4 shard_5 shard_6 shard_7; do
  for y in $shard/*.yaml; do
    TOTAL=$((TOTAL+1))
    name=$(basename $y .yaml)
    # 跳过已完成的（断点续跑）
    if [ -d "$OUT/boltz_results_$name" ]; then
      echo "[SKIP] $name already done"; DONE=$((DONE+1)); continue
    fi
    echo "[$(date +%H:%M:%S)] ($DONE/$TOTAL) START $name"
    time boltz predict $y \
      --out_dir $OUT \
      --seed 1 \
      --diffusion_samples 100 \
      --recycling_steps 3 \
      --sampling_steps 200 \
      --no_kernels \
      --override > /dev/null 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      echo "[$(date +%H:%M:%S)] DONE $name"; DONE=$((DONE+1))
    else
      echo "[$(date +%H:%M:%S)] FAIL($rc) $name"
    fi
    # 每小时打印一次进度
    ELAPSED=$(( $(date +%s) - T0 ))
    echo "  [progress] done=$DONE total=$TOTAL elapsed=${ELAPSED}s"
  done
done
echo "===== ALL DONE at $(date), done=$DONE/$TOTAL ====="
