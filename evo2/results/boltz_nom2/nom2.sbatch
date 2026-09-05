#!/bin/bash
#SBATCH --job-name=nom2
#SBATCH --partition=gpu
#SBATCH --comment=ls_lhz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --time=06:00:00
#SBATCH --output=/home/u22607007/ppri_evo_boltz_nom2/logs/nom2_%a.out
#SBATCH --error=/home/u22607007/ppri_evo_boltz_nom2/logs/nom2_%a.err

mkdir -p ~/ppri_evo_boltz_nom2/logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate boltz221
export NVIDIA_LIB=/opt/app/nvidia/570.195.03/lib
export LD_LIBRARY_PATH=$NVIDIA_LIB:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com

cd ~/ppri_evo_boltz_nom2
echo "===== boltz nom2 shard ${SLURM_ARRAY_TASK_ID} $(date) ====="
boltz predict shard_${SLURM_ARRAY_TASK_ID} \
  --out_dir out_nom2 \
  --seed 1 \
  --diffusion_samples 100 \
  --recycling_steps 3 \
  --sampling_steps 200 \
  --override 2>&1 | tail -30
echo "===== DONE shard ${SLURM_ARRAY_TASK_ID} $(date) ====="
