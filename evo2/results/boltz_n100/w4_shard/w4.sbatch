#!/bin/bash
#SBATCH --job-name=w4n100
#SBATCH --partition=gpu
#SBATCH --account=ls_lhz
#SBATCH --comment=ls_lhz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/u22607007/ppri_evo_boltz_n100/logs/w4n100.out
#SBATCH --error=/home/u22607007/ppri_evo_boltz_n100/logs/w4n100.err

mkdir -p ~/ppri_evo_boltz_n100/logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate boltz221
export NVIDIA_LIB=/opt/app/nvidia/570.195.03/lib
export LD_LIBRARY_PATH=$NVIDIA_LIB:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com

cd ~/ppri_evo_boltz_n100
echo "===== boltz w4 n100 $(date) ====="
boltz predict w4_shard \
  --out_dir out_n100_offline \
  --seed 1 \
  --diffusion_samples 100 \
  --recycling_steps 3 \
  --sampling_steps 200 \
  --override 2>&1 | tail -40
echo "===== DONE w4 n100 $(date) ====="
