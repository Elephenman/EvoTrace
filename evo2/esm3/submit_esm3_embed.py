#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提交 ESM3 embedding 提取到 CPU 集群（10.205.1.3, sugon 分区）。
上传 fasta + 脚本 + sbatch，提交后打印 job id。"""
import os
import paramiko

KEY = "A:/edge/文献/10.205.1.3_0826123315_rsa.txt"
LOCAL = "A:/claudework/out"
REMOTE_DIR = "/public/home/u22607007/ppri_evo/esm3_embed"

BATCH = """#!/bin/bash
#SBATCH --job-name=evo_esm3emb
#SBATCH --partition=sugon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/esm3emb_%j.out
#SBATCH --error=logs/esm3emb_%j.err

source ~/pprI_work/activate_env.sh esm3
cd ~/models/esm3-sm-open-v1
mkdir -p ~/ppri_evo/esm3_embed/out ~/ppri_evo/logs
python ~/ppri_evo/esm3_embed/esm3_embed_cluster.py \\
  --fasta ~/ppri_evo/esm3_embed/esm3_input_seqs.fa \\
  --outdir ~/ppri_evo/esm3_embed/out
echo "===== DONE $(date) ====="
"""


def run(c, cmd, timeout=60):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("10.205.1.3", port=10022, username="u22607007",
              key_filename=os.path.abspath(KEY), timeout=25)
    o, e = run(c, "echo HOME=$HOME; mkdir -p ~/ppri_evo/esm3_embed/out ~/ppri_evo/logs")
    print("[HOME]", o.strip(), e.strip()[:100])
    sftp = c.open_sftp()
    for fn in ["esm3_input_seqs.fa", "esm3_embed_cluster.py"]:
        lp = os.path.join(LOCAL, fn)
        rp = f"{REMOTE_DIR}/{fn}"
        sftp.put(lp, rp)
        print(f"[up] {fn}")
    with sftp.open(f"{REMOTE_DIR}/esm3_embed.sbatch", "w") as f:
        f.write(BATCH)
    run(c, "sed -i 's/\\r$//' ~/ppri_evo/esm3_embed/esm3_embed.sbatch")
    o, e = run(c, "cd ~/ppri_evo/esm3_embed && sbatch esm3_embed.sbatch")
    print("[sbatch]", o.strip(), e.strip()[:200])
    c.close()


if __name__ == "__main__":
    main()
