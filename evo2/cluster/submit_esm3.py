#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU 集群（10.205.1.3 sugon）ESM3 打分作业提交器。

用法: python submit_esm3.py <bench> <n_shards> [walltime_h]
  bench: timetest / avgfp_panel / tem1 / gb1_test / pgym_sub
流程: 上传 payload 目录 + esm3_score_batch.py → sbatch array 作业 → 返回作业号。
集群侧路径: ~/evo2_esm/<bench>/
"""
import os
import sys

import paramiko

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = "A:/edge/文献/10.205.1.3_0826123315_rsa.txt"
HOST, PORT, USER = "10.205.1.3", 10022, "u22607007"

SBATCH = """#!/bin/bash
#SBATCH --job-name=esm3_{bench}
#SBATCH --partition=sugon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time={walltime}
#SBATCH --array=0-{nshards}
#SBATCH --output=logs/{bench}_%A_%a.out
#SBATCH --error=logs/{bench}_%A_%a.err

source ~/pprI_work/activate_env.sh esm3
# 权重按相对路径 data/weights/ 加载 -> 必须在模型目录下运行
cd ~/models/esm3-sm-open-v1
mkdir -p ~/evo2_esm/{bench}/logs ~/evo2_esm/{bench}/scores
python ~/evo2_esm/esm3_score_batch.py --manifest ~/evo2_esm/{bench}/manifest.csv --shard-id $SLURM_ARRAY_TASK_ID --n-shards {nshards} --out-dir ~/evo2_esm/{bench}/scores
"""


def run(c, cmd, timeout=120):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


def main():
    bench = sys.argv[1]
    n_shards = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    walltime = (sys.argv[3] if len(sys.argv) > 3 else "4") + ":00:00"
    local = os.path.join(HERE, "payload", bench)
    man = os.path.join(local, "manifest.csv")
    n_seqs = sum(1 for _ in open(man)) - 1
    print(f"[submit] {bench}: {n_seqs} seqs, {n_shards} shards, walltime {walltime}")

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, key_filename=os.path.abspath(KEY), timeout=25)
    remote = f"~/evo2_esm/{bench}"
    run(c, f"mkdir -p {remote}/logs {remote}/scores")
    sftp = c.open_sftp()
    # 打分脚本
    sftp.put(os.path.join(HERE, "esm3_score_batch.py"),
             f"/public/home/{USER}/evo2_esm/esm3_score_batch.py")
    # payload（单 fasta；manifest 只含 seq_id/dataset）
    sftp.put(os.path.join(local, "all.fasta"), f"/public/home/{USER}/evo2_esm/{bench}/all.fasta")
    sftp.put(man, f"/public/home/{USER}/evo2_esm/{bench}/manifest.csv")
    sb = SBATCH.format(bench=bench, nshards=n_shards - 1, walltime=walltime)
    with sftp.open(f"/public/home/{USER}/evo2_esm/{bench}/run.sbatch", "w") as f:
        f.write(sb)
    # 改写 manifest fasta 路径为集群绝对路径，去 CR
    run(c, f"sed -i 's|A:/claudework/evo2/cluster/payload|/public/home/{USER}/evo2_esm|g' "
           f"{remote}/manifest.csv")
    run(c, f"sed -i 's/\\r$//' {remote}/run.sbatch {remote}/manifest.csv")
    o, e = run(c, f"cd {remote} && sbatch run.sbatch")
    print("[sbatch]", o.strip(), e.strip()[:300])
    c.close()


if __name__ == "__main__":
    main()
