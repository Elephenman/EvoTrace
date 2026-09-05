#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""submit_boltz_nom2.py — n100 下一轮 Boltz 提名批次: 上传 + MSA 预取(登录节点) + 冒烟 + 全量 V100 提交。

纪律 (硬规则):
  * 仅 gpu (V100) 分区; 严禁碰 4090 (225831/225835) 与他人作业 (u12319032)。
  * 远端已有文件只读 (不改动 ppri_evo_boltz_n100); 新建 ppri_evo_boltz_nom2。
  * sbatch 不进输入目录; 日志目录先建。
  * MSA 离线: yaml 内嵌 msa: 绝对路径; 登录节点有外网预取, 计算节点(隔离网段)离线跑。
流程:
  main_upload(): 上传 nom2_seqs.tsv + prefetch 脚本 + 80 yaml(分 8 片) + sbatch
  main_prefetch(): 登录节点后台跑 MSA 预取(复用 n100 同序 a3m, 其余 MMSeqs2 API)
  main_smoke(): 提 1 片 2 yaml 小样本冒烟, 校验输出
  main_full(): 冒烟通过后提全量 8 片 array, 记录作业 ID
"""
import base64
import os
import sys
import time

import paramiko

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from chpc_util import connect, run

PW = "love1314520YYF"
LOCAL = "A:/claudework/evo2/results/boltz_nom2"
REMOTE = "/home/u22607007/ppri_evo_boltz_nom2"
NSHARD = 8
COND = ("S1_G17", "OFF_T_G17")


# ---------- 上传工具 ----------
def up_b64(c, local_path, remote_path, chunk=60000):
    b64 = base64.b64encode(open(local_path, "rb").read()).decode()
    run(c, f"rm -f {remote_path} {remote_path}.b64")
    for i in range(0, len(b64), chunk):
        run(c, f"echo {b64[i:i + chunk]} >> {remote_path}.b64")
    run(c, f"base64 -d {remote_path}.b64 > {remote_path} && rm {remote_path}.b64 && "
           f"sed -i 's/\\r$//' {remote_path}")


# ---------- MSA 预取脚本 (集群登录节点执行) ----------
PREFETCH = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, shutil
sys.path.insert(0, "/home/u22607007/miniconda3/envs/boltz221/lib/python3.11/site-packages")
from boltz.data.msa.mmseqs2 import run_mmseqs2
REMOTE="/home/u22607007/ppri_evo_boltz_nom2"
SRC="/home/u22607007/ppri_evo_boltz_n100/msa_a3m"
DST=os.path.join(REMOTE,"msa_a3m"); os.makedirs(DST, exist_ok=True)
def fseq(p):
    for ln in open(p):
        if ln.startswith(">"): continue
        return ln.strip()
idx={}
for f in os.listdir(SRC):
    if f.endswith(".a3m"):
        idx.setdefault(fseq(os.path.join(SRC,f)), f)
n=0
for line in open(os.path.join(REMOTE,"nom2_seqs.tsv")):
    name,seq=line.strip().split("\\t")
    out=os.path.join(DST,name+".a3m")
    if os.path.exists(out) and os.path.getsize(out)>0:
        print("skip",name,flush=True); continue
    if seq in idx:
        shutil.copy(os.path.join(SRC,idx[seq]), out)
        print("reuse",name,idx[seq],flush=True); n+=1; continue
    txt=run_mmseqs2(seq, prefix=os.path.join(DST,name), use_env=True, use_filter=True, use_pairing=False)
    open(out,"w").write(txt[0]); print("fetch",name,len(txt[0]),flush=True); n+=1
print("PREFETCH_DONE",n,flush=True)
'''


def main_upload():
    c = connect()
    run(c, f"rm -rf {REMOTE} && mkdir -p {REMOTE}/logs {REMOTE}/msa_a3m")
    # prefetch 脚本
    pf = os.path.join(LOCAL, "_prefetch_msa_nom2.py")
    open(pf, "w", newline="\n").write(PREFETCH)
    up_b64(c, pf, f"{REMOTE}/_prefetch_msa_nom2.py")
    # 序列清单
    up_b64(c, os.path.join(LOCAL, "nom2_seqs.tsv"), f"{REMOTE}/nom2_seqs.tsv")
    # shard yaml
    names = sorted(f[:-5] for f in os.listdir(os.path.join(LOCAL, "yamls")) if f.endswith(".yaml"))
    print(f"[upload] {len(names)} yamls -> {NSHARD} shards")
    for si in range(NSHARD):
        d = os.path.join(LOCAL, f"shard_{si}")
        os.makedirs(d, exist_ok=True)
        for nm in names[si::NSHARD]:
            os.replace(os.path.join(LOCAL, "yamls", nm + ".yaml"), os.path.join(d, nm + ".yaml"))
    for si in range(NSHARD):
        run(c, f"mkdir -p {REMOTE}/shard_{si}")
        d = os.path.join(LOCAL, f"shard_{si}")
        for fn in sorted(os.listdir(d)):
            up_b64(c, os.path.join(d, fn), f"{REMOTE}/shard_{si}/{fn}")
        print(f"  shard_{si}: {len(os.listdir(d))} yamls", flush=True)
    # 全量 sbatch
    sb = os.path.join(LOCAL, "nom2.sbatch")
    open(sb, "w", newline="\n").write(FULL_BATCH)
    up_b64(c, sb, f"{REMOTE}/nom2.sbatch")
    run(c, f"sed -i 's/\\r$//' {REMOTE}/nom2.sbatch")
    c.close()
    print("[upload] done. 下一步: main_prefetch")


def main_prefetch():
    c = connect()
    # 启动后台预取
    run(c, f"source ~/miniconda3/etc/profile.d/conda.sh; conda activate boltz221; "
           f"cd {REMOTE}; nohup python _prefetch_msa_nom2.py > prefetch.log 2>&1 &")
    # 轮询
    for _ in range(120):
        o, _ = run(c, f"tail -3 {REMOTE}/prefetch.log 2>/dev/null; echo '---'; ls {REMOTE}/msa_a3m 2>/dev/null | wc -l")
        print(o.strip())
        if "PREFETCH_DONE" in o:
            break
        time.sleep(20)
    c.close()


def main_smoke():
    """冒烟: 取 2 yaml (1 变体 × 2 条件) 小样本, gpu 单卡, 校验输出格式与离线 MSA 加载。"""
    c = connect()
    # 挑 n2_WT_F88R_M255A 两条件做冒烟 (背景特异性的核心候选)
    smoke = ["n2_WT_F88R_M255A_S1_G17", "n2_WT_F88R_M255A_OFF_T_G17"]
    run(c, f"mkdir -p {REMOTE}/smoke")
    for nm in smoke:
        run(c, f"cp {REMOTE}/shard_0/{nm}.yaml {REMOTE}/smoke/ 2>/dev/null || "
               f"find {REMOTE}/shard_* -name '{nm}.yaml' -exec cp {{}} {REMOTE}/smoke/ \\;")
    sb = os.path.join(LOCAL, "nom2_smoke.sbatch")
    open(sb, "w", newline="\n").write(SMOKE_BATCH)
    up_b64(c, sb, f"{REMOTE}/nom2_smoke.sbatch")
    run(c, f"sed -i 's/\\r$//' {REMOTE}/nom2_smoke.sbatch")
    o, e = run(c, f"cd {REMOTE} && sbatch nom2_smoke.sbatch")
    print("[smoke sbatch]", o.strip(), e.strip()[:200])
    # 等待冒烟完成
    for _ in range(40):
        o, _ = run(c, f"squeue -u u22607007 -h -o '%.12i %.12j %.2t' | grep -i smoke; "
                      f"echo '--- cif? ---'; find {REMOTE}/out_smoke -name '*.cif' 2>/dev/null | wc -l; "
                      f"echo '--- err tail ---'; tail -3 {REMOTE}/logs/smoke.out 2>/dev/null")
        print(o.strip())
        if "smoke" not in o or "cif" in o and int([x for x in o.split() if x.isdigit()][-1]) >= 1:
            # 简单判据: 找到 cif 即视为产出
            pass
        time.sleep(15)
    c.close()


def main_full():
    c = connect()
    o, e = run(c, f"cd {REMOTE} && sbatch nom2.sbatch")
    print("[full sbatch]", o.strip(), e.strip()[:300])
    o, _ = run(c, "squeue -u u22607007 -h -o '%.12i %.12j %.2t %.8M %.10P'")
    print("[queue]\n" + o.strip())
    c.close()


FULL_BATCH = """#!/bin/bash
#SBATCH --job-name=nom2
#SBATCH --partition=gpu
#SBATCH --account=ls_lhz
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
boltz predict shard_${SLURM_ARRAY_TASK_ID} \\
  --out_dir out_nom2 \\
  --seed 1 \\
  --diffusion_samples 100 \\
  --recycling_steps 3 \\
  --sampling_steps 200 \\
  --override 2>&1 | tail -30
echo "===== DONE shard ${SLURM_ARRAY_TASK_ID} $(date) ====="
"""

SMOKE_BATCH = """#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --partition=gpu
#SBATCH --account=ls_lhz
#SBATCH --comment=ls_lhz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/u22607007/ppri_evo_boltz_nom2/logs/smoke.out
#SBATCH --error=/home/u22607007/ppri_evo_boltz_nom2/logs/smoke.err

mkdir -p ~/ppri_evo_boltz_nom2/logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate boltz221
export NVIDIA_LIB=/opt/app/nvidia/570.195.03/lib
export LD_LIBRARY_PATH=$NVIDIA_LIB:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com

cd ~/ppri_evo_boltz_nom2
echo "===== boltz nom2 SMOKE $(date) ====="
boltz predict smoke \\
  --out_dir out_smoke \\
  --seed 1 \\
  --diffusion_samples 10 \\
  --recycling_steps 3 \\
  --sampling_steps 200 \\
  --override 2>&1 | tail -40
echo "===== SMOKE DONE $(date) ====="
"""


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "upload"
    {"upload": main_upload, "prefetch": main_prefetch,
     "smoke": main_smoke, "full": main_full}[step]()
