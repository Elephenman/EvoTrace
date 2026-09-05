#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""submit_w4_n100.py — w4 n100 复核: 上传 yaml + MSA 预取(登录节点) + gpu 提交 + 判读.

与 n100 完全一致口径: boltz221, 2.2.1, 每 cell 100 diffusion samples, gpu 分区, msa 离线.
w4 序列: 254aa, 12 突变, F88=Q/Y217=Y/R253=R/M255=M (来自 all_candidate_sequences.json['w4']).
DNA: S1=TCATGAGCAGTTTTTTGTTTTTTT(nt17=G) / OFF=TTGCTATTTTTTATTGCTTTGAGT(nt17=C) 同 n100.
步骤: upload | prefetch | submit | poll | scan | report
"""
import base64
import json
import os
import sys
import time

import paramiko

HOST, PORT, USER, PW = "10.202.94.52", 20009, "u22607007", "love1314520YYF"
REMOTE = "/home/u22607007/ppri_evo_boltz_n100"
LOCAL = "A:/claudework/evo2/results/boltz_n100/w4_shard"
W4SEQ = json.load(open("A:/claudework/ppri_evo/results/all_candidate_sequences.json"))["w4"]

PREFETCH = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, shutil
sys.path.insert(0, "/home/u22607007/miniconda3/envs/boltz221/lib/python3.11/site-packages")
from boltz.data.msa.mmseqs2 import run_mmseqs2
REMOTE="/home/u22607007/ppri_evo_boltz_n100"
DST=os.path.join(REMOTE,"msa_a3m")
os.makedirs(DST, exist_ok=True)
SRC=os.path.join(REMOTE,"msa_a3m")

def patch_query(src_a3m, dst_a3m, seq):
    # 复制 src a3m 并把第一行 query 序列替换为 seq (列对齐一致, 仅替换查询行)
    lines=open(src_a3m).read().splitlines()
    out=[]; replaced=False
    for i,l in enumerate(lines):
        if not replaced and not l.startswith(">") and l.strip():
            out.append(seq); replaced=True
        else:
            out.append(l)
    open(dst_a3m,"w").write("\n".join(out)+"\n")
    return replaced

n=0
for line in open(os.path.join(REMOTE,"w4_shard","w4_seqs.tsv")):
    if not line.strip(): continue
    name,seq=line.strip().split("\t")
    out=os.path.join(DST,name+".a3m")
    if os.path.exists(out) and os.path.getsize(out)>0:
        print("skip",name,flush=True); continue
    try:
        txt=run_mmseqs2(seq, prefix=os.path.join(DST,name), use_env=True, use_filter=True, use_pairing=False)
        open(out,"w").write(txt[0]); print("fetch",name,len(txt[0]),flush=True); n+=1
    except Exception as ex:
        print("MMSEQ2_FAIL",name,repr(ex)[:200],"-> fallback patch w_WT",flush=True)
        src=os.path.join(SRC,"w_WT_S1_G17.a3m")
        if os.path.exists(src):
            patch_query(src, out, seq); print("patched",name,"from w_WT",flush=True); n+=1
        else:
            print("NO_FALLBACK",name,flush=True)
print("PREFETCH_DONE",n,flush=True)
'''

SBAT = r'''#!/bin/bash
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
'''

SCAN = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, csv, json, os, glob, math
sys.path.insert(0, "/home/u22607007/ppri_evo_boltz_n100")
from scan_n100 import (parse_cif, Grid, min_dist, HEXXH, READHEAD, LOCK_R253,
                       LOCK_Y217, LOCK_M255, ANCHORS, E123, CUT)
BASE="/home/u22607007/ppri_evo_boltz_n100/out_n100_offline"
OUT="/home/u22607007/ppri_evo_boltz_n100/w4_fingerprint.csv"
REPORT="/home/u22607007/ppri_evo_boltz_n100/w4_scan_report.json"
TARGETS=["boltz_results_w4_S1_G17", "boltz_results_w4_OFF_T_G17"]
rows=[]
for name in TARGETS:
    d=os.path.join(BASE,name,"predictions",name)
    if not os.path.isdir(d):
        print("MISSING",d,flush=True); continue
    cifs=sorted(glob.glob(os.path.join(d,"*.cif")))
    for cif in cifs:
        model=os.path.basename(cif).rsplit("_model_",1)[-1].replace(".cif","")
        conf={}
        cj=glob.glob(os.path.join(d,f"confidence_{name}_model_{model}.json"))
        if cj:
            try: conf=json.load(open(cj[0]))
            except Exception: conf={}
        prot,dna=parse_cif(cif)
        if not prot or not dna: continue
        grid=Grid(dna)
        all_dna=[(nt,x,y,z) for nt,at in dna.items() for (x,y,z) in at]
        hx=[a for s in HEXXH for a in prot.get(s,[])]
        d_act=min_dist(hx,all_dna) if hx else 99.0
        def sd(resi,nt):
            if resi not in prot or nt not in dna: return 99.0
            return min_dist(prot[resi],[(nt,x,y,z) for (x,y,z) in dna[nt]])
        def sany(resi):
            return min_dist(prot.get(resi,[]),all_dna) if resi in prot else 99.0
        iface_res,cover=set(),set()
        for resi,atoms in prot.items():
            for ax,ay,az in atoms:
                for nt,bx,by,bz in grid.near(ax,ay,az):
                    if (ax-bx)**2+(ay-by)**2+(az-bz)**2<=CUT*CUT:
                        iface_res.add(resi); cover.add(nt); break
        d_read17=sd(READHEAD,17); d_253_23=sd(LOCK_R253,23)
        rows.append(dict(
            pred=name, cond="S1" if "S1" in name else "OFF", model=model,
            conf=round(conf.get("confidence_score",float("nan")),4),
            iptm=round(conf.get("iptm",float("nan")),4),
            ligand_iptm=round(conf.get("ligand_iptm",float("nan")),4),
            act=int(d_act<=CUT), d_act=round(d_act,2),
            lockA=int(d_read17<=CUT), d_read17=round(d_read17,2),
            lockB=int(d_253_23<=CUT), d_253_23=round(d_253_23,2),
            dual=int(d_read17<=CUT and d_253_23<=CUT),
            d_y217_23=round(sd(LOCK_Y217,23),2), d_m255_23=round(sd(LOCK_M255,23),2),
            d_r85=round(sany(ANCHORS[0]),2), d_r207=round(sany(ANCHORS[1]),2), d_r267=round(sany(ANCHORS[2]),2),
            d_e123=round(sany(E123),2), iface=len(iface_res), ntcov=len(cover)))
    print("[ok]",name,len(cifs),flush=True)
with open(OUT,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# aggregate
def mean(xs):
    xs=[x for x in xs if isinstance(x,(int,float)) and not math.isnan(x)]
    return round(sum(xs)/len(xs),4) if xs else None
agg={}
for cond in ("S1","OFF"):
    sub=[r for r in rows if r["cond"]==cond]
    agg[cond]=dict(n=len(sub), act=mean([r["act"] for r in sub]),
                   dual=mean([r["dual"] for r in sub]),
                   conf=mean([r["conf"] for r in sub]),
                   iptm=mean([r["iptm"] for r in sub]),
                   ligand_iptm=mean([r["ligand_iptm"] for r in sub]),
                   d_act=mean([r["d_act"] for r in sub]),
                   d_read17=mean([r["d_read17"] for r in sub]),
                   d_253_23=mean([r["d_253_23"] for r in sub]),
                   d_y217_23=mean([r["d_y217_23"] for r in sub]),
                   d_m255_23=mean([r["d_m255_23"] for r in sub]),
                   iface=mean([r["iface"] for r in sub]),
                   ntcov=mean([r["ntcov"] for r in sub]))
sep_act=round(agg["S1"]["act"]-agg["OFF"]["act"],4)
sep_dual=round(agg["S1"]["dual"]-agg["OFF"]["dual"],4)
rep=dict(agg=agg, sep_act=sep_act, sep_dual=sep_dual,
         pass_crit=(sep_act is not None and sep_act>=0.3 and agg["S1"]["act"]>agg["OFF"]["act"]),
         z=-0.8214693069458008, n_mut=12, n_out53=11)
json.dump(rep, open(REPORT,"w"), indent=1)
print("WROTE",OUT,"rows",len(rows))
print("REPORT",json.dumps(rep,indent=1))
'''


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30,
              look_for_keys=False, allow_agent=False)
    return c


def run(c, cmd, t=60):
    _, o, e = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


def up_b64(c, local_path, remote_path, chunk=60000):
    b64 = base64.b64encode(open(local_path, "rb").read()).decode()
    run(c, f"rm -f {remote_path} {remote_path}.b64")
    for i in range(0, len(b64), chunk):
        run(c, f"echo {b64[i:i + chunk]} >> {remote_path}.b64")
    run(c, f"base64 -d {remote_path}.b64 > {remote_path} && rm {remote_path}.b64 && "
           f"sed -i 's/\\r$//' {remote_path}")


def main_upload():
    c = connect()
    run(c, f"mkdir -p {REMOTE}/w4_shard {REMOTE}/logs")
    up_b64(c, os.path.join(LOCAL, "w4_S1_G17.yaml"), f"{REMOTE}/w4_shard/w4_S1_G17.yaml")
    up_b64(c, os.path.join(LOCAL, "w4_OFF_T_G17.yaml"), f"{REMOTE}/w4_shard/w4_OFF_T_G17.yaml")
    up_b64(c, os.path.join(LOCAL, "w4_seqs.tsv"), f"{REMOTE}/w4_shard/w4_seqs.tsv")
    pf = os.path.join(LOCAL, "_prefetch_w4.py")
    open(pf, "w", newline="\n").write(PREFETCH)
    up_b64(c, pf, f"{REMOTE}/_prefetch_w4.py")
    sb = os.path.join(LOCAL, "w4.sbatch")
    open(sb, "w", newline="\n").write(SBAT)
    up_b64(c, sb, f"{REMOTE}/w4.sbatch")
    run(c, f"sed -i 's/\\r$//' {REMOTE}/w4.sbatch {REMOTE}/_prefetch_w4.py")
    o, _ = run(c, f"ls -la {REMOTE}/w4_shard/")
    print("[upload]", o.strip())
    c.close()


def main_prefetch():
    c = connect()
    run(c, f"source ~/miniconda3/etc/profile.d/conda.sh; conda activate boltz221; "
           f"cd {REMOTE}; nohup python _prefetch_w4.py > prefetch_w4.log 2>&1 &")
    for _ in range(60):
        o, _ = run(c, f"tail -4 {REMOTE}/prefetch_w4.log 2>/dev/null; "
                       f"echo ---; ls -l {REMOTE}/msa_a3m/w4.a3m 2>/dev/null")
        print(o.strip())
        if "PREFETCH_DONE" in o:
            break
        time.sleep(20)
    c.close()


def main_submit():
    c = connect()
    o, e = run(c, f"cd {REMOTE} && sbatch w4.sbatch")
    print("[sbatch]", o.strip(), e.strip()[:200])
    c.close()


def main_poll():
    c = connect()
    for _ in range(80):
        o, _ = run(c, f"squeue -u u22607007 -h -o '%.12i %.12j %.2t %.10M %.10P' | grep -i w4n100; "
                      f"echo ---cif---; "
                      f"find {REMOTE}/out_n100_offline/boltz_results_w4_S1_G17 "
                      f"{REMOTE}/out_n100_offline/boltz_results_w4_OFF_T_G17 "
                      f"-name '*.cif' 2>/dev/null | wc -l; "
                      f"echo ---logtail---; tail -4 {REMOTE}/logs/w4n100.out 2>/dev/null")
        print(o.strip())
        if "w4n100" not in o:
            print("[no running job — checking cif count]")
            # count cifs
            _, cnt = run(c, f"find {REMOTE}/out_n100_offline/boltz_results_w4_S1_G17 "
                            f"{REMOTE}/out_n100_offline/boltz_results_w4_OFF_T_G17 "
                            f"-name '*.cif' 2>/dev/null | wc -l")
            if cnt.strip() and int(cnt.strip()) >= 200:
                print("[200 cifs present — done]")
                break
        time.sleep(30)
    c.close()


def main_scan():
    c = connect()
    up_b64(c, __file__.replace("submit_w4_n100.py", "_scan_w4.py"), f"{REMOTE}/_scan_w4.py")
    # write SCAN text directly
    sc = os.path.join(LOCAL, "_scan_w4.py")
    open(sc, "w", newline="\n").write(SCAN)
    up_b64(c, sc, f"{REMOTE}/_scan_w4.py")
    run(c, f"sed -i 's/\\r$//' {REMOTE}/_scan_w4.py")
    o, e = run(c, f"cd {REMOTE}; source ~/miniconda3/etc/profile.d/conda.sh; "
                 f"conda activate boltz221; python _scan_w4.py 2>&1 | tail -40", t=180)
    print(o.strip())
    print("ERR", e.strip()[:300])
    c.close()


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "upload"
    {"upload": main_upload, "prefetch": main_prefetch, "submit": main_submit,
     "poll": main_poll, "scan": main_scan}[step]()
