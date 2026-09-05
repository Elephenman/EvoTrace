#!/usr/bin/env python3
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
