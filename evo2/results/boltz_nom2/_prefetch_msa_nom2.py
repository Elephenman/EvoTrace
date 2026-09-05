#!/usr/bin/env python3
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
    name,seq=line.strip().split("\t")
    out=os.path.join(DST,name+".a3m")
    if os.path.exists(out) and os.path.getsize(out)>0:
        print("skip",name,flush=True); continue
    if seq in idx:
        shutil.copy(os.path.join(SRC,idx[seq]), out)
        print("reuse",name,idx[seq],flush=True); n+=1; continue
    txt=run_mmseqs2(seq, prefix=os.path.join(DST,name), use_env=True, use_filter=True, use_pairing=False)
    open(out,"w").write(txt[0]); print("fetch",name,len(txt[0]),flush=True); n+=1
print("PREFETCH_DONE",n,flush=True)
