#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe remote CHPC state: queue, dirs, smoke yamls, account config."""
import paramiko

PW = "love1314520YYF"

def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("10.202.94.52", port=20009, username="u22607007", password=PW,
              timeout=25, look_for_keys=False, allow_agent=False)
    return c

def run(c, cmd, timeout=120):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")

c = connect()
print("=== CONNECTED ===")
for label, cmd in [
    ("squeue (me)", "squeue -u u22607007 -h -o '%.12i %.12j %.2t %.8M %.10P'"),
    ("default account", "sacctmgr show user u22607007 format=Account,DefaultAccount -p 2>/dev/null | head"),
    ("allowed accounts for gpu part", "scontrol show partition gpu 2>/dev/null | grep -i 'AllowAccounts\\|PartitionName\\|Default'"),
    ("REMOTE exists?", "ls -la /home/u22607007/ppri_evo_boltz_nom2 2>&1 | head -20"),
    ("smoke dir", "ls -la /home/u22607007/ppri_evo_boltz_nom2/smoke 2>&1"),
    ("n shards", "for i in $(seq 0 7); do echo -n \"shard_$i: \"; ls /home/u22607007/ppri_evo_boltz_nom2/shard_$i 2>/dev/null | wc -l; done"),
    ("msa_a3m count", "ls /home/u22607007/ppri_evo_boltz_nom2/msa_a3m 2>/dev/null | wc -l"),
    ("out_smoke cif", "find /home/u22607007/ppri_evo_boltz_nom2/out_smoke -name '*.cif' 2>/dev/null | wc -l"),
]:
    o, e = run(c, cmd)
    print(f"\n### {label}\n{o.strip()}")
    if e.strip():
        print("  [err]", e.strip()[:300])
c.close()
print("\n=== DONE ===")
