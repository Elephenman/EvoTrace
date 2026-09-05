#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose which partition(s) allow the ls_lhz account / user u22607007."""
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
# 1) Does ls_lhz account exist and what partitions allow it?
for label, cmd in [
    ("account ls_lhz show", "sacctmgr show account ls_lhz format=Account,PartitionName -p 2>&1 | head"),
    ("user u22607007 all accts", "sacctmgr show user u22607007 format=User,Account,Partition -p 2>&1 | head"),
    ("all partitions + AllowAccounts", "for p in $(sinfo -h -o '%P'); do echo \"== $p ==\"; scontrol show partition $p 2>/dev/null | grep -i 'AllowAccounts' | grep -o 'ls_lhz' && echo \"  -> ls_lhz ALLOWED on $p\" || echo '  (ls_lhz not in AllowAccounts)'; done"),
    ("my current jobs/held", "squeue -u u22607007"),
    ("recently denied? sacct", "sacct -u u22607007 --starttime=$(date -d '2 days ago' +%Y-%m-%d) -o JobID,JobName,State,Account,Partition -p 2>/dev/null | head -20"),
]:
    o, e = run(c, cmd, timeout=60)
    print(f"\n### {label}\n{o.strip()}")
    if e.strip():
        print("  [err]", e.strip()[:200])
c.close()
print("\n=== DONE ===")
