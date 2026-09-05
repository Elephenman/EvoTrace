#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Poll smoke2 (225866) and gpu partition node state for up to ~6 min."""
import paramiko, time

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
# gpu node state
o, _ = run(c, "sinfo -p gpu -o '%P %a %N %t %C' 2>/dev/null; echo '---4090---'; sinfo -p 4090 -o '%P %a %N %t %C' 2>/dev/null")
print("=== partition node state ===\n" + o.strip())
for t in range(24):  # 24 * 15s = 6 min
    oq, _ = run(c, "squeue -u u22607007 -h -o '%.12i %.12j %.2t %.8M %.10P %R'")
    print(f"-- t={t*15}s --")
    print(oq.strip())
    if "225866" not in oq:
        # job gone -> check if it ran or was killed
        oh, _ = run(c, "sacct -j 225866 -o JobID,State,NodeList,ExitCode -p 2>/dev/null | head")
        print("sacct:", oh.strip())
        oc, _ = run(c, "tail -5 /home/u22607007/ppri_evo_boltz_nom2/logs/smoke2.out 2>/dev/null; echo '--- err ---'; tail -5 /home/u22607007/ppri_evo_boltz_nom2/logs/smoke2.err 2>/dev/null")
        print("log:", oc.strip()[:500])
        break
    time.sleep(15)
c.close()
print("=== DONE ===")
