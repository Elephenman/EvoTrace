# -*- coding: utf-8 -*-
"""Re-upload (fixed sys.path) + kill prior + relaunch nom2 scan in background."""
import base64
import paramiko

HOST, PORT, USER, PW = "10.202.94.52", 21114, "u22607007", "love1314520YYF"
REMOTE_SCRIPT = "/home/u22607007/ppri_evo_boltz_nom2/scan_nom2_full.py"
LOG = "/home/u22607007/ppri_evo_boltz_nom2/scan_nom2_full.log"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.WarningPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30,
          look_for_keys=False, allow_agent=False)

def ex(cmd, t=60):
    _, o, e = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")

# verify scan_n100.py path on instance
o, e = ex("ls -l /eaas/default/groups/public_cluster/home/u22607007/ppri_evo_boltz_n100/scan_n100.py")
print("[verify scan_n100.py]", o.strip(), e.strip()[:200])

# kill any prior running scan
o, e = ex("pkill -f scan_nom2_full.py; sleep 1; ps -eo pid,cmd | grep -c '[s]can_nom2_full.py'", 30)
print("[killed prior]", o.strip(), e.strip()[:100])

# upload via base64
raw = open("scan_nom2_full.py", "rb").read()
b64 = base64.b64encode(raw).decode()
ex(f"rm -f {REMOTE_SCRIPT} {REMOTE_SCRIPT}.b64")
for i in range(0, len(b64), 60000):
    ex(f"echo {b64[i:i+60000]} >> {REMOTE_SCRIPT}.b64")
ex(f"base64 -d {REMOTE_SCRIPT}.b64 > {REMOTE_SCRIPT} && rm {REMOTE_SCRIPT}.b64 && "
   f"sed -i 's/\\r$//' {REMOTE_SCRIPT} && chmod 644 {REMOTE_SCRIPT}")
print("[uploaded]")

# import sanity
o, e = ex("export PATH=/home/u22607007/miniconda3/envs/boltz221/bin:$PATH; "
          "python -c 'import sys; sys.path.insert(0,\"/eaas/default/groups/public_cluster/home/u22607007/ppri_evo_boltz_n100\"); "
          "import scan_n100; print(\"import-ok CUT=\", scan_n100.CUT)'")
print("[sanity]", o.strip(), e.strip()[:300])

# launch (umask 022 so output CSV is readable for SFTP)
launch = (f"export PATH=/home/u22607007/miniconda3/envs/boltz221/bin:/usr/bin:/bin; "
          f"umask 022; "
          f"cd /home/u22607007/ppri_evo_boltz_nom2; "
          f"setsid nohup python {REMOTE_SCRIPT} > {LOG} 2>&1 </dev/null & echo launched pid=$!")
_, o, e = c.exec_command(launch, timeout=30)
print("[launch]", o.read().decode().strip(), e.read().decode().strip()[:200])
c.close()
print("DONE.")
