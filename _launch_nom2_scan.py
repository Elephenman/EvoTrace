# -*- coding: utf-8 -*-
"""Upload scan_nom2_full.py to instance and launch in background (setsid nohup)."""
import base64
import paramiko

HOST, PORT, USER, PW = "10.202.94.52", 21114, "u22607007", "love1314520YYF"
REMOTE_SCRIPT = "/home/u22607007/ppri_evo_boltz_nom2/scan_nom2_full.py"
LOG = "/home/u22607007/ppri_evo_boltz_nom2/scan_nom2_full.log"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30, look_for_keys=False, allow_agent=False)

# 1) upload via base64 (avoid Windows-PATH heredoc corruption)
raw = open("scan_nom2_full.py", "rb").read()
b64 = base64.b64encode(raw).decode()
_, _, _ = c.exec_command(f"rm -f {REMOTE_SCRIPT} {REMOTE_SCRIPT}.b64")
def ex(cmd):
    _, o, e = c.exec_command(cmd, timeout=60)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")
chunk = 60000
for i in range(0, len(b64), chunk):
    c.exec_command(f"echo {b64[i:i+chunk]} >> {REMOTE_SCRIPT}.b64")
c.exec_command(f"base64 -d {REMOTE_SCRIPT}.b64 > {REMOTE_SCRIPT} && rm {REMOTE_SCRIPT}.b64 && "
               f"sed -i 's/\\r$//' {REMOTE_SCRIPT}")
print("[uploaded]", REMOTE_SCRIPT)

# 2) sanity: confirm scan_n100 module exists & python can import basics
o, e = ex(
    "ls -l /home/u22607007/ppri_evo_boltz_n100/scan_n100.py; "
    "export PATH=/home/u22607007/miniconda3/envs/boltz221/bin:$PATH; "
    "python -c 'import sys; sys.path.insert(0,\"/home/u22607007/ppri_evo_boltz_n100\"); "
    "import scan_n100; print(\"import-ok CUT=\", scan_n100.CUT, \"HEXXH=\", scan_n100.HEXXH)'")
print(o.strip())
if e.strip():
    print("[stderr]", e.strip()[:400])

# 3) launch background job (setsid nohup, detached)
launch = (
    f"export PATH=/home/u22607007/miniconda3/envs/boltz221/bin:/usr/bin:/bin; "
    f"cd /home/u22607007/ppri_evo_boltz_nom2; "
    f"setsid nohup python {REMOTE_SCRIPT} > {LOG} 2>&1 </dev/null & echo launched pid=$!"
)
_, o, e = c.exec_command(launch, timeout=30)
print("[launch]", o.read().decode(errors="replace").strip(), e.read().decode(errors="replace").strip()[:200])
c.close()
print("DONE launching.")
