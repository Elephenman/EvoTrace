# -*- coding: utf-8 -*-
"""增量回拉全部 ESM3 embeddings 到本地 esm3_embeddings/。"""
import paramiko, os
KEY = "A:/edge/文献/10.205.1.3_0826123315_rsa.txt"
REMOTE = "/public/home/u22607007/ppri_evo/esm3_embed/out"
LOCAL = "A:/claudework/out/esm3_embeddings"
os.makedirs(LOCAL, exist_ok=True)
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.205.1.3", port=10022, username="u22607007",
          key_filename=os.path.abspath(KEY), timeout=25)
def run(cmd, t=60):
    _, o, e = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")
o, _ = run(f"ls {REMOTE}/*.npz 2>/dev/null | wc -l")
print("[remote npz]", o.strip())
remote = [f for f in run(f"ls {REMOTE}/")[0].split() if f.endswith(".npz")]
local = set(os.listdir(LOCAL))
todo = [f for f in remote if f not in local]
print(f"[fetch {len(todo)} new / {len(remote)} total]")
sftp = c.open_sftp()
n = 0
for fn in todo:
    try:
        sftp.get(f"{REMOTE}/{fn}", os.path.join(LOCAL, fn))
        n += 1
        if n % 25 == 0:
            print(f"  {n}/{len(todo)}")
    except Exception as ex:
        print("[err]", fn, ex)
print(f"[done {n}]")
c.close()
