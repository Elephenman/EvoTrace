# -*- coding: utf-8 -*-
import paramiko, os
HOST, PORT, USER, PW = "10.202.94.52", 21114, "u22607007", "love1314520YYF"
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30, look_for_keys=False, allow_agent=False)
sftp = c.open_sftp()
# scan_n100.py module (the real scoring logic)
for f in ["/eaas/default/groups/public_cluster/home/u22607007/ppri_evo_boltz_n100/scan_n100.py"]:
    try:
        sftp.get(f, "scan_n100.py"); print("got scan_n100.py", os.path.getsize("scan_n100.py"))
    except Exception as e:
        print("ERR scan_n100.py", e)
sftp.close()
# inspect nom2 structure via shell
def run(cmd, t=60):
    _,o,e = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")
o,e = run("echo '== yamls in shard_0 =='; ls /home/u22607007/ppri_evo_boltz_nom2/shard_0 | head; "
          "echo '== out_nom2_full dir names =='; ls /home/u22607007/ppri_evo_boltz_nom2/out_nom2_full | head -20; "
          "echo '== count =='; ls /home/u22607007/ppri_evo_boltz_nom2/out_nom2_full | wc -l; "
          "echo '== inside one dir =='; ls /home/u22607007/ppri_evo_boltz_nom2/out_nom2_full/$(ls /home/u22607007/ppri_evo_boltz_nom2/out_nom2_full | head -1) | head; "
          "echo '== yaml head of one =='; head -40 /home/u22607007/ppri_evo_boltz_nom2/shard_0/$(ls /home/u22607007/ppri_evo_boltz_nom2/shard_0 | head -1)")
print(o)
if e.strip(): print("[ERR]", e[:500])
c.close()
