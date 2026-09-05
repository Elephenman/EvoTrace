# -*- coding: utf-8 -*-
import paramiko
HOST, PORT, USER, PW = "10.202.94.52", 21114, "u22607007", "love1314520YYF"
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30, look_for_keys=False, allow_agent=False)
def run(cmd, t=60):
    _,o,e = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")
o,e = run(
  "D=$(ls -d /home/u22607007/ppri_evo_boltz_nom2/out_nom2_full/boltz_results_* | head -1); "
  "echo \"DIR=$D\"; "
  "echo '== predictions listing =='; ls \"$D/predictions\"; "
  "echo '== find cifs (depth) =='; find \"$D\" -name '*.cif' | head -3; "
  "echo '== cif count =='; find \"$D\" -name '*.cif' | wc -l; "
  "echo '== confidence files sample =='; find \"$D\" -name 'confidence*json' | head -3; "
  "echo '== total cifs across all 80 =='; find /home/u22607007/ppri_evo_boltz_nom2/out_nom2_full -name '*.cif' | wc -l; "
  "echo '== any empty/failed dirs? (dirs with <100 cif) =='; "
  "for d in /home/u22607007/ppri_evo_boltz_nom2/out_nom2_full/boltz_results_*; do n=$(find \"$d\" -name '*.cif' | wc -l); if [ \"$n\" -ne 100 ]; then echo \"$d -> $n\"; fi; done; echo '== done scan =='")
print(o)
if e.strip(): print("[ERR]", e[:600])
c.close()
