#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import paramiko
HOST, PORT, USER, PW = "10.202.94.52", 20009, "u22607007", "love1314520YYF"
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30)
def run(cmd):
    stdin, stdout, stderr = c.exec_command(cmd)
    return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")
print(run(r"source ~/md/activate_md.sh >/dev/null 2>&1; F=$AMBERHOME/dat/leap/lib/atomic_ions.lib; echo '=== lines 1455-1485 ==='; sed -n '1455,1485p' $F"))
c.close()
