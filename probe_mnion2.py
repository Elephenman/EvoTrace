#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import paramiko, io
HOST, PORT, USER, PW = "10.202.94.52", 20009, "u22607007", "love1314520YYF"
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30)
sftp = c.open_sftp()
def run(cmd):
    stdin, stdout, stderr = c.exec_command(cmd)
    return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")
print("=== frcmod.ions234lm_1264_tip3p 中 Mn 相关行 ===")
print(run("source ~/md/activate_md.sh >/dev/null 2>&1; grep -iE 'mn' $AMBERHOME/dat/leap/parm/frcmod.ions234lm_1264_tip3p | head -20"))
print("=== atomic_ions.lib 中 Mn 段 ===")
print(run("source ~/md/activate_md.sh >/dev/null 2>&1; grep -n -iE '\"Mn2|MN ' $AMBERHOME/dat/leap/lib/atomic_ions.lib | head -20"))
print("=== tleap: loadoff atomic_ions.lib 后 desc Mn2+ ===")
sftp.putfo(io.StringIO("logFile /tmp/d.log\nloadoff atomic_ions.lib\ndesc Mn2+\nquit\n"), "/home/u22607007/md/ff/probe_mn2.in")
print(run("source ~/md/activate_md.sh >/dev/null 2>&1; cd ~/md/ff && tleap -f probe_mn2.in 2>&1 | head -50"))
c.close()
