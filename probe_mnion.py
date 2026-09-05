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
print("=== ion libs / frcmods (AMBERHOME set) ===")
print(run("source ~/md/activate_md.sh >/dev/null 2>&1; echo AMBERHOME=$AMBERHOME; ls $AMBERHOME/dat/leap/lib/ | grep -iE 'ion|solvent'; echo '--- parm ---'; ls $AMBERHOME/dat/leap/parm/ | grep -iE 'ion'"))
print("=== 含 Mn 的 lib ===")
print(run("source ~/md/activate_md.sh >/dev/null 2>&1; grep -rl 'Mn2' $AMBERHOME/dat/leap/lib/ 2>/dev/null; grep -rl '\"MN\"' $AMBERHOME/dat/leap/lib/ 2>/dev/null; echo 'done'"))
# tleap 探查 Mn2+ 单元
sftp.putfo(io.StringIO("desc Mn2+\nquit\n"), "/home/u22607007/md/ff/probe_mn.in")
print("=== tleap desc Mn2+ ===")
print(run("source ~/md/activate_md.sh >/dev/null 2>&1; cd ~/md/ff && tleap -f probe_mn.in 2>&1 | head -40"))
c.close()
