#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import paramiko, io
HOST, PORT, USER, PW = "10.202.94.52", 20009, "u22607007", "love1314520YYF"
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30)
sftp = c.open_sftp()
mk = """source leaprc.gaff2
mnA = createAtom "MN" "MN" 2.0
mnU = createUnit "MN"
add mnU mnA
saveoff mnU mn.off
desc mnU
quit
"""
sftp.putfo(io.StringIO(mk), "/home/u22607007/md/ff/mk_mn.off")
stdin, stdout, stderr = c.exec_command("source ~/md/activate_md.sh >/dev/null 2>&1; cd ~/md/ff && tleap -f mk_mn.off 2>&1; echo '--- ls ---'; ls -la mn.off 2>&1")
print(stdout.read().decode(errors="replace"))
print(stderr.read().decode(errors="replace"))
c.close()
