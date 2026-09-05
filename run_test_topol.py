#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import paramiko
HOST, PORT, USER, PW = "10.202.94.52", 20009, "u22607007", "love1314520YYF"
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30)
sftp = c.open_sftp()
sftp.put(r"A:\Data\设计蛋白\PprI_ssDNA_design\md_specificity\ff\mn_cm12-6.frcmod", "/home/u22607007/md/ff/mn_cm12-6.frcmod")
sftp.put(r"A:\Data\设计蛋白\PprI_ssDNA_design\md_specificity\run_system.sh", "/home/u22607007/md/run_system.sh")
sftp.put(r"A:\claudework\test_topol.sh", "/home/u22607007/md/test_topol.sh")
print("uploaded frcmod + run_system.sh + test_topol.sh")
stdin, stdout, stderr = c.exec_command("rm -f ~/md/ff/mn.off; bash ~/md/test_topol.sh 2>&1")
out = stdout.read().decode(errors="replace")
err = stderr.read().decode(errors="replace")
print("=== STDOUT ===\n" + out)
if err.strip(): print("=== STDERR ===\n" + err)
c.close()
