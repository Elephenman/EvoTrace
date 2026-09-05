#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_cluster.py — 诊断 GPU 集群 ~/md 部署状态与资源上限。"""
import paramiko, sys
HOST, PORT, USER, PW = "10.202.94.52", 20009, "u22607007", "love1314520YYF"
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30)
def run(cmd):
    stdin, stdout, stderr = c.exec_command(cmd)
    return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")

print("=== ~/md 顶层 ===")
print(run("ls -la ~/md/ 2>&1; echo '--- inputs/six ---'; ls ~/md/inputs/six/ 2>&1 | head; echo '--- ff ---'; ls ~/md/ff/ 2>&1"))
print("=== activate_md.sh 是否存在 ===")
print(run("test -f ~/md/activate_md.sh && echo EXISTS || echo MISSING"))
print("=== 若缺失， Amber 实际路径 ===")
print(run("ls -d /opt/app/amber/* 2>&1; which pmemd.cuda tleap MMPBSA.py 2>&1; ls /opt/app/amber/24/bin/pmemd.cuda 2>&1"))
print("=== 分区与最大墙时 ===")
print(run("sinfo -a -o '%P %l %t %N' 2>&1 | head -30"))
print("=== 4090 分区节点 ===")
print(run("sinfo -p 4090 -N -o '%n %G %m' 2>&1 | head"))
print("=== 账户配额/状态 ===")
print(run("sacctmgr show assoc account=ls_lhz format=Account,User,GrpTRES,MaxTRES 2>&1 | head; echo '--- sacct balance ---'; sacct --version 2>&1"))
print("=== 当前我的作业 ===")
print(run("squeue -u $USER 2>&1 | head -20"))
print("=== 测试 source activate ===")
print(run("source ~/md/activate_md.sh 2>/dev/null; echo AMBERHOME=$AMBERHOME; which pmemd.cuda 2>&1"))
c.close()
