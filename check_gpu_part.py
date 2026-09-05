#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_gpu_part.py — 查 gpu* 分区 GPU 型号与空闲情况。"""
import paramiko
HOST, PORT, USER, PW = "10.202.94.52", 20009, "u22607007", "love1314520YYF"
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30)
def run(cmd):
    stdin, stdout, stderr = c.exec_command(cmd)
    return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")
print("=== gpu* 分区节点与 GRES ===")
print(run("sinfo -p gpu -N -o '%n %G %m %t' 2>&1"))
print("=== 4090 分区完整信息 ===")
print(run("sinfo -p 4090 -N -o '%n %G %m %t' 2>&1"))
print("=== 空闲 GPU 资源(squeue 占用) ===")
print(run("squeue -p 4090 -o '%i %u %t %C %b' 2>&1 | head; echo '--- gpu* ---'; squeue -p gpu -o '%i %u %t %C %b' 2>&1 | head"))
print("=== 探测 gpu1 节点 GPU 型号(若空闲, 用 srun 短暂占用) ===")
print(run("sinfo -p gpu -N -o '%n' 2>&1 | tail -n +2 | head -1 | xargs -I{} sh -c 'echo NODE={}; srun -p gpu -w {} --gres=gpu:1 -N1 -n1 --time=02:00 nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1 | head' "))
c.close()
