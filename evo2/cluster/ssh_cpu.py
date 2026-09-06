#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ssh_cpu.py — CPU 集群(10.205.1.3:10022)远程执行 + 文件拉取/推送（密钥认证）。
用法:
  python ssh_cpu.py "cmd"            # 执行远程命令
  python ssh_cpu.py pull r1 r2 ...   # 拉远端文件到 cwd/mirror/
  python ssh_cpu.py push l r         # 上传本地文件到远端绝对路径
"""
import os
import sys
import posixpath
import paramiko

HOST, PORT, USER = "10.205.1.3", 10022, "u22607007"
KEY = r"A:/edge/文献/10.205.1.3_0826123315_rsa.txt"


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, key_filename=KEY,
              timeout=30, look_for_keys=False, allow_agent=False)
    return c


def run(c, cmd, timeout=900):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    if out.strip():
        print(out)
    if err.strip():
        print("[stderr]", err[:3000])
    print(f"==== exit {rc} ====")
    return rc


def pull(c, remote_files):
    sftp = c.open_sftp()
    for rp in remote_files:
        fn = posixpath.basename(rp)
        local = os.path.join("mirror", fn)
        os.makedirs("mirror", exist_ok=True)
        sftp.get(rp, local)
        print(f"[pull] {rp} -> {local}")
    sftp.close()


def push(c, local, remote):
    sftp = c.open_sftp()
    rdir = posixpath.dirname(remote)
    acc = "/" if rdir.startswith("/") else ""
    for seg in rdir.split("/"):
        if not seg:
            continue
        acc = posixpath.join(acc, seg)
        try:
            sftp.stat(acc)
        except FileNotFoundError:
            sftp.mkdir(acc)
            print(f"[mkdir] {acc}")
    sftp.put(local, remote)
    print(f"[push] {local} -> {remote}")
    sftp.close()


if __name__ == "__main__":
    a = sys.argv[1:]
    c = connect()
    try:
        if not a or a[0] == "cmd":
            sys.exit(run(c, a[1] if len(a) > 1 else "hostname"))
        if a[0] == "pull":
            sys.exit(pull(c, a[1:]) or 0)
        if a[0] == "push":
            sys.exit(push(c, a[1], a[2]) or 0)
        raise SystemExit(f"未知子命令 {a[0]}")
    finally:
        c.close()
