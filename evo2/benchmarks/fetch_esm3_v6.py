#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sugon ESM3 v6 嵌入回传: 轮询 4198365/4198369 产物并下载 npz 到本地 out/esm3_embeddings_v6/。

用法: python fetch_esm3_v6.py [poll|get]
"""
import os
import sys

import paramiko

KEY = "A:/edge/文献/10.205.1.3_0826123315_rsa.txt"
REMOTE = "/public/home/u22607007/ppri_evo/esm3_embed_v6/out"
LOCAL = "A:/claudework/out/esm3_embeddings_v6"
os.makedirs(LOCAL, exist_ok=True)


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("10.205.1.3", port=10022, username="u22607007",
              key_filename=KEY, timeout=25)
    return c


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "poll"
    c = connect()
    _, o, _ = c.exec_command(f"ls {REMOTE} 2>/dev/null | grep -c npz; squeue -u u22607007 -h -o '%.12i %.2t' | head", timeout=30)
    print(o.read().decode())
    if mode == "get":
        sftp = c.open_sftp()
        files = sftp.listdir(REMOTE)
        n = 0
        for f in files:
            if f.endswith(".npz"):
                dst = os.path.join(LOCAL, f)
                if not os.path.exists(dst):
                    sftp.get(f"{REMOTE}/{f}", dst)
                    n += 1
        print(f"[get] 下载 {n} 个新 npz, 本地共 {len(os.listdir(LOCAL))}")
    c.close()


if __name__ == "__main__":
    main()
