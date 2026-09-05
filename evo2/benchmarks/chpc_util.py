# -*- coding: utf-8 -*-
"""CHPC SSH 公共工具（杜绝 run(c,...)/run(...) 混用 bug）。"""
import base64

import paramiko

PW = "love1314520YYF"


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("10.202.94.52", port=20009, username="u22607007", password=PW,
              timeout=25, look_for_keys=False, allow_agent=False)
    return c


def run(c, cmd, timeout=120):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


def up_text(c, text_or_path, remote_path, is_path=True, chunk=60000):
    raw = open(text_or_path, "rb").read() if is_path else text_or_path.encode()
    b64 = base64.b64encode(raw).decode()
    run(c, f"rm -f {remote_path} {remote_path}.b64")
    for i in range(0, len(b64), chunk):
        run(c, f"echo {b64[i:i+chunk]} >> {remote_path}.b64")
    run(c, f"base64 -d {remote_path}.b64 > {remote_path} && rm {remote_path}.b64 && "
           f"sed -i 's/\\r$//' {remote_path}")


def queue(c):
    o, _ = run(c, "squeue -u u22607007 -h -o '%.12i %.12j %.2t %.8M'")
    return o.strip()
