# -*- coding: utf-8 -*-
"""CHPC 独立 4090 实例连接器 (port 21114)。

仅用于连接独立实例，不碰登录节点 SLURM(20009)/ssh_run.py。
用法:
  python probe_instance.py            # 交互式跑 health check
  from probe_instance import connect, run, up_text
"""
import base64
import paramiko

HOST = "10.202.94.52"
PORT = 21114
USER = "u22607007"
PW = "love1314520YYF"


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.WarningPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PW,
              timeout=30, look_for_keys=False, allow_agent=False)
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


if __name__ == "__main__":
    c = connect()
    print("[connected] instance 21114")
    o, e = run(c, "echo OK; hostname; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; "
                 "ls /home/u22607007/ppri_evo_boltz_nom2/out_nom2_full/ | wc -l; "
                 "ls /eaas/default/groups/public_cluster/home/u22607007/ppri_evo_boltz_n100/scan_n100_full.py")
    print(o.strip())
    if e.strip():
        print("[stderr]", e.strip()[:500])
    c.close()
