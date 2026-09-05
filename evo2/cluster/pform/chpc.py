# -*- coding: utf-8 -*-
"""pform.chpc — CHPC SSH/SFTP/command client (paramiko), password via env
CHPC_PASS only (never persisted)."""
import os

import paramiko

HOST = "10.202.94.52"
PORT = 20009
USER = "u22607007"


class Chpc:
    def __init__(self, password=None):
        self.pw = password or os.environ.get("CHPC_PASS")
        if not self.pw:
            raise SystemExit("CHPC_PASS env not set")
        self.c = paramiko.SSHClient()
        self.c.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def __enter__(self):
        self.c.connect(HOST, port=PORT, username=USER, password=self.pw,
                       timeout=30, look_for_keys=False, allow_agent=False)
        return self

    def __exit__(self, *a):
        self.c.close()

    def run(self, cmd, timeout=180):
        _, o, e = self.c.exec_command(cmd, timeout=timeout)
        out = o.read().decode("utf-8", errors="replace")
        err = e.read().decode("utf-8", errors="replace")
        return out, err

    def put(self, local, remote):
        """SFTP put with base64 fallback (CHPC SFTP occasionally ENOENT)."""
        try:
            sftp = self.c.open_sftp()
            sftp.put(local, remote)
            sftp.close()
        except Exception:
            import base64
            b64 = base64.b64encode(open(local, "rb").read()).decode()
            self.run(f"echo {b64} | base64 -d > {remote}")
        self.run(f"sed -i 's/\\r$//' {remote} 2>/dev/null || true")

    def get(self, remote, local):
        try:
            sftp = self.c.open_sftp()
            sftp.get(remote, local)
            sftp.close()
        except Exception:
            out, _ = self.run(f"base64 {remote}")
            import base64
            with open(local, "wb") as fh:
                fh.write(base64.b64decode(out.strip()))
