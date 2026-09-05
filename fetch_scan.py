# -*- coding: utf-8 -*-
"""Fetch scan_n100_full.py and cells_n100_merged.csv from instance for inspection."""
import paramiko

HOST, PORT, USER, PW = "10.202.94.52", 21114, "u22607007", "love1314520YYF"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30,
          look_for_keys=False, allow_agent=False)
sftp = c.open_sftp()
SC = "/eaas/default/groups/public_cluster/home/u22607007/ppri_evo_boltz_n100/scan_n100_full.py"
sftp.get(SC, "scan_n100_full.py")
print("fetched scan_n100_full.py", __import__("os").path.getsize("scan_n100_full.py"), "bytes")

# also grab the existing n100 merged csv + n100 cells csv to learn output schema
sftp.get("/eaas/default/groups/public_cluster/home/u22607007/ppri_evo_boltz_n100/cells_n100_merged.csv",
         "cells_n100_merged.csv")
sftp.get("/eaas/default/groups/public_cluster/home/u22607007/ppri_evo_boltz_n100/cells_n100.csv",
         "cells_n100.csv")
sftp.get("/eaas/default/groups/public_cluster/home/u22607007/ppri_evo_boltz_n100/contact_fingerprint_n100.csv",
         "contact_fingerprint_n100.csv")
print("fetched csvs")
sftp.close()
c.close()
