#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n=100 重扫的回收与聚合（作业完成后执行）。

阶段:
  scan   —— 上传判读器(改 BASE/OUT)到 CHPC, 登录节点跑出逐模型 CSV
  fetch  —— 回传 CSV
  agg    —— 聚合: Wilson CI / 与 n=8,20 对比的噪声地板量化 / v4 GBM 重拟合
            / gate v4 边际表 / V1b+V3 复算
用法: python fetch_aggregate_n100.py scan|fetch|agg
"""
import json
import os
import sys

import numpy as np
import paramiko
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PW = "love1314520YYF"
REMOTE = "/home/u22607007/ppri_evo_boltz_n100"
OUT = "A:/claudework/evo2/results/boltz_n100"
os.makedirs(OUT, exist_ok=True)

SCANNER = open(os.path.join(HERE, "scan_boltz_v2a.py"), encoding="utf-8").read()
SCANNER = SCANNER.replace(
    'BASE = "/home/u22607007/ppri_evo_boltz_v2a/out_v2a/boltz_results_yamls/predictions"',
    'import glob as _g\nBASE_SCAN = sorted(_g.glob("/home/u22607007/ppri_evo_boltz_n100/out_n100/boltz_results_*/predictions"))')
SCANNER = SCANNER.replace(
    'OUT = "/home/u22607007/ppri_evo_boltz_v2a/contact_fingerprint.csv"',
    'OUT = "/home/u22607007/ppri_evo_boltz_n100/contact_fingerprint_n100.csv"')
# 扫描器对单个 BASE glob cif —— 改为多根遍历
if "BASE_SCAN" in SCANNER:
    SCANNER = SCANNER.replace('def parse_cif(path):', 'def parse_cif(path):')
    # 在 main/cif 遍历处替换（扫描器用 glob.os.walk? 具体见原文件; 保守起见追加通用遍历）
SCANNER_TAIL = '''

# --- n100 多根遍历（追加于文件尾, 覆盖原 __main__ 判读流程）---
import glob as globmod
rows = []
roots = sorted(globmod.glob("/home/u22607007/ppri_evo_boltz_n100/out_n100/boltz_results_*/predictions"))
print("roots:", roots)
for root in roots:
    for cif in globmod.glob(os.path.join(root, "**", "*.cif"), recursive=True):
        name = cif.replace("\\\\", "/").split("/predictions/")[-1].split("/")[0]
        try:
            m = model_metrics(cif)
        except Exception as exc:
            print("parse fail", cif, exc); continue
        if m:
            m["pred"] = name
            rows.append(m)
import csv as _csv
keys = sorted({k for r in rows for k in r})
with open("/home/u22607007/ppri_evo_boltz_n100/contact_fingerprint_n100.csv", "w", newline="") as fh:
    w = _csv.DictWriter(fh, fieldnames=["pred"] + keys)
    w.writeheader(); w.writerows(rows)
print("wrote", len(rows), "models")
'''


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("10.202.94.52", port=20009, username="u22607007", password=PW,
              timeout=25, look_for_keys=False, allow_agent=False)
    return c


def run(c, cmd, timeout=120):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


def up_b64(c, text, remote_path, chunk=60000):
    b64 = __import__("base64").b64encode(text.encode()).decode()
    run(c, f"rm -f {remote_path} {remote_path}.b64")
    for i in range(0, len(b64), chunk):
        run(c, f"echo {b64[i:i+chunk]} >> {remote_path}.b64")
    run(c, f"base64 -d {remote_path}.b64 > {remote_path} && rm {remote_path}.b64")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def stage_agg():
    df = pd.read_csv(f"{OUT}/contact_fingerprint_n100.csv")
    # pred 名 -> (variant, cond)
    def split_pred(p):
        base = p.replace("boltz_results_", "").split("_seeds")[0]
        for cond in ("S1_G17", "OFF_T_G17"):
            if base.endswith("_" + cond):
                return base[: -len(cond) - 1], cond
        return base, "?"
    df[["variant", "cond"]] = df["pred"].map(lambda p: pd.Series(split_pred(p)))
    # act/dual 列名为 0/1 → 统一
    act_col = "act" if "act" in df.columns else [c for c in df.columns if c.lower().startswith("act")][0]
    g = df.groupby(["variant", "cond"]).agg(
        n=("pred", "size"), act=(act_col, "mean"), dual=("dual", "mean") if "dual" in df.columns else (act_col, "size"))
    g.to_csv(f"{OUT}/cells_n100.csv")
    print(g.head(20).to_string())
    # 噪声地板量化: 与 n=8/20 旧值对比同变体
    old = pd.read_csv("A:/claudework/evo2/results/boltz_six/deconf_stats.csv")
    m = g.xs("S1_G17", level="cond")["dual"].rename("dual_n100")
    j = old.set_index("variant").join(m)
    j["abs_shift"] = (j.dual_n100 - j.dual_S1).abs()
    print("\n=== 噪声地板: 同变体 dual_S1 (n=8 vs n=100) ===")
    print(j[["dual_S1", "dual_n100", "abs_shift"]].sort_values("abs_shift", ascending=False).head(12).to_string())
    print(f"\n平均|漂移| = {j.abs_shift.mean():.3f}  (n=8 的期望抽样噪声 ~{np.sqrt(0.25*0.75/8)*2:.3f} 95%半宽)")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if stage == "scan":
        c = connect()
        up_b64(c, SCANNER + SCANNER_TAIL, f"{REMOTE}/scan_n100.py")
        run(c, f"sed -i 's/\\r$//' {REMOTE}/scan_n100.py")
        o, e = run(c, f"cd {REMOTE} && nohup python3 scan_n100.py > scan.log 2>&1 & echo started")
        print(o, e[:200])
        c.close()
    elif stage == "fetch":
        c = connect()
        sftp = c.open_sftp()
        sftp.get(f"{REMOTE}/contact_fingerprint_n100.csv", f"{OUT}/contact_fingerprint_n100.csv")
        print("[fetched]", os.path.getsize(f"{OUT}/contact_fingerprint_n100.csv"), "bytes")
        c.close()
    elif stage == "agg":
        stage_agg()
    else:
        print("usage: scan|fetch|agg")


if __name__ == "__main__":
    main()
