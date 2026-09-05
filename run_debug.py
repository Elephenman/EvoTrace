#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import paramiko
HOST, PORT, USER, PW = "10.202.94.52", 20009, "u22607007", "love1314520YYF"
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30)
def run(cmd):
    stdin, stdout, stderr = c.exec_command(cmd)
    return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")
print("=== md 环境 parmed 路径 ===")
print(run("ls -la /home/u22607007/miniconda3/envs/md/bin/parmed 2>&1; echo '--- activate 后 which parmed ---'; source ~/md/activate_md.sh >/dev/null 2>&1; which parmed; echo '--- md parmed 自测 ---'; /home/u22607007/miniconda3/envs/md/bin/parmed -i ~/md/topol_test/WT__S1/rec.in 2>&1 | tail -15"))
print("=== 用 md parmed 跑 lig/cp ===")
print(run("source ~/md/activate_md.sh >/dev/null 2>&1; /home/u22607007/miniconda3/envs/md/bin/parmed -i ~/md/topol_test/WT__S1/lig.in 2>&1 | tail -8; /home/u22607007/miniconda3/envs/md/bin/parmed -i ~/md/topol_test/WT__S1/cp.in 2>&1 | tail -8"))
print("=== 写 dbg_mn.py 并算 Mn 配位 ===")
mk = r'''
import parmed as pmd
p=pmd.load_file("/home/u22607007/md/topol_test/WT__S1/vWT__S1.parm7","/home/u22607007/md/topol_test/WT__S1/vWT__S1.rst7")
mn=[a for a in p.atoms if a.name=="MN"][0]
print("Mn coord:", round(mn.xx,2), round(mn.xy,2), round(mn.xz,2))
dists=[]
for a in p.atoms:
    if a.residue.name in ("HIS","GLU") and a.name in ("NE2","ND1","OE1","OE2"):
        d=((a.xx-mn.xx)**2+(a.xy-mn.xy)**2+(a.xz-mn.xz)**2)**0.5
        dists.append((round(d,2), a.residue.name, a.residue.idx+1, a.name))
dists.sort()
print("最近 8 个 HIS/GLU 配位候选:")
for d in dists[:8]: print("  ", d)
print("HEXXH(H92/H96/E123)距离:")
for a in p.atoms:
    if a.residue.name in ("HIS","GLU") and a.residue.idx+1 in (92,96,123) and a.name in ("NE2","ND1","OE1","OE2"):
        dd=((a.xx-mn.xx)**2+(a.xy-mn.xy)**2+(a.xz-mn.xz)**2)**0.5
        print("  ", a.residue.name, a.residue.idx+1, a.name, round(dd,2))
'''
stdin, stdout, stderr = c.exec_command(f"cat > ~/md/topol_test/WT__S1/dbg_mn.py <<'PYEOF'\n{mk}PYEOF\n")
print("write rc:", stdout.read().decode(errors='replace'), stderr.read().decode(errors='replace'))
print(run("source ~/md/activate_md.sh >/dev/null 2>&1; cd ~/md/topol_test/WT__S1 && python3 dbg_mn.py 2>&1"))
c.close()
