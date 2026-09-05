import paramiko, sys
HOST, PORT, USER, PW = "10.202.94.52", 20009, "u22607007", "love1314520YYF"
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect(HOST, port=PORT, username=USER, password=PW, timeout=25)
except Exception as e:
    print("CONNECT_FAIL", repr(e)); sys.exit(1)
def run(cmd):
    stdin, stdout, stderr = c.exec_command(cmd)
    return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")
print("=== host ==="); print(run("hostname"))
print("=== sbatch test-only (4090/gpu:1) ==="); print(run("sbatch --test-only --account=ls_lhz --partition=4090 --gres=gpu:1 --wrap='echo hi' 2>&1 | head -5"))
print("=== tools ==="); print(run("source ~/md/activate_md.sh 2>/dev/null; which pmemd.cuda tleap gmx_MMPBSA mpirun; echo AMBERHOME=$AMBERHOME"))
print("=== partitions ==="); print(run("sinfo -s 2>&1 | head -20"))
c.close()
