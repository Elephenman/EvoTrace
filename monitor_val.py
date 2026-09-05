import paramiko, time, sys
HOST,PORT,USER,PW="10.202.94.52",20009,"u22607007","love1314520YYF"
JOB=225582
def conn():
    c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST,port=PORT,username=USER,password=PW,timeout=25); return c
for i in range(60):  # up to 5h
    time.sleep(240)
    try:
        c=conn()
        _,o,e=c.exec_command(f"sacct -j {JOB} -o State --noheader 2>/dev/null | head -1")
        out=o.read().decode().split()
        state=out[0] if out else '?'
        print(f"[poll {i}] job {JOB} state={state}", flush=True)
        if state in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEM'):
            _,o2,_=c.exec_command(
              f"echo '=== FINAL_RESULTS.dat ==='; cat ~/md/out/WT__S1/FINAL_RESULTS.dat 2>/dev/null | head -50; "
              f"echo '=== tail val log ==='; tail -40 ~/md/log/val_{JOB}.out 2>/dev/null")
            print(o2.read().decode(errors='replace'), flush=True)
            c.close(); break
        # also surface any early failure in md logs
        if state=='RUNNING':
            _,o3,_=c.exec_command(f"tail -5 ~/md/out/WT__S1/prod.mdinfo 2>/dev/null; tail -3 ~/md/out/WT__S1/min1.out 2>/dev/null")
            t=o3.read().decode(errors='replace').strip()
            if t: print("  [running] "+t.replace(chr(10),' | '), flush=True)
        c.close()
    except Exception as ex:
        print("err",repr(ex), flush=True)
print("MONITOR DONE", flush=True)
