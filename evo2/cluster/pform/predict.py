#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pform — predict a protein's final state from input Boltz yamls.

End-to-end CLI (protein-agnostic):
    pform predict <yamls_dir> [--samples 20] [--seeds 1] [--job NAME]
                  [--wait] [--timeout-min 480]
    pform status [--job NAME]
    pform metrics <remote_out_dir> --job NAME      # recompute metrics only

Flow: upload yamls+sbatch (LF-normalized) -> sbatch CHPC 4090 Boltz-2 ->
(wait) -> run pluggable metric script remotely -> pull verdict CSV ->
local per-candidate summary (dual rate / median distances / final-state
verdicts).

Env: CHPC_PASS (password). Cluster: 10.202.94.52:20009 u22607007.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from chpc import Chpc
import metrics_dual_lock as mdl

REMOTE_BASE = "/home/u22607007/pform_jobs"
SBATCH_TMPL = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --partition=4090
#SBATCH --comment=u22607007
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err

mkdir -p {remote}/logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate boltz
export NVIDIA_LIB=/opt/app/nvidia/570.195.03/lib
export LD_LIBRARY_PATH=$NVIDIA_LIB:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com

cd {remote}
echo "===== pform {job} $(date) ====="
for SEED in {seeds}; do
  boltz predict inputs \\
    --out_dir out_s$SEED \\
    --seed $SEED \\
    --diffusion_samples {samples} \\
    --recycling_steps 3 \\
    --sampling_steps 200 \\
    --use_msa_server \\
    --override \\
    --no_trifast \\
    --cache ~/.boltz 2>&1 | tail -8
done
echo "===== PFORM_DONE $(date) ====="
"""


def remote_job_dir(job):
    return f"{REMOTE_BASE}/{job}"


def cmd_predict(args):
    ydir = os.path.abspath(args.yamls)
    yamls = sorted(f for f in os.listdir(ydir) if f.endswith(".yaml"))
    if not yamls:
        raise SystemExit(f"no yaml in {ydir}")
    seeds = " ".join(str(s) for s in args.seeds)
    remote = remote_job_dir(args.job)
    sbatch = SBATCH_TMPL.format(job=args.job, remote=remote, seeds=seeds,
                                samples=args.samples)
    sb_local = os.path.join(ydir, f"pform_{args.job}.sbatch")
    with open(sb_local, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(sbatch)

    with Chpc() as c:
        print(f"[up] {len(yamls)} yamls -> {remote}/inputs")
        c.run(f"rm -rf {remote} && mkdir -p {remote}/inputs {remote}/logs")
        # optional binding-state template: upload crystal/reference CIF and
        # inject a `templates` section into every input yaml
        tmpl_remote = None
        if getattr(args, "template", None):
            import shutil
            import yaml as _yaml
            tmpl_dir = f"{remote}/tmpl"
            c.run(f"mkdir -p {tmpl_dir}")
            tmpl_remote = f"{tmpl_dir}/{os.path.basename(args.template)}"
            c.put(args.template, tmpl_remote)
            for y in yamls:
                lp = os.path.join(ydir, y)
                doc = _yaml.safe_load(open(lp, encoding="utf-8"))
                doc.pop("constraints", None)   # broken in boltz 2.0.3
                doc["templates"] = [{"cif": tmpl_remote,
                                     "chain_id": getattr(args, "template_chain", "A")}]
                tmp = lp + ".tmpl"
                with open(tmp, "w", encoding="utf-8") as fh:
                    _yaml.safe_dump(doc, fh, allow_unicode=True, sort_keys=False)
                c.put(tmp, f"{remote}/inputs/{y}")
                os.remove(tmp)
            print(f"[tmpl] injected {os.path.basename(args.template)} "
                  f"(chain {getattr(args, 'template_chain', 'A')}) into {len(yamls)} yamls")
        else:
            for y in yamls:
                c.put(os.path.join(ydir, y), f"{remote}/inputs/{y}")
        c.put(sb_local, f"{remote}/job.sbatch")
        out, err = c.run(f"cd {remote} && sbatch job.sbatch")
        print("[sbatch]", out.strip(), err.strip()[:200])
        jobid = None
        import re
        m = re.search(r"(\d+)", out)
        if m:
            jobid = m.group(1)

    if args.wait:
        if jobid is None:
            raise SystemExit("no jobid to wait on")
        wait_job(jobid, args.job, args.timeout_min)
        cmd_metrics(args)


def wait_job(jobid, job, timeout_min):
    remote = remote_job_dir(job)
    print(f"[wait] job {jobid} (timeout {timeout_min} min)...")
    t0 = time.time()
    with Chpc() as c:
        while True:
            out, _ = c.run(f"squeue -j {jobid} 2>/dev/null | tail -n +2")
            done_mark, _ = c.run(f"grep -l PFORM_DONE {remote}/logs/*.out 2>/dev/null | head -1")
            n_cif, _ = c.run(f"find {remote} -name '*.cif' 2>/dev/null | wc -l")
            if done_mark.strip() or (not out.strip()):
                if done_mark.strip():
                    if int(n_cif.strip() or 0) > 0:
                        print(f"[done] PFORM_DONE found; cif={n_cif.strip()}")
                    else:
                        errout, _ = c.run(f"tail -10 {remote}/logs/*.out 2>/dev/null | tail -10")
                        print("[warn] PFORM_DONE with 0 cif (likely failed)")
                        print(errout[-900:])
                    return
                # job left queue but no DONE mark -> check error
                errout, _ = c.run(f"tail -6 {remote}/logs/*.err 2>/dev/null | tail -6")
                print("[warn] job left queue without DONE mark")
                print(errout[-800:])
                return
            if time.time() - t0 > timeout_min * 60:
                print(f"[timeout] still running after {timeout_min} min (cif={n_cif.strip()})")
                return
            print(f"[poll {int(time.time()-t0)}s] queued={bool(out.strip())} cif={n_cif.strip()}")
            time.sleep(60)


def cmd_metrics(args):
    remote = remote_job_dir(args.job)
    script = mdl.build_remote_script()
    import base64
    import csv as _csv
    seeds = getattr(args, "seeds", [1])
    local_csvs = []
    with Chpc() as c:
        c.run(f"mkdir -p {remote}/metrics")
        b64 = base64.b64encode(script.encode()).decode()
        c.run(f"echo {b64} | base64 -d > {remote}/metrics/run_metrics.py")
        for s in seeds:
            outdir = f"out_s{s}"
            rcsv = f"metrics/models_s{s}.csv"
            out, err = c.run(
                f"cd {remote} && python3 metrics/run_metrics.py {outdir} {rcsv} 2>&1")
            print(f"[seed {s}]", out.strip(), err.strip()[:200])
            lcsv = args.out_csv.replace(".csv", f"_s{s}.csv")
            c.get(f"{remote}/{rcsv}", lcsv)
            # tag rows with seed
            rows = list(_csv.DictReader(open(lcsv, encoding="utf-8")))
            for r in rows:
                r["seed"] = s
            if rows:
                with open(lcsv, "w", newline="", encoding="utf-8") as fh:
                    w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            local_csvs.append(lcsv)
    print(f"[pulled] {len(local_csvs)} seed csvs")
    if len(local_csvs) == 1:
        summarize(local_csvs[0], getattr(args, "truth", None))
    else:
        summarize_multi(local_csvs, getattr(args, "truth", None))


def summarize(csv_path, truth_path=None):
    import csv as _csv
    import json as _json
    from collections import defaultdict
    rows = list(_csv.DictReader(open(csv_path, encoding="utf-8")))
    truth = {}
    if truth_path and os.path.exists(truth_path):
        truth = _json.load(open(truth_path, encoding="utf-8"))
    by = defaultdict(list)
    for r in rows:
        key = r["model"].split("_model_")[0]
        by[key].append(r)
    print("\n=== per-candidate final-state summary ===")
    summary = {}
    for cand, rs in sorted(by.items()):
        n = len(rs)
        dual = sum(1 for r in rs if r.get("dual") == "1")
        d1 = sorted(float(r["d_P1"]) for r in rs if r.get("d_P1") and r["d_P1"] != "nan")
        d2 = sorted(float(r["d_P2"]) for r in rs if r.get("d_P2") and r["d_P2"] != "nan")
        med1 = d1[len(d1) // 2] if d1 else float("nan")
        med2 = d2[len(d2) // 2] if d2 else float("nan")
        summary[cand] = {"n": n, "dual_rate": round(dual / n, 3),
                         "d_P1_med": round(med1, 2), "d_P2_med": round(med2, 2),
                         "verdict": "FINAL_LOCKED" if dual / n >= 0.5 else "NOT_LOCKED"}
        line = (f"  {cand:>28}: dual={dual}/{n} ({dual/n:.2f})  d_P1 med={med1:5.1f}  "
                f"d_P2 med={med2:5.1f}  -> {summary[cand]['verdict']}")
        # calibration against known experimental truth
        t = truth.get(cand) or truth.get(cand.split("_S1")[0])
        if t and t.get("expected_dual") is not None:
            exp = float(t["expected_dual"])
            pred = dual / n
            bias = round(pred - exp, 3)
            factor = round(exp / pred, 2) if pred > 0 else None
            summary[cand]["calibration"] = {
                "expected_dual": exp, "bias": bias,
                "correction_factor": factor,
                "note": t.get("note", "")}
            line += (f"  | CAL exp={exp:.2f} bias={bias:+.2f}"
                     + (f" factor={factor}" if factor else " (pred=0)")
                     + (f" [{t.get('note','')}]" if t.get("note") else ""))
        else:
            summary[cand]["calibration"] = {"expected_dual": None, "note": "uncalibrated (no truth)"}
        print(line)
    with open(csv_path.replace(".csv", "_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(f"[saved] {csv_path.replace('.csv', '_summary.json')}")


def summarize_multi(csv_paths, truth_path=None):
    """Cross-seed summary: per candidate, dual rate per seed + median."""
    import csv as _csv
    import json as _json
    from collections import defaultdict
    truth = {}
    if truth_path and os.path.exists(truth_path):
        truth = _json.load(open(truth_path, encoding="utf-8"))
    per = defaultdict(lambda: defaultdict(list))  # cand -> seed -> dual(0/1)
    for cp in csv_paths:
        seed = os.path.basename(cp).split("_s")[-1].split(".")[0]
        for r in _csv.DictReader(open(cp, encoding="utf-8")):
            cand = r["model"].split("_model_")[0]
            per[cand][seed].append(int(r.get("dual") == "1"))
    print("\n=== per-candidate cross-seed final-state summary ===")
    summary = {}
    for cand in sorted(per):
        rates = {s: round(sum(v) / len(v), 3) for s, v in per[cand].items()}
        rv = sorted(rates.values())
        med = rv[len(rv) // 2]
        summary[cand] = {"rates_by_seed": rates, "median_dual": med,
                         "verdict": "FINAL_LOCKED" if med >= 0.5 else "NOT_LOCKED"}
        line = (f"  {cand:>28}: seeds={rates}  med={med:.2f}  -> {summary[cand]['verdict']}")
        t = truth.get(cand) or truth.get(cand.split("_S1")[0])
        if t and t.get("expected_dual") is not None:
            exp = float(t["expected_dual"])
            bias = round(med - exp, 3)
            factor = round(exp / med, 2) if med > 0 else None
            summary[cand]["calibration"] = {"expected_dual": exp, "bias": bias,
                                            "correction_factor": factor,
                                            "note": t.get("note", "")}
            line += (f"  | CAL exp={exp:.2f} bias={bias:+.2f}"
                     + (f" factor={factor}" if factor else " (med=0)"))
        else:
            summary[cand]["calibration"] = {"expected_dual": None, "note": "uncalibrated"}
        print(line)
    out = csv_paths[0].replace("_s1.csv", "_summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser(prog="pform", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="upload, submit, wait, metrics")
    p.add_argument("yamls")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--job", default="pform_job")
    p.add_argument("--template", default=None,
                   help="binding-state template CIF (e.g. crystal holo complex); "
                        "injected as Boltz `templates` into every input yaml")
    p.add_argument("--template-chain", default="A",
                   help="template chain id to map onto the input protein (default A)")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--timeout-min", type=int, default=480)
    p.add_argument("--out-csv", default=None)
    p.set_defaults(fn=cmd_predict)

    s = sub.add_parser("status", help="queue status")
    s.add_argument("--job", default=None)
    s.set_defaults(fn=lambda a: status_cmd(a))

    m = sub.add_parser("metrics", help="recompute metrics on existing out dir")
    m.add_argument("--job", default="pform_job")
    m.add_argument("--seeds", type=int, nargs="+", default=[1])
    m.add_argument("--out-csv", default=None)
    m.add_argument("--truth", default=None,
                   help="calibration json: {cand: {expected_dual, note}}")
    m.set_defaults(fn=cmd_metrics)

    c = sub.add_parser("calibrate", help="calibrate a models.csv against known truth")
    c.add_argument("models_csv")
    c.add_argument("--truth", required=True,
                   help="json: {cand_pattern: {expected_dual: 0-1, note}}")
    c.set_defaults(fn=lambda a: summarize(a.models_csv, a.truth))

    args = ap.parse_args()
    oc = getattr(args, "out_csv", None)
    if oc is None and args.cmd in ("predict", "metrics"):
        args.out_csv = os.path.join("A:/claudework/evo2/results",
                                    os.path.basename(remote_job_dir(args.job)) + "_models.csv")
    args.fn(args)


def status_cmd(args):
    with Chpc() as c:
        out, _ = c.run("squeue -u u22607007")
        print(out)
        if args.job:
            r = remote_job_dir(args.job)
            n, _ = c.run(f"find {r} -name '*.cif' 2>/dev/null | wc -l")
            print(f"job {args.job}: cif={n.strip()}")


if __name__ == "__main__":
    main()
