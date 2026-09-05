#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evotrace — 数字定向进化工具 CLI（EvoTrace v2 产品入口）。

子命令：
  evolve    <landscape> [--rounds N] [--labels N] [--Ne N] [--seed S]
            在指定景观上跑进化 campaign（Wright-Fisher + 漏斗 + 标签回流），
            输出精英变体 CSV + 进化轨迹 CSV + 汇总。
  benchmark <name> [--quick]     一键复现基准（b1a/b1b/b2/b3/b4）
  confirm   <elites.csv> [--boltz [--template CIF] [--seeds 1 2 3]]
            对精英候选做确认（可选 Boltz-2 结构确认，调 pform 层）。

内置景观：avGFP（51,714 变体实测景观，ρ 对标 ESM3）。
通用用法：evolve 也接受 --csv <landscape CSV>（列: mutant,DMS_score）+ --wt <序列>。

运行示例：
  python evotrace.py evolve avGFP --rounds 2 --labels 48 --seed 11
  python evotrace.py benchmark b1a --quick
  python evotrace.py confirm results/elites.csv --boltz --template 8SLN.cif --seeds 1 2 3
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

E = "A:/claudework/evo_data/processed/proteingym_benchmark"
DMS_AVGFP = os.path.join(E, "DMS_ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv")
DMS_GB1 = os.path.join(E, "DMS_ProteinGym_substitutions/GB1_2F4K_Wu_2016.csv")

LANDSCAPES = {
    "avGFP": {"dms": DMS_AVGFP, "wt": "benchmarks/data_avgfp_wt.json",
              "priors": "benchmarks/data/priors_avgfp.csv", "Ne": 300,
              "n_gen": 8, "n_pop": 4, "batch": 24, "sites": None},
    "GB1": {"dms": DMS_GB1, "wt": "benchmarks/data_gb1_wt.json",
            "priors": "benchmarks/data/priors_gb1.csv", "Ne": 300,
            "n_gen": 8, "n_pop": 4, "batch": 24, "sites": None},
    "PprI": {"agentic": True, "wt_fasta": "A:/claudework/ppri_evo/inputs/wt_254.fasta",
             "priors": "A:/claudework/ppri_evo/results/priors.csv",
             "Ne": 500, "n_gen": 15, "n_pop": 4},
}


def load_landscape(name, csv=None, wt_fasta=None):
    """Return (wt_seq, sites, prior_dict, oracle_map, extras).

    extras: {"ddg": (L,20), "anchors": {site: family}, "src": ...} for
    agentic (no-experiment) landscapes like PprI; {} for measured ones.
    """
    import pandas as pd
    from engine.seqtools import parse_mutant
    from engine.priors import load_priors_csv
    if csv is None and LANDSCAPES[name].get("agentic"):
        # ---- PprI-style agentic landscape: fasta WT + priors + ddg proxy ----
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "b5", os.path.join(HERE, "benchmarks", "b5_ppri_wave2.py"))
        b5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(b5)
        from engine.seqtools import read_fasta, AA2IDX
        wt = list(read_fasta(LANDSCAPES[name]["wt_fasta"]).values())[0]
        pri, cls, wt_aa = b5.load_ppri_priors()
        sites = sorted(pri.keys())
        ddg, src = b5.build_ddg_tables(wt, sites, cls, wt_aa)
        anchors = {}
        for s in b5.ANCHOR_SITES:
            j = s - b5.PDB_OFFSET
            if j in sites:
                anchors[j] = sorted(b5.ANCHOR_FAMILY)
        return wt, sites, pri, {}, {"ddg": ddg, "anchors": anchors,
                                    "TAU": b5.TAU, "W_STAB": b5.W_STAB}
    if csv is None:
        cfg = LANDSCAPES[name]
        csv, wt_json = cfg["dms"], os.path.join(HERE, cfg["wt"])
        wt = json.load(open(wt_json, encoding="utf-8"))["wt"]
        pri, wt_aa, cls = load_priors_csv(os.path.join(HERE, cfg["priors"]))
    else:
        wt = wt_fasta
        pri = {}  # no structural priors for custom landscape
    df = pd.read_csv(csv)
    oracle_map = {}
    for m, y in zip(df.mutant, df.DMS_score):
        muts = parse_mutant(m)
        if muts:
            oracle_map[tuple(sorted(muts))] = float(y)
    sites = sorted({i for m in oracle_map for i, _ in m})
    return wt, sites, pri, oracle_map, {}


def cmd_evolve(args):
    from engine.kernel import WFKernel
    from engine.funnel import OracleLandscape, run_campaign
    wt, sites, pri, omap, extras = load_landscape(args.landscape, args.csv, args.wt)
    print(f"[landscape] {args.landscape or 'custom'}  wt={len(wt)}aa  "
          f"sites={len(sites)}  measured={len(omap)}  agentic={bool(extras)}")
    if extras:
        return cmd_evolve_agentic(args, wt, sites, pri, extras)
    oracle = OracleLandscape(wt, omap)
    cfg = dict(Ne=args.Ne, n_gen=args.n_gen, n_pop=args.n_pop,
               batch=args.batch, rounds=args.rounds)
    kcfg = dict(mutations_per_genome_per_gen={"lambda": 2.0}, n_mut_max=10, T=0.6,
                w_stab=0.0, tau_stab=2.0, proposal_temp=2.0)
    kernel = WFKernel(wt, sites, pri, kcfg, seed=args.seed, measured_keys=omap)
    final_recs, history, _gep = run_campaign(kernel, oracle, rounds=args.rounds,
                                       n_gen=args.n_gen, Ne=args.Ne,
                                       n_pop=args.n_pop, batch=args.batch,
                                       log_prefix=f"{args.out_prefix}_")
    # report
    import csv as _csv
    outdir = args.out_dir
    os.makedirs(outdir, exist_ok=True)
    elites = sorted(final_recs, key=lambda r: -r[0])[:args.top]
    ecsv = os.path.join(outdir, args.out_prefix + "_elites.csv")
    with open(ecsv, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["rank", "fitness_proxy", "mutations", "labeled_y"])
        for i, (fp, muts, y) in enumerate(elites, 1):
            w.writerow([i, round(fp, 4), ";".join(f"{r+1}{a}" for r, a in muts), y])
    hcsv = os.path.join(outdir, args.out_prefix + "_trajectory.csv")
    with open(hcsv, "w", newline="", encoding="utf-8") as fh:
        if history:
            fields = list(dict.fromkeys(k for h in history for k in h))
            w = _csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(history)
    print(f"[done] elites -> {ecsv}  ({len(elites)} top-{args.top})")
    print(f"[done] trajectory -> {hcsv}  ({len(history)} generations)")


def cmd_evolve_agentic(args, wt, sites, pri, extras):
    """No-experiment landscape (PprI): kernel.run + elites + stability gate."""
    import csv as _csv
    import numpy as np
    from engine.kernel import WFKernel
    from engine.seqtools import AA2IDX
    ddg, anchors = extras["ddg"], extras["anchors"]
    TAU, W_STAB = extras["TAU"], extras["W_STAB"]
    kcfg = dict(mutations_per_genome_per_gen={"lambda": 0.6}, n_mut_max=12, T=0.6,
                proposal_temp=2.0, w_stab=W_STAB, tau_stab=TAU)
    kernel = WFKernel(wt, sites, pri, kcfg, seed=args.seed, anchor_sites=anchors,
                      proposal="prior")
    stats, pops = kernel.run(n_pop=args.n_pop, n_gen=args.n_gen, Ne=args.Ne,
                             record_events=True)
    print(f"[kernel] last best={stats[-1]['best']:.2f} mean={stats[-1]['mean']:.2f} "
          f"unique={stats[-1]['unique']}")
    site_pos = {s: j for j, s in enumerate(sites)}
    cands = kernel.propose_elites(pops, top_k=args.top * 2, diversity=6)
    outdir, prefix = args.out_dir, args.out_prefix
    os.makedirs(outdir, exist_ok=True)
    ecsv = os.path.join(outdir, prefix + "_elites.csv")
    with open(ecsv, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["rank", "fitness", "mutations", "n_mut", "ddg_pred", "gate_pass"])
        n_pass = 0
        for i, (fit, g) in enumerate(sorted(cands, key=lambda c: -c[0])[:args.top], 1):
            muts = kernel.geno_to_muts(g)
            ddg_pred = float(sum(ddg[site_pos[i0], AA2IDX[a]] for i0, a in muts))
            anchor_viol = [(i0, a) for i0, a in muts if i0 in anchors and a not in anchors[i0]]
            gate = bool(ddg_pred <= 3.0 and not anchor_viol)
            n_pass += gate
            w.writerow([i, round(float(fit), 3),
                        ";".join(f"{i0}{a}" for i0, a in muts), len(muts),
                        round(ddg_pred, 2), int(gate)])
    print(f"[done] elites -> {ecsv}  ({n_pass}/{args.top} pass gate)")


def cmd_benchmark(args):
    import subprocess
    script = os.path.join(HERE, "benchmarks", f"{args.name}.py")
    if not os.path.exists(script):
        raise SystemExit(f"no benchmark {args.name} (expected {script})")
    cmd = [sys.executable, script]
    if args.quick:
        cmd.append("--quick")
    print("[run]", " ".join(cmd))
    sys.exit(subprocess.call(cmd))


def cmd_confirm(args):
    """Confirm elites with pform Boltz layer (optional --boltz)."""
    import csv as _csv
    rows = list(_csv.DictReader(open(args.elites, encoding="utf-8")))
    if not args.boltz:
        print(f"[note] {len(rows)} elites loaded; use --boltz to run structure "
              f"confirmation (calls pform predict).")
        return
    # build yamls: WT sequence + each elite's mutations applied
    wt, sites, pri, omap = load_landscape(args.landscape, args.csv, args.wt)
    import yaml
    outdir = args.out_dir
    ydir = os.path.join(outdir, "confirm_yamls")
    os.makedirs(ydir, exist_ok=True)
    from engine.seqtools import apply_muts
    dna = getattr(args, "dna", "TCATGAGCAGTTTTTTGTTTTTTT")
    for r in rows:
        seq = wt
        if r["mutations"]:
            muts = []
            for tok in r["mutations"].split(";"):
                muts.append((int(tok[1:-1]) - 1, tok[-1]))
            seq = apply_muts(wt, muts)
        doc = {"version": 1, "sequences": [
            {"protein": {"id": "A", "sequence": seq}},
            {"dna": {"id": "B", "sequence": dna}},
            {"ligand": {"id": "C", "ccd": "MN"}}]}
        with open(os.path.join(ydir, f"elite_{r['rank']}.yaml"), "w",
                  encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, allow_unicode=True, sort_keys=False)
    print(f"[yaml] {len(rows)} confirm yamls -> {ydir}")
    print("[next] python cluster/pform/predict.py predict <ydir> "
          f"--template {args.template or '<cif>'} --seeds {' '.join(map(str,args.seeds))} --wait")


def main():
    ap = argparse.ArgumentParser(prog="evotrace", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evolve", help="run a digital evolution campaign")
    ev.add_argument("landscape", choices=list(LANDSCAPES) + ["custom"])
    ev.add_argument("--csv", default=None, help="custom landscape CSV (mutant,DMS_score)")
    ev.add_argument("--wt", default=None, help="WT sequence for custom landscape")
    ev.add_argument("--rounds", type=int, default=2)
    ev.add_argument("--Ne", type=int, default=300)
    ev.add_argument("--n-gen", type=int, default=8)
    ev.add_argument("--n-pop", type=int, default=4)
    ev.add_argument("--batch", type=int, default=24)
    ev.add_argument("--seed", type=int, default=11)
    ev.add_argument("--top", type=int, default=16)
    ev.add_argument("--out-dir", default="results")
    ev.add_argument("--out-prefix", default="evolve")
    ev.set_defaults(fn=cmd_evolve)

    bm = sub.add_parser("benchmark", help="reproduce a benchmark")
    bm.add_argument("name", choices=["b1a", "b1b", "b2", "b3", "b4"])
    bm.add_argument("--quick", action="store_true")
    bm.set_defaults(fn=cmd_benchmark)

    cf = sub.add_parser("confirm", help="confirm elites (optional Boltz layer)")
    cf.add_argument("elites")
    cf.add_argument("--boltz", action="store_true")
    cf.add_argument("--landscape", default="avGFP", choices=list(LANDSCAPES) + ["custom"])
    cf.add_argument("--csv", default=None)
    cf.add_argument("--wt", default=None)
    cf.add_argument("--dna", default="TCATGAGCAGTTTTTTGTTTTTTT")
    cf.add_argument("--template", default=None)
    cf.add_argument("--seeds", type=int, nargs="+", default=[1])
    cf.add_argument("--out-dir", default="results")
    cf.set_defaults(fn=cmd_confirm)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
