# -*- coding: utf-8 -*-
"""pform.metrics_dual_lock — default pluggable metric set for predicting the
"final state" of a protein-nucleic acid complex: contact geometry between
designated anchor pairs, catalyst-nucleic acid proximity, and a binary
final-state verdict.

The metric function runs REMOTELY on the CHPC login node (pure stdlib) so we
never transfer hundreds of CIF files. `build_remote_script()` emits the
self-contained python; `parse_remote_csv()` maps the returned rows into a
pandas frame locally.

All residue numbering is the Boltz/MD convention (as written in the yaml).
"""
import textwrap

DEFAULT_CFG = {
    "chains": {"protein": "A", "nucleic": "B", "ligand": "C"},
    "anchors": {"P1": {"res": 67, "na_res": 17},   # F88 - G17
                "P2": {"res": 232, "na_res": 23}},  # R253 - T23
    "hexxh": (71, 75),       # catalytic helix residue range (chain A)
    "metal": "MN",           # ligand ccd (optional)
    "contact_cut": 4.5,
    "lock_cut": 5.0,
}


def build_remote_script(cfg=None):
    cfg = cfg or DEFAULT_CFG
    code = textwrap.dedent(f"""
    import csv, glob, json, math, os, sys

    CFG = {cfg!r}
    CH = CFG["chains"]
    ANCHORS = CFG["anchors"]
    H0, H1 = CFG["hexxh"]
    CUT = CFG["contact_cut"]
    LOCK = CFG["lock_cut"]

    def dist(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def parse_cif(path):
        chains = {{}}
        for line in open(path, encoding="utf-8", errors="replace"):
            if line.startswith(("ATOM", "HETATM")):
                p = line.split()
                if len(p) < 13 or not p[6].isdigit():
                    continue
                asym = p[9]
                if asym not in CH.values():
                    continue
                r = chains.setdefault(asym, {{}}).setdefault(int(p[6]), {{}})
                r[p[3]] = (float(p[10]), float(p[11]), float(p[12]))
        return chains

    def min_pair(ca, a_res, cb, b_res):
        a = ca.get(a_res, {{}})
        b = cb.get(b_res, {{}})
        if not a or not b:
            return float("nan")
        return min(dist(x, y) for x in a.values() for y in b.values())

    def metrics(path):
        ch = parse_cif(path)
        pa, na, lg = ch.get(CH["protein"], {{}}), ch.get(CH["nucleic"], {{}}), ch.get(CH["ligand"], {{}})
        m = {{"model": os.path.basename(path), "n_atoms": sum(len(r) for c in ch.values() for r in c.values())}}
        for name, a in ANCHORS.items():
            m[f"d_{{name}}"] = round(min_pair(pa, a["res"], na, a["na_res"]), 2)
        # dual-lock conjunction
        m["dual"] = 1 if (m.get("d_P1", 9) <= LOCK and m.get("d_P2", 9) <= LOCK) else 0
        # catalytic helix - nucleic acid closest distance
        m["d_cat"] = round(min(min_pair(pa, r, na, 0) for r in range(H0, H1 + 1)), 2) if na else float("nan")
        # metal - nucleic acid (optional)
        if lg:
            m["d_metal"] = round(min_pair(lg, 1, na, 0), 2) if na else float("nan")
        return m

    def main():
        root = sys.argv[1]
        out = sys.argv[2]
        rows = []
        for cif in sorted(glob.glob(root + "/**/*.cif", recursive=True)):
            try:
                rows.append(metrics(cif))
            except Exception as e:
                rows.append({{"model": cif, "error": str(e)}})
        keys = sorted({{k for r in rows for k in r}})
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"METRICS_OK {{len(rows)}}")

    main()
    """)
    return code


def verdict(row, lock_cut=5.0):
    """Final-state verdict from one model's metric row."""
    d1, d2 = row.get("d_P1"), row.get("d_P2")
    if d1 is None or d2 is None or d1 != d1 or d2 != d2:
        return "n.d."
    return "LOCKED" if (d1 <= lock_cut and d2 <= lock_cut) else "unlocked"
