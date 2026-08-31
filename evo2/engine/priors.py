#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoTrace v2 — 结构→位点类→化学先验（机制先验，非记忆；PprI build_priors.py 的泛化）。

输入：PDB（蛋白链 + 可选配体/核酸/发色团）+ WT 序列 + 链/编号映射。
输出（schema 与 PprI results/priors.csv 完全兼容）：
  priors.csv          — seq_idx, wt_aa, aa, prior, dominant_class, min_dist, pdb_resi
  site_classes.json   — 每位点 {dominant_class, contacts, min_dist, buried_density, anchor}
化学先验表（结构机制推导）：
  DNA 三类沿用 PprI 原表（base_edge/phosphate/sugar_methyl）；
  发色团类（chromophore_edge）：平面共轭体系边缘 —— 芳香/平面/H-bond 化学优先；
  buried：疏水包裹（去稳定电荷惩罚体现在先验低权重）；
  surface：极性/带电自由。
锚位标注（M2）：由调用方按机制传入（如 PprI 88/135/171 芳香锚）。
"""
import csv
import json
import math
import os
from collections import defaultdict

from .seqtools import AA20

AA3TO1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
          "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
          "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

# ---- DNA 接触类别（PprI 原表，逐字保留）
PRIOR_DNA = {
    "base_edge":  {"N": 1.0, "Q": 0.9, "S": 0.9, "T": 0.8, "H": 0.8, "R": 0.6, "K": 0.6,
                   "D": 0.5, "E": 0.5, "Y": 0.7, "W": 0.5, "F": 0.4, "M": 0.3, "V": 0.2, "I": 0.2,
                   "L": 0.2, "A": 0.4, "C": 0.2, "G": 0.1, "P": 0.05},
    "phosphate":  {"R": 1.0, "K": 0.95, "H": 0.8, "S": 0.6, "T": 0.6, "Y": 0.5, "N": 0.5, "Q": 0.5,
                   "W": 0.3, "G": 0.3, "A": 0.3, "V": 0.2, "D": 0.1, "E": 0.1, "P": 0.1, "C": 0.1,
                   "M": 0.1, "L": 0.1, "I": 0.1, "F": 0.1},
    "sugar_methyl": {"F": 1.0, "Y": 0.95, "W": 0.9, "M": 0.8, "L": 0.7, "I": 0.7, "V": 0.7,
                     "S": 0.4, "T": 0.4, "N": 0.3, "Q": 0.3, "H": 0.4, "A": 0.3, "R": 0.2, "K": 0.2,
                     "D": 0.1, "E": 0.1, "C": 0.2, "G": 0.1, "P": 0.1},
}
# ---- 发色团/平面共轭配体边缘（GFP CRO：酚环 π-stack + H-bond 网络 + 质子化环境）
PRIOR_CHROMO_EDGE = {
    "Y": 1.0, "F": 0.95, "W": 0.9, "H": 0.85, "T": 0.85, "S": 0.85, "N": 0.8, "Q": 0.8,
    "T": 0.85, "C": 0.5, "M": 0.5, "V": 0.35, "I": 0.35, "L": 0.35, "A": 0.4,
    "D": 0.55, "E": 0.55, "K": 0.45, "R": 0.45, "G": 0.1, "P": 0.1,
}
# ---- 发色团本体（荧光团残基本身，如 GFP 65-67）：高风向，回WT权重高（高突变风险位）
PRIOR_CHROMO_CORE_WT = 1.0
PRIOR_CHROMO_CORE_OTHER = 0.3
# ---- 埋藏核心（疏水包裹优先，带电极性低权重——去稳定代理）
PRIOR_BURIED = {"I": 1.0, "L": 1.0, "V": 0.95, "F": 0.9, "M": 0.85, "A": 0.7, "W": 0.6,
                "Y": 0.5, "C": 0.5, "T": 0.3, "S": 0.25, "G": 0.3, "N": 0.15, "Q": 0.15,
                "D": 0.08, "E": 0.08, "K": 0.08, "R": 0.08, "H": 0.25, "P": 0.15}
# ---- 表面（极性/带电自由，疏水可用但弱）
PRIOR_SURFACE = {"K": 1.0, "R": 0.95, "E": 0.95, "D": 0.95, "S": 0.8, "T": 0.8, "N": 0.75,
                 "Q": 0.75, "A": 0.6, "G": 0.55, "P": 0.45, "Y": 0.5, "H": 0.6, "M": 0.35,
                 "V": 0.35, "I": 0.3, "L": 0.3, "F": 0.3, "W": 0.2, "C": 0.25}

DNA_PHOS = {"P", "OP1", "OP2"}
DNA_SUGAR = {"C7", "C5'", "C4'", "C3'", "O4'", "C2'", "C1'", "O3'", "O5'"}


def parse_pdb(path, protein_chain="A", ligand_resnames=None, ligand_chain=None):
    """解析 PDB：蛋白残基 + 配体重原子（HETATM 指定 resname 或指定链的全部非水）。"""
    prot, lig = {}, []
    for ln in open(path, errors="replace"):
        rec = ln[:6]
        if rec.startswith(("ATOM", "HETATM")):
            altloc = ln[16]
            if altloc not in (" ", "A"):
                continue
            resn = ln[17:20].strip()
            ch = ln[21]
            resi = int(ln[22:26])
            name = ln[12:16].strip()
            el = ln[76:78].strip().upper()
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            if rec.startswith("ATOM") and ch == protein_chain and resn in AA3TO1:
                prot.setdefault(resi, {"resn": resn, "atoms": {}})
                if el != "H":
                    prot[resi]["atoms"][name] = xyz
            else:
                in_lig = (ligand_resnames and resn in ligand_resnames) or \
                         (ligand_chain and ch == ligand_chain and resn not in ("HOH",))
                if in_lig and el not in ("H", "D") and resn != "HOH":
                    lig.append((resn, resi, name, xyz))
    return prot, lig


def build_priors(pdb_path, wt_seq, seq_offset, protein_chain="A",
                 ligand_resnames=None, ligand_chain=None,
                 contact_cut=4.5, facing_cut=6.0, buried_density=70.0,
                 fixed_sites=(), chromophore_core_sites=(), anchor_map=None,
                 out_dir=None, tag=""):
    """主入口。seq_offset: PDB 编号 = seq_idx + seq_offset。

    返回 (priors_rows, site_meta)；out_dir 给定时写 priors.csv / site_classes.json。
    """
    prot, lig = parse_pdb(pdb_path, protein_chain, ligand_resnames, ligand_chain)
    anchor_map = anchor_map or {}
    # 每残基的配体接触统计
    site_meta = {}
    for resi, res in sorted(prot.items()):
        if resi - seq_offset < 0 or resi - seq_offset >= len(wt_seq):
            continue
        best = 1e9
        cls_count = defaultdict(int)
        for lresn, lresi, lname, lxyz in lig:
            for pname, pxyz in res["atoms"].items():
                d = math.dist(pxyz, lxyz)
                best = min(best, d)
                if d < contact_cut:
                    if lname in DNA_PHOS:
                        cls_count["phosphate"] += 1
                    elif lname in DNA_SUGAR:
                        cls_count["sugar_methyl"] += 1
                    else:
                        cls_count["chromophore_edge" if lresn not in ("DA", "DT", "DG", "DC", "A", "U", "G", "C")
                                  else "base_edge"] += 1
        # 埋藏密度：8Å 内重原子邻居数（蛋白+配体）
        all_xyz = [p for r in prot.values() for p in r["atoms"].values()] + \
                  [x for _, _, _, x in lig]
        center = None
        for pname, pxyz in res["atoms"].items():
            if pname == "CA":
                center = pxyz
                break
        dens = 0
        if center is not None:
            dens = sum(1 for p in all_xyz if 0.1 < math.dist(p, center) < 8.0)
        site_meta[resi] = dict(
            resn=res["resn"], aa=AA3TO1.get(res["resn"], "X"),
            min_dist=round(best, 2) if best < 1e9 else None,
            contacts=dict(cls_count), total_contacts=sum(cls_count.values()),
            buried_density=dens)
    # 位点类判定 + 先验
    rows = []
    chromo_core = set(chromophore_core_sites)
    handled = set()
    for resi, meta in sorted(site_meta.items()):
        j = resi - seq_offset
        if j in chromo_core:
            continue  # 单独处理（发色团本体）
        handled.add(j)
        contacts = meta["contacts"]
        if contacts:
            dom = max(contacts.items(), key=lambda kv: kv[1])[0]
        elif meta["buried_density"] >= buried_density:
            dom = "buried"
        else:
            dom = "surface"
        mix = defaultdict(float)
        if contacts:
            tot = meta["total_contacts"]
            tbl = {"base_edge": PRIOR_DNA["base_edge"],
                   "phosphate": PRIOR_DNA["phosphate"],
                   "sugar_methyl": PRIOR_DNA["sugar_methyl"],
                   "chromophore_edge": PRIOR_CHROMO_EDGE}
            for c, n in contacts.items():
                w = 0.7 if c == dom else 0.3
                for aa, pw in tbl[c].items():
                    mix[aa] += pw * w * n / tot
        else:
            tbl = PRIOR_BURIED if dom == "buried" else PRIOR_SURFACE
            for aa, pw in tbl.items():
                mix[aa] = pw
        z = sum(mix.values()) or 1.0
        wt_aa = wt_seq[j]
        for aa in AA20:
            rows.append(dict(seq_idx=j, pdb_resi=resi, wt_aa=wt_aa, aa=aa,
                             prior=round(max(mix.get(aa, 0.02), 0.02) / z, 4),
                             dominant_class=dom, min_dist=meta["min_dist"],
                             buried_density=meta["buried_density"]))
    # 发色团本体位点：WT 优先的高风险类
    for j in sorted(chromo_core):
        if j < 0 or j >= len(wt_seq):
            continue
        wt_aa = wt_seq[j]
        resi = j + seq_offset
        for aa in AA20:
            p = PRIOR_CHROMO_CORE_WT if aa == wt_aa else PRIOR_CHROMO_CORE_OTHER
            rows.append(dict(seq_idx=j, pdb_resi=resi, wt_aa=wt_aa, aa=aa,
                             prior=p, dominant_class="chromophore_core",
                             min_dist=0.0, buried_density=-1))
    meta_out = {}
    for r in sorted({x["seq_idx"] for x in rows}):
        wt_cls = next(x["dominant_class"] for x in rows if x["seq_idx"] == r and x["aa"] == wt_seq[r])
        sm = site_meta.get(r + seq_offset, {})
        meta_out[r] = dict(dominant_class=wt_cls, min_dist=sm.get("min_dist"),
                           contacts=sm.get("contacts", {}),
                           buried_density=sm.get("buried_density", -1),
                           anchor=r in anchor_map)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"priors{tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        json.dump(meta_out, open(os.path.join(out_dir, f"site_classes{tag}.json"), "w"),
                  indent=1)
    return rows, meta_out


def load_priors_csv(path):
    """priors.csv -> (priors{seq_idx:{aa:p}}, wt_aa{seq_idx}, classes{seq_idx:dom})。"""
    priors, wt_aa, cls, anchor = defaultdict(dict), {}, {}, {}
    for r in csv.DictReader(open(path)):
        j = int(r["seq_idx"])
        priors[j][r["aa"]] = max(float(r["prior"]), 1e-3)
        wt_aa[j] = r["wt_aa"]
        cls[j] = r["dominant_class"]
    return dict(priors), wt_aa, cls
