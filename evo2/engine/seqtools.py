#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoTrace v2 — 序列/突变基础工具（蛋白无关）。

突变串格式沿用 ProteinGym："A123B:C45D"（冒号分隔，1-based PDB/UniProt 编号）。
"""
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {a: i for i, a in enumerate(AA20)}
GAP = "-"


def read_fasta(path):
    seqs, name, buf = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        elif line:
            buf.append(line)
    if name:
        seqs[name] = "".join(buf)
    return seqs


def parse_mutant(mutant):
    """'A123B:C45D' -> [(122,'B'), (444,'D')] （0-based seq idx）。"""
    out = []
    if not mutant or str(mutant) in ("nan", "None"):
        return out
    for tok in str(mutant).split(":"):
        if len(tok) < 4:
            continue
        out.append((int(tok[1:-1]) - 1, tok[-1]))
    return out


def apply_mutant(wt_seq, muts):
    s = list(wt_seq)
    for i, a in muts:
        if i < len(s):
            s[i] = a
    return "".join(s)


def diff_seqs(wt_seq, seq):
    """两条序列差异 -> [(i, aa)]。"""
    return [(i, b) for i, (a, b) in enumerate(zip(wt_seq, seq)) if a != b]


def fmt_mutant(muts):
    """[(i, aa)] 0-based -> 'A124B'（1-based）。"""
    return ":".join(f"X{i + 1}{a}" for i, a in muts)
