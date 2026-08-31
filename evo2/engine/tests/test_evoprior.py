#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoPrior 自包含测试（合成数据，无需 PLM / 外部数据）。

要点
----
- 数据驱动的"进化先验" p_evo 的真值就是 MSA-PSSM 本身；MLP 是 PSSM 的近似器。
- 占位嵌入（one-hot + 邻域）不含进化信息，仅用于跑通端到端流程；
  **真实 PLM（ESM-2/ProtBERT）嵌入才编码同源保守性**——这是 EvoPrior
  相对"手工化学规则"的根本优势（规则驱动 vs 数据驱动）。
- 本测试验证：融合层数学、α-CV、MLP 收敛，以及 M135 式盲点修复逻辑。

运行：python evo2/engine/tests/test_evoprior.py
"""
import os
import sys
import tempfile

import numpy as np

EVO2 = os.path.join(os.path.dirname(__file__), "..", "..")  # evo2/
sys.path.insert(0, os.path.abspath(EVO2))

from engine.evoprior import (  # noqa: E402
    read_a3m, msa_to_pssm, placeholder_embeddings, EvoPriorMLP,
    fuse_tables, learn_alpha_cv, build_evoprior, AA20, AA2IDX,
)

N_AA = 20
HYD = [AA2IDX[a] for a in "MFLI"]  # 疏水残基


def synth_seq(L=254, pos135="M"):
    s = ["A"] * L
    s[135] = pos135
    return "".join(s)


def synth_msa(seq, n=400, seed=0):
    """合成 MSA：135 位 80% 疏水(M/F/L)，其余位点 5% 噪声（偏向 WT）。"""
    rng = np.random.default_rng(seed)
    rows = [seq]
    for _ in range(n - 1):
        arr = list(seq)
        if rng.random() < 0.8:
            arr[135] = rng.choice(list("MFL"))
        else:
            arr[135] = rng.choice(list("ACDEGHKNPQRSTVWY"))
        for i in range(len(arr)):
            if i != 135 and rng.random() < 0.05:
                arr[i] = rng.choice(list("ACDEFGHIKLMNPQRSTVWY"))
        rows.append("".join(arr))
    return rows


def pchem_blinded(seq):
    """化学先验（含 M135 盲点）：手工规则 M135H 错误地把 135 位推给 H。

    40/88/171 为正常埋藏疏水位（疏水高）；135 位被错误规则反转：H 高、疏水压低。
    """
    L = len(seq)
    tab = np.full((L, N_AA), 0.05)
    for i in (40, 88, 171):  # 正常埋藏疏水位
        for aa, w in {"M": 0.85, "F": 0.9, "L": 1.0, "I": 1.0, "V": 0.95}.items():
            tab[i, AA2IDX[aa]] = max(tab[i, AA2IDX[aa]], w)
        tab[i, AA2IDX[seq[i]]] = max(tab[i, AA2IDX[seq[i]]], 0.4)
    # 135 位盲点：H 高、疏水压低（模拟手工规则 M135H 的错误先验）
    i = 135
    tab[i] = 0.05
    tab[i, AA2IDX["H"]] = 0.9
    tab[i, AA2IDX["R"]] = 0.3
    tab[i, AA2IDX["K"]] = 0.3
    tab[i, AA2IDX["M"]] = 0.1
    tab[i, AA2IDX["F"]] = 0.1
    tab[i, AA2IDX["L"]] = 0.1
    tab = np.clip(tab, 1e-3, None)
    return tab / tab.sum(axis=1, keepdims=True)


def test_pipeline():
    seq = synth_seq()
    msa = synth_msa(seq)
    with tempfile.NamedTemporaryFile("w", suffix=".a3m", delete=False) as f:
        f.write(">query\n" + seq + "\n")
        for r in msa[1:]:
            f.write(">hit\n" + r + "\n")
        a3m_path = f.name
    try:
        q, seqs = read_a3m(a3m_path)
        assert q == seq
        pssm = msa_to_pssm(seqs, seq, pseudocount=1.0)
        assert pssm.shape == (len(seq), N_AA)

        # 1) MSA 在 135 位确实体现疏水偏好（数据驱动真值）
        assert pssm[135, HYD].sum() > pssm[135, AA2IDX["H"]], \
            "合成 MSA 在 135 位未体现疏水偏好"

        # 2) p_evo = PSSM（数据驱动先验真值）
        p_evo = pssm.copy()

        # 3) 盲点：化学先验把 135 位推给 H
        p_chem = pchem_blinded(seq)
        assert p_chem[135, AA2IDX["H"]] > p_chem[135, AA2IDX["M"]], \
            "未构造出 M135H 盲点"

        # 4) 融合：盲点修复——融合后 135 位疏水概率高于纯化学先验
        hyd_chem = p_chem[135, HYD].sum()
        pf = fuse_tables(p_chem, p_evo, alpha=0.5)
        hyd_fuse = pf[135, HYD].sum()
        assert hyd_fuse > hyd_chem, \
            f"融合未修复盲点：疏水 {hyd_chem:.3f} -> {hyd_fuse:.3f}"

        # 5) α 交叉验证：学到降低 KL(p_ssm||p_final) 的 α
        alpha, curve = learn_alpha_cv(p_chem, p_evo, pssm, folds=4)
        assert 0.0 <= alpha <= 1.0
        kl_a1 = _kl_at(pssm, p_chem, p_evo, 1.0, 135)
        kl_af = _kl_at(pssm, p_chem, p_evo, alpha, 135)
        assert kl_af <= kl_a1 + 1e-6, "CV 学到的 α 未在盲点处降低 KL"

        # 6) MLP 收敛：交叉熵应下降（占位嵌入下也能记忆训练集）
        emb = placeholder_embeddings(seq, window=3)
        m = EvoPriorMLP(emb.shape[1], hidden=64, seed=1)
        hist = m.fit(emb, pssm, epochs=120, lr=0.05, verbose=False)
        assert hist[-1] < hist[0], "MLP 未收敛（CE 未下降）"

        # 7) 端到端 build_evoprior 返回形状正确
        evo = build_evoprior(seq, a3m_path, use_plm=False, epochs=5, verbose=False)
        assert evo["p_evo"].shape == (len(seq), N_AA)
        assert evo["used_plm"] is False

        print(f"[OK] M135 盲点修复: 化学疏水={hyd_chem:.3f} "
              f"-> 融合(α={alpha:.2f})疏水={hyd_fuse:.3f}\n"
              f"     KL(135): α=1 {kl_a1:.3f} -> α={alpha:.2f} {kl_af:.3f}\n"
              f"     MLP CE: {hist[0]:.3f} -> {hist[-1]:.3f}（{'收敛' if hist[-1]<hist[0] else '未降'}）")
    finally:
        os.unlink(a3m_path)


def _kl_at(pssm, p_chem, p_evo, alpha, site):
    pf = fuse_tables(p_chem[site:site + 1], p_evo[site:site + 1], alpha)[0]
    eps = 1e-8
    return float(np.sum(pssm[site] * (np.log(pssm[site] + eps) - np.log(pf + eps))))


if __name__ == "__main__":
    test_pipeline()
    print("\nALL EVO-PRIOR TESTS PASSED")
