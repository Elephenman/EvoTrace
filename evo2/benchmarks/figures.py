#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoTrace v2 — 出版级主图套件（matplotlib，无 seaborn 依赖）。

Fig1  方法总览（分层漏斗 + 回流环）——文字版在文档中，这里跳过
Fig2  B1a avGFP 标签经济曲线 + 终点 top16
Fig3  B1b GB1 test Spearman vs 预算（FLIP 对齐）
Fig4  B2 TEM-1 跨条件迁移热图式条形
Fig5  B4 ProteinGym 打分器地板 vs 官方榜单带
Fig6  B5 PprI wave-2 before/after（sep vs ddG 散点 + 波次演进）
Fig7  B3 理论一致性（固定概率 + 平行进化）
输出：figures/*.pdf + *.png (300dpi)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 120, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
C = {"v2": "#d62728", "v2-prior": "#ff9896", "v1": "#7f7f7f",
     "random": "#bbbbbb", "random-prior": "#999999", "ridge-AL": "#1f77b4",
     "esm3": "#2ca02c"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_b1a():
    p = os.path.join(ROOT, "results", "b1a_results.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    if len(df) < 10:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    for strat, g in df.groupby("strategy"):
        gg = g.groupby("budget_target").agg(m=("best_found", "mean"),
                                            s=("best_found", "std"))
        ax.errorbar(gg.index, gg.m, yerr=gg.s, marker="o", ms=3.5, lw=1.4,
                    color=C.get(strat, None), label=strat, capsize=2)
    ax.axhline(4.123, color="k", ls=":", lw=0.8)
    ax.text(100, 4.126, "landscape max", fontsize=7, va="bottom")
    ax.set_xlabel("assay labels (budget)")
    ax.set_ylabel("best variant found (log10 brightness)")
    ax.set_title("avGFP: label economy")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for strat, g in df.groupby("strategy"):
        gg = g.groupby("budget_target").agg(m=("pred_top16", "mean"),
                                            s=("pred_top16", "std"))
        ax.errorbar(gg.index, gg.m, yerr=gg.s, marker="s", ms=3.5, lw=1.4,
                    color=C.get(strat, None), label=strat, capsize=2)
    ax.set_xlabel("assay labels (budget)")
    ax.set_ylabel("true y of model top-16")
    ax.set_title("avGFP: model extrapolation (panel top-16)")
    ax.legend(frameon=False, fontsize=6, loc="lower right", ncol=2)
    save(fig, "fig2_b1a_avgfp")


def fig_b1b():
    p = os.path.join(ROOT, "results", "b1b_gb1.csv")
    if not os.path.exists(p) or len(pd.read_csv(p)) < 8:
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for strat, g in df.groupby("strategy"):
        gg = g.groupby("budget_target").agg(m=("spearman_test", "mean"),
                                            s=("spearman_test", "std"))
        ax.errorbar(gg.index, gg.m, yerr=gg.s, marker="o", ms=3.5, lw=1.4,
                    color=C.get(strat, None), label=strat, capsize=2)
    ax.set_xscale("log")
    ax.set_xticks([192, 424, 1536])
    ax.set_xticklabels(["192", "424\n(FLIP train)", "1536"])
    ax.set_xlabel("labels")
    ax.set_ylabel("test Spearman (3–4 mut variants)")
    ax.set_title("GB1 four-site landscape (FLIP two-vs-rest)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig3_b1b_gb1")


def fig_b2():
    p = os.path.join(ROOT, "results", "b2_tem1.csv")
    if not os.path.exists(p) or len(pd.read_csv(p)) < 4:
        return
    df = pd.read_csv(p)
    domains = [("v2_rho", "ridge_rho", "Firnberg (train)"),
               ("v2_Deng", "ridge_Deng", "Deng 2012"),
               ("v2_Stiffler", "ridge_Stiffler", "Stiffler 2015"),
               ("v2_Jacquier", "ridge_Jacquier", "Jacquier 2013")]
    budgets = sorted(df.budget.unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(7.2, 2.6), sharey=True)
    if len(budgets) == 1:
        axes = [axes]
    x = np.arange(len(domains))
    for ax, B in zip(axes, budgets):
        g = df[df.budget == B]
        v2m = [g[c1].mean() for c1, _, _ in domains]
        v2s = [g[c1].std() for c1, _, _ in domains]
        rm = [g[c2].mean() for _, c2, _ in domains]
        rs = [g[c2].std() for _, c2, _ in domains]
        ax.bar(x - 0.18, v2m, 0.34, yerr=v2s, color=C["v2"], label="EvoTrace v2",
               capsize=2)
        ax.bar(x + 0.18, rm, 0.34, yerr=rs, color=C["ridge-AL"],
               label="ridge (same labels)", capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([d[2] for d in domains], rotation=20, ha="right",
                           fontsize=7)
        ax.set_title(f"budget {B}")
        ax.axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("Spearman")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "fig4_b2_tem1_transfer")


def fig_b4():
    p = os.path.join(ROOT, "results", "b4_pgym_floor.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    ref = {"VESPA": 0.30, "ESM-1v (single)": 0.374, "Tranception L": 0.434,
           "ESM3 open (1.4B)": 0.466, "SOTA (AIDO/VenusREM)": 0.518}
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    vals = [df.rho_blosum.mean(), df.rho_dms_matrix.mean()]
    if os.path.exists(os.path.join(ROOT, "results", "b4_esm3_pgym.csv")):
        e = pd.read_csv(os.path.join(ROOT, "results", "b4_esm3_pgym.csv"))
        vals.append(e.rho_esm3.mean())
        labels = ["BLOSUM62\nadditive", "DMS-calibrated\n20×20 matrix",
                  "ESM3 open 1.4B\n(ours, zero-shot)"]
    else:
        labels = ["BLOSUM62\nadditive", "DMS-calibrated\n20×20 matrix"]
    ax.bar(range(len(vals)), vals, 0.55,
           color=["#bbbbbb", "#888888", C["esm3"]][:len(vals)])
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for name, v in ref.items():
        ax.axhline(v, color="#d62728", lw=0.6, ls=":", alpha=0.6)
        ax.text(-0.45, v + 0.004, f"{name} {v:.3f}", fontsize=6.5,
                color="#d62728", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
    ax.set_ylabel("mean Spearman (30 DMS)")
    ax.set_title("ProteinGym: cheap-tier floor vs frontier band")
    ax.set_ylim(0, 0.58)
    save(fig, "fig5_b4_pgym_floor")


def fig_b5():
    v = os.path.join(ROOT, "results", "b5_wave2_verdict.csv")
    e = os.path.join(ROOT, "results", "b5_wave2_elites.csv")
    if not os.path.exists(v):
        return
    df = pd.read_csv(v)
    hist = pd.DataFrame([
        dict(wave="WT", sep=45, act_s1=70, act_off=25, ddg=0.0),
        dict(wave="s13_c1\n(old campaign)", sep=53, act_s1=74, act_off=21, ddg=np.nan),
        dict(wave="e18\n(wave-0)", sep=63.4, act_s1=66.7, act_off=3.3, ddg=15.1),
        dict(wave="e3\n(wave-0)", sep=55.0, act_s1=80.0, act_off=25.0, ddg=17.0),
        dict(wave="w4\n(wave-1)", sep=53.3, act_s1=85.0, act_off=31.7, ddg=16.4),
    ])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax = axes[0]
    x = np.arange(len(hist))
    ax.bar(x, hist.sep, 0.6, color=["#bbbbbb", "#999999", "#7f7f7f", "#7f7f7f",
                                    "#7f7f7f"])
    for i, r in hist.iterrows():
        if not np.isnan(r.ddg):
            ax.text(i, r.sep + 1, f"ΔΔG {r.ddg:+.1f}", ha="center", fontsize=6.5,
                    color="#d62728")
    ax.axhline(70.0, color=C["v2"], lw=1.2, ls="--")
    ax.text(4.35, 70.8, "wave2_3 = 70.0 (Boltz-2 confirmed)", fontsize=6.5,
            color=C["v2"], ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["WT", "s13_c1\n(prev best)", "e18\nw0", "e3\nw0",
                        "w4\nw1"], fontsize=7)
    ax.set_ylabel("confirmed separation (S1 − OFF)")
    ax.set_title("PprI campaign: waves 0–1 (all destabilized)")
    ax = axes[1]
    for i, r in df.iterrows():
        ax.errorbar(r.act_off, r.act_s1, xerr=r.act_off_std, yerr=r.act_s1_std,
                    fmt="o", ms=7, color=C["v2"], capsize=2)
        ax.annotate(r.elite, (r.act_off, r.act_s1), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.scatter([25], [70], marker="*", s=90, color="#7f7f7f", zorder5=None) \
        if False else ax.scatter([25], [70], marker="*", s=90, color="#7f7f7f")
    ax.annotate("WT", (25, 70), fontsize=7, xytext=(4, -10),
                textcoords="offset points")
    ax.plot([0, 100], [0, 100], "k:", lw=0.6)
    ax.set_xlabel("OFF-target activation (%)  ↓ better")
    ax.set_ylabel("target S1 activation (%)  ↑ better")
    ax.set_title("PprI wave-2 (Boltz-2 confirm, 3×3×20 models)\n"
                 "ΔΔG_pred ≤ 0 by construction")
    ax.set_xlim(-2, 60)
    ax.set_ylim(60, 92)
    save(fig, "fig6_b5_ppri_wave2")


def fig_b3():
    p = os.path.join(ROOT, "results", "b3_theory.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fix = df[df.test == "fixation"]
    if not len(fix):
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    ax.plot(fix.s, fix.theoretical, "k-", lw=1.2, label="Kimura analytic")
    ax.errorbar(fix.s, fix.empirical, yerr=np.sqrt(fix.empirical * (1 - fix.empirical)
                                                   / 4000), fmt="o", ms=4,
                color=C["v2"], label="WF kernel (simulated)")
    ax.set_xlabel("selection coefficient s")
    ax.set_ylabel("fixation probability")
    ax.set_title("fixation probability vs Kimura")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    par = df[df.test == "parallelism"]
    if len(par):
        r = par.iloc[0]
        ax.bar([0, 1], [r.expected, r.obs], 0.5, color=["#bbbbbb", C["v2"]])
        ax.text(0, r.expected + max(r.obs, 0.01) * 0.03, f"{r.expected:.3f}",
                ha="center", fontsize=7.5)
        ax.text(1, r.obs + max(r.obs, 0.01) * 0.03, f"{r.obs:.3f}",
                ha="center", fontsize=7.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["random\nexpectation", f"observed\n(z={r.z:.0f})"])
        ax.set_ylabel("cross-population (site,AA) overlap")
        ax.set_title(f"parallel evolution (8 populations)")
    save(fig, "fig7_b3_theory")


if __name__ == "__main__":
    fig_b1a()
    fig_b1b()
    fig_b2()
    fig_b4()
    fig_b5()
    fig_b3()
    print("figures done ->", FIG)
