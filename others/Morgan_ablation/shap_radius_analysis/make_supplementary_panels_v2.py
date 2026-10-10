#!/usr/bin/env python
# coding: utf-8
# =====================================================================
# Supplementary figure panels A-F (raw Morgan uses retuned model_2)
#   A  EF0.5% across six similarity bands
#   B  EF1%   across six similarity bands
#   C  radius=0 enrichment among globally influential Morgan positions
#   D  score recovered from high-prevalence positions alone  (% of full)
#   E  score retained after removing those positions         (% of full)
#   F  AUROC on the high-prevalence challenge subset
# Read CSVs only. Match draw_KIBA_figure colors: KinSPEC/LSAE #4C72B0; comparator gray #9E9E9E.
# =====================================================================

# Resolve defaults from this project's root, independently of the working directory.
from pathlib import Path as _KSPath
import sys as _ks_sys
from functools import partial as _ks_partial
_ks_start = _KSPath(__file__).resolve().parent
_ks_root = next((p for p in (_ks_start, *_ks_start.parents) if (p / 'kinspec_paths.py').is_file()), None)
if _ks_root is None:
    raise RuntimeError('Cannot locate the KinSPEC project root containing kinspec_paths.py')
_ks_sys.path.insert(0, str(_ks_root))
from kinspec_paths import ROOT as KINSPEC_ROOT, project_path as _ks_project_path
_kp = _ks_partial(_ks_project_path, script_dir='others/Morgan_ablation/shap_radius_analysis')

import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = _kp('others/Morgan_ablation/shap_radius_analysis')
SRC, AB = f"{BASE}/band_10_50_v2", f"{BASE}/all_bands_v2"
OUT = f"{BASE}/figures_v2"
os.makedirs(OUT, exist_ok=True)

MORGAN, LSAE = "#9E9E9E", "#4C72B0"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})
def tidy(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(length=3)

# ---------------------------------------------------------------- data
ab = pd.read_csv(f"{AB}/per_band_fold_metrics.csv")
BANDS = ["10_50", "20_60", "30_70", "40_80", "50_90", "60_100"]
BLAB = [b.replace("_", "–") for b in BANDS]
pf = pd.read_csv(f"{SRC}/per_fold_metrics.csv")
e1 = pd.read_csv(f"{SRC}/E1_retention_per_fold.csv")
KS = [10, 20, 50]
enr = {k: pf[pf.metric == f"posSHAP_top{k}_r0_enrich"].sort_values("fold").value for k in KS}
e3 = pf[pf.metric == "E3_adversarial_auroc"].pivot(index="fold", columns="model", values="value")
sel = e1[e1.group.isin(["decoy_high (FP)", "active_high (TP)"])].copy()
sel["recovered_frac"] = sel.keep_promiscuous / sel.full
ORDER = [("morgan", "decoy_high (FP)"), ("morgan", "active_high (TP)"),
         ("lsae", "decoy_high (FP)"), ("lsae", "active_high (TP)")]
XLAB = ["Raw Morgan\nhigh-scoring\ndecoys", "Raw Morgan\nhigh-scoring\nactives",
        "LSAE\nhigh-scoring\ndecoys", "LSAE\nhigh-scoring\nactives"]
COL = [MORGAN, MORGAN, LSAE, LSAE]
# Return pandas Series so .std() uses the sample standard deviation (ddof=1), as in the paired table
get = lambda m, g, c: sel[(sel.model == m) & (sel.group == g)].sort_values("fold")[c]
YL = "% of full-fingerprint score"

# ---------------------------------------------------------------- panels
DISP = {"EF0.5%": "EF0.5%", "EF1.0%": "EF1%"}


def band_panel(ax, metric, title):
    x = np.arange(len(BANDS))
    for tag, c, lab in [("morgan", MORGAN, "Raw Morgan"), ("lsae", LSAE, "LSAE")]:
        mu = np.array([ab[(ab.band == b) & (ab.model == tag)][metric].mean() for b in BANDS])
        sd = np.array([ab[(ab.band == b) & (ab.model == tag)][metric].std() for b in BANDS])
        ax.plot(x, mu, marker="o", ms=4.5, color=c, lw=1.8, label=lab, zorder=3)
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.18, lw=0, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(BLAB, fontsize=7)
    ax.set_xlabel("Compound-similarity band (%)")
    ax.set_ylabel(DISP[metric]); ax.set_title(title, color=INK, pad=6)
    # Start the y-axis at zero to avoid exaggerating the relative difference between curves
    ax.set_ylim(0, None); ax.margins(y=0.12)
    ax.legend(frameon=False, loc="lower right"); tidy(ax)

def panel_A(ax): band_panel(ax, "EF0.5%", "Early enrichment across similarity bands\n(EF0.5%)")
def panel_B(ax): band_panel(ax, "EF1.0%", "Early enrichment across similarity bands\n(EF1%)")

def panel_C(ax):
    x = np.arange(len(KS)); mu = np.array([enr[k].mean() for k in KS])
    ax.bar(x, mu, 0.55, color=MORGAN, edgecolor="white", lw=0.8, zorder=2)
    for i, k in enumerate(KS):
        ax.scatter(x[i] + np.linspace(-0.12, 0.12, len(enr[k])), enr[k].values, s=11,
                   color="white", edgecolor=INK, lw=0.7, zorder=4, clip_on=False)
        ax.text(x[i], max(mu[i], enr[k].max()) + 0.8, f"{mu[i]:.1f}×",
                ha="center", va="bottom", fontsize=7.5, color=INK, zorder=5)
    ax.axhline(1.0, color=INK2, ls=(0, (4, 3)), lw=0.8, zorder=1)
    ax.text(len(KS) - 0.55, 1.5, "no enrichment", ha="right", va="bottom",
            fontsize=6.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([f"top {k}" for k in KS])
    ax.set_xlabel("Morgan positions ranked by mean |SHAP|")
    ax.set_ylabel("Radius-0 enrichment\n(vs all 4,096 positions)")
    ax.set_ylim(0, 24); ax.set_xlim(-0.5, len(KS) - 0.5)
    ax.set_title("Radius-0 enrichment among globally\ninfluential Morgan positions", color=INK, pad=6)
    tidy(ax)

def four_bars(ax, col, title, ymax):
    x = np.arange(4)
    for j, (m, g) in enumerate(ORDER):
        v = get(m, g, col) * 100
        ax.bar(x[j], v.mean(), 0.58, color=COL[j], edgecolor="white", lw=0.8, zorder=2)
        ax.scatter(x[j] + np.linspace(-.12, .12, len(v)), v.values, s=11, color="white",
                   edgecolor=INK, lw=0.7, zorder=4, clip_on=False)
        ax.text(x[j], max(v.mean(), v.max()) + ymax * 0.03, f"{v.mean():.1f}%",
                ha="center", va="bottom", fontsize=7.5, color=INK, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(XLAB)
    ax.set_ylabel(YL); ax.set_ylim(0, ymax)
    ax.set_title(title, color=INK, pad=6); tidy(ax)

def panel_D(ax): four_bars(ax, "recovered_frac",
                           "Score recovered from high-prevalence\npositions alone", 85)
def panel_E(ax): four_bars(ax, "delete_retained_frac",
                           "Prediction score retained after removing\nhigh-prevalence positions", 85)

def panel_F(ax):
    xm, xl = 0.0, 1.0
    for f in e3.index:
        ax.plot([xm, xl], [e3.loc[f, "morgan"], e3.loc[f, "lsae"]], color=GRID, lw=0.9, zorder=1)
    ax.scatter([xm]*len(e3), e3.morgan, s=26, color=MORGAN, edgecolor="white", lw=0.8, zorder=3)
    ax.scatter([xl]*len(e3), e3.lsae, s=26, color=LSAE, edgecolor="white", lw=0.8, zorder=3)
    for xx, v, c in [(xm, e3.morgan, MORGAN), (xl, e3.lsae, LSAE)]:
        ax.plot([xx - 0.17, xx + 0.17], [v.mean()]*2, color=c, lw=2.4, zorder=4, solid_capstyle="butt")
        side = 1 if c == LSAE else -1
        ax.text(xx + side * 0.23, v.mean(), f"{v.mean():.3f}\n± {v.std():.3f}",
                ha="left" if side > 0 else "right", va="center",
                fontsize=7.2, color=INK, zorder=5, linespacing=1.35)
    ax.axhline(0.5, color=INK2, ls=(0, (4, 3)), lw=0.8, zorder=2)
    ax.text(1.75, 0.5, "random", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks([xm, xl]); ax.set_xticklabels(["Raw Morgan", "LSAE"])
    lo = min(e3.min().min(), 0.5) - 0.02; hi = max(e3.max().max(), 0.5) + 0.02
    ax.set_xlim(-0.75, 1.75); ax.set_ylim(lo, hi)
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC on the high-prevalence\nchallenge subset", color=INK, pad=6)
    tidy(ax); ax.xaxis.grid(False)

# ---------------------------------------------------------------- render
PANELS = [("A", panel_A), ("B", panel_B), ("C", panel_C),
          ("D", panel_D), ("E", panel_E), ("F", panel_F)]
fig, axes = plt.subplots(3, 2, figsize=(7.2, 9.9), constrained_layout=True)
for (L, fn), ax in zip(PANELS, axes.flat):
    fn(ax)
    ax.text(-0.20, 1.10, L, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", color=INK)
for ext in ["pdf", "png"]:
    fig.savefig(f"{OUT}/FigureSx_AF.{ext}", bbox_inches="tight")
plt.close(fig)
for L, fn in PANELS:
    f, a = plt.subplots(figsize=(3.6, 3.3), constrained_layout=True)
    fn(a)
    for ext in ["pdf", "png"]:
        f.savefig(f"{OUT}/FigureSx_{L}.{ext}", bbox_inches="tight")
    plt.close(f)

print(f"\n[done] -> {OUT}")
