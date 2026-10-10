#!/usr/bin/env python3
"""
Karaman kinase-selectivity benchmark: final figures and paired statistics.

Data
----
  baselines + KinSPEC: KinSPEC_MAC_kinase_selectivity_with_ready2use_v4.xlsx
  KinSPEC_LTS (v4): Karaman_MAC_result_LTS_v4/..._per_molecule.csv,
                   replacing the stale KinSPEC_LTS column in the input workbook.

Statistics (same definitions as KinSPEC_Table_S1.xlsx, with two-sided tests)
-------------------------------------------------------------------------
  - Paired two-sided Wilcoxon signed-rank test (zero_method='pratt').
  - Holm-Bonferroni across the four baseline comparisons (m=4).
    KinSPEC_LTS versus KinSPEC is a separate prespecified comparison (m=1),
    for which Holm is the identity (p_holm = p_raw).
  - Matched-pairs rank-biserial r_rb and bootstrap 95% intervals for median differences and r_rb.
"""

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
_kp = _ks_partial(_ks_project_path, script_dir='others/draw_figures/Karaman_draw_the_violin')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests

# =========================
# 0) Global plot settings
# =========================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 600

ROOT = _kp('others/draw_figures/Karaman_draw_the_violin')
BASE = ROOT
IN_XLSX = f"{BASE}/KinSPEC_MAC_kinase_selectivity_with_ready2use_v4.xlsx"
LTS_CSV = f"{ROOT}/kinspec_mac_training_pool_karaman_per_molecule.csv"
FIG_DIR = f"{BASE}/figures_final"
OUT_PRIMARY_CSV = f"{BASE}/KinSPEC_Karaman_final_paired_stats_primary.csv"
OUT_SENSITIVITY_CSV = f"{BASE}/KinSPEC_Karaman_final_paired_stats_sensitivity.csv"
MERGED_CSV_PREFIX = f"{BASE}/KinSPEC_MAC_kinase_selectivity_final"

METRICS = ["AUROC", "BEDROC", "Precision@10", "Recall@10"]
# xlsx sheet -> column name in the LTS per-molecule csv
LTS_COL = {"AUROC": "AUROC", "BEDROC": "BEDROC_a20",
           "Precision@10": "P@10", "Recall@10": "R@10"}

models = ["KinSPEC", "HitScreen", "PSICHIC", "MolTrans", "TransformerCPI", "KinSPEC_LTS"]

colors = {
    "KinSPEC": "#4C72B0",
    "HitScreen": "#B8B8B8",
    "PSICHIC": "#9E9E9E",
    "MolTrans": "#8C8C8C",
    "TransformerCPI": "#7A7A7A",
    "KinSPEC_LTS": "#6E6E6E",
}

# The four baseline comparisons form an m=4 family; the separate LTS comparison forms an m=1 family
BASELINE_CMPS = [
    ("KinSPEC", "HitScreen"),
    ("KinSPEC", "PSICHIC"),
    ("KinSPEC", "MolTrans"),
    ("KinSPEC", "TransformerCPI"),
]
LTS_CMP = ("KinSPEC_LTS", "KinSPEC")
comparisons = BASELINE_CMPS + [LTS_CMP]

N_BOOT = 10000
SEED = 20251005
ALPHA = 0.05


def p_to_stars(p):
    if p < 1e-4:
        return "****"
    elif p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 5e-2:
        return "*"
    else:
        return "ns"


def rank_biserial(diff):
    """Matched-pairs rank-biserial correlation (W+ - W-)/(W+ + W-); zeros dropped."""
    d = np.asarray(diff, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return np.nan
    r = rankdata(np.abs(d))
    w_pos, w_neg = r[d > 0].sum(), r[d < 0].sum()
    total = w_pos + w_neg
    return (w_pos - w_neg) / total if total else np.nan


def boot_ci(diff, stat_fn, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    """Percentile bootstrap CI, resampling molecules (paired units) with replacement."""
    d = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    vals = np.array([stat_fn(d[i]) for i in idx], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return (np.percentile(vals, 100 * alpha / 2),
            np.percentile(vals, 100 * (1 - alpha / 2)))


def load_metric(metric, drop_too_promiscuous=False):
    """Merge baseline/KinSPEC columns with the v4 LTS results for one metric."""
    df = pd.read_excel(IN_XLSX, sheet_name=metric)
    lts = pd.read_csv(LTS_CSV)
    col = LTS_COL[metric]

    df = df.drop(columns=["KinSPEC_LTS"])          # Drop the stale LTS column from the input workbook
    keep = ["molecule_name", col, "group"]
    df = df.merge(lts[keep], on="molecule_name", how="inner")
    df = df.rename(columns={col: "KinSPEC_LTS"})

    if drop_too_promiscuous:
        df = df[~df["group"].astype(str).str.startswith("EXCLUDED")]
    return df.dropna(subset=models).reset_index(drop=True)


def analyse(df, metric):
    """Two-sided Wilcoxon; Holm over the 4 baseline comparisons, LTS left uncorrected."""
    rows, p_raw_family = [], []

    for m1, m2 in comparisons:
        diff = df[m1].to_numpy(float) - df[m2].to_numpy(float)
        in_family = (m1, m2) in BASELINE_CMPS

        if np.all(diff == 0):
            p, med, med_lo, med_hi = 1.0, 0.0, 0.0, 0.0
            r_rb = rrb_lo = rrb_hi = np.nan
            win = 0.5
        else:
            p = wilcoxon(diff, zero_method='pratt', alternative='two-sided').pvalue
            med = float(np.median(diff))
            med_lo, med_hi = boot_ci(diff, np.median)
            r_rb = rank_biserial(diff)
            rrb_lo, rrb_hi = boot_ci(diff, rank_biserial)
            win = float(np.mean(diff > 0))

        if in_family:
            p_raw_family.append(p)

        rows.append({
            "metric": metric, "Comparison": f"{m1} vs {m2}", "n": int(diff.size),
            "ΔMedian": med, "95% CI lower": med_lo, "95% CI upper": med_hi,
            "r_rb": r_rb, "r_rb CI lower": rrb_lo, "r_rb CI upper": rrb_hi,
            "Win%": win, "p_raw": p, "holm_family": in_family,
        })

    # Apply Holm-Bonferroni correction within the four-baseline comparison family (m=4)
    _, p_adj, _, _ = multipletests(p_raw_family, method='holm')
    it = iter(p_adj)
    for r in rows:
        if r["holm_family"]:
            r["p_holm"] = float(next(it))
            r["p_used"] = r["p_holm"]
        else:
            r["p_holm"] = np.nan           # This comparison forms its own m=1 family, so Holm equals p_raw
            r["p_used"] = r["p_raw"]
        r["stars"] = p_to_stars(r["p_used"])
    return rows


def add_sig_line(ax, x1, x2, y, h, text, dashed=False):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c="black",
            ls="--" if dashed else "-")
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom', fontsize=10)


def make_figure(df, metric, stats_rows):
    data = [df[m].to_numpy(float) for m in models]
    positions = np.arange(1, len(models) + 1)

    fig, ax = plt.subplots(figsize=(8, 5.8))

    parts = ax.violinplot(data, positions=positions, widths=0.8,
                          showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[models[i]])
        body.set_edgecolor("black")
        body.set_linewidth(0.8)
        body.set_alpha(0.75)

    bp = ax.boxplot(data, positions=positions, widths=0.18, patch_artist=True,
                    showfliers=False, showcaps=False,
                    medianprops=dict(color="black", linewidth=1.3),
                    whiskerprops=dict(color="black", linewidth=0.8),
                    capprops=dict(color="black", linewidth=0.8),
                    boxprops=dict(edgecolor="black", linewidth=0.8))
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.95)

    data_min = min(d.min() for d in data)
    data_max = max(d.max() for d in data)
    rng = data_max - data_min

    for pos, d in zip(positions, data):
        med = float(np.median(d))
        ax.text(pos + 0.12, med + 0.01 * rng, f"{med:.3f}",
                ha='left', va='bottom', fontsize=10)

    by_cmp = {r["Comparison"]: r for r in stats_rows}
    model_to_pos = {m: i + 1 for i, m in enumerate(models)}
    line_start = data_max + 0.04 * rng
    step, h = 0.085 * rng, 0.028 * rng

    for idx, (m1, m2) in enumerate(comparisons):
        r = by_cmp[f"{m1} vs {m2}"]
        # The dashed divider distinguishes test families; omit an additional dagger marker
        add_sig_line(ax, model_to_pos[m1], model_to_pos[m2],
                     line_start + idx * step, h, r["stars"],
                     dashed=not r["holm_family"])

    ax.set_xticks(positions)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=11)
    ax.set_ylabel(f"Per-molecule {metric}", fontsize=12)
    ax.set_title(f"Karaman kinase selectivity test (n={len(df)})", fontsize=13, pad=10)
    ax.set_ylim(bottom=data_min - 0.04 * rng,
                top=line_start + (len(comparisons) - 1) * step + h + 0.09 * rng)

    ticks = [t for t in ax.get_yticks() if data_min - 0.04 * rng <= t <= data_max + 0.01 * rng]
    ax.set_yticks(ticks)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis='both', width=1.0, length=4)

    plt.tight_layout()
    tag = metric.replace("@", "_top")
    for ext in ("svg", "pdf"):
        fig.savefig(f"{FIG_DIR}/Karaman_final_{tag}_violin_box.{ext}", bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/Karaman_final_{tag}_violin_box.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def tidy_frame(rows):
    return pd.DataFrame(rows)[[
        "metric", "Comparison", "n", "ΔMedian", "95% CI lower", "95% CI upper",
        "r_rb", "r_rb CI lower", "r_rb CI upper", "Win%",
        "p_raw", "p_holm", "holm_family", "stars"]]


def run(drop_too_promiscuous, make_figs, label):
    rows, n_used, merged = [], None, {}
    for metric in METRICS:
        df = load_metric(metric, drop_too_promiscuous)
        n_used = len(df)
        rr = analyse(df, metric)
        rows.extend(rr)
        merged[metric] = df
        if make_figs:
            make_figure(df, metric, rr)

        print(f"\n=== {metric}  ({label}, n={n_used}) ===")
        hdr = (f"{'Comparison':<28} {'ΔMedian':>10} {'95% CI':>20} {'r_rb':>7} "
               f"{'Win%':>7} {'p_raw':>10} {'p_holm':>10} {'stars':>6}")
        print(hdr); print("-" * len(hdr))
        for r in rr:
            ci = f"[{r['95% CI lower']:+.4f}, {r['95% CI upper']:+.4f}]"
            ph = f"{r['p_holm']:>10.2e}" if r["holm_family"] else f"{'n/a':>10}"
            star = r["stars"] if r["holm_family"] else r["stars"] + "†"
            print(f"{r['Comparison']:<28} {r['ΔMedian']:>+10.4f} {ci:>20} "
                  f"{r['r_rb']:>7.3f} {r['Win%']*100:>6.1f}% "
                  f"{r['p_raw']:>10.2e} {ph} {star:>6}")
    return rows, n_used, merged


def main():
    import os
    os.makedirs(FIG_DIR, exist_ok=True)

    print("#" * 100)
    print("PRIMARY — all molecules shared with the baseline benchmark")
    main_rows, n_main, merged = run(False, True, "primary")

    print("\n" + "#" * 100)
    print("SENSITIVITY — dropping the molecules the LTS pipeline flags EXCLUDED_too_promiscuous")
    sens_rows, n_sens, _ = run(True, False, "sensitivity")

    tidy_frame(main_rows).to_csv(OUT_PRIMARY_CSV, index=False)
    tidy_frame(sens_rows).to_csv(OUT_SENSITIVITY_CSV, index=False)

    for metric, df in merged.items():
        df[["molecule_name"] + models + ["group"]].to_csv(
            f"{MERGED_CSV_PREFIX}_{metric}.csv", index=False)

    print(f"\nFigures -> {FIG_DIR}/Karaman_final_<metric>_violin_box.{{svg,pdf,png}}")
    print(f"Stats   -> {OUT_PRIMARY_CSV}, {OUT_SENSITIVITY_CSV}")
    print(f"Data    -> {MERGED_CSV_PREFIX}_<metric>.csv  (baselines + v4 KinSPEC_LTS)")


if __name__ == "__main__":
    main()
