# -*- coding: utf-8 -*-
"""
Evaluation plots for KIBA seen targets with LUDe decoy augmentation (sim08).

- Per-target violin plots for AUROC, BEDROC, EF1%, and EF5%, with boxes and median labels.
- Two-sided Wilcoxon signed-rank tests with Holm-Bonferroni correction across the four KinSPEC/baseline comparisons.
- Rank-biserial correlation, bootstrap 95% confidence intervals, and win rates.
- Overall horizontal bar charts showing five-fold means and standard deviations.

Match draw_KIBA_figure/figures/plot_KIBA_figures.py plotting conventions and
align statistics columns with draw_KIBA_figure/figures/wilcoxon_holm_per_target_stats.csv.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KIBA_seen_target_LUDe_aug_sim08")
OUT = os.path.join(ROOT, "figure")
os.makedirs(OUT, exist_ok=True)

MODELS = ["KinSPEC", "HitScreen", "PSICHIC", "MolTrans", "TransformerCPI"]
REFERENCE = "KinSPEC"
METRICS = ["AUROC", "BEDROC", "EF1%", "EF5%", "EF10%"]
TITLE = "KIBA seen target (n={n}) with LUDe augmentation"

# Bootstrap settings, consistent with draw_KIBA_figure
N_BOOT = 10000
SEED = 42

# Plot style: highlight KinSPEC and use grayscale for baselines
COLORS = {
    "KinSPEC":        "#4C72B0",
    "HitScreen":      "#B8B8B8",
    "PSICHIC":        "#9E9E9E",
    "MolTrans":       "#8C8C8C",
    "TransformerCPI": "#7A7A7A",
}

# Per-target files and EF column names, which differ among models
PER_TARGET = {
    "KinSPEC":        ("KinSPEC/xgb_eval_per_target_mean.csv",
                       {"EF1%": "EF_1%_mean", "EF5%": "EF_5%_mean",
                         "EF10%": "EF_10%_mean"}),
    "HitScreen":      ("HitScreen/evaluation_metrics_KIBA_augment.csv",
                       {"EF1%": "EF_1%_mean", "EF5%": "EF_5%_mean",
                         "EF10%": "EF_10%_mean"}),
    "PSICHIC":        ("PSICHIC/psichic_eval_per_target_mean.csv",
                       {"EF1%": "EF_1.0%_mean", "EF5%": "EF_5.0%_mean",
                         "EF10%": "EF_10.0%_mean"}),
    "MolTrans":       ("MolTrans/KIBA_seen_LUDe_augment_sim08_result.csv",
                       {"EF1%": "EF_1%_mean", "EF5%": "EF_5%_mean",
                         "EF10%": "EF_10%_mean"}),
    "TransformerCPI": ("TransformerCPI/kiba_10_50_per_target_summary.csv",
                       {"EF1%": "EF_1%_mean", "EF5%": "EF_5%_mean",
                         "EF10%": "EF_10%_mean"}),
}

# Per-fold overall files: paths, fold columns, and metric columns
# KinSPEC/PSICHIC overall_all_folds includes EF_2% but not EF_5%
# Plot overall AUROC, BEDROC, and EF1%; per-target EF5% is unaffected
PER_FOLD = {
    "KinSPEC":        ("KinSPEC/xgb_eval_overall_all_folds.csv", "Fold",
                       {"AUROC": "ROC_AUC", "BEDROC": "BEDROC", "EF1%": "EF_1.0%"}),
    "HitScreen":      ("HitScreen/evaluation_metrics_KIBA_augment_overall.csv", "fold",
                       {"AUROC": "ROC_AUC", "BEDROC": "BEDROC",
                        "EF1%": "EF_1%", "EF5%": "EF_5%"}),
    "PSICHIC":        ("PSICHIC/psichic_eval_overall_all_folds.csv", "Fold",
                       {"AUROC": "ROC_AUC", "BEDROC": "BEDROC", "EF1%": "EF_1.0%"}),
    "TransformerCPI": ("TransformerCPI/kiba_10_50_overall_all_folds.csv", "Fold",
                       {"AUROC": "ROC_AUC", "BEDROC": "BEDROC",
                        "EF1%": "EF_1%", "EF5%": "EF_5%"}),
}
# MolTrans supplies aggregate Mean/Std values rather than per-fold overall files
MOLTRANS_OVERALL = ("MolTrans/KIBA_seen_LUDe_augment_sim08_result_overall.csv",
                    {"AUROC": "ROC_AUC", "BEDROC": "BEDROC", "EF1%": "EF_1%"})

# =========================
# 1) Load data
# =========================
def load_per_target():
    frames = {}
    for model, (rel, ef_cols) in PER_TARGET.items():
        df = pd.read_csv(os.path.join(ROOT, rel))
        df = df[df["Target"].astype(str) != "ALL_TARGETS_MEAN"]  # MolTrans aggregate row
        cols = {"AUROC": "ROC_AUC_mean", "BEDROC": "BEDROC_mean", **ef_cols}
        frames[model] = pd.DataFrame(
            {m: df[c].astype(float).values for m, c in cols.items()},
            index=df["Target"].astype(str).values)
    common = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
    return pd.concat({m: f.loc[common] for m, f in frames.items()}, axis=1)


def load_overall():
    """Return {model: {metric: (mean, std)}}; omit metrics absent from the input file."""
    out = {}
    for model, (rel, fold_col, cols) in PER_FOLD.items():
        df = pd.read_csv(os.path.join(ROOT, rel))
        keys = df[fold_col].astype(str).str.lower()
        df = df[~keys.isin(["mean", "std", "sem", "median"])]
        assert len(df) == 5, f"{model}: expected 5 folds, got {len(df)}"
        out[model] = {m: (float(df[c].astype(float).mean()),
                          float(df[c].astype(float).std(ddof=1)))
                      for m, c in cols.items()}

    if MOLTRANS_OVERALL is not None:
        rel, colmap = MOLTRANS_OVERALL
        sm = pd.read_csv(os.path.join(ROOT, rel)).set_index("Metric")
        out["MolTrans"] = {m: (float(sm.loc[c, "Mean"]), float(sm.loc[c, "Std"]))
                           for m, c in colmap.items()}
    out.setdefault("MolTrans", {})
    return out


# =========================
# 2) Significance and effect sizes
# =========================
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
    """Paired rank-biserial correlation (Kerby 2014).
    r_rb = (W+ - W-) / (W+ + W-), using ranks of |diff| and excluding zero differences."""
    d = diff[diff != 0]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    w_pos, w_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def _rank_biserial_rows(mat):
    ranks = rankdata(np.abs(mat), axis=1)
    pos = np.where(mat > 0, ranks, 0.0).sum(axis=1)
    neg = np.where(mat < 0, ranks, 0.0).sum(axis=1)
    den = pos + neg
    return np.divide(pos - neg, den, out=np.zeros_like(den), where=den > 0)


def bootstrap_ci(diff, n_boot=N_BOOT, seed=SEED):
    """Calculate paired percentile bootstrap 95% intervals for median differences and r_rb by resampling targets."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    mat = diff[idx]
    return (tuple(np.percentile(np.median(mat, axis=1), [2.5, 97.5])),
            tuple(np.percentile(_rank_biserial_rows(mat), [2.5, 97.5])))


def run_stats(per_target, metric):
    baselines = [m for m in MODELS if m != REFERENCE]
    x = per_target[(REFERENCE, metric)].values
    rows, p_raw = [], []
    for b in baselines:
        y = per_target[(b, metric)].values
        diff = x - y
        if np.all(diff == 0):
            p, stat = 1.0, np.nan
            ci_med, ci_rrb = (0.0, 0.0), (0.0, 0.0)
        else:
            stat, p = wilcoxon(diff, zero_method="pratt", alternative="two-sided")
            ci_med, ci_rrb = bootstrap_ci(diff)
        p_raw.append(p)
        rows.append({
            "baseline": b,
            "n_targets": len(diff),
            "n_ties": int(np.sum(diff == 0)),
            "median_KinSPEC": float(np.median(x)),
            "median_baseline": float(np.median(y)),
            "median_diff": float(np.median(diff)),
            "median_diff_ci_low": float(ci_med[0]),
            "median_diff_ci_high": float(ci_med[1]),
            "rank_biserial": rank_biserial(diff),
            "rank_biserial_ci_low": float(ci_rrb[0]),
            "rank_biserial_ci_high": float(ci_rrb[1]),
            "win_rate": float(np.mean(diff > 0)),
            "W": float(stat) if stat == stat else np.nan,
            "p_raw": p,
        })
    _, p_adj, _, _ = multipletests(p_raw, method="holm")
    for r, pa in zip(rows, p_adj):
        r["p_holm"] = float(pa)
        r["stars"] = p_to_stars(pa)
    return rows


# =========================
# 3) Violin plots
# =========================
def plot_violin(metric, per_target, stats_rows):
    data = [per_target[(m, metric)].values for m in MODELS]
    positions = np.arange(1, len(MODELS) + 1)

    fig, ax = plt.subplots(figsize=(8, 5.6))

    parts = ax.violinplot(data, positions=positions, widths=0.8,
                          showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[MODELS[i]])
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

    d_min = min(np.min(d) for d in data)
    d_max = max(np.max(d) for d in data)
    span = max(d_max - d_min, 1e-6)

    for i, m in enumerate(MODELS):
        med = float(np.median(per_target[(m, metric)].values))
        fmt = f"{med:.3f}" if not metric.startswith("EF") else f"{med:.2f}"
        ax.text(positions[i] + 0.11, med + 0.01 * span, fmt,
                ha="left", va="bottom", fontsize=9)

    # Stack significance brackets from narrow to wide; center stars above brackets and effect sizes below
    model_to_pos = {m: i + 1 for i, m in enumerate(MODELS)}
    rows_sorted = sorted(stats_rows,
                         key=lambda r: abs(model_to_pos[r["baseline"]] - model_to_pos[REFERENCE]))
    step, h = 0.155 * span, 0.028 * span
    y0 = d_max + 0.075 * span

    for idx, r in enumerate(rows_sorted):
        x1, x2 = model_to_pos[REFERENCE], model_to_pos[r["baseline"]]
        xc, y = (x1 + x2) / 2, y0 + idx * step
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c="black", clip_on=False)
        star = r["stars"]
        ax.text(xc, y + h + (0.018 * span if star != "ns" else 0.024 * span),
                star, ha="center", va="bottom",
                fontsize=11 if star != "ns" else 9,
                fontweight="bold" if star != "ns" else "normal")
        ax.text(xc, y + h - 0.012 * span, f"$r_{{rb}}$ = {r['rank_biserial']:.2f}",
                ha="center", va="top", fontsize=8, color="#444444")

    ax.set_xticks(positions)
    ax.set_xticklabels(MODELS, rotation=25, ha="right", fontsize=11)
    ax.set_ylabel(f"Per-target {metric}", fontsize=12)
    ax.set_title(TITLE.format(n=len(per_target)), fontsize=13, pad=10)
    ax.set_ylim(bottom=d_min - 0.06 * span,
                top=y0 + (len(rows_sorted) - 1) * step + h + 0.105 * span)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", width=1.0, length=4)

    plt.tight_layout()
    safe = metric.replace("%", "pct").replace(".", "p")
    plt.savefig(os.path.join(OUT, f"violin_KIBA_seen_LUDe_sim08_{safe}.png"),
                dpi=600, bbox_inches="tight")
    plt.close(fig)


# =========================
# 4) Overall bar charts
# =========================
def plot_histogram(metric, overall, n_targets):
    missing = [m for m in MODELS if metric not in overall[m]]
    if missing:
        print(f"[skip] overall {metric} bar chart: this metric is missing from the overall files for these models -> "
              f"{', '.join(missing)}")
        return False
    display_order = MODELS[::-1]  # Show KinSPEC at the top
    means = [overall[m][metric][0] for m in display_order]
    stds = [overall[m][metric][1] for m in display_order]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    y_pos = np.arange(len(display_order))

    ax.barh(y_pos, means, xerr=stds, color=[COLORS[m] for m in display_order],
            edgecolor="black", linewidth=0.8, height=0.65, alpha=0.85,
            error_kw=dict(ecolor="black", capsize=3, capthick=0.8, elinewidth=0.8))

    x_max = max(np.array(means) + np.array(stds))
    for i, (m, s) in enumerate(zip(means, stds)):
        fmt = f"{m:.3f} ± {s:.3f}" if not metric.startswith("EF") else f"{m:.2f} ± {s:.2f}"
        ax.text(m + s + x_max * 0.02, y_pos[i], fmt, va="center", ha="left", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_order, fontsize=10)
    ax.set_xlabel(metric, fontsize=11)
    ax.set_title(f"Overall {metric} on " + TITLE.format(n=n_targets), fontsize=12, pad=10)
    ax.set_xlim(0, x_max * 1.35)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", width=1.0, length=4)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    safe = metric.replace("%", "pct").replace(".", "p")
    plt.savefig(os.path.join(OUT, f"histogram_KIBA_seen_LUDe_sim08_{safe}.png"),
                dpi=600, bbox_inches="tight")
    plt.close(fig)
    return True


# =========================
# 5) Main workflow
# =========================
per_target = load_per_target()
overall = load_overall()
n = len(per_target)

print(f"KIBA seen target + LUDe augmentation (sim08)   n_targets = {n}")
print(f"targets: {len(per_target)} (first 5: {', '.join(list(per_target.index)[:5])} ...)")

all_stats, all_overall = [], []
for metric in METRICS:
    rows = run_stats(per_target, metric)
    plot_violin(metric, per_target, rows)
    plot_histogram(metric, overall, n)

    print(f"\n--- {metric}: Wilcoxon signed-rank (two-sided), "
          f"Holm-Bonferroni within KinSPEC-vs-baseline family (k={len(rows)}); "
          f"CI = {N_BOOT} paired bootstrap, percentile ---")
    header = (f"{'Comparison':<30}{'ΔMedian':>10}{'ΔMedian 95% CI':>24}"
              f"{'r_rb':>8}{'r_rb 95% CI':>18}{'Win%':>8}"
              f"{'p_raw':>11}{'p_holm':>11}{'stars':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        ci_m = f"[{r['median_diff_ci_low']:+.3f}, {r['median_diff_ci_high']:+.3f}]"
        ci_r = f"[{r['rank_biserial_ci_low']:+.2f}, {r['rank_biserial_ci_high']:+.2f}]"
        print(f"{'KinSPEC vs ' + r['baseline']:<30}"
              f"{r['median_diff']:>+10.4f}{ci_m:>24}"
              f"{r['rank_biserial']:>+8.2f}{ci_r:>18}"
              f"{r['win_rate']*100:>7.1f}%"
              f"{r['p_raw']:>11.2e}{r['p_holm']:>11.2e}{r['stars']:>7}")
        all_stats.append({"setting": "seen_LUDe_sim08", "metric": metric, **r})

    for m in MODELS:
        if metric in overall[m]:
            mu, sd = overall[m][metric]
            all_overall.append({"setting": "seen_LUDe_sim08", "metric": metric,
                                "model": m, "mean_5fold": mu, "std_5fold": sd})

pd.DataFrame(all_stats)[
    ["setting", "metric", "baseline", "n_targets", "n_ties",
     "median_KinSPEC", "median_baseline",
     "median_diff", "median_diff_ci_low", "median_diff_ci_high",
     "rank_biserial", "rank_biserial_ci_low", "rank_biserial_ci_high",
     "win_rate", "W", "p_raw", "p_holm", "stars"]
].to_csv(os.path.join(OUT, "wilcoxon_holm_per_target_stats.csv"), index=False)

pd.DataFrame(all_overall).to_csv(
    os.path.join(OUT, "overall_5fold_mean_std.csv"), index=False)

print(f"\nSaved to {OUT}")
