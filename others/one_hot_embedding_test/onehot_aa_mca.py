#!/usr/bin/env python
"""
One-hot amino-acid embedding test for KLIFS pocket sequences + MCA.

1. Each of the 20 standard amino acids -> 20-bit one-hot code (19 zeros, one 1).
   Gap '-' -> all 20 bits are 0.
2. 482 kinases x 85 KLIFS residues -> 482 x 1700 (85 x 20) binary features.
3. MCA on the 85 categorical position variables. The all-zero gap code is
   decoded as an explicit "Gap" category, so every kinase has exactly one active
   category per position (required for MCA; otherwise gapped rows get smaller
   margins and the indicator matrix is no longer a complete disjunctive table).
4. Dimension-wise evaluation:
     Cumulative inertia, cumulative Benzécri-corrected inertia,
     cumulative Greenacre-corrected inertia, mean / median / lower quartile /
     minimum row cos2, and median bootstrap pairwise-distance Spearman.

MCA is computed exactly as correspondence analysis of the indicator matrix
(full SVD), giving the same eigenvalues and row principal coordinates as
prince.MCA but without randomized-SVD approximation for the full spectrum.

Run with the KinSPEC env:
    python onehot_aa_mca.py
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
_kp = _ks_partial(_ks_project_path, script_dir='others/one_hot_embedding_test')


import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
GAP = "-"
N_RESIDUES = 85
N_AA = len(AMINO_ACIDS)

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parent / _kp('feature_embedding/Kinase_test_MCA/human_482_typical_kinase_1955bit.csv')


# ----------------------------------------------------------------------------
# One-hot encoding
# ----------------------------------------------------------------------------
def encode_onehot(sequences):
    """KLIFS sequences -> [n, 85 * 20] binary matrix; gaps are all zeros."""
    X = np.zeros((len(sequences), N_RESIDUES * N_AA), dtype=np.int8)
    for i, seq in enumerate(sequences):
        if len(seq) != N_RESIDUES:
            raise ValueError(f"Sequence {i} has length {len(seq)}, expected {N_RESIDUES}.")
        for r, aa in enumerate(seq):
            if aa == GAP:
                continue
            if aa not in AA_INDEX:
                raise ValueError(f"Sequence {i}, residue {r + 1}: unknown symbol '{aa}'.")
            X[i, r * N_AA + AA_INDEX[aa]] = 1
    return X


def onehot_column_names():
    return [f"res{r + 1:02d}_{aa}" for r in range(N_RESIDUES) for aa in AMINO_ACIDS]


def onehot_to_categorical(X):
    """[n, 1700] one-hot -> [n, 85] categorical table (all-zero block = 'Gap')."""
    X = np.asarray(X)
    if X.shape[1] != N_RESIDUES * N_AA:
        raise ValueError(f"Expected {N_RESIDUES * N_AA} columns, got {X.shape[1]}.")
    data = {}
    for r in range(N_RESIDUES):
        block = X[:, r * N_AA:(r + 1) * N_AA]
        row_sums = block.sum(axis=1)
        if np.any(row_sums > 1):
            raise ValueError(f"res{r + 1:02d}: multiple active amino acids in one row.")
        labels = np.full(X.shape[0], "Gap", dtype=object)
        valid = row_sums == 1
        labels[valid] = np.array(AMINO_ACIDS)[block[valid].argmax(axis=1)]
        data[f"res{r + 1:02d}"] = pd.Categorical(labels, categories=AMINO_ACIDS + ["Gap"])
    df_cat = pd.DataFrame(data)
    # Drop categories never observed, so the indicator matrix has no all-zero columns.
    for col in df_cat.columns:
        df_cat[col] = df_cat[col].cat.remove_unused_categories()
    return df_cat


# ----------------------------------------------------------------------------
# MCA (exact CA of the complete disjunctive table)
# ----------------------------------------------------------------------------
def fit_mca(df_cat):
    """
    Returns eigenvalues (all non-trivial axes), row principal coordinates,
    Q (number of variables) and J (number of observed categories).
    """
    Z = pd.get_dummies(df_cat).to_numpy(dtype=float)
    Z = Z[:, Z.sum(axis=0) > 0]
    n, J = Z.shape
    Q = df_cat.shape[1]

    P = Z / Z.sum()
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, sigma, _ = np.linalg.svd(S, full_matrices=False)

    max_dim = min(n - 1, J - Q)
    eig = sigma[:max_dim] ** 2
    F = (U[:, :max_dim] * sigma[:max_dim]) / np.sqrt(r)[:, None]

    raw_total = (J - Q) / Q
    if not np.isclose(eig.sum(), raw_total, rtol=1e-6):
        raise ValueError(f"Eigenvalue sum {eig.sum():.6g} != (J-Q)/Q = {raw_total:.6g}.")
    return eig, F, Q, J


def row_cos2(F, k):
    d2_all = np.sum(F ** 2, axis=1)
    d2_k = np.sum(F[:, :k] ** 2, axis=1)
    return np.divide(d2_k, d2_all, out=np.zeros_like(d2_k), where=d2_all > 0)


def inertia_spectrum(eig, Q, J):
    """Raw, Benzécri and Greenacre cumulative inertia with fixed denominators."""
    raw_total = (J - Q) / Q
    adjusted = (Q / (Q - 1) * np.maximum(eig - 1 / Q, 0)) ** 2
    benzecri_total = adjusted.sum()
    greenacre_total = Q / (Q - 1) * (np.sum(eig ** 2) - (J - Q) / Q ** 2)

    def cum_pct(values, total):
        if total <= 1e-12:
            return np.full(len(values), np.nan)
        return 100 * np.cumsum(values) / total

    spectrum = pd.DataFrame({
        "n_components": np.arange(1, len(eig) + 1),
        "raw_eigenvalue": eig,
        "benzecri_corrected_eigenvalue": adjusted,
        "cumulative_inertia_pct": cum_pct(eig, raw_total),
        "cumulative_benzecri_inertia_pct": cum_pct(adjusted, benzecri_total),
        "cumulative_greenacre_inertia_pct": cum_pct(adjusted, greenacre_total),
    })
    metadata = {
        "n_variables_Q": Q,
        "n_categories_J": J,
        "n_fitted_components": len(eig),
        "correction_threshold_1_over_Q": 1 / Q,
        "n_axes_above_threshold": int(np.sum(eig > 1 / Q)),
        "raw_total_inertia": raw_total,
        "benzecri_total_inertia": benzecri_total,
        "greenacre_adjusted_total_inertia": greenacre_total,
    }
    return spectrum, metadata


# ----------------------------------------------------------------------------
# Bootstrap stability: Spearman of pairwise distances (reference vs bootstrap)
# ----------------------------------------------------------------------------
def bootstrap_distance_spearman(df_cat, F_ref, dims, n_boot=100, seed=42):
    rng = np.random.default_rng(seed)
    n = len(df_cat)
    rows = []
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        df_boot = df_cat.iloc[idx].reset_index(drop=True)
        for col in df_boot.columns:
            df_boot[col] = df_boot[col].cat.remove_unused_categories()
        _, F_boot, _, _ = fit_mca(df_boot)

        # Duplicated kinases have identical coordinates; keep one per original ID.
        uniq_ids, first_pos = np.unique(idx, return_index=True)
        F_boot = F_boot[first_pos]
        F_ref_u = F_ref[uniq_ids]

        for k in dims:
            k_used = min(k, F_boot.shape[1])
            rho = spearmanr(pdist(F_ref_u[:, :k_used]), pdist(F_boot[:, :k_used])).statistic
            rows.append({
                "bootstrap_round": b + 1,
                "n_components": k,
                "n_components_used": k_used,
                "n_unique_samples": len(uniq_ids),
                "pairwise_distance_spearman": rho,
            })
        print(f"  bootstrap {b + 1}/{n_boot}", end="\r", flush=True)
    print()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--n-components", type=int, default=200,
                    help="Dimension of the saved MCA feature matrix.")
    ap.add_argument("--dims", type=int, nargs="+",
                    default=[10, 20, 30, 50, 75, 100, 150, 200, 250, 300],
                    help="Dimensions to evaluate (max dimension is always added).")
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. One-hot encoding ----------------------------------------------------
    df = pd.read_csv(args.input)
    X = encode_onehot(df["KLIFS"].tolist())
    print(f"One-hot features: {X.shape[0]} kinases x {X.shape[1]} bits")
    onehot_df = pd.DataFrame(X, columns=onehot_column_names())
    onehot_df.insert(0, "UniProt_ID", df["UniProt_ID"].values)
    onehot_df.insert(0, "kinase_name", df["kinase_name"].values)
    onehot_df.to_csv(outdir / "KLIFS_482_onehot_1700D.csv", index=False)

    # 2. MCA -----------------------------------------------------------------
    df_cat = onehot_to_categorical(X)
    eig, F, Q, J = fit_mca(df_cat)
    max_dim = len(eig)
    n_constant = int(sum(df_cat[c].nunique() == 1 for c in df_cat.columns))
    print(f"MCA: Q = {Q} positions, J = {J} observed categories, "
          f"{n_constant} constant positions, {max_dim} non-trivial axes")

    spectrum, metadata = inertia_spectrum(eig, Q, J)
    metadata = {"n_samples": len(df_cat), **metadata, "n_constant_positions": n_constant}

    # 3. Evaluation ----------------------------------------------------------
    dims = sorted({k for k in args.dims + [args.n_components, max_dim] if 1 <= k <= max_dim})
    print(f"Bootstrap ({args.n_boot} rounds) at dims {dims}")
    boot = bootstrap_distance_spearman(df_cat, F, dims, n_boot=args.n_boot, seed=args.seed)
    boot_median = boot.groupby("n_components")["pairwise_distance_spearman"].median()

    rows = []
    for k in dims:
        s = spectrum.iloc[k - 1]
        cos2 = row_cos2(F, k)
        rows.append({
            "n_components": k,
            "Cumulative inertia (%)": s["cumulative_inertia_pct"],
            "Cumulative Benzécri-corrected inertia (%)": s["cumulative_benzecri_inertia_pct"],
            "Cumulative Greenacre-corrected inertia (%)": s["cumulative_greenacre_inertia_pct"],
            "Mean row cos2": cos2.mean(),
            "Median row cos2": np.median(cos2),
            "Lower quartile row cos2": np.quantile(cos2, 0.25),
            "Minimum row cos2": cos2.min(),
            "Median pairwise distance Spearman": boot_median.loc[k],
        })
    evaluation = pd.DataFrame(rows)

    # 4. Save ----------------------------------------------------------------
    k = args.n_components
    coords = pd.DataFrame(F[:, :k], columns=[f"mca_{i + 1}" for i in range(k)])
    coords.to_csv(outdir / f"onehot_MCA_klifs_482_{k}D_features.csv", index=False)
    pd.DataFrame({
        "kinase_name": df["kinase_name"], "UniProt_ID": df["UniProt_ID"],
        f"row_cos2_{k}D": row_cos2(F, k),
    }).to_csv(outdir / f"onehot_MCA_row_cos2_{k}D.csv", index=False)
    spectrum.to_csv(outdir / "onehot_MCA_inertia_spectrum.csv", index=False)
    pd.DataFrame([metadata]).to_csv(outdir / "onehot_MCA_inertia_metadata.csv", index=False)
    boot.to_csv(outdir / "onehot_MCA_bootstrap_stability.csv", index=False)
    evaluation.to_csv(outdir / "onehot_MCA_dimension_evaluation.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in [("cumulative_inertia_pct", "Raw"),
                       ("cumulative_benzecri_inertia_pct", "Benzécri corrected"),
                       ("cumulative_greenacre_inertia_pct", "Greenacre corrected")]:
        ax.plot(spectrum["n_components"], spectrum[col], label=label)
    ax.set(xlabel="Number of MCA dimensions", ylabel="Cumulative inertia (%)",
           title="One-hot amino-acid MCA, 482 KLIFS pockets")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "onehot_MCA_cumulative_inertia.png", dpi=200)

    with pd.option_context("display.width", 250, "display.max_columns", None,
                           "display.float_format", "{:.4f}".format):
        print("\nInertia metadata:")
        print(pd.DataFrame([metadata]).T.to_string(header=False))
        print("\nDimension-wise evaluation:")
        print(evaluation.set_index("n_components").T.to_string())


if __name__ == "__main__":
    main()
