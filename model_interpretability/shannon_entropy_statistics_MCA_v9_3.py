"""VSDS site entropy using V9.3 reference-matched joint-independent attribution.

Use --references to set the reference count (default 2000), and --output-dir
for an independent result directory. Per-reference TreeSHAP checkpoints allow
restarting the same configuration; entropy and plotting follow the v4_2 analysis.
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
_kp = _ks_partial(_ks_project_path, script_dir='model_interpretability')

import os
import argparse
import hashlib
import json
import time
import KinSPEC_interpretability_v9 as v9
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import shap
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
VSDS_CSV =  _kp('dataset/VSDS_truedecoy_integrated_data_with_protein_seq.csv')

KLIFS_DIR = _kp('feature_embedding/Kinase_test_MCA')
KLIFS_FILE = _kp('feature_embedding/Kinase_test_MCA/human_482_typical_kinase_1955bit.csv')
KLIFS_MCA_CSV = Path(_kp('feature_embedding/Kinase_test_MCA/MCA_klifs_482_200D_features.csv'))
KLIFS_ONEHOT = Path(_kp('feature_embedding/Kinase_test_MCA/human_482_typical_kinase_1955bit_onehot.csv'))

MCA_CATEGORY_COORDS =  _kp('feature_embedding/Kinase_test_MCA/MCA_200D_category_coordinates.csv')
MCA_CATEGORY_CONTRIB =  _kp('feature_embedding/Kinase_test_MCA/MCA_200D_category_contributions.csv')
MCA_CATEGORY_LOOKUP =  _kp('feature_embedding/Kinase_test_MCA/MCA_200D_category_lookup.csv')
SITE_ENTROPY_CSV = Path(_kp('model_interpretability/entropy_analysis_output/residue_entropy_by_KLIFS_position.csv'))

MOL_DIR = Path(_kp('feature_embedding/mol'))
MOL_SAE_PATH = MOL_DIR / "LSAE_chembl_1900k_full_dataset_v4_model.pt"
MOL_SAE_IDX = MOL_DIR / "LSAE_chembl_1900k_full_dataset_v4_topk_of1024d_top99.pt"

XGB_MODEL = _kp('model/KinSPEC/KinSPEC_full_training_pool_MCA_200D_v4.json')

OUTDIR = Path(__file__).resolve().parent / "vsds_active_residue_analysis_MCA_v9_3"



# =========================
# Hyperparameters
# =========================
RADIUS = 2
NBITS = 4096
MOL_KEEP_DIMS = 417
TOPK_PROT_COMPONENTS = None  # use all protein latent dimensions
TOPN_RESIDUES = 10
BITS_PER_RESIDUE = 23
N_RESIDUES = 85
RANDOM_SEED = 42
MAX_ACTIVE_PAIRS = None  # set to a small integer for quick smoke tests
BITSCORE_METHOD = "v9_reference_matched_signed_difference_joint_independent"
REFERENCE_COUNT = 2000




# =========================
# KLIFS region map and entropy aggregation
# =========================
KLIFS_DOMAIN_MAP = {}
for i in range(0, 3):    KLIFS_DOMAIN_MAP[i] = "I"
for i in range(3, 9):    KLIFS_DOMAIN_MAP[i] = "g.l."
for i in range(9, 13):   KLIFS_DOMAIN_MAP[i] = "II"
for i in range(13, 19):  KLIFS_DOMAIN_MAP[i] = "III"
for i in range(19, 30):  KLIFS_DOMAIN_MAP[i] = "alphaC"
for i in range(30, 37):  KLIFS_DOMAIN_MAP[i] = "b.l"
for i in range(37, 41):  KLIFS_DOMAIN_MAP[i] = "IV"
for i in range(41, 44):  KLIFS_DOMAIN_MAP[i] = "V"
KLIFS_DOMAIN_MAP[44] = "GK"
for i in range(45, 48):  KLIFS_DOMAIN_MAP[i] = "hinge"
for i in range(48, 52):  KLIFS_DOMAIN_MAP[i] = "Linker"
for i in range(52, 59):  KLIFS_DOMAIN_MAP[i] = "alphaD"
for i in range(59, 64):  KLIFS_DOMAIN_MAP[i] = "alphaE"
for i in range(64, 67):  KLIFS_DOMAIN_MAP[i] = "VI"
for i in range(67, 75):  KLIFS_DOMAIN_MAP[i] = "c.l"
for i in range(75, 78):  KLIFS_DOMAIN_MAP[i] = "VII"
KLIFS_DOMAIN_MAP[78] = "VIII"
for i in range(79, 83):  KLIFS_DOMAIN_MAP[i] = "xDFG"
for i in range(83, 85):  KLIFS_DOMAIN_MAP[i] = "a.l"

REGIONS_ORDERED = [
    "alphaD", "alphaC", "IV", "Linker", "I", "II", "alphaE", "b.l", "VI",
    "GK", "V", "a.l", "hinge", "III", "g.l.", "VII", "VIII", "c.l", "xDFG",
]

def summarize_region_entropy(site_entropy: np.ndarray) -> pd.DataFrame:
    """Summarize the same 85 site entropies used in per-pair statistics.

    std is the sample standard deviation (ddof=1); single-position regions
    retain NaN, matching the existing table and plot convention.
    """
    values = np.asarray(site_entropy, dtype=float)
    if values.shape != (N_RESIDUES,) or not np.isfinite(values).all():
        raise ValueError("Expected 85 finite site entropy values in KLIFS position order")
    sites = pd.DataFrame({
        "region": [KLIFS_DOMAIN_MAP[i] for i in range(N_RESIDUES)],
        "entropy": values,
    })
    return sites.groupby("region")["entropy"].agg(
        mean="mean", std="std"
    ).reindex(REGIONS_ORDERED)


def load_site_entropy(path: Path) -> np.ndarray:
    """
    Load site-level Shannon entropy for the 85 KLIFS positions.

    Expected columns:
        KLIFS_position   : 1-based KLIFS position (1..85)
        Residue_entropy  : Shannon entropy at that position

    Returns
    -------
    np.ndarray
        Length-85 array in KLIFS position order.
        Array index 0 corresponds to KLIFS position 1.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Site-level entropy file not found: {path}"
        )

    entropy_df = pd.read_csv(path)

    required_cols = {
        "KLIFS_position",
        "Residue_entropy",
    }

    missing = required_cols.difference(entropy_df.columns)

    if missing:
        raise ValueError(
            f"Missing required entropy columns: {sorted(missing)}"
        )

    entropy_df = entropy_df[
        ["KLIFS_position", "Residue_entropy"]
    ].copy()

    entropy_df["KLIFS_position"] = pd.to_numeric(
        entropy_df["KLIFS_position"],
        errors="raise"
    ).astype(int)

    entropy_df["Residue_entropy"] = pd.to_numeric(
        entropy_df["Residue_entropy"],
        errors="raise"
    ).astype(float)

    # Check unique KLIFS positions
    if entropy_df["KLIFS_position"].duplicated().any():
        duplicated = entropy_df.loc[
            entropy_df["KLIFS_position"].duplicated(),
            "KLIFS_position"
        ].tolist()

        raise ValueError(
            f"Duplicated KLIFS positions in entropy file: {duplicated}"
        )

    entropy_df = entropy_df.sort_values(
        "KLIFS_position"
    ).reset_index(drop=True)

    expected_positions = np.arange(1, N_RESIDUES + 1)

    if len(entropy_df) != N_RESIDUES:
        raise ValueError(
            f"Expected {N_RESIDUES} site-level entropy values, "
            f"but found {len(entropy_df)}."
        )

    if not np.array_equal(
        entropy_df["KLIFS_position"].to_numpy(),
        expected_positions
    ):
        raise ValueError(
            "KLIFS entropy positions must contain every position "
            f"from 1 to {N_RESIDUES} exactly once."
        )

    site_entropy = entropy_df[
        "Residue_entropy"
    ].to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(site_entropy)):
        raise ValueError(
            "Non-finite values detected in site-level entropy."
        )

    return site_entropy


# =========================
# SAE class for molecule encoder
# =========================
class SparseLinearAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def find_col(df, candidates):
    """Case- and underscore-insensitive column resolver."""
    norm = {c.lower().replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace("_", "")
        if key in norm:
            return norm[key]
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def bitstring_to_array(bitstr: str) -> np.ndarray:
    bitstr = str(bitstr).strip()
    if len(bitstr) != N_RESIDUES * BITS_PER_RESIDUE:
        raise ValueError(f"Expected 1955 KLIFS bits, got {len(bitstr)}")
    return np.frombuffer(bitstr.encode("ascii"), dtype=np.uint8) - ord("0")


def clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert CSV cells such as '[3.8745418E-1]' to numeric floats."""
    cleaned = df.copy()
    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            cleaned[col] = (
                cleaned[col]
                .astype(str)
                .str.strip()
                .str.replace(r"^\[\s*", "", regex=True)
                .str.replace(r"\s*\]$", "", regex=True)
            )
        cleaned[col] = pd.to_numeric(cleaned[col], errors="raise")
    return cleaned


def numeric_to_numpy(df: pd.DataFrame, dtype=np.float64) -> np.ndarray:
    return clean_numeric_frame(df).to_numpy(dtype=dtype)


def load_embedding_matrix(path: Path, csv_skiprows=1, csv_header=None) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, skiprows=csv_skiprows, header=csv_header)
        return numeric_to_numpy(df, dtype=np.float32)
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    if ext == ".pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().astype(np.float32)
        for key in ["emb", "embedding", "X", "features"]:
            if key in obj:
                return obj[key].detach().cpu().numpy().astype(np.float32)
        raise TypeError(f"Unsupported .pt content in {path}")
    raise ValueError(f"Unsupported embedding extension: {path}")


def load_klifs_onehot(klifs_df: pd.DataFrame) -> np.ndarray:
    if KLIFS_ONEHOT.exists():
        onehot = pd.read_csv(KLIFS_ONEHOT, header=None).values.astype(np.uint8)
        if onehot.shape == (len(klifs_df), N_RESIDUES * BITS_PER_RESIDUE):
            return onehot
        print(f"[warn] onehot shape {onehot.shape} does not match KLIFS rows; rebuilding from Binary_string", flush=True)
    bit_col = find_col(klifs_df, ["Binary_string", "binary_string", "KLIFS_bitstring"])
    return np.stack([bitstring_to_array(x) for x in klifs_df[bit_col]]).astype(np.uint8)


def normalise_mca_category_name(name):
    return str(name).replace("__", "_")



def matched_pairs_rank_biserial_from_differences(diff):

    diff = np.asarray(diff,dtype=np.float64)
    diff = diff[np.isfinite(diff)]

    n_positive = int(np.sum(diff > 0))
    n_negative = int(np.sum(diff < 0))
    n_zero = int(np.sum(diff == 0))

    nonzero_diff = diff[diff != 0]

    if len(nonzero_diff) == 0:
        return (float("nan"),0.0,0.0,n_positive,n_negative,n_zero)

    ranks = stats.rankdata(np.abs(nonzero_diff),method="average")
    w_positive = float(ranks[nonzero_diff > 0].sum())
    w_negative = float(ranks[nonzero_diff < 0].sum())

    denominator = (w_positive + w_negative)

    if denominator == 0:
        rank_biserial = float("nan")
    else:
        rank_biserial = (w_positive - w_negative) / denominator

    return float(rank_biserial),w_positive,w_negative,n_positive,n_negative,n_zero





# =========================
# Main
# =========================

def allocation_diagnostics(terms, protein_phi):
    """Per-pair metrics before any reference/dimension signed cancellation.

    Amplification = sum(abs(delta_indicator * projection)) / abs(sum(terms)).
    Max/count concern mapped dimensions with exactly nonzero SHAP; zero
    denominators are reported separately. An empty active set has a NaN max.
    """
    denominator = terms.sum(axis=0)
    mapped = denominator != 0
    unmapped = protein_phi[:, ~mapped]
    amplification = np.divide(np.abs(terms).sum(axis=0), np.abs(denominator),
                              out=np.zeros_like(denominator), where=mapped)
    active = (protein_phi != 0) & mapped[None, :]
    maxima = np.max(np.where(active, amplification[None, :], -np.inf), axis=1)
    maxima[~active.any(axis=1)] = np.nan
    return {
        "unallocated_signed_shap": unmapped.sum(axis=1),
        "unallocated_abs_shap": np.abs(unmapped).sum(axis=1),
        "zero_denominator_nonzero_shap_count": (unmapped != 0).sum(axis=1),
        "zero_denominator_shap_warning_count": (np.abs(unmapped) > v9.ZERO_DENOMINATOR_SHAP_WARNING).sum(axis=1),
        "max_amplification_nonzero_shap": maxima,
        "large_amplification_nonzero_shap_count": (active & (amplification[None, :] > v9.AMPLIFICATION_WARNING)).sum(axis=1),
    }


def summarize_by_kinase(df_pair):
    """Descriptive target-level summaries; shared molecules can link targets."""
    table = df_pair.groupby("uniprot_id", sort=True).agg(
        n_pairs=("delta_site_entropy", "size"),
        n_unique_molecules=("canonical_smiles", "nunique"),
        mean_top_site_entropy=("observed_mean_entropy", "mean"),
        mean_delta=("delta_site_entropy", "mean"),
        median_delta=("delta_site_entropy", "median"),
        std_delta=("delta_site_entropy", "std"),
        fraction_pairs_above_baseline=("delta_site_entropy", lambda x: float((x > 0).mean())),
    ).reset_index()
    return table


def summarize_entropy_effects(df_pair, df_kinase, baseline):
    """Keep benchmark pair weighting separate from equal target weighting."""
    delta = df_pair["delta_site_entropy"]
    kinase_delta = df_kinase["mean_delta"]
    pair_summary = {
        "weighting": "equal weight per analysed pair in the current benchmark",
        "H0": float(baseline), "n_pairs": int(len(delta)),
        "pair_mean_delta": float(delta.mean()),
        "pair_median_delta": float(delta.median()),
        "n_pairs_above_H0": int((delta > 0).sum()),
        "n_pairs_below_H0": int((delta < 0).sum()),
        "n_pairs_equal_H0": int((delta == 0).sum()),
        "fraction_pairs_above_H0": float((delta > 0).mean()),
    }
    kinase_summary = {
        "weighting": "equal weight per kinase after averaging pairs within each kinase",
        "H0": float(baseline), "n_kinases": int(len(kinase_delta)),
        "equal_kinase_weight_mean_delta": float(kinase_delta.mean()),
        "median_of_kinase_mean_deltas": float(kinase_delta.median()),
        "n_kinases_positive_mean_delta": int((kinase_delta > 0).sum()),
        "n_kinases_negative_mean_delta": int((kinase_delta < 0).sum()),
        "n_kinases_zero_mean_delta": int((kinase_delta == 0).sum()),
        "fraction_kinases_positive_mean_delta": float((kinase_delta > 0).mean()),
    }
    return pair_summary, kinase_summary


def check_tree_additivity(booster, inputs, reference, phi, target_margin, base, precision_cache):
    """Recheck float32 margin failures against the same leaves summed in float64.

    No SHAP values or tolerances are modified. The ensemble base score cancels
    in the target-minus-reference difference. Raw errors remain in diagnostics.
    """
    raw_error = base + phi.sum(axis=1) - target_margin
    bounds = np.asarray([v9.tree_additivity_bound(base, t, p) for t, p in zip(target_margin, phi)])
    error = raw_error.copy()
    failed = np.flatnonzero(np.abs(raw_error) > bounds)
    stable_error = np.full(len(phi), np.nan)
    if len(failed):
        if "leaf_values" not in precision_cache:
            model = json.loads(Path(XGB_MODEL).read_text())["learner"]["gradient_booster"]["model"]
            precision_cache["leaf_values"] = [
                np.asarray(tree["split_conditions"], dtype=np.float32) for tree in model["trees"]]
        leaf_values = precision_cache["leaf_values"]
        values = np.vstack([reference, inputs[failed]])
        leaf_ids = booster.predict(xgb.DMatrix(v9.model_frame(booster, values)), pred_leaf=True).astype(np.int64)
        if leaf_ids.shape != (len(values), len(leaf_values)):
            raise ValueError("Unexpected leaf matrix shape in precision validation")
        leaf_outputs = np.column_stack([weights[leaf_ids[:, j]] for j, weights in enumerate(leaf_values)])
        sums = leaf_outputs.sum(axis=1, dtype=np.float64)
        stable_error[failed] = phi[failed].sum(axis=1) - (sums[1:] - sums[0])
        error[failed] = stable_error[failed]
    if not np.isfinite(error).all():
        raise FloatingPointError("Non-finite TreeSHAP additivity error")
    return error, bounds, raw_error, failed, stable_error


def compute_reference_results(actives, sm_col, up_col, embed, row_lookup,
                              metadata, mca, sae, keep_idx, device, booster):
    """Batch TreeSHAP by reference; propagate signed differences as in V9."""
    names, index, projection, lookup = v9.load_mca_projection()
    lookup.to_csv(OUTDIR / "mca_category_lookup.csv", index=False)
    inputs, positions, target_rows = [], [], []
    embeddings = {}
    for i, row in actives.iterrows():
        uid, smi = str(row[up_col]), str(row[sm_col])
        if uid not in row_lookup:
            continue
        if smi not in embeddings:
            embeddings[smi] = embed(smi)
        if embeddings[smi] is None:
            continue
        r = row_lookup[uid]
        positions.append(i); target_rows.append(r)
        inputs.append(np.r_[embeddings[smi], mca[r]])
    if not inputs:
        raise ValueError("No valid active pairs")
    inputs = np.asarray(inputs, dtype=np.float32)
    target_rows = np.asarray(target_rows)
    unique_rows = np.unique(target_rows)
    indicators = {int(r): v9.kinase_indicator(
        v9.parse_binary(metadata.loc[r, "Binary_string"], v9.KLIFS_BITS, str(r)), index, str(r))
        for r in unique_rows}
    molecules, bits, _ = v9.load_molecule_background(REFERENCE_COUNT, RANDOM_SEED, OUTDIR)
    kinases, kinase_mca, kinase_bits, _ = v9.load_kinase_background("all", RANDOM_SEED, OUTDIR)
    kinase_indicators = np.asarray([v9.kinase_indicator(b, index, str(i))
                                  for i, b in enumerate(kinase_bits)])
    validation = []
    for label, indicator, stored in (
        [(f"target_{r}", indicators[int(r)], mca[r]) for r in unique_rows]
        + [(f"background_{r}", kinase_indicators[r], kinase_mca[r]) for r in range(len(kinases))]
    ):
        reconstructed = indicator @ projection
        np.testing.assert_allclose(reconstructed, stored,
                                   rtol=v9.FORWARD_DIFFERENCE_RTOL, atol=v9.FORWARD_DIFFERENCE_ATOL)
        validation.append({"kinase": label, "max_abs_reconstruction_error":
                           float(np.max(np.abs(reconstructed-stored))), "status": "PASS"})
    pd.DataFrame(validation).to_csv(OUTDIR / "mca_linear_projection_validation.csv", index=False)
    mi, ki = v9.reference_design("joint_independent", len(molecules), len(kinases), RANDOM_SEED)
    pd.DataFrame({"reference_index": np.arange(len(mi)), "molecule_source_row":
                  molecules.source_row.to_numpy()[mi], "kinase_source_row":
                  kinases.source_row.to_numpy()[ki]}).to_csv(OUTDIR / "reference_combinations.csv", index=False)
    with torch.no_grad():
        latent = sae.encoder(torch.as_tensor(bits, dtype=torch.float32, device=device))[:, keep_idx].cpu().numpy()
    references = np.c_[latent[mi], kinase_mca[ki]].astype(np.float32)
    config = {"method": BITSCORE_METHOD, "references": len(references), "seed": RANDOM_SEED,
              "valid_pairs": len(inputs), "molecule_dims": MOL_KEEP_DIMS,
              "model": str(Path(XGB_MODEL).resolve())}
    digest = hashlib.sha256(inputs.tobytes() + references.tobytes() + projection.tobytes()
                            + target_rows.tobytes() + str(config).encode()
                            + Path(XGB_MODEL).read_bytes()).hexdigest()
    config["config_hash"] = digest
    config_path = OUTDIR / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text())["config_hash"] != digest:
        raise ValueError("Output configuration changed; choose a new --output-dir")
    config_path.write_text(json.dumps(config, indent=2))
    cache = OUTDIR / "reference_checkpoints"; cache.mkdir(exist_ok=True)
    sum_phi = np.zeros(inputs.shape, dtype=np.float64)
    sum_residue = np.zeros((len(inputs), N_RESIDUES), dtype=np.float64)
    residue_index = lookup.residue.to_numpy() - 1
    target_margin = v9.raw_predict(booster, inputs).astype(np.float64)
    diagnostics = []
    precision_cache = {}
    precision_path = OUTDIR / "tree_additivity_precision_diagnostics.csv"
    pd.DataFrame(columns=["reference_index", "active_row_index", "float32_raw_error",
                          "float64_leaf_error", "error_bound", "status"]).to_csv(precision_path, index=False)
    for r, reference in enumerate(references):
        started = time.perf_counter()
        path = cache / f"reference_{r:05d}.npz"
        cached = path.exists()
        if cached:
            with np.load(path, allow_pickle=False) as data:
                if str(data["config_hash"].item()) != digest:
                    raise ValueError(f"Checkpoint configuration mismatch: {path}")
                phi = data["phi"]
        else:
            explainer = shap.TreeExplainer(booster, data=v9.model_frame(booster, reference),
                                          feature_perturbation="interventional", model_output="raw")
            phi = np.asarray(explainer.shap_values(v9.model_frame(booster, inputs)), dtype=np.float64)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, phi=phi, config_hash=digest)
            temporary.replace(path)
        base = float(v9.raw_predict(booster, reference)[0])
        error, bounds, raw_error, precision_rows, stable_error = check_tree_additivity(
            booster, inputs, reference, phi, target_margin, base, precision_cache)
        if len(precision_rows):
            pd.DataFrame({
                "reference_index": r,
                "active_row_index": np.asarray(positions)[precision_rows],
                "float32_raw_error": raw_error[precision_rows],
                "float64_leaf_error": stable_error[precision_rows],
                "error_bound": bounds[precision_rows],
                "status": np.where(np.abs(error[precision_rows]) <= bounds[precision_rows], "PASS", "FAIL"),
            }).to_csv(precision_path, mode="a", header=False, index=False)
        if np.any(np.abs(error) > bounds):
            failed = np.flatnonzero(np.abs(error) > bounds)
            np.savez_compressed(OUTDIR / "tree_additivity_failure.npz", reference=reference,
                                inputs=inputs[failed], phi=phi[failed], target_margin=target_margin[failed],
                                base=base, error=error[failed], bounds=bounds[failed],
                                active_rows=np.asarray(positions)[failed], reference_index=r)
            raise AssertionError(
                f"TreeSHAP additivity failed at reference {r}: {len(failed)} pairs; "
                f"max_error={np.max(np.abs(error)):.9g}; "
                f"first_failed_pair={positions[failed[0]]}, error={error[failed[0]]:.9g}, "
                f"bound={bounds[failed[0]]:.9g}; saved tree_additivity_failure.npz")
        reference_residues = np.zeros_like(sum_residue)
        max_conservation = 0.0
        pair_diagnostics = []
        for target in unique_rows:
            # Collapse category coefficients to residues before multiplying by SHAP.
            # This is algebraically the same per-reference allocation as V9.
            terms = (indicators[int(target)] - kinase_indicators[ki[r]])[:, None] * projection
            denominators = terms.sum(axis=0)
            np.testing.assert_allclose(denominators, mca[target].astype(np.float64) - kinase_mca[ki[r]],
                                       rtol=v9.FORWARD_DIFFERENCE_RTOL, atol=v9.FORWARD_DIFFERENCE_ATOL)
            mapped = denominators != 0
            fractions = np.divide(terms, denominators, out=np.zeros_like(terms), where=mapped)
            residue_fractions = np.zeros((N_RESIDUES, len(denominators)))
            np.add.at(residue_fractions, residue_index, fractions)
            mask = target_rows == target
            protein_phi = phi[mask, MOL_KEEP_DIMS:]
            metrics = allocation_diagnostics(terms, protein_phi)
            batch_rows = np.flatnonzero(mask)
            frame = pd.DataFrame(metrics)
            frame.insert(0, "reference_index", r)
            frame.insert(1, "active_row_index", np.asarray(positions)[batch_rows])
            frame.insert(2, "uniprot_id", actives.iloc[np.asarray(positions)[batch_rows]][up_col].astype(str).to_numpy())
            frame["amplification_warning_threshold"] = v9.AMPLIFICATION_WARNING
            frame["zero_denominator_shap_warning_threshold"] = v9.ZERO_DENOMINATOR_SHAP_WARNING
            pair_diagnostics.append(frame)
            scores = protein_phi @ residue_fractions.T
            residual = protein_phi[:, ~mapped].sum(axis=1)
            conservation = scores.sum(axis=1) + residual - protein_phi.sum(axis=1)
            magnitude = np.abs(protein_phi) @ np.abs(fractions).sum(axis=0) + np.abs(protein_phi).sum(axis=1)
            tolerance = np.asarray([v9.error_bound(float(x), terms.size) for x in magnitude])
            if np.any(np.abs(conservation) > tolerance):
                raise AssertionError(f"Residue conservation failed at reference {r}, target {target}")
            max_conservation = max(max_conservation, float(np.max(np.abs(conservation))))
            reference_residues[mask] = scores
        reference_diagnostics = pd.concat(pair_diagnostics, ignore_index=True).sort_values("active_row_index")
        reference_diagnostics.to_csv(OUTDIR / "per_reference_pair_diagnostics.csv",
                                     mode="w" if r == 0 else "a", header=r == 0, index=False)
        reference_summary = {
            "unallocated_abs_shap_sum": float(reference_diagnostics.unallocated_abs_shap.sum()),
            "unallocated_signed_shap_sum": float(reference_diagnostics.unallocated_signed_shap.sum()),
            "zero_denominator_nonzero_shap_count": int(reference_diagnostics.zero_denominator_nonzero_shap_count.sum()),
            "zero_denominator_shap_warning_count": int(reference_diagnostics.zero_denominator_shap_warning_count.sum()),
            "max_amplification_nonzero_shap": float(reference_diagnostics.max_amplification_nonzero_shap.max()),
            "large_amplification_nonzero_shap_count": int(reference_diagnostics.large_amplification_nonzero_shap_count.sum()),
            "amplification_warning_threshold": v9.AMPLIFICATION_WARNING,
            "zero_denominator_shap_warning_threshold": v9.ZERO_DENOMINATOR_SHAP_WARNING,
        }
        sum_phi += phi
        sum_residue += reference_residues
        diagnostics.append({**reference_summary, "reference_index": r, "cached": cached,
                            "seconds": time.perf_counter()-started,
                            "max_tree_additivity_error": float(np.max(np.abs(error))),
                            "max_tree_additivity_float32_raw_error": float(np.max(np.abs(raw_error))),
                            "float64_leaf_recheck_pair_count": len(precision_rows),
                            "max_float64_leaf_recheck_error": (float(np.max(np.abs(stable_error[precision_rows])))
                                                               if len(precision_rows) else np.nan),
                            "max_residue_conservation_error": max_conservation})
        pd.DataFrame(diagnostics).to_csv(OUTDIR / "reference_diagnostics.csv", index=False)
        print(f"[reference] {r+1}/{len(references)}: {diagnostics[-1]['seconds']:.2f}s cached={cached}", flush=True)
    mean_phi, mean_residue = sum_phi / len(references), sum_residue / len(references)
    np.savez_compressed(OUTDIR / "average_attributions.npz", active_row_indices=positions,
                        mean_shap=mean_phi, mean_residue=mean_residue)
    return {i: (mean_phi[j], mean_residue[j]) for j, i in enumerate(positions)}

def main():
    print("=" * 70, flush=True)
    print("VSDS active residue entropy analysis - MCA protein embedding", flush=True)
    print("=" * 70, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {device}", flush=True)
    print(f"[init] output = {OUTDIR}", flush=True)

    # --- site-level KLIFS entropy ---

    site_entropy = load_site_entropy(SITE_ENTROPY_CSV)
    region_entropy = summarize_region_entropy(site_entropy)
    # Global null expectation:
    # mean site-level Shannon entropy across all 85 KLIFS positions.
    global_entropy_mean = float(site_entropy.mean())

    print(f"[init] site-level entropy loaded: {len(site_entropy)} KLIFS positions",flush=True)
    print(f"[init] global mean site entropy = {global_entropy_mean:.6f}",flush=True)



    print(f"[init] site-level entropy loaded: "
    f"{len(site_entropy)} KLIFS positions",flush=True)

    print(f"[init] site entropy mean = "
        f"{site_entropy.mean():.6f}, "
        f"min = {site_entropy.min():.6f}, "
        f"max = {site_entropy.max():.6f}",flush=True)


    # --- molecule SAE ---
    mol_sae = SparseLinearAE(NBITS, 1024).to(device)
    mol_sae.load_state_dict(torch.load(MOL_SAE_PATH, map_location=device, weights_only=True))
    mol_sae.eval()
    mol_keep_idx = torch.load(MOL_SAE_IDX, map_location=device, weights_only=True).long()[:MOL_KEEP_DIMS]
    print(f"[init] molecule SAE latents kept = {len(mol_keep_idx)}", flush=True)

    # --- KLIFS lookup and MCA protein embedding ---
    klifs_df = pd.read_csv(KLIFS_FILE)
    uniprot_col = find_col(klifs_df, ["uniprot_id", "Uniprot_ID", "UniProt_ID"])
    print(f"[init] KLIFS uniprot column = '{uniprot_col}'", flush=True)

    uniprot_to_row = {}
    duplicate_uniprots = 0
    for row_idx, uid in enumerate(klifs_df[uniprot_col].astype(str).tolist()):
        if uid in uniprot_to_row:
            duplicate_uniprots += 1
        else:
            uniprot_to_row[uid] = row_idx
    if duplicate_uniprots:
        print(f"[warn] {duplicate_uniprots} duplicate UniProt IDs in KLIFS file; first row used", flush=True)

    klifs_mca_mat = load_embedding_matrix(KLIFS_MCA_CSV, csv_skiprows=1, csv_header=None)
    klifs_onehot_mat = load_klifs_onehot(klifs_df)
    prot_dim = klifs_mca_mat.shape[1]
    print(f"[init] KLIFS MCA shape = {klifs_mca_mat.shape}, onehot shape = {klifs_onehot_mat.shape}", flush=True)

    if klifs_mca_mat.shape[0] != len(klifs_df):
        print(f"[warn] row mismatch: MCA rows={klifs_mca_mat.shape[0]} vs KLIFS rows={len(klifs_df)}", flush=True)
    if klifs_onehot_mat.shape[0] != len(klifs_df):
        print(f"[warn] row mismatch: onehot rows={klifs_onehot_mat.shape[0]} vs KLIFS rows={len(klifs_df)}", flush=True)

    # --- XGBoost + SHAP ---
    booster = xgb.Booster()
    booster.load_model(str(XGB_MODEL))
    expected_features = MOL_KEEP_DIMS + prot_dim
    if booster.num_features() != expected_features:
        raise ValueError(f"Model expects {booster.num_features()} features; embeddings provide {expected_features}")
    feat_cols = booster.feature_names or list(range(expected_features))
    # --- Morgan generator ---
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=NBITS)

    # --- VSDS ---
    vsds = pd.read_csv(VSDS_CSV)
    sm_col = find_col(vsds, ["smiles", "SMILES"])
    up_col = find_col(vsds, ["uniprot_id", "Uniprot_ID", "UniProt_ID"])
    lb_col = find_col(vsds, ["molecule_label", "label", "active"])
    labels = pd.to_numeric(vsds[lb_col], errors="coerce")
    actives = vsds[labels == 1].reset_index(drop=True)
    if MAX_ACTIVE_PAIRS is not None:
        actives = actives.head(int(MAX_ACTIVE_PAIRS)).reset_index(drop=True)
    print(f"[init] VSDS rows={len(vsds)} active rows={len(actives)}", flush=True)
    print(f"[init] columns smiles='{sm_col}' uniprot='{up_col}' label='{lb_col}'", flush=True)

    @torch.no_grad()
    def smiles_to_mol_emb(smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros((NBITS,), dtype=np.float32)
        arr[list(fp.GetOnBits())] = 1.0
        t = torch.from_numpy(arr).unsqueeze(0).to(device)
        z = mol_sae.encoder(t)[:, mol_keep_idx].cpu().numpy().squeeze()
        return z.astype(np.float32)

    reference_results = compute_reference_results(
        actives, sm_col, up_col, smiles_to_mol_emb, uniprot_to_row,
        klifs_df, klifs_mca_mat, mol_sae, mol_keep_idx, device, booster)

    # =========================
    # Per-pair processing
    # =========================
    mol_cache = {}
    prot_cache = {}

    region_hit_count = defaultdict(int)
    per_pair_records = []
    obs_arr = []
    rand_arr = []

    rng = np.random.default_rng(RANDOM_SEED)

    skipped_smiles = 0
    skipped_uniprot = 0
    skipped_zerobitscore = 0

    print("\n[run ] processing active pairs ...", flush=True)
    for i in tqdm(range(len(actives)), file=sys.stdout):
        row = actives.iloc[i]
        smi = str(row[sm_col])
        uid = str(row[up_col])

        if smi not in mol_cache:
            mol_cache[smi] = smiles_to_mol_emb(smi)
        mol_emb = mol_cache[smi]
        if mol_emb is None:
            skipped_smiles += 1
            continue

        if uid not in prot_cache:
            if uid not in uniprot_to_row:
                prot_cache[uid] = (None, None)
            else:
                r = uniprot_to_row[uid]
                prot_cache[uid] = (
                    klifs_mca_mat[r].astype(np.float32),
                    klifs_onehot_mat[r].astype(np.uint8),
                )
        prot_emb, prot_onehot = prot_cache[uid]
        if prot_emb is None:
            skipped_uniprot += 1
            continue

        sv, res_score = reference_results[i]
        prot_shap = sv[MOL_KEEP_DIMS:]
        top_components = np.arange(prot_dim)

        nonzero_idx = np.where(np.abs(res_score) > 0)[0]
        if len(nonzero_idx) == 0:
            skipped_zerobitscore += 1
            continue

        order = nonzero_idx[np.argsort(-np.abs(res_score[nonzero_idx]), kind="stable")]
        top_pos = order[:TOPN_RESIDUES]
        K = len(top_pos)

        for pos in top_pos:
            region_hit_count[KLIFS_DOMAIN_MAP[pos]] += 1

        # Mean site-level Shannon entropy of the top model-important
        # KLIFS positions for this active compound-kinase pair.
        top_site_entropy = site_entropy[top_pos]
        obs_mean_ent = float(top_site_entropy.mean())

        # Exact null expectation.
        # The expected mean entropy of a uniformly sampled set of
        # KLIFS positions equals the mean entropy across all 85 positions.
        baseline_mean_ent = global_entropy_mean
        obs_arr.append(obs_mean_ent)


        per_pair_records.append({
            "active_row_index": i,
            "smiles": smi,
            "canonical_smiles": Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True),
            "uniprot_id": uid,
            "n_top_residues": K,
            "top_positions_0idx": ",".join(map(str, top_pos.tolist())),
            "top_positions_1idx": ",".join(map(str, (top_pos + 1).tolist())),
            "top_regions": ",".join(KLIFS_DOMAIN_MAP[p] for p in top_pos),
            "top_residue_scores": ",".join(f"{res_score[p]:.6g}" for p in top_pos),
            "top_site_entropies": ",".join(f"{site_entropy[p]:.6f}"for p in top_pos),
            "all_mca_components_0idx": ",".join(map(str, top_components.tolist())),
            "all_mca_components_1idx": ",".join(map(str, (top_components + 1).tolist())),
            "all_mca_component_signed_shap": ",".join(f"{prot_shap[c]:.9g}" for c in top_components),
            "protein_shap_sum": float(prot_shap.sum()),
            "mapped_residue_score_sum": float(res_score.sum()),
            "unmapped_signed_shap": float(prot_shap.sum() - res_score.sum()),
            "all_residue_signed_scores": ",".join(f"{v:.12g}" for v in res_score),
            "observed_mean_entropy": obs_mean_ent,
            "global_mean_site_entropy": global_entropy_mean,
            "delta_site_entropy": obs_mean_ent - global_entropy_mean
        })

    print(f"\n[run ] processed pairs   = {len(per_pair_records)}", flush=True)
    print(f"[run ] skipped (smiles)  = {skipped_smiles}", flush=True)
    print(f"[run ] skipped (uniprot) = {skipped_uniprot}", flush=True)
    print(f"[run ] skipped (zero bs) = {skipped_zerobitscore}", flush=True)

    if len(per_pair_records) == 0:
        raise RuntimeError("No successful active pairs were analysed.")

    # =========================
    # Aggregate tables
    # =========================
    df_pair = pd.DataFrame(per_pair_records)
    df_pair.to_csv(OUTDIR / "per_pair_top10_residues.csv", index=False)

    df_kinase = summarize_by_kinase(df_pair)
    df_kinase.to_csv(OUTDIR / "per_kinase_entropy_summary.csv", index=False)
    molecule_targets = df_pair.groupby("canonical_smiles").agg(
        n_pairs=("uniprot_id", "size"), n_kinases=("uniprot_id", "nunique"))
    molecule_targets.to_csv(OUTDIR / "molecule_target_overlap.csv")

    region_count = region_hit_count
    region_n_pos = defaultdict(int)
    for pos in range(N_RESIDUES):
        region_n_pos[KLIFS_DOMAIN_MAP[pos]] += 1

    df_region = pd.DataFrame({
        "region": REGIONS_ORDERED,
        "region_entropy_mean": [
            region_entropy.loc[r, "mean"] for r in REGIONS_ORDERED
        ],
        "region_entropy_std": [
            region_entropy.loc[r, "std"] for r in REGIONS_ORDERED
        ],
        "count": [
            region_count[r] for r in REGIONS_ORDERED
        ],
        "n_positions": [
            region_n_pos[r] for r in REGIONS_ORDERED
        ],
    })

    df_region.to_csv(
        OUTDIR / "region_frequency.csv",
        index=False
    )
    df_region.to_csv(OUTDIR / "region_frequency.csv", index=False)

    # =========================
    # Statistics
    # =========================
    obs = np.array(obs_arr, dtype=np.float64)
    # Difference between the observed top-10 mean entropy
    # and the global mean entropy across all 85 KLIFS positions.
    delta = obs - global_entropy_mean

    # Descriptive effects only: pairs sharing kinases/molecules are dependent.
    # Matched-pairs rank-biserial correlation effect size
    rank_biserial,w_positive,w_negative,n_positive,n_negative,n_zero = matched_pairs_rank_biserial_from_differences(delta)

    pair_summary, kinase_summary = summarize_entropy_effects(df_pair, df_kinase, global_entropy_mean)
    pd.DataFrame([pair_summary]).to_csv(OUTDIR / "pair_level_effect_summary.csv", index=False)
    pd.DataFrame([kinase_summary]).to_csv(OUTDIR / "kinase_equal_weight_effect_summary.csv", index=False)
    pd.DataFrame([{
        "n_pairs": len(delta),
        "n_kinases": len(df_kinase),
        "observed_entropy_mean": obs.mean(),
        "observed_entropy_std": obs.std(ddof=1),
        "global_mean_site_entropy": global_entropy_mean,
        "delta_mean": delta.mean(),
        "delta_std": delta.std(ddof=1),
        "delta_median": np.median(delta),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "n_unique_canonical_molecules": len(molecule_targets),
        "n_molecules_shared_across_targets": (molecule_targets.n_kinases > 1).sum(),
        "W_positive": w_positive,
        "W_negative": w_negative,
        "rank_biserial": rank_biserial,
    }]).to_csv(OUTDIR / "descriptive_statistics.csv", index=False)

    # =========================
    # Plots
    # =========================
    # =========================
    # Figure: KLIFS region entropy ± position-level SD
    # and VSDS important-residue counts
    # =========================

    fig, ax1 = plt.subplots(figsize=(15, 7))

    x = np.arange(len(REGIONS_ORDERED))

    entropy_mean = np.array(
        [region_entropy.loc[r, "mean"] for r in REGIONS_ORDERED],
        dtype=np.float64,
    )

    entropy_std = np.array(
        [region_entropy.loc[r, "std"] for r in REGIONS_ORDERED],
        dtype=np.float64,
    )

    n_positions = np.array(
        [region_n_pos[r] for r in REGIONS_ORDERED],
        dtype=int,
    )

    # ---------------------------------------------------------
    # Left y-axis: regional mean Shannon entropy with SD
    #
    # The error bars represent variation among KLIFS positions
    # within each functional region.
    #
    # GK and VIII each contain only one KLIFS position, so
    # their within-region SD is undefined (np.nan). Matplotlib
    # therefore does not draw an error bar for these positions.
    # ---------------------------------------------------------

    ax1.errorbar(
        x,
        entropy_mean,
        yerr=entropy_std,
        marker="o",
        markersize=6,
        linewidth=2.5,
        capsize=4,
        capthick=1.5,
        elinewidth=1.5,
        label="Regional mean residue entropy ± position-level SD",
    )

    ax1.set_xlabel(
        "KLIFS functional region",
        fontsize=14
    )

    ax1.set_ylabel(
        "Residue Shannon entropy",
        fontsize=14
    )

    ax1.set_xticks(x)

    # Add number of KLIFS positions contributing to each
    # functional-region mean directly to the x-axis labels.
    region_labels = [
        f"{region}\n(n={n})"
        for region, n in zip(REGIONS_ORDERED, n_positions)
    ]

    ax1.set_xticklabels(
        region_labels,
        rotation=45,
        ha="right"
    )

    # ---------------------------------------------------------
    # Automatically determine the y-axis range from mean ± SD.
    # Ignore NaN SD values from single-position regions.
    # ---------------------------------------------------------

    valid_std = np.where(
        np.isnan(entropy_std),
        0.0,
        entropy_std
    )

    entropy_lower = entropy_mean - valid_std
    entropy_upper = entropy_mean + valid_std

    ymin = max(
        0.0,
        float(np.min(entropy_lower)) - 0.10
    )

    ymax = float(np.max(entropy_upper)) + 0.10

    ax1.set_ylim(ymin, ymax)

    # ---------------------------------------------------------
    # Right y-axis: important-residue hit counts
    # ---------------------------------------------------------

    ax2 = ax1.twinx()

    ax2.bar(
        x,
        df_region["count"].values,
        alpha=0.55,
        label=(f"Top-{TOPN_RESIDUES} important-residue counts"
        ),
    )

    ax2.set_ylabel(
        "Important-residue hit count",
        fontsize=14
    )

    # ---------------------------------------------------------
    # Combined legend
    # ---------------------------------------------------------

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper right",
        frameon=True
    )

    ax1.set_title(
        "KLIFS region entropy and VSDS active MCA "
        "important-residue distribution",
        fontsize=15
    )

    plt.tight_layout()

    plt.savefig(
        OUTDIR / "region_entropy_vs_count.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    baseline_val = global_entropy_mean
    bins = np.linspace(min(obs.min(), baseline_val) - 0.02, obs.max() + 0.02, 50)
    ax.hist(obs, bins=bins, alpha=0.7, color="steelblue", edgecolor="white", linewidth=0.4,
            label="Observed (top |signed residue attribution|)")
    ax.axvline(baseline_val, color="dimgray", linestyle="--", linewidth=2,
               label=f"All-position mean {baseline_val:.3f}\n(all {N_RESIDUES} positions)")
    ax.axvline(obs.mean(), color="steelblue", linestyle="--", linewidth=2, label=f"Observed mean {obs.mean():.3f}")
    ax.axvspan(baseline_val, bins[-1], alpha=0.08, color="steelblue")
    ax.set_xlabel("Mean site-level Shannon entropy of top 10 KLIFS positions")
    ax.set_ylabel("Pair count")
    ax.set_title("Site-level entropy enrichment of model-important KLIFS positions\n"
        f"Descriptive distribution: {len(obs)} pairs, {len(df_kinase)} kinases; shared targets/molecules")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTDIR / "site_level_entropy_distribution.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTDIR / "site_level_entropy_distribution.pdf", bbox_inches="tight")
    plt.close()

    ordered_kinases = df_kinase.sort_values("mean_delta")
    fig, ax = plt.subplots(figsize=(max(9, .3 * len(ordered_kinases)), 5.5))
    ax.bar(np.arange(len(ordered_kinases)), ordered_kinases.mean_delta,
           color=np.where(ordered_kinases.mean_delta >= 0, "steelblue", "salmon"))
    ax.axhline(0, color="dimgray", lw=1)
    ax.set_xticks(np.arange(len(ordered_kinases)))
    ax.set_xticklabels([f"{r.uniprot_id} (n={r.n_pairs})" for r in ordered_kinases.itertuples()], rotation=90, fontsize=7)
    ax.set_ylabel("Within-kinase mean of pair entropy minus 85-position baseline")
    ax.set_title("Descriptive entropy differences by kinase\nTargets may share molecules; no independence-based inference")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUTDIR / f"per_kinase_entropy_distribution.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] outputs written to {OUTDIR}/", flush=True)
    for fn in [
        "per_pair_top10_residues.csv",
        "region_frequency.csv",
        "descriptive_statistics.csv",
        "pair_level_effect_summary.csv",
        "kinase_equal_weight_effect_summary.csv",
        "per_kinase_entropy_summary.csv",
        "per_reference_pair_diagnostics.csv",
        "per_kinase_entropy_distribution.png",
        "region_entropy_vs_count.png",
        "site_level_entropy_distribution.png",
    ]:
        print(f"       {fn}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSDS entropy analysis using V9 joint-independent reference attribution")
    parser.add_argument("--references", type=int, default=2000)
    parser.add_argument("--max-active-pairs", type=int)
    parser.add_argument("--output-dir", type=Path, default=OUTDIR)
    args = parser.parse_args()
    if args.references < 1:
        parser.error("--references must be positive")
    OUTDIR = args.output_dir.resolve()
    REFERENCE_COUNT = args.references
    MAX_ACTIVE_PAIRS = args.max_active_pairs
    OUTDIR.mkdir(parents=True, exist_ok=True)
    os.chdir(Path(__file__).resolve().parent)
    torch.set_num_threads(1)
    main()
