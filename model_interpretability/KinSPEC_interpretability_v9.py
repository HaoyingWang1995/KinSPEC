#!/usr/bin/env python3
"""V9 reference-matched linear-rescaled attribution for KinSPEC molecules and kinases.

For each one-row reference, this script computes interventional TreeSHAP on the
complete 417 molecular-latent + 200 protein-MCA input. Both input groups are
exact linear maps of binary encodings, so each group's SHAP is rescaled through
its own signed input difference for that same reference:

    molecule: 4096 Morgan bits  -> linear SAE encoder -> 417 latents
    kinase:   2113 KLIFS categories (85 residues x 5 systems) -> MCA -> 200 dims

Rescaled matrices are averaged only after the per-reference step.

Background modes decide which half of each reference row comes from the
background and which half equals the target itself:

    joint_independent  [background molecule, background kinase]  both attributed
    fixed_target       [background molecule, target kinase]      molecule only
    fixed_molecule     [target molecule, background kinase]      kinase only

The linear decompositions are exact. Propagating nonlinear-model SHAP through
them is a post-hoc approximation, not original-bit Shapley, DeepSHAP, or the
causal effect of independently changing a Morgan bit or a residue property.
"""

from __future__ import annotations

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


import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
from PIL import Image
import shap
import torch
import torch.nn as nn
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "encoder_topk_v9"
BACKGROUND_MODES = ("joint_independent", "fixed_target", "fixed_molecule")
BACKGROUND_MODE = "joint_independent"
MOLECULE_BACKGROUND_SIZE: int | str = 2000
KINASE_BACKGROUND_SIZE: int | str = "all"
RANDOM_SEED = 42
TOP_K = (5, 10, 20, 30, 40, 50)
MOLECULE_DIMS = 417
PROTEIN_DIMS = 200
INPUT_DIMS = MOLECULE_DIMS + PROTEIN_DIMS
MORGAN_BITS = 4096
BATCH_SIZE = 10
SOFT_COOLWARM = colors.LinearSegmentedColormap.from_list(
    "soft_coolwarm",
    matplotlib.colormaps["coolwarm"](np.linspace(0.15, 0.85, 256)),
)
AMPLIFICATION_WARNING = 1e3
COLOR_COMPRESSION_RATIO_WARNING = 100.0
ZERO_DENOMINATOR_SHAP_WARNING = 1e-8
FIXED_GROUP_ZERO_TOLERANCE = 1e-8
FORWARD_DIFFERENCE_RTOL = 2e-5
FORWARD_DIFFERENCE_ATOL = 2e-6

# KLIFS pocket encoding used to build the MCA (feature_embedding/Kinase_test_MCA/MCA_klifs.ipynb).
KLIFS_RESIDUES = 85
KLIFS_SYSTEMS = (
    ("sidechain", ("Aliphatic", "Aromatic", "Sulfur", "Imino", "Amide", "Hydroxyl",
                   "Acidic_sc", "Basic_sc")),
    ("tertiary", ("Exposed", "Buried", "Intermediate")),
    ("functional", ("Acidic_fn", "Basic_fn", "Hydrophobic", "Polar")),
    ("secondary", ("alpha", "beta", "coil")),
    ("size", ("Tiny", "Small", "Medium", "Large", "VeryLarge")),
)
KLIFS_BITS_PER_RESIDUE = sum(len(categories) for _, categories in KLIFS_SYSTEMS)
KLIFS_BITS = KLIFS_RESIDUES * KLIFS_BITS_PER_RESIDUE
SYSTEM_NAMES = tuple(name for name, _ in KLIFS_SYSTEMS)
# Same KLIFS region definition as region_entropy_dotplot.py; short labels for plots.
KLIFS_REGIONS = (
    ("beta-strand-I", "I", 1, 3), ("glycine-rich loop", "g.l", 4, 9),
    ("beta-strand-II", "II", 10, 13), ("beta-strand-III", "III", 14, 19),
    ("alpha-C helix", "αC", 20, 30), ("back loop", "b.l", 31, 37),
    ("beta-strand-IV", "IV", 38, 41), ("beta-strand-V", "V", 42, 44),
    ("gatekeeper", "GK", 45, 45), ("hinge region", "hinge", 46, 48),
    ("linker", "linker", 49, 52), ("alpha-D helix", "αD", 53, 59),
    ("alpha-E helix", "αE", 60, 64), ("beta-strand-VI", "VI", 65, 67),
    ("catalytic loop", "c.l", 68, 75), ("beta-strand-VII", "VII", 76, 78),
    ("beta-strand-VIII", "VIII", 79, 79), ("xDFG", "xDFG", 80, 83),
    ("activation loop", "a.l", 84, 85),
)

PAIRS = (
    {"pair": "Imatinib_ABL1", "drug_name": "Imatinib",
     "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
     "kinase": "P00519"},
    {"pair": "Dacomitinib_EGFR", "drug_name": "Dacomitinib",
     "smiles": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCCC1",
     "kinase": "P00533"},
)

SAE_MODEL = ROOT / _kp('feature_embedding/mol/LSAE_chembl_1900k_full_dataset_v4_model.pt')
SAE_INDICES = ROOT / _kp('feature_embedding/mol/LSAE_chembl_1900k_full_dataset_v4_topk_of1024d_top99.pt')
XGB_MODEL = ROOT / _kp('model/KinSPEC/KinSPEC_full_training_pool_MCA_200D_v4.json')
KINASE_TABLE = ROOT / _kp('feature_embedding/Kinase_test_MCA/human_482_typical_kinase_1955bit.csv')
KINASE_MCA = ROOT / _kp('feature_embedding/Kinase_test_MCA/MCA_klifs_482_200D_features.csv')
MCA_CATEGORY_COORDINATES = ROOT / _kp('feature_embedding/Kinase_test_MCA/MCA_200D_category_coordinates.csv')
MCA_INERTIA_SPECTRUM = ROOT / _kp('feature_embedding/Kinase_test_MCA/MCA_inertia_spectrum.csv')
MOLECULE_BACKGROUND = HERE / "background_data/training_unique_morgan_fingerprints.csv"
KINASE_BACKGROUND = HERE / "background_data/training_unique_kinases_1955bit_MCA200.csv"
MCA_COLUMNS = [f"mca_{i}" for i in range(1, PROTEIN_DIMS + 1)]


class SparseLinearAE(nn.Module):
    def __init__(self, input_dim: int = MORGAN_BITS, latent_dim: int = 1024):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)


def hp_sum(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=np.longdouble), dtype=np.longdouble))


def hp_axis_sum(x: np.ndarray, axis: int) -> np.ndarray:
    return np.asarray(np.sum(np.asarray(x, dtype=np.longdouble), axis=axis,
                             dtype=np.longdouble), dtype=np.float64)


def error_bound(magnitude: float, operations: int) -> float:
    return float(32 * np.finfo(np.float64).eps * max(operations, 1) * max(magnitude, 1.0))


def tree_additivity_bound(reference_margin: float, target_margin: float, shap_values: np.ndarray) -> float:
    # XGBoost prediction inputs and tree thresholds are float32. Scale the bound
    # by the total additive magnitude and the number of accumulated features.
    magnitude = abs(reference_margin) + abs(target_margin) + hp_sum(np.abs(shap_values))
    return float(16 * np.finfo(np.float32).eps * max(magnitude, 1.0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def model_frame(booster: xgb.Booster, values: np.ndarray) -> pd.DataFrame:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    names = booster.feature_names or [str(i) for i in range(values.shape[1])]
    if len(names) != values.shape[1]:
        raise ValueError(f"feature-name count {len(names)} != matrix width {values.shape[1]}")
    return pd.DataFrame(values, columns=names)


def raw_predict(booster: xgb.Booster, values: np.ndarray) -> np.ndarray:
    return booster.predict(xgb.DMatrix(model_frame(booster, values)), output_margin=True)


def probability_predict(booster: xgb.Booster, values: np.ndarray) -> np.ndarray:
    return booster.predict(xgb.DMatrix(model_frame(booster, values)))


def parse_background_size(text: str) -> int | str:
    if text.lower() == "all":
        return "all"
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("background size must be a positive integer or 'all'")
    return value


def parse_binary(value: object, length: int, label: str) -> np.ndarray:
    text = str(value).strip()
    if len(text) != length or set(text) - {"0", "1"}:
        raise ValueError(
            f"{label}: expected a {length}-character binary string; "
            f"length={len(text)}, invalid={sorted(set(text)-{'0','1'})}"
        )
    return np.fromiter((character == "1" for character in text), dtype=np.float32, count=length)


def sample_rows(count: int, size: int | str, rng: np.random.Generator) -> np.ndarray:
    requested = count if size == "all" else int(size)
    if requested >= count:
        return np.arange(count)
    return np.sort(rng.choice(count, requested, replace=False))


def target_molecule(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid target SMILES: {smiles}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_BITS)
    additional = rdFingerprintGenerator.AdditionalOutput()
    additional.AllocateBitInfoMap()
    fp = generator.GetFingerprint(molecule, additionalOutput=additional)
    bits = np.zeros(MORGAN_BITS, dtype=np.float32)
    bits[list(fp.GetOnBits())] = 1.0
    return molecule, bits, additional.GetBitInfoMap()


def load_molecule_background(size: int | str, seed: int, output: Path):
    molecules = pd.read_csv(MOLECULE_BACKGROUND, dtype={"Morgan": str})
    required = {"molecule_id", "first_training_row", "Morgan"}
    missing = required - set(molecules.columns)
    if missing:
        raise ValueError(f"molecule background missing columns: {sorted(missing)}")
    # The training pool has ~300k molecules; parse only the sampled rows to avoid
    # a ~5 GB dense matrix, but check every string's length up front.
    bad_length = molecules.index[molecules.Morgan.str.len().ne(MORGAN_BITS)].tolist()
    if bad_length:
        raise ValueError(f"molecule background rows with Morgan length != {MORGAN_BITS}: {bad_length[:20]}")
    if molecules.duplicated("Morgan").any():
        rows = molecules.index[molecules.duplicated("Morgan", keep=False)].tolist()
        raise ValueError(f"molecule background contains duplicate Morgan strings at rows {rows[:20]}")
    selected_indices = sample_rows(len(molecules), size, np.random.default_rng(seed))
    selected = molecules.iloc[selected_indices].copy()
    selected.insert(0, "source_row", selected_indices)
    selected["requested_background_size"] = str(size)
    selected["actual_background_size"] = len(selected_indices)
    selected["random_seed"] = seed
    selected["used_all_available_because_request_exceeded_pool"] = bool(
        size != "all" and int(size) > len(molecules)
    )
    selected_bits = np.vstack([
        parse_binary(molecules.Morgan.iat[row], MORGAN_BITS, f"molecule background row {row}")
        for row in selected_indices
    ])
    selected.to_csv(output / "molecule_background_selected.csv", index=False)
    return selected.reset_index(drop=True), selected_bits, len(molecules)


def load_kinase_background(size: int | str, seed: int, output: Path):
    kinases = pd.read_csv(KINASE_BACKGROUND)
    missing_mca = [column for column in MCA_COLUMNS if column not in kinases]
    if missing_mca:
        raise ValueError(f"kinase background missing MCA columns: {missing_mca[:10]}")
    required = {"training_kinase_id", "kinase_name", "UniProt_ID", "KLIFS", "Binary_string"}
    if not required.issubset(kinases.columns):
        raise ValueError(f"kinase background is missing columns {sorted(required - set(kinases.columns))}")
    kinases.insert(0, "source_row", np.arange(len(kinases)))
    kinases.to_csv(output / "kinase_background_candidates.csv", index=False)
    pool_size = len(kinases)
    selected_indices = sample_rows(len(kinases), size, np.random.default_rng(seed + 2))
    kinases = kinases.iloc[selected_indices].reset_index(drop=True)
    kinase_mca = kinases[MCA_COLUMNS].to_numpy(np.float32)
    if not np.isfinite(kinase_mca).all():
        row, column = np.argwhere(~np.isfinite(kinase_mca))[0]
        raise ValueError(f"kinase background non-finite MCA at row={row}, column={MCA_COLUMNS[column]}")
    kinase_bits = np.vstack([
        parse_binary(value, KLIFS_BITS, f"kinase background row {row}")
        for row, value in zip(kinases.source_row, kinases.Binary_string)
    ])
    kinases.to_csv(output / "kinase_background_selected.csv", index=False)
    return kinases, kinase_mca, kinase_bits, pool_size


def find_target_protein(identifier: str, metadata: pd.DataFrame) -> int:
    hits = metadata.index[
        metadata["UniProt_ID"].astype(str).eq(identifier)
        | metadata["kinase_name"].astype(str).eq(identifier)
    ].tolist()
    if len(hits) != 1:
        raise ValueError(f"target kinase {identifier!r} matched rows {hits}")
    return hits[0]


def klifs_category_names(bits: np.ndarray, label: str) -> list[str]:
    """Categorical MCA variables for one kinase; an all-zero block is the Gap category."""
    blocks = np.asarray(bits).reshape(KLIFS_RESIDUES, KLIFS_BITS_PER_RESIDUE)
    names = []
    for residue in range(KLIFS_RESIDUES):
        offset = 0
        for system, categories in KLIFS_SYSTEMS:
            active = np.flatnonzero(blocks[residue, offset:offset + len(categories)])
            if len(active) > 1:
                raise ValueError(f"{label}: res{residue+1:02d}_{system} has {len(active)} active categories")
            category = categories[active[0]] if len(active) else "Gap"
            names.append(f"res{residue+1:02d}_{system}__{category}")
            offset += len(categories)
    return names


def load_mca_projection():
    """MCA row coordinates are linear in the category indicator matrix:
    mca_k = sum_c I_c G_ck / (Q sqrt(lambda_k)), with G the category principal
    coordinates and Q the number of categorical variables."""
    coordinates = pd.read_csv(MCA_CATEGORY_COORDINATES, index_col=0)
    missing = [column for column in MCA_COLUMNS if column not in coordinates]
    if missing:
        raise ValueError(f"MCA category coordinates missing columns {missing[:10]}")
    spectrum = pd.read_csv(MCA_INERTIA_SPECTRUM).sort_values("n_components")
    eigenvalues = spectrum["raw_eigenvalue"].to_numpy(np.float64)[:PROTEIN_DIMS]
    if len(eigenvalues) != PROTEIN_DIMS or np.any(eigenvalues <= 0):
        raise ValueError("MCA inertia spectrum must contain 200 positive eigenvalues")
    variables = KLIFS_RESIDUES * len(KLIFS_SYSTEMS)
    projection = (coordinates[MCA_COLUMNS].to_numpy(np.float64)
                  / (variables * np.sqrt(eigenvalues))[None, :])
    names = coordinates.index.astype(str).tolist()
    rows = []
    for name in names:
        variable, category = name.split("__")
        residue_text, system = variable.split("_", 1)
        if system not in SYSTEM_NAMES:
            raise ValueError(f"unexpected MCA category name {name!r}")
        rows.append({"category_name": name, "residue": int(residue_text[3:]),
                     "system": system, "category": category})
    lookup = pd.DataFrame(rows)
    return names, {name: i for i, name in enumerate(names)}, projection, lookup


def kinase_indicator(bits: np.ndarray, index: dict, label: str) -> np.ndarray:
    indicator = np.zeros(len(index), np.float64)
    for name in klifs_category_names(bits, label):
        if name not in index:
            raise ValueError(f"{label}: category {name} is not in the fitted MCA")
        indicator[index[name]] = 1.0
    return indicator


def reference_rescale(
    target_bits: np.ndarray,
    reference_bits: np.ndarray,
    encoder_selected: np.ndarray,
    target_latent: np.ndarray,
    reference_latent: np.ndarray,
    latent_shap: np.ndarray,
    label: str,
) -> dict:
    delta_bits = np.asarray(target_bits, np.float64) - np.asarray(reference_bits, np.float64)
    weights = np.asarray(encoder_selected, np.float64)
    phi = np.asarray(latent_shap, np.float64)
    terms = delta_bits[:, None] * weights
    denominators = np.sum(terms, axis=0, dtype=np.float64)
    forward_delta = np.asarray(target_latent, np.float64) - np.asarray(reference_latent, np.float64)
    difference = denominators - forward_delta
    if not np.allclose(denominators, forward_delta,
                       rtol=FORWARD_DIFFERENCE_RTOL, atol=FORWARD_DIFFERENCE_ATOL):
        latent = int(np.argmax(np.abs(difference)))
        raise AssertionError(
            f"{label}: encoder difference mismatch at latent {latent}: "
            f"sum_terms={denominators[latent]:.17g}, z-z_ref={forward_delta[latent]:.17g}, "
            f"difference={difference[latent]:.6g}"
        )
    l1 = np.sum(np.abs(terms), axis=0, dtype=np.float64)
    mapped = denominators != 0.0
    allocations = np.zeros_like(terms)
    allocations[:, mapped] = (
        terms[:, mapped] / denominators[mapped][None, :] * phi[mapped][None, :]
    )
    bad = np.argwhere(~np.isfinite(allocations))
    if bad.size:
        bit, latent = map(int, bad[0])
        raise FloatingPointError(
            f"{label}: non-finite M at bit={bit}, latent={latent}; "
            f"d={denominators[latent]:.17g}, C={terms[bit, latent]:.17g}, "
            f"phi={phi[latent]:.17g}, L1={l1[latent]:.17g}"
        )
    allocated_sum = hp_axis_sum(allocations, axis=0)
    allocated_abs_sum = hp_axis_sum(np.abs(allocations), axis=0)
    conservation_error = np.full(phi.shape, np.nan)
    conservation_error[mapped] = allocated_sum[mapped] - phi[mapped]
    bounds = np.array([
        error_bound(allocated_abs_sum[k] + abs(phi[k]), len(delta_bits))
        for k in range(len(phi))
    ])
    failed = mapped & (np.abs(conservation_error) > bounds)
    if failed.any():
        latent = int(np.flatnonzero(failed)[0])
        raise AssertionError(
            f"{label}: rescale conservation failed latent={latent}, "
            f"error={conservation_error[latent]:.6g}, bound={bounds[latent]:.6g}"
        )
    amplification = np.full(phi.shape, np.nan)
    amplification[mapped] = l1[mapped] / np.abs(denominators[mapped])
    amplification[(~mapped) & (l1 != 0)] = np.inf
    # Amplification only matters where SHAP is actually allocated to inputs; a
    # latent with phi=0 gets zero allocation however small its denominator is.
    active = mapped & (phi != 0.0)
    active_amplification = np.where(active, amplification, np.nan)
    unallocated = np.where(mapped, 0.0, phi)
    zero_anomaly = (~mapped) & (np.abs(phi) > ZERO_DENOMINATOR_SHAP_WARNING)
    return {
        "delta_bits": delta_bits, "terms": terms, "denominators": denominators,
        "forward_delta": forward_delta, "denominator_difference": difference,
        "l1": l1, "mapped": mapped, "allocations": allocations,
        "allocated_sum": allocated_sum, "allocated_abs_sum": allocated_abs_sum,
        "max_abs_allocation": np.max(np.abs(allocations), axis=0),
        "conservation_error": conservation_error, "conservation_bound": bounds,
        "amplification": amplification, "active": active,
        "active_amplification": active_amplification, "unallocated": unallocated,
        "zero_denominator_nonzero_shap_warning": zero_anomaly,
    }


def nanmax_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, np.float64)
    return float(np.nanmax(values)) if np.any(~np.isnan(values)) else float("nan")


def dynamic_range(values: np.ndarray) -> tuple[float, float, float]:
    magnitude = np.abs(np.asarray(values, np.float64))
    nonzero = magnitude[magnitude > 0]
    if nonzero.size == 0:
        return 0.0, 0.0, np.nan
    maximum = float(nonzero.max())
    median = float(np.median(nonzero))
    return maximum, median, maximum / median if median > 0 else np.inf


def accumulate_group_rescale(group, pair_name, target_vector, target_latent, weights,
                             reference_vector, reference_latent, per_ref_phi, original_indices):
    """Run the per-reference rescale for one input group and average afterwards.

    reference_vector(r) and reference_latent(r) give that reference's binary input
    and latent; per_ref_phi has shape (references, latent dims)."""
    references, dims = per_ref_phi.shape
    sum_alloc = np.zeros((len(target_vector), dims), np.float64)
    sum_abs_alloc = np.zeros_like(sum_alloc)
    sum_unallocated = np.zeros(dims, np.float64)
    amplification = np.full((references, dims), np.nan)
    mapped = np.zeros((references, dims), bool)
    zero_warning = np.zeros((references, dims), bool)
    latent_frames, summary_rows = [], []
    feature_column = "molecular_feature" if group == "molecule" else "protein_mca_dimension"
    for r in range(references):
        phi = per_ref_phi[r]
        result = reference_rescale(target_vector, reference_vector(r), weights, target_latent,
                                   reference_latent(r), phi, f"{pair_name}/{group}/reference_{r}")
        sum_alloc += result["allocations"]
        sum_abs_alloc += np.abs(result["allocations"])
        sum_unallocated += result["unallocated"]
        amplification[r] = result["active_amplification"]
        mapped[r] = result["mapped"]
        zero_warning[r] = result["zero_denominator_nonzero_shap_warning"]
        forward_warning = (~result["mapped"]) & (np.abs(result["forward_delta"]) > FORWARD_DIFFERENCE_ATOL)
        # One DataFrame per reference; thousands of references x hundreds of latents
        # is too many Python dicts to hold in memory.
        latent_frames.append(pd.DataFrame({
            "reference_index": r, "pair": pair_name,
            feature_column: np.arange(dims),
            "original_index": original_indices,
            "latent_shap_signed": phi,
            "target_minus_reference_latent": result["forward_delta"],
            "signed_denominator_sum_terms": result["denominators"],
            "denominator_forward_difference": result["denominator_difference"],
            "signed_difference_term_l1": result["l1"],
            "amplification_factor": result["amplification"],
            "mapped": result["mapped"],
            "nonzero_shap_allocated": result["active"],
            "unallocated_shap": result["unallocated"],
            "allocated_signed_sum": result["allocated_sum"],
            "allocated_absolute_sum": result["allocated_abs_sum"],
            "maximum_absolute_input_allocation": result["max_abs_allocation"],
            "allocation_conservation_error": result["conservation_error"],
            "allocation_conservation_error_bound": result["conservation_bound"],
            "large_amplification_warning": result["active_amplification"] > AMPLIFICATION_WARNING,
            "zero_denominator_nonzero_shap_warning": result["zero_denominator_nonzero_shap_warning"],
            "zero_denominator_nonzero_forward_delta_warning": forward_warning,
        }))
        summary_rows.append({
            "reference_index": r, "pair": pair_name,
            "changed_binary_inputs": int(np.count_nonzero(result["delta_bits"])),
            "mapped_latent_count": int(result["mapped"].sum()),
            "unmapped_latent_count": int((~result["mapped"]).sum()),
            "nonzero_shap_latent_count": int(result["active"].sum()),
            "zero_denominator_nonzero_shap_warning_count": int(
                result["zero_denominator_nonzero_shap_warning"].sum()),
            "zero_denominator_nonzero_forward_delta_warning_count": int(forward_warning.sum()),
            "unallocated_signed_shap": hp_sum(result["unallocated"]),
            "unallocated_absolute_shap": hp_sum(np.abs(result["unallocated"])),
            "maximum_amplification_nonzero_shap": nanmax_or_nan(result["active_amplification"]),
            "maximum_amplification_all_latents": nanmax_or_nan(result["amplification"]),
            "maximum_absolute_allocation": float(np.max(result["max_abs_allocation"])),
            "maximum_denominator_forward_difference": float(
                np.max(np.abs(result["denominator_difference"]))),
            "maximum_mapped_conservation_error": nanmax_or_nan(np.abs(result["conservation_error"])),
        })
    return {
        "average_alloc": sum_alloc / references,
        "average_abs_before": sum_abs_alloc / references,
        "average_unallocated": sum_unallocated / references,
        "amplification": amplification, "mapped": mapped, "zero_warning": zero_warning,
        "latent_frames": latent_frames, "summary_rows": summary_rows,
    }


def selection_totals(rescaled, group_phi, selected_latents, pair_name, group, label):
    """Input-level totals for one latent selection, with the conservation check."""
    average_alloc = rescaled["average_alloc"]
    signed = hp_axis_sum(average_alloc[:, selected_latents], axis=1)
    absolute_after = hp_axis_sum(np.abs(average_alloc[:, selected_latents]), axis=1)
    absolute_before = hp_axis_sum(rescaled["average_abs_before"][:, selected_latents], axis=1)
    unallocated_signed = hp_sum(rescaled["average_unallocated"][selected_latents])
    selected_signed = hp_sum(group_phi[selected_latents])
    selected_abs = hp_sum(np.abs(group_phi[selected_latents]))
    conservation = hp_sum(signed) + unallocated_signed - selected_signed
    bound = error_bound(hp_sum(absolute_after) + abs(unallocated_signed) + selected_abs,
                        average_alloc.shape[0] * len(selected_latents))
    if abs(conservation) > bound:
        raise AssertionError(f"{pair_name}/{group}/{label}: average conservation={conservation}, bound={bound}")
    amplification = rescaled["amplification"][:, selected_latents]
    total_abs = hp_sum(np.abs(group_phi))
    return {
        "signed": signed, "absolute_after": absolute_after, "absolute_before": absolute_before,
        "unallocated_signed": unallocated_signed, "selected_signed": selected_signed,
        "bound": bound,
        "summary": {
            "pair": pair_name, "selection": label,
            "selected_latent_count": len(selected_latents),
            "selected_latent_indices": ",".join(map(str, selected_latents)),
            "absolute_average_latent_shap_coverage": selected_abs / total_abs if total_abs else np.nan,
            "selected_average_latent_shap_signed_sum": selected_signed,
            "selected_average_latent_shap_absolute_sum": selected_abs,
            "average_unallocated_signed_residual": unallocated_signed,
            "maximum_selected_reference_amplification_nonzero_shap": nanmax_or_nan(amplification),
            "large_amplification_reference_latent_count": int(np.sum(amplification > AMPLIFICATION_WARNING)),
            "unmapped_reference_latent_count": int(np.sum(~rescaled["mapped"][:, selected_latents])),
            "zero_denominator_nonzero_shap_warning_count": int(
                np.sum(rescaled["zero_warning"][:, selected_latents])),
            "all_input_signed_sum": hp_sum(signed),
            "input_conservation_error": conservation,
        },
    }


def atoms_for_environment(molecule: Chem.Mol, center: int, radius: int) -> list[int]:
    if radius == 0:
        return [center]
    bonds = Chem.FindAtomEnvironmentOfRadiusN(molecule, radius, center)
    atoms = {center}
    for bond_index in bonds:
        bond = molecule.GetBondWithIdx(bond_index)
        atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return sorted(atoms)


def present_bits_to_atoms(molecule, bit_info, signed, absolute, pre_average_absolute):
    atom_signed = np.zeros(molecule.GetNumAtoms(), np.float64)
    atom_absolute = np.zeros(molecule.GetNumAtoms(), np.float64)
    atom_pre_absolute = np.zeros(molecule.GetNumAtoms(), np.float64)
    rows = []
    for bit in sorted(bit_info):
        occurrences = list(bit_info[bit])
        for occurrence, (center, radius) in enumerate(occurrences):
            atoms = atoms_for_environment(molecule, int(center), int(radius))
            divisor = len(occurrences) * len(atoms)
            atom_signed[atoms] += signed[bit] / divisor
            atom_absolute[atoms] += absolute[bit] / divisor
            atom_pre_absolute[atoms] += pre_average_absolute[bit] / divisor
            rows.append({
                "bit": int(bit), "occurrence": occurrence, "center_atom": int(center),
                "radius": int(radius), "atoms": ",".join(map(str, atoms)),
                "per_atom_signed_net": float(signed[bit] / divisor),
                "per_atom_absolute_after_reference_average": float(absolute[bit] / divisor),
                "per_atom_absolute_before_reference_average": float(pre_average_absolute[bit] / divisor),
            })
    for name, atom_values, bit_values in (
        ("signed", atom_signed, signed), ("absolute", atom_absolute, absolute),
        ("pre-average absolute", atom_pre_absolute, pre_average_absolute),
    ):
        expected = hp_sum(bit_values[np.array(sorted(bit_info), dtype=int)])
        error = hp_sum(atom_values) - expected
        bound = error_bound(hp_sum(np.abs(atom_values)) + hp_sum(np.abs(bit_values)),
                            len(atom_values) + len(bit_values))
        if abs(error) > bound:
            raise AssertionError(f"present-bit to atom {name} conservation error={error}, bound={bound}")
    return atom_signed, atom_absolute, atom_pre_absolute, rows


def signed_or_absolute_norm(mode, limit):
    cmap = SOFT_COOLWARM if mode == "signed" else matplotlib.colormaps["Reds"]
    return cmap, colors.Normalize(vmin=-limit if mode == "signed" else 0, vmax=limit)


def draw_atom_map(molecule, values, mode, limit):
    cmap, norm = signed_or_absolute_norm(mode, limit)
    atoms = np.flatnonzero(np.abs(values) > 0).tolist()
    atom_colors = {int(a): tuple(cmap(norm(float(values[a])))[:3]) for a in atoms}
    drawer = rdMolDraw2D.MolDraw2DCairo(900, 650)
    drawer.drawOptions().addAtomIndices = True
    drawer.drawOptions().bondLineWidth = 2
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, molecule, highlightAtoms=atoms, highlightAtomColors=atom_colors
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")


def map_label(mode):
    return ("Reference-matched encoder-rescaled attribution (signed, present bits only)"
            if mode == "signed" else
            "Absolute allocation after reference averaging (present bits only)")


def save_map(image, mode, limit, title, path):
    cmap, norm = signed_or_absolute_norm(mode, limit)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(image); ax.axis("off"); ax.set_title(title)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap); mapper.set_array([])
    fig.colorbar(mapper, ax=ax, fraction=0.035, pad=0.02).set_label(map_label(mode))
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def save_panel(images, mode, limit, title, path):
    cmap, norm = signed_or_absolute_norm(mode, limit)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, k in zip(axes.flat, TOP_K):
        ax.imshow(images[k]); ax.axis("off"); ax.set_title(f"Top-{k}")
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap); mapper.set_array([])
    fig.colorbar(mapper, ax=axes.ravel().tolist(), fraction=0.018, pad=0.015,
                 label=map_label(mode))
    fig.suptitle(title); fig.subplots_adjust(left=.01, right=.92, bottom=.02, top=.92,
                                             wspace=.02, hspace=.08)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def residue_tick_labels(klifs_sequence: str) -> list[str]:
    return [f"{aa}{i}" for i, aa in enumerate(klifs_sequence, start=1)]


def shade_klifs_regions(ax, label_regions: bool):
    for index, (_, short, start, end) in enumerate(KLIFS_REGIONS):
        if index % 2 == 0:
            ax.axvspan(start - .5, end + .5, color="0.93", zorder=0, lw=0)
        if label_regions:
            ax.text((start + end) / 2, 1.01, short, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7)


def save_residue_bar_panel(values_by_k, mode, limit, tick_labels, title, path):
    cmap, norm = signed_or_absolute_norm(mode, limit)
    x = np.arange(1, KLIFS_RESIDUES + 1)
    fig, axes = plt.subplots(len(values_by_k), 1, figsize=(16, 2.4 * len(values_by_k) + 1.2), sharex=True)
    axes = np.atleast_1d(axes)
    for row, (ax, k) in enumerate(zip(axes, values_by_k)):
        values = values_by_k[k]
        shade_klifs_regions(ax, label_regions=row == 0)
        ax.bar(x, values, width=.8, color=[cmap(norm(v)) for v in values],
               edgecolor="0.35", linewidth=.3, zorder=2)
        if mode == "signed":
            ax.axhline(0, color="0.3", lw=.6, zorder=1)
            ax.set_ylim(-limit * 1.08, limit * 1.08)
        else:
            ax.set_ylim(0, limit * 1.08)
        ax.set_xlim(.4, KLIFS_RESIDUES + .6)
        ax.set_ylabel("All MCA" if k == "all" else f"Top-{k}")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(tick_labels, rotation=90, fontsize=6)
    axes[-1].set_xlabel("KLIFS pocket residue (target kinase amino acid and position)")
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap); mapper.set_array([])
    fig.colorbar(mapper, ax=axes.tolist(), fraction=.015, pad=.01).set_label(
        "Signed MCA-rescaled residue attribution" if mode == "signed"
        else "Absolute allocation after reference averaging")
    fig.suptitle(title)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def save_residue_system_panel(values_by_k, limit, tick_labels, title, path):
    cmap, norm = signed_or_absolute_norm("signed", limit)
    fig, axes = plt.subplots(len(values_by_k), 1, figsize=(16, 1.6 * len(values_by_k) + 1.4), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, values_by_k):
        ax.imshow(values_by_k[k].T, aspect="auto", cmap=cmap, norm=norm,
                  extent=(.5, KLIFS_RESIDUES + .5, len(SYSTEM_NAMES) - .5, -.5),
                  interpolation="nearest")
        for _, _, start, end in KLIFS_REGIONS[:-1]:
            ax.axvline(end + .5, color="0.55", lw=.4)
        ax.set_yticks(range(len(SYSTEM_NAMES))); ax.set_yticklabels(SYSTEM_NAMES, fontsize=7)
        ax.set_ylabel("All MCA" if k == "all" else f"Top-{k}", fontsize=9)
    axes[-1].set_xticks(np.arange(1, KLIFS_RESIDUES + 1))
    axes[-1].set_xticklabels(tick_labels, rotation=90, fontsize=6)
    for _, short, start, end in KLIFS_REGIONS:
        axes[0].text((start + end) / 2, 1.03, short, transform=axes[0].get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=7)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap); mapper.set_array([])
    fig.colorbar(mapper, ax=axes.tolist(), fraction=.015, pad=.01).set_label(
        "Signed attribution per residue property system")
    fig.suptitle(title)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def save_waterfall(average_phi, target_row, feature_names, selected, base, raw_margin,
                   group_label, title, path):
    """Waterfall for selected model inputs; every other input is folded into the base."""
    excluded = np.ones(INPUT_DIMS, bool); excluded[selected] = False
    display_base = float(base + average_phi[excluded].sum())
    np.testing.assert_allclose(display_base + average_phi[selected].sum(), raw_margin,
                               rtol=1e-5, atol=1e-5)
    explanation = shap.Explanation(
        values=average_phi[selected], base_values=display_base, data=target_row[selected],
        feature_names=[feature_names[i] for i in selected]
    )
    shap.plots.waterfall(explanation, max_display=len(selected), show=False)
    fig = plt.gcf(); fig.set_size_inches(10, max(5, .31 * len(selected) + 2.8))
    plt.title(title + f"\ndisplay base = mean reference prediction + SHAP of all inputs other than the "
              f"selected {group_label}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    return display_base


def build_config(args, molecule_count: int, kinase_count: int) -> dict:
    return {
        "version": 8, "background_mode": args.background_mode,
        "requested_molecule_background_size": str(args.molecule_background_size),
        "actual_molecule_background_size": molecule_count,
        "requested_kinase_background_size": str(args.kinase_background_size),
        "actual_kinase_background_size": kinase_count,
        "random_seed": args.random_seed, "top_k": list(TOP_K),
        "molecule_dims": MOLECULE_DIMS, "protein_dims": PROTEIN_DIMS,
        "pairs": list(PAIRS),
        "feature_perturbation": "interventional", "model_output": "raw",
        "diagnostic_thresholds": {
            "amplification_warning": AMPLIFICATION_WARNING,
            "color_compression_ratio_warning": COLOR_COMPRESSION_RATIO_WARNING,
            "zero_denominator_shap_warning": ZERO_DENOMINATOR_SHAP_WARNING,
            "fixed_group_shap_tolerance": FIXED_GROUP_ZERO_TOLERANCE,
            "forward_difference_rtol": FORWARD_DIFFERENCE_RTOL,
            "forward_difference_atol": FORWARD_DIFFERENCE_ATOL,
        },
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "files": {
            str(path): sha256(path) for path in
            (SAE_MODEL, SAE_INDICES, XGB_MODEL, KINASE_TABLE, KINASE_MCA,
             MCA_CATEGORY_COORDINATES, MCA_INERTIA_SPECTRUM,
             MOLECULE_BACKGROUND, KINASE_BACKGROUND)
        },
    }


def reference_design(mode: str, molecule_count: int, kinase_count: int, seed: int):
    """Per-reference source rows; -1 means that half of the reference is the target itself."""
    if mode == "joint_independent":
        rng = np.random.default_rng(seed + 1)
        return np.arange(molecule_count), rng.choice(kinase_count, molecule_count, replace=True)
    if mode == "fixed_target":
        return np.arange(molecule_count), np.full(molecule_count, -1)
    return np.full(kinase_count, -1), np.arange(kinase_count)


def write_reference_combinations(mode, molecule_indices, kinase_indices, selected, kinases, output):
    rows = []
    for r, (m, k) in enumerate(zip(molecule_indices, kinase_indices)):
        row = {"reference_index": r, "background_mode": mode}
        if m >= 0:
            row.update({"molecule_source_row": int(selected.loc[m, "source_row"]),
                        "molecule_id": selected.loc[m, "molecule_id"],
                        "first_training_row": selected.loc[m, "first_training_row"]})
        else:
            row.update({"molecule_source_row": np.nan, "molecule_id": "CURRENT_TARGET_PER_CASE",
                        "first_training_row": np.nan})
        if k >= 0:
            row.update({"kinase_source_row": int(kinases.loc[k, "source_row"]),
                        "training_kinase_id": kinases.loc[k, "training_kinase_id"],
                        "kinase_name": kinases.loc[k, "kinase_name"],
                        "UniProt_ID": kinases.loc[k, "UniProt_ID"]})
        else:
            row.update({"kinase_source_row": np.nan, "training_kinase_id": np.nan,
                        "kinase_name": "CURRENT_TARGET_PER_CASE",
                        "UniProt_ID": "CURRENT_TARGET_PER_CASE"})
        rows.append(row)
    combinations = pd.DataFrame(rows)
    combinations.to_csv(output / "reference_combinations.csv", index=False)
    if mode == "fixed_target":
        frequencies = pd.DataFrame([
            {"kinase_name": pair["kinase"], "sampled_frequency": len(combinations),
             "note": "current target protein repeated for every molecule reference"}
            for pair in PAIRS
        ])
    else:
        frequencies = combinations.groupby(
            ["kinase_source_row", "training_kinase_id", "kinase_name", "UniProt_ID"],
            dropna=False).size().rename("sampled_frequency").reset_index()
    frequencies.to_csv(output / "reference_kinase_frequencies.csv", index=False)
    return combinations


def compute_or_load_reference_shap(args, booster, targets, ref_inputs, ref_index,
                                   config_hash, checkpoint_dir):
    path = checkpoint_dir / f"reference_{ref_index:05d}.npz"
    if path.is_file() and not args.force_recompute:
        data = np.load(path, allow_pickle=False)
        if str(data["config_hash"].item()) != config_hash:
            raise ValueError(f"checkpoint config mismatch: {path}")
        return {key: data[key] for key in data.files if key != "config_hash"}, True
    started = time.perf_counter()
    phis = np.zeros((len(PAIRS), INPUT_DIMS), np.float64)
    reference_margins = raw_predict(booster, ref_inputs).astype(np.float64)
    expected_values = np.zeros(len(PAIRS), np.float64)
    # joint_independent uses the same reference row for every pair, so one
    # explainer explains both targets; fixed modes need one explainer per pair.
    if all(np.array_equal(ref_inputs[0], row) for row in ref_inputs[1:]):
        explainer = shap.TreeExplainer(booster, data=model_frame(booster, ref_inputs[0]),
            feature_perturbation="interventional", model_output="raw")
        phis[:] = np.asarray(explainer.shap_values(model_frame(booster, targets)))
        expected_values[:] = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    else:
        for p in range(len(PAIRS)):
            explainer = shap.TreeExplainer(booster, data=model_frame(booster, ref_inputs[p]),
                feature_perturbation="interventional", model_output="raw")
            phis[p] = np.asarray(explainer.shap_values(model_frame(booster, targets[p]))).reshape(-1)
            expected_values[p] = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    molecule_max = np.max(np.abs(phis[:, :MOLECULE_DIMS]), axis=1)
    protein_max = np.max(np.abs(phis[:, MOLECULE_DIMS:]), axis=1)
    fixed_group = {"fixed_target": ("protein", protein_max),
                   "fixed_molecule": ("molecule", molecule_max)}.get(args.background_mode)
    if fixed_group and np.any(fixed_group[1] > FIXED_GROUP_ZERO_TOLERANCE):
        p = int(np.argmax(fixed_group[1]))
        raise AssertionError(
            f"{args.background_mode} reference {ref_index}, pair {PAIRS[p]['pair']}: "
            f"{fixed_group[0]} SHAP max={fixed_group[1][p]:.6g} > {FIXED_GROUP_ZERO_TOLERANCE}"
        )
    target_margins = raw_predict(booster, targets).astype(np.float64)
    additivity = reference_margins + phis.sum(axis=1) - target_margins
    additivity_bounds = np.array([
        tree_additivity_bound(reference_margins[p], target_margins[p], phis[p])
        for p in range(len(PAIRS))
    ])
    expected_reference_errors = expected_values - reference_margins
    expected_reference_bounds = np.array([
        32 * np.finfo(np.float32).eps
        * max(abs(expected_values[p]) + abs(reference_margins[p]), 1.0)
        for p in range(len(PAIRS))
    ])
    if np.any(np.abs(expected_reference_errors) > expected_reference_bounds):
        raise AssertionError(
            f"reference {ref_index}: TreeSHAP expected-value errors "
            f"{expected_reference_errors} exceed float32 bounds {expected_reference_bounds}"
        )
    if np.any(np.abs(additivity) > additivity_bounds):
        raise AssertionError(
            f"reference {ref_index}: additivity errors {additivity} exceed derived "
            f"float32 bounds {additivity_bounds}"
        )
    result = {
        "shap_values": phis, "reference_margins": reference_margins,
        "expected_values": expected_values,
        "expected_reference_errors": expected_reference_errors,
        "expected_reference_error_bounds": expected_reference_bounds,
        "additivity_errors": additivity,
        "additivity_error_bounds": additivity_bounds,
        "molecule_shap_max_abs": molecule_max,
        "protein_shap_max_abs": protein_max,
        "elapsed_seconds": np.array(time.perf_counter() - started),
    }
    np.savez_compressed(path, config_hash=np.array(config_hash), **result)
    return result, False


def validate_direct_background(booster, targets, ref_input_cube, per_ref_shap,
                               per_ref_margins, output, n=4):
    n = min(n, len(ref_input_cube))
    rows = []
    for p, pair in enumerate(PAIRS):
        background = ref_input_cube[:n, p]
        explainer = shap.TreeExplainer(
            booster, data=model_frame(booster, background),
            feature_perturbation="interventional", model_output="raw")
        direct = np.asarray(explainer.shap_values(model_frame(booster, targets[p]))).reshape(-1)
        single_mean = per_ref_shap[:n, p].mean(axis=0)
        direct_rows = int(np.asarray(explainer.data).shape[0])
        difference = float(np.max(np.abs(direct - single_mean)))
        direct_base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        mean_single_base = float(per_ref_margins[:n, p].mean())
        base_difference = abs(direct_base - mean_single_base)
        base_bound = 16 * np.finfo(np.float32).eps * max(
            abs(direct_base) + abs(mean_single_base), 1.0
        )
        if direct_rows != n or difference > 1e-8 or base_difference > base_bound:
            raise AssertionError(
                f"direct-background comparison failed {pair['pair']}: rows={direct_rows}/{n}, "
                f"SHAP difference={difference}, base difference={base_difference}, "
                f"base bound={base_bound}"
            )
        rows.append({"pair": pair["pair"], "small_background_size": n,
                     "direct_explainer_background_rows": direct_rows,
                     "max_abs_shap_difference": difference,
                     "base_value_difference": base_difference,
                     "base_value_difference_bound": base_bound,
                     "implicit_background_sampling_detected": direct_rows != n,
                     "status": "PASS"})
    pd.DataFrame(rows).to_csv(output / "single_reference_vs_direct_background_validation.csv",
                              index=False)


def validate_mca_projection(projection, target_indicators, target_mca, kinase_indicators,
                            kinase_mca, kinases, output):
    rows = []
    checks = [(pair["pair"], target_indicators[i], target_mca[i]) for i, pair in enumerate(PAIRS)]
    checks += [(f"background_{int(kinases.loc[k, 'source_row'])}_{kinases.loc[k, 'kinase_name']}",
                kinase_indicators[k], kinase_mca[k]) for k in range(len(kinase_mca))]
    for label, indicator, stored in checks:
        reconstructed = indicator @ projection
        error = np.abs(reconstructed - np.asarray(stored, np.float64))
        if not np.allclose(reconstructed, stored, rtol=FORWARD_DIFFERENCE_RTOL,
                           atol=FORWARD_DIFFERENCE_ATOL):
            dim = int(np.argmax(error))
            raise AssertionError(
                f"MCA projection does not reproduce {label} at mca_{dim+1}: "
                f"reconstructed={reconstructed[dim]:.9g}, stored={stored[dim]:.9g}")
        rows.append({"kinase": label, "max_abs_reconstruction_error": float(error.max()),
                     "status": "PASS"})
    pd.DataFrame(rows).to_csv(output / "mca_linear_projection_validation.csv", index=False)
    return max(row["max_abs_reconstruction_error"] for row in rows)


def synthetic_validations() -> pd.DataFrame:
    rows = []
    def check(name, condition, details):
        if not condition:
            raise AssertionError(f"synthetic V9 test failed: {name}: {details}")
        rows.append({"test": name, "status": "PASS", "details": details})
    weights = np.array([[2.0], [-1.0], [3.0]])
    target = np.array([1.0, 1.0, 0.0]); reference = np.array([0.0, 0.0, 1.0])
    bias = .7
    target_z = np.array([bias + target @ weights[:, 0]])
    reference_z = np.array([bias + reference @ weights[:, 0]])
    result = reference_rescale(target, reference, weights, target_z, reference_z,
                               np.array([-0.6]), "synthetic_bias_negative")
    check("bias_cancels_and_negative_shap_is_preserved",
          np.isclose(result["denominators"][0], target_z[0]-reference_z[0])
          and np.isclose(result["allocated_sum"][0], -.6),
          "d=z-z_ref from signed bit differences; bias cancels")
    weights = np.array([[1.0], [-1.0 + 1e-12]])
    d = weights[:, 0].sum()
    result = reference_rescale(np.ones(2), np.zeros(2), weights, np.array([.2+d]),
                               np.array([.2]), np.array([.5]), "synthetic_small_d")
    check("small_nonzero_denominator_is_used", result["mapped"][0]
          and result["amplification"][0] > 1e10, f"d={d}, amp={result['amplification'][0]}")
    result = reference_rescale(np.ones(2), np.zeros(2), weights, np.array([.2+d]),
                               np.array([.2]), np.array([0.0]), "synthetic_small_d_zero_shap")
    check("zero_shap_amplification_excluded_from_diagnostics",
          not result["active"][0] and np.isnan(result["active_amplification"][0])
          and not np.any(result["allocations"]),
          "phi=0 gives zero allocation, so its amplification is not reported")
    weights = np.array([[1.0], [-1.0]])
    result = reference_rescale(np.ones(2), np.zeros(2), weights, np.array([.3]),
                               np.array([.3]), np.array([.4]), "synthetic_zero_d")
    check("exact_zero_denominator_is_unallocated", not result["mapped"][0]
          and result["unallocated"][0] == .4 and np.isfinite(result["allocations"]).all(),
          "zero allocation plus explicit residual")
    result = reference_rescale(np.array([0., 1.]), np.array([1., 0.]),
                               np.array([[2.], [1.]]), np.array([1.]), np.array([2.]),
                               np.array([.5]), "synthetic_absent_bit")
    check("absent_target_bit_can_receive_attribution", result["allocations"][0, 0] != 0,
          "target bit=0 and reference bit=1 gives delta=-1")
    # One categorical variable with three categories: target in A, reference in B.
    projection = np.array([[.9, .1], [-.4, .3], [.2, -.5]])
    target_i = np.array([1., 0., 0.]); reference_i = np.array([0., 1., 0.])
    result = reference_rescale(target_i, reference_i, projection, target_i @ projection,
                               reference_i @ projection, np.array([.8, -.2]), "synthetic_category_swap")
    check("category_swap_allocates_to_target_and_reference_categories",
          np.allclose(result["allocations"].sum(axis=0), [.8, -.2])
          and np.all(result["allocations"][2] == 0) and np.all(result["allocations"][:2] != 0),
          "delta_I=+1 on the target category and -1 on the reference category; unchanged category gets 0")
    return pd.DataFrame(rows)


def explain_molecule(pair, pair_index, molecule, target_bits, bit_info, target_latent,
                     per_ref_shap, molecule_indices, selected_bits, ref_latents,
                     encoder_selected, original_indices, average_phi, average_base,
                     target_row, target_margin, target_probability, feature_names,
                     pair_dir, tables):
    pair_name = pair["pair"]
    molecular_phi = average_phi[:MOLECULE_DIMS]
    rescaled = accumulate_group_rescale(
        "molecule", pair_name, target_bits, target_latent, encoder_selected,
        lambda r: selected_bits[molecule_indices[r]], lambda r: ref_latents[molecule_indices[r]],
        per_ref_shap[:, pair_index, :MOLECULE_DIMS], original_indices)
    tables["molecule_latent_diagnostics"] += rescaled["latent_frames"]
    tables["molecule_reference_diagnostics"] += rescaled["summary_rows"]
    ranking = np.argsort(-np.abs(molecular_phi), kind="stable")
    rank_lookup = np.empty(MOLECULE_DIMS, int); rank_lookup[ranking] = np.arange(1, MOLECULE_DIMS + 1)
    tables["molecule_latents"].append(pd.DataFrame({
        "pair": pair_name, "molecular_feature": np.arange(MOLECULE_DIMS),
        "original_encoder_index": original_indices,
        "rank_by_abs_average_shap": rank_lookup,
        "latent_value": target_latent,
        "average_interventional_shap_signed": molecular_phi,
        "absolute_average_shap_for_ranking": np.abs(molecular_phi),
        "average_unallocated_shap": rescaled["average_unallocated"],
    }))
    present_mask = target_bits.astype(bool); absent_mask = ~present_mask
    plot_results = {}
    # Export all-latent summaries and maps in addition to the requested top-k.
    for label, selected_latents in [(f"top_{k}", ranking[:k]) for k in TOP_K] + [("all", ranking)]:
        totals = selection_totals(rescaled, molecular_phi, selected_latents, pair_name, "molecule", label)
        signed, absolute_after, absolute_before = (
            totals["signed"], totals["absolute_after"], totals["absolute_before"])
        present_signed = hp_sum(signed[present_mask]); absent_signed = hp_sum(signed[absent_mask])
        atoms_signed, atoms_absolute, atoms_pre, environments = present_bits_to_atoms(
            molecule, bit_info, np.where(present_mask, signed, 0.0),
            np.where(present_mask, absolute_after, 0.0), np.where(present_mask, absolute_before, 0.0))
        atom_conservation = (hp_sum(atoms_signed) + absent_signed + totals["unallocated_signed"]
                             - totals["selected_signed"])
        if abs(atom_conservation) > totals["bound"]:
            raise AssertionError(f"{pair_name}/{label}: atom+absent conservation={atom_conservation}")
        signed_atom_max, signed_atom_median, signed_atom_range = dynamic_range(atoms_signed)
        absolute_atom_max, absolute_atom_median, absolute_atom_range = dynamic_range(atoms_absolute)
        tables["molecule_topk"].append({
            **totals["summary"],
            "present_bit_signed_sum": present_signed,
            "absent_bit_signed_sum": absent_signed,
            "present_bit_absolute_after_reference_average": hp_sum(absolute_after[present_mask]),
            "absent_bit_absolute_after_reference_average": hp_sum(absolute_after[absent_mask]),
            "present_bit_absolute_before_reference_average": hp_sum(absolute_before[present_mask]),
            "absent_bit_absolute_before_reference_average": hp_sum(absolute_before[absent_mask]),
            "atom_signed_sum": hp_sum(atoms_signed),
            "atom_absolute_after_reference_average_sum": hp_sum(atoms_absolute),
            "atom_absolute_before_reference_average_sum": hp_sum(atoms_pre),
            "atom_plus_absent_conservation_error": atom_conservation,
            "signed_atom_max_absolute": signed_atom_max,
            "signed_atom_median_nonzero_absolute": signed_atom_median,
            "signed_atom_dynamic_range_ratio": signed_atom_range,
            "absolute_atom_max": absolute_atom_max,
            "absolute_atom_median_nonzero": absolute_atom_median,
            "absolute_atom_dynamic_range_ratio": absolute_atom_range,
            "color_compression_ratio_warning_threshold": COLOR_COMPRESSION_RATIO_WARNING,
            "color_compression_warning": bool(
                (np.isfinite(signed_atom_range) and signed_atom_range > COLOR_COMPRESSION_RATIO_WARNING)
                or (np.isfinite(absolute_atom_range) and absolute_atom_range > COLOR_COMPRESSION_RATIO_WARNING)),
            "average_reference_raw_margin": average_base,
            "target_raw_margin": target_margin,
            "target_probability": target_probability,
        })
        tables["molecule_bits"].append(pd.DataFrame({
            "pair": pair_name, "selection": label, "bit": np.arange(MORGAN_BITS),
            "present_in_target": present_mask,
            "signed_net_allocation": signed,
            "absolute_allocation_after_reference_average": absolute_after,
            "absolute_allocation_before_reference_average": absolute_before,
        }))
        tables["molecule_atoms"].append(pd.DataFrame({
            "pair": pair_name, "selection": label, "atom_id": np.arange(molecule.GetNumAtoms()),
            "atom_symbol": [atom.GetSymbol() for atom in molecule.GetAtoms()],
            "signed_present_bit_allocation": atoms_signed,
            "absolute_present_bit_allocation_after_reference_average": atoms_absolute,
            "absolute_present_bit_allocation_before_reference_average": atoms_pre,
        }))
        tables["molecule_environments"] += [{"pair": pair_name, "selection": label, **row}
                                            for row in environments]
        if label == "all":
            for mode, values in (("signed", atoms_signed), ("absolute", atoms_absolute)):
                limit = float(np.max(np.abs(values))) or 1.0
                image = draw_atom_map(molecule, values, mode, limit)
                save_map(image, mode, limit,
                         f"{pair['drug_name']}–{pair['kinase']}: all {MOLECULE_DIMS} molecular latents, "
                         f"{mode} attribution\natom map shows present-bit portion only",
                         pair_dir / f"{mode}_all.png")
        if label != "all":
            k = len(selected_latents)
            plot_results[k] = (atoms_signed, atoms_absolute)
            save_waterfall(
                average_phi, target_row, feature_names, selected_latents, average_base,
                target_margin, f"molecular latents top-{k}",
                f"{pair['drug_name']}–{pair['kinase']}: average interventional latent TreeSHAP top-{k}",
                pair_dir / f"molecular_waterfall_top{k:02d}.png")
    signed_limit = max(np.max(np.abs(value[0])) for value in plot_results.values()) or 1.0
    absolute_limit = max(np.max(value[1]) for value in plot_results.values()) or 1.0
    signed_images, absolute_images = {}, {}
    for k, (atom_signed, atom_absolute) in plot_results.items():
        signed_images[k] = draw_atom_map(molecule, atom_signed, "signed", signed_limit)
        absolute_images[k] = draw_atom_map(molecule, atom_absolute, "absolute", absolute_limit)
        save_map(signed_images[k], "signed", signed_limit,
                 f"{pair['drug_name']}–{pair['kinase']}: top-{k} reference-matched signed attribution\n"
                 "atom map shows present-bit portion only",
                 pair_dir / f"signed_top{k:02d}.png")
        save_map(absolute_images[k], "absolute", absolute_limit,
                 f"{pair['drug_name']}–{pair['kinase']}: top-{k} absolute allocation\n"
                 "after reference averaging; present-bit portion only",
                 pair_dir / f"absolute_top{k:02d}.png")
    save_panel(signed_images, "signed", signed_limit,
               f"{pair['drug_name']}–{pair['kinase']}: Reference-matched encoder-rescaled attribution",
               pair_dir / "signed_topk_panel.png")
    save_panel(absolute_images, "absolute", absolute_limit,
               f"{pair['drug_name']}–{pair['kinase']}: absolute allocation after reference averaging",
               pair_dir / "absolute_topk_panel.png")


def explain_protein(pair, pair_index, target_indicator, target_mca, klifs_sequence,
                    per_ref_shap, kinase_indices, kinase_indicators, kinase_mca,
                    projection, category_lookup, average_phi, average_base, target_row,
                    target_margin, target_probability, feature_names, pair_dir, tables):
    pair_name = pair["pair"]
    protein_phi = average_phi[MOLECULE_DIMS:]
    rescaled = accumulate_group_rescale(
        "protein", pair_name, target_indicator, target_mca, projection,
        lambda r: kinase_indicators[kinase_indices[r]], lambda r: kinase_mca[kinase_indices[r]],
        per_ref_shap[:, pair_index, MOLECULE_DIMS:], np.arange(1, PROTEIN_DIMS + 1))
    tables["protein_mca_diagnostics"] += rescaled["latent_frames"]
    tables["protein_reference_diagnostics"] += rescaled["summary_rows"]
    ranking = np.argsort(-np.abs(protein_phi), kind="stable")
    rank_lookup = np.empty(PROTEIN_DIMS, int); rank_lookup[ranking] = np.arange(1, PROTEIN_DIMS + 1)
    tables["protein_mca"].append(pd.DataFrame({
        "pair": pair_name, "protein_mca_dimension": np.arange(1, PROTEIN_DIMS + 1),
        "rank_by_abs_average_shap": rank_lookup,
        "mca_value": target_mca,
        "average_interventional_shap_signed": protein_phi,
        "absolute_average_shap_for_ranking": np.abs(protein_phi),
        "average_unallocated_shap": rescaled["average_unallocated"],
    }))
    residue_index = category_lookup["residue"].to_numpy() - 1
    system_index = category_lookup["system"].map({s: i for i, s in enumerate(SYSTEM_NAMES)}).to_numpy()
    cell_index = residue_index * len(SYSTEM_NAMES) + system_index
    in_target = target_indicator.astype(bool)
    residue_sum = lambda weights: np.bincount(residue_index, weights=weights, minlength=KLIFS_RESIDUES)
    cell_sum = lambda weights: np.bincount(
        cell_index, weights=weights, minlength=KLIFS_RESIDUES * len(SYSTEM_NAMES)
    ).reshape(KLIFS_RESIDUES, len(SYSTEM_NAMES))
    region_of = {res: name for name, _, start, end in KLIFS_REGIONS for res in range(start, end + 1)}
    tick_labels = residue_tick_labels(klifs_sequence)
    signed_by_k, absolute_by_k, system_by_k = {}, {}, {}
    for label, selected_latents in [(f"top_{k}", ranking[:k]) for k in TOP_K] + [("all", ranking)]:
        totals = selection_totals(rescaled, protein_phi, selected_latents, pair_name, "protein", label)
        signed, absolute_after, absolute_before = (
            totals["signed"], totals["absolute_after"], totals["absolute_before"])
        residue_signed = residue_sum(signed)
        residue_conservation = (hp_sum(residue_signed) + totals["unallocated_signed"]
                                - totals["selected_signed"])
        if abs(residue_conservation) > totals["bound"]:
            raise AssertionError(f"{pair_name}/{label}: residue conservation={residue_conservation}")
        residue_absolute = residue_sum(absolute_after)
        residue_system_signed = cell_sum(signed)
        signed_max, signed_median, signed_range = dynamic_range(residue_signed)
        tables["protein_topk"].append({
            **totals["summary"],
            "target_category_signed_sum": hp_sum(signed[in_target]),
            "reference_category_signed_sum": hp_sum(signed[~in_target]),
            "target_category_absolute_after_reference_average": hp_sum(absolute_after[in_target]),
            "reference_category_absolute_after_reference_average": hp_sum(absolute_after[~in_target]),
            "residue_signed_sum": hp_sum(residue_signed),
            "residue_absolute_after_reference_average_sum": hp_sum(residue_absolute),
            "residue_conservation_error": residue_conservation,
            "signed_residue_max_absolute": signed_max,
            "signed_residue_median_nonzero_absolute": signed_median,
            "signed_residue_dynamic_range_ratio": signed_range,
            "average_reference_raw_margin": average_base,
            "target_raw_margin": target_margin,
            "target_probability": target_probability,
        })
        tables["protein_categories"].append(pd.DataFrame({
            "pair": pair_name, "selection": label,
            "category_name": category_lookup["category_name"],
            "residue": category_lookup["residue"], "system": category_lookup["system"],
            "category": category_lookup["category"],
            "target_category": in_target,
            "signed_net_allocation": signed,
            "absolute_allocation_after_reference_average": absolute_after,
            "absolute_allocation_before_reference_average": absolute_before,
        }))
        tables["protein_residues"].append(pd.DataFrame({
            "pair": pair_name, "selection": label,
            "klifs_residue": np.arange(1, KLIFS_RESIDUES + 1),
            "klifs_region": [region_of[i] for i in range(1, KLIFS_RESIDUES + 1)],
            "target_amino_acid": list(klifs_sequence),
            "signed_allocation": residue_signed,
            "signed_from_target_categories": residue_sum(np.where(in_target, signed, 0.0)),
            "signed_from_reference_categories": residue_sum(np.where(in_target, 0.0, signed)),
            "absolute_allocation_after_reference_average": residue_absolute,
            "absolute_allocation_before_reference_average": residue_sum(absolute_before),
        }))
        tables["protein_residue_systems"].append(pd.DataFrame({
            "pair": pair_name, "selection": label,
            "klifs_residue": np.repeat(np.arange(1, KLIFS_RESIDUES + 1), len(SYSTEM_NAMES)),
            "system": np.tile(SYSTEM_NAMES, KLIFS_RESIDUES),
            "signed_allocation": residue_system_signed.ravel(),
            "absolute_allocation_after_reference_average": cell_sum(absolute_after).ravel(),
            "absolute_allocation_before_reference_average": cell_sum(absolute_before).ravel(),
        }))
        if label == "all":
            title = f"{pair['drug_name']}–{pair['kinase']}: all {PROTEIN_DIMS} protein MCA dimensions"
            for mode, values in (("signed", residue_signed), ("absolute", residue_absolute)):
                limit = float(np.max(np.abs(values))) or 1.0
                save_residue_bar_panel({"all": values}, mode, limit, tick_labels,
                                       title + f": {mode} residue attribution",
                                       pair_dir / f"protein_residue_{mode}_all.png")
            system_limit_all = float(np.max(np.abs(residue_system_signed))) or 1.0
            save_residue_system_panel({"all": residue_system_signed}, system_limit_all,
                                      tick_labels, title + ": signed attribution by residue and property system",
                                      pair_dir / "protein_residue_system_signed_all.png")
        if label != "all":
            k = len(selected_latents)
            signed_by_k[k], absolute_by_k[k], system_by_k[k] = (
                residue_signed, residue_absolute, residue_system_signed)
            save_waterfall(
                average_phi, target_row, feature_names, MOLECULE_DIMS + selected_latents,
                average_base, target_margin, f"protein MCA dimensions top-{k}",
                f"{pair['drug_name']}–{pair['kinase']}: average interventional MCA TreeSHAP top-{k}",
                pair_dir / f"protein_waterfall_top{k:02d}.png")
    signed_limit = max(np.max(np.abs(v)) for v in signed_by_k.values()) or 1.0
    absolute_limit = max(np.max(v) for v in absolute_by_k.values()) or 1.0
    system_limit = max(np.max(np.abs(v)) for v in system_by_k.values()) or 1.0
    save_residue_bar_panel(signed_by_k, "signed", signed_limit, tick_labels,
                           f"{pair['drug_name']}–{pair['kinase']}: reference-matched MCA-rescaled "
                           "residue attribution (signed)", pair_dir / "protein_residue_signed_topk_panel.png")
    save_residue_bar_panel(absolute_by_k, "absolute", absolute_limit, tick_labels,
                           f"{pair['drug_name']}–{pair['kinase']}: residue absolute allocation after "
                           "reference averaging", pair_dir / "protein_residue_absolute_topk_panel.png")
    save_residue_system_panel(system_by_k, system_limit, tick_labels,
                              f"{pair['drug_name']}–{pair['kinase']}: signed attribution by residue "
                              "and property system", pair_dir / "protein_residue_system_signed_topk_panel.png")


def main(args) -> None:
    started = time.perf_counter()
    mode = args.background_mode
    explains_molecule = mode != "fixed_molecule"
    explains_protein = mode != "fixed_target"
    output = args.output_root / mode
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    synthetic = synthetic_validations(); synthetic.to_csv(output / "synthetic_validation.csv", index=False)

    if explains_molecule:
        selected, selected_bits, molecule_pool_size = load_molecule_background(
            args.molecule_background_size, args.random_seed, output)
    else:
        selected = pd.DataFrame(columns=["source_row", "molecule_id", "first_training_row"])
        selected_bits = np.zeros((0, MORGAN_BITS), np.float32)
        molecule_pool_size = None
    kinases, kinase_mca, kinase_bits, kinase_pool_size = load_kinase_background(
        args.kinase_background_size, args.random_seed, output)
    config = build_config(args, len(selected), len(kinases)); config_hash = stable_json_hash(config)
    config["config_hash"] = config_hash
    config_path = output / "run_config.json"
    if config_path.is_file():
        old = json.loads(config_path.read_text())
        if old.get("config_hash") != config_hash:
            raise ValueError(
                f"cache configuration mismatch in {output}; use a different --output-root "
                "or remove that mode directory explicitly"
            )
    else:
        write_json(config_path, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = SparseLinearAE().to(device)
    sae.load_state_dict(torch.load(SAE_MODEL, map_location=device, weights_only=True)); sae.eval()
    keep_idx = torch.load(SAE_INDICES, map_location=device, weights_only=True).long()[:MOLECULE_DIMS]
    if len(keep_idx) != MOLECULE_DIMS:
        raise ValueError(f"retained molecular dimensions={len(keep_idx)}, expected {MOLECULE_DIMS}")
    original_indices = keep_idx.cpu().numpy()
    encoder_selected = sae.encoder.weight.detach().cpu().numpy()[original_indices].T
    if encoder_selected.shape != (MORGAN_BITS, MOLECULE_DIMS):
        raise ValueError(f"encoder matrix shape={encoder_selected.shape}, expected (4096,417)")

    booster = xgb.Booster(); booster.load_model(XGB_MODEL)
    if booster.num_features() != INPUT_DIMS:
        raise ValueError(f"XGBoost inputs={booster.num_features()}, expected {INPUT_DIMS}")
    target_kinase_meta = pd.read_csv(KINASE_TABLE)
    target_kinase_mca = pd.read_csv(KINASE_MCA).to_numpy(np.float32)
    if target_kinase_mca.shape[1] != PROTEIN_DIMS:
        raise ValueError(f"target kinase MCA width={target_kinase_mca.shape[1]}, expected 200")
    category_names, category_index, projection, category_lookup = load_mca_projection()
    category_lookup.to_csv(output / "mca_category_lookup.csv", index=False)

    targets_data = []
    with torch.no_grad():
        for pair in PAIRS:
            molecule, bits, bit_info = target_molecule(pair["smiles"])
            latent = sae.encoder(torch.as_tensor(bits[None], dtype=torch.float32,
                                                  device=device))[:, keep_idx].cpu().numpy()[0]
            row = find_target_protein(pair["kinase"], target_kinase_meta)
            klifs_bits = parse_binary(target_kinase_meta.loc[row, "Binary_string"], KLIFS_BITS,
                                      f"target kinase {pair['kinase']}")
            targets_data.append({
                "pair": pair, "molecule": molecule, "bits": bits, "bit_info": bit_info,
                "latent": latent, "protein": target_kinase_mca[row],
                "indicator": kinase_indicator(klifs_bits, category_index, f"target {pair['kinase']}"),
                "klifs": str(target_kinase_meta.loc[row, "KLIFS"]),
            })
    targets = np.vstack([np.hstack([t["latent"], t["protein"]]) for t in targets_data]).astype(np.float32)
    target_margins = raw_predict(booster, targets).astype(np.float64)
    target_probabilities = probability_predict(booster, targets).astype(np.float64)
    kinase_indicators = np.vstack([
        kinase_indicator(kinase_bits[k], category_index, f"kinase background row {k}")
        for k in range(len(kinases))
    ])
    mca_projection_error = validate_mca_projection(
        projection, [t["indicator"] for t in targets_data], [t["protein"] for t in targets_data],
        kinase_indicators, kinase_mca, kinases, output)

    if len(selected):
        with torch.no_grad():
            ref_latents = sae.encoder(torch.as_tensor(selected_bits, dtype=torch.float32, device=device)
                                      )[:, keep_idx].cpu().numpy().astype(np.float32)
    else:
        ref_latents = np.zeros((0, MOLECULE_DIMS), np.float32)
    molecule_indices, kinase_indices = reference_design(mode, len(selected), len(kinases), args.random_seed)
    combinations = write_reference_combinations(mode, molecule_indices, kinase_indices,
                                                selected, kinases, output)
    reference_count = len(molecule_indices)
    pd.DataFrame([{
        "background_mode": mode,
        "molecule_background_file_rows": molecule_pool_size,
        "requested_molecule_background_size": str(args.molecule_background_size) if explains_molecule else "not used",
        "actual_molecule_reference_count": len(selected),
        "kinase_background_file_rows": kinase_pool_size,
        "requested_kinase_background_size": str(args.kinase_background_size),
        "kinase_candidate_count": len(kinases),
        "actually_used_kinase_count": (int(combinations["kinase_source_row"].nunique())
                                       if mode != "fixed_target" else len(PAIRS)),
        "reference_count": reference_count,
        "random_seed": args.random_seed,
        "mca_projection_max_abs_reconstruction_error": mca_projection_error,
        "reference_pairing_note": {
            "joint_independent": "independent kinase sampling with replacement; combinations need not be observed training pairs",
            "fixed_target": "current target kinase repeated for each selected background molecule",
            "fixed_molecule": "current target molecule repeated for each selected background kinase",
        }[mode],
    }]).to_csv(output / "background_reference_summary.csv", index=False)
    np.savez_compressed(output / "reference_feature_cache.npz",
                        config_hash=np.array(config_hash), molecule_bits=selected_bits,
                        molecule_latents=ref_latents, molecule_indices=molecule_indices,
                        kinase_indices=kinase_indices)
    reference_cube = np.stack([
        np.stack([
            np.hstack([
                ref_latents[molecule_indices[r]] if molecule_indices[r] >= 0 else targets_data[p]["latent"],
                kinase_mca[kinase_indices[r]] if kinase_indices[r] >= 0 else targets_data[p]["protein"],
            ])
            for p in range(len(PAIRS))
        ])
        for r in range(reference_count)
    ]).astype(np.float32)

    per_ref_shap = np.zeros((reference_count, len(PAIRS), INPUT_DIMS), np.float64)
    per_ref_margins = np.zeros((reference_count, len(PAIRS)), np.float64)
    reference_summary, computed_seconds, observed_seconds = [], [], []
    for r in range(reference_count):
        result, loaded = compute_or_load_reference_shap(
            args, booster, targets, reference_cube[r], r, config_hash, checkpoint_dir)
        per_ref_shap[r] = result["shap_values"]
        per_ref_margins[r] = result["reference_margins"]
        elapsed = float(result["elapsed_seconds"])
        observed_seconds.append(elapsed)
        if not loaded:
            computed_seconds.append(elapsed)
        for p, pair in enumerate(PAIRS):
            reference_summary.append({
                "reference_index": r, "pair": pair["pair"],
                "reference_raw_margin": per_ref_margins[r, p],
                "target_raw_margin": target_margins[p],
                "expected_value_minus_reference_margin": float(result["expected_reference_errors"][p]),
                "expected_reference_error_bound": float(result["expected_reference_error_bounds"][p]),
                "additivity_error": float(result["additivity_errors"][p]),
                "additivity_error_bound": float(result["additivity_error_bounds"][p]),
                "molecule_shap_max_abs": float(result["molecule_shap_max_abs"][p]),
                "protein_shap_max_abs": float(result["protein_shap_max_abs"][p]),
                "elapsed_seconds_for_checkpoint": elapsed,
                "loaded_from_checkpoint": loaded,
            })
        if (r + 1) % args.batch_size == 0 or r + 1 == reference_count:
            pd.DataFrame(reference_summary).to_csv(
                output / "per_reference_tree_shap_summary.partial.csv", index=False)
            write_json(output / "progress.json", {
                "config_hash": config_hash, "completed_references": r + 1,
                "total_references": reference_count, "updated_unix_time": time.time()
            })
    pd.DataFrame(reference_summary).to_csv(output / "per_reference_tree_shap_summary.csv", index=False)
    feature_names = ([f"mol_z_{i}" for i in range(MOLECULE_DIMS)]
                     + [f"protein_mca_{i}" for i in range(1, PROTEIN_DIMS + 1)])
    wide = pd.DataFrame(per_ref_shap.reshape(-1, INPUT_DIMS), columns=feature_names)
    wide.insert(0, "reference_raw_margin", per_ref_margins.ravel())
    wide.insert(0, "pair", np.tile([pair["pair"] for pair in PAIRS], reference_count))
    wide.insert(0, "reference_index", np.repeat(np.arange(reference_count), len(PAIRS)))
    wide.to_csv(output / "per_reference_full_interventional_shap.csv", index=False)
    np.savez_compressed(output / "per_reference_interventional_shap.npz",
                        config_hash=np.array(config_hash), shap_values=per_ref_shap,
                        reference_margins=per_ref_margins)
    validate_direct_background(booster, targets, reference_cube, per_ref_shap, per_ref_margins, output)

    tables = {name: [] for name in (
        "average_features", "molecule_latents", "molecule_reference_diagnostics",
        "molecule_latent_diagnostics", "molecule_bits", "molecule_atoms",
        "molecule_environments", "molecule_topk", "protein_mca", "protein_reference_diagnostics",
        "protein_mca_diagnostics", "protein_categories", "protein_residues",
        "protein_residue_systems", "protein_topk")}
    for p, target in enumerate(targets_data):
        pair = target["pair"]
        pair_dir = output / pair["pair"]; pair_dir.mkdir(exist_ok=True)
        average_phi = per_ref_shap[:, p].mean(axis=0)
        average_base = float(per_ref_margins[:, p].mean())
        average_additivity_error = average_base + average_phi.sum() - target_margins[p]
        average_additivity_bound = tree_additivity_bound(average_base, target_margins[p], average_phi)
        if abs(average_additivity_error) > average_additivity_bound:
            raise AssertionError(
                f"{pair['pair']}: averaged SHAP additivity error {average_additivity_error} "
                f"exceeds derived float32 bound {average_additivity_bound}"
            )
        is_molecule = np.arange(INPUT_DIMS) < MOLECULE_DIMS
        group_rank = np.empty(INPUT_DIMS, int)
        for mask in (is_molecule, ~is_molecule):
            order = np.flatnonzero(mask)[np.argsort(-np.abs(average_phi[mask]), kind="stable")]
            group_rank[order] = np.arange(1, mask.sum() + 1)
        tables["average_features"].append(pd.DataFrame({
            "pair": pair["pair"], "model_input_index": np.arange(INPUT_DIMS),
            "feature_group": np.where(is_molecule, "molecule_latent", "protein_mca"),
            "feature_name": feature_names,
            "average_interventional_shap_signed": average_phi,
            "absolute_average_shap": np.abs(average_phi),
            "rank_within_group_by_abs_average_shap": group_rank,
            "average_reference_raw_margin": average_base,
            "target_raw_margin": float(target_margins[p]),
            "average_shap_additivity_error": average_additivity_error,
        }))
        common = dict(average_phi=average_phi, average_base=average_base, target_row=targets[p],
                      target_margin=float(target_margins[p]),
                      target_probability=float(target_probabilities[p]),
                      feature_names=feature_names, pair_dir=pair_dir, tables=tables)
        if explains_molecule:
            explain_molecule(pair, p, target["molecule"], target["bits"], target["bit_info"],
                             target["latent"], per_ref_shap, molecule_indices, selected_bits,
                             ref_latents, encoder_selected, original_indices, **common)
        if explains_protein:
            explain_protein(pair, p, target["indicator"], target["protein"], target["klifs"],
                            per_ref_shap, kinase_indices, kinase_indicators, kinase_mca,
                            projection, category_lookup, **common)

    pd.concat(tables["average_features"], ignore_index=True).to_csv(
        output / "average_full_input_latent_mca_shap.csv", index=False)
    outputs = {
        "molecule_latents": ("average_molecular_latent_shap_and_ranking.csv", True),
        "molecule_reference_diagnostics": ("per_reference_rescale_diagnostics.csv", False),
        "molecule_latent_diagnostics": ("per_reference_molecular_latent_rescale_diagnostics.csv", True),
        "molecule_bits": ("bit4096_topk_reference_rescale.csv", True),
        "molecule_atoms": ("atom_topk_present_bit_attribution.csv", True),
        "molecule_environments": ("substructure_topk_present_bit_attribution.csv", False),
        "molecule_topk": ("topk_coverage_residual_amplification_summary.csv", False),
        "protein_mca": ("average_protein_mca_shap_and_ranking.csv", True),
        "protein_reference_diagnostics": ("per_reference_protein_rescale_diagnostics.csv", False),
        "protein_mca_diagnostics": ("per_reference_protein_mca_rescale_diagnostics.csv", True),
        "protein_categories": ("protein_category_topk_reference_rescale.csv", True),
        "protein_residues": ("protein_residue_topk_attribution.csv", True),
        "protein_residue_systems": ("protein_residue_system_topk_attribution.csv", True),
        "protein_topk": ("protein_topk_coverage_residual_amplification_summary.csv", False),
    }
    for key, (filename, frames) in outputs.items():
        if not tables[key]:
            continue
        table = pd.concat(tables[key], ignore_index=True) if frames else pd.DataFrame(tables[key])
        table.to_csv(output / filename, index=False)
        if key == "molecule_bits":
            table[table.present_in_target].to_csv(output / "present_bit_topk_reference_rescale.csv", index=False)
            table[~table.present_in_target].to_csv(output / "absent_bit_topk_reference_rescale.csv", index=False)

    mean_observed = float(np.mean(observed_seconds))
    runtime = {
        "background_mode": mode,
        "reference_count": reference_count,
        "total_run_seconds": time.perf_counter() - started,
        "newly_computed_reference_count": len(computed_seconds),
        "mean_new_checkpoint_seconds": float(np.mean(computed_seconds)) if computed_seconds else None,
        "mean_observed_checkpoint_seconds": mean_observed,
        "estimated_tree_shap_seconds_for_all_references": mean_observed * reference_count,
        "note": "Estimate scales the checkpoint-recorded one-reference time linearly and excludes plotting/aggregation."
    }
    write_json(output / "runtime_summary.json", runtime)
    print(json.dumps(runtime, indent=2))
    print(f"Saved V9 {mode} results to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--background-mode", choices=BACKGROUND_MODES, default=BACKGROUND_MODE)
    parser.add_argument("--molecule-background-size", type=parse_background_size,
                        default=MOLECULE_BACKGROUND_SIZE,
                        help="molecule references for joint_independent and fixed_target")
    parser.add_argument("--kinase-background-size", type=parse_background_size,
                        default=KINASE_BACKGROUND_SIZE,
                        help="kinase candidate pool; fixed_molecule uses each one as a reference")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
