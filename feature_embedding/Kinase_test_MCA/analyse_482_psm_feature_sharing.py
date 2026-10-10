#!/usr/bin/env python3
"""Analyse sharing of aligned PSM physicochemical features across 482 kinases.

The amino-acid-to-23-bit mapping is extracted directly from
protein_to_binary_MCA.ipynb.  Feature/group labels are the labels used by the
companion MCA_klifs.ipynb; the conversion notebook itself contains only the
ordered bit strings and does not name the individual bits.
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
_kp = _ks_partial(_ks_project_path, script_dir='feature_embedding/Kinase_test_MCA')


import ast
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
SEQUENCE_CSV = HERE / _kp('feature_embedding/Kinase_test_MCA/human_482_typical_kinase_1955bit.csv')
ONEHOT_CSV = HERE / _kp('feature_embedding/Kinase_test_MCA/human_482_typical_kinase_1955bit_onehot.csv')
CONVERSION_NOTEBOOK = HERE / "protein_to_binary_MCA.ipynb"
MCA_NOTEBOOK = HERE / "MCA_klifs.ipynb"
OUTPUT_DIR = HERE / "M6_PSM_feature_sharing"

N_POSITIONS = 85
N_BITS_PER_POSITION = 23
N_TOTAL_BITS = N_POSITIONS * N_BITS_PER_POSITION

# Bit order and labels used by MCA_klifs.ipynb.  The conversion notebook gives
# the exact ordered 23-bit strings but does not include human-readable labels.
GROUPS = [
    ("side_chain", [
        "Aliphatic", "Aromatic", "Sulfur", "Imino", "Amide", "Hydroxyl",
        "Acidic_sc", "Basic_sc",
    ]),
    ("tertiary_structure", ["Exposed", "Buried", "Intermediate"]),
    ("functional", ["Acidic_fn", "Basic_fn", "Hydrophobic", "Polar"]),
    ("secondary_structure", ["alpha", "beta", "coil"]),
    ("size", ["Tiny", "Small", "Medium", "Large", "VeryLarge"]),
]
FEATURE_NAMES = [f"{group}:{state}" for group, states in GROUPS for state in states]
GROUP_SLICES: dict[str, slice] = {}
_offset = 0
for _group, _states in GROUPS:
    GROUP_SLICES[_group] = slice(_offset, _offset + len(_states))
    _offset += len(_states)
assert _offset == N_BITS_PER_POSITION

GROUP_OUTPUT_NAMES = {
    "side_chain": "side_chain_state",
    "functional": "functional_state",
    "secondary_structure": "secondary_structure_state",
    "tertiary_structure": "tertiary_structure_state",
    "size": "size_state",
}
PERCENTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def extract_mapping_from_notebook(path: Path) -> dict[str, str]:
    """Extract the literal aa_to_binary dictionary without executing notebook code."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "aa_to_binary"
                for target in node.targets
            ):
                mapping = ast.literal_eval(node.value)
                if not isinstance(mapping, dict):
                    raise TypeError("aa_to_binary is not a dictionary")
                return {str(k): str(v) for k, v in mapping.items()}
    raise ValueError(f"aa_to_binary mapping not found in {path}")


def extract_literal_assignments(path: Path, names: set[str]) -> dict[str, object]:
    """Read named literal assignments from a notebook without executing it."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    found: dict[str, object] = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    found[target.id] = ast.literal_eval(node.value)
    missing = names - set(found)
    if missing:
        raise ValueError(f"Assignments {sorted(missing)} not found in {path}")
    return found


def signature_to_group_states(signature: np.ndarray, is_gap: bool) -> dict[str, str]:
    if is_gap:
        return {GROUP_OUTPUT_NAMES[g]: "Gap" for g, _ in GROUPS}
    states: dict[str, str] = {}
    for group, labels in GROUPS:
        block = signature[GROUP_SLICES[group]]
        active = np.flatnonzero(block)
        if len(active) != 1:
            raise ValueError(
                f"Expected exactly one active state in {group}; got {len(active)}"
            )
        states[GROUP_OUTPUT_NAMES[group]] = labels[int(active[0])]
    return states


def sequence_to_bits(sequence: str, mapping: dict[str, str]) -> np.ndarray:
    unknown = sorted(set(sequence) - set(mapping))
    if unknown:
        raise ValueError(f"Sequence contains characters absent from mapping: {unknown}")
    encoded = "".join(mapping[aa] for aa in sequence)
    return np.fromiter((int(bit) for bit in encoded), dtype=np.uint8)


def scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_diagnostics(diagnostics: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean = {key: scalar(value) for key, value in diagnostics.items()}
    (OUTPUT_DIR / "00_input_diagnostics.json").write_text(
        json.dumps(clean, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )



def validate_inputs():
    mapping = extract_mapping_from_notebook(CONVERSION_NOTEBOOK)
    mca_labels = extract_literal_assignments(
        MCA_NOTEBOOK, {"SYSTEM_SIZES", "SYSTEM_NAMES", "SYSTEM_CATEGORIES"}
    )
    sequence_df = pd.read_csv(SEQUENCE_CSV, dtype=str)
    onehot_df = pd.read_csv(ONEHOT_CSV, header=None)

    required = {"kinase_name", "UniProt_ID", "KLIFS", "Binary_string"}
    missing = required - set(sequence_df.columns)
    fatal_errors: list[str] = []
    expected_sizes = [len(states) for _, states in GROUPS]
    expected_names = ["sidechain", "tertiary", "functional", "secondary", "size"]
    expected_categories = [states for _, states in GROUPS]
    if mca_labels["SYSTEM_SIZES"] != expected_sizes:
        fatal_errors.append("PSM group sizes do not match MCA_klifs.ipynb")
    if mca_labels["SYSTEM_NAMES"] != expected_names:
        fatal_errors.append("PSM group names/order do not match MCA_klifs.ipynb")
    if mca_labels["SYSTEM_CATEGORIES"] != expected_categories:
        fatal_errors.append("PSM state names/order do not match MCA_klifs.ipynb")
    if missing:
        fatal_errors.append(f"Missing sequence CSV columns: {sorted(missing)}")

    sequence_lengths = sequence_df["KLIFS"].str.len()
    binary_lengths = sequence_df["Binary_string"].str.len()
    all_binary_values = set(pd.unique(onehot_df.to_numpy().ravel()).tolist())
    mapping_lengths = sorted({len(bits) for bits in mapping.values()})

    if len(sequence_df) != 482 or len(onehot_df) != 482:
        fatal_errors.append(
            f"Expected 482 rows; got {len(sequence_df)} and {len(onehot_df)}"
        )
    if onehot_df.shape[1] != N_TOTAL_BITS:
        fatal_errors.append(f"Expected 1955 one-hot columns; got {onehot_df.shape[1]}")
    if not all_binary_values <= {0, 1}:
        fatal_errors.append(f"One-hot matrix has non-binary values: {all_binary_values}")
    if not (sequence_lengths == N_POSITIONS).all():
        fatal_errors.append("Not all KLIFS sequences have length 85")
    if not (binary_lengths == N_TOTAL_BITS).all():
        fatal_errors.append("Not all Binary_string entries have length 1955")
    if mapping_lengths != [N_BITS_PER_POSITION]:
        fatal_errors.append(f"Mapping bit-string lengths are {mapping_lengths}, not [23]")

    reconstructed = np.vstack(
        [sequence_to_bits(seq, mapping) for seq in sequence_df["KLIFS"]]
    )
    observed = onehot_df.to_numpy(dtype=np.uint8)
    binary_string_matrix = np.vstack(
        [np.fromiter(map(int, value), dtype=np.uint8) for value in sequence_df["Binary_string"]]
    )
    row_matches = np.all(reconstructed == observed, axis=1)
    binary_matches = np.all(binary_string_matrix == observed, axis=1)
    if not row_matches.all():
        fatal_errors.append(
            f"Notebook-derived encoding mismatched rows: {np.flatnonzero(~row_matches).tolist()}"
        )
    if not binary_matches.all():
        fatal_errors.append(
            f"Binary_string mismatched one-hot rows: {np.flatnonzero(~binary_matches).tolist()}"
        )

    # Validate the group partition against every non-gap mapping signature.
    group_partition_valid = True
    group_partition_errors = []
    for aa, bit_string in mapping.items():
        bits = np.fromiter(map(int, bit_string), dtype=np.uint8)
        if aa == "-":
            if bits.sum() != 0:
                group_partition_valid = False
                group_partition_errors.append("Gap is not encoded as 23 zeroes")
            continue
        for group, _ in GROUPS:
            n_active = int(bits[GROUP_SLICES[group]].sum())
            if n_active != 1:
                group_partition_valid = False
                group_partition_errors.append(f"{aa}/{group}: {n_active} active bits")
    if not group_partition_valid:
        fatal_errors.extend(group_partition_errors)

    sample_indices = sorted({0, len(sequence_df) // 4, len(sequence_df) // 2,
                             3 * len(sequence_df) // 4, len(sequence_df) - 1})
    diagnostics = {
        "sequence_csv": str(SEQUENCE_CSV),
        "onehot_csv": str(ONEHOT_CSV),
        "conversion_notebook": str(CONVERSION_NOTEBOOK),
        "feature_label_source": str(MCA_NOTEBOOK),
        "sequence_csv_rows": len(sequence_df),
        "onehot_csv_rows": len(onehot_df),
        "sequence_csv_identifier_columns": ["kinase_name", "UniProt_ID"],
        "onehot_csv_identifier_columns": [],
        "entry_matching_method": "row order; verified against Binary_string and notebook-derived encoding",
        "all_482_entries_match_exactly": bool(row_matches.all() and binary_matches.all()),
        "KLIFS_length_distribution": {
            str(k): int(v) for k, v in sequence_lengths.value_counts().sort_index().items()
        },
        "all_KLIFS_sequences_have_85_positions": bool((sequence_lengths == 85).all()),
        "parsed_binary_feature_columns": onehot_df.shape[1],
        "expected_85_x_23_features": N_TOTAL_BITS,
        "all_parsed_values_are_0_or_1": bool(all_binary_values <= {0, 1}),
        "PSM_feature_names_in_bit_order": FEATURE_NAMES,
        "PSM_groups_in_bit_order": {group: states for group, states in GROUPS},
        "mapping_characters": list(mapping.keys()),
        "gap_encoding": mapping.get("-"),
        "gap_is_23_zeroes": mapping.get("-") == "0" * 23,
        "group_partition_one_active_bit_per_non_gap_AA": group_partition_valid,
        "MCA_label_constants_match_script": not any(
            "MCA_klifs.ipynb" in error for error in fatal_errors
        ),
        "independently_checked_row_indices_zero_based": sample_indices,
        "sample_rows_reproduce_exactly": bool(row_matches[sample_indices].all()),
        "all_rows_reproduce_exactly": bool(row_matches.all()),
        "whole_vector_unique_count": int(np.unique(observed, axis=0).shape[0]),
        "whole_vector_duplicate_row_count": int(len(observed) - np.unique(observed, axis=0).shape[0]),
        "feature_label_provenance_note": (
            "protein_to_binary_MCA.ipynb defines the exact ordered 23-bit mapping but contains "
            "no individual feature names or five-group labels. The labels and contiguous "
            "8/3/4/3/5 grouping are taken from companion MCA_klifs.ipynb and validated "
            "against every amino-acid bit string."
        ),
        "validation_status": "PASS" if not fatal_errors else "FAIL",
        "fatal_errors": fatal_errors,
    }
    write_diagnostics(diagnostics)
    if fatal_errors:
        raise RuntimeError("Input validation failed: " + "; ".join(fatal_errors))
    return sequence_df, observed, reconstructed, mapping, diagnostics


def reconstruct_position_table(sequence_df, bits, mapping):
    signatures = bits.reshape(len(sequence_df), N_POSITIONS, N_BITS_PER_POSITION)
    rows = []
    for kinase_index, record in sequence_df.iterrows():
        for position, aa in enumerate(record["KLIFS"], start=1):
            signature = signatures[kinase_index, position - 1]
            row = {
                "kinase_id": record["kinase_name"],
                "UniProt_ID": record["UniProt_ID"],
                "KLIFS_position": position,
                "amino_acid": aa,
                "is_gap": aa == "-",
            }
            row.update({f"PSM_bit_{i + 1}": int(x) for i, x in enumerate(signature)})
            row.update(signature_to_group_states(signature, aa == "-"))
            rows.append(row)
    long_df = pd.DataFrame(rows)
    long_df.to_csv(OUTPUT_DIR / "00_position_level_reconstructed_features.csv", index=False)
    return long_df, signatures


def group_state_codes(signatures: np.ndarray, sequences: np.ndarray) -> np.ndarray:
    n = signatures.shape[0]
    codes = np.full((n, N_POSITIONS, len(GROUPS)), -1, dtype=np.int8)
    valid = sequences != "-"
    for group_index, (group, _) in enumerate(GROUPS):
        block = signatures[:, :, GROUP_SLICES[group]]
        codes[:, :, group_index] = np.argmax(block, axis=2).astype(np.int8)
        codes[:, :, group_index][~valid] = -1
    return codes


def describe_series(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {k: np.nan for k in ["count", "mean", "SD", "min", "P5", "P25", "median", "P75", "P95", "max"]}
    return {
        "count": int(values.count()),
        "mean": values.mean(),
        "SD": values.std(ddof=1),
        "min": values.min(),
        "P5": values.quantile(0.05),
        "P25": values.quantile(0.25),
        "median": values.median(),
        "P75": values.quantile(0.75),
        "P95": values.quantile(0.95),
        "max": values.max(),
    }


def pairwise_kinase_analysis(sequence_df, bits, signatures, sequences, group_codes):
    n = len(sequence_df)
    pair_i, pair_j = np.triu_indices(n, k=1)
    x = bits.astype(np.int16)
    active_counts = x.sum(axis=1, dtype=np.int32)
    intersections = (x @ x.T)[pair_i, pair_j].astype(np.int32)
    unions = active_counts[pair_i] + active_counts[pair_j] - intersections
    global_jaccard = np.divide(
        intersections, unions, out=np.full(len(pair_i), np.nan), where=unions > 0
    )
    hamming = 1.0 - (
        active_counts[pair_i] + active_counts[pair_j] - 2 * intersections
    ) / N_TOTAL_BITS

    valid_count = np.zeros(len(pair_i), dtype=np.int16)
    aa_same_count = np.zeros(len(pair_i), dtype=np.int16)
    position_jaccard_sum = np.zeros(len(pair_i), dtype=np.float64)
    matching_groups_sum = np.zeros(len(pair_i), dtype=np.int32)
    group_ge_counts = np.zeros((len(pair_i), 5), dtype=np.int16)

    for position in range(N_POSITIONS):
        valid = (sequences[pair_i, position] != "-") & (sequences[pair_j, position] != "-")
        valid_count += valid
        aa_same_count += valid & (sequences[pair_i, position] == sequences[pair_j, position])
        shared = np.sum(
            signatures[pair_i, position] & signatures[pair_j, position], axis=1
        )
        union = np.sum(
            signatures[pair_i, position] | signatures[pair_j, position], axis=1
        )
        jaccard = np.divide(
            shared, union, out=np.zeros(len(pair_i), dtype=float), where=valid & (union > 0)
        )
        position_jaccard_sum += jaccard
        matches = np.sum(
            group_codes[pair_i, position] == group_codes[pair_j, position], axis=1
        ).astype(np.int8)
        matches[~valid] = 0
        matching_groups_sum += matches
        for threshold in range(1, 6):
            group_ge_counts[:, threshold - 1] += valid & (matches >= threshold)

    aa_identity = aa_same_count / valid_count
    mean_position_jaccard = position_jaccard_sum / valid_count
    mean_matching_groups = matching_groups_sum / valid_count
    data = {
        "kinase_1": sequence_df["kinase_name"].to_numpy()[pair_i],
        "kinase_2": sequence_df["kinase_name"].to_numpy()[pair_j],
        "AA_identity": aa_identity,
        "global_PSM_jaccard": global_jaccard,
        "global_hamming_agreement": hamming,
        "mean_position_PSM_jaccard": mean_position_jaccard,
        "mean_matching_PSM_groups": mean_matching_groups,
    }
    for threshold in range(1, 5):
        data[f"fraction_positions_match_ge{threshold}_group" + ("s" if threshold > 1 else "")] = (
            group_ge_counts[:, threshold - 1] / valid_count
        )
    data["fraction_positions_match_5_groups"] = group_ge_counts[:, 4] / valid_count
    data["number_valid_positions"] = valid_count
    pair_df = pd.DataFrame(data)
    pair_df.to_csv(OUTPUT_DIR / "01_pairwise_kinase_similarity.csv", index=False)

    metric_columns = [c for c in pair_df.columns if c not in {"kinase_1", "kinase_2"}]
    summary = pd.DataFrame(
        [{"metric": column, **describe_series(pair_df[column])} for column in metric_columns]
    )
    summary.to_csv(OUTPUT_DIR / "01_pairwise_kinase_similarity_summary.csv", index=False)
    return pair_df


def summarize_residue_class(label, jaccard, shared, matching_groups):
    result = {
        "AA_pair_class": label,
        "number_comparisons": len(jaccard),
        "mean_PSM_jaccard": np.mean(jaccard),
        "median_PSM_jaccard": np.median(jaccard),
        "P5_PSM_jaccard": np.quantile(jaccard, 0.05),
        "P95_PSM_jaccard": np.quantile(jaccard, 0.95),
        "mean_shared_active_properties": np.mean(shared),
        "median_shared_active_properties": np.median(shared),
        "P5_shared_active_properties": np.quantile(shared, 0.05),
        "P95_shared_active_properties": np.quantile(shared, 0.95),
        "mean_matching_PSM_groups": np.mean(matching_groups),
        "median_matching_PSM_groups": np.median(matching_groups),
        "P5_matching_PSM_groups": np.quantile(matching_groups, 0.05),
        "P95_matching_PSM_groups": np.quantile(matching_groups, 0.95),
    }
    for threshold in range(1, 6):
        result[f"fraction_sharing_ge{threshold}_active_properties"] = np.mean(shared >= threshold)
        key = f"probability_matching_ge{threshold}_PSM_groups" if threshold < 5 else "probability_matching_all_5_PSM_groups"
        result[key] = np.mean(matching_groups >= threshold)
    return result


def residue_and_position_analysis(signatures, sequences, group_codes):
    position_rows = []
    different_rows = []
    same_jaccard_parts, different_jaccard_parts = [], []
    same_shared_parts, different_shared_parts = [], []
    same_group_parts, different_group_parts = [], []
    different_group_specific_counts = np.zeros(len(GROUPS), dtype=np.int64)
    total_different = 0

    for position in range(N_POSITIONS):
        valid_indices = np.flatnonzero(sequences[:, position] != "-")
        local_i, local_j = np.triu_indices(len(valid_indices), k=1)
        idx_i, idx_j = valid_indices[local_i], valid_indices[local_j]
        aa_i, aa_j = sequences[idx_i, position], sequences[idx_j, position]
        same_aa = aa_i == aa_j
        shared = np.sum(
            signatures[idx_i, position] & signatures[idx_j, position], axis=1
        ).astype(np.uint8)
        union = np.sum(
            signatures[idx_i, position] | signatures[idx_j, position], axis=1
        )
        jaccard = shared / union
        group_matches_matrix = (
            group_codes[idx_i, position] == group_codes[idx_j, position]
        )
        matching_groups = group_matches_matrix.sum(axis=1).astype(np.uint8)
        signature_strings = ["".join(map(str, row)) for row in signatures[valid_indices, position]]
        aa_counts = Counter(sequences[valid_indices, position])
        signature_counts = Counter(signature_strings)

        row = {
            "KLIFS_position": position + 1,
            "number_non_gap_kinases": len(valid_indices),
            "number_valid_kinase_pairs": len(local_i),
            "number_unique_amino_acids": len(aa_counts),
            "most_common_amino_acid_frequency": max(aa_counts.values()),
            "amino_acid_identity_rate": np.mean(same_aa),
            "number_unique_PSM_signatures": len(signature_counts),
            "most_common_PSM_signature_frequency": max(signature_counts.values()),
            "fraction_pairs_identical_full_PSM_signature": np.mean(shared == 5),
            "mean_active_bit_PSM_jaccard": np.mean(jaccard),
            "median_active_bit_PSM_jaccard": np.median(jaccard),
            "P5_active_bit_PSM_jaccard": np.quantile(jaccard, 0.05),
            "P95_active_bit_PSM_jaccard": np.quantile(jaccard, 0.95),
        }
        for threshold in range(1, 6):
            row[f"fraction_pairs_sharing_ge{threshold}_active_properties"] = np.mean(shared >= threshold)
        for group_index, (group, _) in enumerate(GROUPS):
            row[f"fraction_pairs_same_{group}_state"] = np.mean(group_matches_matrix[:, group_index])
        for threshold in range(1, 6):
            name = f"fraction_pairs_matching_ge{threshold}_groups" if threshold < 5 else "fraction_pairs_matching_all_5_groups"
            row[name] = np.mean(matching_groups >= threshold)
        position_rows.append(row)

        different = ~same_aa
        diff_jaccard = jaccard[different]
        diff_shared = shared[different]
        diff_groups = matching_groups[different]
        diff_group_matrix = group_matches_matrix[different]
        diff_row = {
            "KLIFS_position": position + 1,
            **{k: v for k, v in summarize_residue_class(
                "different_amino_acid", diff_jaccard, diff_shared, diff_groups
            ).items() if k != "AA_pair_class"},
        }
        for group_index, (group, _) in enumerate(GROUPS):
            diff_row[f"probability_same_{group}_state"] = np.mean(diff_group_matrix[:, group_index])
        different_rows.append(diff_row)

        same_jaccard_parts.append(jaccard[same_aa])
        different_jaccard_parts.append(diff_jaccard)
        same_shared_parts.append(shared[same_aa])
        different_shared_parts.append(diff_shared)
        same_group_parts.append(matching_groups[same_aa])
        different_group_parts.append(diff_groups)
        different_group_specific_counts += diff_group_matrix.sum(axis=0)
        total_different += int(different.sum())

    position_df = pd.DataFrame(position_rows)
    position_df.to_csv(OUTPUT_DIR / "02_position_level_psm_sharing.csv", index=False)
    different_by_position = pd.DataFrame(different_rows)
    different_by_position.to_csv(
        OUTPUT_DIR / "03_nonidentical_AA_psm_sharing_by_position.csv", index=False
    )

    same_jaccard = np.concatenate(same_jaccard_parts)
    different_jaccard = np.concatenate(different_jaccard_parts)
    same_shared = np.concatenate(same_shared_parts)
    different_shared = np.concatenate(different_shared_parts)
    same_groups = np.concatenate(same_group_parts)
    different_groups = np.concatenate(different_group_parts)

    different_global = summarize_residue_class(
        "different_amino_acid", different_jaccard, different_shared, different_groups
    )
    for group_index, (group, _) in enumerate(GROUPS):
        different_global[f"probability_same_{group}_state"] = (
            different_group_specific_counts[group_index] / total_different
        )
    pd.DataFrame([different_global]).to_csv(
        OUTPUT_DIR / "03_nonidentical_AA_psm_sharing_global.csv", index=False
    )
    distribution = pd.DataFrame({
        "number_matching_PSM_groups": np.arange(6),
        "count": np.bincount(different_groups, minlength=6),
    })
    distribution["fraction"] = distribution["count"] / len(different_groups)
    distribution.to_csv(
        OUTPUT_DIR / "03_nonidentical_AA_matching_group_distribution.csv", index=False
    )

    same_summary = summarize_residue_class(
        "same_amino_acid", same_jaccard, same_shared, same_groups
    )
    # Step 6 compares common metrics only; individual group-state rates are
    # reported in the dedicated different-AA outputs from Step 5.
    comparison = pd.DataFrame([
        same_summary,
        {key: different_global[key] for key in same_summary},
    ])
    comparison.to_csv(OUTPUT_DIR / "04_same_vs_different_AA_psm_summary.csv", index=False)
    return position_df, different_by_position, different_global, distribution, comparison


def identity_correlation_analysis(pair_df):
    outcomes = [
        "global_PSM_jaccard",
        "mean_position_PSM_jaccard",
        "mean_matching_PSM_groups",
        "fraction_positions_match_ge3_groups",
        "fraction_positions_match_5_groups",
    ]
    rows = []
    for outcome in outcomes:
        valid = pair_df[["AA_identity", outcome]].dropna()
        pearson = stats.pearsonr(valid["AA_identity"], valid[outcome])
        spearman = stats.spearmanr(valid["AA_identity"], valid[outcome])
        rows.extend([
            {"outcome": outcome, "method": "Pearson", "n_pairs": len(valid),
             "correlation": pearson.statistic, "p_value": pearson.pvalue},
            {"outcome": outcome, "method": "Spearman", "n_pairs": len(valid),
             "correlation": spearman.statistic, "p_value": spearman.pvalue},
        ])
    correlations = pd.DataFrame(rows)
    correlations.to_csv(OUTPUT_DIR / "05_AA_identity_vs_PSM_correlations.csv", index=False)

    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    pair_df = pair_df.copy()
    pair_df["AA_identity_bin"] = pd.cut(
        pair_df["AA_identity"], bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
        labels=labels, right=False,
    )
    rows = []
    for label in labels:
        subset = pair_df[pair_df["AA_identity_bin"] == label]
        rows.append({
            "AA_identity_bin": label,
            "number_kinase_pairs": len(subset),
            "median_global_PSM_jaccard": subset["global_PSM_jaccard"].median(),
            "P25_global_PSM_jaccard": subset["global_PSM_jaccard"].quantile(0.25),
            "P75_global_PSM_jaccard": subset["global_PSM_jaccard"].quantile(0.75),
            "median_matching_PSM_groups": subset["mean_matching_PSM_groups"].median(),
            "median_fraction_positions_match_ge3_groups": subset["fraction_positions_match_ge3_groups"].median(),
            "median_fraction_positions_match_5_groups": subset["fraction_positions_match_5_groups"].median(),
        })
    bin_summary = pd.DataFrame(rows)
    bin_summary.to_csv(OUTPUT_DIR / "05_AA_identity_bin_PSM_summary.csv", index=False)
    return correlations, bin_summary


def conservation_analysis(position_df, different_by_position):
    selected = different_by_position[[
        "KLIFS_position", "mean_PSM_jaccard", "median_PSM_jaccard",
        "probability_matching_ge1_PSM_groups", "probability_matching_ge3_PSM_groups",
        "probability_matching_all_5_PSM_groups", "mean_matching_PSM_groups",
    ]].copy()
    selected = selected.rename(columns={
        "probability_matching_ge1_PSM_groups": "fraction_different_AA_pairs_matching_ge1_PSM_group",
        "probability_matching_ge3_PSM_groups": "fraction_different_AA_pairs_matching_ge3_PSM_groups",
        "probability_matching_all_5_PSM_groups": "fraction_different_AA_pairs_matching_all_5_PSM_groups",
        "mean_matching_PSM_groups": "mean_matching_groups_different_AA_pairs",
    })
    selected.insert(
        1, "AA_pair_identity_fraction",
        position_df.set_index("KLIFS_position").loc[selected["KLIFS_position"], "amino_acid_identity_rate"].to_numpy(),
    )
    selected["low_identity_high_physicochemical_sharing_score"] = (
        (1 - selected["AA_pair_identity_fraction"])
        * selected["mean_matching_groups_different_AA_pairs"] / 5
    )
    selected["rank"] = selected[
        "low_identity_high_physicochemical_sharing_score"
    ].rank(method="min", ascending=False).astype(int)
    selected = selected.sort_values("rank")
    selected.to_csv(
        OUTPUT_DIR / "06_position_AA_vs_physicochemical_conservation.csv", index=False
    )
    return selected


def state_reuse_analysis(signatures, sequences, group_codes):
    position_rows, group_rows = [], []
    for position in range(N_POSITIONS):
        valid = sequences[:, position] != "-"
        aa_values = sequences[valid, position]
        sig_values = np.array([
            "".join(map(str, row)) for row in signatures[valid, position]
        ])
        aa_counts = Counter(aa_values)
        sig_counts = Counter(sig_values)
        position_rows.append({
            "KLIFS_position": position + 1,
            "number_valid_kinases": int(valid.sum()),
            "number_unique_amino_acids": len(aa_counts),
            "largest_kinase_count_sharing_one_amino_acid": max(aa_counts.values()),
            "fraction_kinases_whose_amino_acid_is_shared": np.mean([aa_counts[x] >= 2 for x in aa_values]),
            "number_unique_PSM_signatures": len(sig_counts),
            "unique_signature_fraction": len(sig_counts) / valid.sum(),
            "largest_kinase_count_sharing_one_PSM_signature": max(sig_counts.values()),
            "median_kinases_per_observed_PSM_signature": np.median(list(sig_counts.values())),
            "fraction_kinases_whose_full_PSM_signature_is_shared": np.mean([sig_counts[x] >= 2 for x in sig_values]),
        })
        for group_index, (group, states) in enumerate(GROUPS):
            values = group_codes[valid, position, group_index]
            counts = Counter(values)
            group_rows.append({
                "KLIFS_position": position + 1,
                "PSM_property_group": group,
                "number_unique_states": len(counts),
                "dominant_state": states[counts.most_common(1)[0][0]],
                "dominant_state_frequency": counts.most_common(1)[0][1],
                "dominant_state_fraction": counts.most_common(1)[0][1] / len(values),
                "fraction_kinases_sharing_group_state": np.mean([counts[x] >= 2 for x in values]),
            })
    position_reuse = pd.DataFrame(position_rows)
    group_reuse = pd.DataFrame(group_rows)
    position_reuse.to_csv(OUTPUT_DIR / "07_position_state_reuse.csv", index=False)
    group_reuse.to_csv(OUTPUT_DIR / "07_property_group_state_reuse.csv", index=False)

    global_rows = [
        {"metric": "median_unique_amino_acids_per_position", "value": position_reuse["number_unique_amino_acids"].median()},
        {"metric": "median_unique_PSM_signatures_per_position", "value": position_reuse["number_unique_PSM_signatures"].median()},
        {"metric": "median_fraction_kinases_sharing_complete_PSM_state", "value": position_reuse["fraction_kinases_whose_full_PSM_signature_is_shared"].median()},
    ]
    for group, _ in GROUPS:
        subset = group_reuse[group_reuse["PSM_property_group"] == group]
        global_rows.append({
            "metric": f"median_fraction_kinases_sharing_{group}_state",
            "value": subset["fraction_kinases_sharing_group_state"].median(),
        })
    pd.DataFrame(global_rows).to_csv(
        OUTPUT_DIR / "07_state_reuse_global_summary.csv", index=False
    )
    return position_reuse, group_reuse, pd.DataFrame(global_rows)


def save_figure(fig, stem):
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def create_figures(pair_df, position_df, different_by_position, different_global,
                   distribution, position_reuse):
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120, "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.hist(pair_df["global_PSM_jaccard"], bins=40, color="#3A78A1", edgecolor="white")
    ax.set(xlabel="Global active-bit PSM Jaccard similarity", ylabel="Kinase-pair count")
    save_figure(fig, "global_PSM_jaccard_distribution")

    for y, stem, ylabel in [
        ("global_PSM_jaccard", "AA_identity_vs_PSM_jaccard", "Global PSM Jaccard similarity"),
        ("mean_matching_PSM_groups", "AA_identity_vs_matching_PSM_groups", "Mean matching PSM groups per position"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 4.7))
        hb = ax.hexbin(pair_df["AA_identity"], pair_df[y], gridsize=55, bins="log", mincnt=1, cmap="viridis")
        fig.colorbar(hb, ax=ax, label="log10(pair count + 1)")
        ax.set(xlabel="Aligned pocket amino-acid identity", ylabel=ylabel)
        save_figure(fig, stem)

    # Required third AA-identity scatter plot.
    fig, ax = plt.subplots(figsize=(6.2, 4.7))
    hb = ax.hexbin(pair_df["AA_identity"], pair_df["mean_position_PSM_jaccard"],
                   gridsize=55, bins="log", mincnt=1, cmap="viridis")
    fig.colorbar(hb, ax=ax, label="log10(pair count + 1)")
    ax.set(xlabel="Aligned pocket amino-acid identity", ylabel="Mean position-aware PSM Jaccard")
    save_figure(fig, "AA_identity_vs_position_PSM_jaccard")

    fig, ax = plt.subplots(figsize=(6.2, 4.7))
    scatter = ax.scatter(
        position_df["amino_acid_identity_rate"],
        position_df["mean_active_bit_PSM_jaccard"],
        c=position_df["KLIFS_position"], cmap="viridis", s=34,
    )
    fig.colorbar(scatter, ax=ax, label="KLIFS position")
    ax.set(xlabel="AA pair identity fraction", ylabel="Mean active-bit PSM Jaccard")
    save_figure(fig, "position_AA_identity_vs_PSM_similarity")

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.bar(distribution["number_matching_PSM_groups"], distribution["fraction"], color="#497AA7")
    ax.set(xlabel="Matching groups (different-AA pairs)", ylabel="Fraction of comparisons", xticks=range(6))
    save_figure(fig, "nonidentical_AA_matching_group_distribution")

    groups = [g for g, _ in GROUPS]
    rates = [different_global[f"probability_same_{g}_state"] for g in groups]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(range(len(groups)), rates, color="#4F9D73")
    ax.set_xticks(range(len(groups)), [g.replace("_", "\n") for g in groups])
    ax.set(ylabel="P(same group state | different amino acids)", ylim=(0, 1))
    save_figure(fig, "nonidentical_AA_property_group_sharing")

    threshold_columns = [
        "probability_matching_ge1_PSM_groups", "probability_matching_ge2_PSM_groups",
        "probability_matching_ge3_PSM_groups", "probability_matching_ge4_PSM_groups",
        "probability_matching_all_5_PSM_groups",
    ]
    heat = different_by_position.set_index("KLIFS_position")[threshold_columns].T
    fig, ax = plt.subplots(figsize=(13, 3.6))
    im = ax.imshow(heat, aspect="auto", cmap="magma", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Fraction of different-AA pairs")
    ax.set(xlabel="KLIFS position", yticks=range(5), yticklabels=[">=1", ">=2", ">=3", ">=4", "all 5"])
    ax.set_xticks(np.arange(0, 85, 4), np.arange(1, 86, 4))
    save_figure(fig, "nonidentical_AA_physicochemical_sharing_by_KLIFS_position")

    group_columns = [f"fraction_pairs_same_{g}_state" for g in groups]
    heat = position_df.set_index("KLIFS_position")[group_columns].T
    fig, ax = plt.subplots(figsize=(13, 3.6))
    im = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Fraction of valid kinase pairs")
    ax.set(xlabel="KLIFS position", yticks=range(5), yticklabels=[g.replace("_", " ") for g in groups])
    ax.set_xticks(np.arange(0, 85, 4), np.arange(1, 86, 4))
    save_figure(fig, "PSM_property_group_sharing_heatmap")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(position_reuse["KLIFS_position"], position_reuse["fraction_kinases_whose_full_PSM_signature_is_shared"],
            color="#315B7D", lw=1.5, label="Full 23-bit PSM state")
    ax.plot(position_reuse["KLIFS_position"], position_reuse["fraction_kinases_whose_amino_acid_is_shared"],
            color="#C46345", lw=1.2, label="Amino acid")
    ax.set(xlabel="KLIFS position", ylabel="Fraction of kinases with a reused state", ylim=(0, 1.02))
    ax.legend(frameon=False)
    save_figure(fig, "position_state_reuse")






def main():
    sequence_df, bits, reconstructed, mapping, diagnostics = validate_inputs()
    _, signatures = reconstruct_position_table(sequence_df, reconstructed, mapping)
    sequences = np.array([list(seq) for seq in sequence_df["KLIFS"]])
    group_codes = group_state_codes(signatures, sequences)

    pair_df = pairwise_kinase_analysis(
        sequence_df, bits, signatures, sequences, group_codes
    )
    position_df, different_by_position, different_global, distribution, _ = (
        residue_and_position_analysis(signatures, sequences, group_codes)
    )
    correlations, _ = identity_correlation_analysis(pair_df)
    conservation = conservation_analysis(position_df, different_by_position)
    position_reuse, _, _ = state_reuse_analysis(signatures, sequences, group_codes)
    create_figures(
        pair_df, position_df, different_by_position, different_global,
        distribution, position_reuse,
    )
    print(f"Analysis completed successfully. Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
