#!/usr/bin/env python3
"""Evaluate kinase rankings based only on band 10-50 training activity rates.

For each training kinase, activity rate is defined as
``sum(unified_bio_label == 1) / number of training rows``. Training KLIFS
sequences are mapped to UniProt IDs with the project's kinase reference table.
The ranking is restricted to kinases present in Karaman; Karaman-only kinases
are assigned a score below every observed training rate and placed last.

Metrics follow the per-molecule implementation in
``load_model_KinSPEC_karaman_selectivity_test_MCA.ipynb``: AUROC, BEDROC with
alpha=20, precision@10, and recall@10. Overall aggregation also follows that
notebook by excluding molecules with no actives or >=94 active kinases.
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
_kp = _ks_partial(_ks_project_path, script_dir='others/Prior_conflict_pair_analysis/KinSPEC_LTS_v4')


import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from sklearn.metrics import roc_auc_score


DEFAULT_TRAIN_CSV = Path(
    _kp('dataset/spectrum_data/band_0.10_0.50/train_with_smiles_protein_seq.csv')
)
DEFAULT_KARAMAN_CSV = Path(
    _kp('dataset/Karaman_10_50_filtered_final.csv')
)
DEFAULT_KLIFS_REFERENCE = Path(
    _kp('feature_embedding/Kinase_test_MCA/human_482_typical_kinase_1955bit.csv')
)
DEFAULT_OUTPUT_DIR = Path(
    _kp('others/Prior_conflict_pair_analysis/KinSPEC_LTS_v4/training_prior_results')
)
BEDROC_ALPHA = 20.0
TOP_K = 10
EXCLUDE_ACTIVE_AT_LEAST = 94
MISSING_TARGET_SCORE = -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--karaman-csv", type=Path, default=DEFAULT_KARAMAN_CSV)
    parser.add_argument(
        "--klifs-reference", type=Path, default=DEFAULT_KLIFS_REFERENCE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def bedroc(y_true: np.ndarray, y_score: np.ndarray, alpha: float) -> float:
    order = np.argsort(-y_score, kind="stable")
    sorted_labels = y_true[order]
    scored = [[1.0, int(label)] for label in sorted_labels]
    return float(CalcBEDROC(scored, col=1, alpha=alpha))


def calculate_training_rates(
    train: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.to_numeric(train["unified_bio_label"], errors="raise")
    if not set(labels.unique()) <= {0, 1}:
        raise ValueError("Training unified_bio_label contains values other than 0/1")
    train = train.assign(unified_bio_label=labels.astype(np.int8))

    by_klifs = (
        train.groupby("KLIFS", sort=False)["unified_bio_label"]
        .agg(train_total_rows="size", train_active_rows="sum")
        .reset_index()
    )
    by_klifs["training_activity_rate"] = (
        by_klifs["train_active_rows"] / by_klifs["train_total_rows"]
    )

    mapping = reference[["KLIFS", "UniProt_ID", "kinase_name"]].drop_duplicates()
    mapped = by_klifs.merge(mapping, on="KLIFS", how="left", validate="one_to_many")
    if mapped["UniProt_ID"].isna().any():
        examples = mapped.loc[mapped["UniProt_ID"].isna(), "KLIFS"].head(3).tolist()
        raise ValueError(f"Training KLIFS values absent from reference; examples: {examples}")

    # One KLIFS sequence can map to multiple UniProt IDs. Conversely, guard
    # against inconsistent rates if a UniProt ID ever maps from multiple rows.
    inconsistent = mapped.groupby("UniProt_ID")["training_activity_rate"].nunique()
    if (inconsistent > 1).any():
        bad = inconsistent[inconsistent > 1].index.tolist()[:5]
        raise ValueError(f"UniProt IDs map to inconsistent activity rates: {bad}")
    by_uniprot = (
        mapped.sort_values(["training_activity_rate", "UniProt_ID"], ascending=[False, True])
        .drop_duplicates("UniProt_ID")
        .reset_index(drop=True)
    )
    return by_klifs, by_uniprot


def evaluate_molecules(scored_karaman: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    ranked_tables: list[pd.DataFrame] = []

    for molecule_name, sub in scored_karaman.groupby("molecule_name", sort=False):
        sub = sub.copy()
        y_true = pd.to_numeric(sub["label"], errors="raise").to_numpy(dtype=int)
        y_score = sub["ranking_score"].to_numpy(dtype=float)
        if not set(np.unique(y_true).tolist()) <= {0, 1}:
            raise ValueError(f"Karaman label is not binary for {molecule_name}")

        order = np.argsort(-y_score, kind="stable")
        ranked = sub.iloc[order].copy()
        ranked.insert(0, "kinase_rank", np.arange(1, len(ranked) + 1))
        ranked_tables.append(ranked)

        n_total = len(sub)
        n_active = int(y_true.sum())
        result = {
            "molecule_name": molecule_name,
            "n_kinases": n_total,
            "n_active": n_active,
            "n_targets_found_in_training": int(sub["found_in_training"].sum()),
            "n_targets_missing_from_training": int((~sub["found_in_training"]).sum()),
        }
        if n_active == 0 or n_active == n_total:
            result.update(
                {
                    "AUROC": np.nan,
                    f"BEDROC_alpha{int(BEDROC_ALPHA)}": np.nan,
                    f"precision@{TOP_K}": np.nan,
                    f"recall@{TOP_K}": np.nan,
                }
            )
        else:
            sorted_labels = y_true[order]
            active_top_k = int(sorted_labels[: min(TOP_K, n_total)].sum())
            result.update(
                {
                    "AUROC": float(roc_auc_score(y_true, y_score)),
                    f"BEDROC_alpha{int(BEDROC_ALPHA)}": bedroc(
                        y_true, y_score, BEDROC_ALPHA
                    ),
                    f"precision@{TOP_K}": float(active_top_k / min(TOP_K, n_total)),
                    f"recall@{TOP_K}": float(active_top_k / n_active),
                }
            )
        result["included_in_notebook_style_overall"] = bool(
            0 < n_active < EXCLUDE_ACTIVE_AT_LEAST
        )
        metric_rows.append(result)

    return pd.DataFrame(metric_rows), pd.concat(ranked_tables, ignore_index=True)


def summarize_metrics(per_molecule: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "AUROC",
        f"BEDROC_alpha{int(BEDROC_ALPHA)}",
        f"precision@{TOP_K}",
        f"recall@{TOP_K}",
    ]
    rows = []
    masks = {
        "notebook_style_kept_molecules": per_molecule[
            "included_in_notebook_style_overall"
        ],
        "all_molecules_with_defined_metrics": per_molecule["AUROC"].notna(),
    }
    for aggregation, mask in masks.items():
        data = per_molecule.loc[mask, metric_columns]
        for metric in metric_columns:
            values = data[metric].dropna()
            rows.append(
                {
                    "aggregation": aggregation,
                    "metric": metric,
                    "n_molecules": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                }
            )
    summary = pd.DataFrame(rows)
    summary["iqr"] = summary["q75"] - summary["q25"]
    return summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(
        args.train_csv, usecols=["KLIFS", "unified_bio_label"], dtype={"KLIFS": str}
    )
    karaman = pd.read_csv(args.karaman_csv, dtype={"UniProt_ID": str})
    reference = pd.read_csv(
        args.klifs_reference, dtype={"KLIFS": str, "UniProt_ID": str}
    )
    require_columns(train, {"KLIFS", "unified_bio_label"}, args.train_csv)
    require_columns(
        karaman,
        {"molecule_name", "protein_name", "UniProt_ID", "label"},
        args.karaman_csv,
    )
    require_columns(
        reference, {"KLIFS", "UniProt_ID", "kinase_name"}, args.klifs_reference
    )

    if karaman.duplicated(["molecule_name", "UniProt_ID"]).any():
        raise ValueError("Karaman contains duplicate molecule_name/UniProt_ID pairs")

    by_klifs, by_uniprot = calculate_training_rates(train, reference)
    rate_columns = [
        "UniProt_ID",
        "kinase_name",
        "KLIFS",
        "train_total_rows",
        "train_active_rows",
        "training_activity_rate",
    ]
    by_uniprot.loc[:, rate_columns].to_csv(
        args.output_dir / "band_10_50_training_target_activity_rates.csv", index=False
    )
    by_klifs.to_csv(
        args.output_dir / "band_10_50_training_KLIFS_activity_rates.csv", index=False
    )

    # Keep only Karaman targets; left join adds Karaman-only targets. Scores of
    # -1 put missing targets below every valid activity rate (all are >= 0).
    rates = by_uniprot[
        [
            "UniProt_ID",
            "train_total_rows",
            "train_active_rows",
            "training_activity_rate",
        ]
    ]
    scored = karaman.merge(rates, on="UniProt_ID", how="left", validate="many_to_one")
    scored["found_in_training"] = scored["training_activity_rate"].notna()
    scored["ranking_score"] = scored["training_activity_rate"].fillna(
        MISSING_TARGET_SCORE
    )

    target_ranking = (
        scored[
            [
                "protein_name",
                "UniProt_ID",
                "found_in_training",
                "train_total_rows",
                "train_active_rows",
                "training_activity_rate",
                "ranking_score",
            ]
        ]
        .drop_duplicates("UniProt_ID")
        .sort_values("ranking_score", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    target_ranking.insert(0, "kinase_rank", np.arange(1, len(target_ranking) + 1))
    target_ranking.to_csv(
        args.output_dir / "karaman_targets_ranked_by_training_activity_rate.csv",
        index=False,
    )

    per_molecule, ranked_long = evaluate_molecules(scored)
    per_molecule.to_csv(
        args.output_dir / "karaman_activity_rate_per_molecule_metrics.csv", index=False
    )
    ranked_long.to_csv(
        args.output_dir / "karaman_activity_rate_ranked_long.csv", index=False
    )
    summary = summarize_metrics(per_molecule)
    summary.to_csv(
        args.output_dir / "karaman_activity_rate_overall_summary.csv", index=False
    )

    train_ids = set(by_uniprot["UniProt_ID"])
    karaman_ids = set(karaman["UniProt_ID"])
    metadata = {
        "train_csv": str(args.train_csv),
        "karaman_csv": str(args.karaman_csv),
        "klifs_reference": str(args.klifs_reference),
        "training_activity_rate": "train_active_rows / train_total_rows per KLIFS target",
        "karaman_label": "label",
        "bedroc_alpha": BEDROC_ALPHA,
        "top_k": TOP_K,
        "tie_handling": "stable sort preserving Karaman source order",
        "karaman_only_target_score": MISSING_TARGET_SCORE,
        "n_training_klifs": int(by_klifs["KLIFS"].nunique()),
        "n_training_uniprot_ids_after_mapping": int(by_uniprot["UniProt_ID"].nunique()),
        "n_karaman_molecules": int(karaman["molecule_name"].nunique()),
        "n_karaman_targets": int(karaman["UniProt_ID"].nunique()),
        "n_overlapping_targets": len(train_ids & karaman_ids),
        "karaman_only_targets": sorted(karaman_ids - train_ids),
        "n_training_only_targets_removed": len(train_ids - karaman_ids),
        "overall_aggregation": (
            "mean/median/std/quantiles across per-molecule metrics; notebook-style "
            "summary keeps molecules with 1-93 active kinases"
        ),
    }
    (args.output_dir / "karaman_activity_rate_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )

    kept = summary[summary["aggregation"].eq("notebook_style_kept_molecules")]
    print(
        f"Karaman: {karaman['molecule_name'].nunique()} molecules x "
        f"{karaman['UniProt_ID'].nunique()} kinases"
    )
    print(
        f"Target overlap: {len(train_ids & karaman_ids)}; "
        f"Karaman-only: {sorted(karaman_ids - train_ids)}; "
        f"training-only removed: {len(train_ids - karaman_ids)}"
    )
    print("\nNotebook-style overall summary (kept molecules only):")
    print(kept[["metric", "n_molecules", "mean", "median", "std"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
