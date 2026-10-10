#!/usr/bin/env python3
"""Build compact, deduplicated backgrounds for KinSPEC interpretation.

The merged training CSV already stores each 4096-bit Morgan fingerprint as a
compact binary string.  Reading that column avoids materialising the 5.3 GB
dense one-hot CSV.  The dense file is sampled to verify row correspondence.
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
import csv
import hashlib
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGED = ROOT / _kp('dataset/train_ready2use_merged_4datasets_deduplicated_with_smiles.csv')
DEFAULT_MORGAN = ROOT / _kp('dataset/KinSPEC_training_pool_morgan_one_hot.csv')
DEFAULT_KINASE = ROOT / _kp('feature_embedding/Kinase_test_MCA/human_482_typical_kinase_1955bit.csv')
DEFAULT_MCA = ROOT / _kp('feature_embedding/Kinase_test_MCA/MCA_klifs_482_200D_features.csv')
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "background_data"


def fingerprint_digest(bits: str) -> bytes:
    return hashlib.blake2b(bits.encode("ascii"), digest_size=16).digest()


def verify_dense_prefix(merged_path: Path, dense_path: Path, n_rows: int) -> None:
    merged = pd.read_csv(merged_path, usecols=["Morgan"], nrows=n_rows)
    dense = pd.read_csv(dense_path, header=None, nrows=n_rows, dtype=np.uint8)
    if dense.shape != (len(merged), 4096):
        raise ValueError(f"Dense Morgan prefix has shape {dense.shape}; expected ({len(merged)}, 4096)")
    compact = ["".join(row.astype(str)) for row in dense.to_numpy()]
    if compact != merged["Morgan"].tolist():
        raise ValueError("Dense Morgan rows are not aligned with the merged training CSV")


def build_backgrounds(merged_path: Path, kinase_path: Path, mca_path: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    mol_final = outdir / _kp('dataset/training_unique_morgan_fingerprints.csv')
    mol_tmp = outdir / ".training_unique_morgan_fingerprints.csv.tmp"

    seen_digests: set[bytes] = set()
    kinase_counts: dict[str, int] = defaultdict(int)
    n_rows = 0
    with merged_path.open(newline="") as src, mol_tmp.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        required = {"Morgan", "KLIFS"}
        if not required.issubset(reader.fieldnames or []):
            raise KeyError(f"Missing columns {required}; found {reader.fieldnames}")
        writer = csv.writer(dst)
        writer.writerow(["molecule_id", "first_training_row", "Morgan"])
        for row_idx, row in enumerate(reader):
            bits = row["Morgan"]
            if len(bits) != 4096 or set(bits) - {"0", "1"}:
                raise ValueError(f"Invalid Morgan fingerprint at training row {row_idx}")
            digest = fingerprint_digest(bits)
            if digest not in seen_digests:
                seen_digests.add(digest)
                writer.writerow([len(seen_digests) - 1, row_idx, bits])
            kinase_counts[row["KLIFS"]] += 1
            n_rows += 1
    os.replace(mol_tmp, mol_final)

    kinase = pd.read_csv(kinase_path)
    mca = pd.read_csv(mca_path)
    if len(kinase) != len(mca):
        raise ValueError(f"Kinase metadata rows ({len(kinase)}) != MCA rows ({len(mca)})")

    records = []
    unmatched = sorted(set(kinase_counts) - set(kinase["KLIFS"]))
    if unmatched:
        raise ValueError(f"{len(unmatched)} training KLIFS sequences are absent from the 482-kinase table")
    for training_id, sequence in enumerate(sorted(kinase_counts)):
        hits = kinase.index[kinase["KLIFS"].eq(sequence)].tolist()
        first = hits[0]
        if len(hits) > 1:
            reference = mca.iloc[first].to_numpy(float)
            for hit in hits[1:]:
                np.testing.assert_allclose(mca.iloc[hit].to_numpy(float), reference, rtol=0, atol=1e-12)
        rec = {
            "training_kinase_id": training_id,
            "human_482_rows": "|".join(map(str, hits)),
            "kinase_name": "|".join(kinase.loc[hits, "kinase_name"].astype(str)),
            "UniProt_ID": "|".join(kinase.loc[hits, "UniProt_ID"].astype(str)),
            "KLIFS": sequence,
            "Binary_string": kinase.loc[first, "Binary_string"],
            "training_pair_count": kinase_counts[sequence],
        }
        rec.update(mca.iloc[first].to_dict())
        records.append(rec)
    kinase_out = pd.DataFrame(records)
    kinase_out.to_csv(outdir / "training_unique_kinases_1955bit_MCA200.csv", index=False)

    pd.DataFrame(
        [{
            "training_rows": n_rows,
            "unique_morgan_fingerprints": len(seen_digests),
            "unique_training_kinase_sequences": len(records),
            "human_kinase_rows": len(kinase),
        }]
    ).to_csv(outdir / "background_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--dense-morgan", type=Path, default=DEFAULT_MORGAN)
    parser.add_argument("--kinase", type=Path, default=DEFAULT_KINASE)
    parser.add_argument("--mca", type=Path, default=DEFAULT_MCA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--verify-prefix-rows", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_dense_prefix(args.merged, args.dense_morgan, args.verify_prefix_rows)
    build_backgrounds(args.merged, args.kinase, args.mca, args.outdir)
    print(pd.read_csv(args.outdir / "background_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
