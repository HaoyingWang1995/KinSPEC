#!/usr/bin/env python3
"""Download the KinSPEC dataset from Hugging Face (https://huggingface.co/datasets/Lumineb/KinSPEC).

Files are saved next to this script (the project's `dataset/` directory) by
default, so they end up where the KinSPEC notebooks expect them.

The cross-validation archive data_split_from_merged_true_fold.tar.gz is unpacked
after download into dataset/data_split_from_merged_true_fold/. Use --no-extract
to skip this, or --remove-archives to delete the archive after unpacking.

Usage:
    python download_dataset_hf.py                  # download everything (about 70 GB)
    python download_dataset_hf.py --list           # list available files and sizes
    python download_dataset_hf.py --files VSDS_truedecoy_integrated_data_with_protein_seq.csv
    python download_dataset_hf.py --include "train_band_10_50_*"
    python download_dataset_hf.py --out /path/to/dir

Downloads are resumable: re-running the script skips files that are already
complete and continues interrupted ones.

Requirements: pip install -U huggingface_hub
"""
import argparse
import fnmatch
import sys
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# ===== Configuration =====
HF_TOKEN = ""  # Hugging Face access token (read-only is sufficient)
# =========================

REPO_ID = "Lumineb/KinSPEC"
REPO_TYPE = "dataset"
DEFAULT_DIR = Path(__file__).resolve().parent
SKIP = {"README.md", ".gitattributes"}   # repository metadata, not dataset files
EXTRACT = "data_split_from_merged_true_fold.tar.gz"                    # archives unpacked after download


def remote_files(api):
    """Return {filename: size_in_bytes} for the files in the repository."""
    info = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=True)
    return {s.rfilename: s.size or 0 for s in info.siblings if s.rfilename not in SKIP}


def select(available, files, patterns):
    if not files and not patterns:
        return sorted(available)
    unknown = [f for f in files if f not in available]
    if unknown:
        sys.exit(f"Not found in {REPO_ID}: {', '.join(unknown)}. Use --list to see available files.")
    chosen = set(files)
    for pattern in patterns:
        matched = fnmatch.filter(available, pattern)
        if not matched:
            sys.exit(f"No file matches pattern: {pattern!r}")
        chosen.update(matched)
    return sorted(chosen)


def extract(archive, out_dir):
    print(f"Extracting {archive.name} ...", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):   # Python >= 3.12 / patched 3.8-3.11
            tar.extractall(out_dir, filter="data")
        else:
            tar.extractall(out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="list available files and exit")
    parser.add_argument("--files", nargs="+", default=[], metavar="NAME", help="exact file names to download")
    parser.add_argument("--include", nargs="+", default=[], metavar="PATTERN", help="glob patterns, e.g. '*.tar.gz'")
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR, help=f"target directory (default: {DEFAULT_DIR})")
    parser.add_argument("--no-extract", action="store_true", help="do not unpack downloaded archives")
    parser.add_argument("--remove-archives", action="store_true", help="delete archives after unpacking")
    args = parser.parse_args()

    token = HF_TOKEN.strip() or None
    api = HfApi(token=token)
    try:
        available = remote_files(api)
    except Exception as exc:
        sys.exit(f"Cannot access {REPO_ID} ({REPO_TYPE}): {exc}\n"
                 "Check that HF_TOKEN is set and has read access to the repository.")

    if args.list:
        for name in sorted(available):
            print(f"{available[name] / 1e6:10.1f} MB  {name}")
        print(f"{sum(available.values()) / 1e9:10.2f} GB  total ({len(available)} files)")
        return

    names = select(available, args.files, args.include)
    total = sum(available[n] for n in names)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(names)} file(s), {total / 1e9:.2f} GB, from {REPO_ID} to {out}", flush=True)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=out,
        allow_patterns=names,
        token=token,
    )

    for name in names:
        size = (out / name).stat().st_size
        status = "OK" if size == available[name] else f"SIZE MISMATCH (expected {available[name]})"
        print(f"  {status:4s} {name} ({size / 1e6:.1f} MB)")

    if not args.no_extract:
        for name in names:
            if fnmatch.fnmatch(name, EXTRACT):
                extract(out / name, out)
                if args.remove_archives:
                    (out / name).unlink()
    print("Done.")


if __name__ == "__main__":
    main()
