"""Check release inputs, or copy existing local files into the release layout."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import tarfile

CODE_DIR = Path(__file__).resolve().parent


def prepare(root: Path, *, copy_existing: bool = False) -> int:
    root = root.expanduser().resolve()
    with (CODE_DIR / "release_input_manifest.csv").open(newline="") as handle:
        entries = list(csv.DictReader(handle))
    problems = []
    copied = 0
    for entry in entries:
        target = root / entry["repository_path"]
        source = root / entry["original_repository_path"]
        if copy_existing and not target.exists() and source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
        if not target.is_file():
            problems.append(f"Missing: {entry['repository_path']}")
        elif target.stat().st_size != int(entry["bytes"]):
            problems.append(f"Size mismatch: {entry['repository_path']} (expected {entry['bytes']} bytes)")
    print(f"Release inputs: {len(entries) - len(problems)}/{len(entries)} available with expected sizes")
    if copied:
        print(f"Copied {copied} files; original files were retained.")
    for problem in problems:
        print(problem)
    return int(bool(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinspec-dir", type=Path, default=CODE_DIR.parents[1])
    parser.add_argument("--copy-existing", action="store_true",
                        help="Copy missing inputs from the previous repository layout; no downloads or overwrites")
    parser.add_argument("--pack-additional", type=Path,
                        help="Create a gzip tar archive of the nine additional CSVs for the data release")
    args = parser.parse_args()
    result = prepare(args.kinspec_dir, copy_existing=args.copy_existing)
    if result or args.pack_additional is None:
        return result
    with (CODE_DIR / "release_input_manifest.csv").open(newline="") as handle:
        extra = [entry for entry in csv.DictReader(handle)
                 if entry["repository_path"] != entry["original_repository_path"]]
    with tarfile.open(args.pack_additional, "x:gz") as archive:
        for entry in extra:
            source = args.kinspec_dir.expanduser().resolve() / entry["repository_path"]
            archive.add(source, arcname=source.name)
    print(f"Packed {len(extra)} additional CSVs: {args.pack_additional}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
