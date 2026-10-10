"""Run every analysis once, with resumable audit records and immutable input snapshots."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from analysis_scripts import workflow as w

# Run the expensive computations once; their downstream consumers share the outputs.
SELECTED = (
    "fullpool", "vsds_eval", "karaman_verify", "karaman", "decomp", "shortcut",
    "guilt", "adversarial", "strat", "geom", "derive_aa23", "mech", "backproj", "ablate",
    "prepare_encoding", "prepare_mca", "mech2_setup", "mech2_retain", "mech2_pairs", "mech2_pairs2",
    "invariance", "pca_vs_mca", "stability", "null_check", "null_audit", "own_space", "imptype",
    "tie_check", "causal", "zero_ambiguity", "make_tables",
    "pool_attr", "shap_pertarget", "pool_analyse", "redund_shap", "zero_shap", "path_reuse", "perkinase2", "perkinase3",
    "dedup", "cross", "control", "stability2",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(paths):
    result = {}
    for path in sorted(set(paths)):
        stat = path.stat()
        result[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest(path)}
    return result


def stage_signature(name, config):
    paths = [w.CODE_DIR / "analysis_scripts" / f"{name}.py", Path(w.__file__)]
    if name in {"pool_attr", "shap_pertarget"}:
        paths.append(w.CODE_DIR / "analysis_scripts/shap_checks.py")
    value = {"config": config.to_dict(), "sources": {str(p): digest(p) for p in paths}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def helper_hashes(name):
    names = ["training_checks"] if w.STAGES[name].retrains else []
    if name in {"control", "stability2"}:
        names.append("pca_features")
    return {str(w.CODE_DIR / "analysis_scripts" / f"{helper}.py"):
            digest(w.CODE_DIR / "analysis_scripts" / f"{helper}.py") for helper in names}


def run_audit(config, resume=False, force_steps=()):
    plan = w.resolve_steps(SELECTED)
    if set(plan) != set(w.STAGES):
        raise ValueError("Audit plan must cover every registered stage")
    w.prepare_inputs(config)
    missing = {name: [w.display_path(p, config) for p in w.required_inputs(name, config) if not p.is_file()] for name in plan}
    if any(missing.values()):
        raise FileNotFoundError(json.dumps({k: v for k, v in missing.items() if v}, indent=2))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logs = config.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    state_path = config.output_dir / "audit_state.json"
    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        if state["config"] != config.to_dict():
            raise ValueError("Resume requires the same data paths, output path and compute settings")
        if any(importlib.metadata.version(package) != version for package, version in state["versions"].items()):
            raise ValueError("Analysis package versions changed since this audit started")
        before = json.loads((config.output_dir / "inputs_before.json").read_text())
        print("Checking input hashes before resuming ...", flush=True)
        current = snapshot({Path(p) for p in before})
        if current != before:
            changed = [p for p in before if before[p] != current[p]]
            raise ValueError(f"Inputs changed since this audit started: {changed}")
    else:
        if state_path.exists():
            raise ValueError("An audit already exists in this output directory; use --resume or another directory")
        raw_inputs = {p for name in plan for p in w.required_inputs(name, config)}
        # Capture the original directory at this run's start, including data and models.
        originals = {p for p in config.project_dir.rglob("*") if p.is_file()}
        print(f"Hashing {len(raw_inputs | originals)} input/original files ...", flush=True)
        start_snapshot = snapshot(raw_inputs | originals)
        (config.output_dir / "inputs_before.json").write_text(json.dumps(start_snapshot, indent=2) + "\n")
        versions = {package: importlib.metadata.version(package) for package in ("xgboost", "numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")}
        state = {"config": config.to_dict(), "plan": plan, "started_utc": datetime.now(timezone.utc).isoformat(),
                 "python": sys.version, "host": platform.node(), "versions": versions,
                 "gpu": gpu_description(),
                 "stages": {}, "status": "running"}

    def save():
        temporary = state_path.with_suffix(".json.new")
        temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(state_path)

    state["status"] = "running"
    save()
    invalidated = set()
    for name in plan:
        signature = stage_signature(name, config)
        previous = state["stages"].get(name, {})
        dependencies = w.STAGES[name].dependencies
        if any(state["stages"].get(dep, {}).get("status") != "completed" for dep in dependencies):
            state["stages"][name] = {"status": "blocked", "reason": "A required dependency failed"}
            save()
            continue
        outputs = previous.get("output_hashes", {})
        can_reuse = resume and name not in force_steps and previous.get("status") == "completed" and previous.get("signature") == signature
        can_reuse = can_reuse and previous.get("helper_hashes", {}) == helper_hashes(name)
        can_reuse = can_reuse and not any(dep in invalidated for dep in dependencies)
        can_reuse = can_reuse and all(Path(p).is_file() and digest(p) == h for p, h in outputs.items())
        can_reuse = can_reuse and all((config.output_dir / p).is_file() for p in w.STAGES[name].outputs)
        if can_reuse:
            print(f"Reusing audited stage {name}", flush=True)
            continue
        invalidated.add(name)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        log_path = logs / f"{stamp}_{name}.log"
        record = {"status": "running", "signature": signature, "helper_hashes": helper_hashes(name), "log": str(log_path), "started_utc": stamp,
                  "thread_environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
        state["stages"][name] = record
        save()
        started = time.monotonic()
        with log_path.open("w") as stream, redirect_stdout(w.Tee(sys.stdout, stream)), redirect_stderr(w.Tee(sys.stderr, stream)):
            try:
                print(f"Starting audit {name}", flush=True)
                w.import_step(name).run(config)
                absent = [p for p in w.STAGES[name].outputs if not (config.output_dir / p).is_file()]
                if absent:
                    raise RuntimeError(f"Missing outputs: {absent}")
                record["output_hashes"] = {str(config.output_dir / p): digest(config.output_dir / p) for p in w.STAGES[name].outputs}
                if name == "vsds_eval":
                    extras = list(config.output_dir.glob("scores_*.npy"))
                elif w.STAGES[name].retrains:
                    extras = list((config.output_dir / "training_models" / name).glob("*"))
                    if name == "control":
                        extras.append(config.output_dir / "control_pca_200.npy")
                    if name == "stability2":
                        extras.extend(config.output_dir / f for f in ("stability2_pca_200.npy", "stability2_pca_components.npy"))
                else:
                    extras = []
                record["output_hashes"].update({str(p): digest(p) for p in extras if p.is_file()})
                record["status"] = "completed"
                print(f"Completed audit {name} ({time.monotonic() - started:.1f}s)", flush=True)
            except Exception as exc:
                record.update(status="failed", error=str(exc))
                traceback.print_exc()
            except BaseException as exc:
                record.update(status="interrupted", error=type(exc).__name__)
                state["status"] = "interrupted"
                raise
            finally:
                record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                save()
                gc.collect()
    before = json.loads((config.output_dir / "inputs_before.json").read_text())
    print("Verifying input/original files remained unchanged ...", flush=True)
    after = snapshot({Path(p) for p in before})
    changed = [p for p in before if before[p] != after[p]]
    (config.output_dir / "inputs_after.json").write_text(json.dumps(after, indent=2) + "\n")
    state["changed_inputs"] = changed
    state["status"] = "computations_completed" if not changed and all(r["status"] == "completed" for r in state["stages"].values()) else "needs_review"
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print("Audit computation status:", state["status"], flush=True)
    print("Numerical result comparisons still require review.", flush=True)
    return 0 if state["status"] == "computations_completed" else 1


def gpu_description():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "No GPU information available"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=w.CODE_DIR / "validation/full_audit")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--kinspec-dir", type=Path, default=w.KINSPEC_DIR, help="Root of the KinSPEC repository")
    parser.add_argument("--source-dir", type=Path, help="Original layout only: raw-PSM model/result directory")
    parser.add_argument("--final-dir", type=Path, help="Original layout only: KinSPEC_final directory")
    parser.add_argument("--split-dir", type=Path, help="Original layout only: similarity-band split directory")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun", help="Comma-separated completed stages to recompute during resumption")
    args = parser.parse_args()
    forced = tuple(args.rerun.split(",")) if args.rerun else ()
    if forced and (not args.resume or any(name not in w.STAGES for name in forced)):
        parser.error("--rerun requires --resume and valid stage names")
    return run_audit(w.AnalysisConfig(kinspec_dir=args.kinspec_dir, project_dir=args.source_dir, final_dir=args.final_dir, split_dir=args.split_dir,
                                     output_dir=args.output_dir, device=args.device,
                                     include_band_shap=True, nthread=args.threads), resume=args.resume, force_steps=forced)


if __name__ == "__main__":
    raise SystemExit(main())
