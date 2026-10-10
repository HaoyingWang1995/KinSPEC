"""Shared configuration, dependency scheduling and CLI for the copied PSM analyses."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import gc
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
import traceback

CODE_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = CODE_DIR.parent / "pure_PSM"
USER_DIR = CODE_DIR.parents[1]
# Release layout: this directory is KinSPEC/others/pure_PSM_workflows inside the KinSPEC repository.
KINSPEC_DIR = CODE_DIR.parents[1]
# Repository folders that hold inputs; outputs may never be written inside them.
RELEASE_INPUT_DIRS = ("dataset", "feature_embedding", "model", "others/PSM_model")
# Original three-directory layout on the analysis server; used only when these paths are passed explicitly.
LEGACY_DIRS = {"project_dir": KINSPEC_DIR / "external/legacy_pure_PSM", "final_dir": KINSPEC_DIR,
               "split_dir": KINSPEC_DIR / "dataset/data_split_from_merged_true_fold"}
VIEW_ROOTS = {"final_dir": "KinSPEC_final", "project_dir": "pure_PSM", "split_dir": "splits"}
BANDS = ("10_50", "20_60", "30_70", "40_80", "50_90", "60_100")
BAND_DIRS = ("band_0.10_0.50", "band_0.20_0.60", "band_0.30_0.70", "band_0.40_0.80", "band_0.50_0.90", "band_0.60_1.00")


@dataclass(frozen=True)
class AnalysisConfig:
    # With no legacy directory given, inputs are read from the KinSPEC repository layout through a
    # symlink view that recreates the original layout, so the analysis scripts keep their relative paths.
    kinspec_dir: Path = KINSPEC_DIR
    project_dir: Path | None = None
    final_dir: Path | None = None
    split_dir: Path | None = None
    output_dir: Path = CODE_DIR / "results"
    device: str = "cuda"
    include_band_shap: bool = False
    nthread: int = 8
    release: bool = field(default=True, init=False)

    def __post_init__(self):
        legacy = any(getattr(self, key) is not None for key in LEGACY_DIRS)
        object.__setattr__(self, "release", not legacy)
        for key in ("kinspec_dir", "output_dir"):
            object.__setattr__(self, key, Path(getattr(self, key)).expanduser().resolve())
        for key, default in LEGACY_DIRS.items():
            value = getattr(self, key)
            if self.release:
                value = self.output_dir / "_input_view" / VIEW_ROOTS[key]
            elif value is None:
                value = default
            object.__setattr__(self, key, Path(value).expanduser().resolve())
        if self.nthread < 1:
            raise ValueError("nthread must be positive")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("device must be 'cuda' or 'cpu'")
        inputs = (ORIGINAL_DIR.resolve(), *(self.kinspec_dir / d for d in RELEASE_INPUT_DIRS)) if self.release else \
                 (ORIGINAL_DIR.resolve(), self.project_dir, self.final_dir, self.split_dir)
        for protected in inputs:
            if self.output_dir == protected or protected in self.output_dir.parents:
                raise ValueError(f"Output directory must be outside the input directory: {protected}")
        # Also prevent placing an input tree inside the output tree.
        for protected in inputs:
            if self.output_dir in protected.parents:
                raise ValueError(f"Output directory must not contain the input directory: {protected}")

    @property
    def psm_model_dir(self):
        return self.project_dir / "model"

    @property
    def embedding_dir(self):
        return self.final_dir / "feature_embedding/Kinase_test_MCA"

    def _init_values(self):
        values = {key: value for key, value in asdict(self).items() if key != "release"}
        if self.release:
            values.update({key: None for key in LEGACY_DIRS})
        return values

    def evolve(self, **changes):
        """Copy with changed settings; a release-layout configuration stays in the release layout."""
        return AnalysisConfig(**{**self._init_values(), **changes})

    def to_dict(self):
        return {key: str(value) if isinstance(value, Path) else value for key, value in self._init_values().items()}


def release_path(key, relative):
    """Data live directly in dataset/; model weights retain their model/ subfolders."""
    relative = PurePosixPath(relative)
    name, parent = relative.name, relative.parent.name
    band = BANDS[BAND_DIRS.index(parent)] if parent in BAND_DIRS else None
    if name == "kinspec_mac_karaman_per_molecule_all_folds.csv":
        return f"dataset/kinspec_{'mac_psm' if key == 'final_dir' else 'pure_psm'}_v4_karaman_per_molecule_all_folds.csv"
    if key == "split_dir":
        return f"dataset/train_band_{band}_with_smiles_protein_seq.csv"
    if key == "project_dir":
        return f"model/PSM_raw/{name}"
    if band:
        return f"dataset/train_band_{band}_{name[len('train_'):]}"
    if name == "KinSPEC_training_pool_merged_4_datasets_deduplicated_with_smiles.csv":
        # Same rows and columns as the original training pool, with an added SMILES column.
        return "dataset/train_ready2use_merged_4datasets_deduplicated_with_smiles.csv"
    if relative.parts[0] == "dataset":
        if name == "VSDS_truedecoy_integrated_data.csv":
            return "dataset/VSDS_truedecoy_integrated_data_with_protein_seq.csv"
        return f"dataset/{name}"
    if relative.parts[0] == "model":
        return f"model/KinSPEC/{name}"
    return f"dataset/{name}"


def _view_entries(config, paths):
    """Map each input path inside the view to its file in the repository."""
    entries = {}
    for path in paths:
        for key in LEGACY_DIRS:
            root = getattr(config, key)
            if root == path or root in path.parents:
                entries[path] = config.kinspec_dir / release_path(key, path.relative_to(root).as_posix())
                break
    return entries


def all_inputs(config):
    full = config.evolve(include_band_shap=True) if not config.include_band_shap else config
    return sorted({p for name in STAGES for p in required_inputs(name, full)})


def prepare_inputs(config):
    """Create the symlink view of the repository inputs; no effect for explicit legacy directories."""
    if not config.release:
        return
    entries = _view_entries(config, all_inputs(config))
    sources = list(entries.values())
    if len(set(sources)) != len(sources):
        raise RuntimeError("Two inputs map to the same repository file")
    for link, source in entries.items():
        target = Path(os.path.relpath(source, link.parent))
        if link.is_symlink() and link.readlink() == target:
            continue
        if link.is_symlink():
            link.unlink()
        if source.is_file():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target)


def display_path(path, config):
    """Report inputs by their location in the repository."""
    if not config.release:
        return str(path)
    return str(_view_entries(config, [path]).get(path, path))


def load_model(booster, path, config):
    """Load a saved model and explicitly select the inference device."""
    booster.load_model(path)
    booster.set_param({"device": config.device, "nthread": config.nthread})


@dataclass(frozen=True)
class Stage:
    dependencies: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    retrains: bool = False
    requires_band_shap: bool = False
    historical: bool = False


STAGES = {
    "prepare_mca": Stage(outputs=("Z.npy", "G.npy", "lam.npy")),
    "mech2_setup": Stage(("prepare_mca",), ("strata.npz",), historical=True),
    "mech2_retain": Stage(("mech2_setup",), ("mech2_retain.csv",), historical=True),
    "mech2_pairs": Stage(("mech2_setup",), ("mech2_pairs.csv",), historical=True),
    "mech2_pairs2": Stage(("mech2_setup",), ("mech2_pairs2.csv",), historical=True),
    "invariance": Stage(("prepare_mca",), historical=True),
    "pca_vs_mca": Stage(historical=True),
    "causal": Stage(historical=True),
    "stability": Stage(("mech2_setup",), ("stability.csv",), historical=True),
    "stability2": Stage(outputs=("stability2.npz",), retrains=True, historical=True),
    "null_check": Stage(historical=True),
    "null_audit": Stage(historical=True),
    "own_space": Stage(historical=True),
    "imptype": Stage(historical=True),
    "tie_check": Stage(historical=True),
    "perkinase2": Stage(("shap_pertarget",), requires_band_shap=True, historical=True),
    "perkinase3": Stage(("shap_pertarget",), requires_band_shap=True, historical=True),
    "zero_ambiguity": Stage(("mech2_setup",), historical=True),
    "make_tables": Stage(outputs=tuple(f"archived_SI_tables/{filename}" for filename in (
        "Table_SM2_hyperparameters.csv", "Table_SM3_performance.csv", "Table_SM4_redundancy.csv",
        "Table_SM5_attribution_by_correlation.csv", "Table_SM6_zero_branch_by_weighting.csv",
        "Table_SM7_zero_branch_distribution.csv", "Table_SM8_cross_kinase_overlap.csv",
        "Table_SM9_cross_dataset.csv", "Table_SM10_2x2_sensitivity.csv")), historical=True),
    "fullpool": Stage(outputs=("fullpool_vsds.csv", "fullpool_karaman.csv", "fullpool_scores_PSM.npy", "fullpool_scores_MCA.npy")),
    "vsds_eval": Stage(outputs=("vsds_perfold_PSM_vs_MCA.csv",)),
    "karaman": Stage(outputs=("karaman_summary.csv",)),
    "karaman_verify": Stage(outputs=("karaman_verification.csv",)),
    "decomp": Stage(outputs=("decomp.csv",)),
    "shortcut": Stage(("vsds_eval",)),
    "prepare_encoding": Stage(outputs=("strata.npz",)),
    "pool_attr": Stage(outputs=("pool_attr.npz", "pool_attr_checks.json")),
    "shap_pertarget": Stage(outputs=("shap_pertarget.npz", "shap_pertarget_checks.json")),
    "pool_analyse": Stage(("pool_attr", "shap_pertarget")),
    "redund_shap": Stage(("pool_attr",)),
    "zero_shap": Stage(("prepare_encoding", "pool_attr")),
    "path_reuse": Stage(outputs=("path_reuse.json",)),
    "geom": Stage(),
    "dedup": Stage(outputs=("dedup.csv",), retrains=True),
    "cross": Stage(outputs=("cross.csv",), retrains=True),
    "control": Stage(outputs=("control.csv",), retrains=True),
    "derive_aa23": Stage(outputs=("aa23.json",)),
    "mech": Stage(("derive_aa23",), ("mech_gain.csv",)),
    "backproj": Stage(("derive_aa23",), ("mech_mca_backproj.csv",)),
    "ablate": Stage(outputs=("ablate.csv",)),
    "guilt": Stage(outputs=("guilt.csv",)),
    "adversarial": Stage(outputs=("adversarial.csv",)),
    "strat": Stage(),
}

GROUPS = {
    "history": (("null_check", "null_audit", "own_space", "imptype", "tie_check", "causal"), (
        "prepare_mca", "mech2_setup", "mech2_retain", "mech2_pairs", "mech2_pairs2", "invariance",
        "pca_vs_mca", "causal", "stability", "stability2", "null_check", "null_audit", "own_space",
        "imptype", "tie_check", "perkinase2", "perkinase3", "zero_ambiguity")),
    "tables": (("make_tables",), ("make_tables",)),
    "performance": (("fullpool",), ("fullpool", "vsds_eval", "karaman", "karaman_verify", "decomp", "shortcut")),
    "attribution": (("pool_analyse", "redund_shap", "zero_shap", "path_reuse"), ("prepare_encoding", "pool_attr", "shap_pertarget", "pool_analyse", "redund_shap", "zero_shap", "path_reuse")),
    "controls": (("geom", "dedup", "cross", "control"), ("geom", "dedup", "cross", "control", "derive_aa23", "mech", "backproj", "ablate")),
    "selectivity": (("guilt", "adversarial", "strat"), ("guilt", "adversarial", "strat")),
}


def resolve_steps(selected):
    """Topologically order steps, running shared dependencies only once."""
    plan, visiting = [], set()

    def visit(name):
        if name not in STAGES:
            raise ValueError(f"Unknown step: {name}")
        if name in visiting:
            raise ValueError(f"Dependency cycle at {name}")
        if name in plan:
            return
        visiting.add(name)
        for dependency in STAGES[name].dependencies:
            visit(dependency)
        visiting.remove(name)
        plan.append(name)

    for name in selected:
        visit(name)
    return plan


def required_inputs(name, config):
    """Raw inputs only; intermediate files are produced by scheduled dependencies."""
    e, d = config.embedding_dir, config.final_dir / "dataset"
    ref = e / "human_482_typical_kinase_1955bit.csv"
    psm = e / "human_482_typical_kinase_1955bit_onehot.csv"
    mca = e / "MCA_klifs_482_200D_features.csv"
    contrib = e / "MCA_200D_category_contributions.csv"
    lookup = e / "MCA_200D_category_lookup.csv"
    vsds = [d / "VSDS_truedecoy_integrated_data.csv", d / "VSDS_truedecoy_LSAE_latent_top95_features_v4.csv"]
    karaman = [d / "Karaman_Kinase_selectivity_filtered_withseq.csv", d / "Karaman_10_50_filtered_SAE_latent_top95_features_from_merge_v4.csv"]
    full_models = [config.psm_model_dir / "KinSPEC_full_training_pool_pure_PSM_v4.json", config.final_dir / "model/KinSPEC_full_training_pool_MCA_200D_v4.json"]
    stored = [config.project_dir / "Karaman_pure_PSM_v4_result/kinspec_mac_karaman_per_molecule_all_folds.csv", config.final_dir / "model_training/Karaman_MAC_v4_result/kinspec_mac_karaman_per_molecule_all_folds.csv"]

    def models(bands=BANDS, p=True, m=True):
        files = []
        for band in bands:
            for fold in range(1, 6):
                if p:
                    files.append(config.psm_model_dir / f"KinSPEC_MCA_training_band_{band}_fold{fold}_pure_PSM_model_v4.json")
                if m:
                    files.append(config.final_dir / f"model/KinSPEC_MCA_training_band_{band}_fold{fold}_model_v4.json")
        return files

    standard = [ref, psm, mca]
    if name == "prepare_mca": return [psm, lookup, e / "MCA_200D_category_coordinates.csv", e / "MCA_inertia_spectrum.csv", mca]
    if name == "mech2_setup": return [e / "MCA_200D_category_coordinates.csv", lookup]
    if name in {"mech2_retain", "mech2_pairs", "mech2_pairs2"}: return standard + karaman + [lookup] + models(("10_50",))
    if name == "invariance": return [psm]
    if name == "pca_vs_mca": return [psm, mca]
    if name == "stability": return [psm, contrib, lookup, e / "MCA_200D_category_coordinates.csv"] + models()
    if name == "stability2": return standard + [contrib, config.split_dir / "band_0.10_0.50/train.csv", d / "spectrum_data/band_0.10_0.50/train_LSAE_latent_top95_features_v4.csv"]
    if name in {"causal", "null_check", "null_audit", "imptype"}: return [psm, contrib] + models()
    if name in {"own_space", "tie_check"}: return models()
    if name in {"perkinase2", "perkinase3"}: return [contrib]
    if name == "zero_ambiguity": return [psm, lookup] + models(("10_50",), m=False)
    if name == "make_tables": return []
    if name == "fullpool": return standard + vsds + karaman + full_models
    if name == "pool_attr": return standard + full_models + [d / "KinSPEC_training_pool_merged_4_datasets_deduplicated_with_smiles.csv", d / "KinSPEC_training_pool_LSAE_mol_latent_top95_features_v4.csv"]
    if name == "shap_pertarget": return standard + vsds + full_models + (models(("10_50",)) if config.include_band_shap else [])
    if name == "pool_analyse": return [ref, contrib]
    if name == "redund_shap": return [psm]
    if name == "prepare_encoding": return [psm, lookup, e / "MCA_200D_category_coordinates.csv"]
    if name == "zero_shap": return [psm, lookup, full_models[0]]
    if name == "path_reuse": return [lookup, full_models[0]]
    if name in {"guilt", "adversarial", "strat"}: return standard + karaman + full_models
    if name == "geom": return standard + [karaman[0]]
    if name in {"dedup", "cross", "control"}: return standard + vsds + [config.split_dir / "band_0.10_0.50/train.csv", d / "spectrum_data/band_0.10_0.50/train_LSAE_latent_top95_features_v4.csv"]
    if name == "vsds_eval": return standard + vsds + models()
    if name == "decomp": return standard + vsds + models(("10_50",))
    if name == "karaman": return stored
    if name == "karaman_verify": return standard + karaman + stored + models(("10_50",))
    if name == "shortcut": return [psm, mca, vsds[0]] + [config.split_dir / band / "train.csv" for band in BAND_DIRS]
    if name == "derive_aa23": return [ref, psm, lookup]
    if name == "mech": return [psm] + models()
    if name == "backproj": return [psm, contrib, lookup] + models(p=False)
    if name == "ablate": return standard + vsds + models(("10_50", "30_70", "60_100"))
    raise ValueError(f"No input declaration for step: {name}")


class Tee:
    """Write output to both the terminal and the current step log."""
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def import_step(name):
    package = __package__ or "analysis_scripts"
    if not __package__:
        root = str(CODE_DIR)
        if root not in sys.path:
            sys.path.insert(0, root)
    return importlib.import_module(f"{package}.{name}")


def run_steps(selected, config=None, *, dry_run=False, check=False):
    config = config or AnalysisConfig()
    plan = resolve_steps(selected)
    if any(STAGES[name].requires_band_shap for name in plan):
        config = config.evolve(include_band_shap=True)
    print("Output:", config.output_dir)
    print("Plan:", " -> ".join(plan))
    for name in plan:
        if STAGES[name].historical:
            print(f"  {name}: archived analysis; see its limitations in the report.")
        if STAGES[name].retrains:
            print(f"  {name}: retrains five-fold models; may take hours.")
    if dry_run:
        return plan
    prepare_inputs(config)
    missing = {name: [display_path(p, config) for p in required_inputs(name, config) if not p.is_file()] for name in plan}
    missing = {name: paths for name, paths in missing.items() if paths}
    if missing:
        details = "\n".join(f"{name}:\n  " + "\n  ".join(paths) for name, paths in missing.items())
        raise FileNotFoundError("Missing required input files:\n" + details)
    if check:
        print("Input file checks passed. No analysis was run.")
        return plan
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logs = config.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    manifest_path = logs / f"run_{stamp}.json"
    manifest = {"config": config.to_dict(), "plan": plan, "started_utc": stamp, "steps": []}

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    save_manifest()
    for name in plan:
        log_path = logs / f"{stamp}_{name}.log"
        record = {"name": name, "status": "running", "log": str(log_path)}
        manifest["steps"].append(record)
        save_manifest()
        started = time.monotonic()
        with log_path.open("w") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                print(f"\nStarting {name}", flush=True)
                module = import_step(name)
                module.run(config)
                absent = [f for f in STAGES[name].outputs if not (config.output_dir / f).is_file()]
                if absent:
                    raise RuntimeError(f"Step {name} did not produce expected outputs: {absent}")
                record["status"] = "completed"
                record["outputs"] = [str(config.output_dir / f) for f in STAGES[name].outputs]
                print(f"Completed {name} ({time.monotonic() - started:.1f}s)", flush=True)
            except BaseException as exc:
                record.update(status="failed", error=str(exc))
                traceback.print_exc()
                raise
            finally:
                record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                save_manifest()
                gc.collect()
    print("Run record:", manifest_path)
    return plan


def cli(group=None, single=None, argv=None):
    if group is not None:
        defaults, allowed = GROUPS[group]
    else:
        defaults, allowed = (single,), tuple(STAGES)
    parser = argparse.ArgumentParser(description=f"PSM analysis workflow: {group or single}")
    parser.add_argument("--steps", help="Comma-separated steps; dependencies are added automatically. Available: " + ", ".join(allowed))
    parser.add_argument("--list", action="store_true", help="List available steps and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show the execution plan without creating outputs")
    parser.add_argument("--check", action="store_true", help="Check input file availability without running analyses")
    parser.add_argument("--output-dir", type=Path, default=CODE_DIR / "results")
    parser.add_argument("--kinspec-dir", type=Path, default=KINSPEC_DIR, help="Root of the KinSPEC repository (default: two levels above this directory)")
    parser.add_argument("--source-dir", type=Path, help="Original layout only: raw-PSM model/result directory")
    parser.add_argument("--final-dir", type=Path, help="Original layout only: KinSPEC_final directory")
    parser.add_argument("--split-dir", type=Path, help="Original layout only: similarity-band split directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads per model")
    parser.add_argument("--include-band-shap", action="store_true", help="Also calculate all-example and five-fold VSDS SHAP profiles")
    args = parser.parse_args(argv)
    if args.list:
        print("Default steps:", ", ".join(defaults))
        print("Available steps:", ", ".join(allowed))
        return 0
    selected = tuple(x.strip() for x in args.steps.split(",")) if args.steps is not None else defaults
    if not selected or any(x not in allowed for x in selected):
        parser.error("--steps must contain available step names")
    try:
        config = AnalysisConfig(kinspec_dir=args.kinspec_dir, project_dir=args.source_dir, final_dir=args.final_dir, split_dir=args.split_dir, output_dir=args.output_dir, device=args.device, include_band_shap=args.include_band_shap, nthread=args.threads)
        run_steps(selected, config, dry_run=args.dry_run, check=args.check)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, str(exc) + "\n")
    return 0


def single_main(name):
    raise SystemExit(cli(single=name))
