"""Compare audit artifacts to archives and independently verify key metric calculations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts.workflow import AnalysisConfig, ORIGINAL_DIR, BANDS

CSV_KEYS = {
    "vsds_perfold_PSM_vs_MCA.csv": ["band", "fold", "rep", "scope"],
    "decomp.csv": ["fold", "rep", "cond"], "ablate.csv": ["band", "fold"],
    "guilt.csv": ["m"], "adversarial.csv": ["m"],
    "mech_gain.csv": ["band", "fold", "stratum"],
    "mech_mca_backproj.csv": ["band", "fold", "stratum"],
    "stability.csv": ["band", "rep", "level"],
    "cross.csv": ["kinase", "hp", "fold"], "control.csv": ["rep", "fold"],
    "mech2_retain.csv": ["fold", "rep", "case"],
    "mech2_pairs.csv": ["fold", "rep"], "mech2_pairs2.csv": ["fold", "TOP", "rep"],
}


def compare_frame(new, old, keys, atol=5e-6):
    if list(new.columns) != list(old.columns) or new.shape != old.shape:
        return {"status": "different", "reason": "Schema or dimensions differ", "new_shape": new.shape, "old_shape": old.shape}
    keys = [key for key in keys if key in new.columns]
    if keys:
        if new.duplicated(keys).any() or old.duplicated(keys).any():
            return {"status": "different", "reason": "Duplicate comparison keys"}
        new, old = (frame.sort_values(keys).reset_index(drop=True) for frame in (new, old))
    errors = {}
    nonnumeric = []
    for column in new:
        if pd.api.types.is_numeric_dtype(new[column]) and pd.api.types.is_numeric_dtype(old[column]):
            a, b = new[column].to_numpy(float), old[column].to_numpy(float)
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                errors[column] = "Missing-value patterns differ"
            else:
                finite = np.isfinite(a) & np.isfinite(b)
                errors[column] = float(np.abs(a[finite]-b[finite]).max()) if finite.any() else 0.0
        elif not new[column].fillna("<missing>").equals(old[column].fillna("<missing>")):
            nonnumeric.append(column)
    match = not nonnumeric and all(isinstance(v, float) and v <= atol for v in errors.values())
    return {"status": "match" if match else "difference_requires_review", "rows": len(new),
            "tolerance": atol, "max_absolute_errors": errors, "different_text_columns": nonnumeric}


def auc_from_ranks(y, s):
    """Mann-Whitney definition, independent of sklearn's ROC implementation."""
    positive = y == 1
    a, b = int(positive.sum()), int((~positive).sum())
    return float((rankdata(s)[positive].sum() - a*(a+1)/2) / (a*b))


def metric_checks(out, config):
    v = pd.read_csv(config.final_dir / "dataset/VSDS_truedecoy_integrated_data.csv", usecols=["uniprot_id", "molecule_label"])
    y, targets = v.molecule_label.to_numpy(int), v.uniprot_id.to_numpy()
    frame = pd.read_csv(out / "fullpool_vsds.csv")
    result = {}
    for tag in ("PSM", "MCA"):
        scores = np.load(out / f"fullpool_scores_{tag}.npy")
        if scores.shape != y.shape or not np.isfinite(scores).all():
            raise ValueError("Full-pool scores have invalid dimensions or values")
        n, active = len(y), int(y.sum())
        observed = y[np.argsort(scores)[::-1]]
        independently = {"AUROC": auc_from_ranks(y, scores),
            "within_target": float(np.mean([auc_from_ranks(y[targets == t], scores[targets == t]) for t in np.unique(targets) if np.unique(y[targets == t]).size == 2]))}
        for alpha, name in ((20, "BEDROC20"), (80.5, "BEDROC80.5")):
            weights = np.exp(-alpha*np.arange(1,n+1)/n)
            independently[name] = float((weights[observed == 1].sum()-weights[-active:].sum()) / (weights[:active].sum()-weights[-active:].sum()))
        for frac, name in ((.005, "EF_0.5%"), (.01, "EF_1%"), (.05, "EF_5%")):
            k = max(1, round(n*frac))
            independently[name] = float(observed[:k].sum()/k/(active/n))
        row = frame[(frame.rep == tag) & (frame.condition == "intact")].iloc[0]
        errors = {name: abs(value - row[name]) for name, value in independently.items()}
        result[tag] = {"status": "match" if max(errors.values()) < 1e-10 else "different", "absolute_errors": errors}
    k = pd.read_csv(out / "karaman_verification.csv")
    result["karaman_saved_model_origin"] = {tag: float(np.abs(k[f"stored_{stored}"]-k[f"AUROC_{tag}"]).max()) for tag,stored in (("PSM","pure_PSM"),("MCA","MCA"))}
    return result


def score_checks(out, config):
    """Recompute all saved band-score AUROCs using the Mann-Whitney definition."""
    v = pd.read_csv(config.final_dir / "dataset/VSDS_truedecoy_integrated_data.csv", usecols=["uniprot_id", "molecule_label"])
    y, targets = v.molecule_label.to_numpy(int), v.uniprot_id.to_numpy()
    frame = pd.read_csv(out / "vsds_perfold_PSM_vs_MCA.csv")
    records = []
    for band in BANDS:
        for fold in range(1, 6):
            for tag in ("PSM", "MCA"):
                values = np.load(out / f"scores_{tag}_{band}_f{fold}.npy")
                if values.shape != y.shape or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                    raise ValueError(f"Invalid score vector for {band} fold {fold} {tag}")
                rows = frame[(frame.band == band) & (frame.fold == fold) & (frame.rep == tag)]
                pooled = auc_from_ranks(y, values)
                macro = float(np.mean([auc_from_ranks(y[targets == t], values[targets == t]) for t in np.unique(targets)]))
                error = max(abs(pooled-rows[rows.scope == "pooled"].AUROC.iloc[0]), abs(macro-rows[rows.scope == "macro_target"].AUROC.iloc[0]))
                records.append({"band": band, "fold": fold, "rep": tag, "max_auc_error": float(error)})
    return {"vectors_checked": len(records), "auc_values_checked": 2*len(records),
            "max_auc_error": max(r["max_auc_error"] for r in records), "details": records}


def compare(out):
    state = json.loads((out / "audit_state.json").read_text())
    config = AnalysisConfig(**{key: value for key, value in state["config"].items()})
    report = {"computation_status": state["status"], "stages": {name: record["status"] for name,record in state["stages"].items()}, "archive_comparisons": {}}
    for filename, keys in CSV_KEYS.items():
        new, old = out / filename, ORIGINAL_DIR / "analysis_scripts" / filename
        if not old.exists():
            old = out.parent / "archived_references" / filename
        if new.exists() and old.exists():
            report["archive_comparisons"][filename] = compare_frame(pd.read_csv(new), pd.read_csv(old), keys)
    old, new = ORIGINAL_DIR / "analysis_scripts/stability2.npz", out / "stability2.npz"
    if old.exists() and new.exists():
        with np.load(old) as a, np.load(new) as b:
            report["archive_comparisons"]["stability2.npz"] = {key: {"shape": b[key].shape, "max_absolute_error": float(np.abs(a[key]-b[key]).max())} for key in a.files}
    if (out / "fullpool_vsds.csv").exists() and (out / "karaman_verification.csv").exists():
        report["independent_metric_checks"] = metric_checks(out, config)
    if (out / "vsds_perfold_PSM_vs_MCA.csv").exists():
        report["saved_score_vector_checks"] = score_checks(out, config)
    # Previous CPU predictions provide another device-independent reference.
    previous = out.parent / "full_audit"
    for filename in ("fullpool_vsds.csv", "fullpool_karaman.csv"):
        if (previous / filename).exists() and (out / filename).exists():
            report["archive_comparisons"]["previous_CPU/"+filename] = compare_frame(pd.read_csv(out/filename), pd.read_csv(previous/filename), ["condition","rep"])
    shap = {}
    for filename in ("pool_attr_checks.json", "shap_pertarget_checks.json"):
        if (out / filename).exists():
            shap[filename] = json.loads((out / filename).read_text())
    report["shap_raw_margin_additivity"] = shap
    references = out.parent / "archived_references"
    arrays = {}
    for filename in ("Z.npy", "G.npy", "lam.npy", "strata.npz", "pool_attr.npz", "shap_pertarget.npz"):
        if not (out/filename).exists() or not (references/filename).exists():
            continue
        a, b = np.load(out/filename, allow_pickle=True), np.load(references/filename, allow_pickle=True)
        pairs = {key: (a[key], b[key]) for key in a.files} if filename.endswith(".npz") else {"array": (a, b)}
        details = {}
        for key,(new,old) in pairs.items():
            if new.shape != old.shape:
                details[key] = {"status": "different_shape", "new": new.shape, "old": old.shape}
            elif np.issubdtype(new.dtype, np.number):
                details[key] = {"max_absolute_error": float(np.abs(new-old).max()),
                                "mean_absolute_error": float(np.abs(new-old).mean())}
            else:
                details[key] = {"identical": bool(np.array_equal(new, old))}
        arrays[filename] = details
        if filename.endswith(".npz"):
            a.close(); b.close()
    report["archived_array_comparisons"] = arrays
    devices = out.parent / "device_check"
    for filename in ("guilt.csv", "mech2_pairs2.csv"):
        old = ORIGINAL_DIR / "analysis_scripts" / filename
        if not old.exists():
            old = references / filename
        if (devices/filename).exists() and old.exists():
            cpu = compare_frame(pd.read_csv(devices/filename), pd.read_csv(old), CSV_KEYS[filename], atol=1e-12)
            report.setdefault("CPU_device_crosscheck", {})[filename] = cpu
            if cpu["status"] == "match" and report["archive_comparisons"].get(filename, {}).get("status") == "difference_requires_review":
                report["archive_comparisons"][filename]["resolution"] = "CPU reproduces archived values exactly; GPU floating-point differences affect tied ranks"
    if (devices / "karaman_verification.csv").exists():
        d = pd.read_csv(devices / "karaman_verification.csv")
        report.setdefault("CPU_device_crosscheck", {})["karaman_saved_model_origin"] = {tag: float(np.abs(d[f"stored_{stored}"]-d[f"AUROC_{tag}"]).max()) for tag,stored in (("PSM","pure_PSM"),("MCA","MCA"))}
    (out / "numerical_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    compare(parser.parse_args().output_dir.resolve())
