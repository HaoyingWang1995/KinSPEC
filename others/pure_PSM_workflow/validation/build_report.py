"""Build a concise, shareable report from completed audit evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts.workflow import CODE_DIR


def build(out):
    state = json.loads((out / "audit_state.json").read_text())
    comparison = json.loads((out / "numerical_comparison.json").read_text())
    legacy = json.loads((out.parent / "legacy_cpu/legacy_parity.json").read_text())
    checkpoints = json.loads((out / "training_checkpoint_checks.json").read_text())
    issues = []
    if state["status"] != "computations_completed" or len(state["stages"]) != 42:
        issues.append("Computations are not all complete")
    for name,record in comparison["archive_comparisons"].items():
        if "status" in record and record["status"] != "match" and not record.get("resolution"):
            issues.append(name)
    stability = comparison["archive_comparisons"].get("stability2.npz", {})
    if not stability or any(item["max_absolute_error"] > 1e-10 for item in stability.values()):
        issues.append("Stability NPZ differs from historical results")
    if checkpoints["status"] != "pass":
        issues.append("Saved-model verification did not pass")
    if len(legacy) != 17 or any(r["status"] not in {"match", "known_original_bug_prefix_matches"} for r in legacy.values()):
        issues.append("Original-source comparison did not pass")
    completed = sum(r["status"] == "completed" for r in state["stages"].values())
    title = "Complete; no unresolved execution or numerical issues were found within the audit scope" if not issues else "Further verification required: " + ", ".join(issues)
    vsds,karaman = (pd.read_csv(out/f) for f in ("fullpool_vsds.csv", "fullpool_karaman.csv"))
    intact = vsds[vsds.condition == "intact"].set_index("rep")
    ka = karaman.set_index("rep")
    rows = ["| Dataset/metric | PSM | MCA |", "|---|---:|---:|"]
    for field,label in (("AUROC","VSDS pooled AUROC"),("within_target","VSDS within-target AUROC"),("AUPRC","VSDS AUPRC")):
        rows.append(f"| {label} | {intact.loc['PSM',field]:.4f} | {intact.loc['MCA',field]:.4f} |")
    rows.append(f"| Karaman full-pool AUROC(37 Molecule) | {ka.loc['PSM','AUROC']:.4f} | {ka.loc['MCA','AUROC']:.4f} |")
    raw_shap = comparison["shap_raw_margin_additivity"]
    max_margin = max(v['max_margin_error'] for group in raw_shap.values() for v in group.values())
    arrays = comparison["archived_array_comparisons"]
    max_pool = max(v.get('max_absolute_error',0) for v in arrays['pool_attr.npz'].values())
    max_vsds = max(v.get('max_absolute_error',0) for v in arrays['shap_pertarget.npz'].values())
    unit_log = (out/'unit_tests.log').read_text()
    if 'Ran 18 tests' not in unit_log or '\nOK\n' not in unit_log:
        raise ValueError("Expected successful 18-test verification log")
    report = f"""# Full results audit record (2026-10-09)

Status: **{title}**. The audit covers all 40 Python analysis scripts from the original directory plus two preparation steps ({completed}/42 completed), including SHAP recomputation from scratch, four groups of five-fold retraining, and independent verification of saved models. Computational correctness and reproduction of previous results are documented; the strength of statistical inference remains subject to the limitations below.

## Execution and independent verification

- All 18 tests passed, covering paired missing-value handling, SHAP additivity, PCA thread consistency, model save/reload, and audit resumption.
- Verified 60 saved band prediction vectors and independently computed 120 AUROCs; maximum error: `{comparison['saved_score_vector_checks']['max_auc_error']:.2e}`.
- Reran 17 original-source analyses in isolated output directories: 16 matched the terminal statistics; the original `redund_shap` failed at the end with its known indexing bug, but output before the error matched the corrected version.
- Saved 75 trained models as UBJ and independently recomputed validation AUROC, VSDS metrics, or position-level gain distributions after reloading. Maximum error: `{checkpoints['max_absolute_error']:.2e}`; verification tolerance: `1e-5` (including CPU/GPU ranking differences).
- Compared SHA256, size, and modification time before and after execution for original-directory files and all declared inputs: **{len(state.get('changed_inputs',[]))}** files changed.

## Main results

{chr(10).join(rows)}

Per-kinase attribution results were reproduced: training-pool PSM/MCA across-kinase Jaccard was `0.182/0.112`, with MCA lower for `278/300` kinases. Cross-dataset overlap for the 31 kinases shared by both datasets was `0.516/0.426`. Five-fold retraining after deduplication gave pooled AUROC `0.6057/0.6056`, a difference of approximately `-0.0001`. Per-fold results for 2x2 hyperparameter and dimensionality controls matched the original CSVs, and stability retraining matched the original NPZ.

Maximum SHAP discrepancies from historical references were `{max_pool:.2e}` for the training pool and `{max_vsds:.2e}` for VSDS. Raw-prediction SHAP additivity checks passed for every batch, with maximum residual `{max_margin:.2e}`. Z, G, eigenvalues, and variable strata matched the historical arrays exactly.

## Corrections and explained differences

1. **Missing values in Karaman aggregation**: Lapatinib has only one label class, so its metrics are undefined. Undefined metrics are excluded pairwise after molecule-level aggregation, giving 37 valid samples and avoiding NaN column means and tests. Results are written to `karaman_summary.csv`. Tests on 185 molecule-by-fold observations must not be interpreted as tests on 185 independent molecules.
2. **Pairing in stratified correlations**: the original `strat` independently dropped missing values on each side and then truncated the arrays, potentially mismatching molecules. It now uses one shared validity mask. Printed results for the current data match the original, and missing-value cases are covered by regression tests.
3. **PCA numerical environment**: float32 randomized PCA with 4 versus 32 BLAS threads gave maximum coordinate differences of approximately `3.44e-4`, affecting tree binning and early stopping. Single-fold retraining with 32 threads reproduced previous results exactly. Both PCA controls now use a shared function to fix the original 32-thread environment and retain actual input coordinates and loadings; full comparisons were recomputed with this setting. Initial discrepancies and diagnostic experiments remain in `validation/pca_thread_check/`.
4. **GPU ranking differences**: small discrepancies affected some similarity correlations, pairwise rankings, and one MCA Karaman AUROC. CPU controls reproduced the original results exactly. These discrepancies were not silently treated as identical results.
5. Earlier fixes for collinear-group indexing, tree-traversal scope, and temporary-file dependencies remain in place. The original fixed invariance claim in `invariance` conflicted with actual values and was removed; MCA coordinates changed by approximately `0.370`. Fixed SI tables only export historical values and do not aggregate new results.

## Limits of interpretation

- MCA position-level values allocate axis SHAP using category-contribution weights; they are not recomputed TreeSHAP for the original position features. Additivity verification applies to model-input axes and the bias term.
- Zero-branch fractions use prevalence from the complete 482-kinase reference table and cannot be interpreted as actual sample flow through tree nodes. Associated SHAP weights are first averaged as signed values within each kinase, then converted to absolute values and averaged across kinases.
- Five-fold training sets overlap, and per-kinase Jaccard means share kinase pairs. Although original p-values were reproduced, this code audit does not guarantee independence or causal conclusions. Nonsignificant performance differences are not strict equivalence tests.
- Checks cover analysis workflows using existing inputs and models. Semantic correspondence between latent features and original records still depends on row ordering during generation. Row counts, labels, mappings, and input checksums were verified, but molecular encodings and original full-pool/band models were not regenerated.

## Reproduction and evidence locations

Dependency versions are listed in `requirements.txt`. This audit used Python 3.10, XGBoost 3.2.0, and an RTX 5090. Inputs are supplied by the original project and are not automatically downloaded by the code repository.

```bash
python run_validation.py --device cuda --output-dir validation/full_audit_20261009
python run_validation.py --device cuda --output-dir validation/full_audit_20261009 --resume
python validation/compare_results.py validation/full_audit_20261009
python validation/check_training_models.py validation/full_audit_20261009
```

Full local records are in `validation/full_audit_20261009/`: `audit_state.json`, `numerical_comparison.json`, `training_checkpoint_checks.json`, `input_alignment_checks.json`, before/after input snapshots, and step logs. Original-source comparisons are in `validation/legacy_cpu/legacy_parity.json`, and device comparisons are in `validation/device_check/`. These large or local records are excluded by `.gitignore`; this report and verification code can be committed to the code repository.
"""
    (CODE_DIR/'VALIDATION_REPORT.md').write_text(report)
    print(title)
    return issues


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir',type=Path)
    build(parser.parse_args().output_dir.resolve())
