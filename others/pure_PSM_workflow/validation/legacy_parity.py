"""Read-only legacy re-execution with outputs redirected to an isolated audit directory.

Reuses freshly calculated intermediates; never trains or recalculates SHAP. The original
source text is not edited on disk. Only path constants and OUT assignments are rewritten.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import redirect_stdout, redirect_stderr
import difflib
import io
import json
from pathlib import Path
import shutil
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts.workflow import AnalysisConfig

STEPS = ("geom", "shortcut", "strat", "invariance", "pca_vs_mca", "null_check", "null_audit",
         "own_space", "imptype", "tie_check", "causal", "zero_ambiguity", "pool_analyse",
         "redund_shap", "zero_shap", "perkinase2", "perkinase3")


class RedirectPaths(ast.NodeTransformer):
    def __init__(self, config, destination):
        self.destination = destination
        self.prefixes = (
            ("/tmp/claude-1947933/-home-s1856122/bca708e7-bfc5-4b83-a840-d982a9520480/scratchpad", str(destination)),
            ("/home/s1856122/KinSPEC_review/pure_PSM/analysis_scripts", str(destination)),
            ("/home/s1856122/KinSPEC_review/pure_PSM/SI_tables", str(destination / "archived_SI_tables")),
            ("/home/s1856122/KinSPEC_review/pure_PSM", str(config.project_dir)),
            ("/home/s1856122/KinSPEC_final", str(config.final_dir)),
            ("/home/s1856122/dataset/SPECTRA_data/SPECTRA_dataset/data_split_from_merged_true_fold", str(config.split_dir)),
        )

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for prefix, target in self.prefixes:
                if node.value.startswith(prefix):
                    return ast.copy_location(ast.Constant(target + node.value[len(prefix):]), node)
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any(isinstance(t, ast.Name) and t.id == "OUT" for t in node.targets):
            node.value = ast.Constant(str(self.destination))
        return node


def normalized(text, out):
    lines = []
    skip_prefixes = ("Starting audit ", "Completed audit ", "Archived analysis:", "saved", "wrote")
    for line in text.splitlines():
        if line.startswith(skip_prefixes):
            continue
        # These fixed conclusions were removed because the measured values contradict them.
        if "MCA's retained-axis solution is unchanged" in line or "PCA's coordinates move, because" in line:
            continue
        if "Compare the empirical coordinate changes" in line or "does not guarantee the exact distribution-equivalence" in line:
            continue
        line = line.replace(str(out), "<OUTPUT>").rstrip()
        if line:
            lines.append(line)
    return lines


def verify(out):
    import xgboost as xgb
    xgb.set_config(nthread=4)
    state = json.loads((out / "audit_state.json").read_text())
    config = AnalysisConfig(**state["config"])
    destination = out.parent / "legacy_cpu"
    destination.mkdir(exist_ok=True)
    for filename in ("Z.npy", "G.npy", "lam.npy", "strata.npz", "aa23.json", "pool_attr.npz", "shap_pertarget.npz", "vsds_perfold_PSM_vs_MCA.csv"):
        if (out / filename).exists():
            shutil.copyfile(out / filename, destination / filename)
    # shortcut also consumes all per-fold score vectors.
    for source in out.glob("scores_*.npy"):
        shutil.copyfile(source, destination / source.name)
    report_path = destination / "legacy_parity.json"
    results = json.loads(report_path.read_text()) if report_path.exists() else {}
    for name in STEPS:
        if results.get(name, {}).get("status") in {"match", "known_original_bug_prefix_matches"}:
            continue
        source = config.project_dir / "analysis_scripts" / f"{name}.py"
        tree = ast.fix_missing_locations(RedirectPaths(config, destination).visit(ast.parse(source.read_text())))
        stream, errors = io.StringIO(), io.StringIO()
        exception = None
        print("Checking archived source:", name, flush=True)
        with redirect_stdout(stream), redirect_stderr(errors):
            try:
                exec(compile(tree, str(destination / f"{name}.py"), "exec"),
                     {"__name__": "__legacy_validation__", "__file__": str(destination / f"{name}.py")})
            except Exception as exc:
                exception = type(exc).__name__ + ": " + str(exc)
                traceback.print_exc()
        (destination / f"{name}.log").write_text(stream.getvalue() + errors.getvalue())
        new = Path(state["stages"][name]["log"]).read_text()
        a, b = normalized(stream.getvalue(), destination), normalized(new, out)
        if exception is None and a == b:
            status = "match"
        elif name == "redund_shap" and exception and exception.startswith("IndexError") and b[:len(a)] == a:
            status = "known_original_bug_prefix_matches"
        else:
            status = "difference_requires_review"
        difference = list(difflib.unified_diff(a, b, fromfile="legacy", tofile="refactored"))
        results[name] = {"status": status, "original_exception": exception,
                         "diff": difference, "note": "CPU legacy inference versus configured audit device; shared freshly computed intermediates"}
        report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        print(name, status, flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    verify(parser.parse_args().output_dir.resolve())
