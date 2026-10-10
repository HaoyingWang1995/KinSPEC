"""Verify input migration and GitHub exclusions without external services."""
import ast
from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
import prepare_release_inputs as prepare
from analysis_scripts import workflow as w


class ReleaseInputTests(unittest.TestCase):
    def test_manifest_matches_runtime_paths(self):
        config = w.AnalysisConfig()
        with (CODE_DIR / "release_input_manifest.csv").open(newline="") as handle:
            entries = list(csv.DictReader(handle))
        expected = {p.relative_to(config.kinspec_dir).as_posix()
                    for p in w._view_entries(config, w.all_inputs(config)).values()}
        self.assertEqual(expected, {entry["repository_path"] for entry in entries})
        self.assertEqual(len(entries), 84)

    def test_migration_preserves_original_and_does_not_overwrite(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "feature_embedding/example.csv"
            original.parent.mkdir()
            original.write_text("original\n")
            manifest = root / "release_input_manifest.csv"
            manifest.write_text("repository_path,original_repository_path,bytes\n"
                                "dataset/example.csv,feature_embedding/example.csv,9\n")
            with patch.object(prepare, "CODE_DIR", root), redirect_stdout(io.StringIO()):
                self.assertEqual(prepare.prepare(root), 1)
                self.assertEqual(prepare.prepare(root, copy_existing=True), 0)
                self.assertEqual(original.read_text(), "original\n")
                target = root / "dataset/example.csv"
                target.write_text("changed!\n")
                self.assertEqual(prepare.prepare(root, copy_existing=True), 0)
                self.assertEqual(target.read_text(), "changed!\n")
                target.write_text("wrong size\n")
                self.assertEqual(prepare.prepare(root), 1)

    def test_both_uploaders_exclude_inputs_and_generated_views(self):
        project = CODE_DIR.parents[1]
        spec = importlib.util.spec_from_file_location("kinspec_uploader", project / "upload_to_github.py")
        uploader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(uploader)
        notebook = json.loads((project / "upload_to_github.ipynb").read_text())
        notebook_functions = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                for node in ast.parse("".join(cell["source"])).body:
                    if isinstance(node, ast.FunctionDef) and node.name in {"exclude_release_input", "build_plan"}:
                        notebook_functions.append(node)
        notebook_namespace = dict(vars(uploader))
        exec(compile(ast.Module(body=notebook_functions, type_ignores=[]), "<notebook>", "exec"), notebook_namespace)
        for namespace in (vars(uploader), notebook_namespace):
            with self.subTest(uploader=namespace["build_plan"].__code__.co_filename), TemporaryDirectory() as tmp:
                root = Path(tmp)
                files = ("dataset/README.md", "dataset/data.csv", "model/PSM_raw/weights.json",
                         "others/pure_PSM_workflows/run_performance.py",
                         "others/pure_PSM_workflows/release_input_manifest.csv",
                         "others/pure_PSM_workflows/results/output.npz",
                         "others/pure_PSM_workflows/validation/full_audit/model.ubj",
                         "others/pure_PSM_workflows/validation/check_training_models.py")
                for name in files:
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("fixture\n")
                link = root / "others/pure_PSM_workflows/results/_input_view/input.csv"
                link.parent.mkdir(parents=True)
                link.symlink_to(root / "dataset/data.csv")
                with patch.dict(namespace, BASE_DIR=root), redirect_stdout(io.StringIO()):
                    targets = {target for _, target in namespace["build_plan"]([root], ["."])}
                self.assertEqual(targets, {"dataset/README.md",
                                          "others/pure_PSM_workflows/run_performance.py",
                                          "others/pure_PSM_workflows/release_input_manifest.csv",
                                          "others/pure_PSM_workflows/validation/check_training_models.py"})


if __name__ == "__main__":
    unittest.main()
