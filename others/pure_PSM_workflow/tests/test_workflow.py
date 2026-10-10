"""Scheduling, isolation and failure handling; no production data or GPU needed."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts import workflow as w


class WorkflowTests(unittest.TestCase):
    def test_every_stage_declares_valid_input_paths(self):
        config = w.AnalysisConfig(include_band_shap=True)
        for name in w.STAGES:
            with self.subTest(step=name):
                paths = w.required_inputs(name, config)
                self.assertIsInstance(paths, list)
                self.assertTrue(all(isinstance(path, Path) for path in paths))

    def test_release_repository_layout(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "KinSPEC"
            config = w.AnalysisConfig(kinspec_dir=root, output_dir=root / "others/pure_PSM_workflows/results")
            self.assertTrue(config.release)
            inputs = w.all_inputs(config)
            sources = list(w._view_entries(config, inputs).values())
            self.assertEqual(len(sources), len(inputs))
            self.assertEqual(len(set(sources)), len(sources))
            for expected in ("dataset/train_band_10_50_with_smiles_protein_seq.csv",
                             "dataset/train_band_60_100_with_smiles_protein_seq.csv",
                             "dataset/train_band_10_50_LSAE_latent_top95_features_v4.csv",
                             "dataset/train_ready2use_merged_4datasets_deduplicated_with_smiles.csv",
                             "dataset/kinspec_pure_psm_v4_karaman_per_molecule_all_folds.csv",
                             "dataset/MCA_200D_category_lookup.csv",
                             "model/KinSPEC/KinSPEC_full_training_pool_MCA_200D_v4.json",
                             "model/PSM_raw/KinSPEC_full_training_pool_pure_PSM_v4.json"):
                self.assertIn(root / expected, sources)
            self.assertTrue(all(path.relative_to(root).parts[0] in {"dataset", "model"} for path in sources))
            self.assertEqual(sum(path.parent == root / "dataset" for path in sources), 22)
            lookup = root / "dataset/MCA_200D_category_lookup.csv"
            lookup.parent.mkdir(parents=True)
            lookup.write_text("x\n")
            w.prepare_inputs(config)
            link = config.embedding_dir / "MCA_200D_category_lookup.csv"
            self.assertTrue(link.is_symlink())
            self.assertFalse(link.readlink().is_absolute())
            self.assertEqual(link.read_text(), "x\n")
            self.assertEqual(w.display_path(config.psm_model_dir / "x.json", config), str(root / "model/PSM_raw/x.json"))
            # Copies and saved records stay in the release layout.
            self.assertTrue(config.evolve(include_band_shap=True).release)
            self.assertTrue(w.AnalysisConfig(**config.to_dict()).release)
            with self.assertRaises(ValueError):
                w.AnalysisConfig(kinspec_dir=root, output_dir=root / "dataset/results")

    def test_input_links_survive_repository_move_and_refresh_old_links(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "KinSPEC"
            out = root / "others/pure_PSM_workflows/results"
            config = w.AnalysisConfig(kinspec_dir=root, output_dir=out)
            source = root / "dataset/MCA_200D_category_lookup.csv"
            source.parent.mkdir(parents=True)
            source.write_text("release data\n")
            old = root / "feature_embedding/Kinase_test_MCA/MCA_200D_category_lookup.csv"
            old.parent.mkdir(parents=True)
            old.write_text("old data\n")
            link = config.embedding_dir / source.name
            link.parent.mkdir(parents=True)
            link.symlink_to(old)
            w.prepare_inputs(config)
            self.assertEqual(link.read_text(), "release data\n")
            moved = Path(tmp) / "reviewer_checkout"
            root.rename(moved)
            self.assertEqual((moved / link.relative_to(root)).read_text(), "release data\n")

    def test_dependency_order_and_deduplication(self):
        plan = w.resolve_steps(["pool_analyse", "redund_shap", "zero_shap"])
        self.assertEqual(plan.count("pool_attr"), 1)
        for name in plan:
            for dependency in w.STAGES[name].dependencies:
                self.assertLess(plan.index(dependency), plan.index(name))
        self.assertEqual(w.resolve_steps(["mech", "backproj"]), ["derive_aa23", "mech", "backproj"])

    def test_outputs_cannot_modify_input_trees(self):
        with self.assertRaises(ValueError):
            w.AnalysisConfig(output_dir=w.ORIGINAL_DIR / "results")
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()
            (base / "link").symlink_to(source, target_is_directory=True)
            with self.assertRaises(ValueError):
                w.AnalysisConfig(project_dir=source, output_dir=base / "link/results")

    def test_dry_run_is_side_effect_free(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            with redirect_stdout(io.StringIO()):
                plan = w.run_steps(["pool_analyse"], w.AnalysisConfig(output_dir=out), dry_run=True)
            self.assertEqual(plan, ["pool_attr", "shap_pertarget", "pool_analyse"])
            self.assertFalse(out.exists())

    def test_archived_perkinase_requests_band_shap_before_preflight(self):
        with TemporaryDirectory() as tmp:
            config = w.AnalysisConfig(output_dir=Path(tmp) / "out")
            inspected = []

            def inputs(name, received):
                inspected.append((name, received.include_band_shap))
                return []

            with patch.object(w, "required_inputs", side_effect=inputs), redirect_stdout(io.StringIO()):
                plan = w.run_steps(["perkinase2", "perkinase3"], config, check=True)
            self.assertEqual(plan, ["shap_pertarget", "perkinase2", "perkinase3"])
            self.assertTrue(all(flag for _, flag in inspected))
            self.assertFalse(config.include_band_shap)
            self.assertFalse(config.output_dir.exists())

    def test_missing_inputs_fail_before_running(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = w.AnalysisConfig(project_dir=base / "source", final_dir=base / "final", split_dir=base / "splits", output_dir=base / "out")
            with patch.object(w, "import_step") as importer, redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):
                    w.run_steps(["guilt"], config)
            importer.assert_not_called()
            self.assertFalse(config.output_dir.exists())

    def test_function_calls_and_outputs_are_logged(self):
        with TemporaryDirectory() as tmp:
            config = w.AnalysisConfig(output_dir=Path(tmp) / "out")
            calls = []

            def importer(name):
                def run(received):
                    self.assertIs(received, config)
                    calls.append(name)
                    print("calculated", name)
                    for filename in w.STAGES[name].outputs:
                        (config.output_dir / filename).write_text("fixture")
                return SimpleNamespace(run=run)

            with patch.object(w, "required_inputs", return_value=[]), patch.object(w, "import_step", side_effect=importer), redirect_stdout(io.StringIO()):
                w.run_steps(["mech", "backproj"], config)
            self.assertEqual(calls, ["derive_aa23", "mech", "backproj"])
            manifests = list((config.output_dir / "logs").glob("run_*.json"))
            result = json.loads(manifests[0].read_text())
            self.assertEqual([step["status"] for step in result["steps"]], ["completed"] * 3)
            self.assertIn("calculated mech", Path(result["steps"][1]["log"]).read_text())

    def test_failure_is_logged_and_stops_later_steps(self):
        with TemporaryDirectory() as tmp:
            config = w.AnalysisConfig(output_dir=Path(tmp) / "out")
            calls = []

            def importer(name):
                def run(received):
                    calls.append(name)
                    raise RuntimeError("fixture failure")
                return SimpleNamespace(run=run)

            with patch.object(w, "required_inputs", return_value=[]), patch.object(w, "import_step", side_effect=importer), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    w.run_steps(["guilt", "adversarial"], config)
            self.assertEqual(calls, ["guilt"])
            manifest = json.loads(next((config.output_dir / "logs").glob("run_*.json")).read_text())
            self.assertEqual(manifest["steps"][0]["status"], "failed")
            self.assertIn("fixture failure", Path(manifest["steps"][0]["log"]).read_text())


if __name__ == "__main__":
    unittest.main()
