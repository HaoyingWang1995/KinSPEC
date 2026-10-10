"""Regression for the inherited zero-attribution collinear-group failure."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts.workflow import AnalysisConfig, STAGES, import_step

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = pd = None


class AnalysisTests(unittest.TestCase):
    @unittest.skipIf(np is None, "Requires the documented analysis environment")
    def test_pca_features_are_independent_of_callers_blas_thread_setting(self):
        from threadpoolctl import threadpool_limits
        from analysis_scripts.pca_features import fitted_features
        matrix = np.random.default_rng(42).integers(0, 2, size=(350, 600)).astype(np.float32)
        with threadpool_limits(limits=1, user_api="blas"):
            first = fitted_features(matrix)
        with threadpool_limits(limits=4, user_api="blas"):
            second = fitted_features(matrix)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])

    @unittest.skipIf(np is None, "Requires the documented analysis environment")
    def test_strat_missing_values_preserve_molecule_pairs(self):
        from analysis_scripts.strat import finite_pairs
        a, b = finite_pairs([0.1, np.nan, 0.3, 0.4], [0.5, 0.6, np.nan, 0.8])
        np.testing.assert_array_equal(a, [0.1, 0.4])
        np.testing.assert_array_equal(b, [0.5, 0.8])

    @unittest.skipIf(np is None, "Requires the documented analysis environment")
    def test_karaman_undefined_pairs_do_not_poison_molecule_summary(self):
        from analysis_scripts.karaman import paired_summary
        result = paired_summary([0.6, np.nan, 0.8, np.inf], [0.7, np.nan, 0.9, 0.4])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["wins"], 2)
        self.assertAlmostEqual(result["psm"], 0.7)
        self.assertAlmostEqual(result["mca"], 0.8)
        self.assertTrue(np.isfinite(result["p"]))
        self.assertEqual(paired_summary([0.6, 0.8], [0.6, 0.8])["p"], 1.0)
        self.assertEqual(paired_summary([np.nan], [np.nan])["n"], 0)

    @unittest.skipIf(np is None, "Requires the documented analysis environment")
    def test_shap_check_validates_bias_and_rejects_corrupted_margin(self):
        import xgboost as xgb
        from analysis_scripts.shap_checks import checked_contributions
        from unittest.mock import patch
        matrix = xgb.DMatrix(np.arange(48, dtype=np.float32).reshape(16, 3), label=[0]*8+[1]*8)
        model = xgb.train({"objective": "binary:logistic", "device": "cpu", "nthread": 1, "max_depth": 2}, matrix, num_boost_round=3)
        checks = {}
        values = checked_contributions(model, matrix, checks)
        self.assertEqual(values.shape, (16, 4))
        self.assertEqual(checks["rows"], 16)
        predict = model.predict

        def corrupt(data, **kwargs):
            result = predict(data, **kwargs)
            return result + 0.1 if kwargs.get("output_margin") else result

        with patch.object(model, "predict", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "additivity"):
                checked_contributions(model, matrix, {})

    @unittest.skipIf(np is None, "Requires the documented analysis environment")
    def test_training_checkpoint_preserves_predictions_and_metadata(self):
        import json
        import xgboost as xgb
        from analysis_scripts.training_checks import save_checkpoint
        matrix = xgb.DMatrix(np.arange(48, dtype=np.float32).reshape(16, 3), label=[0]*8+[1]*8)
        params = {"objective": "binary:logistic", "eval_metric": "auc", "device": "cpu", "nthread": 1, "seed": 0}
        model = xgb.train(params, matrix, num_boost_round=5, evals=[(matrix, "validation")], early_stopping_rounds=1, verbose_eval=False)
        with TemporaryDirectory() as tmp:
            config = AnalysisConfig(output_dir=Path(tmp), device="cpu")
            save_checkpoint(model, config, "fixture", "representation", 1, params)
            target = next((Path(tmp) / "training_models/fixture").glob("*.ubj"))
            restored = xgb.Booster(model_file=target)
            np.testing.assert_array_equal(model.predict(matrix), restored.predict(matrix))
            metadata = json.loads(target.with_suffix(".json").read_text())
            self.assertEqual(metadata["num_features"], 3)
            self.assertEqual(metadata["best_iteration"], model.best_iteration)

    def test_imports_do_not_start_analysis(self):
        for name in STAGES:
            with self.subTest(step=name), redirect_stdout(io.StringIO()) as output:
                module = import_step(name)
                self.assertTrue(callable(module.run))
                self.assertEqual(output.getvalue(), "")

    def test_all_original_analysis_scripts_have_callable_stages(self):
        from analysis_scripts.workflow import ORIGINAL_DIR
        originals = {p.stem for p in (ORIGINAL_DIR / "analysis_scripts").glob("*.py")}
        if not originals:
            self.skipTest("Original source directory is not available")
        self.assertTrue(originals <= STAGES.keys(), originals - STAGES.keys())

    @unittest.skipIf(np is None, "Requires the documented analysis environment")
    def test_indicator_reconstructs_gaps_in_coordinate_row_order(self):
        from analysis_scripts.prepare_mca import build_indicator
        lookup = pd.DataFrame({
            "category_name": ["res1_system_A", "res1_system_B", "res1_system_Gap"],
            "residue": [1, 1, 1], "system": ["system"] * 3,
            "category": ["A", "B", "Gap"], "bit_index_0_based": [0, 1, np.nan],
        })
        psm = np.array([[1, 0], [0, 1], [0, 0]])
        categories = ["res1_system_Gap", "res1_system_B", "res1_system_A"]
        indicator, variables = build_indicator(psm, lookup, categories)
        np.testing.assert_array_equal(indicator, [[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        self.assertEqual(variables, 1)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build_indicator(np.array([[1, 1]]), lookup, categories)
        with self.assertRaisesRegex(ValueError, "binary"):
            build_indicator(np.array([[0.5, 0]]), lookup, categories)

    @unittest.skipIf(np is None, "Requires the documented analysis environment")
    def test_zero_attribution_group_does_not_shift_group_indices(self):
        from analysis_scripts import redund_shap
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            final = base / "final"
            e = final / "feature_embedding/Kinase_test_MCA"
            e.mkdir(parents=True)
            out = base / "out"
            out.mkdir()
            # Two collinear groups; the first group is unused by the model.
            psm = np.zeros((482, 1955))
            psm[:241, :2] = 1
            psm[::2, 2:4] = 1
            pd.DataFrame(psm).to_csv(e / "human_482_typical_kinase_1955bit_onehot.csv", header=False, index=False)
            shap = np.zeros((2, 1955))
            shap[:, 2:4] = [0.75, 0.25]
            np.savez(out / "pool_attr.npz", PSM_sgn=shap)
            config = AnalysisConfig(final_dir=final, output_dir=out, device="cpu")
            with redirect_stdout(io.StringIO()) as output:
                redund_shap.run(config)
            self.assertIn("Collinear groups: 2 groups, covering 4 columns", output.getvalue())
            self.assertIn("2 member groups (n=1): mean largest-member fraction 0.750", output.getvalue())
            self.assertIn("At the same KLIFS position: 2/2 groups   Across positions: 0 groups", output.getvalue())


if __name__ == "__main__":
    unittest.main()
