"""Audit resumption must not reuse corrupted outputs or modified input data."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_validation as audit
from analysis_scripts import workflow as w


class AuditTests(unittest.TestCase):
    def test_resume_checks_outputs_and_invalidates_dependents(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            input_file = source / "fixture.txt"
            input_file.write_text("immutable input")
            config = w.AnalysisConfig(project_dir=source, output_dir=root / "out", device="cpu")
            stages = {"derive_aa23": w.Stage(outputs=("aa23.json",)), "mech": w.Stage(("derive_aa23",), ("mech_gain.csv",))}
            calls = []

            def importer(name):
                def run(received):
                    calls.append(name)
                    for filename in stages[name].outputs:
                        (received.output_dir / filename).write_text("calculated fixture")
                return SimpleNamespace(run=run)

            with patch.object(audit, "SELECTED", ("mech",)), patch.object(w, "STAGES", stages), \
                 patch.object(w, "required_inputs", return_value=[input_file]), \
                 patch.object(w, "import_step", side_effect=importer), patch.object(audit, "gpu_description", return_value="fixture"), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(audit.run_audit(config), 0)
                self.assertEqual(calls, ["derive_aa23", "mech"])
                calls.clear()
                self.assertEqual(audit.run_audit(config, resume=True), 0)
                self.assertEqual(calls, [])
                (config.output_dir / "aa23.json").write_text("corrupted result")
                self.assertEqual(audit.run_audit(config, resume=True), 0)
                self.assertEqual(calls, ["derive_aa23", "mech"])
                input_file.write_text("different input")
                with self.assertRaisesRegex(ValueError, "Inputs changed"):
                    audit.run_audit(config, resume=True)


if __name__ == "__main__":
    unittest.main()
