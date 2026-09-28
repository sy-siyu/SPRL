"""Fast CLI checks; smoke settings are not paper results.

Run with: python -m unittest discover -s tests -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments import setting_a, setting_b


class SyntheticSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_nonempty_unrelated_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "user_data.txt").write_text("keep me", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonempty output"):
                setting_a.run(setting_a.load_config(smoke=True), 1, 1, output)
            with self.assertRaisesRegex(ValueError, "nonempty output"):
                setting_b.run(setting_b.load_config(smoke=True), 1, 1, output)
            self.assertEqual((output / "user_data.txt").read_text(encoding="utf-8"), "keep me")

    def test_setting_a_smoke_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "a"
            argv = ["setting_a", "--smoke", "--output", str(output)]
            with patch.object(sys, "argv", argv):
                setting_a.main()
                setting_a.main()  # bounded resume
            path = output / "wu_05" / "seed_001.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            config = setting_a.load_config(smoke=True)
            self.assertEqual(list(row["methods"]), config["method_order"])
            self.assertAlmostEqual(row["analytic_target_value"], 0.7)
            self.assertEqual(sorted(p.name for p in path.parent.glob("seed_*.json")), ["seed_001.json"])
            self.assertTrue(all(np.isfinite(metrics["width"]) and metrics["width"] >= 0
                                for metrics in row["methods"].values()))
            summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["n_complete_seeds"], 1)
            self.assertFalse(summary["complete_paper_protocol"])

    def test_setting_b_smoke_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "b"
            argv = ["setting_b", "--smoke", "--output", str(output)]
            with patch.object(sys, "argv", argv):
                setting_b.main()
                setting_b.main()
            path = output / "wu_05" / "seed_001.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            config = setting_b.load_config(smoke=True)
            self.assertEqual(list(row["methods"]), config["method_order"])
            self.assertEqual(row["methods"]["raw"]["amplification"], 1.0)
            self.assertIsNotNone(row["methods"]["cprl_lam_1.0"]["trans_mse"])
            self.assertEqual(sorted(p.name for p in path.parent.glob("seed_*.json")), ["seed_001.json"])
            self.assertTrue(all(np.isfinite(value) for metrics in row["methods"].values()
                                for value in metrics.values() if value is not None))
            summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["n_complete_seeds"], 1)
            self.assertFalse(summary["complete_paper_protocol"])


if __name__ == "__main__":
    unittest.main()
