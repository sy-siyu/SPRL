"""Integrity and camera-ready agreement checks for the packaged seed results."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("render_results", ROOT / "scripts" / "render_results.py")
assert SPEC and SPEC.loader
render = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render)


class SavedResultsTest(unittest.TestCase):
    def test_complete_paired_cells_and_manuscript_tables(self):
        tables = render.build_tables()  # validates every cell and its exact 100-seed set
        render.check(tables)  # validates every rounded entry in main Tables 1–3
        self.assertEqual(len(tables["table_1_setting_a"]), 7)
        self.assertEqual(len(tables["table_2_setting_b"]), 7)
        self.assertEqual(len(tables["table_3_summary"]), 4)
        self.assertEqual(len(tables["figure_4a_lambda_sweep"]), 5)
        # Exact published verification-summary bootstrap values, with no private-source dependency.
        a = next(r for r in tables["appendix_setting_a_endpoints"]
                 if r["w_u"] == "0.3" and r["method"] == "sprl")
        self.assertAlmostEqual(a["width_ratio_bootstrap95_lower"], 0.9871763550036884, places=12)
        self.assertAlmostEqual(a["width_ratio_bootstrap95_upper"], 1.0150453921200089, places=12)
        ihdp = next(r for r in tables["appendix_benchmark_methods"]
                    if (r["benchmark"], r["w_u"], r["method"]) == ("ihdp", "0.5", "sprl"))
        self.assertAlmostEqual(ihdp["width_ratio_bootstrap95_lower"], 1.030647886599691, places=12)
        self.assertAlmostEqual(ihdp["width_ratio_bootstrap95_upper"], 1.0505291360453903, places=12)
        half = next(r for r in tables["appendix_benchmark_paired"]
                    if (r["benchmark"], r["comparison"], r["metric"]) ==
                    ("halfcheetah", "sprl_minus_ldm", "fresh_amplification_p95"))
        self.assertAlmostEqual(half["paired_median_bootstrap95_lower"], -7.978391952562147, places=12)
        self.assertAlmostEqual(half["paired_median_bootstrap95_upper"], -0.7680027733220226, places=12)

    def test_saved_rows_are_numeric_summaries_and_retain_extremes(self):
        a = render.read_rows("setting_a_per_seed.csv")
        b = render.read_rows("setting_b_per_seed.csv")
        semi = render.read_rows("benchmark_per_seed.csv")
        self.assertEqual((len(a), len(b), len(semi)), (2100, 3000, 1600))
        for rows in (a, b, semi):
            self.assertNotIn("path", " ".join(rows[0]).lower())
            self.assertNotIn("historical", " ".join(rows[0]).lower())
        half = [r for r in semi if r["benchmark"] == "halfcheetah"]
        self.assertAlmostEqual(max(float(r["fresh_amplification_p95"]) for r in half
                                   if r["method"] == "balanced"), 1177.0091889031285)
        self.assertAlmostEqual(max(float(r["fresh_amplification_p95"]) for r in half
                                   if r["method"] == "sprl"), 95.72684107400242)


if __name__ == "__main__":
    unittest.main()
