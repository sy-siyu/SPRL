"""Small generated-data checks; no external benchmark assets or downloads."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dgp.benchmark_support import EmpiricalShiftSupport, load_ihdp_covariates
from experiments.benchmark_utils import (inject_confounding_multistep,
                                         make_default_policy_weights,
                                         split_halfcheetah)
from experiments.benchmarks import (CONFIG, METHODS, complete, configure_torch,
                                    prepare_output_dir, resolve_seed_range, train_case)


class BenchmarkTests(unittest.TestCase):
    def test_ihdp_loader_uses_only_true_covariates(self):
        rng = np.random.RandomState(4)
        source = rng.normal(size=(747, 30))
        source[:, 0] = 0
        source[:139, 0] = 1
        source[:, 5:30] = np.arange(747)[:, None] + np.arange(25)[None, :]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ihdp_npci_1.csv"
            np.savetxt(path, source, delimiter=",")
            x = load_ihdp_covariates(path)
        self.assertEqual(x.shape, (747, 25))
        np.testing.assert_allclose(x.mean(0), 0, atol=1e-12)
        np.testing.assert_allclose(x.std(0), 1, atol=1e-12)
        self.assertEqual(np.unique(x[:, 0]).size, 747)

    def test_exact_empirical_shift_posterior_counts(self):
        x = np.zeros((3, 25), dtype=float)
        x[1:, :10] = 2.0
        support = EmpiricalShiftSupport(x, 10, 2.0)
        # At shifted state 2, two U=0 subjects and one U=1 subject overlap.
        np.testing.assert_allclose(support.posterior_u1(x[1:2]), [1 / 3])
        self.assertEqual(support.overlap_points, 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            support.posterior_u1(np.full((1, 25), 77.0))

    def test_halfcheetah_episode_split_and_train_only_scaler(self):
        episodes = []
        for i in range(7):
            s = np.full((20, 17), float(i), dtype=np.float32)
            episodes.append({"s": s, "s_next": s + 1,
                             "r": np.full(20, i, dtype=np.float32)})
        first = inject_confounding_multistep(episodes, (0, 1, 8, 11, 12), 0.5,
                                            0.5, make_default_policy_weights(), 0.0,
                                            seed=101)
        second = inject_confounding_multistep(episodes, (0, 1, 8, 11, 12), 0.5,
                                             0.5, make_default_policy_weights(), 0.0,
                                             seed=101)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first["a"].dtype, np.float32)
        self.assertEqual(first["u"].dtype, np.float32)
        with tempfile.TemporaryDirectory() as folder:
            case = split_halfcheetah(episodes, 101, Path(folder), smoke=True)
        ids = case["episode_split"]
        perm = np.random.RandomState(101).permutation(7)
        self.assertEqual(ids, {"train": perm[:5].tolist(),
                               "fit": perm[5:6].tolist(),
                               "fresh": perm[6:7].tolist()})
        self.assertTrue(all(not (set(ids[a]) & set(ids[b])) for a, b in
                            (("train", "fit"), ("train", "fresh"), ("fit", "fresh"))))
        np.testing.assert_allclose(case["train"]["s"].mean(0), 0, atol=1e-5)
        self.assertEqual(len(case["train"]["s"]), 100)
        self.assertEqual(len(case["fit"]["s"]), 20)
        self.assertEqual(len(case["fresh"]["s"]), 20)
        np.testing.assert_array_equal(case["fresh"]["r"],
                                      np.full(20, ids["fresh"][0], dtype=np.float32))

    def test_public_protocol_and_completion(self):
        self.assertEqual(METHODS, ("raw", "balanced", "world_model", "sprl_neural"))
        self.assertEqual(CONFIG["seeds"], [101, 200])
        self.assertEqual(CONFIG["conditions"], {"ihdp": [0.3, 0.5, 0.8],
                                                "halfcheetah": [0.5]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.assertFalse(complete(path))
            (path / "result.json").write_text(json.dumps({}))
            self.assertFalse(complete(path))

    def test_smoke_seed_default_and_explicit_range(self):
        self.assertEqual(resolve_seed_range(101, None, True), (101, 101))
        self.assertEqual(resolve_seed_range(105, None, True), (105, 105))
        self.assertEqual(resolve_seed_range(101, None, False), (101, 200))
        self.assertEqual(resolve_seed_range(101, 103, True), (101, 103))
        with self.assertRaises(ValueError):
            resolve_seed_range(100, None, True)

    def test_nonempty_unconfigured_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            marker = path / "unrelated.txt"
            marker.write_text("keep me")
            with self.assertRaisesRegex(ValueError, "Nonempty output"):
                prepare_output_dir(path)
            self.assertEqual(marker.read_text(), "keep me")
            self.assertEqual(sorted(p.name for p in path.iterdir()), ["unrelated.txt"])

    def test_generated_ihdp_end_to_end_smoke(self):
        rng = np.random.RandomState(7)
        x = rng.normal(size=(747, 25))
        x = (x - x.mean(0)) / x.std(0)
        support = EmpiricalShiftSupport(x, 10, 2.0)
        configure_torch()
        with tempfile.TemporaryDirectory() as folder:
            result = train_case("ihdp", 0.5, 101, Path(folder),
                                x=x, support=support, smoke=True)
            self.assertTrue(complete(Path(folder)))
            self.assertEqual(result["n_training"], 128)
            self.assertEqual(result["n_fresh"], 128)
            self.assertEqual(set(result["methods"]), set(METHODS))
            self.assertTrue(result["smoke_nonpaper"])
            for metrics in result["methods"].values():
                self.assertTrue(np.isfinite(metrics["fresh_amplification_median"]))
                self.assertTrue(np.isfinite(metrics["interval_width"]))

    def test_generated_halfcheetah_end_to_end_smoke(self):
        rng = np.random.RandomState(8)
        episodes = []
        for i in range(7):
            s = rng.normal(size=(20, 17)).astype(np.float32)
            episodes.append({"s": s, "s_next": s + 0.1,
                             "r": rng.normal(size=20).astype(np.float32)})
        configure_torch()
        with tempfile.TemporaryDirectory() as folder:
            result = train_case("halfcheetah", 0.5, 101, Path(folder),
                                episodes=episodes, smoke=True)
            self.assertTrue(complete(Path(folder)))
            self.assertEqual(result["n_training"], 100)
            self.assertEqual(result["n_readout_fit"], 20)
            self.assertEqual(result["n_fresh"], 20)
            self.assertIsNone(result["analytic_target"])
            self.assertNotIn("interval_width", result["methods"]["sprl_neural"])
            self.assertTrue(np.isfinite(result["methods"]["sprl_neural"]["fresh_transition_mse"]))


if __name__ == "__main__":
    unittest.main()
