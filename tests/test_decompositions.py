"""Numerical decomposition and experiment-generation checks."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from parity_synthesis.benchmark import load_cases
from parity_synthesis import CompiledCircuit
from parity_synthesis import benchmark
from parity_synthesis.decompositions import cs_decomposition
from parity_synthesis.targets import haar_unitary


class DecompositionRegression(unittest.TestCase):
    def test_csd_reconstructs_seeded_and_degenerate_targets(self):
        matrices = [np.eye(8, dtype=complex)]
        matrices += [haar_unitary(n, 720+n) for n in (1, 2, 3, 4, 5)]
        for matrix in matrices:
            with self.subTest(dimension=len(matrix)):
                left0, left1, theta, right0, right1 = cs_decomposition(matrix)
                zero = np.zeros_like(left0)
                left = np.block([[left0, zero], [zero, left1]])
                right = np.block([[right0, zero], [zero, right1]])
                cosine, sine = np.diag(np.cos(theta)), np.diag(np.sin(theta))
                middle = np.block([[cosine, -sine], [sine, cosine]])
                self.assertLess(np.max(np.abs(left @ middle @ right - matrix)), 1e-12)

    def test_seed_groups_define_sixteen_distinct_unitaries(self):
        root = Path(__file__).resolve().parents[1]
        cases = load_cases(root / "benchmarks/cases.json")
        fingerprints = set()
        for group_seed in cases["seed_groups"]:
            for n in cases["sizes"]:
                for sample in range(cases["samples_per_size"]):
                    target = haar_unitary(n, group_seed + 1000*n + sample)
                    self.assertLess(np.linalg.norm(target.conj().T @ target - np.eye(2**n)), 1e-12)
                    fingerprints.add(target.tobytes())
        self.assertEqual(len(fingerprints), 16)

    def test_experiment_aggregation_uses_distinct_targets_and_sample_means(self):
        config = {
            "seed_groups": [2026, 7], "sizes": [2], "samples_per_size": 2,
            "reference_rotation_weight": 44, "reference_costs": {"2": 660},
        }
        seen = []

        def recording_backend(target, **kwargs):
            seen.append(target.copy())
            return CompiledCircuit(2, tuple(("T", (0,)) for _ in range(len(seen))), 0.0, 0.0)

        with patch.object(benchmark, "compile_unitary", side_effect=recording_backend):
            with patch.object(CompiledCircuit, "error_against", return_value=1e-9):
                report = benchmark.run_benchmark(config)
        self.assertEqual(len(seen), 4)
        expected_seeds = (4026, 4027, 2007, 2008)
        for actual, seed in zip(seen, expected_seeds):
            np.testing.assert_array_equal(actual, haar_unitary(2, seed))
        self.assertEqual(report["cases"][0]["mean_weighted_cost_by_size"], {"2": 1.5})
        self.assertEqual(report["cases"][1]["mean_weighted_cost_by_size"], {"2": 3.5})
        self.assertTrue(all(case["within_requested_tolerance"] for case in report["cases"]))
