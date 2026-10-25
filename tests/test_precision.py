"""Precision regressions using exact words and tiny matrices; no lattice search."""
import unittest
from unittest.mock import patch

import numpy as np

from parity_synthesis import (
    backends, compiler, decompositions, emission, multiplexors, precision, simulation, words,
)


class PrecisionBudgetRegression(unittest.TestCase):

    def test_budget_limits_even_coherent_product_errors(self):
        budget = 8e-9
        for n in (2, 3, 4, 5):
            count = (2**n - 1) * 2**(n - 1)
            eps = precision.word_epsilon(n, 4e-9, budget)
            # All residual rotations point the same way: no random cancellation.
            residual = decompositions.Ry(2 * eps)
            product = np.linalg.matrix_power(residual, count)
            self.assertLessEqual(np.linalg.norm(product - np.eye(2), 2), budget / 2 + 1e-14)
        self.assertAlmostEqual(precision.word_epsilon(5, 4e-9, budget), 8e-9 / 992)

    def test_per_word_cap_and_budget_settings(self):
        self.assertEqual(precision.word_epsilon(5, 4e-9), 4e-9)
        self.assertEqual(precision.word_epsilon(5, 1e-12, 8e-9), 1e-12)
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                precision.word_epsilon(5, invalid)
            with self.assertRaises(ValueError):
                precision.word_epsilon(5, 4e-9, invalid)

    def test_word_check_detects_error_lost_by_trace_overlap(self):
        target = decompositions.Ry(4e-8)
        old_check = abs(abs(np.trace(target.conj().T)) / 2 - 1)
        self.assertLess(old_check, 1e-12)
        self.assertGreater(words.word_error("", target), 1e-8)
        with patch.object(backends, "pygridsynth_synth", return_value=""):
            synth = backends.make_synth("pygridsynth")
            with self.assertRaisesRegex(RuntimeError, "phase-aligned word error"):
                synth(target, 1e-9)

    def test_word_check_preserves_global_phase_freedom(self):
        target = np.exp(0.713j) * words.mat("H")
        with patch.object(backends, "pygridsynth_synth", return_value="H"):
            self.assertEqual(backends.make_synth("pygridsynth")(target, 1e-10), "H")

    def test_mux_bound_covers_measured_block_error(self):
        blocks = np.array([np.eye(2), decompositions.Ry(4e-10)], dtype=complex)
        result = multiplexors.synth_mux(blocks, lambda target, eps: "", 1e-9)
        gates, phase = emission.mux_gates(result, 1, [0])
        actual = np.exp(1j * phase) * emission.simulate(gates, 2)
        target = np.zeros((4, 4), complex)
        for x in range(2):
            target[2*x:2*x+2, 2*x:2*x+2] = (
                np.exp(-1j * result["ctrl_phase"][x]) * blocks[x]
            )
        self.assertLessEqual(np.linalg.norm(actual - target, 2), result["error_bound"] + 1e-13)
        self.assertGreater(result["error_bound"], 1e-10)

    def test_bad_factorization_is_rejected_before_synthesis(self):
        blocks = np.array([np.eye(2), decompositions.Ry(0.2)], dtype=complex)
        with patch.object(multiplexors, "factorize", return_value=(np.eye(2), [np.eye(2)])):
            with self.assertRaisesRegex(RuntimeError, "factorization matrix error"):
                multiplexors.synth_mux(blocks, lambda target, eps: self.fail("synthesis started"), 1e-9)

    def test_budget_reaches_all_words_in_a_full_small_circuit(self):
        calls = []

        def exact_identity(target, eps):
            self.assertLess(np.linalg.norm(target - np.eye(2)), 1e-12)
            calls.append(eps)
            return ""

        target = np.eye(4, dtype=complex)
        gates, phase = compiler.synthesize(target, decompositions.cs_decomposition, exact_identity,
                                    4e-9, optimize=False, error_budget=8e-9)
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(eps == precision.word_epsilon(2, 4e-9, 8e-9) for eps in calls))
        actual = simulation.circuit_matrix(gates, 2, phase)
        self.assertLess(np.max(abs(actual - target)), 1e-12)

    def test_full_circuit_budget_rejects_bad_custom_backend(self):
        with self.assertRaisesRegex(RuntimeError, "exceeds requested circuit budget"):
            compiler.synthesize(np.eye(4), decompositions.cs_decomposition, lambda target, eps: "H",
                         4e-9, optimize=False, error_budget=8e-9)




if __name__ == "__main__":
    unittest.main()
