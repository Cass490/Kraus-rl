"""Backend routing checks; native synthesis searches are explicitly opt-in."""
import unittest
import os
from unittest.mock import patch

import numpy as np

from parity_synthesis import (
    backends, pairs, parity, words,
)


class RunnerWiringRegression(unittest.TestCase):
    """Integration checks that do not execute numerical synthesis."""

    def test_paired_words_request_even_t_without_fallback(self):
        target = np.eye(2, dtype=complex)
        with patch.object(backends, "cyclosynth_synth", return_value="") as native:
            with patch.object(backends, "pygridsynth_synth", side_effect=AssertionError("fallback used")):
                synth = backends.make_synth("cyclosynth")
                self.assertEqual(backends.even_word(target, synth, 1e-8), "")
        native.assert_called_once()
        self.assertIs(native.call_args.args[0], target)
        self.assertEqual(native.call_args.args[1], 1e-8)
        self.assertEqual(native.call_args.kwargs, {"even_t": True})


@unittest.skipUnless(os.environ.get("PARITY_SYNTHESIS_NATIVE_TESTS") == "1",
                     "native searches are opt-in; set PARITY_SYNTHESIS_NATIVE_TESTS=1")
class SynthesisRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            backends.require_cyclosynth()
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        if not hasattr(backends.require_cyclosynth().Synthesizer, "even_t"):
            raise RuntimeError("Rebuild cyclosynth with even_t support before running these regressions")

    def assert_approximates(self, word, target, eps):
        matrix = words.mat(word)
        phase = np.angle(np.trace(target.conj().T @ matrix))
        self.assertLess(np.linalg.norm(np.exp(-1j * phase) * matrix - target, 2), eps)

    def test_joint_u3_angle_convention(self):
        target = parity.su2_exp(np.array([0.21, -0.37, 0.49]))
        eps = 1e-3
        self.assert_approximates(backends.cyclosynth_synth(target, eps), target, eps)

    def test_python_even_t_api(self):
        cls = backends.require_cyclosynth().Synthesizer
        self.assertFalse(cls(1e-3).even_t)
        self.assertTrue(cls(1e-3, even_t=True).even_t)
        with self.assertRaisesRegex(ValueError, "even_t requires Clifford"):
            cls(1e-3, even_t=True, sqrt_t=True)

    def test_generic_even_t_does_not_use_pygridsynth(self):
        target = parity.su2_exp(np.array([0.21, -0.37, 0.49]))
        synth = backends.make_synth("cyclosynth")
        with patch.object(backends, "pygridsynth_synth", side_effect=AssertionError("fallback used")):
            word = backends.even_word(target, synth, 1e-3)
        self.assertEqual(words.tcount(word) % 2, 0)
        self.assert_approximates(word, target, 1e-3)

    def test_odd_exact_rotations_and_pair(self):
        synth = backends.make_synth("cyclosynth")
        eps = 4e-9
        for exact_word in ("T", "HTH", "SHTHs"):
            with self.subTest(axis_word=exact_word):
                target = words.mat(exact_word)
                target /= np.sqrt(np.linalg.det(target))
                self.assertEqual(words.tcount(synth(target, eps)) % 2, 1)
                with patch.object(backends, "pygridsynth_synth", side_effect=AssertionError("fallback used")):
                    word = backends.even_word(target, synth, eps)
                self.assertEqual(words.tcount(word) % 2, 0)
                self.assert_approximates(word, target, eps)
                if exact_word == "T":
                    pair = pairs.compile_pair_best(word)
                    a = words.mat(pair["A"])
                    expected = np.zeros((4, 4), dtype=complex)
                    expected[:2, :2] = a
                    expected[2:, 2:] = a @ words.mat(word)
                    self.assertLess(np.max(np.abs(pairs.pair_matrix(pair) - expected)), 1e-11)


if __name__ == "__main__":
    unittest.main()
