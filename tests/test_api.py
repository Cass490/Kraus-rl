"""Public API, simulator, caching, and CLI checks without lattice searches."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from parity_synthesis import (
    CompiledCircuit, compile_unitary,
)
from parity_synthesis import (
    api, cli,
)
from parity_synthesis.simulation import circuit_matrix
from parity_synthesis.words import GATE


class PublicAPIRegression(unittest.TestCase):
    def test_invalid_input_is_rejected_before_backend_construction(self):
        bad = (np.ones((2, 3)), np.eye(3), np.zeros((4, 4)),
               np.full((4, 4), np.nan), np.eye(1))
        with patch.object(api, "make_synth") as backend:
            for matrix in bad:
                with self.subTest(shape=matrix.shape), self.assertRaises(ValueError):
                    compile_unitary(matrix)
        backend.assert_not_called()

    def test_single_qubit_global_phase_is_retained(self):
        target = np.exp(0.37j) * GATE["H"]
        circuit = compile_unitary(target, backend=lambda target, eps: "H")
        self.assertLess(circuit.error_against(target), 1e-13)
        self.assertEqual(circuit.num_qubits, 1)

    def test_identity_compiles_with_a_controlled_exact_backend(self):
        calls = []

        def exact_identity(target, eps):
            self.assertLess(np.linalg.norm(target - np.eye(2)), 1e-12)
            calls.append(eps)
            return ""

        circuit = compile_unitary(np.eye(4), backend=exact_identity)
        self.assertEqual(len(calls), 6)
        self.assertLess(circuit.error_against(np.eye(4)), 1e-12)
        self.assertEqual(circuit.statistics()["weighted_cost"], 0)

    def test_checkpoint_hit_does_not_construct_a_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(api, "backend_fingerprint", return_value="test-backend"):
                with patch.object(api, "make_synth", return_value=lambda target, eps: ""):
                    first = compile_unitary(np.eye(4), backend="pygridsynth", cache_dir=directory)
                self.assertFalse(first.reused_checkpoint)
                with patch.object(api, "make_synth", side_effect=AssertionError("search started")):
                    second = compile_unitary(np.eye(4), backend="pygridsynth",
                                             cache_dir=directory, cache_only=True)
                self.assertTrue(second.reused_checkpoint)
                self.assertEqual(first.gates, second.gates)
                self.assertEqual(first.global_phase, second.global_phase)
                with self.assertRaisesRegex(RuntimeError, "prevented a new synthesis run"):
                    compile_unitary(np.eye(4), backend="pygridsynth", epsilon=2e-9,
                                    cache_dir=directory, cache_only=True)

    def test_unsafe_cache_configurations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires cache_dir"):
            compile_unitary(np.eye(4), cache_only=True)
        with self.assertRaisesRegex(ValueError, "requires a named backend"):
            compile_unitary(np.eye(4), backend=lambda target, eps: "", cache_dir="unused")

    def test_statistics_distinguish_physical_gates_from_weighted_cost(self):
        gates = (("H", (0,)), ("T", (0,)), ("CNOT", (0, 1)), ("RZ", (1,), 0.2))
        circuit = CompiledCircuit(2, gates, 0.3, 0.1)
        stats = circuit.statistics()
        self.assertEqual(stats["weighted_cost"], 45)
        self.assertEqual(stats["physical_gate_count"], 5)
        self.assertEqual(stats["rotation_count"], 1)
        self.assertEqual(json.loads(json.dumps(circuit.to_dict()))["global_phase"], 0.3)

    def test_dry_run_avoids_compilation(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(cli, "compile_unitary") as compiler:
            status = cli.main(["compile", "--qubits", "5", "--dry-run", "--error-budget", "8e-9"])
        compiler.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(plan["planned_words"], 376)
        self.assertEqual(plan["fallback_maximum_words"], 496)
        self.assertAlmostEqual(plan["per_word_epsilon"], 8e-9 / 992)

    @unittest.skipUnless(importlib.util.find_spec("pennylane"), "PennyLane extra not installed")
    def test_pennylane_queue_matches_the_direct_matrix(self):
        import pennylane as qml
        circuit = CompiledCircuit(2, (("H", (0,)), ("CNOT", (0, 1)),
                                     ("T", (1,)), ("RZ", (0,), -0.3)), 0.17, 0.0)
        tape = qml.tape.make_qscript(circuit.queue)()
        actual = qml.matrix(tape, wire_order=[0, 1])
        self.assertLess(np.max(np.abs(actual - circuit.matrix())), 1e-13)


class MatrixSimulationRegression(unittest.TestCase):
    def test_optimized_simulator_matches_independent_dense_products(self):
        rng = np.random.default_rng(838)
        num_qubits = 3
        gates = []
        expected = np.eye(8, dtype=complex)
        for _ in range(40):
            name = str(rng.choice(["H", "S", "T", "X", "Y", "Z", "RZ", "CNOT"]))
            if name == "CNOT":
                control, target = map(int, rng.choice(3, size=2, replace=False))
                gate = (name, (control, target))
                full = np.zeros((8, 8), complex)
                for column in range(8):
                    row = column ^ (1 << (2-target)) if column & (1 << (2-control)) else column
                    full[row, column] = 1
            else:
                wire = int(rng.integers(3))
                if name == "RZ":
                    angle = float(rng.normal())
                    gate = (name, (wire,), angle)
                    local = np.diag(np.exp(np.array([-0.5j, 0.5j]) * angle))
                else:
                    gate, local = (name, (wire,)), GATE[name]
                full = np.array([[1]], complex)
                for selected in range(num_qubits):
                    full = np.kron(full, local if selected == wire else np.eye(2))
            gates.append(gate)
            expected = full @ expected
        self.assertLess(np.max(np.abs(circuit_matrix(gates, 3, 0.37)
                                      - np.exp(0.37j) * expected)), 1e-12)
