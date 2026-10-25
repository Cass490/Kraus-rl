"""Public matrix-to-circuit API, independent of benchmark infrastructure."""

from collections import Counter
from dataclasses import dataclass
import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np

from .backends import make_synth, require_cyclosynth
from .cache import CircuitCache
from .compiler import optimize_circuit, synthesize
from .decompositions import cs_decomposition
from .emission import queue_ops, word_gates, word_phase
from .precision import word_epsilon
from .simulation import circuit_matrix, max_entrywise_error
from .words import phase_of


@dataclass(frozen=True)
class CompiledCircuit:
    """A time-ordered gate sequence and its explicitly retained scalar phase."""

    num_qubits: int
    gates: tuple
    global_phase: float
    elapsed_seconds: float
    reused_checkpoint: bool = False

    def matrix(self) -> np.ndarray:
        """Evaluate the circuit matrix; practical for the small sizes studied."""
        return circuit_matrix(self.gates, self.num_qubits, self.global_phase)

    def error_against(self, target) -> float:
        return max_entrywise_error(self.matrix(), target)

    def queue(self) -> None:
        """Queue gates in an active PennyLane recording context (optional extra)."""
        import pennylane as qml
        queue_ops(self.gates, self.global_phase, qml)

    def statistics(self, rotation_weight: float = 44.0) -> dict:
        """Report physical counts and a configurable weighted resource metric."""
        if not np.isfinite(rotation_weight) or rotation_weight < 0:
            raise ValueError("rotation_weight must be finite and nonnegative")
        counts = Counter(gate[0] for gate in self.gates)
        rotations = sum(counts[name] for name in ("RX", "RY", "RZ"))
        return {
            "num_qubits": self.num_qubits,
            "gate_counts": dict(sorted(counts.items())),
            "physical_gate_count": len(self.gates) + 1,
            "global_phase_operations": 1,
            "T_count": counts["T"],
            "rotation_count": rotations,
            "rotation_weight": rotation_weight,
            "weighted_cost": counts["T"] + rotation_weight * rotations,
            "elapsed_seconds": self.elapsed_seconds,
            "reused_checkpoint": self.reused_checkpoint,
        }

    def to_dict(self) -> dict:
        """Return JSON-serializable circuit data, including its scalar phase."""
        return {
            "num_qubits": self.num_qubits,
            "gates": [list(gate) for gate in self.gates],
            "global_phase": self.global_phase,
        }


def backend_fingerprint(kind: str) -> str:
    """Identify the installed backend without starting a synthesis search."""
    if kind == "pygridsynth":
        return version("pygridsynth")
    folder = Path(require_cyclosynth().__file__).parent
    binaries = sorted(path for path in folder.iterdir()
                      if path.name.endswith(tuple(EXTENSION_SUFFIXES)))
    if not binaries:
        raise RuntimeError("cannot fingerprint the installed cyclosynth extension")
    return hashlib.sha256(b"".join(path.read_bytes() for path in binaries)).hexdigest()


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compile_unitary(unitary, *, epsilon=1e-9, backend="cyclosynth",
                    method="phase11", fold_common=True, error_budget=None,
                    cache_dir=None, cache_only=False, verbose=False) -> CompiledCircuit:
    """Compile a finite unitary matrix into gates, preserving its global phase.

    epsilon is a per-word cap, not a guarantee on final circuit error. A
    callable backend(target, epsilon) may be supplied for controlled experiments.
    Persistent checkpoints are available only for named, fingerprinted backends.
    """
    started = time.monotonic()
    matrix = np.ascontiguousarray(unitary, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("unitary must be a square matrix")
    dimension = len(matrix)
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("matrix dimension must be a power of two, at least two")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("unitary must contain finite entries")
    if np.linalg.norm(matrix.conj().T @ matrix - np.eye(dimension), 2) > 1e-10:
        raise ValueError("matrix is not unitary to numerical tolerance 1e-10")
    num_qubits = dimension.bit_length() - 1
    epsilon = word_epsilon(num_qubits, epsilon, error_budget)
    if method not in ("baseline", "phase11"):
        raise ValueError("method must be baseline or phase11")
    if isinstance(backend, str):
        if backend not in ("cyclosynth", "pygridsynth"):
            raise ValueError("unknown synthesis backend")
    elif not callable(backend):
        raise TypeError("backend must be a supported name or callable")
    if cache_only and cache_dir is None:
        raise ValueError("cache_only requires cache_dir")
    if cache_dir is not None and not isinstance(backend, str):
        raise ValueError("persistent caching requires a named backend")

    cache = None
    saved = None
    if cache_dir is not None:
        signature = {
            "format": "parity-synthesis-v1",
            "source": _source_fingerprint(),
            "backend": backend,
            "backend_fingerprint": backend_fingerprint(backend),
            "epsilon": epsilon,
            "method": method,
            "fold_common": bool(fold_common),
            "error_budget": error_budget,
        }
        cache = CircuitCache(cache_dir, signature)
        saved = cache.load(matrix)
    reused = saved is not None
    if saved is None:
        if cache_only:
            raise RuntimeError("No checkpoint; cache_only prevented a new synthesis run")
        synthesizer = make_synth(backend) if isinstance(backend, str) else backend
        if num_qubits == 1:
            word = synthesizer(matrix, epsilon)
            gates = word_gates(word.replace("I", ""), 0)
            phase = word_phase(word) - phase_of(word, matrix)
        else:
            gates, phase = synthesize(
                matrix, cs_decomposition, synthesizer, epsilon,
                verbose=verbose, optimize=False, error_budget=error_budget,
                mux_method=method, fold_common=fold_common,
            )
        if cache is not None:
            cache.save(matrix, gates, phase)
    else:
        gates, phase = saved
    gates, phase = optimize_circuit(gates, phase, verbose=verbose)
    return CompiledCircuit(num_qubits, tuple(gates), float(phase),
                           time.monotonic() - started, reused)
