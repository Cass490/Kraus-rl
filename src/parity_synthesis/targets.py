"""Seeded matrix generators used by examples and reproducible experiments."""

import numpy as np


def haar_unitary(num_qubits: int, seed: int) -> np.ndarray:
    """Generate a Haar-random unitary using complex Gaussian QR."""
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    rng = np.random.default_rng(seed)
    dimension = 2 ** num_qubits
    matrix = (rng.normal(size=(dimension, dimension))
              + 1j * rng.normal(size=(dimension, dimension))) / np.sqrt(2)
    q, r = np.linalg.qr(matrix)
    diagonal = np.diag(r)
    return np.asarray(q * (diagonal / np.abs(diagonal)), dtype=complex)
