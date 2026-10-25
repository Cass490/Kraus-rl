"""Explore the continuous 11-mask model without a synthesis backend."""

import numpy as np

from parity_synthesis.phase_factorization import factorize_phase, reconstruct
from parity_synthesis.targets import haar_unitary


def main():
    blocks = []
    for seed in range(16):
        matrix = haar_unitary(1, seed)
        blocks.append(matrix / np.sqrt(np.linalg.det(matrix)))
    common, betas, phases, error = factorize_phase(np.array(blocks))
    print("Parity factors:", len(betas))
    print("Maximum block operator error:", error)
    print("Maximum entry residual:", np.max(np.abs(reconstruct(common, betas, phases) - blocks)))


if __name__ == "__main__":
    main()
