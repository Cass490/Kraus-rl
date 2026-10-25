"""Compile a small target using the native backend; run explicitly."""

from parity_synthesis import compile_unitary
from parity_synthesis.targets import haar_unitary


def main():
    target = haar_unitary(num_qubits=2, seed=7)
    circuit = compile_unitary(target, epsilon=1e-9, cache_dir=".cache/parity-synthesis")
    print(circuit.statistics())
    print("Maximum entrywise error:", circuit.error_against(target))


if __name__ == "__main__":
    main()
