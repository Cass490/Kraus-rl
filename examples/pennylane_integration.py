"""Optional PennyLane recording of a compiled circuit; run explicitly."""

import pennylane as qml

from parity_synthesis import compile_unitary
from parity_synthesis.targets import haar_unitary


def main():
    circuit = compile_unitary(haar_unitary(2, 7), cache_dir=".cache/parity-synthesis")
    tape = qml.tape.make_qscript(circuit.queue)()
    print("Queued operations:", len(tape.operations))
    print("Wires:", tape.wires)


if __name__ == "__main__":
    main()
