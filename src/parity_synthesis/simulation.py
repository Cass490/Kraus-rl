"""Direct matrix evaluation of time-ordered circuits, without a framework tape."""

import numpy as np

from .words import GATE


def circuit_matrix(gates, num_qubits: int, global_phase: float = 0.0):
    """Evaluate a small circuit; wire 0 is the most significant tensor factor.

    Consecutive one-qubit gates are accumulated per wire. CNOTs swap slices
    directly, avoiding construction of a dense matrix for each gate.
    This is deterministic matrix simulation, not shot-based execution.
    """
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    dimension = 2 ** num_qubits
    states = np.eye(dimension, dtype=complex).reshape((2,) * num_qubits + (dimension,))
    pending = {}

    def index(fixed):
        selection = [slice(None)] * (num_qubits + 1)
        for wire, bit in fixed.items():
            selection[wire] = bit
        return tuple(selection)

    def flush(wire):
        local = pending.pop(wire, None)
        if local is None:
            return
        zero = states[index({wire: 0})]
        one = states[index({wire: 1})]
        new_zero = local[0, 0] * zero + local[0, 1] * one
        one *= local[1, 1]
        one += local[1, 0] * zero
        zero[...] = new_zero

    for gate in gates:
        name, wires = gate[:2]
        if any(not 0 <= wire < num_qubits for wire in wires):
            raise ValueError("gate wire is outside the circuit")
        if name == "CNOT":
            if len(wires) != 2 or wires[0] == wires[1]:
                raise ValueError("CNOT needs two distinct wires")
            control, target = wires
            flush(control)
            flush(target)
            low = index({control: 1, target: 0})
            high = index({control: 1, target: 1})
            temporary = states[low].copy()
            states[low] = states[high]
            states[high] = temporary
        else:
            if len(wires) != 1:
                raise ValueError("single-qubit gate needs one wire")
            wire = wires[0]
            if name == "RZ":
                local = np.diag(np.exp(np.array([-0.5j, 0.5j]) * gate[2]))
            elif name in GATE:
                local = GATE[name]
            else:
                raise ValueError(f"unsupported gate: {name}")
            pending[wire] = local @ pending[wire] if wire in pending else local
    for wire in list(pending):
        flush(wire)
    return np.exp(1j * global_phase) * states.reshape(dimension, dimension)


def max_entrywise_error(actual, target) -> float:
    """Largest absolute matrix-entry difference, with global phase included."""
    actual, target = np.asarray(actual), np.asarray(target)
    if actual.shape != target.shape:
        raise ValueError("matrix shapes differ")
    return float(np.max(np.abs(actual - target)))
