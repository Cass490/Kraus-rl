"""Time-ordered gate emission and PennyLane integration."""

import numpy as np

from .words import GATE


def word_gates(word, w):
    """Time-ordered gates for a matrix-order word on wire w (t,s expanded into T/S/Z)."""
    out = []
    for g in reversed(word):
        if g == "t": out += [("T", (w,)), ("S", (w,)), ("Z", (w,))]
        elif g == "s": out += [("S", (w,)), ("Z", (w,))]
        elif g == "W": pass                       # global phase handled by caller
        else: out.append((g, (w,)))
    return out


def word_phase(word):
    return np.pi / 4 * word.count("W")


def mux_gates(res, target, controls):
    """controls: list of wires, big-endian w.r.t. mask bits. Returns (gates time order, global phase rad)."""
    c = len(controls); gates = []
    gph = res["global"] + word_phase(res["C"])
    for mask, pg in res["pairs"]:
        mem = [controls[i] for i in range(c) if (mask >> (c - 1 - i)) & 1]
        pw = mem[-1]
        ladder = [("CNOT", (o, pw)) for o in mem[:-1]]
        gates += ladder
        for name, ws in pg:
            gates.append((name, tuple(pw if x == "c" else target for x in ws)))
        gates += ladder[::-1]
    gates += word_gates(res["C"], target)
    return gates, gph


def simulate(gates, n):
    """Full 2^n matrix of a gate list (big-endian wires 0..n-1)."""
    N = 2 ** n; M = np.eye(N, dtype=complex)
    for name, ws in gates:
        if name == "CNOT":
            a, b = ws; G = np.zeros((N, N))
            for i in range(N):
                j = i ^ (1 << (n - 1 - b)) if (i >> (n - 1 - a)) & 1 else i
                G[j, i] = 1
        else:
            (w,) = ws; ops = [np.eye(2)] * n; ops = list(ops); ops[w] = GATE[name]
            G = ops[0]
            for o in ops[1:]: G = np.kron(G, o)
        M = G @ M
    return M


def queue_ops(gates, gph, qp):
    for gt in gates:
        name, ws = gt[0], gt[1]
        if name == "CNOT": qp.CNOT(wires=list(ws))
        elif name == "RZ": qp.RZ(gt[2], wires=ws[0])
        elif name == "H": qp.Hadamard(wires=ws[0])
        elif name == "S": qp.S(wires=ws[0])
        elif name == "T": qp.T(wires=ws[0])
        elif name == "X": qp.PauliX(wires=ws[0])
        elif name == "Y": qp.PauliY(wires=ws[0])
        elif name == "Z": qp.PauliZ(wires=ws[0])
        else: raise ValueError(name)
    qp.GlobalPhase(-gph)
