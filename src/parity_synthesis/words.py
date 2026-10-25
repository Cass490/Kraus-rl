"""Exact Clifford+T word algebra, phase conventions, and error measurement."""

import numpy as np


OM = np.exp(1j * np.pi / 4)


GATE = {
    "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "S": np.diag([1, 1j]), "s": np.diag([1, -1j]),
    "T": np.diag([1, OM]), "t": np.diag([1, np.conj(OM)]),
    "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]),
    "W": np.eye(2) * OM, "I": np.eye(2),
}


INV = {"H": "H", "S": "s", "s": "S", "T": "t", "t": "T", "X": "X", "Y": "Y", "Z": "Z", "I": "I"}


def mat(word):
    M = np.eye(2, dtype=complex)
    for g in word:
        M = M @ GATE[g]
    return M


def inverse(word):
    """Inverse word (exact up to global phase W^-k, tracked separately by callers via skeleton phase)."""
    out = []
    for g in reversed(word):
        if g == "W":
            out.append("WWWWWWW")
        else:
            out.append(INV[g])
    return "".join(out)


def tcount(word):
    return sum(g in "Tt" for g in word)


def skeleton(word):
    """Reduce a word to  phase * T^e0 H T^e1 H ... H T^er  (e_i mod 8).
    Returns (exps, phase_k) with global phase OM**phase_k.  Exact identities used:
    S=T^2, Z=T^4, s=T^6, t=T^7, X = H T^4 H, Y = i X Z = OM^2 * H T^4 H T^4, W = OM."""
    toks = []      # list of ints (T powers) and 'H'
    ph = 0
    for g in word:
        if g == "H": toks.append("H")
        elif g == "T": toks.append(1)
        elif g == "t": toks.append(7)
        elif g == "S": toks.append(2)
        elif g == "s": toks.append(6)
        elif g == "Z": toks.append(4)
        elif g == "X": toks += ["H", 4, "H"]
        elif g == "Y": toks += ["H", 4, "H", 4]; ph += 2
        elif g == "W": ph += 1
        elif g == "I": pass
        else: raise ValueError(g)
    # normalise into alternating exps: e0 H e1 H ... (merge powers, cancel H H)
    exps = [0]
    for tk in toks:
        if tk == "H":
            if len(exps) >= 2 and exps[-1] % 8 == 0:
                exps.pop()               # ... H T^0 H  ->  ... (H H = I)
            else:
                exps.append(0)
        else:
            exps[-1] = (exps[-1] + tk) % 8
    return [e % 8 for e in exps], ph % 8


def skeleton_word(exps, ph=0):
    """Word realising OM^ph * T^e0 H T^e1 H ... H T^er using only H,S,T,Z (T^k as T/S/Z combos)."""
    def tp(k):
        k %= 8
        return ("T" if k & 1 else "") + ("S" if k & 2 else "") + ("Z" if k & 4 else "")
    w = "W" * (ph % 8)
    for i, e in enumerate(exps):
        if i: w += "H"
        w += tp(e)
    return w


def to_basic(word):
    """Rewrite into pygridsynth's alphabet {H,T,S,X,W} exactly."""
    rep = {"s": "SSS", "t": "TTTTTTT", "Z": "SS", "Y": "WWXSS", "I": ""}
    return "".join(rep.get(g, g) for g in word)


def ma_normal_form(word):
    """Exact Matsumoto-Amano normal form (as a word over H,T,S,X,W) via pygridsynth exact synthesis."""
    from pygridsynth.domega_unitary import DOmegaUnitary
    from pygridsynth.synthesis_of_cliffordT import decompose_domega_unitary
    u = DOmegaUnitary.from_gates(to_basic(word))
    return decompose_domega_unitary(u, wires=[0], up_to_phase=False).to_simple_str()


def phase_of(word, U):
    """gamma with mat(word) ~ exp(i gamma) U."""
    return float(np.angle(np.trace(U.conj().T @ mat(word))))


def word_error(word, U):
    """Phase-aligned operator-norm error, stable for errors near 1e-8.

    The trace overlap alone loses small errors quadratically and can round to
    exactly one even when the matrix error exceeds the requested tolerance.
    """
    M = mat(word)
    gamma = np.angle(np.trace(U.conj().T @ M))
    return float(np.linalg.norm(np.exp(-1j * gamma) * M - U, 2))
