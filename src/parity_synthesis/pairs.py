"""Dynamic-programming compilation of parity-compatible gate-word pairs."""

import functools
import itertools as _it
import sys

import numpy as np

from .words import GATE, inverse, mat, ma_normal_form, skeleton, skeleton_word
from .emission import word_gates


V_CH = "SHT"


_CLIFF = []


def dcost(x, y):
    k1 = ((x + y) // 2) % 8
    d = ((y - x) // 2) % 8
    return (k1 & 1) + (d & 1)


def diagonal_params(x, y):
    """Equivalent half-angle lifts; a control-only phase is deferred by callers."""
    a = ((x + y) // 2) % 8
    d = ((y - x) // 2) % 8
    def gate_count(pair):
        k, delta = pair
        return k.bit_count() + (2 + ((-delta) % 8).bit_count() if delta else 0)
    return min(((a, d), ((a + 4) % 8, (d + 4) % 8)), key=gate_count)


def diagonal_gate_count(x, y):
    a, d = diagonal_params(x, y)
    return a.bit_count() + (2 + ((-d) % 8).bit_count() if d else 0)


def align(a, b, band=2):
    """DP alignment of skeleton segments a (word A) and b (word B), matrix order.
    Returns (cost, ops) with ops in matrix order: ('D',x,y) | ('H',) | ('HA',) (H only in A) | ('HB',)."""
    N, M = len(a) - 1, len(b) - 1
    INF = 10 ** 9
    sys.setrecursionlimit(10000)
    @functools.lru_cache(maxsize=None)
    def best(i, j, ra, rb):
        if abs(i - j) > band:
            return INF, INF, ()
        res = (INF, INF, ())
        if i == N and j == M:
            if (ra - rb) % 2 == 0:
                return dcost(ra, rb), diagonal_gate_count(ra, rb), (("D", ra, rb),)
            return res
        if i < N and j < M and (ra - rb) % 2 == 0:
            c, ng, ops = best(i + 1, j + 1, a[i + 1], b[j + 1])
            c += dcost(ra, rb)
            ng += diagonal_gate_count(ra, rb) + 1
            if (c, ng) < res[:2]:
                res = (c, ng, (("D", ra, rb), ("H",)) + ops)
        if i < N:
            for x in range(ra % 2, 8, 2):
                c, ng, ops = best(i + 1, j, a[i + 1], (rb - x) % 8)
                c += dcost(ra, x) + 2
                ng += diagonal_gate_count(ra, x) + 12
                if (c, ng) < res[:2]:
                    res = (c, ng, (("D", ra, x), ("HA",)) + ops)
        if j < M:
            for x in range(rb % 2, 8, 2):
                c, ng, ops = best(i, j + 1, (ra - x) % 8, b[j + 1])
                c += dcost(x, rb) + 2
                ng += diagonal_gate_count(x, rb) + 10
                if (c, ng) < res[:2]:
                    res = (c, ng, (("D", x, rb), ("HB",)) + ops)
        return res
    out = best(0, 0, a[0], b[0])
    best.cache_clear()
    return out[0], out[2]


def split_options(word, max_pos=3):
    """Ways to write word = X Y at a T-exponent position, giving (A,B) = (X^-1, Y) skeletons.
    Only near-balanced split points (equal T in both halves) and one exponent split per parity class."""
    f, ph = skeleton(word)
    nodd = sum(e & 1 for e in f)
    cands = []
    for j in range(len(f)):
        left_odd = sum(e & 1 for e in f[:j])
        for alpha in range(8):
            beta = (f[j] - alpha) % 8
            ta = left_odd + (alpha & 1); tb = nodd - left_odd - (f[j] & 1) + (beta & 1)
            if ta != tb:
                continue
            cands.append((abs(j - len(f) / 2), j, alpha))
    cands.sort()
    opts, seen = [], set()
    for _, j, alpha in cands:
        key = (j, alpha & 1, ((f[j] - alpha) % 8) & 1)
        if key in seen: continue
        seen.add(key)
        a = [(-alpha) % 8] + [(-e) % 8 for e in reversed(f[:j])]
        b = [(f[j] - alpha) % 8] + f[j + 1:]
        opts.append((a, b, ph))
        if len(opts) >= max_pos * 2: break
    return opts


def _emit_pair_ops(ops, ctrl_phase=0):
    """Emit a skeleton alignment, deferring its exact control phase."""
    gates = []            # time order list of (name, wires)
    def tpow(k, w):
        k %= 8
        out = []
        if k & 1: out.append(("T", (w,)))
        if k & 2: out.append(("S", (w,)))
        if k & 4: out.append(("Z", (w,)))
        return out
    def word_gates(wd, w):   # time order gates for word wd (matrix order string)
        out = []
        for g in reversed(wd):
            if g == "t": out += tpow(7, w)
            elif g == "s": out += tpow(6, w)
            else: out.append((g, (w,)))
        return out
    for op in reversed(ops):              # time order = reverse matrix order
        if op[0] == "D":
            x, y = op[1], op[2]
            k1, d = diagonal_params(x, y)
            gates += tpow(k1, "q")
            if d % 8:
                gates += [("CNOT", ("c", "q"))] + tpow(-d, "q") + [("CNOT", ("c", "q"))]
            ctrl_phase += d
        elif op[0] == "H":
            gates.append(("H", ("q",)))
        else:
            ch = word_gates(inverse(V_CH), "q") + [("CNOT", ("c", "q"))] + word_gates(V_CH, "q")
            if op[0] == "HA":                 # H only when parity = 0
                gates += [("X", ("c",))] + ch + [("X", ("c",))]
            else:
                gates += ch
    tc = sum(g[0] == "T" for g in gates)
    return {"gates": gates, "t": tc, "ctrl_phase": ctrl_phase % 8}


def compile_pair(word, band=2):
    """Best split pair with A^-1 B == word; retain exact deferred phases."""
    best = None
    for a, b, phB in split_options(word):
        cost, ops = align(a, b, band)
        ng = sum(diagonal_gate_count(op[1], op[2]) if op[0] == "D"
                 else {"H": 1, "HA": 12, "HB": 10}[op[0]] for op in ops)
        if best is None or (cost, ng) < best[:2]:
            best = (cost, ng, ops, a, phB)
    if best is None or best[0] >= 10 ** 9:
        raise RuntimeError("no parity-compatible split-word alignment")
    _, _, ops, a, phB = best
    result = _emit_pair_ops(ops, phB)
    result["A"] = skeleton_word(a)
    return result


def compile_word_pair(word0, word1, band=2):
    """Compile two full words; P(w0,w1)=exp(i global) D(ctrl) raw gates."""
    a, ph0 = skeleton(word0)
    b, ph1 = skeleton(word1)
    if sum(e & 1 for e in a) % 2 != sum(e & 1 for e in b) % 2:
        raise ValueError("full paired words must have matching T parity")
    band = max(band, abs(len(a) - len(b)) + 1)
    cost, ops = align(a, b, band)
    if cost >= 10 ** 9:
        cost, ops = align(a, b, max(len(a), len(b)))
    if cost >= 10 ** 9:
        raise RuntimeError("no full-word pair alignment")
    result = _emit_pair_ops(ops, ph1 - ph0)
    result["global"] = ph0 * np.pi / 4
    return result


def pair_matrix(res):
    """4x4 matrix (c,q) of the compiled gates including the deferred control phase."""
    I = np.eye(2); M = np.eye(4, dtype=complex)
    CN = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], complex)
    for name, w in res["gates"]:
        if name == "CNOT": G = CN
        elif w == ("c",): G = np.kron(GATE[name], I)
        else: G = np.kron(I, GATE[name])
        M = G @ M
    ph = np.exp(1j * np.pi / 4 * res["ctrl_phase"])
    return np.exp(1j * res.get("global", 0.0)) * np.kron(np.diag([1, ph]), I) @ M


def cliffords():
    if not _CLIFF:
        mats = []
        for L in range(0, 6):
            for w in _it.product("HS", repeat=L):
                w = "".join(w); M = mat(w)
                if not any(abs(abs(np.trace(M.conj().T @ m)) - 2) < 1e-9 for m in mats):
                    _CLIFF.append(w); mats.append(M)
    return _CLIFF


def compile_pair_best(word):
    """Best over the 24 Clifford conjugations. Returns res with 'gates' (time order, incl. the free
    uncontrolled K^-1 applied first), 't', 'ctrl_phase', and 'A' (= A' K^-1) such that the gates realise
    P(A, A word) up to the returned control phase."""
    best = None
    for K in cliffords():
        Kinv = inverse(K)
        w2 = ma_normal_form(Kinv + word + K)
        res = compile_pair(w2)
        pre_count = sum(2 if gate == "s" else 1 for gate in Kinv)
        if best is None or (res["t"], len(res["gates"]) + pre_count) < best[2]:
            best = (res, Kinv, (res["t"], len(res["gates"]) + pre_count))
    res, Kinv, _ = best
    pre = []
    for g in reversed(Kinv):
        if g == "s": pre += [("S", ("q",)), ("Z", ("q",))]
        else: pre.append((g, ("q",)))
    return {"gates": pre + res["gates"], "t": res["t"], "ctrl_phase": res["ctrl_phase"], "A": res["A"] + Kinv}


def compile_word_pair_best(word0, word1):
    """Search simultaneous Clifford conjugations of two full words."""
    best = None
    for K in cliffords():
        Kinv = inverse(K)
        result = compile_word_pair(ma_normal_form(Kinv + word0 + K),
                                   ma_normal_form(Kinv + word1 + K))
        pre = [(name, ("q",)) for name, _ in word_gates(Kinv, 0)]
        post = [(name, ("q",)) for name, _ in word_gates(K, 0)]
        result["gates"] = pre + result["gates"] + post
        score = result["t"], len(result["gates"])
        if best is None or score < best[0]:
            best = score, result
    return best[1]
