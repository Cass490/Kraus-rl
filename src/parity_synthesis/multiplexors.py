"""Phase-carry multiplexor compilation and common-word folding."""

import numpy as np

from .backends import even_word
from .pairs import compile_pair_best, compile_word_pair_best
from .parity import factorize, model, parities
from .words import mat, phase_of, tcount, word_error


def factor_mux_blocks(V, mux_method="baseline", allow_output_z=True, verbose=False):
    """Return factors, actual masks, and output-Z carry; fall back before synthesis."""
    if mux_method not in ("baseline", "phase11"):
        raise ValueError(f"unknown multiplexor method: {mux_method}")
    count = len(V)
    c = int(round(np.log2(count)))
    if mux_method == "phase11" and c == 4 and allow_output_z:
        from .phase_factorization import MASKS, PhaseFactorizationError, factorize_phase
        try:
            R, betas, z, error = factorize_phase(V)
            if verbose:
                print(f"    phase11: 11 masks, factor error={error:.2e}", flush=True)
            return R, betas, MASKS, z, error, "phase11"
        except PhaseFactorizationError as exc:
            if verbose:
                print(f"    phase11 fallback to 15 masks: {exc}", flush=True)
    R, betas = factorize(V)
    error = max(float(np.linalg.norm(M - target, 2))
                for M, target in zip(model(R, betas, parities(c)), V))
    if error > 1e-11:
        raise RuntimeError(f"parity factorization matrix error {error:.2e} exceeds 1e-11")
    return R, betas, tuple(range(1, count)), np.zeros(count), error, "baseline"


def synth_mux(V, synth, eps, verbose=False, mux_method="baseline", fold_common=False,
              allow_output_z=True):
    """V: (2^c,2,2) SU(2) blocks. Returns dict:
       pairs: list (time order) of (mask, gates) with gates on symbolic wires 'c','q'
       C: word for the common gate (applied last), global: global phase (radians, multiply by e^{i*global})
       ctrl_phase: array over x (2^c) of control-pattern phases (radians) to push into the carry."""
    n = V.shape[0]; c = int(round(np.log2(n)))
    R, betas, masks, output_z, factor_error, method = factor_mux_blocks(
        V, mux_method, allow_output_z, verbose)
    q = len(masks)
    P = np.array([[(x & mask).bit_count() % 2 for mask in masks] for x in range(n)])
    error_bound = factor_error
    G = np.eye(2, dtype=complex)
    pairs_rev = []
    ctrl = np.zeros(n)
    tsum = 0
    folded = fold_common and q > 0
    for j in range(q - 1, 0 if folded else -1, -1):
        W = G @ betas[j] @ G.conj().T
        w = even_word(W, synth, eps)
        gam = phase_of(w, W)
        error_bound += word_error(w, W)
        res = compile_pair_best(w)
        tsum += res["t"]
        # exactness: pair realises P(A, A w) * phase(ctrl_phase*pi/4 on parity 1); we want P(A, A W):
        corr = res["ctrl_phase"] * np.pi / 4 - gam
        ctrl += corr * P[:, j]
        pairs_rev.append((masks[j], res["gates"]))
        G = mat(res["A"]) @ G
    if folded:
        target0 = R @ G.conj().T
        target1 = R @ betas[0] @ G.conj().T
        word0, word1 = even_word(target0, synth, eps), even_word(target1, synth, eps)
        gamma0, gamma1 = phase_of(word0, target0), phase_of(word1, target1)
        last_pair = compile_word_pair_best(word0, word1)
        ctrl += (last_pair["ctrl_phase"] * np.pi / 4 - gamma1 + gamma0) * P[:, 0]
        pairs_rev.append((masks[0], last_pair["gates"]))
        global_phase = last_pair["global"] - gamma0
        error_bound += max(word_error(word0, target0), word_error(word1, target1))
        tsum += last_pair["t"]
        wc = ""
    else:
        Cmat = R @ G.conj().T
        wc = synth(Cmat, eps)
        global_phase = -phase_of(wc, Cmat)
        error_bound += word_error(wc, Cmat)
    if verbose:
        print(f"    mux c={c}: {method}, {q} masks, pairs T={tsum}, "
              f"common={'folded' if folded else str(tcount(wc)) + ' T'}", flush=True)
    return {"pairs": pairs_rev, "C": wc, "global": global_phase, "ctrl_phase": ctrl, "c": c,
            "output_z": output_z, "factorization": method, "error_bound": error_bound}
