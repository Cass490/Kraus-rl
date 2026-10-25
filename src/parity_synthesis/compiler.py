"""Recursive unitary compiler with indexed diagonal-phase carry."""

import time

import numpy as np

from .decompositions import csd_items, Ry, full_index, diagonal_gates
from .emission import mux_gates
from .multiplexors import synth_mux
from .precision import synthesis_word_counts, word_epsilon


def optimize_circuit(gates, gph, verbose=False):
    """Reduce physical gates exactly, retaining all removed scalar phases."""
    from .cleanup import cleanup_gates
    started = time.monotonic()
    if verbose:
        print(f"  cleaning {len(gates) + 1:,} emitted gates (including global phase)", flush=True)
    reduced, phase = cleanup_gates(gates)
    if verbose:
        print(f"  gate cleanup: {len(gates) + 1:,} -> {len(reduced) + 1:,}; "
              f"T={sum(g[0] == 'T' for g in reduced)} "
              f"({time.monotonic() - started:.1f}s)", flush=True)
    return reduced, gph + phase


def synthesize(u_mat, cs_decomposition, synth, eps, verbose=False, optimize=True,
               error_budget=None, mux_method="baseline", fold_common=False):
    """Returns (gates, global_phase): gates time-ordered tuples ('NAME', wires[, angle])."""
    items, n = csd_items(u_mat, cs_decomposition)
    eps = word_epsilon(n, eps, error_budget)
    if verbose:
        expected, worst = synthesis_word_counts(n, mux_method)
        print(f"  n={n}: {expected} planned words ({worst} with fallbacks); "
              f"per-word epsilon={eps:.3e}; fold_common={fold_common}", flush=True)
    N = 2 ** n
    carry = np.zeros(N)
    gates, gph = [], 0.0
    error_bound = 0.0
    t0 = time.time()
    for item_index, it in enumerate(items):
        if it[0] == "G":
            t = n - 1; blocks = list(it[1])
        else:
            t = it[1]; blocks = [Ry(a) for a in it[2]]
        controls = [w for w in range(n) if w != t]
        c = len(controls)
        V = np.zeros((2 ** c, 2, 2), complex); phi = np.zeros(2 ** c)
        for x in range(2 ** c):
            i0 = full_index(n, t, controls, x, 0); i1 = full_index(n, t, controls, x, 1)
            B = blocks[x] @ np.diag([np.exp(1j * carry[i0]), np.exp(1j * carry[i1])])
            d = np.linalg.det(B); phi[x] = np.angle(d) / 2
            V[x] = B / np.exp(1j * phi[x])
        # Keep the final carry independent of the last target bit, avoiding
        # 16 additional arbitrary RZ rotations at five qubits.
        res = synth_mux(V, synth, eps, verbose=verbose, mux_method=mux_method,
                        fold_common=fold_common, allow_output_z=item_index < len(items) - 1)
        error_bound += res["error_bound"]
        g, ph = mux_gates(res, t, controls)
        gates += g; gph += ph
        carry = np.zeros(N)
        for x in range(2 ** c):
            for tb in (0, 1):
                carry[full_index(n, t, controls, x, tb)] = (
                    phi[x] + res["ctrl_phase"][x] + (2 * tb - 1) * res["output_z"][x])
    gph += diagonal_gates(carry, list(range(n)), gates)
    if verbose:
        print(f"  sum of multiplexor error bounds: {error_bound:.3e} "
              "(excludes CSD/roundoff)", flush=True)
    if error_budget is not None and error_bound > error_budget:
        raise RuntimeError(f"sum of multiplexor error bounds {error_bound:.2e} "
                           f"exceeds requested circuit budget {error_budget:.2e}")
    if optimize:
        gates, gph = optimize_circuit(gates, gph, verbose=verbose)
    if verbose:
        print(f"  synthesized n={n} in {time.time()-t0:.0f}s")
    return gates, gph
