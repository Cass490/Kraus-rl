"""Joint single-qubit synthesis backends and native even-T constraints."""

import numpy as np

from .words import tcount, word_error


_CYC = {}


def require_cyclosynth():
    """Reject a source-directory namespace import without the compiled API."""
    import cyclosynth
    if not callable(getattr(cyclosynth, "Synthesizer", None)):
        raise ImportError(
            "cyclosynth.Synthesizer is unavailable; Python found a module or source "
            "folder without the compiled API. Install the cyclosynth Python extension "
            "into the interpreter running this script."
        )
    return cyclosynth


def euler_zyz(U):
    """U ~ e^{i phi} Rz(a) Ry(b) Rz(g), Rz(t)=diag(e^{-it/2},e^{it/2}), Ry(t)=[[c,-s],[s,c]]."""
    U = U / np.sqrt(np.linalg.det(U))
    b = 2 * np.arctan2(abs(U[1, 0]), abs(U[0, 0]))
    s = np.angle(U[1, 1]) ; d = np.angle(U[1, 0])          # U11 = e^{i(a+g)/2} cos, U10 = e^{i(a-g)/2} sin
    if abs(U[0, 0]) < 1e-14: s = -np.angle(U[0, 1]) + np.pi if False else d
    if abs(U[1, 0]) < 1e-14: d = s
    a = s + d; g = s - d
    return a, b, g


def pygridsynth_synth(U, eps):
    """Phase-preserving stand-in, typically producing longer general-unitary words."""
    import mpmath
    from pygridsynth.unitary_approximation import approximate_one_qubit_unitary
    U = U / np.sqrt(np.linalg.det(U))
    circ, _ = approximate_one_qubit_unitary(
        mpmath.matrix(U.tolist()), mpmath.mpf(eps), up_to_phase=False
    )
    return circ.to_simple_str()


def cyclosynth_synth(U, eps, *, even_t=False):
    """Joint SU(2) synthesis via cyclosynth (build locally: maturin develop --release)."""
    cyclosynth = require_cyclosynth()
    key = (eps, even_t)
    if key not in _CYC:
        _CYC[key] = cyclosynth.Synthesizer(eps, even_t=even_t)
    a, b, g = euler_zyz(U)
    # U3(theta, phi, lambda) represents Rz(phi) Ry(theta) Rz(lambda) up to phase.
    # synthesize_zyz instead approximates three rotations separately in this API.
    r = _CYC[key].synthesize_u3(float(b), float(a), float(g))
    if r is None or r.gates is None:
        raise RuntimeError("cyclosynth found no circuit")
    return r.gates


def even_word(U, synth, eps):
    """Use the backend's even-T search for paired words when available."""
    even_synth = getattr(synth, "even", None)
    if even_synth is not None:
        w = even_synth(U, eps)
        if tcount(w) % 2:
            raise RuntimeError("even-T backend returned an odd T-count word")
        return w
    # Custom/stand-in backends can still use phase-preserving synthesis.
    w = synth(U, eps)
    if tcount(w) % 2 == 0:
        return w
    w = pygridsynth_synth(U, eps)
    if tcount(w) % 2:
        raise RuntimeError("phase-preserving SU(2) synthesis returned an odd T-count word")
    return w


def make_synth(kind, eps_scale=1.0):
    if kind == "cyclosynth" and not hasattr(require_cyclosynth().Synthesizer, "even_t"):
        raise RuntimeError(
            "The installed cyclosynth extension predates even_t support. "
            "Run scripts/setup_cyclosynth.sh in the project environment before synthesis."
        )
    base = {"cyclosynth": cyclosynth_synth, "pygridsynth": pygridsynth_synth}[kind]
    def checked(U, eps, even_t=False):
        w = cyclosynth_synth(U, eps, even_t=True) if even_t else base(U, eps)
        err = word_error(w, U)
        # Dense products of long words have a small floating-point floor.
        roundoff = 32 * np.finfo(float).eps * max(1, len(w))
        if not np.isfinite(err) or err > eps + roundoff:
            raise RuntimeError(f"{kind}: phase-aligned word error {err:.2e} "
                               f"exceeds epsilon {eps:.2e} (roundoff {roundoff:.1e})")
        return w
    def synth(U, eps):
        return checked(U, eps)
    if kind == "cyclosynth":
        def even(U, eps):
            return checked(U, eps, even_t=True)
        synth.even = even
    return synth
