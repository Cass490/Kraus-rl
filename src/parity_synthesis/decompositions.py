"""Cosine-sine decomposition and diagonal circuit primitives.

These are numerical linear-algebra helpers, not a benchmark or grading harness.
The CSD convention and implementation provenance are recorded in NOTICE.md.
"""

import numpy as np

_TOL = 1e-8


def cs_decomposition(u_mat):
    """Cosine-sine decomposition of a unitary of even dimension 2m. Returns
    (l_0, l_1, theta, r_0, r_1) such that, in m x m blocks,

        u_mat == [[l_0, 0], [0, l_1]] @ [[C, -S], [S, C]] @ [[r_0, 0], [0, r_1]]

    with C = diag(cos(theta)) and S = diag(sin(theta)).
    """
    half = len(u_mat) // 2
    u_00, u_01 = u_mat[:half, :half], u_mat[:half, half:]
    u_10, u_11 = u_mat[half:, :half], u_mat[half:, half:]
    l_0, cosines, r_0 = np.linalg.svd(u_00)
    lower = u_10 @ r_0.conj().T
    sines = np.linalg.norm(lower, axis=0)
    theta = np.arctan2(sines, cosines)
    big_s, big_c = sines > _TOL, cosines > _TOL
    l_1 = _complete_basis(lower / np.where(big_s, sines, 1.0), big_s)
    r_1 = np.where(
        big_s[:, None],
        -(l_0.conj().T @ u_01) / np.where(big_s, sines, 1.0)[:, None],
        (l_1.conj().T @ u_11) / np.where(big_c, cosines, 1.0)[:, None],
    )
    # l_1 and r_1 come from dividing by small sines, so they drift from unitarity; snap
    # each to the nearest unitary so the drift is not amplified by a deep recursion.
    return l_0, _nearest_unitary(l_1), theta, r_0, _nearest_unitary(r_1)


def _nearest_unitary(mat):
    """The unitary closest to `mat` in Frobenius norm (its polar factor)."""
    left, _, right = np.linalg.svd(mat)
    return left @ right


def _complete_basis(partial, mask):
    """Fills the columns of `partial` where `mask` is False with an orthonormal completion."""
    missing = int((~mask).sum())
    if missing == 0:
        return partial
    good = partial[:, mask]
    extra, _, _ = np.linalg.svd(np.eye(len(partial)) - good @ good.conj().T)
    out = np.array(partial, dtype=complex)
    out[:, ~mask] = extra[:, :missing]
    return out


def csd_items(u_mat, cs_decomposition):
    n = int(round(np.log2(len(u_mat))))
    def rec(blocks, k):
        if k == n - 1:
            return [("G", blocks)]
        lefts, rights, cs = [], [], []
        for block in blocks:
            l0, l1, th, r0, r1 = cs_decomposition(block)
            lefts += [l0, l1]; rights += [r0, r1]; cs.append(2 * np.asarray(th))
        return rec(rights, k + 1) + [("CS", k, np.concatenate(cs))] + rec(lefts, k + 1)
    return rec([np.array(u_mat, dtype=complex)], 0), n


def Ry(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)


def full_index(n, t, controls, x, tb):
    idx = 0
    for i, w in enumerate(controls):
        if (x >> (len(controls) - 1 - i)) & 1:
            idx |= 1 << (n - 1 - w)
    if tb:
        idx |= 1 << (n - 1 - t)
    return idx


def uc_rz(angles, controls, target, out):
    """Gray-code uniformly controlled RZ; skips (near-)zero rotations."""
    angles = np.asarray(angles, float); k = len(controls)
    if k == 0:
        if abs(angles[0]) > 1e-13: out.append(("RZ", (target,), float(angles[0])))
        return
    size = 2 ** k
    gray = [i ^ (i >> 1) for i in range(size)]
    signs = np.array([[(-1) ** bin(j & g).count("1") for j in range(size)] for g in gray])
    th = signs @ angles / size
    for i in range(size):
        if abs(th[i]) > 1e-13: out.append(("RZ", (target,), float(th[i])))
        bit = (gray[i] ^ gray[(i + 1) % size]).bit_length() - 1
        out.append(("CNOT", (controls[k - 1 - bit], target)))


def diagonal_gates(phases, wires, out):
    """diag(exp(i*phases)) on wires (big-endian); returns leftover global phase (radians)."""
    phases = np.asarray(phases, float); m = len(wires)
    while m > 0:
        even, odd = phases[0::2], phases[1::2]
        uc_rz(odd - even, wires[:m - 1], wires[m - 1], out)
        phases = (even + odd) / 2; m -= 1
    return float(phases[0])
