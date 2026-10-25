"""Full parity-product factorization of SU(2) multiplexor blocks."""

import numpy as np


PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]


def su2_exp(v):
    th = np.linalg.norm(v)
    if th < 1e-300:
        return np.eye(2, dtype=complex)
    n = v / th
    return np.cos(th) * np.eye(2) - 1j * np.sin(th) * (n[0] * PAULI[0] + n[1] * PAULI[1] + n[2] * PAULI[2])


def su2_log(U):
    """v with su2_exp(v) = U (U in SU(2)), principal branch, full precision near identity."""
    a = np.real(U[0, 0] + U[1, 1]) / 2
    v = np.array([-np.imag(U[0, 1] + U[1, 0]) / 2,
                  np.real(U[1, 0] - U[0, 1]) / 2,
                  -np.imag(U[0, 0] - U[1, 1]) / 2])
    s = np.linalg.norm(v)
    if s < 1e-300:
        return np.zeros(3)
    th = np.arctan2(s, a)
    return th * v / s


def parities(c):
    q = 2 ** c - 1
    P = np.zeros((2 ** c, q), dtype=int)
    for x in range(2 ** c):
        for j, m in enumerate(range(1, 2 ** c)):
            P[x, j] = bin(x & m).count("1") & 1
    return P


def model(R, betas, P):
    out = []
    for x in range(P.shape[0]):
        M = R.copy()
        for j in range(P.shape[1]):
            if P[x, j]:
                M = M @ betas[j]
        out.append(M)
    return out


def factorize(V, max_steps=400, tol=1e-13):
    """V: array (2^c, 2, 2) of SU(2) blocks. Returns (R, betas[q]) with model == V to ~1e-13."""
    n = V.shape[0]; c = int(round(np.log2(n))); q = n - 1
    P = parities(c)
    R = V[0].copy()
    Rinv = R.conj().T
    L = [su2_log(Rinv @ V[x]) for x in range(n)]          # targets relative to R
    th = np.zeros((q, 3))                                 # beta_j = exp(th_j)
    def resid(th, s):
        betas = [su2_exp(t) for t in th]
        r = []
        for x in range(1, n):
            M = np.eye(2, dtype=complex)
            for j in range(q):
                if P[x, j]:
                    M = M @ betas[j]
            Tgt = su2_exp(s * L[x])
            r.append(su2_log(Tgt.conj().T @ M))
        return np.concatenate(r)
    def jac(th, s, h=1e-7):
        f0 = resid(th, s); J = np.zeros((f0.size, th.size))
        flat = th.ravel()
        for k in range(flat.size):
            t2 = flat.copy(); t2[k] += h
            J[:, k] = (resid(t2.reshape(q, 3), s) - f0) / h
        return f0, J
    s = 0.0; ds = 0.05
    while s < 1.0 - 1e-15:
        s_new = min(1.0, s + ds)
        th_try = th.copy(); ok = False
        for it in range(30):
            f, J = jac(th_try, s_new)
            if np.abs(f).max() < (tol if s_new == 1.0 else 1e-10):
                ok = True; break
            try:
                step = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                break
            th_try = th_try + step.reshape(q, 3)
            if np.abs(step).max() > 1.0:
                break
        if ok:
            th, s = th_try, s_new
            ds = min(0.25, ds * 1.5)
        else:
            ds /= 2
            if ds < 1e-6:
                raise RuntimeError("parity factorisation continuation failed")
    betas = [su2_exp(t) for t in th]
    # polish with exact-ish Newton at s=1
    return R, betas
