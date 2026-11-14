"""Exact linear algebra over Q with Fractions: kernels, and positivity by elimination."""
from fractions import Fraction as Fr


def kernel(rows, n):
    """rows: list of dicts {column: Fraction}; returns a basis of the kernel in Q^n, as lists"""
    piv = {}                                  # pivot column -> reduced row (dict), leading coefficient 1
    for r in rows:
        r = dict(r)
        while r:
            c = min(r)
            if c in piv:
                f = r[c]; p = piv[c]
                for cc, v in p.items():
                    nv = r.get(cc, 0) - f * v
                    if nv == 0: r.pop(cc, None)
                    else: r[cc] = nv
            else:
                f = r[c]; piv[c] = {cc: v / f for cc, v in r.items()}; break
    # back substitution
    free = [c for c in range(n) if c not in piv]
    basis = []
    cols = sorted(piv, reverse=True)
    for fcol in free:
        v = [Fr(0)] * n; v[fcol] = Fr(1)
        for c in cols:
            p = piv[c]; s = Fr(0)
            for cc, val in p.items():
                if cc != c and v[cc] != 0: s += val * v[cc]
            v[c] = -s
        basis.append(v)
    return basis


def pivots(M):
    """pivots of the symmetric elimination of M (list of lists of Fractions), without row exchanges; all positive iff M is positive definite"""
    n = len(M); A = [row[:] for row in M]; out = []
    for i in range(n):
        p = A[i][i]; out.append(p)
        if p <= 0: return out
        for j in range(i + 1, n):
            if A[j][i] != 0:
                f = A[j][i] / p
                for c in range(i, n):
                    if A[i][c] != 0: A[j][c] -= f * A[i][c]
    return out
