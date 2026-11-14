"""Exact Jones-Wenzl word distribution nu_l(v) = <v|p_l|v>/d_l on the Fourier words v of length l, for N = 6.
p_l is the projection onto r_l = zero-sum tensors killed by the l-1 consecutive merges. The merges preserve the
grading of the words by the elements of the group (walk endpoint and counters), so p_l is block diagonal, and
each block is computed with Fractions, in unnormalized coordinates: P = B (B^t W B)^{-1} B^t W."""
import itertools, pickle, os
from fractions import Fraction as Fr
import words as W
from linalg import kernel


def signature(w):
    d = 0; cnt = [0, 0, 0]
    for l in w:
        if l == 0: continue
        e = (d + W.CHAR[l]) % 3
        if W.BLOCK[l] == 1: cnt[d] += 1; cnt[e] -= 1
        d = e
    return (d,) + tuple(cnt)


def solve_diag(G, B, wt):
    """diagonal of B G^{-1} B^t W, exactly"""
    n = len(G)
    if n == 0: return [Fr(0)] * len(wt)
    M = [G[i][:] + [Fr(int(i == j)) for j in range(n)] for i in range(n)]
    for i in range(n):
        if M[i][i] == 0:
            j = next(j for j in range(i + 1, n) if M[j][i] != 0); M[i], M[j] = M[j], M[i]
        p = M[i][i]; M[i] = [v / p for v in M[i]]
        for j in range(n):
            if j != i and M[j][i] != 0:
                f = M[j][i]; M[j] = [a - f * b if b else a for a, b in zip(M[j], M[i])]
    Gi = [row[n:] for row in M]
    out = []
    for x in range(len(wt)):
        col = [B[i][x] for i in range(n)]; nz = [i for i in range(n) if col[i]]
        out.append(wt[x] * sum(col[i] * Gi[i][j] * col[j] for i in nz for j in nz))
    return out


def nu_exact(l, cache="jw-cache"):
    os.makedirs(cache, exist_ok=True); fn = os.path.join(cache, "nu-%d.pkl" % l)
    if os.path.exists(fn): return pickle.load(open(fn, "rb"))
    groups = {}
    for w in itertools.product(range(5), repeat=l): groups.setdefault(signature(w), []).append(w)
    diag = {}; d = Fr(0)
    for sig, ws in groups.items():
        idx = {w: i for i, w in enumerate(ws)}; rows = {}
        for w in ws:
            for t in range(l - 1):
                rest = w[:t] + w[t + 2:]
                for m, v in W.product(w[t], w[t + 1]).items():
                    r = rows.setdefault((t, m, rest), {}); r[idx[w]] = r.get(idx[w], 0) + v
        B = kernel([r for r in rows.values() if r], len(ws)); wt = [W.weight(w) for w in ws]
        G = [[sum(wt[i] * a[i] * b[i] for i in range(len(ws)) if a[i] and b[i]) for b in B] for a in B]
        dg = solve_diag(G, B, wt)
        for w, p in zip(ws, dg): diag[w] = p
        d += len(B)
    assert sum(diag.values()) == d
    nu = {w: p / d for w, p in diag.items()}
    pickle.dump((nu, int(d)), open(fn, "wb"))
    return nu, int(d)


if __name__ == "__main__":
    import time, numpy as np
    from marginal import jw_diag
    for l in (1, 2, 3, 4, 5):
        t0 = time.time(); nu, d = nu_exact(l)
        ws = list(itertools.product(range(5), repeat=l))
        fl, d2 = jw_diag(l) if l <= 5 else (None, None)
        err = max(abs(float(nu[w]) - fl[i]) for i, w in enumerate(ws))
        print("l=%d: d_l=%d (float %d), max |exact - float| = %.1e, %.1fs" % (l, d, d2, err, time.time() - t0), flush=True)


def words_with_signature(l, sigs, chunk=5 ** 9):
    """all words of length l whose signature is in the set sigs, by numpy enumeration: dict signature -> list of words"""
    import numpy as np
    BLK = np.array([0, 0, 0, 1, 1]); CH = np.array([0, 1, 2, 1, 2]); out = {s: [] for s in sigs}
    for start in range(0, 5 ** l, chunk):
        idx = np.arange(start, min(5 ** l, start + chunk), dtype=np.int64); dig = np.empty((len(idx), l), np.int8); r = idx.copy()
        for t in range(l - 1, -1, -1): dig[:, t] = r % 5; r //= 5
        d = np.zeros(len(idx), np.int8); cnt = np.zeros((len(idx), 3), np.int16); rows = np.arange(len(idx))
        for t in range(l):
            x = dig[:, t]; e = ((d + CH[x]) % 3).astype(np.int8); b = BLK[x] == 1
            np.add.at(cnt, (rows[b], d[b]), 1); np.add.at(cnt, (rows[b], e[b]), -1); d = e
        for s in sigs:
            ok = (d == s[0]) & (cnt[:, 0] == s[1]) & (cnt[:, 1] == s[2]) & (cnt[:, 2] == s[3])
            out[s] += [tuple(int(a) for a in w) for w in dig[ok]]
    return out


def jw_dim(l):
    a, b = 1, 5
    for _ in range(l - 1): a, b = b, 4 * b - a
    return b if l >= 1 else 1


def nu_words(l, wanted):
    """nu_l on the words of the list wanted, exactly, computing only the blocks of their signatures"""
    if l <= 5:
        nu, d = nu_exact(l); return {w: nu[w] for w in wanted}
    sigs = {signature(w) for w in wanted}; groups = words_with_signature(l, sigs); d = jw_dim(l); out = {}
    for sig, ws in groups.items():
        idx = {w: i for i, w in enumerate(ws)}; rows = {}
        for w in ws:
            for t in range(l - 1):
                rest = w[:t] + w[t + 2:]
                for m, v in W.product(w[t], w[t + 1]).items():
                    r = rows.setdefault((t, m, rest), {}); r[idx[w]] = r.get(idx[w], 0) + v
        B = kernel([r for r in rows.values() if r], len(ws)); wt = [W.weight(w) for w in ws]
        G = [[sum(wt[i] * a[i] * b[i] for i in range(len(ws)) if a[i] and b[i]) for b in B] for a in B]
        dg = solve_diag(G, B, wt)
        for w in wanted:
            if signature(w) == sig: out[w] = dg[idx[w]] / d
    return out
