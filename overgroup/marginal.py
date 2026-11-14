"""Module for certify.py, certify_sparse.py; as a program: python3 marginal.py k ...
Classical marginal problem for a first extra tensor over D: is there a probability measure on the identity words of
length k whose windows of l consecutive letters (cyclically) have the Jones-Wenzl distribution nu_l = diag(p_l)/d_l,
for 2l < k, and for k = 2l the merged constraints (first l-1 letters, sign eps of the next one)?  LP in floating point,
searching for a dual certificate: weights s with min over identity words of score_s(w) - k mu(s) > 0."""
import sys, itertools, time, numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, coo_matrix
import words as W

BLK = np.array([0, 0, 0, 1, 1]); CH = np.array([0, 1, 2, 1, 2]); FL = np.array(W.FLIP); CJ = np.array(W.CONJ)
EPS = np.array([0, 1, 1, -1, -1])


def identity_words_np(k, chunk=5 ** 9):
    """all identity words of length k as an array of digit rows"""
    out = []
    for start in range(0, 5 ** k, chunk):
        idx = np.arange(start, min(5 ** k, start + chunk), dtype=np.int64)
        dig = np.empty((len(idx), k), np.int8); r = idx.copy()
        for t in range(k - 1, -1, -1): dig[:, t] = r % 5; r //= 5
        d = np.zeros(len(idx), np.int8); cnt = np.zeros((len(idx), 3), np.int16); rows = np.arange(len(idx))
        for t in range(k):
            l = dig[:, t]; e = (d + CH[l]) % 3; b = (BLK[l] == 1) & (l != 0)
            np.add.at(cnt, (rows[b], d[b]), 1); np.add.at(cnt, (rows[b], e[b]), -1); d = e.astype(np.int8)
        ok = (d == 0) & (cnt == 0).all(axis=1); out.append(dig[ok])
    return np.concatenate(out)


def jw_diag(l):
    """nu_l(v) = (p_l)_{vv}/d_l in the normalized Fourier word basis, floating point; p_l = zero-sum tensors killed by consecutive merges"""
    om = np.exp(2j * np.pi / 3); f = np.zeros((5, 6), complex); f[0] = [1, 1, 1, -1, -1, -1]
    for a in range(1, 5):
        for x in range(3): f[a, 3 * W.BLOCK[a] + x] = om ** (W.CHAR[a] * x)
    f /= np.linalg.norm(f, axis=1)[:, None]
    U = np.array([[1.0 + 0j]])
    for _ in range(l): U = np.kron(U, f)                      # rows: Fourier words in physical coordinates
    sh = (6,) * l; mask = np.ones(sh, bool); ar = np.arange(6)
    for t in range(l - 1):
        a = ar.reshape([6 if j == t else 1 for j in range(l)]); b = ar.reshape([6 if j == t + 1 else 1 for j in range(l)])
        mask &= (a != b)
    # r_l = (zero-sum) cap (support on mask): kernel of U restricted to the complement of the mask
    C = U[:, ~mask.ravel()]                                   # x = sum c_w U_w vanishes off the mask: c C = 0
    u, s, vh = np.linalg.svd(C, full_matrices=True)
    r = int((s > 1e-9).sum()); K = u[:, r:]                   # orthonormal basis of the left kernel, coefficients on words
    return (np.abs(K) ** 2).sum(axis=1) / K.shape[1], K.shape[1]


def code(dig, cols):
    c = np.zeros(len(dig), np.int64)
    for j in cols: c = c * 5 + dig[:, j]
    return c


def run(k, verbose=True):
    t0 = time.time(); idw = identity_words_np(k); n = len(idw)
    l = (k - 1) // 2; nu, d = jw_diag(l)
    # symmetry classes of words under rotation, flip, conj: canonical code = min over the group
    best = None
    for r in range(k):
        rot = np.roll(idw, -r, axis=1)
        for g in (rot, FL[rot], CJ[rot], FL[CJ[rot]]):
            c = code(g, range(k)); best = c if best is None else np.minimum(best, c)
    cls, inv = np.unique(best, return_inverse=True); m = len(cls)
    rep = np.zeros(m, np.int64); rep[inv] = np.arange(n); reps = idw[rep]
    # observables: l-windows (2l < k), classes under flip, conj; for k = 2l+2 also merged: (l)-window with eps of next / previous letter
    def obs_classes(nl, signed):
        v = np.array(list(itertools.product(range(5), repeat=nl)), np.int8)
        cs = [code(g, range(nl)) for g in (v, FL[v], CJ[v], FL[CJ[v]])]
        return v, cs
    rows, rhs, names = [], [], []
    v, cs = obs_classes(l, False); can = np.minimum.reduce(cs); ucl = np.unique(can)
    cnt = np.zeros((m, 5 ** l))
    for t in range(k):
        c = code(reps, [(t + j) % k for j in range(l)]); np.add.at(cnt, (np.arange(m), c), 1)
    for u in ucl:
        sel = np.where(can == u)[0]; rows.append(cnt[:, sel].sum(axis=1)); rhs.append(k * nu[sel].sum()); names.append(("w", tuple(v[sel[0]])))
    if k % 2 == 0:                                            # merged: window of l letters, times eps(next) or eps(previous); l+1 = k/2
        nu2, d2 = jw_diag(l + 1); v2 = np.array(list(itertools.product(range(5), repeat=l + 1)), np.int8)
        for side in (0, 1):
            # antisymmetric under flip (eps changes sign): combine v with flip(v) with a minus sign; conj: plus sign
            cntp = np.zeros((m, 5 ** l))
            for t in range(k):
                c = code(reps, [(t + j) % k for j in range(l)])
                e = EPS[reps[:, (t + l) % k]] if side == 0 else EPS[reps[:, (t - 1) % k]]
                np.add.at(cntp, (np.arange(m), c), e)
            ref = np.zeros(5 ** l)
            if side == 0: np.add.at(ref, code(v2, range(l)), nu2 * EPS[v2[:, l]])
            else: np.add.at(ref, code(v2, range(1, l + 1)), nu2 * EPS[v2[:, 0]])
            cF = code(FL[v], range(l)); cC = code(CJ[v], range(l)); cFC = code(FL[CJ[v]], range(l)); c0 = code(v, range(l))
            seen = set()
            for i in range(5 ** l):
                orb = (c0[i], cC[i]); orbm = (cF[i], cFC[i]); key = min(min(orb), min(orbm))
                if key in seen: continue
                seen.add(key)
                r_ = cntp[:, list(set(orb))].sum(axis=1) - cntp[:, list(set(orbm))].sum(axis=1)
                rf = ref[list(set(orb))].sum() - ref[list(set(orbm))].sum()
                if np.abs(r_).max() == 0 and abs(rf) < 1e-12: continue
                rows.append(r_); rhs.append(k * rf); names.append(("m%d" % side, tuple(v[i])))
    A = np.array(rows); b = np.array(rhs)                     # (observables) x (word classes)
    # dual: maximize t s.t. for every word class: sum_j s_j (A[j, c] - b[j]) >= t, |s_j| <= 1
    q = len(b); cobj = np.zeros(q + 1); cobj[-1] = -1
    G = np.hstack([-(A - b[:, None]).T, np.ones((m, 1))])
    res = linprog(cobj, A_ub=G, b_ub=np.zeros(m), bounds=[(-1, 1)] * q + [(None, None)], method="highs")
    t = -res.fun if res.status == 0 else None
    if verbose:
        print("level %d: %d identity words, %d classes; l = %d, d_l = %d; %d observables; dual optimum t = %s  (%.0fs)"
              % (k, n, m, l, d, q, "%.6f" % t if t is not None else res.message, time.time() - t0), flush=True)
    return t, res, names, A, b, reps


if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:]]:
        run(k)
