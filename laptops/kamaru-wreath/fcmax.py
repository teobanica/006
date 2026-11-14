"""Maximality of the free wreath product S_n^+ wr_* S_m^+ = G^+(m K_n) inside S_N^+ (N = nm), level by level.
FC_k = span of xi_{pi,sigma}, sigma <= pi noncrossing; TL_k = span of xi_{sigma,sigma}.
An intermediate category TL subset C subset FC automatically has C_2 = TL_2 (otherwise it contains the
biprojection E, hence all of FC). Assuming C_j = TL_j for j < k, a vector w of C_k orthogonal to TL_k must
(1) have all lowerings of all rotations in TL_{k-1}, (2) give zero modulo TL in all planar quadratic
reductions R_j(w,w) (pairing, 2j<k) and M_j(w,w) (merging, j<k). If no unit w satisfies (1) and (2),
then C_k = TL_k."""
import itertools, numpy as np
from functools import lru_cache
from scipy.optimize import minimize

def set_partitions(k):
    if k == 0: yield []; return
    for p in set_partitions(k - 1):
        for i in range(len(p)): yield p[:i] + [p[i] + [k - 1]] + p[i + 1:]
        yield p + [[k - 1]]

def noncrossing(p):
    for A, B in itertools.combinations(p, 2):
        for a1, a2 in itertools.combinations(sorted(A), 2):
            for b1, b2 in itertools.combinations(sorted(B), 2):
                if a1 < b1 < a2 < b2 or b1 < a1 < b2 < a2: return False
    return True

@lru_cache(None)
def NC(k): return [tuple(tuple(b) for b in p) for p in set_partitions(k) if noncrossing(p)]

def refines(s, p):
    lab = {x: i for i, b in enumerate(p) for x in b}
    return all(len({lab[x] for x in b}) == 1 for b in s)

def xi(D, Bl, pi, sigma):
    ok = np.ones(D.shape[1], bool)
    for b in sigma:
        for x in b[1:]: ok &= D[x] == D[b[0]]
    for b in pi:
        for x in b[1:]: ok &= Bl[x] == Bl[b[0]]
    return ok.astype(np.float64)

def orth_cols(vecs, tol=1e-9):
    """orthonormal basis of the span of a list of vectors, through the Gram matrix (memory friendly)."""
    X = np.stack(vecs, 1); G = X.T @ X
    ev, EV = np.linalg.eigh(G); keep = ev > tol * ev.max()
    return X @ (EV[:, keep] / np.sqrt(ev[keep]))

def spaces(n, m, k):
    N = n * m
    D = np.array(list(itertools.product(range(N), repeat=k)), dtype=np.int8).T; Bl = D // n
    TL = orth_cols([xi(D, Bl, s, s) for s in NC(k)])
    FC = orth_cols([xi(D, Bl, p, s) for p in NC(k) for s in NC(k) if refines(s, p)])
    del D, Bl
    V = FC - TL @ (TL.T @ FC)
    G = V.T @ V; ev, EV = np.linalg.eigh(G); keep = ev > 1e-9 * max(ev.max(), 1)
    V = V @ (EV[:, keep] / np.sqrt(ev[keep]))
    return TL, FC, V

def rot(v, N, k, a=1):
    X = v.reshape((N,) * k)
    for _ in range(a): X = np.moveaxis(X, 0, -1)
    return X.reshape(-1)

def lowerings(v, N, k):
    X = v.reshape((N,) * k); idx = np.arange(N)
    for s in range(k - 1): yield X[(slice(None),) * s + (idx, idx)].reshape(-1)
    for s in range(k): yield X.sum(axis=s).reshape(-1)

def linear_survivors(V, TLprev, N, k):
    d = V.shape[1]; G = np.zeros((d, d))
    for a in range(k):
        for li in range(2 * k - 1):
            Lw = np.stack([list(lowerings(rot(V[:, i], N, k, a), N, k))[li] for i in range(d)], 1)
            Lw -= TLprev @ (TLprev.T @ Lw)
            G += Lw.T @ Lw
    ev, EV = np.linalg.eigh(G)
    return V @ EV[:, ev < 1e-12 * max(ev.max(), 1)]

def forms(Vlin, TL, FC, N, k):
    d = Vlin.shape[1]; out = []
    R = [np.stack([rot(Vlin[:, i], N, k, a) for i in range(d)], 1) for a in range(k)]
    def add(M):       # M: (len, d, d); keep only the directions actually reached
        flat = M.reshape(M.shape[0], -1)
        if np.abs(flat).max() < 1e-10: return
        U, s, _ = np.linalg.svd(flat, full_matrices=False); U = U[:, s > 1e-9 * s.max()]
        for c in range(U.shape[1]):
            f = np.tensordot(U[:, c], M, axes=(0, 0)); out.append((f + f.T) / 2)
    for a in range(k):
        for b in range(k):
            for j in range(1, (k - 1) // 2 + 1):            # pairings, output level 2j < k
                Q = FC[2 * j] - TL[2 * j] @ (TL[2 * j].T @ FC[2 * j])
                ev, EV = np.linalg.eigh(Q.T @ Q); Q = Q @ (EV[:, ev > 1e-9] / np.sqrt(ev[ev > 1e-9]))
                M = np.zeros((Q.shape[1], d, d))
                for p in range(d):
                    A = R[a][:, p].reshape(N ** j, -1)
                    for q in range(d):
                        Mm = (A @ R[b][:, q].reshape(N ** j, -1).T).reshape((N,) * (2 * j))
                        Mm = np.moveaxis(Mm, list(range(j, 2 * j)), list(range(2 * j - 1, j - 1, -1))).reshape(-1)
                        M[:, p, q] = Q.T @ Mm
                add(M)
            for j in range(1, k):                            # merges, output level j < k
                M = np.zeros((N ** j, d, d))
                for p in range(d):
                    A = R[a][:, p].reshape(N ** j, -1)
                    for q in range(d):
                        r = (A * R[b][:, q].reshape(N ** j, -1)).sum(axis=1)
                        M[:, p, q] = r - TL[j] @ (TL[j].T @ r)
                add(M)
    return out

def min_isotropy(F, d, restarts=20, seed=0):
    Ft = np.stack(F)
    def obj(x):
        x = x / np.linalg.norm(x); v = np.einsum('fij,i,j->f', Ft, x, x); return float(v @ v)
    rng = np.random.default_rng(seed); best = np.inf
    for _ in range(restarts):
        best = min(best, minimize(obj, rng.standard_normal(d), method='BFGS').fun)
    return best
