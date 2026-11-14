"""Intermediate categories TL subset C subset FC, for FC = Hom(S_n^+ wr_* S_m^+) = Hom(G^+(m K_n)),
inside the spin planar algebra of N = nm points. Level-by-level test of: C_2 = TL_2 forces C_k = TL_k."""
import itertools, numpy as np
from functools import lru_cache

def set_partitions(k):
    if k == 0: yield []; return
    for p in set_partitions(k - 1):
        for i in range(len(p)):
            yield p[:i] + [p[i] + [k - 1]] + p[i + 1:]
        yield p + [[k - 1]]

def noncrossing(p):
    for A, B in itertools.combinations(p, 2):
        for a1, a2 in itertools.combinations(sorted(A), 2):
            for b1, b2 in itertools.combinations(sorted(B), 2):
                if a1 < b1 < a2 < b2 or b1 < a1 < b2 < a2: return False
    return True

@lru_cache(None)
def NC(k):
    return [tuple(tuple(b) for b in p) for p in set_partitions(k) if noncrossing(p)]

def refines(s, p):
    lab = {}
    for i, b in enumerate(p):
        for x in b: lab[x] = i
    return all(len({lab[x] for x in b}) == 1 for b in s)

class Spin:
    def __init__(self, n, m, k):
        self.n, self.m, self.N, self.k = n, m, n * m, k
        N = self.N
        self.D = np.array(list(itertools.product(range(N), repeat=k))).T if k else np.zeros((0, 1), int)
        self.Bl = self.D // n
    def xi(self, pi, sigma):
        ok = np.ones(self.D.shape[1], bool)
        for b in sigma:
            for x in b[1:]: ok &= self.D[x] == self.D[b[0]]
        for b in pi:
            for x in b[1:]: ok &= self.Bl[x] == self.Bl[b[0]]
        return ok.astype(float)

def orth(A, tol=1e-9):
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, s > tol * max(s.max(), 1)]

def spaces(n, m, k):
    S = Spin(n, m, k)
    tl = np.stack([S.xi(s, s) for s in NC(k)], 1)
    fc = np.stack([S.xi(p, s) for p in NC(k) for s in NC(k) if refines(s, p)], 1)
    TL = orth(tl); FC = orth(fc)
    # V = FC minus TL
    Pfc = FC - TL @ (TL.T @ FC)
    V = orth(Pfc)
    return S, TL, FC, V

def rotate(v, N, k):
    return np.moveaxis(v.reshape((N,) * k), 0, -1).reshape(-1)

def lowerings(v, N, k):
    X = v.reshape((N,) * k); idx = np.arange(N); out = []
    for s in range(k - 1):
        out.append(X[(slice(None),) * s + (idx, idx)].reshape(-1))
    for s in range(k):
        out.append(X.sum(axis=s).reshape(-1))
    return out

def reduce_open(v, w, N, k, j):
    """open the first j legs: M[x,y] = sum_z v(x,z) conj w(y,z); as a 2j-vector (x_1..x_j, y_j..y_1)."""
    A = v.reshape(N ** j, -1); B = w.reshape(N ** j, -1)
    M = A @ B.conj().T
    M = M.reshape((N,) * (2 * j))
    M = np.moveaxis(M, list(range(j, 2 * j)), list(range(2 * j - 1, j - 1, -1)))
    return M.reshape(-1)

def level_test(n, m, k, prev):
    """prev: dict j -> TL_j orthonormal basis for j < k (assumed C_j = TL_j). Returns V_lin and quadratic forms."""
    N = n * m
    S, TL, FC, V = spaces(n, m, k)
    if V.shape[1] == 0: return TL, FC, V, V, []
    # linear constraints: for all rotations, all lowerings land in TL_{k-1}
    TLp = prev[k - 1]
    rows = []
    for a in range(k):
        Vr = np.stack([fc_rot(V[:, i], N, k, a) for i in range(V.shape[1])], 1)
        for s_idx in range(2 * k - 1):
            Lw = np.stack([lowerings(Vr[:, i], N, k)[s_idx] for i in range(V.shape[1])], 1)
            Lw = Lw - TLp @ (TLp.T @ Lw)
            rows.append(Lw)
    A = np.concatenate(rows, 0)
    G = A.T @ A
    ev, EV = np.linalg.eigh((G + G.T) / 2)
    null = EV[:, ev < 1e-12 * max(ev.max(), 1)]                      # coefficients of V_lin in the basis V
    Vlin = V @ null
    return TL, FC, V, Vlin, None

def fc_rot(v, N, k, a):
    for _ in range(a): v = rotate(v, N, k)
    return v


def quad_forms(n, m, k, Vlin, prevTL, prevFC, jmax=None):
    """For each pair of rotations (a,b) and each j with 2j < k: the components of R_j(rot^a w, rot^b w')
    orthogonal to TL_{2j} inside FC_{2j}, as bilinear forms on Vlin coordinates. Returns list of matrices."""
    N = n * m; d = Vlin.shape[1]; forms = []
    for j in range(1, (k - 1) // 2 + 1):
        if jmax and j > jmax: break
        TLj, FCj = prevTL[2 * j], prevFC[2 * j]
        Q = FCj - TLj @ (TLj.T @ FCj); Q = orth(Q)          # FC_2j minus TL_2j
        rots = [np.stack([fc_rot(Vlin[:, i], N, k, a) for i in range(d)], 1) for a in range(k)]
        for a in range(k):
            for b in range(k):
                M = np.zeros((Q.shape[1], d, d))
                for p in range(d):
                    for q in range(d):
                        r = reduce_open(rots[a][:, p], rots[b][:, q], N, k, j)
                        M[:, p, q] = Q.T @ r.real
                for c in range(Q.shape[1]): forms.append(M[c])
    return forms

def reduce_merge(v, w, N, k, j):
    """open the first j legs, merged pairwise with the mirrored legs of w: out(x) = sum_z v(x,z) w(x,z)."""
    A = v.reshape(N ** j, -1); B = w.reshape(N ** j, -1)
    return (A * B).sum(axis=1)

def merge_forms(n, m, k, Vlin, prevTL):
    N = n * m; d = Vlin.shape[1]; forms = []
    rots = [np.stack([fc_rot(Vlin[:, i], N, k, a) for i in range(d)], 1) for a in range(k)]
    for j in range(1, k):
        TLj = prevTL[j]
        for a in range(k):
            for b in range(k):
                M = None
                for p in range(d):
                    for q in range(d):
                        r = reduce_merge(rots[a][:, p], rots[b][:, q], N, k, j)
                        r = r - TLj @ (TLj.T @ r)
                        if M is None: M = np.zeros((len(r), d, d))
                        M[:, p, q] = r
                U = orth(M.reshape(len(M), -1).T) if np.abs(M).max() > 1e-10 else None
                if U is not None:
                    # project onto the span actually reached, to keep the number of forms small
                    B = orth(M.reshape(len(M), -1))
                    for c in range(B.shape[1]):
                        forms.append(np.tensordot(B[:, c], M, axes=(0, 0)))
    return forms
