"""Sparse version of certify.py for odd levels (windows only): word certificate at level k = 2l+1 by linear programming.
Output: found.json (one entry per level)"""
import sys, os, json, itertools, time, numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack, identity
import words as W
from marginal import identity_words_np, code, FL, CJ
from jw import nu_exact

def find(k):
    t0 = time.time(); assert k % 2 == 1; l = (k - 1) // 2
    idw = identity_words_np(k); n = len(idw); best = None
    for r in range(k):
        rot = np.roll(idw, -r, axis=1)
        for g in (rot, FL[rot], CJ[rot], FL[CJ[rot]]):
            c = code(g, range(k)); best = c if best is None else np.minimum(best, c)
        del rot
    cls, inv = np.unique(best, return_inverse=True); m = len(cls); del best
    rep = np.zeros(m, np.int64); rep[inv] = np.arange(n); reps = idw[rep]; del idw, inv
    v = np.array(list(itertools.product(range(5), repeat=l)), np.int8); nv = len(v)
    can = np.minimum.reduce([code(g, range(l)) for g in (v, FL[v], CJ[v], FL[CJ[v]])])
    ucl, cidx = np.unique(can, return_inverse=True); q = len(ucl)
    nu, d = nu_exact(l); nuf = np.array([float(nu[tuple(int(x) for x in w)]) for w in v])
    b = np.zeros(q); np.add.at(b, cidx, nuf)
    rows = np.repeat(np.arange(m), k); cols = np.concatenate([cidx[code(reps, [(t + j) % k for j in range(l)])][:, None] for t in range(k)], axis=1).ravel()
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, q)).tocsr()          # classes x observables: window counts
    D = (A - csr_matrix(np.ones((m, 1))) @ csr_matrix(k * b[None, :])).tocsr() if False else None
    print("level %d: %d identity words, %d classes, l = %d, %d observables (%.0fs)" % (k, n, m, l, q, time.time() - t0), flush=True)
    # D s >= t with D = A - k 1 b^t: introduce the scalar y = k b.s ; A s - y >= t
    # variables (s, y, t): maximize t; constraints: -A s + y + t <= 0 ; k b.s - y = 0 ; |s| <= 1
    Aub = hstack([-A, csr_matrix(np.ones((m, 1))), csr_matrix(np.ones((m, 1)))]).tocsr()
    Aeq = csr_matrix(np.r_[k * b, -1, 0][None, :])
    res = linprog(np.r_[np.zeros(q + 1), -1], A_ub=Aub, b_ub=np.zeros(m), A_eq=Aeq, b_eq=[0],
                  bounds=[(-1, 1)] * q + [(None, None), (None, None)], method="highs")
    t = -res.fun; print("   margin LP optimum %.6f (%.0fs)" % (t, time.time() - t0), flush=True)
    if t < 1e-7: return
    # sparse: minimize |s|_1 with A s - y >= 1
    Aub2 = hstack([-A, A, csr_matrix(np.ones((m, 1)))]).tocsr(); Aeq2 = csr_matrix(np.r_[k * b, -k * b, -1][None, :])
    res2 = linprog(np.r_[np.ones(2 * q), 0], A_ub=Aub2, b_ub=-np.ones(m), A_eq=Aeq2, b_eq=[0],
                   bounds=[(0, None)] * (2 * q) + [(None, None)], method="highs")
    s = res2.x[:q] - res2.x[q:2 * q]
    for Qs in [1, 2, 3, 4, 5, 6, 8, 10, 12, 20, 30, 60, 120, 1000, 10000]:
        si = np.rint(s / np.abs(s).max() * Qs).astype(np.int64); mar = (A @ si).min() - k * (b @ si)
        if mar > 1e-6 * Qs: break
    print("   sparse certificate: %d nonzero weights of %d, integers of size <= %d, floating margin %.4f (%.0fs)" % ((si != 0).sum(), q, Qs, mar, time.time() - t0), flush=True)
    tab = {}
    for i in range(nv):
        if si[cidx[i]]: tab["".join(str(int(x)) for x in v[i])] = int(si[cidx[i]])
    found = json.load(open("found.json")) if os.path.exists("found.json") else {}
    found[str(k)] = {"level": k, "l": l, "w": tab, "m0": {}, "m1": {}}
    json.dump(found, open("found.json", "w"), indent=0)

if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:]]: find(k)
