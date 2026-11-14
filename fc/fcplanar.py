"""fcplanar.py -- corrected test for the maximality of S_n^+ wr_* S_m^+ inside S_{nm}^+ (Proposition 10.1).
Same setting as fcmax.py: TL subset C subset FC, C_j = TL_j for j < k, and w in C_k orthogonal to TL_k.
(1) linear conditions: all lowerings of all rotations of w lie in TL_{k-1};
(2) quadratic conditions: every planar contraction of w (x) w* and of w (x) w landing at a level < k lies in TL there.
The planar contractions are: r nested pairs at the junction, x_{k-s+1} with y_s for s = 1..r, possibly followed by
the merge of the next pair, at all rotations of both factors. (fcmax.py used parallel instead of nested merges,
which are not planar: this is the error corrected here.) Complex w is handled through real coordinates (Re c, Im c).
Conclusion C_k = TL_k is reached when a positive definite combination of the real quadratic forms is found
(certificate, as in sdpcert.py); the minimum of the sum of squares over the unit sphere is also reported."""
import sys, json, time, itertools, numpy as np
from scipy.optimize import minimize
import fcmax, sdpcert

def rot(v, N, k, a): return fcmax.rot(v, N, k, a)

def contract(x, y, N, k, r, merge):
    X = x.reshape((N,) * k); Y = y.reshape((N,) * k)
    Z = np.tensordot(X, Y, axes=(list(range(k - 1, k - r - 1, -1)), list(range(r))))   # x_{k-s+1} ~ y_s
    if merge:
        p = k - r - 1                                   # last free leg of x, first free leg of y at p+1
        Z = np.diagonal(Z, axis1=p, axis2=p + 1)        # merged leg goes last
        Z = np.moveaxis(Z, -1, p)
    return Z.reshape(-1)

def forms(Vlin, TLs, N, k):
    d = Vlin.shape[1]
    R = [[rot(Vlin[:, i], N, k, a) for i in range(d)] for a in range(k)]
    Rrev = [[R[a][i].reshape((N,) * k).transpose(tuple(reversed(range(k)))).reshape(-1) for i in range(d)] for a in range(k)]   # w* for real basis
    F = []
    for r in range(1, k):
        for merge in (False, True):
            L = 2 * (k - r) - (1 if merge else 0)
            if L >= k or L < 1: continue
            TL = TLs[L]
            for kind, Ys in (("sesq", Rrev), ("bil", R)):
                for a in range(k):
                    for b in range(k):
                        B = np.zeros((N ** L, d, d))
                        for p in range(d):
                            for q in range(d):
                                z = contract(R[a][p], Ys[b][q], N, k, r, merge)
                                B[:, p, q] = z - TL @ (TL.T @ z)
                        if np.abs(B).max() < 1e-10: continue
                        flat = B.reshape(B.shape[0], -1)
                        U, s, _ = np.linalg.svd(flat, full_matrices=False); U = U[:, s > 1e-9 * s.max()]
                        F.append((kind, np.tensordot(U, B, axes=(0, 0)).transpose(1, 2, 0)))   # (d, d, m)
    return F

def run(n, m, K, allpart=False, log=print):
    N = n * m; TLs = {}; out = {}
    for k in range(1, K + 1):
        t0 = time.time()
        if allpart:      # sanity check: the category of S_N instead of FC
            D = np.array(list(itertools.product(range(N), repeat=k)), dtype=np.int8).T; Bl = D // n
            TL = fcmax.orth_cols([fcmax.xi(D, Bl, s, s) for s in fcmax.NC(k)])
            P = [tuple(tuple(b) for b in p) for p in fcmax.set_partitions(k)]
            FC = fcmax.orth_cols([fcmax.xi(D, D, p, p) for p in P])
            V = FC - TL @ (TL.T @ FC); G = V.T @ V; ev, EV = np.linalg.eigh(G); kp = ev > 1e-9 * max(ev.max(), 1)
            V = V @ (EV[:, kp] / np.sqrt(ev[kp]))
        else:
            TL, FC, V = fcmax.spaces(n, m, k)
        TLs[k] = TL
        if k <= 2:
            log("n=%d m=%d level %d: TL %d, FC %d, FC-TL %d (C_2 = TL_2 by hypothesis)" % (n, m, k, TL.shape[1], FC.shape[1], V.shape[1])); continue
        Vlin = fcmax.linear_survivors(V, TLs[k - 1], N, k) if V.shape[1] else V; d = Vlin.shape[1]
        rec = {"TL": TL.shape[1], "FC": FC.shape[1], "FC-TL": V.shape[1], "linear": d}
        if d:
            F = forms(Vlin, TLs, N, k)
            basis = sdpcert.real_forms_span(F, d) if F else []
            if not basis:
                rec.update({"forms": 0, "certificate": None, "min": 0.0, "seconds": round(time.time() - t0), "verdict": "no quadratic constraint"})
                out[k] = rec; log("n=%d m=%d level %d: %s" % (n, m, k, rec)); continue
            lam, _ = sdpcert.certify(basis, iters=5)
            St = np.array(basis)
            def obj(x):
                x = x / np.linalg.norm(x); v = np.einsum('fij,i,j->f', St, x, x); return float(v @ v)
            rng = np.random.default_rng(0)
            mn = min(minimize(obj, rng.standard_normal(2 * d), method="BFGS").fun for _ in range(20))
            rec.update({"forms": len(basis), "certificate": float(lam), "min": float(mn)})
            verdict = "C_k = TL_k (certified)" if lam > 1e-6 else "no certificate"
        else:
            verdict = "C_k = TL_k (linear)"
        rec["seconds"] = round(time.time() - t0); rec["verdict"] = verdict; out[k] = rec
        log("n=%d m=%d level %d: %s -> %s" % (n, m, k, rec, verdict))
    return out

if __name__ == "__main__":
    n, m, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); allp = len(sys.argv) > 4
    res = run(n, m, K, allp)
    json.dump(res, open("fcplanar-%d-%d%s.json" % (n, m, "-sanity" if allp else ""), "w"), indent=1)
