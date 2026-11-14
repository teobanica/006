"""manul.py -- second order coefficients a_i for the order 8 families F_{8,2}, K_{8,2}, M_{8,1} at their Dita points
(section 8.3 of invariants.tex). The families are affine, H(e) = H_0 o exp(i e Phi), with base points:
  M_{8,1}(a) at a = 1, deforming a;  K_{8,2}(a,b) at a = e(1/8), b = 1, deforming b;  F_{8,2}(a,b) at a = 1, b = e(1/8),
  deforming a. These base points have c_2,c_3 = 4,19 ; 4,19 ; 5,28.
For each family, level t and rotation sector w:
  1. the lowest weight fixed vectors of G_{H_0} at level t (filtered random block, dimension 8^t),
  2. the exact quadratic form a(v) = (1/2N) sum_xi ||(id (x) delta)(u^{otimes t})(v (x) xi)||^2 on them, and its eigenvalues,
  3. a check that T_t'(0) vanishes on these vectors.
A zero eigenvalue of a would be a fixed vector of the first order quantum group; positive values mean that these fixed
vectors disappear along the family. Tasks: levels 3,4,5 for the 3 families (quick), then level 6 for M_{8,1}, K_{8,2},
F_{8,2}, as time allows (about 2 to 3 hours each). Memory: at most about 3 GB.
Run:  python3 manul.py      Log: manul-output/manul.log ; numbers: manul-output/manul.json
Each sector is saved as soon as it is done, so running again resumes. Stops by itself after 7 hours."""
import os, sys, json, time, threading
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import matrices as m
from lwsector import lw_fixed

OUT = os.path.join(HERE, "manul-output"); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "manul.log"), "a"); JS = os.path.join(OUT, "manul.json")
try:
    res = json.load(open(JS))
except Exception:
    res = {}
T0 = time.time()
def say(s):
    s = "[%s | %5.1f min] %s" % (time.strftime("%H:%M:%S"), (time.time() - T0) / 60, s)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def watchdog():
    time.sleep(7 * 3600); say("7 hours reached, stopping. Results so far are in " + JS); os._exit(0)
threading.Thread(target=watchdog, daemon=True).start()

N = 8
e8 = np.exp(2j * np.pi / 8)
FAM = {"M81": (lambda p: m.M81(*p), np.array([1.0 + 0j]), np.array([1.0])),
       "K82": (lambda p: m.K82(*p), np.array([e8, 1.0 + 0j]), np.array([0.0, 1.0])),
       "F82": (lambda p: m.F82(*p), np.array([1.0 + 0j, e8]), np.array([1.0, 0.0]))}
def base(name):
    """base point H0 and the derivative data Pm, Pd, for the path p0 * exp(i e d)"""
    f, p0, d = FAM[name]; h = 1e-5
    Hf = lambda e: f(p0 * np.exp(1j * e * d))
    H0 = Hf(0.0)
    H1 = (8 * (Hf(h) - Hf(-h)) - (Hf(2 * h) - Hf(-2 * h))) / (12 * h)
    Phi = (H1 / (1j * H0)).real
    xi = lambda i, j: H0[i] * np.conj(H0[j]) / np.sqrt(N)
    Pm = np.array([[np.outer(xi(i, j), xi(i, j).conj()) for j in range(N)] for i in range(N)])
    Dl = [np.diag(Phi[i]) for i in range(N)]
    Pd = np.array([[1j * ((Dl[i] - Dl[j]) @ Pm[i, j] - Pm[i, j] @ (Dl[i] - Dl[j])) for j in range(N)] for i in range(N)])
    return H0, Pm, Pd

def apply_leg(X, M, s):
    Y = np.moveaxis(X, s, 0); sh = Y.shape; Y = Y.reshape(N, -1, N)
    Z = np.einsum('ijab,jrb->ira', M, Y)
    return np.moveaxis(Z.reshape(sh), 0, s)
def dprop(X, t, Pm, Pd):
    Xd = np.zeros_like(X)
    for s in reversed(range(t)):
        Xd = apply_leg(Xd, Pm, s) + apply_leg(X, Pd, s)
        X = apply_leg(X, Pm, s)
    return Xd, X
def a_form(L, t, Pm, Pd):
    f = L.shape[1]; G = np.zeros((f, f), complex); worst = 0.0; colw = np.zeros(f)
    Tp = np.zeros((N ** t, f), complex)       # T_t'(0) applied to the fixed vectors
    for x in range(N):
        C = np.empty((N ** (t + 1), f), np.complex64)
        for c in range(f):
            X = np.multiply.outer(L[:, c].reshape((N,) * t), np.eye(N)[x]).astype(complex)
            Xd, UX = dprop(X, t, Pm, Pd); colw[c] = max(colw[c], float(np.abs(UX - X).max()))
            C[:, c] = Xd.reshape(-1); Tp[:, c] += Xd[..., x].reshape(-1) / N
        Cd = C.astype(complex); G += Cd.conj().T @ Cd
    good = colw < 1e-6                       # drop spurious vectors (not fixed), seen at level 3
    G = G[np.ix_(good, good)]; Lg = L[:, good]; Tp = Tp[:, good]
    worst = float(colw[good].max()) if good.any() else 0.0
    tprime = float(np.abs(Lg.conj().T @ Tp).max()) if good.any() else 0.0
    return G / (2 * N), worst, tprime, int((~good).sum())

say("start; numpy %s, python %s. Second order coefficients, order 8 families at Dita points." % (np.__version__, sys.version.split()[0]))
TASKS = [(f, t) for f in ("M81", "K82", "F82") for t in (3, 4, 5)] + [("M81", 6), ("K82", 6), ("F82", 6)]
cache = {}
for p, t in TASKS:
    if p not in cache:
        H0, Pm, Pd = base(p); cache[p] = (H0, Pm, Pd)
        say("base point %s: |H0| - 1 = %.1e, H0 H0^* - N = %.1e" % (p, np.abs(np.abs(H0) - 1).max(), np.abs(H0 @ H0.conj().T - N * np.eye(N)).max()))
    H0, Pm, Pd = cache[p]
    allA = []; tag = p
    for w in range(t):
        key = "%s, level %d, sector %d" % (p, t, w)
        if key in res:
            say("%s already done" % key); allA += res[key]["a"]; continue
        tt = time.time(); npz = os.path.join(OUT, "%s_L%d_%d.npz" % (tag, t, w))
        if os.path.exists(npz):
            V = np.load(npz)["V"]
        else:
            b = 48
            while True:
                V, resid, _ = lw_fixed(H0, t, w, b=b)
                if V.shape[1] <= b - 8: break
                b *= 2
            np.savez_compressed(npz, V=V)
        say("%s: %d fixed vectors (%.0f s)" % (key, V.shape[1], time.time() - tt))
        if V.shape[1] == 0:
            res[key] = {"f": 0, "a": []}; json.dump(res, open(JS, "w"), indent=1); continue
        K, worst, tprime, dropped = a_form(V, t, Pm, Pd)
        if dropped: say("%s: %d spurious vector(s) dropped" % (key, dropped))
        if K.shape[0] == 0:
            res[key] = {"f": 0, "a": [], "dropped": dropped}; json.dump(res, open(JS, "w"), indent=1); continue
        a = np.sort(np.linalg.eigvalsh((K + K.conj().T) / 2)).tolist(); allA += a
        res[key] = {"f": int(K.shape[0]), "dropped": dropped, "a": a, "fixed_check": worst, "Tprime_check": tprime, "seconds": round(time.time() - tt)}
        json.dump(res, open(JS, "w"), indent=1)
        say("%s: a_min = %.10f, a_max = %.6f, fixed check %.1e, T' check %.1e, %.0f s" % (key, a[0], a[-1], worst, tprime, time.time() - tt))
    if allA:
        say("%s LEVEL %d: f = %d, a_min = %.10f, smallest values %s" % (p, t, len(allA), min(allA), np.round(np.sort(allA)[:6], 6).tolist()))
    else:
        say("%s LEVEL %d: no fixed vectors" % (p, t))
say("all done.")
