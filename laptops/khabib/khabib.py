"""khabib.py -- completes the previous order 8 run: second order coefficients a_i for F_{8,2} at its Dita point
(a = 1, b = e(1/8), deforming a), at level 6, in the rotation sectors 2, 3, 4, 5 (sectors 0, 1 were done before).
For each sector: the lowest weight fixed vectors at level 6 (dimension 8^6), the exact quadratic form a on them, and a
check that T_6'(0) vanishes on them. Expected time: about 6 hours. Memory: at most about 3 GB.
Run:  python3 khabib.py      Log: khabib-output/khabib.log ; numbers: khabib-output/khabib.json
Each sector is saved as soon as it is done, so running again resumes. Stops by itself after 7 hours."""
import os, sys, json, time, threading
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import matrices as m
from lwsector import lw_fixed

OUT = os.path.join(HERE, "khabib-output"); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "khabib.log"), "a"); JS = os.path.join(OUT, "khabib.json")
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
TASKS = [("F82", 6)]
SECTORS = {("F82", 6): [2, 3, 4, 5]}   # sectors 0, 1 were done by the previous run
cache = {}
for p, t in TASKS:
    if p not in cache:
        H0, Pm, Pd = base(p); cache[p] = (H0, Pm, Pd)
        say("base point %s: |H0| - 1 = %.1e, H0 H0^* - N = %.1e" % (p, np.abs(np.abs(H0) - 1).max(), np.abs(H0 @ H0.conj().T - N * np.eye(N)).max()))
    H0, Pm, Pd = cache[p]
    allA = []; tag = p
    for w in SECTORS.get((p, t), range(t)):
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
