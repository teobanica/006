"""kodkod.py -- Karlsson family K_6: second order coefficients a_i at ALL the Dita points of the pi/6 grid
(section 8.3 of invariants.tex, "Other base points"). So far only (0,4,1), (0,0,1), (0,2,1), (0,1,1), (0,2,3),
(0,3,1), (3,0,2) were done, falling into 3 classes. Here:
  stage 1: scan the 216 grid points pi/6*(n1,n2,n3), keep the valid ones of Dita type (c_2 >= 3), points with
           n1 = 0 taken as limits theta -> 0; at each, levels 3, 4, 5 (about 1 minute per point);
  stage 2: level 6 at one representative of each class (same spectra up to a scalar factor);
  stage 3: level 7 at the representatives of the new classes, while time remains.
Deformation along theta in all cases. A negative or zero a_min, or a large T' check, would be news: it is flagged.
Expected time: 7 hours (stops by itself). Memory: at most about 2 GB.
Run:  python3 kodkod.py      Log: kodkod-output/kodkod.log ; numbers: kodkod-output/kodkod.json
Everything is saved as soon as computed, so running again resumes."""
import os, sys, json, time, threading
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import matrices as m
import ck
from lwsector import lw_fixed

OUT = os.path.join(HERE, "kodkod-output"); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "kodkod.log"), "a"); JS = os.path.join(OUT, "kodkod.json")
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

N = 6
d = np.array([1.0, 0, 0]); h = 1e-4
def base(p):
    """base point H0 (as a limit along theta) and the derivative data Pm, Pd"""
    q0 = np.pi / 6 * np.array(p, float)
    Hf = lambda e: m.K6(*(q0 + e * d))
    fp1, fm1, fp2, fm2 = Hf(h), Hf(-h), Hf(2 * h), Hf(-2 * h)
    H0 = (4 * (fp1 + fm1) - (fp2 + fm2)) / 6
    H1 = (8 * (fp1 - fm1) - (fp2 - fm2)) / (12 * h)
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


def save():
    json.dump(res, open(JS, "w"), indent=1)

def level(p, t, H0, Pm, Pd):
    """all rotation sectors of level t at point p; returns the list of a values (or None if a sector failed)"""
    allA = []; tag = "%d%d%d" % p; flags = []
    for w in range(t):
        key = "point %s, level %d, sector %d" % (p, t, w)
        if key in res:
            allA += res[key]["a"]; continue
        tt = time.time(); npz = os.path.join(OUT, "P%s_L%d_%d.npz" % (tag, t, w))
        if os.path.exists(npz):
            V = np.load(npz)["V"]
        else:
            b = 48
            while True:
                V, resid, _ = lw_fixed(H0, t, w, b=b)
                if V.shape[1] <= b - 8: break
                b *= 2
            if t >= 6: np.savez_compressed(npz, V=V)
        if V.shape[1] == 0:
            res[key] = {"f": 0, "a": []}; save(); continue
        K, worst, tprime, dropped = a_form(V, t, Pm, Pd)
        if K.shape[0] == 0:
            res[key] = {"f": 0, "a": [], "dropped": dropped}; save(); continue
        a = np.sort(np.linalg.eigvalsh((K + K.conj().T) / 2)).tolist(); allA += a
        res[key] = {"f": int(K.shape[0]), "dropped": dropped, "a": a, "fixed_check": worst, "Tprime_check": tprime, "seconds": round(time.time() - tt)}
        save()
        if t >= 6 or a[0] < 1e-6 or tprime > 1e-6:
            say("  %s: f = %d, a_min = %.10f, fixed check %.1e, T' check %.1e, %.0f s%s" % (key, K.shape[0], a[0], worst, tprime, time.time() - tt,
                "   <<< FLAG" if (a[0] < 1e-6 or tprime > 1e-6) else ""))
    allA = sorted(allA)
    say("POINT %s LEVEL %d: f = %d, a_min = %s, smallest %s" % (p, t, len(allA), ("%.10f" % allA[0]) if allA else "-", np.round(allA[:6], 6).tolist()))
    return allA

say("start; numpy %s, python %s. Second order coefficients, K_6 at all the Dita points of the grid." % (np.__version__, sys.version.split()[0]))
# ---- stage 1: scan
if "scan" not in res:
    pts = []
    for p in [(a, b, c) for a in range(6) for b in range(6) for c in range(6)]:
        try:
            with np.errstate(all="ignore"):
                H0, Pm, Pd = base(p)
            ok = np.isfinite(H0).all() and np.abs(np.abs(H0) - 1).max() < 1e-6 and np.abs(H0 @ H0.conj().T - N * np.eye(N)).max() < 1e-6 \
                 and np.isfinite(Pd).all()
        except Exception:
            ok = False
        if not ok: continue
        c2 = ck.ck(H0, 2)[0]
        if c2 >= 3:
            pts.append([list(p), int(c2), int(ck.ck(H0, 3)[0])])
    res["scan"] = pts; save()
pts = [tuple(x[0]) for x in res["scan"]]
say("stage 1: %d valid Dita points on the grid: %s" % (len(pts), [(tuple(x[0]), x[1], x[2]) for x in res["scan"]]))
C2 = {tuple(x[0]): x[1] for x in res["scan"]}
spec = {}
for p in pts:   # the Fourier-like points (c_2 >= 6) have huge fixed spaces: levels 3, 4 only
    H0, Pm, Pd = base(p)
    spec[p] = [level(p, t, H0, Pm, Pd) for t in ((3, 4, 5) if C2[p] < 6 else (3, 4))]
# ---- classes: same spectra at levels 3, 4, 5 up to a common scalar factor
def signature(S):
    flat = [x for L in S for x in L]
    if not flat or flat[0] <= 1e-9: return ("dims",) + tuple(len(L) for L in S)
    return tuple(len(L) for L in S) + tuple(np.round(np.array(flat) / flat[0], 5))
classes = {}
for p in pts:
    classes.setdefault(signature(spec[p]), []).append(p)
reps = []
for i, (sg, ps) in enumerate(classes.items()):
    fl = [x for L in spec[ps[0]] for x in L]
    say("CLASS %d: %d points %s; dims %s; a_min at levels 3,4,5 = %s" % (i + 1, len(ps), ps, [len(L) for L in spec[ps[0]]],
        [round(min(L), 6) if L else None for L in spec[ps[0]]]))
    reps.append(ps[0])
res["classes"] = [[list(q) for q in ps] for ps in classes.values()]; save()
# ---- stage 2 and 3
known = [(0, 4, 1), (0, 0, 1), (0, 2, 1), (0, 1, 1), (0, 2, 3), (0, 3, 1), (3, 0, 2)]
for t in (6, 7):
    for ps in classes.values():
        if t == 7 and any(q in known for q in ps):
            continue
        p = ps[0]
        if C2[p] >= 6: continue
        H0, Pm, Pd = base(p)
        level(p, t, H0, Pm, Pd)
say("all done.")
