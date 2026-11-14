"""lyoto.py -- second order coefficients a_i for the Karlsson family K_6 at its Dita point (section 8.3 of
invariants.tex): base point H_0 = lim K_6(theta, pi/3, pi/6) as theta -> 0, which is F_2 (x)_Q F_3 with Q of order 12,
deformation along theta. For each level t and rotation sector w:
  1. the lowest weight fixed vectors of G_{H_0} at level t (filtered random block, dimension 6^t),
  2. the exact quadratic form a(v) = (1/2N) sum_xi ||(id (x) delta)(u^{otimes t})(v (x) xi)||^2 on them, and its eigenvalues.
Levels: 5 (quick check: minimum must be exactly 1), then 7. Expected time: about 1 to 2 hours. Memory: at most about 3 GB.
Run:  python3 lyoto.py      Log: lyoto-output/lyoto.log ; numbers: lyoto-output/lyoto.json
Each sector is saved as soon as it is done, so running again resumes. Stops by itself after 7 hours."""
import os, sys, json, time, threading
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import matrices as m
from lwsector import lw_fixed

OUT = os.path.join(HERE, "lyoto-output"); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "lyoto.log"), "a"); JS = os.path.join(OUT, "lyoto.json")
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
q0 = np.pi / 6 * np.array([0, 2, 1]); d = np.array([1.0, 0, 0])
Hf = lambda e: m.K6(*(q0 + e * d))
h = 1e-4
fp1, fm1, fp2, fm2 = Hf(h), Hf(-h), Hf(2 * h), Hf(-2 * h)
H0 = (4 * (fp1 + fm1) - (fp2 + fm2)) / 6
H1 = (8 * (fp1 - fm1) - (fp2 - fm2)) / (12 * h)
Phi = (H1 / (1j * H0)).real
xi = lambda i, j: H0[i] * np.conj(H0[j]) / np.sqrt(N)
Pm = np.array([[np.outer(xi(i, j), xi(i, j).conj()) for j in range(N)] for i in range(N)])
Dl = [np.diag(Phi[i]) for i in range(N)]
Pd = np.array([[1j * ((Dl[i] - Dl[j]) @ Pm[i, j] - Pm[i, j] @ (Dl[i] - Dl[j])) for j in range(N)] for i in range(N)])

def apply_leg(X, M, s):
    Y = np.moveaxis(X, s, 0); sh = Y.shape; Y = Y.reshape(N, -1, N)
    Z = np.einsum('ijab,jrb->ira', M, Y)
    return np.moveaxis(Z.reshape(sh), 0, s)
def dprop(X, t):
    Xd = np.zeros_like(X)
    for s in reversed(range(t)):
        Xd = apply_leg(Xd, Pm, s) + apply_leg(X, Pd, s)
        X = apply_leg(X, Pm, s)
    return Xd, X
def a_form(L, t):
    f = L.shape[1]; G = np.zeros((f, f), complex); worst = 0.0
    for x in range(N):
        C = np.empty((N ** (t + 1), f), np.complex64)
        for c in range(f):
            X = np.multiply.outer(L[:, c].reshape((N,) * t), np.eye(N)[x]).astype(complex)
            Xd, UX = dprop(X, t); worst = max(worst, float(np.abs(UX - X).max()))
            C[:, c] = Xd.reshape(-1)
        Cd = C.astype(complex); G += Cd.conj().T @ Cd
    return G / (2 * N), worst

say("start; numpy %s, python %s. Second order coefficients, K_6 at its Dita point, levels 5 and 7." % (np.__version__, sys.version.split()[0]))
say("base point check: |H0| - 1 = %.1e, H0 H0^* - N = %.1e" % (np.abs(np.abs(H0) - 1).max(), np.abs(H0 @ H0.conj().T - N * np.eye(N)).max()))
for t in [5, 7]:
    allA = []
    for w in range(t):
        key = "level %d, sector %d" % (t, w)
        if key in res:
            say("%s already done" % key); allA += res[key]["a"]; continue
        tt = time.time(); npz = os.path.join(OUT, "L%d_%d.npz" % (t, w))
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
        K, worst = a_form(V, t)
        a = np.sort(np.linalg.eigvalsh((K + K.conj().T) / 2)).tolist(); allA += a
        res[key] = {"f": int(V.shape[1]), "a": a, "fixed_check": worst, "seconds": round(time.time() - tt)}
        json.dump(res, open(JS, "w"), indent=1)
        say("%s: a_min = %.10f, a_max = %.6f, fixed check %.1e, %.0f s" % (key, a[0], a[-1], worst, time.time() - tt))
    say("LEVEL %d: f = %d, a_min = %.10f, smallest values %s" % (t, len(allA), min(allA), np.round(np.sort(allA)[:6], 6).tolist()))
say("all done.")
