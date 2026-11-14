"""cheetah.py -- Jordan test at level 7 for the generic 2x3 Dita point F_6(r,s), r = e^{0.4i}, s = e^{1.9i}
(section 6.9 of invariants.tex). For each rotation sector w = 0..6:
  1. lowest weight fixed vectors L_7^w = Fix(G_0^{otimes 7}) cap Lambda_7, by a filtered random block (dimension 6^7),
  2. certificate: best smallest eigenvalue of a positive combination of the contraction forms (landing levels <= 5);
     a positive value means that no planar algebra between TL and Fix(G_0) with trivial 2-boxes has new vectors there.
Run:  python3 cheetah.py      Log: cheetah-output/cheetah.log ; numbers: cheetah-output/cheetah.json
Each sector is saved as soon as it is done (vectors in cheetah-output/sector*.npz), so running again resumes.
Stops by itself after 7 hours. Memory: about 2 GB. Expected time: 1 to 2 hours."""
import os, sys, json, time, threading
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import matrices as m
from lwsector import lw_fixed
from jordan_full import forms
from sdpcert import real_forms_span, certify

OUT = os.path.join(HERE, "cheetah-output"); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "cheetah.log"), "a"); JS = os.path.join(OUT, "cheetah.json")
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

H = m.F6rs(np.exp(0.4j), np.exp(1.9j)); N = 6; k = 7
say("start; numpy %s, python %s. Jordan test, 2x3 Dita point, level 7, sectors 0..6." % (np.__version__, sys.version.split()[0]))
for w in range(k):
    key = "level 7, sector %d" % w
    if key in res:
        say("%s already done: dim %d, lambda_min %.3e" % (key, res[key]["dim"], res[key]["lambda_min"])); continue
    t = time.time(); npz = os.path.join(OUT, "sector%d.npz" % w)
    if os.path.exists(npz):
        V = np.load(npz)["V"]; say("%s: vectors reloaded, dim %d" % (key, V.shape[1]))
    else:
        b = 48
        while True:
            say("%s: filtered block of size %d" % (key, b))
            V, resid, nxt = lw_fixed(H, k, w, b=b)
            if V.shape[1] <= b - 8: break
            b *= 2
        np.savez_compressed(npz, V=V)
        say("%s: dim %d, residual %.1e, next Ritz value %s, %.0f s" % (key, V.shape[1], resid, None if nxt is None else round(nxt, 4), time.time() - t))
    if V.shape[1] == 0:
        lam = float("inf")
    else:
        F = forms(V, N, k, max_level=5)
        lam, y = certify(real_forms_span(F, V.shape[1]), iters=3)
    res[key] = {"dim": int(V.shape[1]), "lambda_min": float(lam), "seconds": round(time.time() - t)}
    json.dump(res, open(JS, "w"), indent=1)
    say("%s: certificate lambda_min = %.3e   %s" % (key, lam, "<<< CERTIFIED" if lam > 1e-6 else "<<< NO CERTIFICATE"))
say("all done; total dim f_7 = %d" % sum(r["dim"] for kk, r in res.items() if kk.startswith("level 7")))
