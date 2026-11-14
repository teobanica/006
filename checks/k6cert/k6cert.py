"""Jordan test of section 10.2 at the generic 2x3 Dita limit point of K_6, namely the limit of K_6(theta, 0, pi/6)
as theta -> 0+, the point (0,0,1) of section 10.3. The limit is obtained by Richardson extrapolation,
H0 = 2 H(t) - H(2t), with t = 1e-5, followed by the normalization of the entries to modulus 1.
For each level k and rotation sector w: lowest weight fixed vectors L_k^w (lwsector.py), then the best smallest
eigenvalue of a unit norm combination of the contraction forms (jordan_full.py, sdpcert.py).
The copy of lwsector.py used here keeps only the Ritz vectors with residual below 1e-6.
Usage: python3 k6cert.py KMIN KMAX [SECTORS, e.g. 4,5,6]     Output: k6cert-KMIN-KMAX.log, k6cert-KMIN-KMAX.json
(the runs in the paper: levels 3 to 6, then level 7 alone, with landing levels <= 5 at level 7, as in cheetah.py)"""
import sys, json, time, numpy as np
import matrices as m
from lwsector import lw_fixed
from jordan_full import forms
from sdpcert import real_forms_span, certify
KMIN, KMAX = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (3, 6)
SECTORS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
t = 1e-5
H0 = 2 * m.K6(t, 0, np.pi / 6) - m.K6(2 * t, 0, np.pi / 6); H0 = H0 / np.abs(H0)
tag = "%d-%d" % (KMIN, KMAX) + ("-s" + sys.argv[3].replace(",", "") if SECTORS else ""); log = open("k6cert-%s.log" % tag, "w"); res = {}
def say(s): print(s, flush=True); log.write(s + "\n"); log.flush()
say("limit point: max |HH* - 6| = %.1e" % np.abs(H0 @ H0.conj().T - 6 * np.eye(6)).max())
N = 6
for k in range(KMIN, KMAX + 1):
    tot, worst = 0, np.inf
    for w in (SECTORS if SECTORS is not None else range(k)):
        t0 = time.time(); b = 48
        while True:
            V, resid, nxt = lw_fixed(H0, k, w, b=b)
            if V.shape[1] <= b - 8: break
            b *= 2
        lam = float("inf")
        if V.shape[1]:
            F = forms(V, N, k, max_level=5 if k >= 7 else None)
            lam, _ = certify(real_forms_span(F, V.shape[1]), iters=3)
        tot += V.shape[1]; worst = min(worst, lam)
        res["%d,%d" % (k, w)] = {"dim": int(V.shape[1]), "lambda_min": lam, "residual": resid, "next": nxt, "seconds": round(time.time() - t0)}
        say("level %d sector %d: dim %d, residual %.1e, next Ritz %s, certificate %.4f, %d s"
            % (k, w, V.shape[1], resid, None if nxt is None else round(nxt, 4), lam, time.time() - t0))
    say("level %d: dim L_k = %d, worst certificate %.4f" % (k, tot, worst))
    json.dump(res, open("k6cert-%s.json" % tag, "w"), indent=1)
