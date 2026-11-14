# sanity check: replace FC by the span of ALL partitions (category of S_N). Since S_N is a genuine
# intermediate quantum group with C_2 = TL_2, the test must NOT conclude C_4 = TL_4.
import numpy as np, fc
from scipy.optimize import minimize
N = 6; n, m = 6, 1
def spacesP(k):
    S = fc.Spin(n, m, k)
    tl = np.stack([S.xi(s, s) for s in fc.NC(k)], 1)
    allp = [tuple(tuple(b) for b in p) for p in fc.set_partitions(k)]
    P = np.stack([S.xi(p, p) for p in allp], 1)
    TL = fc.orth(tl); FC = fc.orth(P); V = fc.orth(FC - TL @ (TL.T @ FC))
    return S, TL, FC, V
fc.spaces = lambda n_, m_, k: spacesP(k)
prevTL, prevFC = {}, {}
for k in range(1, 6):
    S, TL, FC, V = spacesP(k)
    if k <= 2: Vlin = V[:, :0]
    else: _, _, _, Vlin, _ = fc.level_test(n, m, k, prevTL)
    prevTL[k], prevFC[k] = TL, FC
    msg = "S_N sanity, level %d: TL %d, P %d, P-TL %d, after linear %d" % (k, TL.shape[1], FC.shape[1], V.shape[1], Vlin.shape[1])
    if Vlin.shape[1]:
        F = fc.quad_forms(n, m, k, Vlin, prevTL, prevFC) + fc.merge_forms(n, m, k, Vlin, prevTL)
        F = [(f + f.T) / 2 for f in F if np.abs(f).max() > 1e-10]
        d = Vlin.shape[1]; rng = np.random.default_rng(0); best = np.inf
        obj = lambda x: sum(((x / np.linalg.norm(x)) @ f @ (x / np.linalg.norm(x))) ** 2 for f in F)
        for t in range(20):
            r = minimize(obj, rng.standard_normal(d), method='BFGS'); best = min(best, r.fun)
        msg += ", %d forms, min = %.3e" % (len(F), best)
    print(msg, flush=True)
    if k == 4: break
