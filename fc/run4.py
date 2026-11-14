import sys, numpy as np, fc
from scipy.optimize import minimize
n, m, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
prevTL, prevFC = {}, {}
for k in range(1, K + 1):
    if k <= 2:
        S, TL, FC, V = fc.spaces(n, m, k); Vlin = V[:, :0]
    else:
        TL, FC, V, Vlin, _ = fc.level_test(n, m, k, prevTL)
    prevTL[k], prevFC[k] = TL, FC
    msg = "n=%d m=%d level %d: TL %d, FC %d, FC-TL %d, after linear %d" % (n, m, k, TL.shape[1], FC.shape[1], V.shape[1], Vlin.shape[1])
    if Vlin.shape[1]:
        F = fc.quad_forms(n, m, k, Vlin, prevTL, prevFC) + fc.merge_forms(n, m, k, Vlin, prevTL)
        F = [(f + f.T) / 2 for f in F if np.abs(f).max() > 1e-10]
        d = Vlin.shape[1]; rng = np.random.default_rng(0); best = np.inf
        obj = lambda x: sum(((x / np.linalg.norm(x)) @ f @ (x / np.linalg.norm(x))) ** 2 for f in F)
        for t in range(20):
            r = minimize(obj, rng.standard_normal(d), method='BFGS'); best = min(best, r.fun)
        msg += ", %d forms, min sum q(w,w)^2 = %.3e  %s" % (len(F), best, "=> C_k = TL_k" if best > 1e-6 else "INCONCLUSIVE")
    else:
        msg += "  => C_k = TL_k" if k >= 3 else ""
    print(msg, flush=True)
    if Vlin.shape[1] and best <= 1e-6: break
