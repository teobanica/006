import sys, numpy as np, fc
n, m, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
prevTL, prevFC = {}, {}
for k in range(1, K + 1):
    if k <= 2:
        S, TL, FC, V = fc.spaces(n, m, k); Vlin = V if k == 1 else V[:, :0]   # C_2 = TL_2 by hypothesis
    else:
        TL, FC, V, Vlin, _ = fc.level_test(n, m, k, prevTL)
    prevTL[k], prevFC[k] = TL, FC
    msg = "n=%d m=%d level %d: TL %d, FC %d, FC-TL %d, after linear %d" % (n, m, k, TL.shape[1], FC.shape[1], V.shape[1], Vlin.shape[1])
    if Vlin.shape[1] and k >= 3:
        F = fc.quad_forms(n, m, k, Vlin, prevTL, prevFC)
        F = [f for f in F if np.abs(f).max() > 1e-10]
        # does a nonzero w with all f(w,w)=0 exist?  minimize sum f(w,w)^2 on the unit sphere
        d = Vlin.shape[1]; best = np.inf
        rng = np.random.default_rng(0)
        from scipy.optimize import minimize
        def obj(x):
            x = x / np.linalg.norm(x); return sum((x @ f @ x) ** 2 for f in F)
        for t in range(20):
            r = minimize(obj, rng.standard_normal(d), method='BFGS'); best = min(best, r.fun)
        msg += ", %d nonzero quadratic forms, min sum q(w,w)^2 on unit sphere = %.3e" % (len(F), best)
    print(msg, flush=True)
