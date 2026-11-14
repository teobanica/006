"""Search for word certificates over D at level k >= 4, by linear programming (floating point), and rounding to integers.
A certificate is a triple of integer tables on the Fourier words of length l = [(k-1)/2]:
   s_w(v): weight of the window v;  for even k also s_m0(v), s_m1(v): weight of the window v times the sign eps of the
   next, respectively previous, letter (eps = 0, 1, 1, -1, -1 on the letters 0..4), these being the merged observables.
The exact verification is done by overgroup.py, where the tables used in the paper are stored. Output: found.json (one entry per level)"""
import sys, os, json, itertools, time, numpy as np
from scipy.optimize import linprog
import words as W
from marginal import identity_words_np, jw_diag, code, FL, CJ, EPS


def problem(k):
    idw = identity_words_np(k); n = len(idw); l = (k - 1) // 2
    best = None
    for r in range(k):
        rot = np.roll(idw, -r, axis=1)
        for g in (rot, FL[rot], CJ[rot], FL[CJ[rot]]):
            c = code(g, range(k)); best = c if best is None else np.minimum(best, c)
    cls, inv = np.unique(best, return_inverse=True); m = len(cls)
    rep = np.zeros(m, np.int64); rep[inv] = np.arange(n); reps = idw[rep]
    v = np.array(list(itertools.product(range(5), repeat=l)), np.int8); nv = len(v); ar = np.arange(m)
    c0, cF, cC, cFC = (code(g, range(l)) for g in (v, FL[v], CJ[v], FL[CJ[v]]))
    nu, d = jw_diag(l)
    obs = []                                                # (kind, list of (word code, sign)), row, reference
    cnt = np.zeros((m, nv))
    for t in range(k): np.add.at(cnt, (ar, code(reps, [(t + j) % k for j in range(l)])), 1)
    seen = set()
    for i in range(nv):
        orb = sorted({c0[i], cF[i], cC[i], cFC[i]})
        if orb[0] in seen: continue
        seen.add(orb[0]); obs.append(("w", [(int(c), 1) for c in orb], cnt[:, orb].sum(axis=1), nu[orb].sum()))
    if k % 2 == 0:
        nu2, d2 = jw_diag(l + 1); v2 = np.array(list(itertools.product(range(5), repeat=l + 1)), np.int8)
        for side in (0, 1):
            cm = np.zeros((m, nv))
            for t in range(k):
                e = EPS[reps[:, (t + l) % k]] if side == 0 else EPS[reps[:, (t - 1) % k]]
                np.add.at(cm, (ar, code(reps, [(t + j) % k for j in range(l)])), e)
            ref = np.zeros(nv)
            if side == 0: np.add.at(ref, code(v2, range(l)), nu2 * EPS[v2[:, l]])
            else: np.add.at(ref, code(v2, range(1, l + 1)), nu2 * EPS[v2[:, 0]])
            seen = set()
            for i in range(nv):
                op = sorted({c0[i], cC[i]}); om = sorted({cF[i], cFC[i]}); key = min(op[0], om[0])
                if key in seen: continue
                seen.add(key)
                if set(op) == set(om): continue             # symmetric class: the antisymmetric observable vanishes
                row = cm[:, op].sum(axis=1) - cm[:, om].sum(axis=1); rf = ref[op].sum() - ref[om].sum()
                if np.abs(row).max() == 0 and abs(rf) < 1e-13: continue
                obs.append(("m%d" % side, [(int(c), 1) for c in op] + [(int(c), -1) for c in om], row, rf))
    return l, n, m, obs


def find(k, Q=None, log=print):
    t0 = time.time(); l, n, m, obs = problem(k); q = len(obs)
    A = np.array([o[2] for o in obs]); b = np.array([o[3] for o in obs]); D = (A - k * b[:, None]).T     # (classes) x (observables)
    # margin LP: maximize t with D s >= t, |s| <= 1
    res = linprog(np.r_[np.zeros(q), -1], A_ub=np.hstack([-D, np.ones((m, 1))]), b_ub=np.zeros(m),
                  bounds=[(-1, 1)] * q + [(None, None)], method="highs")
    t = -res.fun
    log("level %d: %d identity words, %d classes, l = %d, %d observables, margin LP optimum %.6f (%.0fs)" % (k, n, m, l, q, t, time.time() - t0))
    if t < 1e-7: return None
    # sparse LP: minimize |s|_1 with D s >= 1
    res2 = linprog(np.ones(2 * q), A_ub=-np.hstack([D, -D]), b_ub=-np.ones(m), bounds=[(0, None)] * (2 * q), method="highs")
    s = res2.x[:q] - res2.x[q:]
    for Qs in ([Q] if Q else [1, 2, 3, 4, 5, 6, 8, 10, 12, 20, 30, 60, 120, 1000, 10000]):
        si = np.rint(s / np.abs(s).max() * Qs).astype(np.int64)
        if (D @ si).min() > 1e-6 * Qs: break
    else:
        raise RuntimeError("rounding failed")
    log("   sparse certificate: %d nonzero weights of %d, integers of size <= %d, floating margin %.4f" % ((si != 0).sum(), q, Qs, (D @ si).min()))
    tab = {"w": {}, "m0": {}, "m1": {}}
    def word(c):
        w = []
        for _ in range(l): w.append(c % 5); c //= 5
        return "".join(str(x) for x in reversed(w))
    for (kind, members, _, _), val in zip(obs, si):
        if val:
            for c, sg in members: tab[kind][word(c)] = int(sg * val)
    found = json.load(open("found.json")) if os.path.exists("found.json") else {}
    found[str(k)] = {"level": k, "l": l, "w": tab["w"], "m0": tab["m0"], "m1": tab["m1"]}
    json.dump(found, open("found.json", "w"), indent=0)
    return tab


if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:]]: find(k)
