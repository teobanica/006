"""Exact check of the certificates for the overgroups of D (generic F_2 x_Q F_3), levels 3 to 7.
For each level k: identity words, the space L_k = Fix(D^{ok}) cap Lambda_k in unnormalized Fourier coordinates
(exact kernel over Q), the reference mean mu = Tr(p_l s)/d_l, and the test: either the cyclic score is >= 0 > k mu
on all identity words (levels 5, 7), or the form S - k mu is positive definite on L_k (levels 3, 4, 6).
Output: seven.log"""
import itertools, sys, time
from fractions import Fraction as Fr
import words as W
from linalg import kernel, pivots

out = open("seven.log", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n"); out.flush()

# ---------------------------------------------------------------- observables, diagonal on Fourier words
def orbit(pat):
    o = set(); 
    for a in (pat, W.flip(pat)):
        for b in (a, W.conj(a)): o.add(b)
    return o

def table(weights):
    t = {}
    for pat, v in weights.items():
        for q in orbit(tuple(int(c) for c in pat)):
            assert t.get(q, v) == v; t[q] = Fr(v)
    return t

S3 = {(0,): Fr(1)}
EPS = (0, 1, 1, -1, -1)
S4 = {(a, b): Fr(1 + EPS[a] * EPS[b], 2) for a in range(5) for b in range(5)}     # same triple, compressed
S5 = table({"12": 2, "13": -1, "14": -1})
S7 = table({"013": -3, "014": -6, "102": 24, "111": 15, "123": 6, "124": 9, "130": -9, "131": 8, "132": -15,
            "133": -6, "134": 3, "140": -6, "141": -12, "142": -9, "143": 6})
def beta(h, l): return Fr(1, 2) if l == 0 else Fr(int(W.BLOCK[l] == h))
S6 = {}
for (a, b, h) in [(1, 0, 1), (1, 4, 0)]:
    for (a2, b2, h2) in {(a, b, h), (W.FLIP[a], W.FLIP[b], 1 - h), (W.CONJ[a], W.CONJ[b], h),
                         (W.CONJ[W.FLIP[a]], W.CONJ[W.FLIP[b]], 1 - h)}:
        for c in range(5):
            v = -2 * beta(h2, c)
            if v: S6[(a2, b2, c)] = S6.get((a2, b2, c), 0) + v
OBS = {3: (1, S3), 4: (2, S4), 5: (2, S5), 6: (3, S6), 7: (3, S7)}

def score(w, l, s):
    k = len(w); return sum(s.get(tuple(w[(t + j) % k] for j in range(l)), 0) for t in range(k))

# ---------------------------------------------------------------- merge conditions
def merge_rows(words, coef, k, pairs):
    """equations: for each pair (t, t+1) of legs, the merged tensor vanishes. words: iterable; coef(w) -> (column, sign) or None"""
    rows = {}
    for w in words:
        cs = coef(w)
        if cs is None: continue
        col, sg = cs
        for (t, u) in pairs:
            pr = W.product(w[t], w[u])
            rest = tuple(w[j] for j in range(k) if j != t and j != u)
            for m, v in pr.items():
                key = (t, m, rest); r = rows.setdefault(key, {})
                nv = r.get(col, 0) + sg * v
                if nv == 0: r.pop(col, None)
                else: r[col] = nv
    return [r for r in rows.values() if r]

def jw_mean(l, s):
    """mu = Tr(p_l s)/d_l, with p_l the projection onto the zero-sum tensors killed by the l-1 consecutive merges"""
    allw = list(itertools.product(range(5), repeat=l)); idx = {w: i for i, w in enumerate(allw)}
    rows = merge_rows(allw, lambda w: (idx[w], 1), l, [(t, t + 1) for t in range(l - 1)])
    B = kernel(rows, len(allw)); d = len(B)
    wt = [W.weight(w) for w in allw]; sv = [s.get(w, Fr(0)) for w in allw]
    G = [[sum(wt[i] * a[i] * b[i] for i in range(len(allw)) if a[i] and b[i]) for b in B] for a in B]
    A = [[sum(wt[i] * sv[i] * a[i] * b[i] for i in range(len(allw)) if a[i] and b[i] and sv[i]) for b in B] for a in B]
    # trace of G^{-1} A, by solving G X = A
    n = d; M = [G[i][:] + A[i][:] for i in range(n)]
    for i in range(n):
        p = M[i][i]
        if p == 0:
            j = next(j for j in range(i + 1, n) if M[j][i] != 0); M[i], M[j] = M[j], M[i]; p = M[i][i]
        M[i] = [v / p for v in M[i]]
        for j in range(n):
            if j != i and M[j][i] != 0:
                f = M[j][i]; M[j] = [a - f * b for a, b in zip(M[j], M[i])]
    return d, sum(M[i][n + i] for i in range(n)) / d

# ---------------------------------------------------------------- main
levels = [int(a) for a in sys.argv[1:]] or [3, 4, 5, 6, 7]
for k in levels:
    t0 = time.time(); l, s = OBS[k]
    idw = W.identity_words(k); d, mu = jw_mean(l, s)
    sc = {w: score(w, l, s) for w in idw}
    say("level %d: %d identity words; l = %d, d_l = %d, mu = %s, k mu = %s; cyclic score on identity words: min %s, max %s"
        % (k, len(idw), l, d, mu, k * mu, min(sc.values()), max(sc.values())))
    if min(sc.values()) > k * mu:
        say("   score > k mu on all identity words: no first extra tensor at level %d (no linear algebra needed)" % k)
    reps, where = W.orbit_reps(k)
    if k <= 6:
        rows = merge_rows(idw, lambda w: where[w], k, [(t, (t + 1) % k) for t in range(k)])
        B = kernel(rows, len(reps))
        say("   dim L_%d = %d (unknowns %d, equations %d)" % (k, len(B), len(reps), len(rows)))
        # Gram and form on L_k: sum over all identity words; x_{flip w} = +- x_w and score, weight are flip invariant
        mult = [1 if W.flip(w) == w else 2 for w in reps]
        dg = [mult[i] * W.weight(w) * (sc[w] - k * mu) for i, w in enumerate(reps)]
        n = len(reps)
        M = [[sum(dg[i] * a[i] * b[i] for i in range(n) if a[i] and b[i]) for b in B] for a in B]
        pv = pivots(M)
        say("   form S - k mu on L_%d: %d pivots, %d positive, smallest %s" % (k, len(pv), sum(1 for p in pv if p > 0),
            "%.6g" % float(min(pv)) if pv else "-"))
    say("   time %.1fs" % (time.time() - t0))
