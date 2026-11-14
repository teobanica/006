"""caracal.py / lebanner.py (same program; default family z1 for caracal, all for lebanner) -- word certificate at an odd level k (default 15) for the overgroups of D (generic F_2 x_Q F_3).

What it does:
  1. enumerates the identity words of length k (closed walks with pauses on the kagome lattice), by meeting in the middle,
     and keeps one representative per class under rotation, reversal, block interchange and character conjugation;
  2. computes the Jones-Wenzl law nu_l on the words of length l = (k-1)/2 (floating point, block by block);
  3. solves the linear program: maximize t with score_s(w) - k mu(s) >= t for all identity words, |s| <= 1 (in dual form,
     by an interior point method, with generation of classes when there are too many of them);
  4. if t > 0, looks for a sparse certificate (minimal sum of |weights|), rounds it to integers, and checks it on all classes.
Everything is floating point: the exact verification of the table found is done afterwards, by overgroup.py.

Usage:   python3 caracal.py                 (level 15, default family, 6.5 hours at most)
         python3 caracal.py 13              (another odd level)
         python3 caracal.py 15 6.5 nz       (level, time budget in hours, family of patterns carrying weights:
                                             all = all the words, z1 = at most one letter 0, nz = no letter 0)
Needs numpy and scipy >= 1.7. Output folder: <name>-output/, with <name>-k.log (progress) and <name>-k.json (result);
the classes are saved in <name>-k-classes.npy, so a second run does not enumerate again. The classes are also taken
up to the reversal of the words, which is a symmetry of the problem, and the table found is symmetric under it."""
import sys, os, time, json, itertools
import numpy as np

K = int(sys.argv[1]) if len(sys.argv) > 1 else 15
HOURS = float(sys.argv[2]) if len(sys.argv) > 2 else 6.5
NAME = os.path.splitext(os.path.basename(__file__))[0]
FAMILY = sys.argv[3] if len(sys.argv) > 3 else ("all" if NAME == "lebanner" else "z1")
TAG = "%s-%d%s" % (NAME, K, "" if FAMILY == "all" else "-" + FAMILY)
assert K % 2 == 1 and K >= 5
L = (K - 1) // 2
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), NAME + "-output"); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, TAG + ".log"), "a"); T0 = time.time()
def say(s):
    s = "[%7.0fs] %s" % (time.time() - T0, s); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

FL = np.array([0, 3, 4, 1, 2]); CJ = np.array([0, 2, 1, 4, 3]); BLK = np.array([0, 0, 0, 1, 1]); CH = np.array([0, 1, 2, 1, 2])
NORM = (6, 3, 3, 3, 3)

def all_digits(m):
    idx = np.arange(5 ** m, dtype=np.int64); dig = np.empty((5 ** m, m), np.int8)
    for t in range(m - 1, -1, -1): dig[:, t] = idx % 5; idx //= 5
    return dig

def to_digits(codes, m):
    idx = codes.copy(); dig = np.empty((len(codes), m), np.int8)
    for t in range(m - 1, -1, -1): dig[:, t] = idx % 5; idx //= 5
    return dig

def signatures(dig):
    """walk endpoint d and the 3 counters, for each word"""
    n, m = dig.shape; d = np.zeros(n, np.int64); cnt = np.zeros((n, 3), np.int64); rows = np.arange(n)
    for t in range(m):
        x = dig[:, t]; e = (d + CH[x]) % 3; b = BLK[x] == 1
        np.add.at(cnt, (rows[b], d[b]), 1); np.add.at(cnt, (rows[b], e[b]), -1); d = e
    return d, cnt

def key(d, cnt, off=40, B=81):
    return d + 3 * ((cnt[:, 0] + off) + B * ((cnt[:, 1] + off) + B * (cnt[:, 2] + off)))

def code_of(dig, cols=None):
    c = np.zeros(len(dig), np.int64)
    for j in (range(dig.shape[1]) if cols is None else cols): c = c * 5 + dig[:, j]
    return c

# ---------------------------------------------------------------- 1. classes of identity words
def classes(k):
    fn = os.path.join(OUT, "%s-%d-classes.npy" % (NAME, k))
    if os.path.exists(fn):
        reps = np.load(fn); say("classes loaded from file: %d" % len(reps)); return reps, None
    a, b = (k + 1) // 2, k // 2
    da, db = all_digits(a), all_digits(b)
    (pd, pc), (sd, sc) = signatures(da), signatures(db)
    # required suffix signature for a prefix (d, cnt): d' = -d, cnt'[c] = -cnt[c + d]
    need = np.empty_like(pc)
    for c in range(3): need[:, c] = -pc[np.arange(len(pc)), (c + pd) % 3]
    pkey = key((-pd) % 3, need); skey = key(sd, sc)
    so = np.argsort(skey, kind="stable"); sk = skey[so]; uk, st = np.unique(sk, return_index=True); en = np.r_[st[1:], len(sk)]
    pos = {int(x): i for i, x in enumerate(uk)}
    po = np.argsort(pkey, kind="stable"); pk = pkey[po]; upk, pst = np.unique(pk, return_index=True); pen = np.r_[pst[1:], len(pk)]
    maps = (("F", FL), ("C", CJ), ("FC", FL[CJ]))
    Tb = {name: code_of(tab[db]) for name, tab in maps}; Ta = {name: code_of(tab[da]) for name, tab in maps}
    Rb = {name: code_of(tab[db][:, ::-1]) for name, tab in (("I", np.arange(5)),) + maps}        # reversed halves
    Ra = {name: code_of(tab[da][:, ::-1]) for name, tab in (("I", np.arange(5)),) + maps}
    pa, pb, top = 5 ** a, 5 ** b, 5 ** (k - 1)
    total, keep, buf, nbuf = 0, [], [], 0
    def flush():
        nonlocal buf, nbuf
        if not buf: return
        codes = np.concatenate(buf); buf, nbuf = [], 0
        hi, lo = codes // pb, codes % pb; best = codes.copy()
        variants = [codes] + [Ta[n][hi] * pb + Tb[n][lo] for n in ("F", "C", "FC")] + [Rb[n][lo] * pa + Ra[n][hi] for n in ("I", "F", "C", "FC")]
        for v in variants:
            v = v.copy()
            for r in range(k):
                np.minimum(best, v, out=best)
                v = (v % top) * 5 + v // top
        keep.append(codes[codes == best])
    t1 = time.time()
    for gi in range(len(upk)):
        j = pos.get(int(upk[gi]))
        if j is None: continue
        P = po[pst[gi]:pen[gi]]; S = so[st[j]:en[j]]
        for i0 in range(0, len(P), 4000):
            c = (P[i0:i0 + 4000, None] * pb + S[None, :]).ravel(); buf.append(c); nbuf += len(c); total += len(c)
            if nbuf >= 3000000: flush()
        if time.time() - t1 > 600:
            t1 = time.time(); say("   enumeration: %d of %d prefix groups, %d identity words so far" % (gi, len(upk), total))
    flush()
    reps = np.sort(np.concatenate(keep)); np.save(fn, reps)
    say("level %d: %d identity words, %d classes" % (k, total, len(reps))); return reps, total

# ---------------------------------------------------------------- 2. the Jones-Wenzl law, floating point, block by block
def product(a, b):
    if a == 0 and b == 0: return ((-1, 1.0),)
    if a == 0 or b == 0:
        x = a + b; return ((x, 1.0 if BLK[x] == 0 else -1.0),)
    if BLK[a] != BLK[b]: return ()
    j = int(BLK[a]); ca, cb = (1 if CH[a] == 1 else -1), (1 if CH[b] == 1 else -1); c = ca + cb
    if c == 0: return ((-1, 0.5), (0, 0.5 if j == 0 else -0.5))
    return ((1 + 2 * j + (1 if c == 2 else 0), 1.0),)        # c = 2 is the character -1, c = -2 the character 1

def jw_law(l):
    dig = all_digits(l); d, cnt = signatures(dig); ky = key(d, cnt); order = np.argsort(ky, kind="stable")
    uk, st = np.unique(ky[order], return_index=True); en = np.r_[st[1:], len(order)]
    wt = np.ones(5 ** l)
    for t in range(l): wt *= np.array(NORM)[dig[:, t]]
    nu = np.zeros(5 ** l); dim = 0.0; PROD = [[product(a, b) for b in range(5)] for a in range(5)]
    for g in range(len(uk)):
        idx = order[st[g]:en[g]]; n = len(idx); rows = {}; ri, ci, vi = [], [], []
        for col, w in enumerate(idx):
            word = dig[w]
            for t in range(l - 1):
                pr = PROD[word[t]][word[t + 1]]
                if not pr: continue
                rest = (t,) + tuple(word[:t]) + tuple(word[t + 2:])
                for m, v in pr:
                    r = rows.setdefault(rest + (m,), len(rows)); ri.append(r); ci.append(col); vi.append(v / np.sqrt(wt[w]))
        if not rows: nu[idx] = 1.0; dim += n; continue
        M = np.zeros((len(rows), n)); np.add.at(M, (ri, ci), vi)
        G = M.T @ M; ev, V = np.linalg.eigh(G); rng_ = ev > 1e-10 * max(ev.max(), 1.0)
        p = 1.0 - (V[:, rng_] ** 2).sum(axis=1); nu[idx] = p; dim += p.sum()
    a_, b_ = 1, 5
    for _ in range(l - 1): a_, b_ = b_, 4 * b_ - a_
    say("Jones-Wenzl law on %d-letter words: dimension %.6f (expected %d)" % (l, dim, b_))
    return nu / b_

# ---------------------------------------------------------------- 3, 4. linear programs by row generation
def main():
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, hstack, vstack, identity
    k, l = K, L
    reps, total = classes(k); m = len(reps)
    dl = all_digits(l); dr = dl[:, ::-1]
    wc = np.minimum.reduce([code_of(g) for g in (dl, FL[dl], CJ[dl], FL[CJ[dl]], dr, FL[dr], CJ[dr], FL[CJ[dr]])])
    nzero = (dl == 0).sum(axis=1); allowed = {"all": nzero >= 0, "z1": nzero <= 1, "nz": nzero == 0}[FAMILY]
    wc = np.where(allowed, wc, -1)                      # the words outside the family carry no weight: class number 0, weight 0
    ucl, cidx = np.unique(np.r_[-1, wc], return_inverse=True); cidx = cidx[1:].astype(np.int32); q = len(ucl)
    nu = jw_law(l); b = np.zeros(q); np.add.at(b, cidx, nu); b[0] = 0.0
    Wn = np.empty((m, k), np.int32 if q > 32000 else np.int16)
    for i0 in range(0, m, 2000000):
        dg = to_digits(reps[i0:i0 + 2000000], k)
        for t in range(k): Wn[i0:i0 + 2000000, t] = cidx[code_of(dg, [(t + j) % k for j in range(l)])]
    say("%d classes, %d observables (classes of windows of %d letters, family %s)" % (m, q - 1, l, FAMILY))
    def values(s):                                      # score_s(w) for all classes
        s = s.copy(); s[0] = 0.0; out = np.empty(m)
        for i0 in range(0, m, 1000000): out[i0:i0 + 1000000] = s[Wn[i0:i0 + 1000000]].sum(axis=1)
        return out
    def rows_matrix(act):
        r = np.repeat(np.arange(len(act)), k); c = Wn[act].ravel().astype(np.int64); d = (c != 0).astype(float)   # column 0 = outside the family
        return csr_matrix((d, (r, c)), shape=(len(act), q))
    CAP = int(os.environ.get("CAP", 700000))                                        # maximal number of classes in a linear program
    rng = np.random.default_rng(0); act = np.arange(m) if m <= CAP else np.unique(rng.integers(0, m, size=CAP // 2))
    deadline = T0 + HOURS * 3600; result = {"level": k, "l": l, "family": FAMILY, "classes": int(m), "identity_words": total}
    one = lambda n: csr_matrix(np.ones((1, n)))
    left = lambda: max(60.0, deadline - time.time())
    def margin_lp(act):
        """dual form: minimize the l1 distance between the window statistics of a probability p on the classes and k nu;
        the multipliers of the q equations are the weights s, and the optimum is the margin"""
        na = len(act); AT = rows_matrix(act).T.tocsr()
        Aeq = vstack([hstack([AT, identity(q), -identity(q)]), hstack([one(na), csr_matrix((1, 2 * q))])]).tocsr()
        res = linprog(np.r_[np.zeros(na), np.ones(2 * q)], A_eq=Aeq, b_eq=np.r_[k * b, 1], bounds=(0, None), method="highs-ipm", options={"time_limit": left()})
        if res.status != 0: raise RuntimeError("margin program: " + str(res.message))
        return float(res.fun), -res.eqlin.marginals[:q], res.x[:na] > 1e-11
    def sparse_lp(act):
        """dual form of: minimize the sum of |s| with score - k mu >= 1 on the classes"""
        na = len(act); AT = rows_matrix(act).T.tocsr(); col = csr_matrix((k * b)[:, None])
        Aub = vstack([hstack([AT, -col]), hstack([-AT, col])]).tocsr(); Aeq = hstack([one(na), csr_matrix([[-1.0]])]).tocsr()
        res = linprog(np.r_[-np.ones(na), 0], A_ub=Aub, b_ub=np.ones(2 * q), A_eq=Aeq, b_eq=[0], bounds=(0, None), method="highs-ipm", options={"time_limit": left()})
        if res.status != 0: raise RuntimeError("sparse program: " + str(res.message))
        mg = res.ineqlin.marginals; return float(-res.fun), -(mg[:q] - mg[q:]), res.x[:na] > 1e-11
    def renew(act, supp, val):
        """the classes of the next program: the support of the current solution, and the classes with the smallest values"""
        keep = act[supp]; n = min(m, CAP) - len(keep); low = np.argpartition(val, n)[:n] if n < m else np.arange(m)
        return np.union1d(keep, low)
    # ---- margin program, with generation of classes when there are many
    t, s, it, worst = None, None, 0, -1.0
    while True:
        it += 1; t, s, supp = margin_lp(act); val = values(s) - k * (b @ s); worst = float(val.min())
        say("margin program, round %d: %d classes, optimum %.6f, worst class %.6f" % (it, len(act), t, worst))
        np.save(os.path.join(OUT, TAG + "-weights.npy"), s)
        if worst >= t - 1e-6 or t < 1e-7: break
        if time.time() > deadline: say("time budget reached in the margin program"); break
        act = renew(act, supp, val)
    result["margin"] = t; result["margin_proved_over_all_classes"] = bool(worst >= t - 1e-6)
    json.dump(result, open(os.path.join(OUT, TAG + ".json"), "w"), indent=1)
    if t < 1e-7 or not result["margin_proved_over_all_classes"]:
        say("no certificate established: last optimum %.6f, worst class %.6f" % (t, worst)); return
    say("LEVEL %d: margin %.6f > 0 over all the classes: a word certificate exists (floating point)" % (k, t))
    # ---- sparse program
    it = 0; s1 = None
    while True:
        it += 1; tot, s1, supp = sparse_lp(act); val = values(s1) - k * (b @ s1); worst = float(val.min())
        say("sparse program, round %d: %d classes, sum of |weights| %.4f, %d nonzero, worst class %.6f" % (it, len(act), np.abs(s1).sum(), (np.abs(s1) > 1e-9).sum(), worst))
        if worst >= 1 - 1e-5: break
        if time.time() > deadline: say("time budget reached in the sparse program"); s1 = None; break
        act = renew(act, supp, val)
    cand = [("margin", s)] + ([("sparse", s1)] if s1 is not None else [])
    for name, sv in reversed(cand):
        for Q in [1, 2, 3, 4, 5, 6, 8, 10, 12, 20, 30, 60, 120, 1000, 10000, 100000]:
            si = np.rint(sv / np.abs(sv).max() * Q); mar = float((values(si) - k * (b @ si)).min())
            if mar > 1e-6 * Q:
                si[0] = 0; tab = {"".join(str(int(x)) for x in dl[np.where(wc == ucl[j])[0][0]]): int(si[j]) for j in np.where(si != 0)[0]}
                result.update({"table_from": name, "scale": Q, "integer_margin": mar, "mu": float(b @ si), "weights": tab})
                json.dump(result, open(os.path.join(OUT, TAG + ".json"), "w"), indent=1)
                say("integer table from the %s program: %d patterns, weights of size <= %d, minimum of score - k mu = %.6f" % (name, len(tab), Q, mar))
                say("DONE. Result in %s-output/%s.json" % (NAME, TAG)); return
    say("rounding failed; the floating point margin is in the json file")

if __name__ == "__main__":
    try:
        import scipy
        say("%s, level %d, family %s, python %s, numpy %s, scipy %s" % (NAME, K, FAMILY, sys.version.split()[0], np.__version__, scipy.__version__))
        main()
    except Exception as e:
        import traceback; say("ERROR: " + traceback.format_exc()); raise
