"""Exact verification of the word certificates for the quantum 2-transitive overgroups of D, the quantum group of a
generic Dita matrix F_2 x_Q F_3 (section 3). For each level k the program checks that there is no first extra tensor:
   level 3: the form S - 3 mu is positive definite on the space L_3 (dimension 1), S = number of letters 0;
   level k >= 4: an integer table s on the Fourier words of length L <= k/2 (for k = 2L, depending on the last letter
   only through its sign eps), with cyclic score > k mu on all the identity words of length k, mu = sum_v nu_L(v) s(v).
Tables: even k: the zigzag table s(Z, y) = eps(z) eps(y), Z zigzag of length L-1 with last letter z;
odd k = 5, 7, 9, 11, 13: the tables of the paper.
The identity words are enumerated completely for k <= 11, and for larger k the words containing a window of
negative weight are enumerated completely (the other words have score >= 0 > k mu).
Everything is exact: integers and Fractions. Usage: python3 overgroup.py [levels], default 3..14. Output: overgroup.log"""
import sys, os, json, itertools, time, numpy as np
from fractions import Fraction as Fr
import words as W
from linalg import kernel, pivots
from jw import nu_exact, nu_words, jw_dim, signature

BLK = np.array([0, 0, 0, 1, 1]); CH = np.array([0, 1, 2, 1, 2])
out = open("overgroup.log", "a")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n"); out.flush()

# ---------------------------------------------------------------- letters and tables
INV = W.CONJ                                   # inverse letter
def alt(a, b): return a != 0 and b != 0 and W.BLOCK[a] != W.BLOCK[b]
def zig(a, b): return alt(a, b) and W.CHAR[a] != W.CHAR[b]
def is_zig(w): return all(w) and all(zig(w[i], w[i + 1]) for i in range(len(w) - 1))
EPS = (0, 1, 1, -1, -1)

def zigzag_table(k):
    """even k = 2L: s(Z, y) = eps(z) eps(y) for Z a zigzag of length L-1 with last letter z"""
    L = k // 2; s = {}
    for z in itertools.product(range(1, 5), repeat=L - 1):
        if is_zig(z):
            for y in range(1, 5): s[z + (y,)] = EPS[z[-1]] * EPS[y]
    return L, s

def orbit_table(weights):
    t = {}
    for pat, v in weights.items():
        p = tuple(int(c) for c in pat)
        for q in {p, W.flip(p), W.conj(p), W.flip(W.conj(p))}: t[q] = v
    return len(next(iter(t))), t

PAPER = {5: {"12": 2, "13": -1, "14": -1},
         7: {"013": -3, "014": -6, "102": 24, "111": 15, "123": 6, "124": 9, "130": -9, "131": 8, "132": -15,
             "133": -6, "134": 3, "140": -6, "141": -12, "142": -9, "143": 6},
         9: {"1323": -1, "1324": -1, "1413": -1, "1414": -1, "1412": 2, "1421": 2},
         11: {"01323": -1, "11323": -1, "12323": 1, "12324": 1, "12413": 1, "13142": -1, "13232": -1, "13234": 1,
              "13241": -1, "13242": -1, "13243": 1, "14102": 2, "14111": 2, "14130": -1, "14132": -1, "14133": -1,
              "14141": -2, "14142": -1, "14143": 1, "14201": 2, "14222": 2, "14231": -1, "14232": -1, "14242": 2},
         13: {"132323": -1, "132324": -1, "132413": -1, "132414": -1, "141323": -1, "141324": -1, "141413": -1, "141414": -1,
              "141232": 2, "141241": 2, "142132": 2, "142141": 2}}

def table_for(k):
    if k % 2 == 0: return ("zigzag",) + zigzag_table(k)
    if k in PAPER: return ("of the paper",) + orbit_table(PAPER[k])
    d = json.load(open("found.json"))[str(k)]                       # a table found by certify.py
    return ("found.json", d["l"], {tuple(int(c) for c in w): v for w, v in d["w"].items()})

# ---------------------------------------------------------------- enumeration
def identity_filter(dig):
    n, k = dig.shape; d = np.zeros(n, np.int8); cnt = np.zeros((n, 3), np.int16); rows = np.arange(n)
    for t in range(k):
        x = dig[:, t]; e = ((d + CH[x]) % 3).astype(np.int8); b = BLK[x] == 1
        np.add.at(cnt, (rows[b], d[b]), 1); np.add.at(cnt, (rows[b], e[b]), -1); d = e
    return dig[(d == 0) & (cnt == 0).all(axis=1)]

def digits(start, stop, k):
    idx = np.arange(start, stop, dtype=np.int64); dig = np.empty((len(idx), k), np.int8)
    for t in range(k - 1, -1, -1): dig[:, t] = idx % 5; idx //= 5
    return dig

def scores(dig, L, s):
    n, k = dig.shape; lut = np.zeros(5 ** L, np.int64)
    for w, v in s.items(): lut[sum(x * 5 ** (L - 1 - i) for i, x in enumerate(w))] = v
    sc = np.zeros(n, np.int64)
    for t in range(k):
        c = np.zeros(n, np.int64)
        for j in range(L): c = c * 5 + dig[:, (t + j) % k]
        sc += lut[c]
    return sc

def min_score(k, L, s, full, chunk=5 ** 9):
    """(number of identity words examined, minimal score); full: all identity words; otherwise those starting with a window of negative weight"""
    n, best = 0, None
    if full:
        for start in range(0, 5 ** k, chunk):
            idw = identity_filter(digits(start, min(5 ** k, start + chunk), k))
            if len(idw): n += len(idw); m = int(scores(idw, L, s).min()); best = m if best is None else min(best, m)
        return n, best
    best = 0
    for v in [w for w, x in s.items() if x < 0]:
        for start in range(0, 5 ** (k - L), chunk):
            tail = digits(start, min(5 ** (k - L), start + chunk), k - L)
            dig = np.hstack([np.tile(np.array(v, np.int8), (len(tail), 1)), tail]); idw = identity_filter(dig)
            if len(idw): n += len(idw); best = min(best, int(scores(idw, L, s).min()))
    return n, best

# ---------------------------------------------------------------- level 3
def level3():
    idw = W.identity_words(3); reps, where = W.orbit_reps(3); rows = {}
    for w in idw:
        if where[w] is None: continue
        col, sg = where[w]
        for t in range(3):
            u = (t + 1) % 3; rest = tuple(w[j] for j in range(3) if j != t and j != u)
            for m, v in W.product(w[t], w[u]).items():
                r = rows.setdefault((t, m, rest), {}); r[col] = r.get(col, 0) + sg * v
    rows = [{c: v for c, v in r.items() if v != 0} for r in rows.values()]
    B = kernel([r for r in rows if r], len(reps)); mu = Fr(1, 5)
    M = [[sum((1 if W.flip(w) == w else 2) * W.weight(w) * (w.count(0) - 3 * mu) * a[i] * b[i] for i, w in enumerate(reps)) for b in B] for a in B]
    pv = pivots(M)
    say("level 3: %d identity words, dim L_3 = %d, mu = %s, form S - 3 mu on L_3: pivots %s  -> %s"
        % (len(idw), len(B), mu, [str(p) for p in pv], "OK" if all(p > 0 for p in pv) and len(pv) == len(B) else "FAILS"))
    return all(p > 0 for p in pv)

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    levels = [int(a) for a in sys.argv[1:]] or list(range(3, 15)); allok = True
    for k in levels:
        t0 = time.time()
        if k == 3: allok &= level3(); continue
        name, L, s = table_for(k)
        assert 2 * L <= k
        if 2 * L == k:                         # merged type: s(x, v) = eps(x) c(v)
            for w, v in s.items():
                assert w[-1] != 0 and all(s.get(w[:-1] + (y,), 0) == EPS[y] * EPS[w[-1]] * v for y in range(5))
        nu = nu_words(L, list(s)); mu = sum(nu[w] * v for w, v in s.items())
        n, m = min_score(k, L, s, full=(k <= 11))
        ok = (m > k * mu); allok &= ok
        say("level %d: table %s on %d-letter words (%d nonzero weights), d_%d = %d, mu = %s, k mu = %s; %s identity words examined: %d, minimal score %d  -> %s  (%.0fs)"
            % (k, name, L, len(s), L, jw_dim(L), mu, k * mu, "all" if k <= 11 else "with a negative window,", n, m, "OK" if ok else "FAILS", time.time() - t0))
    say("all levels verified" if allok else "SOME LEVEL FAILS")
