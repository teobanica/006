"""The family of tables at the levels k = 4m+1 (windows of l = 2m letters), tested on the identity words:
  -1 on the words with alternating letters whose even pairs (2nd, 4th, ...) are straight (opposite characters);
  +2 on the words whose first pair is a backtrack c c^{-1}, whose other odd pairs are straight, the even pairs being alternating.
Complete enumeration of the identity words of length k containing a window of weight -1 (the others have score >= 0):
the tail after the window is cut in 2 pieces, matched through the group element. Exact (integers). Output: staircase.log
Usage: python3 staircase.py [levels], default 5 9 13 17 21 25 29."""
import sys, os, itertools, time, numpy as np
BLK = np.array([0, 0, 0, 1, 1]); CH = np.array([0, 1, 2, 1, 2]); INV = (0, 2, 1, 4, 3)
log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "staircase.log"), "a")
def say(s): print(s, flush=True); log.write(s + "\n"); log.flush()
def alt(a, b): return a != 0 and b != 0 and BLK[a] != BLK[b]
def straight(a, b): return alt(a, b) and CH[a] != CH[b]

def table(l):
    neg, pos = [], []
    def grow(w, kind):                      # kind 0: negative words, kind 1: positive words
        n = len(w)                          # the next pair has number n
        if n == l: (neg if kind == 0 else pos).append(tuple(w)); return
        for x in range(1, 5):
            p = (w[-1], x)
            if kind == 0: ok = straight(*p) if n % 2 == 0 else alt(*p)
            else: ok = alt(*p) if n % 2 == 0 else ((x == INV[w[-1]]) if n == 1 else straight(*p))
            if ok: grow(w + [x], kind)
    for a in range(1, 5): grow([a], 0); grow([a], 1)
    return neg, pos

def all_digits(m):
    idx = np.arange(5 ** m, dtype=np.int64); dig = np.empty((5 ** m, m), np.int8)
    for t in range(m - 1, -1, -1): dig[:, t] = idx % 5; idx //= 5
    return dig
def signatures(dig):
    n, m = dig.shape; d = np.zeros(n, np.int64); cnt = np.zeros((n, 3), np.int64); rows = np.arange(n)
    for t in range(m):
        x = dig[:, t]; e = (d + CH[x]) % 3; b = BLK[x] == 1
        np.add.at(cnt, (rows[b], d[b]), 1); np.add.at(cnt, (rows[b], e[b]), -1); d = e
    return d, cnt
def key(d, cnt, off=60, B=121): return d + 3 * ((cnt[:, 0] + off) + B * ((cnt[:, 1] + off) + B * (cnt[:, 2] + off)))
def code(dig):
    c = np.zeros(len(dig), np.int64)
    for j in range(dig.shape[1]): c = c * 5 + dig[:, j]
    return c

def check(k):
    t0 = time.time(); l = (k - 1) // 2; r = k - l; a, b = r // 2, r - r // 2
    neg, pos = table(l); negc = np.sort(code(np.array(neg, np.int8))); posc = np.sort(code(np.array(pos, np.int8)))
    da, db = all_digits(a), all_digits(b); (ad, ac), (bd, bc) = signatures(da), signatures(db)
    bk = key(bd, bc); bo = np.argsort(bk, kind="stable"); bks = bk[bo]; ar = np.arange(len(da))
    nwords, best, hist = 0, 0, {}
    for v in neg:
        v = np.array(v, np.int8); (vd, vc) = signatures(v[None, :]); vd, vc = int(vd[0]), vc[0]
        d = (vd + ad) % 3; cnt = np.empty_like(ac)                     # after the window and the first piece of the tail
        for c in range(3): cnt[:, c] = vc[c] + ac[:, (c - vd) % 3]
        need = np.empty_like(cnt)
        for c in range(3): need[:, c] = -cnt[ar, (c + d) % 3]
        nk = key((-d) % 3, need); lo, hi = np.searchsorted(bks, nk, "left"), np.searchsorted(bks, nk, "right"); num = hi - lo
        if num.sum() == 0: continue
        pi = np.repeat(ar, num); off = np.arange(num.sum()) - np.repeat(np.cumsum(num) - num, num); si = bo[np.repeat(lo, num) + off]
        words = np.hstack([np.tile(v, (len(pi), 1)), da[pi], db[si]]); n = len(words); nwords += n; sc = np.zeros(n, np.int64)
        for t in range(k):
            c = np.zeros(n, np.int64)
            for j in range(l): c = c * 5 + words[:, (t + j) % k]
            i1 = np.searchsorted(negc, c); i1[i1 == len(negc)] = 0; i2 = np.searchsorted(posc, c); i2[i2 == len(posc)] = 0
            sc += 2 * (posc[i2] == c) - (negc[i1] == c)
        best = min(best, int(sc.min()))
        for x, y in zip(*np.unique(sc, return_counts=True)): hist[int(x)] = hist.get(int(x), 0) + int(y)
    say("level %d: windows of %d letters, %d words of weight -1, %d of weight 2; %d identity words starting with a window of weight -1; minimal score %d; scores: %s  (%.0fs)"
        % (k, l, len(neg), len(pos), nwords, best, dict(sorted(hist.items())), time.time() - t0))
    return best

if __name__ == "__main__":
    for k in [int(x) for x in sys.argv[1:]] or [5, 9, 13, 17, 21, 25, 29]: check(k)
