"""Word model for D, the quantum group of a generic Dita matrix F_2 x_Q F_3, acting on the 6 points (i,a), i in Z_2, a in Z_3.
Letters 0..4 = f_0, f_{0,1}, f_{0,-1}, f_{1,1}, f_{1,-1}, with f_0(i,a) = (-1)^i and f_{j,c}(i,a) = [i=j] w^{ca}, w = e^{2 pi i/3}.
A fixed vector with zero leg sums is supported on the identity words, with x_{flip(w)} = (-1)^{#0(w)} x_w."""
import itertools
from fractions import Fraction as Fr

BLOCK = {1: 0, 2: 0, 3: 1, 4: 1}
CHAR = {1: 1, 2: -1, 3: 1, 4: -1}
FLIP = (0, 3, 4, 1, 2)          # block interchange
CONJ = (0, 2, 1, 4, 3)          # character conjugation
NORM = (6, 3, 3, 3, 3)          # squared norms of the 5 functions


def is_identity(w):
    d = 0; cnt = [0, 0, 0]
    for l in w:
        if l == 0: continue
        e = (d + CHAR[l]) % 3
        if BLOCK[l] == 1: cnt[d] += 1; cnt[e] -= 1
        d = e
    return d == 0 and cnt == [0, 0, 0]


def identity_words(k):
    return [w for w in itertools.product(range(5), repeat=k) if is_identity(w)]


def flip(w): return tuple(FLIP[l] for l in w)
def conj(w): return tuple(CONJ[l] for l in w)
def weight(w):
    n = 1
    for l in w: n *= NORM[l]
    return n


def product(a, b):
    """pointwise product f_a f_b on the basis 1, f_0, ..., f_4, as a dict {index: coefficient}, index -1 for the constant 1"""
    if a == 0 and b == 0: return {-1: Fr(1)}
    if a == 0 or b == 0:
        l = a + b; return {l: Fr((-1) ** BLOCK[l])}
    if BLOCK[a] != BLOCK[b]: return {}
    j = BLOCK[a]; c = CHAR[a] + CHAR[b]
    if c == 0: return {-1: Fr(1, 2), 0: Fr((-1) ** j, 2)}
    c = -1 if c == 2 else 1          # 2 = -1 and -2 = 1 mod 3
    return {1 + 2 * j + (0 if c == 1 else 1): Fr(1)}


def orbit_reps(k):
    """identity words up to flip: list of representatives, and the map word -> (index, sign); words killed by the sign rule are dropped"""
    reps, where = [], {}
    for w in identity_words(k):
        if w in where: continue
        fw = flip(w); s = (-1) ** w.count(0)
        if fw == w:
            if s == -1: where[w] = None; continue
            where[w] = (len(reps), 1)
        else:
            where[w] = (len(reps), 1); where[fw] = (len(reps), s)
        reps.append(w)
    return reps, where
