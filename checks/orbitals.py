"""Exact check of c_2 = 2 at rational points of the Karlsson family, by reduction modulo primes.
Parameters: the half-angle tangents (t_1, t_2, t_3) of (theta, phi, psi_1), rational. The matrix K_6 has its entries in the ring
generated over Q(i, sqrt3) by square roots z_3, z_4, z_2 of the 3 Moebius values, and by inverses. Modulo a prime p where
-1, 3 and the 3 Moebius values are squares, and no denominator vanishes, we get a matrix over F_p, with H Hbar^t = 6
(Hbar = entrywise inverse), and the profile sums S[(i,j),(k,l)] = sum_x H_ix Hbar_kx Hbar_jx H_lx. A sum which is nonzero
modulo p is nonzero for the corresponding complex matrix, which is the matrix K_6 at these parameters, up to a Galois
conjugation of Q(i, sqrt3) and to changes of signs of z_2, z_3, z_4; none of these changes the vanishing pattern, up to
permutations of rows and columns. Thus if the graph of the nonzero sums on the 36 pairs has 2 components modulo p, then c_2 = 2.
Usage: python3 orbitals.py [t_1 t_2 t_3], default 1 1 2. Output: orbitals.log"""
import sys, os, itertools
from fractions import Fraction as Fr
log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "orbitals.log"), "a")
def say(x): print(x, flush=True); log.write(x + "\n")

def sqrt_mod(a, p):
    a %= p
    for x in range(p):
        if x * x % p == a: return x
    return None

def run(ts, p):
    inv = lambda a: pow(a % p, p - 2, p)
    fr = lambda q: q.numerator * inv(q.denominator) % p
    I, S = sqrt_mod(-1, p), sqrt_mod(3, p)
    if I is None or S is None or any((1 + t * t).numerator % p == 0 or t.denominator % p == 0 for t in ts): return None
    t1, t2, t3 = (fr(t) for t in ts)
    def build(I):                                   # the matrices A, B and the phase z_1, for a given square root of -1
        c = (1 - t1 * t1) * inv(1 + t1 * t1) % p; s = 2 * t1 * inv(1 + t1 * t1) % p
        e = (1 + I * t2) ** 2 * inv(1 + t2 * t2) % p; z1 = (1 + I * t3) ** 2 * inv(1 + t3 * t3) % p
        h = inv(2); Lam = [[c, e * s % p], [inv(e) * s % p, -c % p]]
        M = [[(-h * (i == j) + I * S * h * Lam[i][j]) % p for j in range(2)] for i in range(2)]
        A = [[(M[0][j] + M[1][j]) % p for j in range(2)], [(M[0][j] - M[1][j]) % p for j in range(2)]]
        F2 = [[1, 1], [1, p - 1]]; B = [[(-F2[i][j] - A[i][j]) % p for j in range(2)] for i in range(2)]
        return A, B, z1
    (A, B, z1), (Ac, Bc, z1c) = build(I), build(-I % p)        # the second one is the conjugate
    if z1 * z1c % p != 1: return None
    def mob(X, Xc, z):
        al, be, alc, bec = X[0][1] ** 2, X[0][0] ** 2, Xc[0][1] ** 2, Xc[0][0] ** 2; den = (bec * z - alc) % p
        return None if den == 0 else (al * z - be) * inv(den) % p
    def mobinv(X, Xc, w):
        al, be, alc, bec = X[0][1] ** 2, X[0][0] ** 2, Xc[0][1] ** 2, Xc[0][0] ** 2; den = (bec * w - al) % p
        return None if den == 0 else (alc * w - be) * inv(den) % p
    w3, w4 = mob(A, Ac, z1 * z1 % p), mob(B, Bc, z1 * z1 % p)
    if w3 is None or w4 is None: return None
    w2 = mobinv(A, Ac, w4)
    if w2 is None or 0 in (w2, w3, w4): return None
    z3, z4, z2 = sqrt_mod(w3, p), sqrt_mod(w4, p), sqrt_mod(w2, p)
    if None in (z2, z3, z4): return None
    mul = lambda X, Y: [[sum(X[i][k] * Y[k][j] for k in range(2)) % p for j in range(2)] for i in range(2)]
    half = lambda X: [[x * inv(2) % p for x in r] for r in X]
    Zr = lambda z: [[1, 1], [z, -z % p]]; Zc = lambda z: [[1, z], [1, -z % p]]; F2 = [[1, 1], [1, p - 1]]
    Z1, Z2, Z3, Z4 = Zr(z1), Zr(z2), Zc(z3), Zc(z4)
    blocks = [[F2, Z1, Z2], [Z3, half(mul(mul(Z3, A), Z1)), half(mul(mul(Z3, B), Z2))], [Z4, half(mul(mul(Z4, B), Z1)), half(mul(mul(Z4, A), Z2))]]
    H = [[blocks[i // 2][j // 2][i % 2][j % 2] for j in range(6)] for i in range(6)]
    if any(x == 0 for r in H for x in r): return None
    Hb = [[inv(x) for x in r] for r in H]
    had = all(sum(H[i][x] * Hb[j][x] for x in range(6)) % p == (6 if i == j else 0) for i in range(6) for j in range(6))
    pairs = list(itertools.product(range(6), repeat=2)); comp = {q: q for q in pairs}
    def find(q):
        while comp[q] != q: q = comp[q]
        return q
    edges = 0
    for (i, j), (k, l) in itertools.product(pairs, repeat=2):
        if sum(H[i][x] * Hb[k][x] * Hb[j][x] * H[l][x] for x in range(6)) % p:
            edges += 1; comp[find((i, j))] = find((k, l))
    sizes = sorted(sum(1 for q in pairs if find(q) == r) for r in {find(q) for q in pairs})
    return dict(p=p, I=I, S=S, phases=(z1, z2, z3, z4), hadamard=had, edges=edges, sizes=sizes)

if __name__ == "__main__":
    ts = [Fr(a) for a in sys.argv[1:4]] or [Fr(1), Fr(1), Fr(2)]
    say("half-angle tangents %s" % ", ".join(str(t) for t in ts)); done = 0
    for p in range(5, 2000):
        if all(p % q for q in range(2, int(p ** 0.5) + 1)):
            r = run(ts, p)
            if r is not None:
                say("   p = %d: i -> %d, sqrt3 -> %d, phases z_1, z_2, z_3, z_4 -> %s; H Hbar^t = 6: %s; %d nonzero profile sums, components of sizes %s"
                    % (r["p"], r["I"], r["S"], r["phases"], r["hadamard"], r["edges"], r["sizes"])); done += 1
                if done == 6: break
