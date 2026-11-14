"""Two checks on the Karlsson family K_6 (section 10), using matrices.py.
(1) Base points: Haagerup sets {H_ij conj(H_kj) H_kl conj(H_il)} of the limits theta -> 0+ of
    K_6(theta, phi, psi) at (phi, psi) = (pi/6)(n_2, n_3). At (0,4,1) and (0,2,1) all the values are 12th roots
    of unity (Butson points, finite quantum groups); at (0,0,1) the set contains h with h^2 = (-13-3i sqrt3)/14,
    which is not a root of unity, so this limit is a generic 2x3 Dita matrix.
(2) The curve theta -> K_6(theta, pi/2, 0): c_2 at 72 points, giving 4, except 6 at theta = pi (Fourier),
    theta = 0 being a degenerate point of the parametrization, with the Fourier matrix as limit.
Output: k6points.log"""
import itertools, numpy as np, matrices as m
out = []
def say(s): print(s); out.append(s)
z = (-13 - 3j * np.sqrt(3)) / 14
for pt in [(0, 0, 1), (0, 4, 1), (0, 2, 1)]:
    H = m.K6(1e-9, pt[1] * np.pi / 6, pt[2] * np.pi / 6)
    S = set()
    for i, k, j, l in itertools.product(range(6), repeat=4):
        S.add(complex(round((H[i, j] * np.conj(H[k, j]) * H[k, l] * np.conj(H[i, l])).real, 7),
                      round((H[i, j] * np.conj(H[k, j]) * H[k, l] * np.conj(H[i, l])).imag, 7)))
    roots = all(min(abs(v - np.exp(2j * np.pi * a / 12)) for a in range(12)) < 1e-5 for v in S)
    sq = [v for v in S if abs(v * v - z) < 1e-5]
    say("base point %s: %d Haagerup values; all 12th roots of unity: %s; values h with h^2 = (-13-3i sqrt3)/14: %d"
        % (pt, len(S), roots, len(sq)))
def c2(H):
    n = len(H); P = np.zeros((n, n, n, n), complex)
    for x in range(n):
        for y in range(n):
            v = H[x] / H[y]; P[x, y] = np.outer(v, v.conj()) / n
    T = np.einsum('ijab,klba->ikjl', P, P).reshape(n * n, n * n) / n
    return int(sum(np.linalg.eigvalsh((T + T.conj().T) / 2) > 1 - 1e-9))
vals = {}
with np.errstate(all='ignore'):
    for a in range(1, 72):
        th = a * np.pi / 36; H = m.K6(th, np.pi / 2, 0)
        assert np.abs(np.abs(H) - 1).max() < 1e-8 and np.abs(H @ H.conj().T - 6 * np.eye(6)).max() < 1e-8
        vals[a] = c2(H)
say("curve theta -> K_6(theta, pi/2, 0), theta = a pi/36, a = 1..71: c_2 = 4 except at a = %s, where c_2 = %s"
    % ([a for a, c in vals.items() if c != 4], sorted(set(c for c in vals.values() if c != 4))))
open("k6points.log", "w").write("\n".join(out) + "\n")
