"""The limit point of the Karlsson family used in section 3: lim K_6(theta, 0, pi/6), theta -> 0+ (floating point).
(1) the limit matrix L, built from A = F_2 Omega, B = F_2 Omega^2, z_1 = e^{i pi/6}, z_3 = z_4 = 1 and the exact value
    z_2^2 = (-13 - 3 i sqrt3)/14 of k6limits.py, is the limit of K_6(theta, 0, pi/6);
(2) after the relabeling of the paper, L is the Dita matrix F_2 x_Q F_3 with Q = (1,1,1; 1,z_1,z_2), entry by entry;
(3) this Dita matrix has the fixed spaces of the word model at the levels 2, 3, 4, so its quantum group is D;
(4) the criterion: the 3 monomials z_1^{2 n_1} z_2^{2 n_2}, z_1^{2 n_2} z_2^{2 n_0}, z_1^{2 n_0} z_2^{2 n_1}, n_0 + n_1 + n_2 = 0,
    are equal only for n = 0 (search in a box, the proof being in the paper).
Uses ../wild (matrices.py, ck.py) and ../overgroup (words.py). Output: karlsson.log"""
import sys, os, itertools, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'wild')); sys.path.insert(0, os.path.join(here, '..', 'overgroup'))
import matrices as mt, ck, words as W
log = open(os.path.join(here, "karlsson.log"), "w")
def say(x): print(x, flush=True); log.write(x + "\n")

om = np.exp(2j * np.pi / 3); F2 = np.array([[1, 1], [1, -1]], complex); Om = np.diag([om, om.conjugate()])
z1 = np.exp(1j * np.pi / 6); z2 = np.sqrt((-13 - 3j * np.sqrt(3)) / 14); z = [1, z1, z2]
# (1) the limit matrix, in the block form of matrices.K6
Zr = lambda t: np.array([[1, 1], [t, -t]]); A, B = F2 @ Om, F2 @ Om @ Om
L = np.block([[F2, Zr(z1), Zr(z2)], [F2, F2 @ A @ Zr(z1) / 2, F2 @ B @ Zr(z2) / 2], [F2, F2 @ B @ Zr(z1) / 2, F2 @ A @ Zr(z2) / 2]])
say("(1) L is a complex Hadamard matrix: %s" % np.allclose(L @ L.conj().T, 6 * np.eye(6)))
for th in (1e-2, 1e-3, 1e-4, 1e-5):
    say("    theta = %.0e: max |K_6(theta, 0, pi/6) - L| = %.2e" % (th, np.abs(mt.K6(th, 0, np.pi / 6) - L).max()))
# (2) entries H_{is,jt} = omega^{(-1)^s i j} z_j^s (-1)^{st}, and the relabeling rows (s, (-1)^s i), columns (t, j)
E = np.array([[om ** ((-1) ** s * i * j) * z[j] ** s * (-1) ** (s * t) for j in range(3) for t in range(2)] for i in range(3) for s in range(2)])
say("(2) max |L - formula of the paper| = %.1e" % np.abs(L - E).max())
Q = np.array([[1, 1, 1], [1, z1, z2]]); D = mt.dita(mt.fourier(2), mt.fourier(3), Q)       # rows (s, a) -> 3 s + a, columns (t, b) -> 3 t + b
R = np.zeros((6, 6), complex)
for i, s, j, t in itertools.product(range(3), range(2), range(3), range(2)):
    R[3 * s + ((-1) ** s * i) % 3, 3 * t + j] = L[2 * i + s, 2 * j + t]
say("    max |relabeled L - F_2 x_Q F_3| = %.1e" % np.abs(R - D).max())
# (3) fixed spaces of F_2 x_Q F_3 against the word model
f = np.zeros((5, 6), complex); f[0] = [1, 1, 1, -1, -1, -1]
for l in range(1, 5):
    for a in range(3): f[l, 3 * W.BLOCK[l] + a] = om ** (W.CHAR[l] * a)
def vec(w):
    v = np.array([1.0 + 0j])
    for l in w: v = np.kron(v, f[l])
    return v
P1 = np.eye(6) - np.ones((6, 6)) / 6
for k in (2, 3, 4):
    T = ck.T_matrix(D, k); e, U = np.linalg.eigh((T + T.conj().T) / 2); Fix = U[:, e > 1 - 1e-8]; P = np.array([[1.0]])
    for _ in range(k): P = np.kron(P, P1)
    dz = int((np.linalg.svd(P @ Fix, compute_uv=False) > 1e-8).sum()); reps, where = W.orbit_reps(k)
    M = np.array([vec(w) + (-1) ** w.count(0) * vec(W.flip(w)) if W.flip(w) != w else vec(w) for w in reps]).T
    say("(3) k = %d: c_k = %d, zero-sum fixed dimension %d, word model dimension %d, max |T v - v| on the model = %.1e"
        % (k, Fix.shape[1], dz, np.linalg.matrix_rank(M, tol=1e-8), np.abs(T @ M - M).max()))
# (4) the criterion, in a box
a, b = z1 ** 2, z2 ** 2; bad = []
for n0, n1 in itertools.product(range(-40, 41), repeat=2):
    n2 = -n0 - n1; m = [a ** n1 * b ** n2, a ** n2 * b ** n0, a ** n0 * b ** n1]
    if (n0, n1) != (0, 0) and abs(m[0] - m[1]) < 1e-9 and abs(m[1] - m[2]) < 1e-9: bad.append((n0, n1, n2))
say("(4) z_2^2 + conj = %.12f (-13/7 = %.12f); nonzero exponents |n_0|, |n_1| <= 40 with 3 equal monomials: %s" % ((b + 1 / b).real, -13 / 7, bad))
