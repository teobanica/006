"""Check of the word model against the matrix: H = F_2 x_Q F_3 with Q = (1,1,1;1,p,1), p = (3+4i)/5 (floating point).
For k = 2,3,4 the zero-sum part of Fix(u^{ok}), computed from T_k, is compared with the span of the vectors
f_w + (-1)^{#0(w)} f_{flip(w)}, w identity word. For k = 5,6 these vectors are checked to be fixed by T_k.
Uses matrices.py and ck.py of the folder wild. Output: wordmodel.log"""
import sys, os, itertools, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'wild'))
import matrices as mt, ck
import words as W

p = (3 + 4j) / 5
H = mt.dita(mt.fourier(2), mt.fourier(3), np.array([[1, 1, 1], [1, p, 1]]))
assert np.allclose(H @ H.conj().T, 6 * np.eye(6))
om = np.exp(2j * np.pi / 3)
f = np.zeros((5, 6), complex)
f[0] = [1, 1, 1, -1, -1, -1]
for l in range(1, 5):
    j, c = W.BLOCK[l], W.CHAR[l]
    for a in range(3): f[l, 3 * j + a] = om ** (c * a)

log = open('wordmodel.log', 'w')
def say(x): print(x, flush=True); log.write(x + '\n')

def vec(w):
    v = np.array([1.0 + 0j])
    for l in w: v = np.kron(v, f[l])
    return v

for k in (2, 3, 4):
    T = ck.T_matrix(H, k); n = 6 ** k
    e, U = np.linalg.eigh((T + T.conj().T) / 2)
    Fix = U[:, e > 1 - 1e-8]
    # zero-sum projection
    P1 = np.eye(6) - np.ones((6, 6)) / 6; P = np.array([[1.0]])
    for _ in range(k): P = np.kron(P, P1)
    Z = P @ Fix; s = np.linalg.svd(Z, compute_uv=False); dz = int((s > 1e-8).sum())
    reps, where = W.orbit_reps(k)
    M = np.array([vec(w) + (-1) ** w.count(0) * vec(W.flip(w)) if W.flip(w) != w else vec(w) for w in reps]).T
    res = np.abs(T @ M - M).max()
    rk = np.linalg.matrix_rank(M, tol=1e-8)
    say("k=%d  c_k=%d  zero-sum fixed dim=%d  identity words=%d  model dim=%d (rank %d)  max|T v - v|=%.1e"
          % (k, Fix.shape[1], dz, len(W.identity_words(k)), len(reps), rk, res))

for k in (5, 6):
    Tb = ck.T_block_blas(H, k); reps, where = W.orbit_reps(k); rng = np.random.default_rng(0); worst = 0.0
    for _ in range(3):
        c = rng.standard_normal(len(reps)); v = np.zeros(6 ** k, complex)
        for a, w in zip(c, reps): v += a * (vec(w) + ((-1) ** w.count(0) * vec(W.flip(w)) if W.flip(w) != w else 0))
        v /= np.linalg.norm(v); worst = max(worst, np.abs(Tb(v[:, None].copy())[:, 0] - v).max())
    say("k=%d  identity words=%d  model dim=%d  max|T v - v| on random model vectors=%.1e" % (k, len(W.identity_words(k)), len(reps), worst))
