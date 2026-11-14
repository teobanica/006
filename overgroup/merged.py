"""Numerical validation of the partial trace principle on a known first extra tensor: G = S_6, k = 4.
The fixed vectors of S_6 at level 4 are the 15 partition vectors; the 14 noncrossing ones span TL_4, and the component x
of the crossing orthogonal to TL_4 is a first extra tensor. Check: rho_1 = p_1/5 at each leg, and the merged tensors of
rho_2 and of p_2/19 coincide (entries rho_2[(i,a),(j,a)]), at each cyclic interval; rho_2 itself is not p_2/19."""
import itertools, numpy as np
N, k = 6, 4
def parts(n):
    if n == 0: yield []; return
    for p in parts(n - 1):
        for i in range(len(p)): yield p[:i] + [p[i] + [n - 1]] + p[i + 1:]
        yield p + [[n - 1]]
def crossing(p):
    for A, B in itertools.combinations(p, 2):
        for a, c in itertools.combinations(A, 2):
            for b in B:
                for d in B:
                    if a < b < c < d or b < a < d < c and False: return True
                    if a < b < c and (d < a or d > c): return True
    return False
def vec(p):
    v = np.zeros((N,) * k)
    for w in itertools.product(range(N), repeat=k):
        if all(len({w[i] for i in B}) == 1 for B in p): v[w] = 1
    return v.ravel()
P = list(parts(k)); nc = [p for p in P if not crossing(p)]; cr = [p for p in P if crossing(p)]
assert len(nc) == 14 and len(cr) == 1
T = np.array([vec(p) for p in nc]).T; c = vec(cr[0])
x = c - T @ np.linalg.lstsq(T, c, rcond=None)[0]; x /= np.linalg.norm(x); X = x.reshape((N,) * k)
# Jones-Wenzl projections p_1, p_2: zero leg sums, zero on equal adjacent letters
P1 = np.eye(N) - np.ones((N, N)) / N
rows = []
for i in range(N):
    e = np.zeros((N, N)); e[i, :] = 1; rows.append(e.ravel()); e = np.zeros((N, N)); e[:, i] = 1; rows.append(e.ravel())
    e = np.zeros((N, N)); e[i, i] = 1; rows.append(e.ravel())
u, sv, vh = np.linalg.svd(np.array(rows)); q = vh[(sv > 1e-10).sum():].T; p2 = q @ q.T; d2 = int(round(np.trace(p2)))
for r in range(k):
    Y = np.moveaxis(X, list(range(k)), [(a + r) % k for a in range(k)])
    rho1 = np.einsum('abcd,ebcd->ae', Y, Y); rho2 = np.einsum('abcd,efcd->abef', Y, Y)
    m_rho = np.einsum('iaja->ija', rho2); m_ref = np.einsum('iaja->ija', p2.reshape(N, N, N, N)) / d2
    m_rho0 = np.einsum('aiaj->aij', rho2); m_ref0 = np.einsum('aiaj->aij', p2.reshape(N, N, N, N)) / d2
    print("rotation %d: |rho_1 - p_1/5| = %.1e ; merged at the last leg |.| = %.1e, at the first leg %.1e ; |rho_2 - p_2/%d| = %.3f"
          % (r, np.abs(rho1 - P1 / 5).max(), np.abs(m_rho - m_ref).max(), np.abs(m_rho0 - m_ref0).max(), d2, np.abs(rho2.reshape(36, 36) - p2 / d2).max()))
