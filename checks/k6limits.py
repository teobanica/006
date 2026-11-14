"""Exact one-sided limits theta -> 0+ of the phases of the Karlsson matrix K_6(theta, phi, pi/6) (section 10.3),
in the coordinates of matrices.py: z_1 = e^{i pi/6}, z_3^2 = M_A(z_1^2), z_4^2 = M_B(z_1^2), z_2^2 = M_A^{-1}(M_B(z_1^2)),
with M_X(z) = (X_12^2 z - X_11^2)/(conj(X_11)^2 z - conj(X_12)^2), A = F_2 Omega, B = -F_2 - A.
At theta = 0 numerators and denominators vanish, and the limits are quotients of first order coefficients.
Exact arithmetic: numbers of Q(i, sqrt3) as 4 Fractions on the basis 1, i, r, ir (r = sqrt3), and first order
jets a + b theta (theta real). Output: k6limits.log"""
from fractions import Fraction as F
class K:                                   # element of Q(i, sqrt3)
    def __init__(s, a=0, b=0, c=0, d=0): s.v = tuple(F(x) for x in (a, b, c, d))
    def __add__(s, o): return K(*[x + y for x, y in zip(s.v, o.v)])
    def __sub__(s, o): return K(*[x - y for x, y in zip(s.v, o.v)])
    def __neg__(s): return K(*[-x for x in s.v])
    def __mul__(s, o):
        a, b, c, d = s.v; e, f, g, h = o.v   # (a + bi + cr + dir)(e + fi + gr + hir), i^2 = -1, r^2 = 3
        return K(a*e - b*f + 3*c*g - 3*d*h, a*f + b*e + 3*c*h + 3*d*g, a*g + c*e - b*h - d*f, a*h + d*e + b*g + c*f)
    def conj(s): a, b, c, d = s.v; return K(a, -b, c, -d)
    def zero(s): return all(x == 0 for x in s.v)
    def inv(s):                            # 1/x = conj(x) (x conj(x))^{-1}, x conj(x) = p + q r
        n = s * s.conj(); p, _, q, _ = n.v; den = p * p - 3 * q * q
        return s.conj() * K(p / den, 0, -q / den, 0)
    def __truediv__(s, o): return s * o.inv()
    def __repr__(s): return "%s + %s i + %s sqrt3 + %s i sqrt3" % s.v
class J:                                   # first order jet x0 + x1 theta
    def __init__(s, x0, x1=None): s.x0, s.x1 = x0, (x1 if x1 is not None else K())
    def __add__(s, o): return J(s.x0 + o.x0, s.x1 + o.x1)
    def __sub__(s, o): return J(s.x0 - o.x0, s.x1 - o.x1)
    def __mul__(s, o): return J(s.x0 * o.x0, s.x0 * o.x1 + s.x1 * o.x0)
    def conj(s): return J(s.x0.conj(), s.x1.conj())
def limit(n, d):                           # limit of n/d at theta = 0+
    if not d.x0.zero(): return n.x0 / d.x0
    assert n.x0.zero() and not d.x1.zero(); return n.x1 / d.x1
half, c = K(F(1, 2)), K(0, 0, F(1, 2))      # c = sqrt3/2
I = K(0, 1)
z1sq = J(K(F(1, 2), 0, 0, F(1, 2)))         # e^{i pi/3} = (1 + i sqrt3)/2
out = []
for name, eminus in [("(0,0,1), phi = 0", K(1)), ("(0,4,1), phi = 2pi/3", K(F(-1, 2), 0, 0, F(-1, 2)))]:
    eplus = eminus.conj()                  # e^{+i phi}
    A11 = J(K() - half + I * c, I * c * eminus)              # -1/2 + i c (cos t + e^{-i phi} sin t), first order
    A12 = J(K() - half - I * c, I * c * eplus)               # -1/2 + i c (e^{i phi} sin t - cos t)
    B11, B12 = J(-K(1)) - A11, J(-K(1)) - A12
    def mob(X11, X12, z): al, be = X12 * X12, X11 * X11; return al * z - be, be.conj() * z - al.conj()
    nA, dA = mob(A11, A12, z1sq); nB, dB = mob(B11, B12, z1sq)
    al, be = A12 * A12, A11 * A11
    z2n, z2d = al.conj() * nB - be * dB, be.conj() * nB - al * dB     # M_A^{-1}(nB/dB)
    z3, z4, z2 = limit(nA, dA), limit(nB, dB), limit(z2n, z2d)
    out.append("%s: z3^2 = %s ; z4^2 = %s ; z2^2 = %s" % (name, z3, z4, z2))
print("\n".join(out)); open("k6limits.log", "w").write("\n".join(out) + "\n")
