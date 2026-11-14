import os
import sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup import *
N=6
H0,H1,_=jets(Hf,h=1e-4)
Phi=(H1/(1j*H0)).real
xi=lambda i,j: H0[i]*np.conj(H0[j])/np.sqrt(N)
Pm=np.array([[np.outer(xi(i,j),xi(i,j).conj()) for j in range(N)] for i in range(N)])           # P_ij
Dl=[np.diag(Phi[i]) for i in range(N)]
Pd=np.array([[1j*((Dl[i]-Dl[j])@Pm[i,j]-Pm[i,j]@(Dl[i]-Dl[j])) for j in range(N)] for i in range(N)])  # P'_ij
def apply_leg(X, M, s, t):
    """X: array (N,)*t + (N,) [legs..., model]; M[i,j] = operator on model; apply sum_ij e_ij^{(s)} (x) M_ij"""
    Y=np.moveaxis(X,s,0)                       # (leg s, ..., model)
    sh=Y.shape; Y=Y.reshape(N,-1,N)            # (j, rest, model)
    Z=np.einsum('ijab,jrb->ira',M,Y)          # out leg i, model a
    return np.moveaxis(Z.reshape(sh),0,s)
def U_apply(X,t,deriv_leg=None):
    # U = U^(1) ... U^(t): apply U^(t) first
    for s in reversed(range(t)):
        X=apply_leg(X, Pd if s==deriv_leg else Pm, s, t)
    return X
def a_operator(L,t):
    f=L.shape[1]; G=np.zeros((f,f),complex)
    for xi_ in range(N):
        cols=[]
        for c in range(f):
            X=np.multiply.outer(L[:,c].reshape((N,)*t), np.eye(N)[xi_]).astype(complex)
            assert np.abs(U_apply(X,t)-X).max()<1e-9
            Y=sum(U_apply(X,t,deriv_leg=s) for s in range(t)); cols.append(Y.reshape(-1))
        C=np.array(cols).T; G+=C.conj().T@C
    return G/(2*N)
if __name__=="__main__":
    from annular import lam_basis
    np.set_printoptions(precision=10,suppress=True,linewidth=150)
    for t in [3,4]:
        Q=lam_basis(N,t).astype(complex); M=Q.conj().T@Tk(H0,t)@Q; e,V=np.linalg.eigh((M+M.conj().T)/2)
        L=Q@V[:,e>1-1e-9]
        a=np.linalg.eigvalsh(a_operator(L,t)); print("level %d: f=%d, a_i ="%(t,L.shape[1]),np.sort(a))
