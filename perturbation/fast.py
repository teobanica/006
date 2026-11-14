import os
import sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact import Pm, Pd, N, H0, apply_leg, a_operator
def dprop(X,t):
    """returns U'(X) via forward propagation of (X, X'), applying U^(t) first"""
    Xd=np.zeros_like(X)
    for s in reversed(range(t)):
        Xd=apply_leg(Xd,Pm,s,t)+apply_leg(X,Pd,s,t)
        X=apply_leg(X,Pm,s,t)
    return Xd
def a_fast(L,t,chunk=8):
    f=L.shape[1]; G=np.zeros((f,f),complex)
    for a0 in range(0,f,chunk):
        A=L[:,a0:a0+chunk]
        # batch: legs + model_out, with model_in as batch via identity
        XA=[dprop(np.einsum('v,m->vm',A[:,c],np.ones(1)).reshape((N,)*t+(1,))*0+np.multiply.outer(A[:,c].reshape((N,)*t),np.eye(N)).reshape((N,)*t+(N,N))[...,0],t) for c in range(0)]
    # simple version: loop xi, keep all columns in complex64
    for xi_ in range(N):
        C=np.empty((N**(t+1),f),np.complex64)
        for c in range(f):
            X=np.multiply.outer(L[:,c].reshape((N,)*t),np.eye(N)[xi_]).astype(complex)
            C[:,c]=dprop(X,t).reshape(-1)
        Cd=C.astype(complex); G+=Cd.conj().T@Cd
    return G/(2*N)
if __name__=="__main__":
    from lwsector import lw_fixed
    t=int(sys.argv[1]); w=int(sys.argv[2])
    V,_,_=lw_fixed(H0,t,w,b=24)
    t0=time.time(); a1=np.sort(np.linalg.eigvalsh(a_fast(V,t))); t1=time.time()
    a2=np.sort(np.linalg.eigvalsh(a_operator(V,t))); t2=time.time()
    print("level %d sector %d: f=%d, fast %s (%.1f s), old %s (%.1f s), diff %.1e"%(t,w,V.shape[1],np.round(a1[:4],8),t1-t0,np.round(a2[:4],8),t2-t1,np.abs(a1-a2).max()))
