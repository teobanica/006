import os
import sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup import *
N=6
def analyze(t, sec=None):
    T0,T1,T2=jets(lambda e: Tk(Hf(e),t), h=2e-3)
    Q=lam_basis(N,t); Qc=Q.astype(complex)
    P0=Qc.conj().T@T0@Qc; P1=Qc.conj().T@T1@Qc; P2=Qc.conj().T@T2@Qc
    P0=(P0+P0.conj().T)/2; P1=(P1+P1.conj().T)/2; P2=(P2+P2.conj().T)/2
    e,V=np.linalg.eigh(P0); fix=e>1-1e-7; L=V[:,fix]; O=V[:,~fix]; eo=e[~fix]
    f=L.shape[1]
    first=L.conj().T@P1@L
    R=O@np.diag(1/(1-eo))@O.conj().T
    term1=-0.5*L.conj().T@P2@L; term2=-L.conj().T@P1@R@P1@L
    M=term1+term2; a=np.sort(np.linalg.eigvalsh((M+M.conj().T)/2))
    print("level %d: f=%d, ||first order|| = %.1e"%(t,f,np.abs(first).max()))
    print("   a_i =",np.round(a,4))
    print("   eigenvalues of term1 (-T''/2):",np.round(np.sort(np.linalg.eigvalsh((term1+term1.conj().T)/2)),4))
    print("   eigenvalues of term2 (-T'RT'):",np.round(np.sort(np.linalg.eigvalsh((term2+term2.conj().T)/2)),4))
    # spectral content of T1 L in eigenbasis of P0
    W=O.conj().T@P1@L; w2=(np.abs(W)**2).sum(axis=1); idx=np.argsort(-w2)[:8]
    print("   T'L hits eigenvalues of Psi_t(H0):",[(round(eo[i],4),round(w2[i],4)) for i in idx if w2[i]>1e-8])
    return a
analyze(3); analyze(4)
