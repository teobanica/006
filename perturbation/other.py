import os
import sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matrices as m
from lwsector import lw_fixed
from phis import Tk
N=6
def run(p, dvec=np.array([1.0,0,0]), levels=(3,4,5)):
    q0=np.pi/6*np.array(p); Hf=lambda e: m.K6(*(q0+e*dvec)); h=1e-4
    fp1,fm1,fp2,fm2=Hf(h),Hf(-h),Hf(2*h),Hf(-2*h)
    H0=(4*(fp1+fm1)-(fp2+fm2))/6; H1=(8*(fp1-fm1)-(fp2-fm2))/(12*h); Phi=(H1/(1j*H0)).real
    cks=[int((np.linalg.eigvalsh(Tk(H0,k))>1-1e-8).sum()) for k in (2,3,4)]
    xi=lambda i,j: H0[i]*np.conj(H0[j])/np.sqrt(N)
    Pm=np.array([[np.outer(xi(i,j),xi(i,j).conj()) for j in range(N)] for i in range(N)])
    Dl=[np.diag(Phi[i]) for i in range(N)]
    Pd=np.array([[1j*((Dl[i]-Dl[j])@Pm[i,j]-Pm[i,j]@(Dl[i]-Dl[j])) for j in range(N)] for i in range(N)])
    def leg(X,Mx,s):
        Y=np.moveaxis(X,s,0); sh=Y.shape; Y=Y.reshape(N,-1,N); return np.moveaxis(np.einsum('ijab,jrb->ira',Mx,Y).reshape(sh),0,s)
    def dprop(X,t):
        Xd=np.zeros_like(X)
        for s in reversed(range(t)): Xd=leg(Xd,Pm,s)+leg(X,Pd,s); X=leg(X,Pm,s)
        return Xd
    out={"c2..c4":cks}
    for t in levels:
        A=[]
        for w in range(t):
            V,_,_=lw_fixed(H0,t,w,b=24 if t<5 else 48)
            if V.shape[1]==0: continue
            f=V.shape[1]; G=np.zeros((f,f),complex)
            for x in range(N):
                C=np.array([dprop(np.multiply.outer(V[:,c].reshape((N,)*t),np.eye(N)[x]).astype(complex),t).reshape(-1) for c in range(f)]).T
                G+=C.conj().T@C
            A+=list(np.linalg.eigvalsh(G/(2*N)))
        A=np.sort(A); out[t]=(len(A),round(A[0],6),np.round(A[:5],4).tolist())
    return out
for p in [(0,0,1),(0,4,1),(3,0,2)]:
    print(p, run(p), flush=True)
