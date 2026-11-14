import os
import sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matrices as m
from phis import Tk
from annular import lam_basis, sectors
q0=np.pi/6*np.array([0,2,1]); d=np.array([1.0,0,0])
Hf=lambda e: m.K6(*(q0+e*d))
def jets(F, h=1e-3):
    # Richardson-type stencil on eps = +-h, +-2h (avoids eps=0, where the formula jumps)
    fp1,fm1,fp2,fm2=F(h),F(-h),F(2*h),F(-2*h)
    f0=(4*(fp1+fm1)-(fp2+fm2))/6
    f1=(8*(fp1-fm1)-(fp2-fm2))/(12*h)
    f2=((fp1+fm1)-2*f0)/h**2          # crude second derivative (uses f0)
    f2=(-(fp2+fm2)+16*(fp1+fm1)-30*f0)/(12*h**2)
    return f0,f1,f2
if __name__=="__main__":
    H0,H1,H2=jets(Hf)
    N=6; print("H0 Hadamard check:",np.abs(np.abs(H0)-1).max(),np.abs(H0@H0.conj().T-N*np.eye(N)).max())
    print("one-sided check: ||H(1e-6)-H0||=",np.abs(Hf(1e-6)-H0).max()," ||H(-1e-6)-H0||=",np.abs(Hf(-1e-6)-H0).max())
    phi=(H1/H0/1j); print("phase derivative real?",np.abs(phi.imag).max())
    phi=phi.real
    # remove trivial directions alpha_i+beta_j: project out
    A=np.zeros((N*N,2*N))
    for i in range(N):
        for j in range(N): A[i*N+j,i]=1; A[i*N+j,N+j]=1
    c,*_=np.linalg.lstsq(A,phi.reshape(-1),rcond=None); phir=(phi.reshape(-1)-A@c).reshape(N,N)
    np.set_printoptions(precision=4,suppress=True,linewidth=140)
    print("H0 (phases / 2pi):\n",np.angle(H0)/(2*np.pi)%1)
    print("phase derivative, mod dephasing:\n",phir)
