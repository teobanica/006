import os
import sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact import *
from lwsector import lw_fixed
for t in [int(x) for x in sys.argv[1:]]:
    allA=[]; t0=time.time()
    for w in range(t):
        V,res,_=lw_fixed(H0,t,w,b=48 if t>=6 else 24)
        if V.shape[1]==0: continue
        K=a_operator(V,t); a=np.sort(np.linalg.eigvalsh((K+K.conj().T)/2)); allA+=list(a)
        print("level %d sector %d: f=%d, a_min=%.10f, a_max=%.6f"%(t,w,V.shape[1],a[0],a[-1]),flush=True)
    allA=np.sort(allA); print("LEVEL %d: f=%d, a_min=%.10f, smallest: %s  [%.0f s]"%(t,len(allA),allA[0],np.round(allA[:6],6),time.time()-t0),flush=True)
