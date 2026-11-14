import numpy as np, itertools
M,N=2,3; q=np.exp(2j*np.pi/12)
Q=np.array([[1,1,1],[1,q,q**2]])
def theta(i,c,k,e): return Q[i,(e-c)%N]*Q[(i-k)%M,e]/(Q[i,e]*Q[(i-k)%M,(e-c)%N])
def gen(i,c):
    G=np.zeros((M*N,M*N),complex)
    for k in range(M):
        for e in range(N): G[k*N+(e-c)%N, k*N+e]=theta(i,c,k,e)
    return G
gens=[gen(i,c) for i in range(M) for c in range(N)]
key=lambda A: tuple(np.round(np.concatenate([A.real.reshape(-1),A.imag.reshape(-1)])*1e6).astype(np.int64).tolist())
# BFS closure
I=np.eye(M*N,dtype=complex); elems={key(I):I}; frontier=[I]
while frontier:
    new=[]
    for A in frontier:
        for g in gens:
            B=g@A; k_=key(B)
            if k_ not in elems: elems[k_]=B; new.append(B)
    frontier=new
    if len(elems)>200000: print("too big"); break
print("|Gamma_bar| =",len(elems))
# identity word counts via convolution on the group
keys=list(elems); idx={k:n for n,k in enumerate(keys)}; mats=[elems[k] for k in keys]
gi=[[idx[key(g@A)] for A in mats] for g in gens]       # left multiplication tables
vec=np.zeros(len(keys)); vec[idx[key(I)]]=1
for p in range(1,8):
    nv=np.zeros_like(vec)
    for t in gi: np.add.at(nv,t,vec)
    vec=nv; print("p=%d: #identity words = %d, /M = %g"%(p,vec[idx[key(I)]],vec[idx[key(I)]]/M))
# blockwise-scalar elements
lam0=np.zeros(len(keys),complex); inZ=np.zeros(len(keys),bool)
for n,A in enumerate(mats):
    ok=True; l0=None
    for k in range(M):
        B=A[k*N:(k+1)*N,k*N:(k+1)*N]
        if np.abs(B-B[0,0]*np.eye(N)).max()>1e-9 or np.abs(A[k*N:(k+1)*N,:]).sum()-np.abs(B).sum()>1e-9: ok=False; break
        if k==0: l0=B[0,0]
    if ok: inZ[n]=True; lam0[n]=l0
print("|Z| =",inZ.sum(), " lambda_0 values:",np.round(lam0[inZ],3))
vec=np.zeros(len(keys),complex); vec[idx[key(I)]]=1
for p in range(1,8):
    nv=np.zeros_like(vec)
    for t in gi: np.add.at(nv,t,vec)
    vec=nv; print("p=%d: c_p = %s"%(p,np.round((vec[inZ]*lam0[inZ]).sum()/M,6)))
