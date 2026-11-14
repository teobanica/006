import sys, numpy as np, fc
n, m, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
N = n * m
lev = {}
for k in range(1, K + 1):
    S, TL, FC, V = fc.spaces(n, m, k); lev[k] = (TL, FC, V)
    print("n=%d m=%d level %d: dim TL %d, dim FC %d, dim FC-TL %d" % (n, m, k, TL.shape[1], FC.shape[1], V.shape[1]), flush=True)
