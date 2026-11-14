import sys, numpy as np, fc
n, m, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
prev = {}
for k in range(1, K + 1):
    TL, FC, V, Vlin, _ = fc.level_test(n, m, k, prev) if k > 1 else (*fc.spaces(n, m, 1)[1:], None, None)[:5] if False else (None,)*5
    if k == 1:
        S, TL, FC, V = fc.spaces(n, m, 1); Vlin = V
    prev[k] = TL
    print("n=%d m=%d level %d: dim TL %d, FC %d, FC-TL %d, after linear constraints %d" % (n, m, k, TL.shape[1], FC.shape[1], V.shape[1], Vlin.shape[1]), flush=True)
