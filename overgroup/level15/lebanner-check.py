"""Verify an enclosure of the seven-letter reference mean using integer trial vectors.

The eigensolver only proposes trial vectors. All quantities used for certification
are subsequently obtained from those integers. See README.md for the inequalities.
"""
import itertools
import math
import os
import subprocess
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix

L = 7
D = 13775
Q = 262144
ALPHABET = range(5)
STEP = (0, 1, 2, 1, 2)
SCALE = 32

def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)

def grade(word):
    state = 0
    counts = [0, 0, 0]
    for x in word:
        dest = (state + STEP[x]) % 3
        if x >= 3:
            counts[state] += 1
            counts[dest] -= 1
        state = dest
    return (state, *counts)

def products(a, b):
    """Twice each coefficient of the pointwise product. Digit 5 is the constant."""
    if a == b == 0:
        return [(5, 2)]
    if a == 0 or b == 0:
        t = a + b
        return [(t, 2 if t < 3 else -2)]
    if (a <= 2) != (b <= 2):
        return []
    if a == b:
        return [(1 + ((a - 1) ^ 1), 2)]
    return [(5, 1), (0, 1 if a <= 2 else -1)]

def orbit(word):
    return min(tuple(0 if x == 0 else 1 + ((x - 1) ^ g) for x in direction)
               for direction in (word, word[::-1]) for g in range(4))

def read_table():
    table = {}
    for line in Path(__file__).with_name("weights.txt").read_text().splitlines():
        a, b = line.split()
        word = tuple(map(int, a))
        require(word == orbit(word) and word not in table, "bad orbit key")
        table[word] = int(b)
    require(max(map(abs, table.values())) == SCALE, "weight range")
    return table

def exact_square_total(a):
    """Bound each vectorized integer sum against overflow, then use Python ints."""
    if not a.size:
        return 0
    bound = int(np.abs(a).max())
    width = a.shape[-1]
    if width * bound**2 < 2**63:
        return sum(map(int, (a*a).sum(axis=-1).ravel()))
    return sum(int(x)**2 for x in a.ravel())

def inspect_block(words, table):
    n = len(words)
    rows = {}
    row, col, data, output_norm = [], [], [], []
    norms = np.array([2**w.count(0) for w in words], dtype=np.int64)
    weights = np.array([table[orbit(w)] for w in words], dtype=np.int64)
    for j, w in enumerate(words):
        for t in range(L-1):
            for x, coefficient in products(w[t], w[t+1]):
                output = w[:t] + (x,) + w[t+2:]
                key = (t, output)
                if key not in rows:
                    rows[key] = len(rows)
                    output_norm.append(2**(output.count(0)+output.count(5)))
                row.append(rows[key])
                col.append(j)
                data.append(coefficient)
    M = coo_matrix((data,(row,col)), shape=(len(rows),n), dtype=np.int64).tocsr()
    if not rows:
        U = np.eye(n)
    else:
        A = M.astype(float).multiply(1/np.sqrt(norms)).tocsr()
        gram = (A.T @ A).toarray()
        _, U = eigh(gram, subset_by_value=(-np.inf, 1e-8), driver="evr",
                    overwrite_a=True, check_finite=False)
    r = U.shape[1]
    if not r:
        return (0,0,0,0)
    B = np.rint(U*(Q/np.sqrt(norms))[:,None]).astype(np.int64)
    # This small bound also keeps the intermediate sparse merge products safe.
    require(int(np.abs(B).max()) <= Q+1 and n <= 5**7, "trial coefficient range")
    # For scale Q, individual and columnwise sums are checked before int64 summation.
    max_entry = int(np.abs(B).max())
    require(n*int(norms.max())*max_entry**2 < 2**63, "column norm could overflow")
    colnorm = (norms[:,None]*B**2).sum(axis=0)
    # Every partial sum of signed integer products is bounded in absolute
    # value by max(colnorm), by weighted Cauchy--Schwarz.
    require(int(max(colnorm)) < 2**63, "integer Gram product could overflow")
    left = np.ascontiguousarray(B.T)
    right = np.asfortranarray(norms[:,None]*B)
    gram = left @ right
    gram.flat[::r+1] -= Q**2
    gram_error = exact_square_total(gram)
    # Numerator of Tr(sR). An integer sum per word is exact; accumulate as Python ints.
    require(r*max_entry**2 < 2**63, "row norm could overflow")
    diag = (B**2).sum(axis=1)
    weighted_trace = sum(int(w)*int(h)*int(v) for w,h,v in zip(weights,norms,diag))
    merged = M @ B
    require(int(np.abs(merged).max(initial=0)) < 2**30, "merge coefficient range")
    energy = 0
    if merged.size:
        require(r*int(np.abs(merged).max())**2 < 2**63, "energy row could overflow")
        energy = sum(int(h)*int(v) for h,v in zip(output_norm,(merged**2).sum(axis=1)))
    return (r,gram_error,weighted_trace,energy)

def main():
    table = read_table()
    blocks = defaultdict(list)
    for w in itertools.product(ALPHABET, repeat=L):
        blocks[grade(w)].append(w)
    total = [0]*4
    for i, words in enumerate(sorted(blocks.values(),key=len,reverse=True),1):
        result = inspect_block(words,table)
        total = [a+b for a,b in zip(total,result)]
        if i <= 3 or i % 50 == 0:
            print(f"block {i}/{len(blocks)}, block size {len(words)}, cumulative columns {total[0]}",flush=True)
    r, gram_error, weighted_trace, energy = total
    require(r == D, "trial column count")
    # H >= (1 - 2/sqrt(6))(I-P) >= (I-P)/6, so
    # ||P-R||_F^2 <= gram_error / Q^4 + energy / Q^2 when r=D.
    hs2 = Fraction(gram_error,Q**4) + Fraction(energy,Q**2)
    err2 = Fraction(SCALE**2 * (D+r),D**2)*hs2
    denominator = 10**8
    numerator = math.isqrt((err2.numerator*denominator**2)//err2.denominator) + 1
    error = Fraction(numerator,denominator)
    mean = Fraction(weighted_trace,D*Q**2)
    print(f"trial columns {r}\nGram squared error numerator {gram_error}\nweighted trace numerator {weighted_trace}\nenergy numerator {energy}")
    print(f"trial mean {mean}\nabsolute error at most {error}")
    print(f"reference mean in [{float(mean-error):.10f}, {float(mean+error):.10f}] (display rounded)")
    upper = 15*(mean+error)
    print(f"rational upper bound for 15 times mean {upper}")
    require(upper < 48, "reference bound failed")

if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temp:
        executable = Path(temp)/"word-check"
        subprocess.run([os.environ.get("CXX","c++"), "-O3", "-std=c++17",
                        str(folder/"lebanner-words.cpp"), "-o", str(executable)], check=True)
        reps = np.load(folder/"lebanner-15-classes.npy", mmap_mode="r", allow_pickle=False)
        require(reps.ndim == 1 and reps.dtype.kind in ("i","u"), "representative list format")
        with subprocess.Popen([str(executable),str(folder/"weights.txt")],
                              stdin=subprocess.PIPE) as child:
            for start in range(0,len(reps),200000):
                child.stdin.write(reps[start:start+200000].astype("<u8").tobytes())
            child.stdin.close()
            require(child.wait() == 0, "closed-word verification failed")
    main()
    print("Verified: min score = 75 and 15 times reference mean < 48.",flush=True)
