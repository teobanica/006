# Level fifteen

Run the following in this directory with Python 3, NumPy, SciPy and a C++17 compiler available.

```sh
gzip -dc lebanner-15-classes.npy.gz > lebanner-15-classes.npy
OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 python3 lebanner-check.py
```

The verifier compiles lebanner-words.cpp in a temporary directory. It verifies the closed-word list and minimum score, then constructs integer trial columns and checks the reference-mean bound. Both tests must pass. The integers from the recorded run are in verification.txt. Another eigensolver can give different integers while verifying the same inequalities. Section 5.5 gives the proof.

The table records all 9,891 window orbits, using their least seven-digit base-five word as key. It was obtained by rounding 32 times the unrestricted Le Banner weights. The script used to find those weights is in ../../laptops/level15/.
