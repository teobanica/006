# Karlsson matrices and the kagome lattice

**Astra, Banica and Fable**, November 2026

For a generic Karlsson matrix of order 6, the first 18 fixed-space dimensions of its quantum permutation group are the Catalan numbers. These are also the dimensions of the higher relative commutants of the associated subfactor. The [preceding paper](https://github.com/teobanica/005) computed these dimensions through level 9. Here a Diţă boundary point turns the question into one about closed walks on the kagome lattice.

- A first invariant tensor outside Temperley–Lieb would give a probability measure on closed walks with prescribed distributions on short windows. At even length a straight segment forces its return path, ruling out every even level as the first exception.
- At level 15 an integer weight on seven-letter windows gives every closed word a score of at least 75. The prescribed distribution would give a mean total score below 48. Together with the staircase tests, this leaves 19 as the first open level. More precisely, a first exception must be congruent to 3 modulo 4 and at least 19, or to 1 modulo 4 and at least 37.
- The same restrictions hold for every strict overgroup in $S_6^+$ of the free wreath product $S_3^+\wr_* S_2^+$ acting on two triples. The last section examines how the fixed spaces vary to second order at special matrices of orders 6 and 8.

## Contents

- [006.pdf](006.pdf) — the paper (26 pages)
- [006.tex](006.tex) — the self-contained LaTeX source, including the kagome figure. Compile twice with `pdflatex`.

| Folder | Computation | Paper |
|---|---|---|
| [overgroup/](overgroup/) | Fourier words, exact local distributions, even-level and staircase tests | Sections 3–5 |
| [overgroup/level15/](overgroup/level15/) | Integer window weights, closed-word check and certified bound on the reference mean | Section 5.5 |
| [checks/](checks/) | The Diţă boundary point, the two-orbital calculation and one-sided Karlsson limits | Sections 6 and 8 |
| [fc/](fc/) and [checks/k6cert/](checks/k6cert/) | Numerical tests for free wreath products and for the Karlsson boundary action | Sections 7.2–7.3 |
| [perturbation/](perturbation/) and [wild/](wild/) | Second-order forms at special parameters, including exact calculations at the order-24 limit group | Section 8 |
| [laptops/](laptops/) | Longer searches and perturbation runs, listed below | Sections 5, 7 and 8 |

| Laptop program | Computation | Paper |
|---|---|---|
| `level15/caracal.py`, `level15/lebanner.py` | Search for separating weights on short windows. Caracal restricts the pauses by default. Le Banner uses all windows and found the level-15 certificate. | Section 5.5 |
| `kamaru-wreath/kamaru.py` | Level-6 test for free wreath products, subsequently repeated by `fc/fcplanar.py` | Section 7.2 |
| `cheetah` | Numerical level-7 test at the generic Diţă point on two triples | Section 7.3 |
| `lyoto` | Second-order coefficients through level 7 at the order-24 limit point | Section 8.2 |
| `kamaru-karlsson/kamaru.py`, `oncilla`, `kodkod` | Second-order coefficients along other Karlsson directions and across the grid of Diţă points | Section 8.3 |
| `manul`, `khabib` | Order-8 comparisons, including the remaining level-6 rotation sectors for the deformed Fourier family | Section 8.4 |

## Running the code

Use Python 3 with NumPy and SciPy. The level-15 verifier also needs a C++17 compiler named `c++`, or one selected with `CXX`. Start in the repository root. A quick exact check of two odd levels is

```sh
(cd overgroup && python3 overgroup.py 5 7)
```

To reproduce the Le Banner certificate from the supplied integer weights and compressed list of closed-word representatives, run

```sh
cd overgroup/level15
gzip -dc lebanner-15-classes.npy.gz > lebanner-15-classes.npy
OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 python3 lebanner-check.py
```

The verifier checks that the 7,751,763 representatives account for all 928,353,697 closed words. It then bounds the reference mean. The last line should read

```text
Verified: min score = 75 and 15 times reference mean < 48.
```

The recorded run is in [verification.txt](overgroup/level15/verification.txt). To regenerate the representative list independently, run the following from the repository root and then rerun the verifier.

```sh
(cd laptops/level15 && python3 -c 'import lebanner; lebanner.classes(15)')
cp laptops/level15/lebanner-output/lebanner-15-classes.npy overgroup/level15/
```
