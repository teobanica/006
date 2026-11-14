# Karlsson matrices and the kagome lattice

The code is grouped by calculation. Each Python program uses imports from its own directory. The level-fifteen verifier also needs a C++17 compiler. Run the checks from the indicated directory.

| folder | content | section |
|---|---|---|
| `fc/` | maximality tests for the free wreath products: `fcplanar.py`, with `fcmax.py` and `sdpcert.py`, see `fc/NOTE.md` | 7 |
| `checks/k6cert/` | certificates at the limit point of K_6: `k6cert.py`, with `lwsector.py`, `sdpcert.py`, `jordan_full.py` | 5, 7 |
| `overgroup/` | Exact word and annular tests, the even-level obstruction, and staircase certificates. | 3–5 |
| `overgroup/level15/` | Integer witness and reproducible verification of the Le Banner certificate. See its README.md. | 5.5 |
| `wild/` | perturbation at the Dita points: `pert.py`, `pert6.py`, and variants | 8 |
| `perturbation/` | second order coefficients for K_6: `group.py`, `exact.py`, `exact56.py` | 8 |
| `checks/` | `karlsson.py` (the limit point of K_6 is the Dita matrix F_2 x_Q F_3, with quantum group D), `orbitals.py` (exact check of c_2 = 2 for K_6, modulo primes) | 6 |
| `checks/` | `k6limits.py` (exact one-sided limits at the base points of K_6), `k6points.py` (base points and Dita curve of K_6) | 8 |

| laptop program | computation | section |
|---|---|---|
| `kamaru-wreath/kamaru` | first version of the level 6 test for the free wreath products, redone by `fc/fcplanar.py` | 7 |
| `cheetah` | certificates at level 7 for the generic 2x3 Dita point | 5, 7 |
| `lyoto` | second order coefficients for K_6 at H_lim, up to level 7 | 8 |
| `kamaru-karlsson/kamaru` | second order coefficients for K_6 at 3 other Dita points | 8 |
| `oncilla` | second order coefficients for K_6 at 4 further Dita points | 8 |
| `kodkod` | second order coefficients for K_6 at all the Dita points of the grid | 8 |
| `manul` | second order coefficients for F_{8,2}, K_{8,2}, M_{8,1} | 8 |
| `khabib` | second order coefficients for F_{8,2} at level 6, remaining sectors | 8 |

The name `kamaru` was used for 2 different programs, kept in 2 different folders.
