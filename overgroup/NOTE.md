# Exact certificates for the overgroups of D (Sections 3–5)

D is the quantum group of a generic Dita matrix F_2 x_Q F_3, on the 6 points (i,a), i in Z_2, a in Z_3.
Everything here works in the word model: letters 0..4 for the functions f_0, f_{0,1}, f_{0,-1}, f_{1,1}, f_{1,-1}.

| program | content | arithmetic | output |
|---|---|---|---|
| `overgroup.py` | verification of the finite-level tests: no first extra tensor at the levels 3 to 12 (level 3: pivot on L_3; levels 4 to 12: word certificates, the zigzag table at the even levels, the tables of the paper at the levels 5, 7, 9, 11) | exact | `overgroup.log` |
| `seven.py` | independent check of the early-level counts and local tests: identity words, reference means, pivots at the levels 3, 4, 6, minimal scores at the levels 5, 7 | exact | `seven.log` |
| `even.py` | further checks of the zigzag inequality of Section 4: all the identity words at level 12, and the identity words with a negative window at the levels 14 to 20 | exact | `even.log` |
| `wordmodel.py` | check of the word model against the matrix F_2 x_Q F_3, Q = (1,1,1; 1,p,1), p = (3+4i)/5, levels 2 to 6 (uses `../wild/matrices.py`, `../wild/ck.py`) | floating point | `wordmodel.log` |
| `merged.py` | check of the partial trace principle (rho_1 = p_1/5, merged tensors of rho_2 and p_2/19) on the first extra tensor of the classical group S_6, at level 4 | floating point | `merged.log` |
| `certify.py`, `certify_sparse.py` | search for word certificates by linear programming (scipy, HiGHS), then rounding to integers; `certify_sparse.py` is for the odd levels, up to 11 | floating point | `found.json` |

Modules: `words.py` (letters, identity test, products of functions), `linalg.py` (kernels and pivots with Fractions),
`jw.py` (the laws nu_l = diag(p_l)/d_l, exactly, block by block with respect to the grading by the group p3),
`marginal.py` (enumeration of the identity words with numpy, linear programs).

Usage: `python3 overgroup.py` (levels 3 to 12, about 1 minute), or `python3 overgroup.py 9 11` for given levels.
`python3 certify.py 4 5 6 8 9 10`, `python3 certify_sparse.py 7 11` (level 11: about 2 minutes, 3 GB of memory).

The tables found by the search at the levels 9 and 11 are those stored in `overgroup.py` (dictionary `PAPER`), and printed in
the paper. The verification by `overgroup.py` does not use `found.json`, nor any floating point number.

The application to the Karlsson family is checked by `../checks/karlsson.py` and `../checks/orbitals.py`.

Added for the levels 13 and beyond:
`caracal.py` (floating point): word certificate at an odd level, by meeting in the middle and linear programming in dual form;
found the table of level 13, stored in `overgroup.py`. Usage: `python3 caracal.py 13`. Same program as the laptop script.
`staircase.py` (exact): the family of tables at the levels 4m+1 (Section 5.4), checked on the identity words containing
a window of weight -1, for m = 1 to 8, that is, up to level 33. Output: `staircase.log`.
`overgroup.py` now checks the levels 3 to 14.
