# Maximality of the free wreath products (Proposition 10.1)

The computation of Proposition 10.1 is done by `fcplanar.py`:

    python3 drive.py 2 3 6        # n, m, top level; writes log-2-3.txt and fcplanar-2-3.json
    python3 fcplanar.py 6 1 4 s   # sanity check: the category of all partitions instead of FC

`fcplanar.py` uses `fcmax.py` for the spaces TL_k, FC_k and the linear conditions, and `sdpcert.py`
for the certificates. Its output is in `fcplanar-output/`.

The quadratic conditions are the contractions of w (x) w* and of w (x) w along r nested pairs of legs at the
junction, x_{k-s+1} with y_s for s = 1..r, possibly followed by the merge of the next pair, at all rotations
of both factors. These are planar operations. The older scripts `fcmax.py` (its function `forms`), `run*.py`,
`fc.py`, and the laptop program `laptops/kamaru-wreath/kamaru.py`, also used contractions retaining j legs
and summing the others in parallel, which are not planar operations for j >= 2 (they can send Temperley-Lieb
vectors to the crossing), and their outputs `o_*.txt` are superseded by `fcplanar-output/`.
The linear conditions, and the counts of linear survivors, are the same in both versions.

The reflection w -> w* used for the sesquilinear forms reverses the order of the legs, (Rv)_{i_1..i_k} = v_{i_k..i_1}
(tensor transpose with reversed axes), and the outputs in fcplanar-output/ were computed with this convention.
