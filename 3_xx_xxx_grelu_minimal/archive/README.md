# archive: the earlier two-algorithm version

The first version of this folder, kept for reference. It compares Algorithm 9 (margin loss) with Algorithm 8 (squared loss) and holds out 16 whole chains as the test set. `REPORT.md` / `REPORT.pdf` write it up, and `figures/` holds its four figures.

The scripts were written to run from the folder above (`run.jl` reads `../data/...` and writes `results/` next to itself), so copy them back up a level to re-run them. `pipeline_alg8_vs_alg9.ipynb` is their notebook version, with outputs stripped.
