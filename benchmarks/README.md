# benchmarks/ — measurement harnesses

Performance and comparison measurements, plus their CSV outputs. Figures are
rendered from these CSVs into `../figures/`.

Run as plain scripts from the repo root; they import `notebook_test_utils` from
the root. Prefix matplotlib work with `MPLCONFIGDIR=/tmp/matplotlib`.

---

## Scripts

### `benchmark_scaling.py` → `scaling_results.csv`

Times **one** loss-and-gradient pass across methods, qubit counts and sample
counts. One-time setup (dataset reduction, representation construction) is
reported separately as `setup_seconds` — mixing it into the pass time is how
aggregation optimizations get misreported. `--quick` for a CI-sized subset.

Analysis: [`../docs/scaling_comparison.md`](../docs/scaling_comparison.md).

### `benchmark_paper_comparison.py` → three `paper_*_2_4_7.csv`

Paper-style comparison at n ∈ {2, 4, 7}. The original dense path is expensive
(~10.5 h for a full n = 7 run), so the script **times a short probe first** and
runs the full original only if the extrapolation lands under
`--full-original-threshold-seconds` (300 by default). `--paper` raises the
scale to 500 epochs / 1000 training / 500 validation states.

Guide: [`../docs/paper_comparison_guide.md`](../docs/paper_comparison_guide.md).

### `plot_paper_comparison.py`

Renders the three paper-comparison figures from the CSVs.

### `mps_bond_dimensions.py`

Blocked vs interleaved JW ordering for the purification MPS, over every
(molecule, kT) block in a run file. The ancilla bond is the thermal rank and is
identical for both layouts, so the comparison is purely about physical
inter-qubit bonds.

```bash
.venv/bin/python -m benchmarks.mps_bond_dimensions --file results/qh9_dense_cas8-6_kT0p25.h5
```

> ⚠️ **The docstring's stated expectation is superseded.** It says "interleaved
> is expected to win"; measurement showed **blocked** gives ~2× smaller χ.
> See [`../docs/RESEARCH_LOG.md`](../docs/RESEARCH_LOG.md) 2026-07-27 and
> [`../docs/OPEN_QUESTIONS.md`](../docs/OPEN_QUESTIONS.md) Q7.

---

## Headline results

Measured on Intel Core i5-8350U (4C/8T), NumPy 2.4.6, PennyLane 0.45, CPU only.

| | |
|---|---|
| **Label aggregation (R±)** | n = 4, 1000 samples: 1.34 s → **0.0077 s (174×)**; runtime flat in sample count while every other method scales linearly |
| **Static memory** | 3.99 MiB → 0.0156 MiB (**256×**) at 1000 samples |
| **Chebyshev crossover** | **n = 9**. At n = 10: degree 18, 6.9× faster than exact diagonalization, 42× less memory |
| **Chebyshev accuracy** | tol 1e-5 → max absolute loss difference **1.06e-6**, median 1.71e-7 |
| **Diagonal FCIM** | 345× (16 samples) to 20,478× (1000 samples) over dense FCIM |

**Caveats that travel with these numbers:** timings are machine- and
BLAS-specific; static byte counts omit transient workspace; Chebyshev is
approximate; mini-batch Adam changes optimization noise, so per-pass speedups
do not directly predict time-to-accuracy.

---

## Adding a benchmark

- Report **setup separately** from the measured pass.
- Use medians over repeats with warmups (`notebook_test_utils.median_runtime`)
  for anything fast enough to repeat.
- Record the environment — these numbers are meaningless without it.
- Write the CSV to `benchmarks/`, the figure to `../figures/`, and add both to
  [`../docs/DATA_CATALOG.md`](../docs/DATA_CATALOG.md).
- If a result contradicts a stated expectation, **say so explicitly** and log it
  in [`../docs/RESEARCH_LOG.md`](../docs/RESEARCH_LOG.md). That has happened
  twice here and both times mattered.
