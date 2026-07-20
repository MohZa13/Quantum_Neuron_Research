# Paper-style comparison plots for the log-loss notebooks

This folder now contains a reproducible pipeline for comparing the original
`notebooks/paper/logloss.ipynb` implementation against
`notebooks/pennylane/logloss_pennylane.ipynb` for selected qubit counts,
defaulting to 2, 4, and 7 qubits.

The pipeline produces:

- paper-style log-loss training curves;
- training-time and speedup comparisons;
- static memory comparisons;
- numerical loss/gradient error checks;
- sampling-efficiency plots as the number of training states increases.

## Quick run

From the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python benchmarks/benchmark_paper_comparison.py
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python benchmarks/plot_paper_comparison.py
```

The quick run uses:

- qubits: `2 4 7`;
- epochs: `80`;
- training states: `64`;
- validation states: `128`;
- full original dense training through `n=4`;
- a short original dense timing probe for `n=7`, unless the probe estimates the
  complete original run will finish within 300 seconds.

That last point is deliberate: full original dense `n=7` training can be slow,
so the script now follows a 5-minute gate. It first times a short probe, then
fully runs the original dense path only when the extrapolated complete runtime is
at or below `--full-original-threshold-seconds`.

## Paper-strength run

To match the paper-style scale more closely:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python benchmarks/benchmark_paper_comparison.py --paper
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python benchmarks/plot_paper_comparison.py
```

`--paper` sets:

- epochs: `500`;
- training states: `1000`;
- validation states: `500`.

This can take a while, especially because the original dense `n=4` path is run
fully by default. The original dense `n=7` path remains a timing probe unless
you explicitly raise `--max-full-original-n`.

## Custom run examples

Faster development run:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python benchmarks/benchmark_paper_comparison.py \
  --epochs 40 \
  --samples 32 \
  --validation-samples 64
```

More serious 2/4/7 comparison, still avoiding full dense `n=7`:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python benchmarks/benchmark_paper_comparison.py \
  --epochs 500 \
  --samples 1000 \
  --validation-samples 500 \
  --original-probe-epochs 3 \
  --full-original-threshold-seconds 0
```

Force full original dense training at `n=7` only if you really want to wait:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python benchmarks/benchmark_paper_comparison.py \
  --paper \
  --max-full-original-n 7
```

Use a custom full-run gate:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python benchmarks/benchmark_paper_comparison.py \
  --full-original-threshold-seconds 300
```

## Output files

Benchmark CSVs:

- `benchmarks/paper_training_curves_2_4_7.csv`
- `benchmarks/paper_efficiency_summary_2_4_7.csv`
- `benchmarks/paper_sampling_efficiency_2_4_7.csv`

Figures:

- `figures/paper_training_curves_2_4_7.png`
- `figures/paper_efficiency_comparison_2_4_7.png`
- `figures/paper_sampling_efficiency_2_4_7.png`

## How to read the figures

### Training curves

The training-curve figure is a 2-by-3 grid:

- top row: quantum Heisenberg/log-loss model;
- bottom row: classical FCIM model;
- columns: 2, 4, and 7 qubits.

For `n=7`, the original dense line is marked as a probe unless you force a full
run. The PennyLane exact and Chebyshev curves are fully measured.

### Efficiency comparison

The efficiency figure has four panels:

1. estimated training time;
2. speedup versus the original dense notebook;
3. static data/operator memory;
4. numerical error versus the dense reference.

The exact PennyLane path should have near-machine-precision loss error. The
Chebyshev path is approximate, so it should show small tolerance-controlled
error.

### Sampling efficiency

The sampling-efficiency figure shows one quantum loss/gradient pass as the
number of training states increases.

This is where label aggregation matters most: the exact PennyLane aggregate path
becomes mostly insensitive to sample count after the aggregates are formed,
whereas the original dense path scales roughly linearly with samples.

## Important note about gradients

The original notebook's `dfj` gradient has been corrected to use
`dH/dω_j = H_j` directly. The benchmark CSVs report direct
`gradient_error_vs_dense` values against this corrected dense reference.
