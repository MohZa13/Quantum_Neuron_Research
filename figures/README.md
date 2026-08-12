# figures/ — figure generation and rendered output

Scripts and their PNG outputs live together here. Prefix everything with
`MPLCONFIGDIR=/tmp/matplotlib`.

---

## `quantum_training_impls.py` — the code of record

Not a plotting script. Both training passes reproduced **verbatim** side by
side, so benchmarks compare implementations rather than reimplementations:

- `run_original` — the `notebooks/paper/logloss.ipynb` pass: dense Pauli
  matrices, per-state loss loop, per-parameter `dfj` gradient loop
- `run_pennylane` — the `optimize_phase2` pass: symbolic Paulis, `R±` label
  aggregation, vectorized gradient
- `run_pennylane_matched` — quantum + classical FCIM trained together on one
  target/state/label set

Given the same seed they consume the RNG in the same order, so they train on
**bit-identical** states, labels and initial weights. `run_original` applies the
`dH/dω_j = H_j` gradient correction so the comparison is matched
([`../docs/INVARIANTS.md`](../docs/INVARIANTS.md) I7).

> This file is also the clearest place to read the model's structure:
> line 78 is the decision rule, line 124 the paper's own labeling scheme, lines
> 139–140 the loss. See [`../docs/QUANTUM_NEURON.md`](../docs/QUANTUM_NEURON.md) §1.

## Data-producing scripts

| Script | Produces |
|---|---|
| `run_pennylane_vs_original.py` | `results/pennylane_quantum_{2..7}qubit.csv` (500-epoch curves) + `results/equivalence_check.csv` (per-n max abs diff over a shared 10-epoch window, plus per-epoch timing) |
| `run_matched_pennylane.py` | `results/pennylane_matched_{2..7}qubit.csv` — a genuinely matched single trial, unlike the earlier version that paired a fresh quantum curve against a *digitized* classical one from a different trial |
| `digitize_fig8_classical.py` | `results/digitized/fig8_classical_{2..7}qubit_digitized.csv` from `docs/fig_8.png`. Pixel↔data calibration was derived programmatically once and is recorded as fixed constants rather than re-derived |

## Plotting scripts

| Script | Figure |
|---|---|
| `plot_fig8_pennylane.py` | `fig8_pennylane_reproduction.png` — the paper's 2×3 grid |
| `plot_efficiency_comparison.py` | `efficiency_comparison.png` |
| `benchmark_notebook_comparison.py` | `notebook_comparison_2_4_7.png` — 3×3: loss, val accuracy, ms/epoch |
| `generate_paper_training_curves.py` | `paper_training_curves_2_4_7.png` |

### Why `generate_paper_training_curves.py` uses a purely quantum target

With a random quantum target (XX+YY+ZZ nearest-neighbour + X+Y+Z) on
Haar-random states, the **classical** FCIM reached similar loss to the
Heisenberg model — all-to-all ZZ can fit the target's ZZ component, so the two
models were solving statistically equivalent sub-tasks.

The fix is a target with no diagonal part at all: **XX+YY nearest-neighbour
only**. The diagonal FCIM then has near-zero gradient, because ZZ expectations
carry no information about XX+YY labels.

> This is the same lesson as the coherence confound, two months earlier and on
> synthetic data: **if the classical model can fit your target, your target was
> classical.** [`../docs/RESEARCH_LOG.md`](../docs/RESEARCH_LOG.md) 2026-06.

---

## Rendered figures

| PNG | Producer |
|---|---|
| `qh9_quantum_neuron_training.png` | `../scripts/demo_train_curve.py` |
| `spin_quantum_vs_classical.png` | `../scripts/train_spin_comparison.py` (one neuron) |
| `hybrid_spin_8q.png`, `hybrid_spin_10q.png` | `../scripts/train_hybrid_spin.py` (the network) — both re-render from their JSON with `--replot`, no retraining |
| `paper_training_curves_2_4_7.png`, `paper_efficiency_comparison_2_4_7.png`, `paper_sampling_efficiency_2_4_7.png` | `../benchmarks/plot_paper_comparison.py` |
| `fig8_pennylane_reproduction.png` | `plot_fig8_pennylane.py` |
| `efficiency_comparison.png` | `plot_efficiency_comparison.py` |
| `notebook_comparison_2_4_7.png` | `benchmark_notebook_comparison.py` |
| `h2o_cas8-6_H_rho_kT0p025.png` | **ad-hoc**, no committed script |
| `qh9_cas8-8_kT0p1_diagnostics.png` | **ad-hoc** — 50-molecule diagnostics |
| `omol25_assessment.png` | `../scripts/plot_omol25_assessment.py` — the three-panel evidence figure for [`../docs/OMOL25_ASSESSMENT.md`](../docs/OMOL25_ASSESSMENT.md); reads the audit JSON/npz only, so restyling can never change a number |
| `qh9_cas8-8_kT0p1_diagnostics_1000mol.png` | **ad-hoc** — 1000-molecule version. ⚠️ [orphan](../docs/DATA_CATALOG.md), but `deck/fig_diagnostics.png` now regenerates the same two panels |

---

## `deck/` — the group-meeting deck's figures

All nine, plus `deck/eq/` (display equations as transparent PNGs, cached by
content hash). **Generated — do not edit, and do not hand-place them into the
slides**; the deck itself is built from `../scripts/presentation/`.

`deck_theory/` (11) and `deck_gap/` (4) are the same arrangement for the other
two decks, produced by `figures_theory.py` and `figures_gap.py`. `deck_gap/`
plots *only* from `../results/gap_diagnosis*.json`, so restyling it can never
change a number.

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=scripts/presentation \
    .venv/bin/python scripts/presentation/figures.py [name ...]
```

They share one validated palette and one set of matplotlib defaults
(`scripts/presentation/style.py`) rather than the per-script `rcParams` blocks
used elsewhere in this directory. If you restyle the deck, restyle there.

---

## Adding a figure

Separate **producing data** from **plotting it** — write the CSV, then render
from it. That is why the comparison figures can be restyled without re-running
a 10-hour training pass.

Then: add the PNG↔script pair to the table above and to
[`../docs/DATA_CATALOG.md`](../docs/DATA_CATALOG.md) §6. **Every figure needs a
committed producer** — three here do not, and they cannot be regenerated or
verified.
