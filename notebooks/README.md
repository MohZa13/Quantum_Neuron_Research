# notebooks/

> **Scope.** These are the **single-neuron** reference implementations, on
> synthetic Haar-random states, faithful to the source paper's §II–VI and Fig. 8.
> They are not where experiments run. Real molecular thermal states are handled
> by `../scripts/train_spin_comparison.py` (one neuron) and
> `../scripts/train_hybrid_spin.py` (the hybrid **network** of the paper's
> §VII.C, over `../qnn/`). — the classifier

The Fermi–Dirac machine lives here: one paper-faithful reference and one
optimized production implementation, held equal by 12 automated tests.

> ## ⚠️ Cell order is load-bearing
>
> `notebook_test_utils.py` executes cells **by index** — cells `(0,1,2,3)` of
> `paper/logloss.ipynb` and `(1..7)` of `pennylane/logloss_pennylane.ipynb`.
>
> **Reordering, inserting, or deleting a cell breaks the equivalence tests in a
> way that looks like a numerical failure.** You will chase a physics bug that
> is actually an off-by-one. If you must restructure, update the index tuples in
> `notebook_test_utils.py` in the same commit.
>
> [`../docs/INVARIANTS.md`](../docs/INVARIANTS.md) I6.

---

## `paper/logloss.ipynb` — the reference

The paper's algorithm as written: dense Pauli matrices, per-state Python loop
for the loss, per-parameter per-state loop (`dfj`) for the gradient. 9 cells.

Its value is being *exactly* what the paper describes — **including the
extraneous `1/T` factor in `dfj`** (an error relative to Theorem 5 / Eq. 63).
Do not fix it here. Corrections live in the optimized paths and in
`figures/quantum_training_impls.py::run_original`, which applies the fix so
comparisons are matched
([`../docs/INVARIANTS.md`](../docs/INVARIANTS.md) I7).

Not fast: measured 0.7 s/epoch at n = 2 up to 76 s/epoch at n = 7 — a full
500-epoch n = 7 run would take ~10.5 hours.

## `pennylane/logloss_pennylane.ipynb` — production

The optimized implementation. 15 cells; tests execute 1–7.

| Cell | Contents |
|---|---|
| 2 | Original definitions (Paulis, training states, `fdd_logloss_matrix`, `dfj`) |
| 3 | Phase 1 — vectorized gradients: one eigendecomposition per parameter vector, batched einsum |
| 4 | Validation set, accuracy, state-vector helpers |
| 5 | **The heavy machinery** (29 KB): symbolic Paulis, sparse + fused-bitwise kernels, matrix-free Hamiltonian application, backend resolution |
| 6 | `optimize` — the paper's training protocol |
| 7 | **`optimize_phase2`** — label-aggregated `R±`, GD or Adam, adaptive Chebyshev loss. **The production training pass** |
| 9–11 | Benchmark cells |

**The key idea** — the loss is linear in each ρ, so it depends on the training
set only through `R± = Σ_{y=±1} ρ_i`. Sum each class once and per-epoch cost
becomes *independent of dataset size*: measured 174× at 1,000 samples, 256×
less static memory. The same linearity pins the model's hypothesis class —
see [`../docs/QUANTUM_NEURON.md`](../docs/QUANTUM_NEURON.md) §2.

## `pennylane/phase1_optimization.ipynb` — exploratory

Autodiff experiments and bottleneck profiling. **Cells are in reverse narrative
order** (section 7 first, section 1 last) — an artifact of assembly, not a
convention. Historical; the autodiff route was evaluated and deliberately not
adopted ([`../docs/DECISIONS.md`](../docs/DECISIONS.md) D6).

## `pennylane/sampling_demo.py` — the model in 40 lines

A 2-qubit TFIM demonstration of `output = Tr[g_T(H(ω))ρ]` with
`g_T(x) = tanh(x/T)`. **The clearest single statement of what the neuron is** —
read this before the notebooks.

---

## Working here

```bash
.venv/bin/python -m pytest tests/test_notebook_equivalence.py -q   # run after ANY edit
```

Prefer adding new work to `figures/quantum_training_impls.py` (plain Python,
importable, benchmarkable) over adding notebook cells. That file already
reproduces both training passes verbatim and is the code of record for
comparisons.

**Current limitation:** these train on **synthetic Haar-random states**, as in
the paper's Fig. 8. Connecting them to real molecular thermal states is the
project's open work item — [`../docs/QUANTUM_NEURON.md`](../docs/QUANTUM_NEURON.md).
