# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **📍 Read [`AGENTS.md`](AGENTS.md) first if you are starting with no context.**
> This file holds the operational rules (environment, commands, invariants);
> `AGENTS.md` holds the map — routes by task, the repo diagram, and current
> project state. The knowledge base lives in [`docs/`](docs/README.md):
> [`docs/ORIENTATION.md`](docs/ORIENTATION.md) (what and why),
> [`docs/QUANTUM_NEURON.md`](docs/QUANTUM_NEURON.md) (the research problem),
> [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md) (what to do next),
> [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) (what is already known),
> [`docs/DATA_CATALOG.md`](docs/DATA_CATALOG.md) (what every artifact is).
> **Update them as you work** — rules in [`docs/AGENT_PROTOCOL.md`](docs/AGENT_PROTOCOL.md).

## Environment

`qthermal` is installed editable into the in-tree virtualenv, so imports work
from any working directory — do not add `sys.path` manipulation to new files.

```bash
.venv/bin/python ...                                    # always this interpreter
.venv/bin/pip install -e . --no-build-isolation         # after moving/adding modules
```

Python 3.12, with PySCF (2.13), PennyLane + pennylane-lightning (0.45), h5py,
numpy. Matplotlib scripts need `MPLCONFIGDIR=/tmp/matplotlib` prefixed.

Two dependency tiers: `pyproject.toml` carries loose ranges (PennyLane is
capped to one minor — it breaks API across minors and lightning must match it),
and `requirements.lock` is a `pip freeze` of the environment that produced the
committed results and figures. Reproducing published numbers means the lock
file, not the ranges. Flat layout, no `src/`: `pyproject.toml` lists
`packages = ["qthermal", "qnn"]` explicitly because auto-discovery cannot
disambiguate this many top-level directories.

Only `qthermal/`, `qnn/`, and the root-level `notebook_test_utils.py` are importable
package code. `scripts/`, `benchmarks/`, and `figures/` are run as plain
scripts (`python benchmarks/foo.py`), so sibling imports within those
directories resolve via the script's own directory.

## Commands

```bash
# one-time setup in a fresh clone
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.lock && \
    .venv/bin/pip install -e . --no-deps --no-build-isolation

# thermal-state pipeline (Module H CLI)
.venv/bin/python -m qthermal.run --qh9-path data/QH9Stable.db --out results/demo.h5 \
    --limit 5 --n-act-occ 4 --n-act-virt 4 --kT-list 0.05,0.10,0.25 --workers 4

# beyond dense reach: certified matrix-free Krylov, larger active space
.venv/bin/python -m qthermal.run --qh9-path data/QH9Stable.db --out results/demo10.h5 \
    --limit 5 --n-act-occ 5 --n-act-virt 5 --solver iterative --kT-list 0.025

# eigenblocks -> extended-Heisenberg Pauli coefficients (Module I CLI)
.venv/bin/python -m qthermal.encode_run --in results/demo.h5 --out results/demo_extheis.h5 --taper

# hybrid quantum-classical network (paper VII.C) on real thermal states
.venv/bin/python -m scripts.train_hybrid_spin \
    --labels results/spin_labels_kT0p1.npz --rho /path/rho_10q.npy
#   --project-qubits 8   restrict the register: the quantum layer is O(K^3) and
#                        dataset-size-independent, so this is the only fast lever

# tests (337 collected: 156 qthermal/notebook + 181 qnn)
.venv/bin/python -m pytest tests/
.venv/bin/python -m pytest tests/qthermal/test_thermal.py           # one module
.venv/bin/python -m pytest tests/qthermal/test_run.py::test_name    # one test
.venv/bin/python -m pytest tests/qnn/ -q                            # the network (fast, ~2 s)
.venv/bin/python -m pytest tests/test_notebook_equivalence.py -q    # slow: executes notebook cells

# classifier benchmarks / figures
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python benchmarks/benchmark_paper_comparison.py
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python benchmarks/plot_paper_comparison.py
```

`*.h5`, `*.db`, `*.pt` are gitignored — `results/` HDF5 files and
`data/QH9Stable.db` (~30 GB) exist only locally and must be regenerated, never
assumed present.

## Architecture

Two halves that do not yet meet.

**Half 1 — `qthermal/`: QH9 → thermal states.** A strict linear pipeline, one
module per stage, lettered in the module docstrings:

`loader` (A: raw SQLite `data(id,N,Z,pos,Ham)` → `MoleculeRecord`, empirical
unit detection) → `orbitals` (B: PySCF `Mole`, overlap, `eigh(F,S)` recovery)
→ `active_space` (C: frontier window; every dimension derived, nothing
hardcoded) → `hamiltonian` (D: frozen-core CASCI `(ecore, h1eff, g)` with
injected orbitals) → `diagonalize` (E) → `thermal` (F: Boltzmann truncation +
diagnostics) → `io_hdf5` (G) → `run` (H: CLI + multiprocessing) → `encode` /
`encode_run` (I: Jordan-Wigner + Pauli coefficients) → `mps` (J: purification
MPS from eigenblocks).

The one real seam is `SpectralSolver` (a `Protocol` in `diagonalize.py`, with
`SOLVERS` as the registry). Three implementations with *different contracts* —
read `qthermal/README.md#Solvers` before touching either solver path:
- `DenseEDSolver` — full `eigh`, stores the whole spectrum as `evals`, has
  dimension guardrails and a stored-eigenvector keep cap.
- `IterativeWindowSolver` — matrix-free PySCF FCI Davidson, no `evals` dataset,
  escalates root count until a rigorous counting bound certifies the weight
  cutoff. Correct at any kT but caps out once the thermal window holds
  thousands of states; `--kT-relative` requires dense.
- `NonInteractingSolver` — the g = 0 Gaussian reference used for the
  quantumness audit (trace distance to the non-interacting state).

State flows as `TruncatedEnsemble` / `ThermalBlock` dataclasses, never as dense
density matrices — `rho = V^T diag(p) V` in the CI sector basis. Anything that
would materialize a 2^(2·ncas) matrix is a bug (65,536² for CAS(8,8)).

HDF5 is the interchange format; the layout and its resume-safety rule
(`complete=True` written last, incomplete groups deleted and rewritten) are
documented at the top of `io_hdf5.py`. Readers must not assume `evals` exists.

**Half 2 — the Fermi-Dirac classifier.** Two models, from two parts of the same
paper.

*One neuron* (paper §II–VI). `notebooks/paper/logloss.ipynb` is the
paper-faithful reference; `notebooks/pennylane/` holds the optimized versions;
`figures/quantum_training_impls.py` reproduces both passes verbatim side by side
for benchmarking. The loss is a *spectral* function of H(ω), not a circuit
expectation — hence the label-aggregation optimization (loss and gradient depend
on the training set only through R± = Σ_{y=±1} ρ_i) rather than parameter-shift
autodiff. See `docs/classifier_optimization.md`.

*A network* (paper §VII.C) — `qnn/`. One layer of quantum neurons reading ρ
directly, then a classical MLP. The paper defines this and leaves its training
open; the rule is derived in `docs/HYBRID_BACKPROP.md`:

```
    dL/dTheta_ij = Tr[ H_j . Dphi(B_i)[R_i] ],   R_i = (1/M) sum_m delta_{m,i} rho_m
```

i.e. the classical `delta * phi'(z) * x` with the input vector replaced by an
operator pool, the per-sample sum by a matrix-valued aggregate, and `phi'(z)` by
the **Fréchet derivative** of the activation observable — which is *not*
`phi'(B)`. Read that document before touching `qnn/quantum_layer.py`.
Note what does *not* carry over: the hybrid loss is nonlinear in each ρ_m, so
R± aggregation dies; what survives is one aggregate per neuron per epoch, which
keeps the eigendecomposition count independent of dataset size.

**The bridge exists but the label does not.** `scripts/spin_labels.py` produces
real `{ρ_m, y_m}` and both models train on it
(`train_spin_comparison.py`, `train_hybrid_spin.py`);
`scripts/export_thermal_training.py` exports eigenblocks for the Julia trainers
in `tensor-network-testing/` (Yao/ITensor, Algorithms 8/9). What is open is
*which label* — see `docs/OPEN_QUESTIONS.md` Q1.

## Invariants that are easy to break

1. **Raw `QH9Stable.db` Hamiltonians are already PySCF-ordered — apply NO AO
   reorder.** Doing so double-transforms and silently corrupts every record
   (this happened; 284 GB of derived data was discarded 2026-07-13). The
   QHBench transform helpers in `loader.py` are for QHBench *processed* /
   model-output matrices only. Evidence: `data/qh9_raw_sqlite_audit.md`.
2. **Phase 1 runs CASCI on the stored B3LYP/def2-SVP Kohn-Sham orbitals.** No
   SCF, no geometry stretching. This is intentional — do not "fix" it.
3. **MO signs are canonicalized in `orbitals()`** (largest-magnitude AO
   coefficient made positive). Without it `h1eff`, `g`, and `civecs` carry
   arbitrary per-column sign flips that are pure noise to a downstream model.
4. **The paper notebook's `dfj` divides the gradient by T** — an extraneous
   factor relative to Theorem 5 / Eq. (63). Optimized implementations use
   `dH/dw_j = H_j`; `run_original` also fixes it so comparisons are matched.
   Do not "restore" the notebook's version to make numbers agree.
5. **`notebook_test_utils.py` executes notebook cells by index.**
   Reordering, inserting, or deleting cells in either log-loss notebook breaks
   the equivalence tests in a way that looks like a numerical failure.
6. **`g` is stored full `ncas^4` chemist-notation, not s8-packed** — gzip
   absorbs the redundancy and readers need no unpacking step.
7. **Never add off-diagonal terms to a `z_only` / `diagonal_full` pool.** A
   commuting pool provably reduces the *entire* hybrid network, forward pass and
   gradient at any depth, to a classical model on `diag(rho)`. That theorem is
   what makes the ablation stronger than a benchmark; breaking it fails nothing.
8. **Normalize projected thermal states to unit trace before training.** Traces
   run 0.967-1.000 (they are `1 - truncation_error`) and every pool contains the
   identity, so truncation error would become a free feature.

## Further reading

Full index: [`docs/README.md`](docs/README.md). Most useful first:

- `AGENTS.md` — **the map**: routes by task, repo diagram, current state.
- `docs/QUANTUM_NEURON.md` — the model's exact hypothesis class and the
  coherence-label program. The project's focus document.
- `docs/HYBRID_BACKPROP.md` — **the derivation**: backpropagation from a
  classical layer into a quantum neuron. Read before changing `qnn/`.
- `qnn/README.md` — the network package: layout, cost, and why its ablation is
  a theorem rather than a baseline.
- `docs/OPEN_QUESTIONS.md` — prioritized agenda, each with a decisive test.
- `docs/RESEARCH_LOG.md` — dated findings, including the negative ones.
- `docs/DATA_CATALOG.md` — every artifact: provenance, schema, regeneration cost.
- `docs/INVARIANTS.md` — the do-not-break list with verification commands
  (expanded form of the six above).
- `docs/WORKFLOWS.md` — runbooks for every recurring task.
- `qthermal/README.md` — physics conventions, solver contracts, and the full
  list of evidence-driven deviations. Read before changing pipeline behavior.
- `docs/classifier_optimization.md` — bottleneck analysis, what was optimized,
  what was deliberately skipped.
- `docs/paper_comparison_guide.md`, `docs/scaling_comparison.md` — benchmark
  procedures and measured results.
- `Papers/` — the source papers (Fermi-Dirac Machines, QBM ground-state energies).
