# Quantum Neuron Research

Building on the [QH9 dataset](https://arxiv.org/abs/2306.09549) — DFT Fock
matrices for ~130,000 small organic molecules — this project rebuilds the
original molecular orbitals, selects a physically meaningful active-space
slice, computes inter-electronic interactions explicitly, and solves the
multi-electron problem exactly in that slice. The products are mixed quantum
thermal states at several temperatures, intended as inputs to a hybrid
quantum-classical neural network (a Fermi-Dirac machine) for binary
classification of molecular features such as the HOMO-LUMO gap.

## Pipeline

| Stage | Where |
|---|---|
| QH9 Fock matrices (raw SQLite, def2-SVP) | `data/QH9Stable.db` (untracked, ~30 GB), size index in `data/qh9_scan.jsonl` |
| MO reconstruction: solve H C = S C eps with PySCF overlap | `qthermal/loader.py`, `qthermal/orbitals.py` |
| Active-space selection (frontier window around HOMO/LUMO) | `qthermal/active_space.py` |
| Explicit two-electron integrals in the active space | `qthermal/hamiltonian.py` |
| Exact diagonalization: dense + certified matrix-free Krylov | `qthermal/diagonalize.py` |
| Truncated thermal (Gibbs) ensembles + diagnostics | `qthermal/thermal.py`, CLI in `qthermal/run.py`, HDF5 in `qthermal/io_hdf5.py` |
| Fermi-Dirac classifier (paper reproduction + optimized) | `notebooks/paper/`, `notebooks/pennylane/`, `figures/quantum_training_impls.py` |

The bridge between the thermal-state factory and the classifier (encoding
molecular thermal density matrices as qubit-register inputs, with gap labels)
is the current open work item — the classifier presently trains on synthetic
random states as in the paper's Fig. 8 experiment.

See `qthermal/README.md` for the physics conventions, solver contracts, and
documented deviations; `docs/classifier_optimization.md` for the classifier's
bottleneck analysis and optimization history.

## Layout

- `qthermal/` — QH9 → thermal-states pipeline (the core library)
- `tests/` — pytest suite (`tests/qthermal/`, 90 tests) + notebook
  equivalence test
- `benchmarks/` — classifier benchmark/plot scripts and their CSV outputs
- `notebooks/` — paper-faithful and PennyLane-optimized classifier notebooks
- `figures/` — figure-generation scripts and rendered PNGs
- `results/` — generated HDF5 thermal-state files and benchmark CSVs
- `data/` — QH9 database, Slater-weight builder (`build_slater.py`), and the
  raw-SQLite AO-ordering audit (`qh9_raw_sqlite_audit.md`)
- `docs/` — reports and guides

## Data advisory (2026-07)

Raw `QH9Stable.db` Hamiltonian blobs are **already PySCF-ordered**; applying
the QHBench AO reorder to them corrupts every record. Two readers did exactly
that until 2026-07-09/10; both are fixed, and the corrupted derived dataset
(`data/groups/`, 284 GB) was removed on 2026-07-13. Details, evidence, and
regeneration instructions: `data/qh9_raw_sqlite_audit.md` and
`qthermal/README.md` (Deviations, item 6).

## Quick start

```bash
# thermal states for the first 5 QH9 molecules, CAS(8,8), dense solver
.venv/bin/python -m qthermal.run --qh9-path data/QH9Stable.db --limit 5 \
    --out results/demo.h5

# beyond dense reach: certified iterative solver, larger active space
.venv/bin/python -m qthermal.run --qh9-path data/QH9Stable.db --limit 5 \
    --n-act-occ 5 --n-act-virt 5 --solver iterative --kT-list 0.025 \
    --out results/demo_ncas10.h5

# tests
.venv/bin/python -m pytest tests/
```
