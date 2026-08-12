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
| Fermi-Dirac classifier — one neuron (paper §II–VI) | `notebooks/paper/`, `notebooks/pennylane/`, `figures/quantum_training_impls.py`, `scripts/train_spin_comparison.py` |
| Hybrid quantum-classical **network** (paper §VII.C) | `qnn/`, derivation in `docs/HYBRID_BACKPROP.md`, experiment in `scripts/train_hybrid_spin.py` |

Both classifiers now train on real molecular thermal states, via
`scripts/spin_labels.py`. What is still open is *which label* — see
[`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md) Q1. The notebooks continue to
use synthetic random states as in the paper's Fig. 8 and are the reference
implementations, not the experiment.

`qnn/` implements something the source paper explicitly leaves open: it defines
the hybrid architecture (one layer of quantized neurons reading a density matrix
directly, then ordinary classical layers) and then says *"We leave it open to
future work to simulate the performance and training of hybrid
quantum–classical neural networks."* The missing training rule — the chain rule
crossing from a classical layer into a matrix-valued nonlinearity — is derived
in [`docs/HYBRID_BACKPROP.md`](docs/HYBRID_BACKPROP.md).

See `qthermal/README.md` for the physics conventions, solver contracts, and
documented deviations; `qnn/README.md` for the network; and
`docs/classifier_optimization.md` for the single neuron's bottleneck analysis
and optimization history.

**New here?** [`AGENTS.md`](AGENTS.md) is the map — routes by task, a repo
diagram, and current project state. The knowledge base is
[`docs/`](docs/README.md): [`docs/ORIENTATION.md`](docs/ORIENTATION.md) explains
the project in plain language, [`docs/QUANTUM_NEURON.md`](docs/QUANTUM_NEURON.md)
is the research problem, and [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md)
is what to work on next.

## Layout

Every directory has its own `README.md`; every file is catalogued in
[`docs/REPO_MAP.md`](docs/REPO_MAP.md).

- `qthermal/` — QH9 → thermal-states pipeline (the core library)
- `qnn/` — the hybrid quantum-classical network (quantum layer + classical MLP)
- `scripts/` — screening, the training-data bridge, training runs, demonstrations
- `tests/` — pytest suite (`tests/qthermal/`, `tests/qnn/`, notebook equivalence — 337 tests)
- `benchmarks/` — classifier benchmark/plot scripts and their CSV outputs
- `notebooks/` — paper-faithful and PennyLane-optimized classifier notebooks
- `figures/` — figure-generation scripts and rendered PNGs
- `tensor-network-testing/` — Julia (Yao/ITensor) trainers for Algorithms 8/9
- `results/` — generated HDF5 thermal-state files and benchmark CSVs
- `data/` — QH9 database, Slater-weight builder (`build_slater.py`), and the
  raw-SQLite AO-ordering audit (`qh9_raw_sqlite_audit.md`)
- `docs/` — the knowledge base (index: [`docs/README.md`](docs/README.md))
- `Papers/` — the source literature

## Data advisory (2026-07)

Raw `QH9Stable.db` Hamiltonian blobs are **already PySCF-ordered**; applying
the QHBench AO reorder to them corrupts every record. Two readers did exactly
that until 2026-07-09/10; both are fixed, and the corrupted derived dataset
(`data/groups/`, 284 GB) was removed on 2026-07-13. Details, evidence, and
regeneration instructions: `data/qh9_raw_sqlite_audit.md` and
`qthermal/README.md` (Deviations, item 6).

## Quick start

```bash
# setup (fresh clone). requirements.lock reproduces the environment that
# produced the committed results; pyproject.toml carries loose ranges.
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e . --no-deps --no-build-isolation

# thermal states for the first 5 QH9 molecules, CAS(8,8), dense solver
.venv/bin/python -m qthermal.run --qh9-path data/QH9Stable.db --limit 5 \
    --out results/demo.h5

# beyond dense reach: certified iterative solver, larger active space
.venv/bin/python -m qthermal.run --qh9-path data/QH9Stable.db --limit 5 \
    --n-act-occ 5 --n-act-virt 5 --solver iterative --kT-list 0.025 \
    --out results/demo_ncas10.h5

# train the hybrid network on real thermal states (needs the labels + rho stack;
# see docs/WORKFLOWS.md section 5 for how to produce them)
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m scripts.train_hybrid_spin \
    --labels results/spin_labels_kT0p1.npz --rho <scratch>/rho_10q.npy \
    --project-qubits 8

# tests (337)
.venv/bin/python -m pytest tests/
.venv/bin/python -m pytest tests/qnn/ -q          # the network only, ~2 s
```
