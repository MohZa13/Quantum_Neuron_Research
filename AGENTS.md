# AGENTS.md — start here

Single entry point for any agent (or human) arriving with **zero context**.
Read this file top to bottom, then follow exactly one route below. Do not
explore the repo by `ls` — everything is catalogued.

> `CLAUDE.md` holds the terse operational rules (environment, commands,
> invariants) and is auto-loaded. **This file holds the map.** They are
> complementary: `CLAUDE.md` tells you how to run things, `AGENTS.md` tells you
> where things are and why they exist.

---

## 1. What this project is, in four sentences

We are building a **quantum neuron** — a Fermi–Dirac machine that classifies
quantum states directly, without ever measuring them into classical features.
To feed it we built `qthermal/`, a pipeline that turns ~130,000 real molecules
from the QH9 dataset into **exact interacting thermal (Gibbs) states** at
chosen temperatures. The scientific bet is that these states carry
**coherence** — genuinely quantum structure invisible to any classical model —
and that a quantum classifier can exploit it. The open frontier is proving
that: finding a **label that only coherence can predict**.

Longer version: `docs/ORIENTATION.md`. Terminology: `docs/GLOSSARY.md`.

---

## 2. Routes — find your task, read only those files

| If you are asked to… | Read, in order |
|---|---|
| **Understand the project cold** | `docs/ORIENTATION.md` → `docs/GLOSSARY.md` → `docs/QUANTUM_NEURON.md` |
| **Work on the quantum neuron / classifier / labels** | `docs/QUANTUM_NEURON.md` → `docs/OPEN_QUESTIONS.md` → `docs/classifier_optimization.md` |
| **Build or train the hybrid network (quantum layer + classical layers)** | `docs/HYBRID_BACKPROP.md` → `qnn/README.md` → `scripts/train_hybrid_spin.py` |
| **Change the thermal-state pipeline** | `qthermal/README.md` → `docs/INVARIANTS.md` → the module's own docstring |
| **Touch data, or wonder what a file is** | `docs/DATA_CATALOG.md` (every artifact, with provenance) |
| **Run something / reproduce a result** | `docs/WORKFLOWS.md` (runbooks) |
| **Find a specific file** | `docs/REPO_MAP.md` (every file in the repo, one line each) |
| **Know what is already known** | `docs/RESEARCH_LOG.md` (dated findings, append-only) |
| **Know why something is the way it is** | `docs/DECISIONS.md` (ADR log) |
| **Know what to work on next** | `docs/OPEN_QUESTIONS.md` (prioritized live agenda) |
| **Avoid breaking things** | `docs/INVARIANTS.md` (the do-not-break list, with verification commands) |
| **Record what you did** | `docs/AGENT_PROTOCOL.md` (update rules) + `docs/templates/` |

---

## 3. The repository in one diagram

```
                 ┌──────────────────────────────────────────┐
                 │  data/QH9Stable.db  (~30 GB, untracked)  │
                 │  130,831 molecules, B3LYP/def2-SVP Fock  │
                 └────────────────────┬─────────────────────┘
                                      │
      scripts/screen_conjugation.py ──┤  cheap triage: gap, DoU, π-system
                                      │  → results/qh9_conjugation_screen*.csv
                                      ▼
   ┌──────────────────────── qthermal/ (HALF 1) ────────────────────────┐
   │  loader → orbitals → active_space → hamiltonian → diagonalize →    │
   │  thermal → io_hdf5 → run (CLI)                                     │
   │      A        B           C             D            E             │
   │      F         G          H                                        │
   └────────────────────────────┬───────────────────────────────────────┘
                                │  results/*.h5  (eigenblocks: p, civecs)
                 ┌──────────────┴───────────────┐
                 ▼                              ▼
       qthermal/encode(_run).py         qthermal/mps.py
       (I) 248 Pauli features           (J) purification MPS
       → *_extheis.h5                   → tensor-network-testing/*.jl
                 │
                 ▼
      scripts/export_thermal_training.py   ← ★ THE BRIDGE (open work item)
      {rho_m, y_m} training file            the LABEL choice lives here
                 │
                 ▼
   ┌──────────────────── the classifier (HALF 2) ────────────────────┐
   │  ONE NEURON (paper §II–VI)                                       │
   │    notebooks/paper/logloss.ipynb      paper-faithful reference   │
   │    notebooks/pennylane/*.ipynb        optimized (R± aggregation) │
   │    figures/quantum_training_impls.py  both, verbatim, for bench  │
   │    tensor-network-testing/*.jl        Yao/ITensor Alg. 8 & 9     │
   │    scripts/train_spin_comparison.py   on real states             │
   │                                                                  │
   │  A NETWORK (paper §VII.C)                                        │
   │    qnn/                  quantum layer + classical MLP           │
   │    docs/HYBRID_BACKPROP.md   ← the training rule, derived here   │
   │    scripts/train_hybrid_spin.py  on real states                  │
   └──────────────────────────────────────────────────────────────────┘
```

**Both halves now meet on real data**, via `scripts/spin_labels.py` →
`train_spin_comparison.py` (one neuron) and `train_hybrid_spin.py` (the
network). What is *not* settled is the label: see `docs/QUANTUM_NEURON.md` §5
and `docs/OPEN_QUESTIONS.md` Q1. The notebook classifiers still train on
synthetic Haar-random states, as in the source paper's Fig. 8, and remain the
reference implementations rather than the experiment.

---

## 4. Non-negotiables (full list + verification: `docs/INVARIANTS.md`)

1. **Never AO-reorder raw `QH9Stable.db` Hamiltonians.** They are already
   PySCF-ordered. Doing so silently corrupts every record (it happened; 284 GB
   discarded 2026-07-13).
2. **Never run SCF in the pipeline.** Phase 1 runs CASCI on the *stored*
   B3LYP Kohn–Sham orbitals. This is intentional.
3. **Never materialize a dense density matrix.** State flows as
   `(weights p, eigenvectors civecs)`. A `2^(2·ncas)` matrix is 65,536² at
   CAS(8,8) — building one is a bug.
4. **Never reorder notebook cells.** `notebook_test_utils.py` executes cells
   *by index*; reordering breaks equivalence tests in a way that looks
   numerical.
5. **Never "fix" the paper notebook's `dfj` 1/T factor** to make numbers agree.
6. **Never assume `evals` exists** in a run file — iterative-solver runs omit it.
7. **Never normalize away the ablation.** A commuting (Z-only / diagonal) pool
   provably reduces the *whole* hybrid network, at any depth, to a classical
   model on `diag(ρ)`. Adding one off-diagonal term to "make it fair" destroys
   the only claim here that is a theorem rather than a benchmark, and nothing
   will fail. `INVARIANTS.md` I15.
8. **Always normalize projected thermal states to unit trace.** Their traces run
   0.967–1.000 and every pool contains the identity, so truncation error becomes
   a free feature correlated with molecular complexity. `INVARIANTS.md` I14.

---

## 5. Environment (one line)

```bash
.venv/bin/python ...          # ALWAYS this interpreter. Never bare `python`.
```

`MPLCONFIGDIR=/tmp/matplotlib` prefix for anything that imports matplotlib.
Full setup and command list: `CLAUDE.md` → *Commands*; runbooks:
`docs/WORKFLOWS.md`.

---

## 6. Current state, in one glance

| | |
|---|---|
| **Production dataset** | `results/qh9_dense_cas8-8_kT0p1.h5` — **1000 molecules**, CAS(8,8), kT = 0.1 Ha, 45.4 GB |
| **Feature set** | `results/qh9_dense_cas8-8_kT0p1_extheis.h5` — 248 Pauli features × 1000 molecules |
| **Labels** | `results/spin_labels_kT0p1.npz` — `⟨S²⟩`, its diagonal part `D`, and the coherence-only `c = ⟨S²⟩ − D` |
| **Models** | one neuron: `scripts/train_spin_comparison.py` · the **network** (paper §VII.C): `qnn/` + `scripts/train_hybrid_spin.py`, training rule derived in `docs/HYBRID_BACKPROP.md` |
| **In flight** | `results/qh9_conjugated_top45.h5` — 28 of ~45 conjugated molecules at kT ∈ {0.1, 0.25} |
| **Beyond dense reach** | `QThermalMPS/` (Module K, Julia): thermal states as purification MPS by imaginary-time TDVP — validated to CAS(10,10)/sector 63,504 (`results/qh9_mps_ncas10.h5`), bridged into `qnn` by `scripts/train_mps_thermal.py` (trains at native K = 1024; conventions pinned by `tests/test_mps_bridge.py`) |
| **Tests** | 337 Python + 7 bridge (skip w/o local h5), `.venv/bin/python -m pytest tests/` · 730 Julia, `julia --project=QThermalMPS QThermalMPS/test/runtests.jl` |
| **Uncommitted** | `--indices` targeted-subset feature (loader + run + tests); `notebook_test_utils.py` moved to repo root; all of `qnn/` and the hybrid-network work |
| **Blocking question** | Which label do we train on? See `docs/QUANTUM_NEURON.md` §5 and `docs/OPEN_QUESTIONS.md` Q1 and **Q13**. The *machinery* is no longer the constraint: on a coherence-only label the network scores 93.0% where the diagonal ablation gets 77.0% and the old single neuron got 66.7%. What is missing is a **physical** label that is not also predictable from chemistry — and note the candidate list in `QUANTUM_NEURON.md` §5 was written for the single neuron's narrower hypothesis class (§2.3) |

---

## 7. Before you finish any task

Update the knowledge base. This is not optional — it is what makes the next
cold start cheap. Rules in `docs/AGENT_PROTOCOL.md`, but the short version:

- Learned something true about the science? → append to `docs/RESEARCH_LOG.md`
- Made a choice a future reader would question? → append to `docs/DECISIONS.md`
- Created or deleted a file/dataset? → update `docs/DATA_CATALOG.md` **and**
  `docs/REPO_MAP.md`
- Answered or raised a research question? → update `docs/OPEN_QUESTIONS.md`
- Found a stale claim in any doc? → fix it and log the correction

An artifact whose provenance is not written down is a liability. We already
have two (see `docs/DATA_CATALOG.md` → *Orphans*); do not create a third.
