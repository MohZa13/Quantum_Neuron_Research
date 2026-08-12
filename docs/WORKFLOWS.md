# Workflows — runbooks

*Exact commands for every recurring task. All paths are repo-relative; run from
the repository root. **Always `.venv/bin/python`**, never bare `python`.*

Timings are for this machine (Intel Core i5-8350U, 4C/8T) — treat them as
orders of magnitude, not guarantees.

---

## 0. Environment

```bash
# fresh clone
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e . --no-deps --no-build-isolation

# after moving or adding modules
.venv/bin/pip install -e . --no-build-isolation

# anything importing matplotlib needs this prefix
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python ...
```

`requirements.lock` reproduces the exact environment behind every committed
result. `pyproject.toml`'s ranges are for "does it import and run".

**Preconditions to check before any pipeline run:**

```bash
ls -la data/QH9Stable.db        # ~30 GB; not in git — must exist locally
df -h .                          # a 1000-molecule CAS(8,8) run writes ~45 GB
```

---

## 1. Generate thermal states

### 1a. Smoke test (minutes)

```bash
.venv/bin/python -m qthermal.run \
    --qh9-path data/QH9Stable.db --out results/demo.h5 \
    --limit 5 --n-act-occ 4 --n-act-virt 4 --kT-list 0.05,0.10,0.25 --workers 4
```

### 1b. Production settings (~192 s/molecule)

```bash
.venv/bin/python -m qthermal.run \
    --qh9-path data/QH9Stable.db --out results/run.h5 \
    --limit 1000 --n-act-occ 4 --n-act-virt 4 --kT-list 0.1 \
    --keep-cap 0 --workers 4 --log-level INFO 2>&1 | tee results/run.log
```

`--keep-cap 0` lifts the storage cap so the weight cutoff alone decides.
**Recommended for kT ≥ 0.25**, where the default cap discards percent-level
weight. Costs `m × dim` float64 per ensemble (~192 MB/molecule pre-gzip at
dim = 4,900 when the window spans the sector).

### 1c. Targeted subset from the conjugation screen

*(`--indices` is currently uncommitted — see `OPEN_QUESTIONS.md` Q10)*

```bash
# take the 45 most-conjugated ids from the full screen
.venv/bin/python - <<'EOF' > /tmp/ids.txt
import csv
rows = sorted(csv.DictReader(open("results/qh9_conjugation_screen_full.csv")),
              key=lambda r: float(r["gap_Ha"]))
print("\n".join(r["idx"] for r in rows[:45]))
EOF

.venv/bin/python -m qthermal.run \
    --qh9-path data/QH9Stable.db --out results/conjugated.h5 \
    --indices @/tmp/ids.txt --limit 45 --kT-list 0.1,0.25 --keep-cap 0 --workers 4
```

`--indices` also accepts inline form: `--indices '0-9,42,6485'`.

### 1d. Beyond dense reach (larger active space)

```bash
.venv/bin/python -m qthermal.run \
    --qh9-path data/QH9Stable.db --out results/ncas10.h5 \
    --limit 5 --n-act-occ 5 --n-act-virt 5 --solver iterative --kT-list 0.025
```

Dense refuses above dim 70,000 and needs `--allow-large-dense` above 5,000.
Krylov has no dim guardrail (only a RAM precheck) but is a **low-kT** tool —
watch for `cap_hit` in the log. `--kT-relative` requires `--solver dense`.

### Resuming

**Every run is resumable.** Rerun the identical command: complete molecule
groups are skipped, incomplete ones deleted and rewritten. Safe to `Ctrl-C` or
kill at any point.

### Reading the log

| Line | Meaning |
|---|---|
| `Coordinate unit detected: Angstrom` | Unit detection succeeded; re-verified every 100th record |
| `mol N done in T s` | Written and marked complete |
| `mol N skipped: <reason>` | Validation failure — logged, never fatal |
| `ensemble cap bound at kT_max=...` | **Storage cap**, not the weight cutoff, bound the truncation. The discarded weight is recorded per block with `cap_hit=True`. Rerun with `--keep-cap 0` if it matters |
| `Davidson left roots unconverged` | Degenerate multiplet at the window edge; the solver escalates automatically |

---

## 2. Extract Pauli features

```bash
.venv/bin/python -m qthermal.encode_run \
    --in results/run.h5 --out results/run_extheis.h5 --ordering blocked --taper
```

Produces 248 coefficients `Tr(ρP)` per molecule per kT (at ncas = 8), plus the
Z₂-tapered 14-qubit basis. ~1 s/molecule — the contraction happens in the
determinant basis, never on the 65,536-dimensional register.

`--taper` can be added later to an existing output file: molecules resume, the
taper datasets are filled in. **The tool refuses to mix orderings in one file.**

---

## 3. Export training data (the bridge)

```bash
.venv/bin/python -m scripts.export_thermal_training \
    --in results/run.h5 --out results/training.h5 \
    --kT 0.1 --ordering interleaved --label h5:static_corr --label-threshold median
```

**Label specs:**
- `h5:<key>` — a per-block diagnostic from the run file
  (`static_corr`, `entropy`, `c_max_sq`, `tracedist_gaussian`, `truncation_error`)
- `csv:<path>:<col>` — join an external table on molecule idx, e.g.
  `csv:results/qh9_conjugation_screen_full.csv:gap_Ha`

> ⚠️ **The default `h5:static_corr` is a placeholder.** It is a one-body
> quantity a classical model reads directly. See
> [`QUANTUM_NEURON.md`](QUANTUM_NEURON.md) §3–§5 before choosing.

`--weight-keep 0.99` trims the long tail of rare eigenstates;
`--limit N` exports the first N complete molecules.

---

## 4. Screen molecules

```bash
.venv/bin/python -m scripts.screen_conjugation \
    --qh9-path data/QH9Stable.db --limit 1000 --kT 0.1 \
    --out results/screen.csv --top 20
```

Omit `--limit` for the full 130,831-molecule database (multi-hour). Resumable
via `<out>.partial` — rerunning picks up where it stopped. Tier 3 (structural
conjugation) needs RDKit; without it those columns are blank and the script
warns rather than failing.

---

## 5. Run the classifier

### Quick demonstration on real thermal features

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/demo_train_curve.py
```

Trains on the 248 Pauli features of 1000 molecules, labeled by median
HOMO–LUMO gap, 70/30 split → `figures/qh9_quantum_neuron_training.png`.
**Input paths are hardcoded.** Note this is a *classical* logistic rule on
quantum features, not the full FD machine — a demonstration, not the result.

### The hybrid network on real thermal states (paper §VII.C)

One layer of quantum neurons + a classical MLP, trained with the rule derived in
[`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md). Needs the labels **and** the
projected ρ stack; the stack is large and not committed, so make it first if it
is missing:

```bash
# 1. labels + the projected rho stack (reads the 45 GB run file; ~15 min)
.venv/bin/python -m scripts.spin_labels \
    --in results/qh9_dense_cas8-8_kT0p1.h5 \
    --out results/spin_labels_kT0p1.npz \
    --rho-out "$SCRATCH/rho_10q.npy" --n-qubits 10

# 2. the experiment: 3 labels x 4 models
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m scripts.train_hybrid_spin \
    --labels results/spin_labels_kT0p1.npz --rho "$SCRATCH/rho_10q.npy"

# fast iteration: 8-qubit register, ~3 min for the whole grid
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m scripts.train_hybrid_spin \
    --labels results/spin_labels_kT0p1.npz --rho "$SCRATCH/rho_10q.npy" \
    --project-qubits 8 --json-out "$SCRATCH/pilot.json" --out "$SCRATCH/pilot.png"

# re-render the figure from stored metrics — no retraining, no rho stack
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m scripts.train_hybrid_spin \
    --labels results/spin_labels_kT0p1.npz --rho /dev/null --replot
```

**Cost.** The quantum layer is `O(K³)` per neuron per epoch and **independent of
dataset size**; the diagonal pools take an `O(K)` path and are ~2000× faster.
Measured at J₁ = 8: 3.6 s/epoch at K = 1024, 0.14 s at K = 256. So
`--project-qubits` is the only lever that matters for a sweep, and adding
molecules is nearly free.

**Reading the log.** `sat` is the mean divided difference in units of `φ'(0)`
and `rho(B)` is the largest eigenvalue of the pre-activation operator. If `sat`
falls toward 0 while `rho(B)` grows past a few times `T`, the quantum layer is
**saturating** — the neurons are going dead, and the fix is a larger `--l2` or a
larger `--temperature`, not more epochs.

### Paper reproduction / benchmarks

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python benchmarks/benchmark_paper_comparison.py
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python benchmarks/plot_paper_comparison.py

# paper-strength: 500 epochs, 1000 training states, 500 validation
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python benchmarks/benchmark_paper_comparison.py --paper
```

The original dense path at n = 7 is gated behind a timing probe (a full run is
~10.5 h). Details: [`paper_comparison_guide.md`](paper_comparison_guide.md).

### Scaling study

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python benchmarks/benchmark_scaling.py
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python benchmarks/benchmark_scaling.py --quick
```

### Julia trainers

```bash
julia
julia> import Pkg; Pkg.add(["Yao", "ITensors", "ITensorMPS", "Optimisers"])
julia> include("tensor-network-testing/train_alg9.jl")     # Algorithm 9 + Optimisers.jl
julia> include("tensor-network-testing/convergence_checks.jl")
julia> run_all(; n=4)          # correctness (0–3) + convergence (4–7)
julia> cost_vs_n(; ns=4:2:12)  # check 8, slower
```

Use `Alg9Yao` (statevector) for training; `Alg9ITensor` (MPS) for validation and
larger *n* — too slow to drive an optimizer.

---

## 6. MPS bond-dimension comparison

```bash
.venv/bin/python -m benchmarks.mps_bond_dimensions --file results/qh9_dense_cas8-6_kT0p25.h5
```

Reports physical bond profiles for both wire orderings. The ancilla bond is the
thermal rank and is identical for both, so the comparison is purely about
physical bonds. **Measured result: blocked wins** (~2× smaller χ) — the
docstring's stated expectation is superseded (`OPEN_QUESTIONS.md` Q7).

---

## 7. Tests

```bash
.venv/bin/python -m pytest tests/                              # all 337
.venv/bin/python -m pytest tests/qthermal/ -q                  # pipeline only
.venv/bin/python -m pytest tests/qnn/ -q                       # the network, ~2 s
.venv/bin/python -m pytest tests/qthermal/test_thermal.py      # one module
.venv/bin/python -m pytest tests/qthermal/test_run.py::test_name
.venv/bin/python -m pytest tests/test_notebook_equivalence.py -q   # slow: executes notebook cells
```

**Run the full suite before committing anything in `qthermal/` or `qnn/`.**
Two correctness gates, one per half:

- `tests/qthermal/test_diagonalize.py` — the lowest dense eigenvalue plus
  `ecore` must reproduce PySCF's own CASCI energy to 1e-8 Ha, at both ncas = 6
  and ncas = 8.
- `tests/qnn/test_gradients.py` — the **composite** network gradient must match
  central finite differences of the loss, per parameter, for every activation,
  pool, depth, loss and temperature. The hybrid backward pass is a derivation
  ([`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md)), not a library call, and every
  structurally wrong version of it still descends a loss curve.

---

## 8. Inspecting a run file

```bash
.venv/bin/python - <<'EOF'
import h5py
p = "results/qh9_dense_cas8-8_kT0p1.h5"
with h5py.File(p, "r") as f:
    print("meta:", dict(f["meta"].attrs))
    mols = sorted((k for k in f if k.startswith("mol_")),
                  key=lambda s: int(s.split("_")[1]))
    print(f"{len(mols)} molecules; "
          f"{sum(f[m].attrs.get('complete', False) for m in mols)} complete")
    g = f[mols[0]]
    print("datasets:", list(g))
    print("has evals (dense only):", "evals" in g)
    for t in (k for k in g if k.startswith("kT_")):
        b = g[t]
        print(f"  {t}: m={b['p'].shape[0]}, "
              f"trunc={b['truncation_error'][()]:.2e}, "
              f"cap_hit={b.attrs.get('cap_hit')}")
EOF
```

**Always check `cap_hit` and `truncation_error` before trusting a block.**

---

## 9. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: qthermal` | Wrong interpreter, or package not installed editable | `.venv/bin/python`; `.venv/bin/pip install -e . --no-build-isolation` |
| Matplotlib config/permission error | `MPLCONFIGDIR` unset | Prefix `MPLCONFIGDIR=/tmp/matplotlib` |
| `FileNotFoundError: results/*.h5` | HDF5 files are gitignored and local only | Regenerate — [`DATA_CATALOG.md`](DATA_CATALOG.md) |
| `SectorTooLargeError` | dim above a dense guardrail | `--allow-large-dense` (≤70,000) or `--solver iterative` |
| `UnitDetectionError` | Both/neither unit hypothesis passed | Real data problem — inspect the record; do not widen the windows casually ([`INVARIANTS.md`](INVARIANTS.md) I1's lesson) |
| Notebook equivalence test fails numerically | Cells were reordered/inserted | [`INVARIANTS.md`](INVARIANTS.md) I6 — check cell indices in `notebook_test_utils.py` first |
| `--kT-relative` rejected | Needs the sector spectral width | Use `--solver dense` |
| Run appears to redo finished molecules | Groups were incomplete (crash mid-write) | Expected: incomplete groups are deleted and rewritten |
