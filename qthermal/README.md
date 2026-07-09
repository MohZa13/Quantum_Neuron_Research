# qthermal — QH9 interacting thermal-state pipeline (Phase 1)

Converts QH9 equilibrium-geometry records into labeled quantum thermal-state
data: a frozen-core CASCI Hamiltonian in a configurable frontier active space,
dense exact diagonalization in the (nelecas, S_z=0) sector, and Boltzmann-
truncated thermal ensembles with a "quantumness audit" (entropy, natural
occupations, static-correlation score, leading-determinant weight, and a
subspace-projected trace distance to the non-interacting reference).

Phase 1 excludes geometry stretching and SCF; orbitals are the stored QH9
B3LYP/def2-SVP ones ("CASCI on KS orbitals" — intentional, do not "fix" by
running SCF).

## Layout (one module per pipeline stage)

| file | stage |
|---|---|
| `loader.py` | A — `MoleculeRecord`, raw `QH9Stable.db` adapter, empirical unit detection |
| `orbitals.py` | B — PySCF molecule, overlap, orbital validation/recovery |
| `active_space.py` | C — frontier-window `ActiveSpace` (all dims derived, nothing hardcoded) |
| `hamiltonian.py` | D — frozen-core `(ecore, h1eff, g)` via injected-orbital CASCI |
| `diagonalize.py` | E — `SpectralSolver` seam + Phase-1 `DenseEDSolver` (TN backends in Phase 2) |
| `thermal.py` | F — weights, truncation, diagnostics, Gaussian-reference audit |
| `io_hdf5.py` | G — resume-safe gzip HDF5 writer |
| `run.py` | H — CLI + multiprocessing orchestration |

Tests: `tests/qthermal/` (pytest; synthetic B3LYP H2O record generated
end-to-end with PySCF).

## Usage

```bash
python -m qthermal.run --qh9-path /path/to/QH9Stable.db --out run.h5 \
    --limit 100 --n-act-occ 4 --n-act-virt 4 --kT-list 0.05,0.10,0.25 --workers 4
```

## Deviations from the original instructions (all evidence-driven)

1. **Synthetic test record is B3LYP, not RHF.** The `detect_units` physicality
   window (gap in [0.02, 0.6] Ha) rejects RHF H2O (gap 0.67 Ha); QH9 is B3LYP
   anyway.
2. **`detect_units` has a geometric tiebreak.** The spectral test is
   one-sided: misreading Bohr as Angstrom only *stretches* the molecule and
   the eigh(F, S) spectrum can stay physical. When both hypotheses pass, the
   hypothesis whose shortest interatomic distance lies in the covalent window
   [0.7, 1.7] Å wins.
3. **Loader schema taken from `data/build_slater.py`** (raw QH9Stable SQLite
   table `data(id, N, Z, pos, Ham)` + AO reordering) instead of asking for a
   sample record — the repo's own reader documents the format. The
   `# TODO(user)` adapter hook remains in `loader.py`.
4. **`g` is stored full (not s8-packed)**: ncas^4 float64 is ~32 KB at
   defaults and gzip absorbs the redundancy; readers need no unpacking.
5. **Orbital sign gauge is canonicalized in `orbitals()`** (`_canonicalize_signs`):
   each MO column is flipped so its largest-magnitude AO coefficient is
   positive. MO coefficients are only defined up to a per-column sign, so
   without this, `h1eff`, `g`, and `civecs` pick up arbitrary sign flips
   between otherwise-identical molecules — noise to any model trained on the
   raw tensors, not a physical effect. Applied uniformly whether `C` is
   provided or recovered from `eigh(F, S)`, before the orthonormality check.

## Performance (this machine, dim = 4,900)

Sector build ~6 s + `eigh` ~25 s per Hamiltonian (2 Hamiltonians per molecule
including the Gaussian audit) with full BLAS threads — about 60 s/molecule at
defaults, matching the Phase-1 target. The `--workers` pool trades per-worker
BLAS threads for molecule-level parallelism.
