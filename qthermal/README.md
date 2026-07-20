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
| `diagonalize.py` | E — `SpectralSolver` seam + `DenseEDSolver` + matrix-free `IterativeWindowSolver` (TN backends in Phase 2) |
| `thermal.py` | F — weights, truncation, diagnostics, Gaussian-reference audit |
| `io_hdf5.py` | G — resume-safe gzip HDF5 writer |
| `run.py` | H — CLI + multiprocessing orchestration |
| `encode_run.py` | I-CLI — batch `run.h5` → extended-Heisenberg Pauli coefficient vectors (`Tr(rho P)` per molecule per kT), determinant-basis contraction, resume-safe |
| `encode.py` | I — eigenblock → qubit states: Jordan-Wigner (2·ncas qubits; `ordering="blocked"` sign-free, `"interleaved"` with parity signs) and sector compression (⌈log2 dim⌉ qubits); PennyLane JW Hamiltonian for validation/Pauli inspection; `extended_heisenberg_paulis/_labels/_expectations` for the weight-≤2 classifier basis. Measured: blocked reads ~10x more connected-ZZ + the only nonzero XX/YY under the nearest-neighbor Heisenberg ansatz at kT=0.25 (interleaved's adjacent pairs are all spin-flip ⇒ XX/YY vanish by S_z conservation); interleaved wins only the small low-kT same-orbital pairing signal, and remains the right layout for future MPS backends |

Tests: `tests/qthermal/` (pytest; synthetic B3LYP H2O record generated
end-to-end with PySCF).

## Usage

```bash
python -m qthermal.run --qh9-path /path/to/QH9Stable.db --out run.h5 \
    --limit 100 --n-act-occ 4 --n-act-virt 4 --kT-list 0.05,0.10,0.25 --workers 4

# eigenblocks -> extended-Heisenberg Pauli-string coefficients (Module I);
# --taper adds the Z2-tapered basis (block parities removed, 2*ncas-2 qubits)
python -m qthermal.encode_run --in run.h5 --out run_extheis.h5 --taper
```

## Solvers

- `--solver dense` (default): full `eigh` on the sector matrix. Exact tail
  weights, stores the full spectrum (`evals`). Guardrails: silent to
  dim = 5,000; `--allow-large-dense` + RAM precheck to 70,000; refuses beyond.
  Stored eigenvectors (per ensemble and per kT block) are capped at
  `max(1024, dim // 4)` by default; when the Boltzmann window at kT_max wants
  more (e.g. kT = 0.25 at dim = 4,900), the run warns `ensemble cap bound`,
  keeps the cap, and records the exact discarded mass in `truncation_error`
  with `cap_hit=True` — energy-only quantities stay exact via `evals`.
  `--keep-cap N` overrides the cap; `--keep-cap 0` lifts it entirely so the
  `--weight-cutoff` alone decides, at m x dim float64 storage per ensemble
  (~192 MB/molecule/ensemble pre-gzip at dim = 4,900 when the window spans
  the sector).
- `--solver iterative`: matrix-free Krylov (`IterativeWindowSolver`) — PySCF
  FCI Davidson converges only the low-energy window through `contract_2e`
  matvecs; the dense matrix is never formed, so it has no dim guardrails,
  only an O(k·dim) RAM precheck. The root count k doubles until the rigorous
  counting bound

      tail <= sum_{i=m..k-1} e^{-(E_i-E0)/kT_max} + (dim-k) e^{-(E_{k-1}-E0)/kT_max}

  certifies the weight cutoff (with one converged-but-unkept buffer root), or
  `--max-nroots`/RAM caps escalation (`cap_hit`, honest oversized bound).
  No `evals` dataset is stored — per-kT weights normalize over the kept
  window and fold the certified `tail_weight` into `truncation_error`.
  This is the **low-kT** backend: correct at any temperature, but once the
  thermal window holds thousands of states (e.g. kT = 0.25 Ha at large ncas)
  it will cap out — that regime belongs to a Phase-2 sampling backend
  (TPQ/METTS). `--kT-relative` requires the dense solver (needs the spectral
  width). The certificate assumes Davidson missed no root below `E[k-1]`;
  mitigations: pspace-seeded guesses, the buffer root, and a prefix-
  consistency check across escalations (all roots must also converge, else
  the solve raises rather than emit an invalid bound).

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
6. **Raw `QH9Stable.db` Hamiltonians are already PySCF-ordered — the loader
   applies NO AO reordering** (found 2026-07-09). `data/build_slater.py`'s
   raw-SQLite path applies the QHBench `PYSCF_DEF2SVP_CONVENTION` reorder to
   these blobs, which double-transforms them: fresh-B3LYP comparison on
   records 0-4 shows the reordered spectra carry ~1 Ha spurious core shifts
   on compact molecules (inside the loose physicality windows — detection
   passed anyway) and intruder eigenvalues on linear molecules (HCN -57.7 Ha,
   C2H2 -173.7 Ha, both failing unit detection individually). Un-reordered
   blobs agree with fresh B3LYP to <= 3.7e-3 Ha across the full spectrum on
   all five. The transform helpers remain in `loader.py` for QHBench
   processed/model-output matrices only. **Audit outcome (2026-07-10)**:
   `build_slater.py`'s raw-SQLite path had the same bug (now fixed); all 95
   `data/groups/*.h5` files (125,013 records) were corrupted — the tree was
   removed on 2026-07-13 with the single-determinant branch retired;
   `data/qh9_scan.jsonl` and everything else in the repo are clean — see
   `data/qh9_raw_sqlite_audit.md`.

## Performance (this machine)

Dense, dim = 4,900: sector build ~6 s + `eigh` ~25 s per Hamiltonian
(2 Hamiltonians per molecule including the Gaussian audit) with full BLAS
threads — about 60 s/molecule at defaults, matching the Phase-1 target. The
`--workers` pool trades per-worker BLAS threads for molecule-level parallelism.

Iterative (kT_max = 0.05, QH9 mols 0-2, dim = 4,900): 29-76 s/molecule —
comparable to dense at this size; the win is reach, not constant factors.
At ncas = 10 (dim = 63,504, dense would need ~97 GB here): 4.5-6 min/molecule
at kT_max = 0.025 with escalation to k = 64 (C2H2/HCN pi degeneracies want
the wide window). Each escalation currently re-solves from scratch —
warm-starting Davidson from the previous roots is the obvious next
optimization if escalations dominate. Davidson stalls on exactly degenerate
multiplets straddling the window edge (typical in the g = 0 reference sector
of high-symmetry molecules) are handled by the same escalation; the solver
only raises if unconverged at the root ceiling with nothing to fall back on.
