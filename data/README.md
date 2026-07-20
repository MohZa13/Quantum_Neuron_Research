# QH9 data directory

Contents:

- `QH9Stable.db` — raw QH9 SQLite database (~30 GB, untracked): DFT Fock
  matrices, geometries, atomic numbers for 130,831 molecules.
  **AO ordering: the Ham blobs are already in PySCF def2-SVP order** — do not
  apply the QHBench qh9→pyscf reorder to them (see
  [`qh9_raw_sqlite_audit.md`](qh9_raw_sqlite_audit.md)).
- `qh9_scan.jsonl` — geometry-derived size index of every record (nao, nelec,
  spin-orbital count). Hamiltonian-independent; reusable for group selection.
- `build_slater.py` — retired-baseline builder (see below).
- `qh9_raw_sqlite_audit.md` — 2026-07-10 audit of the AO-ordering
  double-transform bug: mechanism, scope, error magnitudes, and regeneration
  instructions.

The primary pipeline consuming this data is `qthermal/` (see
`qthermal/README.md` and the top-level `README.md`).

## build_slater.py — retired single-determinant baseline

`build_slater.py` builds compact Slater-determinant weight matrices `W`
(and `D_occ`, HOMO/LUMO/gap labels) per molecule by solving the generalized
eigenproblem `H C = S C eps` — a *mean-field, zero-temperature*
representation with no explicit inter-electronic interactions. It predates
`qthermal` and is superseded by it for the project's thermal-state goals.

The `data/groups/qh9_slater_*.h5` tree it produced (125,013 records, 284 GB,
built June 2026) was **corrupted** by the AO-ordering bug and was **removed
on 2026-07-13** together with its downstream encoders
(`active_space_encode.py`, `build_state_vectors.py` — recoverable from git
history). The builder itself is kept, with the bug fixed and verified,
because it is the regeneration route if a single-determinant baseline is
ever wanted for comparison against the interacting thermal-state pipeline.

To regenerate (multi-day job; resumable): follow the "Regeneration" section
of [`qh9_raw_sqlite_audit.md`](qh9_raw_sqlite_audit.md). The script's raw-db
path expects `{root}/QH9Stable/raw/QH9Stable.db`; run
`python3 data/build_slater.py --help` for the full CLI (group selection by
qubit count, chunked `--all-groups` mode, scan reuse via
`--scan-from data/qh9_scan.jsonl`).
