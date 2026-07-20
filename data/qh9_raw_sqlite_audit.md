# Audit: raw-SQLite AO-ordering corruption (2026-07-10)

**Finding.** Raw `QH9Stable.db` Hamiltonian blobs are **already in PySCF
def2-SVP AO ordering**. Two readers in this repo applied the QHBench
qh9→pyscf reorder (`PYSCF_DEF2SVP_CONVENTION`) on top of that — a double
transform that corrupts every record:

- `qthermal/loader.py` `_record_from_row` (fixed 2026-07-09, see
  `qthermal/README.md` Deviations item 6);
- `data/build_slater.py` `RawQH9SQLiteDataset.__getitem__` (fixed
  2026-07-10, this audit). The LMDB fallback path in the same script never
  transformed, and the CLI even refuses `--ham-ordering qh9` — the raw-SQLite
  adapter silently violated its own contract.

**Evidence.** Untransformed blobs for records 0–4 match freshly converged
B3LYP/def2-SVP spectra at the stored geometries to ≤ 3.7×10⁻³ Ha across the
full spectrum (B3LYP-variant-level agreement). Transformed blobs show ~1 Ha
spurious core shifts on compact molecules and catastrophic intruder
eigenvalues on linear ones (HCN −57.7 Ha, C₂H₂ −173.7 Ha; true 1s levels are
−14.4 / −10.2 Ha). For 9/9 sampled `data/groups` records, re-applying the old
double transform reproduces the stored HOMO/gap values **exactly** —
mechanism and provenance are certain.

## Scope

| Artifact | Verdict | Basis |
|---|---|---|
| `data/groups/qh9_slater_*.h5` (95 files, 125,013 records, 284 GB) | **Corrupted — removed 2026-07-13** | every file's attrs say `loader_module='raw SQLite QH9Stable.db'`; 9/9 sampled records reproduce the double transform exactly. Tree deleted with the branch retired; regenerate per below only if the single-determinant baseline is needed |
| `data/qh9_scan.jsonl` | Clean | contains only geometry-derived signatures (nao, nelec, n_spin_orbitals); Ham never read |
| `data/build_state_vectors.py`, `data/active_space_encode.py` | Clean (code) — removed 2026-07-13 with the retired branch | no raw-DB or transform usage; consumed the corrupted groups files; recoverable from git history |
| `results/*.csv`, `figures/*` (quantum-neuron benchmarks) | Clean | no dependence on QH9 data |
| `results/qh9_krylov_ncas10.h5`, validation runs | Regenerated 2026-07-10 on corrected loader | qthermal outputs |
| `results/h2o_cas8-6_kT0p025.h5` | Clean | built from a synthetic PySCF record, not the DB |

No code in the repository currently *consumes* `data/groups/*.h5`
(verified by repo-wide search), so the corruption has not propagated into any
downstream result — the damage is confined to the group files themselves.

## Error magnitude in the corrupted group files

Sampled across the 8 smallest files (n=9 records):

- HOMO error: median **0.74 eV**, max **3.8 eV**
- gap error: median **0.92 eV**, max **3.0 eV**
- Slater weight matrices `W`: entries shift by up to **0.93**
- `y_gap_binary`: labels flip near the threshold (the stored threshold
  itself — a gap median — is corrupted: 8.107 eV stored vs 9.175 eV correct
  for the 48q group)

## Fix verification

`data/build_slater.py --source raw-db --target-nao 24` (fixed) on a 10-row
copy of the DB rebuilds the water (qh9_index=2) group with
homo = −7.8451 eV, gap = 9.1752 eV — matching an independent
`eigh(Ham, S)` recompute of the untransformed blob to <10⁻⁴ eV, and matching
`qthermal`'s corrected loader.

## Regeneration

**Update 2026-07-13:** the corrupted `data/groups/` tree was deleted and the
single-determinant branch retired (`qthermal` supersedes it). The
instructions below remain valid if the baseline is ever rebuilt.

The 284 GB `data/groups/` tree must be rebuilt with the fixed script. The
raw-db path expects `{root}/QH9Stable/raw/QH9Stable.db`; the DB currently
sits at `data/QH9Stable.db`, so create the layout once:

```bash
mkdir -p data/qh9root/QH9Stable/raw
ln -s ../../../QH9Stable.db data/qh9root/QH9Stable/raw/QH9Stable.db
```

Then (resumable chunks append; safe to re-run overlapping ranges — existing
`original_index` values are skipped):

```bash
python3 data/build_slater.py \
  --source raw-db --root data/qh9root \
  --scan-from data/qh9_scan.jsonl \
  --all-groups --out-dir data/groups_v2 --out-prefix qh9_slater \
  --max-qubits 400 --max-samples 0 \
  --start-index 0 --stop-index 10000
```

(`data/qh9_scan.jsonl` is geometry-only and therefore reusable as-is.)
Expect a similar cost to the original build (~4 days for the full 125k
records; the June build ran Jun 24–28). Write into `groups_v2` and swap the
directories only after spot-checking, so corrupted and corrected data are
never mixed in one tree. The old tree should then be deleted or archived —
its `ham_ordering='pyscf'` attribute is wrong.
