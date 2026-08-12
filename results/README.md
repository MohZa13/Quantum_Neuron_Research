# results/ — generated artifacts

**Authoritative catalog — provenance, schema, status, regeneration cost:**
[`../docs/DATA_CATALOG.md`](../docs/DATA_CATALOG.md). This file is the quick
orientation.

---

## Read this first

**`*.h5` files are gitignored and exist only on this machine.** Never assume
one is present — check, and regenerate per the catalog if not. The production
file alone is 45 GB and takes 13.3 hours to rebuild.

**Always check `cap_hit` and `truncation_error` before trusting a block.** They
record exactly what was discarded. A block with `cap_hit=True` at kT = 0.25 may
be missing over 1% of its thermal weight — recorded, but not negligible.

**`evals` exists only in dense-solver runs.** Iterative (Krylov) runs omit it by
contract ([`../docs/INVARIANTS.md`](../docs/INVARIANTS.md) I5).

---

## What is here

### Thermal-state run files (`qthermal.run` output)

| File | Contents |
|---|---|
| **`qh9_dense_cas8-8_kT0p1.h5`** | **★ Production set** — 1000 molecules, CAS(8,8), kT = 0.1 Ha, 45.4 GB |
| `qh9_conjugated_top45.h5` | Conjugated subset, kT ∈ {0.1, 0.25} — **28 complete, resumable** |
| `qh9_dense_cas8-8_kT0p25.h5` | 4 molecules, kT = 0.25 — predates `--keep-cap 0`; capped blocks miss 0.1–4% weight |
| `qh9_krylov_ncas12_hcn.h5` | HCN at ncas = 12, **dim 853,776** — the Krylov reach milestone |
| `qh9_krylov_ncas10.h5` | 2 molecules at ncas = 10, dim 63,504 |
| `qh9_dense_cas88_5mols.h5`, `qh9_dense_cas8-6*.h5`, `h2o_cas8-6_kT0p025.h5` | Small dev/test fixtures |
| `qh9_krylov_ncas8.h5` | **Empty** — abandoned run, safe to delete |

### Derived

- `qh9_dense_cas8-8_kT0p1_extheis.h5` — 248 Pauli features × 1000 molecules,
  plus the Z₂-tapered 14-qubit basis
- `thermal_training_5mol.h5` — the `{ρ_m, y_m}` bridge format (5 examples,
  placeholder label)
- `spin_labels_kT0p1.npz` — the real labels: `⟨S²⟩`, its diagonal part `D`, and
  the coherence-only part `c = ⟨S²⟩ − D`, for all 1000 molecules
- `spin_comparison_metrics.json` — the **single-neuron** result on those labels
- `hybrid_spin_metrics_8q.json`, `hybrid_spin_metrics_10q.json` — the
  **network** result (paper §VII.C) on the same labels, same split, same seed,
  so the two read side by side. Each re-renders its figure with `--replot`

### Tables

- `qh9_conjugation_screen_full.csv` — **the full 130,812-molecule screen**;
  the selection instrument for targeted runs
- `qh9_conjugation_screen.csv` — first 1000 records
- `coherence_share_kT0p1.csv` — ⚠️ **orphan**, no producer script; basis of the
  coherence-confound finding
- Classifier benchmark CSVs (`pennylane_*`, `logloss_*`, `equivalence_check`)
  and `digitized/` — all from **synthetic** Haar-random states, not molecules

### Logs

`hybrid_spin_8q.log`, `hybrid_spin_10q.log` — per-epoch training logs. Worth
keeping beyond provenance: they carry `sat` (mean divided difference, in units
of `φ'(0)`) and `rho(B)` (largest eigenvalue of the pre-activation operator),
the only record of whether the quantum layer stayed in a trainable regime.

`qh9_kT0p1_extend_to_1000.log`, `qh9_conjugated_top45.log`,
`qh9_conjugation_screen_full.log`.

Logs are not disposable — the cap warnings in them are the only record of which
blocks hit the storage ceiling. Keep them with their datasets.

---

## Naming convention

```
qh9_<solver>_cas<nelec>-<ncas>_kT<value>[_<variant>].h5
       │              │            │
       │              │            └── 0.1 → kT0p1
       │              └── cas8-8 = 8 electrons, 8 orbitals
       └── dense | krylov
```

Non-QH9 sources lead with the molecule (`h2o_cas8-6_kT0p025.h5`).

## When you add a file here

Add a row to [`../docs/DATA_CATALOG.md`](../docs/DATA_CATALOG.md) **in the same
session**, with the exact generating command. An artifact without provenance
cannot be verified or reproduced — we already carry two such orphans and they
are a standing liability.
