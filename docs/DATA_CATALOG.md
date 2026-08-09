# Data catalog

*Every dataset and generated artifact: what it is, what made it, what it costs
to remake, and whether you can trust it. Last audited **2026-08-05**.*

**Rule: no artifact without provenance.** If you generate a file, add a row
here in the same session. We already carry two orphans (bottom of this file) —
do not create a third.

`*.h5`, `*.db`, `*.pt` are gitignored. They exist only on this machine and
**must never be assumed present** — check first, regenerate if absent.

---

## 1. Source data

### `data/QH9Stable.db` — ~30 GB, untracked

The raw QH9 SQLite database: 130,831 small organic molecules (H, C, N, O, F),
each with geometry, atomic numbers, and a converged B3LYP/def2-SVP Fock matrix.

| | |
|---|---|
| **Schema** | `data(id INTEGER, N INTEGER, Z BLOB int32, pos BLOB float64 (N,3), Ham BLOB float64 (nao,nao))` |
| **AO ordering** | **Already PySCF def2-SVP order. Apply NO reorder.** ([`INVARIANTS.md`](INVARIANTS.md) #1) |
| **Units** | Not declared — detected empirically per run (`qthermal/loader.py`), observed **Ångström** on every run so far |
| **Read by** | `qthermal/loader.py::iter_records`, `scripts/screen_conjugation.py`, `data/build_slater.py` |
| **Provenance** | Public dataset — [arXiv:2306.09549](https://arxiv.org/abs/2306.09549) |

### `data/qh9_scan.jsonl` — 15 MB, tracked

Geometry-derived size index of every record (nao, nelec, spin-orbital count).
**Hamiltonian-independent, therefore unaffected by the AO-ordering bug** and
reusable as-is for group selection.

---

## 2. Thermal-state run files (`qthermal.run` output, Module G layout)

Schema for every file in this section:

```
/meta                          attrs: basis, n_act_occ, n_act_virt, ncas, nelecas,
                                      solver_name, kT_list, kT_convention,
                                      weight_cutoff, keep_cap (-1 = default),
                                      code_version, pyscf_version, unit
/mol_{idx}/                    attrs: complete=True   ← written LAST (resume marker)
    Z, R, active_idx, nocc     provenance
    ecore, h1eff, g            the Hamiltonian (g = full ncas⁴, chemist notation)
    evals                      full sector spectrum — DENSE SOLVER ONLY; readers
                               must not assume it exists
    kT_{tag}/                  E, p, civecs, truncation_error, entropy, nat_occs,
                               static_corr, c_max_sq, tracedist_gaussian,
                               tracedist_bound;  attrs: kT, cap_hit
```

| File | Size | Contents | Status |
|---|---:|---|---|
| **`results/qh9_dense_cas8-8_kT0p1.h5`** | **45.4 GB** | **★ PRODUCTION SET.** 1000 molecules (QH9 ids 0–999), CAS(8,8), kT = 0.1 Ha, dense ED, `keep_cap` default = 1225 | Complete, 1000/1000 |
| `results/qh9_conjugated_top45.h5` | 3.97 GB | Top conjugated molecules from the full screen, CAS(8,8), kT ∈ {0.1, 0.25}, `keep_cap = 2450` | **In flight — 28 complete.** Resumable: rerun the same command |
| `results/qh9_dense_cas8-8_kT0p25.h5` | 179 MB | 4 molecules at kT = 0.25 Ha | Partial; predates `--keep-cap 0`, so capped blocks are missing 0.1–4% of thermal weight (recorded) |
| `results/qh9_dense_cas88_5mols.h5` | 1.6 MB | 5 molecules, CAS(8,8), kT = 0.025 | Complete — the smoke-test input for the bridge |
| `results/qh9_dense_cas8-6.h5` | 135 KB | 3 molecules, CAS(8,6), kT = 0.025 | Complete — small-scale dev fixture |
| `results/qh9_dense_cas8-6_kT0p25.h5` | 1.1 MB | 3 molecules, CAS(8,6), kT = 0.25 | Complete — the MPS-ordering benchmark input |
| `results/qh9_krylov_ncas10.h5` | 10.5 MB | 2 molecules, **ncas = 10** (dim 63,504), iterative Krylov, kT = 0.025 | Complete. Regenerated 2026-07-10 on the corrected loader |
| `results/qh9_krylov_ncas12_hcn.h5` | 73.5 MB | **HCN at ncas = 12, dim 853,776** — the Krylov reach milestone | Complete, certified |
| `results/h2o_cas8-6_kT0p025.h5` | 52 KB | Synthetic PySCF H₂O, not from the DB | Complete — clean by construction |
| `results/qh9_krylov_ncas8.h5` | 6.8 KB | **Empty** — meta only, zero molecules | **Failed/abandoned run.** Safe to delete |

**Regeneration cost** (this machine, CAS(8,8) defaults): median **192 s per
molecule**, max 313 s. The 1000-molecule production run took **47,725 s
(~13.3 h)** wall with `--workers`. Commands: [`WORKFLOWS.md`](WORKFLOWS.md) §1.

---

## 3. Derived feature and training files

### `results/qh9_dense_cas8-8_kT0p1_extheis.h5` — 5.3 MB

248 extended-Heisenberg Pauli coefficients `Tr(ρP)` per molecule, for all 1000
molecules. Produced by `qthermal.encode_run` from the production set,
`ordering = blocked`. Includes the Z₂-tapered basis (14 qubits).

```
/meta                  ordering, ansatz, n_terms=248, ncas/nalpha/nbeta,
                       source_file, n_qubits_tapered, taper_removed_wires,
                       taper_sector + copied run provenance
/pauli_labels          (248,) bytes
/pauli_labels_tapered  (248,) bytes      ← --taper
/taper_signs           (248,) int8 ±1    ← coefficient on the tapered register
/taper_kept_wires      (14,) int64          is taper_signs[k] * coeffs[k]
/mol_{idx}/Z, kT_{tag}/coeffs   (248,) float64; attrs kT, trace,
                                truncation_error, cap_hit
```

Cheap to regenerate (contraction happens in the determinant basis, ~1 s per
molecule) — see [`WORKFLOWS.md`](WORKFLOWS.md) §2.

### `results/thermal_training_5mol.h5` — 1.3 MB

The bridge format: `{ρ_m, y_m}` for the Julia trainers, from
`scripts/export_thermal_training.py`. 5 examples, `ordering = interleaved`,
`label_spec = h5:static_corr`.

```
/meta            n_system_qubits=16, ncas, nalpha, nbeta, ordering,
                 sector_dim=4900, encoding='jw_sector_sparse', kT,
                 label_spec, label_threshold, n_examples
/basis_indices   (4900,) int64   — shared by every example, stored once
/labels, /mol_indices, /formulas, /label_values
/example_{k}/    amps (m, 4900) float64 (JW parity signs folded in),
                 weights (m,);  attrs mol_idx, label, formula, trace,
                 truncation_error, n_states, n_states_full
```

Recover a state vector: `v = zeros(2**16); v[basis_indices] = amps[k]`
(Julia: `v[basis_indices .+ 1] = amps[k, :]`).

> ⚠️ **`label_spec = h5:static_corr` is an acknowledged placeholder.**
> `static_corr` is a one-body quantity that a classical model reads directly.
> Choosing the real label is the project's open question — see
> [`QUANTUM_NEURON.md`](QUANTUM_NEURON.md) §4 and [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) Q1.

---

## 3b. Spin-label study (2026-08-05)

### `results/spin_labels_kT0p1.npz` — 30 KB, tracked

Thermal spin observables for all 1000 molecules of the production set, from
`scripts/spin_labels.py`. On the S_z = 0 sector `S² = D + S²_od`, so:

| key | meaning |
|---|---|
| `S2` | `Tr(ρ S²)` — total thermal open-shell / triplet character |
| `D` | `Tr(ρ D)` — the **diagonal**, classically readable part (count of singly-occupied orbitals holding β) |
| `c` | `Tr(ρ S²_od)` — the **coherence-only** part, `c = S2 − D` |
| `idx`, `formula`, `trace`, `n_states`, `rho_trace_kept`, `keep_idx`, `n_qubits` | provenance |

The sector `S²` matrix is molecule-independent and built once from PySCF's
`contract_ss`; its diagonal is cross-checked against an independently derived
open-shell count (agreement 0.0e+00). Regeneration: ~15 min.

```bash
.venv/bin/python -m scripts.spin_labels \
    --in results/qh9_dense_cas8-8_kT0p1.h5 --out results/spin_labels_kT0p1.npz \
    --rho-out <scratch>/rho_10q.npy --n-qubits 10 --sample 80
```

### `<scratch>/rho_10q.npy` — 4.2 GB, **not tracked, not kept**

`(1000, 1024, 1024)` float32 stack of each thermal state projected onto the 1024
most-populated determinants (10 qubits). Retains ≥96.7% of thermal weight
(median 99.8%) and **99.79% of off-diagonal Frobenius weight** (min 96.41%) —
verified so the projection cannot be blamed for a null coherence result.
Regenerate with `--rho-out` above; scratch-only by design.

### `results/spin_comparison_metrics.json` — tracked

Full output of `scripts/train_spin_comparison.py`: per-label R± screening
scores, per-model final/best accuracy and loss, complete epoch histories, the
classical-descriptor baseline, and the positive control's exact-solution
diagnostic. The figure can be re-rendered from it alone (`--replot`), no
retraining and no ρ stack needed.

### `figures/spin_quantum_vs_classical.png`

Quantum vs classical Fermi-Dirac neuron on three labels (`⟨S²⟩`, `c`, and a
synthetic purely off-diagonal positive control). **Headline: quantum − classical
= +0.00 points on both physical labels.** Producer:
`scripts/train_spin_comparison.py`. See `RESEARCH_LOG.md` 2026-08-05.

---

## 3c. Hybrid-network study (2026-08-05)

The same labels, split, and seed as §3b, run through the **network** of
`Papers/Fermi-Dirac Machines.pdf` §VII.C instead of the single neuron —
`scripts/train_hybrid_spin.py` over [`qnn/`](../qnn/README.md). Five models per
label: `quantum`, `z_only`, `diagonal_full`, `quantum_shallow` (no classical
layers), `quantum_linear` (identity activation).

### `results/hybrid_spin_metrics_8q.json` — tracked

Full output at an 8-qubit register (the 256 most-populated determinants,
sub-projected from the 10-qubit stack). Per-label R± screening scores, per-model
final/best accuracy and loss, complete epoch histories **including the
`saturation` and `spectral_radius` diagnostics**, the classical-descriptor
baseline, and the run configuration. The figure re-renders from it alone
(`--replot`) — no retraining, no ρ stack.

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m scripts.train_hybrid_spin \
    --labels results/spin_labels_kT0p1.npz --rho <scratch>/rho_10q.npy \
    --project-qubits 8 --epochs 600 --temperature 4.0 --lr 0.02 --l2 0.0 \
    --out figures/hybrid_spin_8q.png --json-out results/hybrid_spin_metrics_8q.json
```

Cost: **1063 s measured** (of which ~90 s is loading and projecting the 4.2 GB
stack). Hyperparameters were chosen by a sweep on the **control label only**
(`RESEARCH_LOG.md` 2026-08-05) and then applied unchanged to all three labels.

Register retention: **97.4% of the population, 95.5% of the off-diagonal
Frobenius weight**. The second number is the one that makes a coherence result
interpretable; it was backfilled into the file's `config.projection` by
re-running `project_register` on the same stack, because the diagnostic was
added after this run launched — the file records that provenance in a `note`.

### `results/hybrid_spin_metrics_10q.json`, `figures/hybrid_spin_10q.png`

The **control label only**, at the full 10-qubit register, 500 epochs, models
`quantum, z_only, diagonal_full`. Its purpose is not to re-confirm the 8-qubit
result but to make the comparison against the *single neuron* controlled: at
`n_qubits = 10` and `seed = 7` the synthetic control operator is bit-identical to
the one in `results/spin_comparison_metrics.json` — verified by the R± screen
ratio, **0.3435 in both files**. The single neuron scored 66.7% on it (classical
pool 66.3%, descriptor baseline 67.0%, and an exactly-representable solution that
scores 100% but which its spectral loss declines to find).

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m scripts.train_hybrid_spin \
    --labels results/spin_labels_kT0p1.npz --rho <scratch>/rho_10q.npy \
    --epochs 500 --temperature 4.0 --lr 0.02 --l2 0.0 \
    --labels-subset control --models quantum,z_only,diagonal_full \
    --out figures/hybrid_spin_10q.png --json-out results/hybrid_spin_metrics_10q.json
```

Cost: **4133 s measured**, essentially all of it the one `quantum` run (4069 s
for 500 epochs; the two diagonal models took 13 s and 18 s). The quantum layer
is `O(K³)` per neuron per epoch and **independent of dataset size**, while a
commuting pool takes the `O(K)` path — a ~250× gap here. That asymmetry is why
`--project-qubits` exists.

**Result**: quantum 93.0%, Z-only 77.0%, unconstrained diagonal 77.3%,
descriptors 67.0% — against the single neuron's 66.7 / 66.3 / 69.3 / 67.0 on the
same label. `RESEARCH_LOG.md` 2026-08-06.

The file's `descriptor_baseline` was recomputed on the canonical split after a
fix (the run predates it and reported 62.0% on its own draw); the entry records
that provenance in a `note`.

### `figures/hybrid_spin_8q.png`

Held-out cross-entropy and accuracy per epoch, three labels × five models, with
the R± screen and the classical-descriptor baseline annotated. Producer:
`scripts/train_hybrid_spin.py`.

### `results/hybrid_spin_8q.log`, `results/hybrid_spin_10q.log`

Per-epoch training logs. Worth keeping for one reason beyond provenance: they
carry `sat` (mean divided difference, in units of `φ'(0)`) and `rho(B)` (largest
eigenvalue of the pre-activation operator), which is the only record of whether
the quantum layer was in a trainable regime or had saturated.

---

## 3f. Purification-MPS thermal states, ncas = 10 (2026-08-08, Module K)

### `results/qh9_mps_ncas10.h5` — ~130 MB, untracked

**Producer:** `QThermalMPS/bin/thermal.jl`-style scripts (this instance:
session scratchpad `prod10.jl` / `prod10_mol4.jl`), imaginary-time TDVP on the
purification chain, `maxdim = 256`, `cutoff = 1e-8`, graded `dbeta = 0.05`,
`:blocked` ordering.
**Input:** `results/qh9_krylov_ncas10.h5` (mol_3 = C2H2, mol_4 = HCN;
CAS(10,10), sector 63,504).

```
/meta                 attrs: ncas, nelecas, nalpha, nbeta, ordering=blocked,
                             nwires=20, nsites=40, sector_dim, dbeta, maxdim,
                             cutoff, source_file
/mol_{i}/             attrs: complete, ecore, dmrg_E0, krylov_E0,
                             beta0_energy_error   (per-molecule validation)
    Z
    kT_{tag}/         attrs: kT, beta;  logZ, energy, free_energy, entropy,
                      maxlinkdim, steps, seconds
        psi           the purification MPS (ITensorMPS HDF5 format)
        rho           dense 1024x1024 float64: reduced density matrix over
                      rho_wires = 0..9 — the ALPHA spin-orbitals (blocked
                      ordering), wire 0 = MOST significant bit (Module I).
                      Unit trace BY CONSTRUCTION (partial trace of a
                      normalised purification): no truncation-trace leak.
        rho_wires
```

Validation carried in the file: `beta0_energy_error` (~1e-14, vs the closed
form `sector_mean_energy`) and `dmrg_E0` vs `krylov_E0` (~1e-8, two codebases).
**Error bar:** rungs at kT <= 0.5 hit the `maxdim` cap and carry ~1-2e-2 Ha,
dominated by `cutoff = 1e-8` (RESEARCH_LOG 2026-08-08). Regeneration: ~1.5 h
per molecule on 8 threads + ~45 min RDM export.

### `results/mps_thermal_training.json` — tracked

**Producer:** `scripts/train_mps_thermal.py`. The first Julia->Python bridge
run: qnn `HybridNetwork` trained on the 12 states (both molecules; hotcold,
molecule, kT-regression; LOO; quantum vs z_only pools; native-K=1024 timing).
**Consumes** `qh9_mps_ncas10.h5` after bit-reversal to qnn's little-endian
register (tests: `tests/test_mps_bridge.py`). Carries its own caveat string:
every label here is diagonal-visible, so this is a plumbing result, not a
coherence result.

### `results/pair_screen.json` — tracked (2026-08-09)

**Producer:** `scripts/pair_screen.py`. 450 molecule-pair screens (isomer /
isoelectronic / control, 150 each) on the shared 1024-determinant register:
is molecular identity at matched kT an off-diagonal-dominated label?
Headline: isomer pairs median ratio 2.76, 87% >= 1. Consumes the production
h5 + `spin_labels_kT0p1.npz` (reuses its `keep_idx` register and
`rho_trace_kept` as a per-molecule consistency check).

### `results/pair_transfer_top45.json` — tracked (2026-08-09)

**Producer:** `scripts/pair_transfer_top45.py`. Few-shot cross-temperature
molecule identification on the top45 set (40 isoelectronic pairs, 2-shot
train at one kT, test at the other): chance for both pools — the honest
negative that motivates multi-temperature training sets. See RESEARCH_LOG
2026-08-09 for the two confounds.

## 4. Screening tables

| File | Rows | Contents |
|---|---:|---|
| `results/qh9_conjugation_screen_full.csv` | 130,812 | **Full QH9 screen.** Columns: `idx, formula, n_heavy, DoU, gap_Ha, n_frontier_within_kT, largest_pi_atoms, n_aromatic_atoms`. Sorted ascending by gap (most quantum first) |
| `results/qh9_conjugation_screen.csv` | 1,000 | Same columns, first 1000 records |
| `results/qh9_conjugation_screen_full.csv.partial` | — | Resume checkpoint (gitignored). **Differs from the final file** — it is append-ordered, the final is gap-sorted |

Producer: `scripts/screen_conjugation.py`. Tier 3 columns are blank if RDKit is
unavailable. Multi-hour job over the full DB; resumable.

---

## 5. Classifier benchmark outputs

All produced from synthetic Haar-random states, not molecular data.

| File(s) | Producer |
|---|---|
| `benchmarks/scaling_results.csv`, `scaling_results_quick.csv` | `benchmarks/benchmark_scaling.py` |
| `benchmarks/paper_{training_curves,efficiency_summary,sampling_efficiency}_2_4_7.csv` | `benchmarks/benchmark_paper_comparison.py` |
| `results/pennylane_quantum_{2..7}qubit.csv` | `figures/run_pennylane_vs_original.py` — 500-epoch loss history |
| `results/pennylane_matched_{2..7}qubit.csv` | `figures/run_matched_pennylane.py` — matched quantum + FCIM trial |
| `results/equivalence_check.csv` | `figures/run_pennylane_vs_original.py` — per-n max abs diff over a shared 10-epoch window + per-epoch timing |
| `results/logloss_{3,4}qubit_cl_heisenberg.csv`, `results/logloss_{4,5,7}qubit_phase2.csv` | The notebooks' own benchmark cells |
| `results/digitized/fig8_classical_{2..7}qubit_digitized.csv` | `figures/digitize_fig8_classical.py` from `docs/fig_8.png` |

---

## 6. Figures

| File | Producer |
|---|---|
| `figures/qh9_quantum_neuron_training.png` | `scripts/demo_train_curve.py` |
| `figures/paper_training_curves_2_4_7.png`, `paper_efficiency_comparison_2_4_7.png`, `paper_sampling_efficiency_2_4_7.png` | `benchmarks/plot_paper_comparison.py` |
| `figures/fig8_pennylane_reproduction.png` | `figures/plot_fig8_pennylane.py` |
| `figures/efficiency_comparison.png` | `figures/plot_efficiency_comparison.py` |
| `figures/notebook_comparison_2_4_7.png` | `figures/benchmark_notebook_comparison.py` |
| `figures/h2o_cas8-6_H_rho_kT0p025.png` | Ad-hoc (no committed script) — H and ρ heatmaps for the H₂O fixture |
| `figures/qh9_cas8-8_kT0p1_diagnostics.png` | Ad-hoc — 50-molecule diagnostics summary |
| `figures/qh9_cas8-8_kT0p1_diagnostics_1000mol.png` | Ad-hoc — 1000-molecule version |
| `docs/fig_8.png` | The source paper's Fig. 8 (input, not output) |

---

## 7. Run logs

| File | |
|---|---|
| `results/qh9_kT0p1_extend_to_1000.log` | The production run, 2026-07-20/21. Ends: *"run finished in 47725.1 s: 950 written, 0 skipped, 50 resumed"* — i.e. it extended an existing 50-molecule file to 1000. Begins with a traceback from an earlier aborted attempt |
| `results/qh9_conjugated_top45.log` | The conjugated-subset run, 2026-08-05. Shows repeated `ensemble cap bound at kT_max=0.25` warnings with tails up to 1.7e-2 |
| `results/qh9_conjugation_screen_full.log` | The full-DB screen |

**How to read the cap warnings.** `ensemble cap bound at kT_max=... kept N
states, tail weight X exceeds cutoff 1.0e-06` means the *storage cap*, not the
weight cutoff, decided the truncation. The state is still exact for everything
derived from `evals`; the missing Boltzmann weight X is recorded per block in
`truncation_error` with `cap_hit=True`. At kT = 0.1 the production run's tails
land around 1e-4…1e-3. At kT = 0.25 they reach 1.7e-2 — large enough that
uncapped reruns (`--keep-cap 0`) are the right call for hot ensembles.

---

## 8. Orphans — artifacts with no producer

**These are liabilities.** They cannot be regenerated, verified, or trusted
beyond what is written here. If you use one, treat its numbers as
unreproducible until a script exists.

| File | What it appears to be | Missing |
|---|---|---|
| `results/coherence_share_kT0p1.csv` | 1000 rows: `idx, formula, coh_share, coh_max, coh_nonzero, diag_max` joined with `gap_Ha, DoU, largest_pi_atoms, n_aromatic_atoms, n_frontier_within_kT` from the conjugation screen | Still no producer — but **its numbers are now reproduced**, see below |
| `figures/qh9_cas8-8_kT0p1_diagnostics_1000mol.png` | Diagnostics summary for the 1000-molecule set | Same. `scripts/presentation/figures.py::diagnostics` regenerates the same two panels from the run file |

> **Update 2026-08-06 — the coherence-confound numbers are no longer
> unreproducible.** `scripts/presentation/build_cache.py` recomputes them
> directly from the eigenblocks and gets **median off-diagonal share 6.70%**
> and **Spearman 0.787 vs DoU**, matching the logged 6.7 / 0.79 exactly.
>
> It also settles what the orphan CSV actually holds: its `coh_share` column is
> the off-diagonal share of the **248-component Pauli feature vector**
> (median 0.0197%), *not* of the density matrix — a different quantity, three
> orders of magnitude smaller, also reproduced to four significant figures. The
> two were being read as the same number. See
> [`RESEARCH_LOG.md`](RESEARCH_LOG.md) 2026-08-06.

**Action remaining:** a standalone `scripts/coherence_audit.py` with a CLI.
Tracked as [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) Q6.

---

## 3d. HOMO–LUMO gap label audit (2026-08-06)

Why the project's first positive label shows no quantum advantage.
`RESEARCH_LOG.md` 2026-08-06 carries the findings; these are the artifacts.

| Artifact | What it is | Regenerate |
|---|---|---|
| `results/gap_diagnosis.json` | The accuracy ladder (8 feature sets × 25 splits), the regression ladder, accuracy by quintile of \|gap − median\|, the residual test, the Pauli-space screen ratio, `⟨XX⟩ = ⟨YY⟩` check, feature-block norm and variance shares. Tracked, 9 KB | `scripts/gap_diagnosis.py`, ~15 min |
| `results/gap_rho_pass.npz` | Per-molecule `diag(ρ)` (1000 × 4900, float32), diagonal/off-diagonal Frobenius norms, and the class-aggregated `R±` diagonal. 14.9 MB, **gitignored** | `scripts/gap_rho_pass.py` — **~25 min**, one pass over the 45 GB run file |
| `results/gap_diagnosis_data.npz` | Feature + label + state-scalar cache so the follow-ups do not re-read the run file. 1.3 MB, **gitignored** | written on first run of `gap_diagnosis_followup.py` |
| `results/gap_diagnosis_followup.json` | The same ablation under four labels (mean-field gap, CASCI gap, correlation correction, synthetic off-diagonal control), the coherence-stratified split, and conjugated-subset coherence at two temperatures | `scripts/gap_diagnosis_followup.py`, ~10 min |
| `results/gap_diagnosis_ceiling.json` | `diag(ρ)`-as-features baseline (300 PCs) and the **exact ρ-space R± screen ratio, 0.1345** | `scripts/gap_diagnosis_ceiling.py`, ~5 min |
| `results/gap_diagnosis_controls.json` | Coherence features vs equal-count Gaussian noise; what fraction of `‖diag ρ‖²` / `‖offdiag ρ‖²` the weight-≤2 pool sees; global-coherence-vs-residual probes | `scripts/gap_diagnosis_controls.py`, ~5 min |
| `figures/deck_gap/*.png` (4) | The deck's figures, plotted only from the four JSON files above | `scripts/presentation/figures_gap.py` |

> Run order matters: `gap_diagnosis.py` → `gap_rho_pass.py` (independent, and
> the long one) → `_followup` → `_ceiling` (needs the npz) → `_controls`. The
> follow-ups import `gap_diagnosis` as a module, so run them with
> `PYTHONPATH=scripts`.

---

## 3e. Second-quantization labels and the basis audit (2026-08-06)

The OMol25 assessment's evidence base ([`OMOL25_ASSESSMENT.md`](OMOL25_ASSESSMENT.md)).
All of it is derived from the stored `h1eff`/`g` of the existing run file, so
none of it needs QH9, new SCF, or the 45 GB eigenvector blocks.

| Artifact | What it is | Regenerate |
|---|---|---|
| `results/second_quantization_labels.npz` | Per molecule: FCI ground states of the neutral / cation / anion / triplet sectors, their single-determinant references, the correlation corrections to IP / EA / S–T gap, quasiparticle pole strength `Z`, `N_unpaired`, orbital entropy, and the excitation-rank weight profile. Tracked, ~120 KB | `scripts/second_quantization_labels.py --limit 1000 --workers 10`, ~4 min |
| `results/sq_label_screen.json` | The full diagnostic over those ten labels plus the gap reference: ladders, regression, residual tests, magnitudes in eV, mean-field share, composition confounds | `PYTHONPATH=scripts scripts/sq_label_screen.py`, ~25 min |
| `results/localized_basis.npz` | Per molecule and per basis (canonical / full ER / block ER): off-diagonal share of the correlated ground state **and of the zero-correlation reference determinant**, FCI-invariance deviations, and the 248 features in the canonical and block-ER bases. ~2 MB | `scripts/localized_basis_experiment.py --limit 1000 --workers 8`, ~4 min |
| `results/basis_dependence_probe.json` | C₂H₄ torsion scan, 12 angles: SCF energy, S–T gap, natural occupations, and the diagonal/coherence split in three bases with an FCI-invariance gate | `PYTHONPATH=scripts scripts/basis_dependence_probe.py`, ~3 min |
| `results/spin_ladder_pilot.json` | Earlier, coarser version of the same scan. **Superseded** by the probe above, which adds SCF continuation and the invariance gate | `scripts/spin_ladder_pilot.py` |
| `results/multireference_stratification.json` | Ablation by `N_unpaired` quartile, and learning curves for the full pool against the diagonal pool | `PYTHONPATH=scripts scripts/multireference_stratification.py`, ~10 min |
| `figures/omol25_assessment.png` | The three-panel summary: basis control, QH9 vs ethylene on the invariant, and the ten-label ablation | `scripts/plot_omol25_assessment.py` |

> These scripts import `gap_diagnosis` as a module, so run them with
> `PYTHONPATH=scripts`. `basis_dependence_probe.py` imports `spin_ladder_pilot`
> the same way.

---

## 8b. The three decks (2026-08-06)

All generated from `scripts/presentation/`; see that directory's README.
`build_deck.py` takes the deck name as its argument.

### The background talk

| Artifact | What it is | Regenerate |
|---|---|---|
| `Papers/molecular_hamiltonians_and_thermal_states.pptx` | **29 slides.** Conference background lecture: the fermionic algebra and the Slater-Condon rules, active spaces, thermal-state construction ordered by reachable size, representation, Jordan-Wigner, qubit preparation, and applications across chemistry, condensed matter, and quantum information. ~5,000 words of speaker notes | `... build_deck.py theory` |
| `.html`, `.pdf` (same basename) | Pixel-mirror and its Chrome render. Fonts substituted in the PDF | same command, then Chrome |
| `Papers/theory_references.md` | 102 citations grouped by the claim each supports, marking those needed to implement the construction | hand-written |
| `figures/deck_theory/*.png` (11) | Its figures. Two are labelled schematics; the rest are computed from project data or from published active-space estimates | `scripts/presentation/figures_theory.py` |
| `results/theory_cache.npz` | Orbital energies for one molecule (needs `data/QH9Stable.db`), 250 spectra from the run file, and the Pauli decomposition of one active-space Hamiltonian (needs PennyLane, ~30 s). ~0.4 MB | `figures_theory.py cache` |

> This deck declares `lint=True`, so the build rejects em dashes in slide text
> and in speaker notes, and reports contrastive constructions ("not X but Y",
> "rather than") in the slide text for review. A clean build prints nothing.

### The results talk

| Artifact | What it is | Regenerate |
|---|---|---|
| `Papers/thermal_states_presentation.pptx` | **20 slides**, editable text runs, presenter notes on every slide | `... build_deck.py results` |
| `Papers/thermal_states_presentation.html` | Pixel-mirror of the same slide model, for review and for the PDF route | same command |
| `Papers/thermal_states_presentation.pdf` | Chrome render of the HTML. **Fonts are substituted** (no Calibri/Cambria here) — re-export from the `.pptx` for brand-faithful output | `google-chrome --headless --print-to-pdf` on the HTML |
| `Papers/presentation_references.md` | Companion citation list, organised by what each work supports; marks which were needed to get the *code* right | hand-written |
| `figures/deck/*.png` (9) | Every figure in the deck | `scripts/presentation/figures.py` |
| `figures/deck/eq/*.png` (19) | Display equations, mathtext/Computer Modern, cached by content hash | `scripts/presentation/equations.py` |
| `results/presentation_cache.npz` | Per-molecule summaries of the production run (rank, `p₀`, entropy, gap, trace distance, purity, dephased purity, feature-weight shares, descriptors). ~0.4 MB | `scripts/presentation/build_cache.py` — **~28 min**, one full pass over the 45 GB run file |

The predecessor, `Papers/thermal_states_presentation_final.pdf` (2026-07-28,
18 slides), is **kept** — it is the only copy of that version, and its `.pptx`
source never lived in this repo.

### The gap-label diagnosis

| Artifact | What it is | Regenerate |
|---|---|---|
| `Papers/homo_lumo_gap_diagnosis.pptx` | **4 slides**, `lint=True`: the exact diagonal/off-diagonal ablation and its null, regression as the control for binarisation, the redundancy result, and the verdict with the OMol25 assessment. Full speaker notes carry the caveats (pool coverage, the R± screen's blind spot) | `... build_deck.py gap` |
| `.html`, `.pdf` (same basename) | Pixel-mirror and its Chrome render | same command, then Chrome |
| `figures/deck_gap/*.png` (4) | Its figures | `scripts/presentation/figures_gap.py` |

Its inputs are §3d, not `presentation_cache.npz`.

---

## 9. Deleted / retired

| Artifact | Fate |
|---|---|
| `data/groups/qh9_slater_*.h5` — 95 files, 125,013 records, **284 GB** | **Deleted 2026-07-13.** Corrupted by the AO-ordering double transform. Full audit: `data/qh9_raw_sqlite_audit.md`. Regeneration instructions preserved there; only needed if a single-determinant baseline is ever wanted |
| `data/active_space_encode.py`, `data/build_state_vectors.py` | Removed with the retired branch; recoverable from git history (commit `d466c589`) |
| `notebooks/paper/mse.ipynb`, `softplus.ipynb` | Removed in `d466c589` — alternative loss functions, superseded by the log-loss line |
| `docs/bottleneck_analysis.md`, `optimization_roadmap.md`, `pennylane_optimization_summary.md`, `quick_reference_phase1.md` | Consolidated into `docs/classifier_optimization.md` |
