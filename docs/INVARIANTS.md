# Invariants — the do-not-break list

*Each entry: the rule, why it exists, the evidence, how it fails, and a command
that verifies it. These are not style preferences. Every one of them either
already cost the project something or is guarding a silent-corruption path.*

Silent failures dominate this list. That is the theme: in numerical quantum
chemistry, the dangerous bugs do not raise — they return plausible numbers.

---

## I1 — Never AO-reorder raw `QH9Stable.db` Hamiltonians

**Rule.** The `Ham` blobs in the raw SQLite database are **already in PySCF
def2-SVP AO order**. Apply no permutation. The QHBench
`PYSCF_DEF2SVP_CONVENTION` transform helpers in `qthermal/loader.py` exist
**only** for QHBench *processed* / model-output matrices.

**Why.** Applying the reorder double-transforms the matrix. It does not crash;
it produces a physically plausible-looking spectrum that is wrong.

**Evidence** (`data/qh9_raw_sqlite_audit.md`, 2026-07-10):

- Untransformed blobs agree with freshly converged B3LYP/def2-SVP to
  **≤ 3.7×10⁻³ Ha** across the full spectrum on records 0–4.
- Transformed blobs show **~1 Ha spurious core shifts** on compact molecules —
  *inside* the loose physicality windows, so unit detection passed anyway — and
  catastrophic intruder eigenvalues on linear ones (HCN −57.7 Ha, C₂H₂ −173.7 Ha;
  true 1s levels are −14.4 / −10.2 Ha).
- 9/9 sampled records from the derived tree reproduced the double transform
  *exactly* — mechanism and provenance certain.

**Cost when broken.** 284 GB / 125,013 records corrupted; deleted 2026-07-13.
HOMO error median 0.74 eV (max 3.8), gap error median 0.92 eV (max 3.0), and
binary gap labels flipped near threshold — including the stored threshold
itself.

**Verify.**
```bash
grep -n "qh9_ham_to_pyscf\|qh9_to_pyscf_transform" qthermal/loader.py
# _record_from_row must NOT call either. Only the module-level helpers define them.
```

---

## I2 — Never run SCF; CASCI runs on the stored Kohn–Sham orbitals

**Rule.** Phase 1 builds the active-space Hamiltonian on QH9's stored
B3LYP/def2-SVP orbitals. **Never call `mf.kernel()`.**
`hamiltonian.make_injected_rhf` deliberately builds a non-iterated RHF wrapper
whose only job is to carry injected `mo_coeff`.

**Why.** The dataset's value is that it is faithful to QH9's own electronic
structure. Re-converging orbitals would make every record a different physical
object from the source dataset, silently. It is a deliberate choice, not an
oversight — do not "fix" it.

**Trade-off, stated honestly.** CASCI on KS orbitals is not variationally
optimal. That is accepted and documented (`qthermal/README.md`).

**Verify.**
```bash
grep -rn "\.kernel()" qthermal/ | grep -v "cis.kernel\|fci"
# Only the FCI solver's kernel may appear. No mf.kernel().
```

---

## I3 — Never materialize a dense density matrix in the pipeline

**Rule.** State flows as `TruncatedEnsemble` / `ThermalBlock` dataclasses —
weights `p` plus eigenvectors `civecs`, i.e. `ρ = Vᵀ diag(p) V`. Anything that
builds a `dim × dim` or `2^(2·ncas)` matrix inside the pipeline is a bug.

**Scale.** CAS(8,8): the determinant sector is 4,900² and the qubit register
is 65,536² ≈ 4×10⁹ complex entries ≈ 64 GB. Per molecule.

**Where the discipline is enforced.**
- `thermal.trace_distance_projected` works in the joint span of two eigenblocks
  (rank ≤ m₁+m₂), never `dim × dim`.
- `encode.extended_heisenberg_expectations` contracts in the determinant basis
  via alpha/beta Gram matrices — 248 features in under a second.
- `mps.purification_mps` is bounded by `m · 2^Q`, never `dim²`.

**Sanctioned exceptions**, all validation-only and documented as such:
`mps.to_dense_ket`, `mps.reduced_density_matrix`, `encode.encode_jw`,
`encode.thermal_density_matrix`.

**Verify.**
```bash
grep -rn "np.zeros((dim, dim)\|np.outer\|einsum(\"i,ip,iq->pq" qthermal/
# Any hit outside encode.py/mps.py validation helpers needs justification.
```

---

## I4 — MO signs are canonicalized; do not remove

**Rule.** `orbitals._canonicalize_signs` flips each MO column so its
largest-magnitude AO coefficient is positive, applied uniformly whether `C` is
provided or recovered, **before** the orthonormality check.

**Why.** MO coefficients are defined only up to a per-column sign. Without
canonicalization, `h1eff`, `g`, and `civecs` carry arbitrary sign flips between
otherwise-identical molecules — pure noise to any model trained on the raw
tensors, and not a physical effect. It also makes sign-structure labels (see
[`QUANTUM_NEURON.md`](QUANTUM_NEURON.md) §5) well-defined at all.

**Verify.** `.venv/bin/python -m pytest tests/qthermal/test_orbitals.py -q`

---

## I5 — Readers must not assume `evals` exists

**Rule.** The full sector spectrum is written **only** by the dense solver.
`IterativeWindowSolver` returns `evals_full = None` by contract.

**Why.** The three solvers have genuinely different contracts, and the seam
(`SpectralSolver`, a `Protocol` in `diagonalize.py`) exists precisely so
tensor-network backends can slot in later without a full spectrum.

| Solver | `evals` | Tail bound | Notes |
|---|---|---|---|
| `DenseEDSolver` | yes | exact | dim guardrails: silent ≤5,000; `--allow-large-dense` + RAM check ≤70,000; refuses beyond |
| `IterativeWindowSolver` | **no** | rigorous counting bound | low-kT backend; caps out when the thermal window holds thousands of states. `--kT-relative` requires dense |
| `NonInteractingSolver` | yes | exact | g = 0 audit reference only; raises if `g is not None` |

**Downstream consequence.** With `evals`, per-kT weights are exact and the tail
is exact. Without it, weights normalize over the kept window and the certified
`tail_weight` folds into `truncation_error` (`thermal._weights_for_kT`).

**Verify.** `.venv/bin/python -m pytest tests/qthermal/test_diagonalize.py -q`

---

## I6 — Never reorder, insert, or delete notebook cells

**Rule.** `notebook_test_utils.load_notebook_namespace` executes cells **by
index**: cells `(0,1,2,3)` of `notebooks/paper/logloss.ipynb` and `(1..7)` of
`notebooks/pennylane/logloss_pennylane.ipynb`.

**Why.** Editing cell structure breaks the equivalence tests in a way that
looks like a *numerical* failure — you will chase a physics bug that is
actually an off-by-one in cell indexing.

**If you must restructure**, update the index tuples in
`notebook_test_utils.py` in the same commit, and say so in the commit message.

**Verify.** `.venv/bin/python -m pytest tests/test_notebook_equivalence.py -q`

---

## I7 — Do not "restore" the notebook's `1/T` gradient factor

**Rule.** `notebooks/paper/logloss.ipynb`'s `dfj` divides the gradient by `T`,
an extraneous factor relative to Theorem 5 / Eq. (63). Optimized
implementations use `dH/dω_j = H_j`; `figures/quantum_training_impls.py::run_original`
also applies the fix so comparisons are matched.

**Why.** It is tempting to reintroduce the factor to make old numbers agree.
Don't — the corrected version is the correct one, and the loss was never
affected (only the training step).

**Verify.** `test_notebook_equivalence.py` checks analytic gradients against
central finite differences.

---

## I8 — `g` is stored full `ncas⁴`, chemist notation, not s8-packed

**Rule.** Two-electron integrals go to disk as the complete `(ncas,)*4` tensor
in chemist notation `(pq|rs)`, via `ao2mo.restore(1, ...)`.

**Why.** ~32 KB at defaults; gzip absorbs the 8-fold redundancy; readers need
no unpacking step. Physicist-notation conversion is a Phase-2 concern and does
not happen in the pipeline.

**Verify.** `hamiltonian.assert_g_8fold_symmetry` runs on every build.

---

## I9 — Resume safety: `complete=True` is written last

**Rule.** `io_hdf5.RunWriter.write_molecule` flushes all data, *then* sets
`complete=True`, then flushes again. On rerun, complete groups are skipped;
incomplete ones are **deleted and rewritten**. `encode_run` mirrors this.

**Why.** A crashed run must never leave a half-written group that a reader
mistakes for real data. This is what makes 13-hour runs safe to interrupt.

**Verify.** `.venv/bin/python -m pytest tests/qthermal/test_io_hdf5.py -q`

---

## I10 — Truncation is always recorded, never silently dropped

**Rule.** Every truncation records exactly what it discarded:
`truncation_error` per block, `cap_hit` when the storage cap (not the weight
cutoff) bound, and `tracedist_bound` for the audit's additive error.

**Why.** Every downstream number carries its own error bar. A dataset that
quietly loses 1.7% of its thermal weight (which the kT = 0.25 runs do) is
worthless if the loss is not written down.

**Reading a run**: check `cap_hit` and `truncation_error` before trusting a
block. See [`DATA_CATALOG.md`](DATA_CATALOG.md) §7.

---

## I11 — Nothing hardcodes active-space dimensions

**Rule.** Every dimension derives from the `ActiveSpace` object: `ncas`,
`ncore`, `nelecas`, `na_strings`, `nb_strings`, `dim`. No literal `8`, `(4,4)`,
`4900`, or `65536` in pipeline code.

**Why.** The pipeline is genuinely parametric — proven by tests running at both
ncas = 6 and ncas = 8, and by production runs at ncas = 10 and 12.

**Verify.**
```bash
grep -rn "4900\|65536\|ncas=8\|ncas = 8" qthermal/ | grep -v "docstring\|#"
```

---

## I12 — Wire ordering is a per-consumer choice; record it

**Rule.** `blocked` and `interleaved` JW layouts are **not** interchangeable.
Whichever you use, it is written into the output file's `/meta` — check it
before combining files. `encode_run` refuses to mix conventions in one file.

**Current best knowledge** (see `RESEARCH_LOG.md`):

| Consumer | Use | Why |
|---|---|---|
| Extended-Heisenberg features | **blocked** | ~10× more connected-ZZ signal, and the only nonzero XX/YY under a nearest-neighbour ansatz (interleaved's adjacent pairs are all spin-flip ⇒ XX/YY vanish by S_z conservation) |
| Purification MPS | **blocked** | Measured ~2× smaller χ. *This overturned the original hypothesis* — see `RESEARCH_LOG.md` 2026-07-27 |
| Coherence-label operators (`S²_od`) | **interleaved** | Same-orbital α/β wires adjacent ⇒ JW strings cancel ⇒ the spin-exchange operator is 4-local and string-free |

> ⚠️ `qthermal/README.md` (Module I row) and `benchmarks/mps_bond_dimensions.py`
> (docstring) both still assert interleaved is right for MPS. **That claim is
> superseded.** Tracked in [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) Q7.

---

## I13 — Always use the in-tree interpreter

**Rule.** `.venv/bin/python`, never bare `python`. `qthermal` is installed
editable, so imports work from any working directory — **do not add `sys.path`
manipulation to new files** (that pattern was removed in the current
uncommitted diff).

`MPLCONFIGDIR=/tmp/matplotlib` prefix for anything importing matplotlib.

After moving or adding modules:
`.venv/bin/pip install -e . --no-build-isolation`

---

## I14 — Normalize projected thermal states before training on them

**Rule.** `qnn.StateBatch(..., normalise=True)` — the default. Any other path
that feeds thermal states to a classifier must divide each ρ by its trace.

**Why.** A truncated thermal block has `Tr ρ = 1 − truncation_error`, not 1. On
the production 1000-molecule projection the traces run **0.967 to 1.000**
(median 0.998; `results/spin_labels_kT0p1.npz → rho_trace_kept`). Every pool in
this project contains the identity, so `Tr(ρI) = Tr ρ` is a *feature the model
can read*, and truncation error correlates with molecular complexity — the same
thing most candidate labels correlate with. The model would be reading how hard
the molecule was to converge.

**How it fails.** Silently, and in the most flattering direction: accuracy goes
*up*. Nothing raises, and the artifact looks like signal.

**Verify.**
```bash
grep -n "normalise" qnn/states.py scripts/train_hybrid_spin.py
# StateBatch must default normalise=True; callers must not pass False for real data.
.venv/bin/python -m pytest tests/qnn/test_states.py -q
```

---

## I15 — A commuting pool is a theorem, not a baseline. Do not "improve" it

**Rule.** `qnn.build_pool(n, "z_only")` and `"diagonal_full"` must stay strictly
diagonal. Do not add off-diagonal terms to make the ablation "fairer", and do
not delete the `U is None` fast path in `quantum_layer._spectra` without keeping
`tests/qnn/test_gradients.py::test_commuting_fast_path_matches_dense_construction`.

**Why.** [`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md) §5.2 proves that for a
mutually commuting pool, both the forward pass **and the gradient** of an
arbitrarily deep hybrid network depend on ρ only through `diag(ρ)`. That is what
makes the ablation a statement about the model class rather than an empirical
comparison someone can attack as undertuned — the single strongest thing this
project has. Adding one XX term destroys it and nothing will fail.

**Evidence.** `tests/qnn/test_pools.py::test_commuting_pool_sees_only_the_diagonal`
deletes every off-diagonal entry of every state and asserts the outputs and *all*
gradients are bit-identical, for three activations at depth 3. Its companion
asserts the quantum pool's output *does* change — without which the first test
would pass for a trivial reason.

**Measured** (2026-08-05, 8-qubit register, 1000 molecules, 600 epochs;
`results/hybrid_spin_metrics_8q.json`): on a synthetic purely off-diagonal
label, `z_only` sits at **51.0%** held-out — chance, and below that label's own
classical-descriptor baseline of 53.3% — while the same network with XX/YY added
reaches **83.0%**. The unconstrained diagonal pool, with 4× the parameters,
manages 55.3%.

**Verify.** `.venv/bin/python -m pytest tests/qnn/test_pools.py -q`

---

## I16 — Screen the label's *residual*, not the raw label

**Rule.** Before quoting `‖offdiag(ΔR)‖_F / ‖diag(ΔR)‖_F` as evidence that a
candidate label is or is not learnable by a coherence-reading model, compute it
on the label **residualized against an out-of-fold model of `diag(ρ)`**. And
never compare a ρ-space screen ratio against a Pauli-feature-space one: they
differ by more than an order of magnitude for the same label.

**Why.** The screen asks whether the two classes' aggregated states differ
off-diagonally. It does *not* ask whether that difference carries information
the diagonal has not already supplied. In molecular data those are routinely
different questions, because chemistry drives coherence and populations
together.

**Evidence** (2026-08-06, `scripts/gap_rho_pass.py` + `gap_diagnosis*.py`).
The median-split HOMO–LUMO gap screens at **0.1345** on the true
`R± = Σ_m ρ_m` over 1000 molecules — between `⟨S²⟩` (0.122) and `c` (0.162),
so the screen ranks it as no worse than labels it was used to reject. Its
coherence channel nevertheless sits at chance (53.0%, AUC 0.555, regression
`R² = 0.000`), while a synthetic off-diagonal label on the same states reaches
+42.0 points. The diagnostic that *does* separate them is the residual: the
exact off-diagonal share of ρ correlates −0.571 with the gap and **+0.040**
with the gap residual after a diagonal model.

Same label in the 248-term Pauli basis screens at **0.0069**, ~20× smaller than
the ρ-space number. This is the same class of error as the `coh_share`
conflation (`RESEARCH_LOG.md` 2026-08-06): one name, two quantities, different
orders of magnitude.

**Verify.**
```bash
PYTHONPATH=scripts .venv/bin/python scripts/gap_diagnosis_ceiling.py   # screen_ratio_rho
PYTHONPATH=scripts .venv/bin/python scripts/gap_diagnosis_controls.py  # residual probes
```

---

## I17 — Coherence is measured relative to a reference determinant. Carry the zero-correlation control

**Rule.** Any statement of the form "this state is X% off-diagonal" is
meaningless without naming the orbital basis, and must be accompanied by the
**same measurement applied to a single Slater determinant** carried through the
same rotation. Only rotations that preserve the reference determinant
(occupied-occupied and virtual-virtual) may be used to compare coherence across
molecules. Never quote a number obtained under an occupied-virtual rotation
without the control.

**Why.** The diagonal/off-diagonal split of ρ in the determinant basis is not
invariant. An orthogonal rotation inside the active space leaves every energy,
every eigenvalue of ρ, and every observable untouched while moving weight
arbitrarily between the two channels. Without a control it is trivial to
manufacture a spectacular and completely empty result.

**Evidence** (2026-08-06, `scripts/localized_basis_experiment.py`, 1000
molecules, FCI ground states, FCI invariance verified to 2.6e-12 Ha):

| basis | correlated ground state | zero-correlation control |
|---|---|---|
| canonical | 0.0582 | 0.0000 |
| full Edmiston-Ruedenberg | 0.9293 | **0.9348** |
| block ER (reference-preserving) | 0.0580 | 0.0000 |

Full ER makes the correlated state look 93% coherent — and makes a state with no
correlation at all look **93.5%** coherent. Under the reference-preserving
subgroup the share moves by a median of **5.3e-5**.

**Corollary, and the useful part.** A molecule offers genuine coherence exactly
insofar as it has no dominant reference determinant. The basis-invariant
discriminant is the natural-occupation spectrum, summarised by
`N_unpaired = Σᵢ min(nᵢ, 2−nᵢ)`. Report it alongside any coherence claim. QH9 at
CAS(8,8) spans 0.0003-0.4815 (median 0.113, **zero molecules above 0.5**); a
single ethylene torsion spans 0.194-1.997.

**Verify.**
```bash
.venv/bin/python scripts/localized_basis_experiment.py --limit 200 --workers 8
PYTHONPATH=scripts .venv/bin/python scripts/basis_dependence_probe.py
```

→ [`RESEARCH_LOG.md`](RESEARCH_LOG.md) 2026-08-06,
[`OMOL25_ASSESSMENT.md`](OMOL25_ASSESSMENT.md) §2

---

## Adding an invariant

Only add one when it is *load-bearing* — when breaking it causes silent wrong
answers or expensive rework. Include: the rule, the mechanism of failure, the
evidence (with date), and a verification command. Then log it in
`RESEARCH_LOG.md` and reference it from `CLAUDE.md` if it belongs in the
always-loaded set.
