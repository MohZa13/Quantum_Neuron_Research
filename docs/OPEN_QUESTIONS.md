# Open questions — the live research agenda

*Prioritized. Each question states what would settle it, what it costs, and
what it blocks. When one is answered, move the finding to
[`RESEARCH_LOG.md`](RESEARCH_LOG.md) and mark the entry **ANSWERED** with a
link — do not delete it.*

Priority: **P0** blocks the main result · **P1** needed for a defensible claim ·
**P2** improves reach or cost · **P3** housekeeping with real consequences

---

## Q11 — **P1** — Align the training objective with the decision rule

> **Fix #1 built and measured, 2026-08-05.** The hybrid network
> ([`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md), [`qnn/`](../qnn/README.md))
> optimizes an ordinary cross-entropy on a **classical scalar** rather than a
> spectral functional of `H`, so the pathology below cannot occur by
> construction: loss and accuracy are the same objective's two faces again.
> Measured on the positive control, `scripts/train_hybrid_spin.py` — see
> [`RESEARCH_LOG.md`](RESEARCH_LOG.md) 2026-08-05.
>
> **Measured on the identical label, 2026-08-06 — the objective was the whole
> problem.** On the *same* control operator, *same* split, *same* molecules:
> the single neuron scored 66.7% quantum against 66.3% classical, both sitting
> at the 67.0% descriptor baseline — a 0.4-point separation, i.e. none. The
> hybrid network scores **93.0% against 77.0%**, a 16-point separation, and
> beats the single neuron by **+26.3 points**. Nothing about the label or the
> pool changed; only the loss did.
>
> **What remains open** is the last part of the decisive test: 93.0% is not the
> ~100% an exactly-representable label should permit. That residue is
> optimization and conditioning, not the objective — fixes #2 and #3 (whitening
> the pool, subtracting the common mode) are untried, and the training curves
> show the learning rate is too aggressive late (±10-point oscillations with no
> decay). Reprioritized from **P0** to **P1**.

**New 2026-08-05, and now the top blocker — ahead of the label question.**

Measured (`RESEARCH_LOG.md`): on a synthetic label that is purely off-diagonal
and **exactly representable** by the quantum pool, the exact solution scores
100% held-out at Fermi-Dirac loss 3.52, while the optimizer converges to loss
1.37 at 66%. The loss prefers a worse-accuracy operator. The same machine
reaches 93.8–96.4% in the paper's Haar-random setting, so this is a property of
molecular thermal states — highly structured, strongly overlapping, ~93%
diagonal — not of the method.

**Why it blocks everything.** No label, however well designed, can demonstrate
quantum advantage while the objective declines to chase off-diagonal signal.
Q1 is downstream of this.

**Candidate fixes, cheapest first.**
1. **Train the decision rule directly** — logistic loss on `Tr(ρH)`, which *is*
   the neuron's output. Already known to work: plain logistic regression on the
   same features gets 100% train / 91% held-out on the control.
2. **Whiten / rescale the operator pool** so off-diagonal directions are not
   swamped by the common diagonal mode (e.g. normalise each `P_j` by the
   dataset spread of `Tr(ρ P_j)`).
3. **Subtract the common mode**: train on `ρ_m − mean(ρ)` rather than `ρ_m`.
4. Lower the FD temperature `T`, or anneal it, so the loss weights the
   expectation more and the spectrum less.

**Decisive test.** Re-run the positive control. Success = quantum reaches ~100%
and classical stays at chance. Until that passes, no negative result on any
label is fully informative.

---

## Q13 — **P1** — Re-examine the discarded labels now that the class is wider

**New 2026-08-05.** [`QUANTUM_NEURON.md`](QUANTUM_NEURON.md) §5's candidate list
was written for a model that can express exactly one thing: a threshold on a
*linear* functional of ρ. That is why it rules out purity, entropy, negativity,
and every ratio of expectations.

**The hybrid network is not restricted that way** ([`QUANTUM_NEURON.md`](QUANTUM_NEURON.md)
§2.3): its output is an arbitrary classical function of `J₁` linear functionals,
so nonlinear functionals of ρ are expressible whenever they factor through
finitely many of them. Ratios are in range outright. Purity is
`Σⱼ Tr(ρPⱼ)²/K` over a complete basis — exact form is exponentially wide, but a
sub-basis approximation is cheap and is exactly what a classical layer on top of
`J₁` neurons computes.

**Decisive test.** Take §5 candidate 5 — logarithmic negativity across the α|β
cut — which was discarded *only* for being nonlinear. It is a rigorous coherence
witness (identically zero on any diagonal state), which is precisely the property
Q1 wants and which `⟨S²⟩` and `c` failed to have. Compute it per molecule from
the existing eigenblocks, screen it with the R± metric, and train the hybrid net
with the `z_only` ablation alongside.

**Why it might now work where `c` failed.** `c` fails because
`corr(c, D) = 0.919` — coupling tracks unpaired-electron count (Q12). Negativity
is a spectral property of a partial transpose, not a weighted count of
determinants, so it has no reason to inherit that correlation. Whether it
actually escapes it is the measurement.

**Cost.** Low. No pipeline rerun; a contraction over stored eigenblocks plus one
training run, and the register can be the 8-qubit one (~20 min for a full grid).

---

## Q14 — **P0** — Is a different *dataset* the fix, and is OMol25 it?

> **Assessed in full 2026-08-06 → [`OMOL25_ASSESSMENT.md`](OMOL25_ASSESSMENT.md).
> Raised to P0: it now subsumes Q1.** Three measurements moved it.
>
> 1. **Ten second-quantization labels tested on the existing 1000 molecules; all
>    ten fail** (Q − C from −2.37 to +0.13), including labels that are
>    identically zero for any single determinant. "No mean-field analogue" is not
>    a sufficient condition, so the label question alone cannot be the fix.
> 2. **The "diagonal-dominated" result survives the basis attack**, but only
>    relative to a reference determinant ([`INVARIANTS.md`](INVARIANTS.md) I17):
>    full ER localization inflates a *zero-correlation determinant* to 0.935
>    against the correlated state's 0.929.
> 3. **QH9 has zero admissible molecules.** The basis-invariant discriminant
>    `N_unpaired` spans 0.0003–0.4815 across all 1000; **not one exceeds 0.5**,
>    and the median is below *planar* ethylene's 0.194. On twisted ethylene, at
>    `N_unpaired` ≈ 2 the singlet/triplet screen ratio goes 1.06 → 2451 under a
>    physics-preserving rotation.
>
> **So every negative result to date was measured in a regime where the
> hypothesis is untestable in principle.** They are not evidence against it.
> Estimated probability of advantage ≈ **0.10**, with essentially all the risk in
> one condition: that off-diagonal structure is not redundant with `diag(ρ)` *at
> the dataset level* — which has failed three times and has never been tested on
> an admissible system.
>
> **Blocker:** OMol25 does not currently ship Fock matrices (Appendix A.2: "will
> be released in future versions"). Routes and prices in the assessment §1.2.
>
> **The staged decisive test is §9 of the assessment.** Stage 0 is metadata-only
> and free; Stage 2 is the experiment this project has never been able to run.

**New 2026-08-06**, out of the HOMO–LUMO gap audit (`RESEARCH_LOG.md`). That
audit answers the narrow question — the gap label dies because it is a one-body
mean-field observable, and no dataset repairs that — but it sharpens the wider
one, because two of the three attempted repairs moved *towards* harder chemistry
and made the separation worse.

**What is actually needed** is not more correlation but correlation that varies
**at fixed composition**. Measured on QH9: off-diagonal share of ρ vs degree of
unsaturation, Spearman **+0.787**; the same confound that killed `⟨S²⟩`
(`corr(c, D) = 0.919`). QH9 is 130k neutral closed-shell organics of ≤ 9 heavy
atoms and appears to have no such axis.

**Headroom already measured.** The 28 most conjugated molecules carry **12.4%**
off-diagonal weight at kT = 0.1 and **14.8%** at kT = 0.25, against **6.70%**
median for the full 1000. Roughly 2×. Useful, and not by itself decisive.

**OMol25** (Meta FAIR, 2025): >100M calculations at ωB97M-V/def2-TZVPD, ~83M
unique systems, 83 elements, up to 350 atoms, **variable charge and spin**,
four domains including metal complexes. That variable-charge/spin axis is
exactly what QH9 lacks.

Two constraints before anyone plans a campaign:

1. **We need the AO-basis Fock matrix**, which is what lets Phase 1 skip SCF
   entirely (`INVARIANTS.md` I2). OMol25 proper ships energies and forces. The
   usable artifact is **OMol_CSH_58k**, the closed-shell Hamiltonian subset
   (58 elements, systems up to 150 atoms, def2-TZVPD). Note *closed shell*: it excludes the
   open-shell metal complexes carrying the most static correlation, which was
   half the reason to want OMol25. **Verify this before committing** — it is
   read off the dataset cards, not measured here.
2. **Cost does not land where people expect.** The CI dimension is set by the
   active space, not the molecule: CAS(8,8) = 4,900, CAS(10,10) = 63,504,
   CAS(12,12) = 853,776. What grows with OMol25 is (a) the integral transform —
   def2-TZVPD on 100 atoms is thousands of AOs against ~120 for def2-SVP on nine
   heavy atoms — and (b) the active space genuinely *required*: a metal complex
   needs the d shell plus ligand frontier orbitals, so CAS(8,8) is indefensible
   there. Reference points: 4.5–6 min/molecule for `--solver iterative` at
   CAS(10,10), kT = 0.025; the 1000-molecule CAS(8,8) run is 45 GB.

**And a warning from our own data (Q3):** CAS(8,8) already truncates the
*thermal* states of high-DoU QH9 molecules (top-decile median edge slack 0.167).
Moving to more correlated chemistry without widening the window means
representing the interesting molecules worst.

**Decisive test, cheap version.** Before any new pipeline work: take the
conjugated subset we already have (`results/qh9_conjugated_top45.h5`, 28
molecules, kT = 0.1 and 0.25), encode it through `qthermal.encode_run`, and
measure whether the *residual* screen (I16) improves with coherence. If a 2×
increase in off-diagonal weight does not move it, dataset difficulty is not the
lever and Q13 (a label coherence actually determines) is the whole game.

**Blocks.** Any decision to spend compute on a second dataset.

---

## Q12 — **P1** — Find a label where coupling varies at fixed open-shell count

The `⟨S²⟩` / `c` labels failed partly because `corr(c, D) = 0.919`: spin
*coupling* and unpaired-electron *count* move together across QH9, so there is
nothing to separate. Concrete follow-ups, in order:

1. **Residualize `c` against `D`** (not merely against chemical descriptors) and
   label on the sign of the residual. Cheap — both quantities are already in
   `results/spin_labels_kT0p1.npz`. Screen it with the R± metric before training.
2. **Stratify**: within molecules matched on `D` (and on DoU), does `c` still
   vary usefully? If the residual spread collapses to numerical noise, the
   answer is that QH9 at kT = 0.1 simply has no such axis.
3. **Change the regime** rather than the label: the conjugated subset at
   kT = 0.25 (`results/qh9_conjugated_top45.h5`) has much larger coherence.
   Higher kT raises the off-diagonal fraction directly.

**Screening bar.** Aim for `‖offdiag(ΔR)‖/‖diag(ΔR)‖` well above the 0.12–0.16
measured here — note even a *synthetic purely off-diagonal* label only reached
0.34 on this dataset, which is itself a strong signal about how classical these
states are.

---

## Q1 — **P0** — Which label do we train on?

> **Partially answered 2026-08-05.** Candidate #1 (spin coupling, `⟨S²⟩` and
> `c = Tr(ρS²_od)`) was implemented and tested: **quantum − classical = +0.00
> points** on both. See `RESEARCH_LOG.md`. The R± screening metric is now
> implemented (`scripts/train_spin_comparison.py::screening_score`) and it
> predicted the null result before training — use it on every remaining
> candidate. Candidates #2–#5 in `QUANTUM_NEURON.md` §5 are untested, but see
> Q11: the objective must be fixed first.
>
> **Re-confirmed on the hybrid network, and the constraint that shaped the
> candidate list has changed.** The deep model reproduces the null on both
> labels (`⟨S²⟩`: diagonal models *beat* it, 98.0 vs 97.0; `c`: +1.0 point,
> within scatter) while reaching 83.0% against the ablation's 51.0% on a
> synthetic coherence-only label — so the machinery is not what is failing.
> What has changed is that **§5's list was written for a model that can only
> threshold a linear functional of ρ, and the hybrid network is not so
> restricted** (`QUANTUM_NEURON.md` §2.3). Candidate #5 — logarithmic
> negativity — was discarded *only* for nonlinearity and is now the best
> remaining option: **Q13**.
>
> **A third label is now closed, 2026-08-06: the raw HOMO–LUMO gap.**
> Quantum − classical **−1.25 points**; the coherence channel alone is at
> chance (53.0%, regression `R² = 0.000`) while `diag(ρ)` alone explains
> `R² = 0.856`. It fails for a *structural* reason the other two did not — the
> gap is a one-body mean-field eigenvalue difference, defined before correlation
> enters — and its repairs (correlated CASCI gap, correlation correction) fail
> too. Full audit in `RESEARCH_LOG.md`; the dataset half of the question became
> **Q14**. **Net effect here: the screening bar is no longer sufficient on its
> own** — a candidate must clear it *after* residualization against `diag(ρ)`
> ([`INVARIANTS.md`](INVARIANTS.md) I16).

**The project's blocking question.** Everything else is downstream.

The classifier can express exactly one kind of rule: `sign(Tr(ρH))`. We need a
label that is (a) inside that class, (b) determined by coherences and not by
`diag(ρ)`, and (c) not predictable from classical molecular descriptors.

**Decisive test.** Run the R± screening metric
`‖offdiag(R₊−R₋)‖_F / ‖diag(R₊−R₋)‖_F` over all candidates in
[`QUANTUM_NEURON.md`](QUANTUM_NEURON.md) §5 and their residualized variants.
Large numerator + small denominator wins; both tiny means unlearnable by *any*
model in this class.

**Cost.** Hours, not days. `D = R₊ − R₋` is 4,900² ≈ 190 MB; a few seconds to
build from eigenblocks. **No training run required** — this is the highest
information-per-hour step in the project.

**Recommended starting candidate.** Spin coupling, `c_m = Tr(ρ_m S²_od)` —
rigorous (singlet vs triplet is 100% off-diagonal by construction), cheap
(computable from the existing export file), chemically real (thermally
accessible diradical character), and linear in ρ.

**Blocks.** Q2, Q4, and every downstream result.

---

## Q2 — **P0** — Does a Z-only pool actually land at chance?

> **ANSWERED 2026-08-05 — see [Answered](#q2--does-a-z-only-pool-actually-land-at-chance--answered-2026-08-05)
> at the foot of this file.** Kept here in full because the reasoning below is
> still the reason the ablation matters.
>
> Two things happened. First, the claim was **strengthened to a theorem about
> the whole hybrid model class**: a commuting pool makes the forward pass *and
> the gradient* of an arbitrarily deep network a function of `diag(ρ)` alone
> ([`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md) §5.2), asserted bit-identically in
> `tests/qnn/test_pools.py`. Second, it **was** observed at chance on real data:
> the hybrid `z_only` model reaches 51.0% held-out on a synthetic purely
> off-diagonal label, against the quantum pool's 83.0%.
>
> That does not contradict the earlier note that the shallow classical model hit
> 66.3% on *its* control. The two controls are different random operators on
> different registers, and the earlier one was genuinely confounded — its
> classical-descriptor baseline was 67.0%, which the classical pool simply
> matched. The later one's descriptor baseline is 53.3%. The lesson is about the
> control, not the ablation: **always read the ablation against the descriptor
> baseline for the same label**, never against 50%.
>
> Still open on the *physical* labels (`⟨S²⟩`, `c`), where the diagonal model
> ties the quantum one at 99% / 94% — but that is Q1 and Q12, not this question.

The proof that a coherence label is invisible to classical models is a theorem
(Z-only ⇒ diagonal H ⇒ reads only `diag(ρ)`). **But the theorem must be
demonstrated in the actual code path**, or a reviewer is entitled to doubt the
implementation matches the claim.

**Decisive test.** Train twice on the chosen label — full pool, then a pool
restricted to Z-strings. The second must sit at chance (50% on a median-split
balanced label). Any accuracy above chance means a bug: leaked diagonal
information, an unintended non-Z term, or label leakage through preprocessing.

**Cost.** One extra training run. Same code path, one argument different.

**Blocked by.** Q1.

---

## Q3 — **P1** — Is CAS(8,8) big enough for the *thermal* states?

**Measured** (`RESEARCH_LOG.md` ~2026-07-30): ground states are bracketed fine
(T = 0 edge slack < 0.05), but the kT = 0.1 thermal states of high-DoU
molecules are **not** — high-DoU decile median edge slack **0.167**, 92% above
0.1. The excitable π* ceiling is real.

**Why it matters now.** If the winning label depends on states pressed against
the active-space boundary, the numbers are an artifact of window size, not
chemistry. This is the most likely way a good-looking result turns out to be
wrong.

**Decisive test.** Rerun a stratified subset (say 30 molecules spanning the DoU
range) at CAS(8,10) and CAS(8,12); recompute the label and check that both the
value and the ranking are stable.

**Cost.** CAS(8,10) is dim 63,504 — beyond dense reach, so use
`--solver iterative` (measured 4.5–6 min/molecule at kT = 0.025). At kT = 0.1
the Krylov window is wider and escalation may cap out; budget accordingly and
watch `cap_hit`.

---

## Q4 — **P1** — Can a classical descriptor model predict the chosen label?

Distinguish two claims carefully (`QUANTUM_NEURON.md` §7):

- **A.** "No diagonal-only quantum model can learn this" — proven by
  construction, free.
- **B.** "No classical model can learn this" — **not implied by A**, because
  chemistry correlates coherence with composition.

**Decisive test.** Fit gradient-boosted trees / logistic regression on the
classical features we already have (`DoU`, `gap_Ha`, `n_aromatic_atoms`,
`largest_pi_atoms`, `n_frontier_within_kT`, atom counts, occupation numbers).
If it matches the quantum model, residualize the label against those features
and rerun (`QUANTUM_NEURON.md` §7), then **verify the residual's dynamic range
still exceeds the per-block `truncation_error`** — otherwise you are training
on numerical noise.

**Blocked by.** Q1.

---

## Q5 — **P2** — Is the Gaussian-reference trace distance measuring interaction, or compactness?

The non-interacting reference omits mean-field electron repulsion, which
inflates the trace distance for compact molecules. The clean-looking family
split (saturated skeletons 0.34–0.51, N/O π-systems 0.93–0.99;
`RESEARCH_LOG.md` 2026-07-15) may be partly that artifact.

**Decisive test.** Build a mean-field-corrected reference — one-body `h1eff`
plus the Hartree/exchange potential of the reference density — and recompute.
If the family split survives, it is physics; if it collapses, the current
"quantumness audit" is partly measuring molecular size.

**Why it matters.** `tracedist_gaussian` was floated as a classification
target. It should not be used as one until this is settled.

---

## Q6 — **P3** — Reproduce the two orphan artifacts

> **Largely answered 2026-08-06.** `scripts/presentation/build_cache.py`
> recomputes both from the run file. The density-matrix off-diagonal share is
> **6.70% median, Spearman 0.787 vs DoU** — matching the logged 6.7 / 0.79
> exactly, so the coherence-confound finding is no longer resting on an
> unreproducible number. The diagnostics figure is regenerated by
> `scripts/presentation/figures.py::diagnostics`.
>
> **And it found a real conflation:** the orphan CSV's `coh_share` is the
> off-diagonal share of the 248-component *Pauli feature vector* (median
> 0.0197%), not of ρ. Two quantities, three orders of magnitude apart, were
> being read under one name. See [`RESEARCH_LOG.md`](RESEARCH_LOG.md)
> 2026-08-06.

**What remains** is packaging, not verification: a standalone
`scripts/coherence_audit.py` with a CLI (per-molecule off-diagonal share, max
coherence, nonzero count, joined against the conjugation screen), and a
`scripts/plot_run_diagnostics.py`. Both are ~30 lines lifted out of
`build_cache.py`, which currently does the work as a side effect of building
the deck — a producer script whose *purpose* is something else is still a
liability.

**Cost.** Low. Both are straightforward contractions over the eigenblocks.

---

## Q7 — **P3** — Correct the superseded MPS-ordering claims

Measurement showed **blocked** gives ~2× smaller χ, overturning the
interleaved-for-MPS hypothesis. Two places still assert the old claim:

- `qthermal/README.md`, Module I row: *"remains the right layout for future MPS
  backends"*
- `benchmarks/mps_bond_dimensions.py` docstring: *"Interleaved is expected to
  win"*

Not corrected here because they sit inside technical prose that the pipeline
owner should review. Interim guidance lives in [`INVARIANTS.md`](INVARIANTS.md) I12.

---

## Q8 — **P2** — Warm-started Krylov escalation

`IterativeWindowSolver` doubles its root count until the counting bound
certifies the cutoff, and **each escalation re-solves from scratch**. Known
waste ≈ 40% when escalations dominate (`qthermal/README.md`, Performance).

**Action.** Seed Davidson from the previous escalation's converged roots.
Worth doing before any large ncas = 10/12 campaign (i.e. before Q3 at scale).

---

## Q9 — **P2** — A hot-ensemble backend (TPQ / METTS)

The Krylov certificate makes it a **low-kT** tool by design: once the thermal
window holds thousands of states (kT = 0.25 at large ncas) escalation caps out.
The `SpectralSolver` protocol in `diagonalize.py` is the seam — a sampling
backend slots in without touching anything upstream.

**Trigger.** Needed when we want hot ensembles at active spaces beyond dense
reach. Not on the critical path for Q1.

---

## Q10 — **P3** — Land the uncommitted work

Currently uncommitted: the `--indices` targeted-subset feature
(`qthermal/loader.py`, `qthermal/run.py`, plus tests including a
SQLite-variable-limit chunking test), and the move of `notebook_test_utils.py`
to the repo root with `tests/conftest.py`'s `sys.path` shim deleted.

Both are complete and tested (156 collected, `pyproject.toml` already lists
`py-modules = ["notebook_test_utils"]`). They should not sit uncommitted —
`--indices` is what makes the conjugation screen actionable.

---

## Answered

### Q2 — Does a Z-only pool actually land at chance? — **ANSWERED 2026-08-05**

**Yes, and it is now a theorem rather than a measurement.**

[`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md) §5.2 proves that for a mutually
commuting pool, both the forward pass **and the gradient** of an arbitrarily
deep hybrid network depend on ρ only through `diag(ρ)`. Off-diagonal entries
appear nowhere in either expression, so no depth of classical layers can
recover coherence a commuting pool never admitted.

Demonstrated in the code path that runs it, three ways:

- `tests/qnn/test_pools.py::test_commuting_pool_sees_only_the_diagonal` — delete
  every off-diagonal entry of every state; a `z_only` network's outputs and *all*
  gradients are bit-identical, for three activations at depth 3. Its companion
  asserts the quantum pool's output *does* change, so the first is not vacuous.
- On real data: `scripts/train_hybrid_spin.py`, 1000 molecules, 600 epochs,
  synthetic purely-off-diagonal label → `z_only` **51.0%** held-out (chance, and
  below that label's own descriptor baseline of 53.3%) against the quantum
  pool's **83.0%**. `results/hybrid_spin_metrics_8q.json`.
- The single-neuron harness reaches the same conclusion independently
  (`scripts/train_spin_comparison.py`).

The original question asked for this on *the chosen label*, which is still open
(Q1) — but the instrument is now verified, and the ablation is stronger than
originally scoped: it holds for the whole hybrid model class, not one neuron.

→ [`RESEARCH_LOG.md`](RESEARCH_LOG.md) 2026-08-05, [`INVARIANTS.md`](INVARIANTS.md) I15
