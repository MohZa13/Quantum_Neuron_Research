# Research log

**Append-only, newest first.** This is the project's memory across context
resets. A finding belongs here the moment it is established — including
negative results, which are often the expensive ones.

**Format.** One entry per finding. Date, one-line claim in bold, then: what was
measured, what it means, what it changes. Cite the artifact or command that
produced it. If a later entry supersedes an earlier one, edit the earlier entry
to say so — never delete it.

Template: [`templates/FINDING.md`](templates/FINDING.md).

---

## 2026-08-09 — **Isomer identity is a coherence-dominated label at scale** (`scripts/pair_screen.py`, 450 pairs)

The ncas = 10 anecdote (C2H2-vs-HCN screen ratio 1.15) tested on the
1000-molecule CAS(8,8) production set: for pairs (A, B) at matched kT = 0.1,
`ratio = ||offdiag(rho_A - rho_B)|| / ||diag(rho_A - rho_B)||` on the shared
1024-determinant register (`spin_labels_kT0p1.npz` `keep_idx`), 150 pairs per
class:

| pair class | med \|offdiag\| | med \|diag\| | med ratio | frac >= 1 |
|---|---|---|---|---|
| **isomer** (same formula) | 0.142 | **0.046** | **2.76** | **87%** |
| isoelectronic (same N, diff formula) | 0.146 | 0.105 | 1.32 | 61% |
| control (N differs by >= 4) | 0.157 | 0.152 | 1.06 | 53% |

Reference points on this data family: S^2 label screens at 0.13, coherence-only
c at 0.18, the synthetic pure-off-diagonal control at 0.81 (8q register).

**The structure is exactly what the coherence program wants.** The off-diagonal
separation is class-independent (~0.15 for every pair type) while the diagonal
separation collapses as composition is matched — for isomers the diagonal
channel is 3x weaker than the off-diagonal one. Since isomers share every
formula-derived descriptor (DoU included), whatever separates their thermal
states is not composition wearing a disguise: **"which isomer is this thermal
state from" is a label whose signal is predominantly coherence, by measurement
rather than by construction.** This answers the screening half of
OPEN_QUESTIONS Q1's demand for a physical, composition-orthogonal label
family; learnability-with-generalization is the remaining half.

Caveats: (i) ratios are register-dependent (1024 shared determinants); (ii)
one pathological isomer pair (QH9 907 vs 911, near-identical states, ratio
6.5e4 with |offdiag| only 0.03) shows the ratio alone can be inflated when both
norms vanish — medians and the absolute norms table are the honest summary;
(iii) a high screen does not guarantee learnability (the single-neuron program
already showed a model can fail to exploit an existing exact solution).

**Learnability check, negative under the extreme protocol**
(`scripts/pair_transfer_top45.py`): on the 28-molecule x 2-kT conjugated set,
training on a pair's two states at ONE temperature and testing at the other
gives chance for both pools (quantum 54.4%, z_only 53.8%, 160 decisions;
uniform mediocrity across pairs, not bimodal). Two confounds: the shared
register kept only 85.8% of the population for these spread-out states, and
one-example-per-class training is the hardest possible protocol. Read together
with the ncas = 10 result (molecule ID at 9/12 with SIX temperatures per
molecule in training), the operational conclusion is that this label family
needs multi-temperature ladders per molecule — which is what the Module K
generation plan produces — not single-temperature snapshots.

---

## 2026-08-08 — **The bridge closes: qnn trains on MPS-produced CAS(10,10) thermal states** (`scripts/train_mps_thermal.py`)

First contact between the repo's two halves: the `results/qh9_mps_ncas10.h5`
states (Module K, sector 63,504 — beyond every dense route) fed to the
`qnn` hybrid network as density matrices. **It trains.**

- **Native register, K = 1024:** 4.1 s/epoch, loss 0.73 -> 0.29 in 8 epochs.
  The quantum layer's O(K^3) eigendecompositions are fine at this size.
- **Projected register (K = 256, exactly lossless here):** only C(10,5) = 252
  of the 1024 rows are populated — N_alpha = 5 is sharp — so `project_register`
  keeps 100.0000% of population *and* off-diagonal weight. On it:
  - *hotcold* (kT >= 1 vs < 1): LOO **5/6** both pools; train loss
    0.69 -> 6e-4. The one miss is kT = 0.5, the boundary point, called hot by
    a model trained with no negative example warmer than 0.25 — a labeling
    artifact, not a machinery failure.
  - *kT regression* (standardized log10 kT, squared loss): LOO MAE **0.14 sigma**
    (quantum) / 0.11 (z_only) — predictions track the ladder monotonically
    with mild shrinkage at the extrapolation edges.

**Conventions were the real risk, and are now pinned by tests**
(`tests/test_mps_bridge.py`, 7 tests + the 188-test qnn suite green): the h5
register is wire-0-MSB (Module I), qnn pools are little-endian, and the loader
bit-reverses. The razor: the coldest state's dominant determinant must land at
row 0b0000011111 = **31** (HF string, orbitals 0-4 occupied) — a loader that
forgot the reversal puts it at 992 and cannot pass.

**The two-molecule run** (mol_4 = HCN generated overnight: beta0 err 2.1e-14,
DMRG-vs-Krylov 6.3e-9; note HCN saturates chi = 256 already at kT = 1 where
C2H2 needed only 64 — a materially more entangled thermal state). All 12
states, LOO:

| task | quantum | z_only | R+- screen ratio |
|---|---|---|---|
| hotcold | **12/12** | 12/12 | 0.09 |
| molecule (C2H2 vs HCN, matched kT) | **9/12** | 7/12 | **1.15** |
| kT regression (MAE, sigma units) | **0.047** | 0.068 | — |

**What this does and does not show.** The bridge works end-to-end at
production size: unit-trace-by-construction states (no truncation-trace leak
for I8 to worry about), native-K = 1024 training at 2.8 s/epoch, and the
temperature tasks solved essentially perfectly by both pools — as expected,
since Boltzmann weights are diagonal data.

The *molecule* label is the one to remember. Its class-difference aggregate
R+ - R- carries MORE off-diagonal than diagonal Frobenius weight (ratio 1.15,
vs 0.09 for the temperature label) — the discriminating structure between two
isoelectronic molecules at matched temperature lives substantially in the
coherences — and the quantum pool beats the diagonal ablation 9/12 vs 7/12,
with both models' misses concentrated at the uninformative extremes (kT = 4,
both states near P/dim; kT = 0.1, both near-pure). Twelve samples is far too
few for the 2-fold gap to be evidence of anything; the sample-free screen
ratio is the observation worth keeping, and it suggests "molecular identity
at matched temperature" as a label family for the coherence program —
cheap to generate at scale with Module K, and not obviously a composition
proxy since the compared molecules are chosen isoelectronic. Logged against
OPEN_QUESTIONS Q1.

---

## 2026-08-08 — **Production run at ncas = 10 succeeds: CAS(10,10) thermal states as MPS, validated at both ends** (`results/qh9_mps_ncas10.h5`)

The run Module K was built for. QH9 `mol_3` (acetylene, C2H2), CAS(10,10), CI
sector **63,504** — two orders of magnitude past the dense `eigh` guardrail and
past Module J's `2^Q` scatter. Full kT ladder in one pass, 88 minutes on 8
threads.

**Bracketed by two external references, because no spectrum exists at this
size:**

| end | quantity | this run | reference | error |
|---|---|---|---|---|
| hot (beta = 0) | `<Psi(0)|H|Psi(0)>` | -21.663909237992 | -21.663909237992 (closed form) | **1.1e-14** |
| cold | ground state | -25.2277762909 (DMRG) | -25.2277763085 (Python Krylov) | **1.8e-08** |

The hot end uses a new closed form for `Tr[H P]/dim`
(`QThermalMPS.sector_mean_energy`): under a uniform average over sector
determinants only the direct (`p=q, r=s`) and exchange (`p=s, r=q`, equal spin,
sign flipped) two-body terms have a diagonal, so the beta = 0 energy is an
`O(ncas^2)` formula. Verified to 1e-14 against `mean(evals)` on CAS(8,6) and
CAS(8,8). **This is the only exact check that survives past dense reach**, and
it validates the MPO and the beta = 0 purification simultaneously.

The cold end is two different codebases and two different algorithms — Julia
ITensor MPS-DMRG against PySCF FCI-Davidson — agreeing to 1.8e-8 Ha.

**The ladder** (maxdim 256, cutoff 1e-8, graded dbeta from 0.05):

| kT | beta | chi | E | S | log Z | wall |
|---|---|---|---|---|---|---|
| 4.00 | 0.25 | 19 | -21.94845447 | 11.023741 | 16.5109 | 104 s |
| 2.00 | 0.50 | 38 | -22.21847859 | 10.922979 | 22.0322 | 96 s |
| 1.00 | 1.00 | 64 | -22.70637369 | 10.561502 | 33.2679 | 329 s |
| 0.50 | 2.00 | **246** | -23.44480442 | 9.477765 | 56.3674 | 799 s |
| 0.25 | 4.00 | **256** | -24.27225250 | 7.100572 | 104.1896 | 1726 s |
| 0.10 | 10.00 | **256** | -25.04565217 | 2.321989 | 252.7785 | 2248 s |

`S` starts at `ln(63504) = 11.0589` and falls monotonically; `E` falls
monotonically toward the DMRG ground state, reaching `E - E0 = 0.182` at
kT = 0.1, consistent with the 0.207 gap the Krylov solver reports.

**The honest limitation: the production artifact's cold rungs carry ~1-2e-2
Ha, but the METHOD converges faster than the first two sweep points
suggested.** The completed bond-dimension sweep at beta = 2 (kT = 0.5), all at
`dbeta = 0.05`, `cutoff = 1e-10`:

| maxdim | chi | E | shift | ratio |
|---|---|---|---|---|
| 64 | 64 | -23.41979142 | — | |
| 128 | 128 | -23.44191070 | -2.21e-02 | |
| 256 | 256 | -23.45596820 | -1.41e-02 | 0.64 |
| 512 | 512 | -23.45950851 | **-3.54e-03** | **0.25** |

(An earlier version of this entry, written before the 512 point landed,
extrapolated the 0.64 ratio to a ~2-3e-2 residual at 256. The third doubling
refutes that: convergence *accelerates* as chi approaches the state's true
Schmidt tail.) Geometric tails from the last ratio put the truncation residual
at **~5e-3 Ha at chi = 256** and **~1e-3 at chi = 512** (tight cutoff).

Two other error terms, measured:

- **cutoff dominates the production artifact.** The production ladder ran
  `cutoff = 1e-8` and gives -23.44480 at maxdim 256 where the `1e-10` sweep
  gives -23.45597 — a **1.1e-2** gap at identical maxdim. So the artifact's
  kT <= 0.5 rungs carry ~1-2e-2 Ha total, dominated by cutoff, not maxdim.
- **the step sweep is confounded with the expansion schedule.**
  E(dbeta=0.025) = -23.45803 vs E(0.05) = -23.45597 vs E(0.1) = -23.45580:
  the 0.1 -> 0.05 change is 1.7e-4 but 0.05 -> 0.025 moves 2.1e-3 — because
  `expand_every` defaults to `10*dbeta`, the smaller-step run also expanded
  its bond space twice as often. At these settings the "step error" is mostly
  expansion-frequency sensitivity. Future sweeps should pin `expand_every`
  explicitly.

**Corrected outlook:** chemical accuracy (1.6e-3) at kT = 0.5 needs
`chi ~ 512-1024` at `cutoff <= 1e-10` — one 2-4x-cost rerun, not the "chi in
the low thousands" a two-point extrapolation implied. The warm rungs are
*better* but not immune: re-running kT = 4/2/1 as independent ladders at
`cutoff = 1e-9`, `maxdim = 400` moves E by 5.8e-4 / 6.4e-3 / 6.9e-3 vs the
production values (settings and step grid both differ, so this is a
sensitivity envelope, not a pure cutoff study). Only kT = 4 is at the
sub-1e-3 level; treat kT = 1-2 as ~7e-3.

**What is nonetheless established:** the pipeline runs, is exactly right at
beta = 0 and at the ground state, and produces a full ladder at a sector size
where dense diagonalization is simply impossible. At CAS(8,6) the exact
purification needed `chi ~ 200` against a sector of 225 — essentially no
compression, which is why `ncas = 6` is a validation size and not a payoff
size. At CAS(10,10) `chi = 256` against 63,504 is ~250x compression on the
state but is demonstrably not yet enough for converged energies; the honest
statement is that the method **reaches** ncas = 10 and that converging it there
costs more bond dimension than this run spent.

**Artifact.** `results/qh9_mps_ncas10.h5` carries, per temperature, the
purification MPS plus a dense 1024x1024 `rho` reduced onto JW wires 0-9 — the
form `qnn/states.py` consumes. Read back in Python: `tr(rho) = 1.0000000`
exactly at every rung, symmetric to 1e-16, PSD to 1e-17. Note the subsystem
RDM is unit-trace *by construction*, so unlike the top-determinants projection
it carries no truncation error for a threshold model to read as signal
(INVARIANTS I8 is satisfied for free).

**Cost note for the next run:** most of the wall time is the global Krylov
subspace expansion, not TDVP — `expand` builds `H|psi>` at bond `chi * 103`
before truncating. Bounding that intermediate is the obvious next
optimisation.

---

## 2026-08-07 — **Module K: thermal states as purification MPS by imaginary-time evolution** (`QThermalMPS/`)

A Julia package that takes the `(ecore, h1eff, g)` Module G already stores and
produces `rho(kT)` as an MPS, with nothing diagonalized and no `2^Q` object
formed anywhere — the two ceilings that stop `qthermal/thermal.py` (dense
`eigh` on the sector) and `qthermal/mps.py` (Module J's `2^Q` scatter, dead at
`ncas = 8`).

```
|Psi(0)> = (1/sqrt(dim)) sum_{n in sector} |n>_phys |n>_anc      exactly rho(0) = P/dim
|Psi(b)> = e^{-b H/2} |Psi(0)> / norm                            two-site TDVP vs the JW MPO
rho(b)   = Tr_anc |Psi(b)><Psi(b)|                               plain qubit partial trace
```

One pass yields the whole kT ladder (evolution sweeps *through* every
intermediate beta), and `Z(beta) = dim * ||e^{-bH/2}|Psi(0)>||^2` comes out of
the evolution norm, so `logZ`, free energy and entropy are free.

**Validated** for `ncas <= 6` against two independent references: a by-hand
second-quantized Hamiltonian with explicit JW parity signs (pins the *basis*,
not just the spectrum) and dense `exp(-beta H)`; plus Boltzmann sums over the
stored `evals`. At beta = 0, `<Psi0|H|Psi0>` and `<Psi0|H^2|Psi0>` reproduce
`mean(evals)` and `mean(evals^2)` to ~1e-13 on `h2o` CAS(8,6) and QH9 CAS(8,8).

### Four findings, three of them traps

**1. `:blocked` beats `:interleaved` by 4–5x, measured exactly.** Dense
diagonalization then TT-SVD on the actual chain — no TDVP, so this is the
intrinsic cost of the representation:

| case | `:blocked` | `:interleaved` |
|---|---|---|
| random CAS(4,4), any `beta > 0` | 36 (= sector dim) | **256** (no compression at all) |
| `h2o` CAS(8,6), `kT = 0.25`, cutoff 1e-10 | 221 | **821** |
| `h2o` CAS(8,6), `kT = 0.25`, cutoff 1e-6 | 118 | **582** |

This **confirms and extends** the 2026-07-27 entry (which measured the
eigenblock purification) to the imaginary-time one, and settles
[`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) Q7 for this consumer.

**2. TDVP silently converges to the WRONG state unless the bond space is
opened first.** TDVP moves within the manifold of its current bond dimension;
the beta = 0 purification starts at `chi = O(ncas^2)` and under U(1)xU(1) whole
QN blocks are absent, so two-site TDVP often cannot grow into them. On `h2o`
CAS(8,6) `chi` stayed at **6** from beta = 0 to beta = 40 and `logZ` was wrong
by 0.4. **Both natural diagnostics miss it:**

- thermodynamic self-consistency proves nothing — `d logZ/d beta = -<H>` holds
  *exactly* on the wrong manifold, since the tangent-space projection preserves
  `<Psi|H|Psi>`. Both quantities stay consistent and both are wrong.
- a step-size study proves nothing — halving `dbeta` reproduced the same wrong
  number to every digit, because the manifold is exactly invariant.

Fixed with a global Krylov subspace expansion while `chi` is still growing (on
by default). The one visible symptom is `maxlinkdim` never rising above its
beta = 0 value.

**3. `prime(dag(psi), "Link")` is a trap.** `expand` returns bonds tagged
`"sum"`, not `"Link"`, so the idiomatic spelling misses them, ket and bra
contract straight through, and `tr(rho)` comes out a power of two too large
(16 on an eight-site chain) with the state itself perfectly fine. Prime by
index identity instead.

**4. Compiling the MPO on the purification chain is impossible, not merely
slower.** ITensor's `QN` holds at most four `QNVal`s and the chain already
spends all four on `(Nf, Sz, Na, Sza)`. `MPO(os, sites)` dies outright;
`ITensorMPOConstruction` does *not* die and instead returns an operator with
sign errors in some off-diagonal elements — invisible to `<H>` and `<H^2>` at
beta = 0 (both are sums of squares) and exposed only by an evolved state. The
Hamiltonian is therefore built on the bare `2*ncas` physical chain and
identity tensors spliced in at the ancillas.

**Ancillas are bosonic**, which is load-bearing rather than incidental: JW
strings then pass through them as the identity, so `Tr_anc` is the Module I
encoded `rho` with no dressing. With fermionic ancillas the naive partial
trace returns exactly **0** for `<10|rho|01>` on a two-wire hopping model where
the truth is `beta*t/2`. The cost is that the honest pair creator is odd and
`(sum_w X_w)^2 = 0`, so the beta = 0 state is built from parity-free `"Raise"`
operators instead.

**Not yet done:** a production run at `ncas >= 10`, which is the point of the
whole thing — at `ncas = 6` the exact purification needs `chi ~ 200` against a
sector of 225, so dense still wins at validation size.

---

## 2026-08-06 — **"Diagonal-dominated" is basis-relative, but the negative result survives: the control is a zero-correlation determinant**

The sharpest possible attack on every coherence number this project has quoted:
they are all computed in the determinant basis of **canonical, delocalized**
Kohn-Sham orbitals, and an orthogonal rotation inside the active space changes
no energy and no observable while redistributing weight between `diag(ρ)` and
`offdiag(ρ)`. Settled by `scripts/localized_basis_experiment.py`, all 1000
molecules, FCI ground states from the stored `h1eff`/`g`, with the
**zero-correlation reference determinant carried through every rotation** by
`fci.addons.transform_ci_for_orbital_rotation`:

| basis | correlated ground state | zero-correlation control |
|---|---|---|
| canonical (the pipeline's) | 0.0582 | 0.0000 |
| full Edmiston-Ruedenberg | 0.9293 | **0.9348** |
| block ER (occ-occ, virt-virt only) | **0.0580** | **0.0000** |

(median `‖offdiag ρ‖²_F / ‖ρ‖²_F`; FCI energies invariant to **2.6e-12 Ha**.)

**Full ER inflates a state with no correlation whatsoever to 0.935 — more than
the correlated state's 0.929.** That coherence is bookkeeping from mixing
occupied with virtual orbitals. Under the subgroup that preserves the reference
determinant the share is unchanged: max deviation **1.6e-3**, median **5.3e-5**.

**The rule.** The diagonal/coherence split is well defined *relative to a
reference determinant* and invariant under every rotation preserving it. A
molecule offers genuine coherence exactly insofar as it **has no dominant
reference**. Recorded as [`INVARIANTS.md`](INVARIANTS.md) I17.

**Where it does flip** (`scripts/basis_dependence_probe.py`): C₂H₄ torsion,
B3LYP/def2-SVP, CASCI(8,8), SCF continuation, FCI-invariance gate 2.3e-14 Ha,
barrier 98.5 kcal/mol. Lowest singlet vs lowest triplet, screen ratio
`‖offdiag(ρ_S−ρ_T)‖/‖diag(ρ_S−ρ_T)‖`:

| twist | `N_unpaired` | canonical cos / screen | localized cos / screen |
|---|---|---|---|
| 0° | 0.194 | 0.0000 / 0.65 | 0.9229 / 4.7 |
| 70° | 0.650 | 0.0000 / 0.83 | 0.9879 / 11.0 |
| **90°** | **1.997** | 0.0000 / 1.06 | **1.0000 / 2451** |

In the canonical basis the two states have **orthogonal populations at every
angle**, so a diagonal-only model separates them perfectly everywhere. The
occupied-virtual rotation that flips this is legitimate *only* at
`N_unpaired` ≈ 2, where the natural occupations are 1.009/0.991 and there is no
reference determinant to preserve.

→ Full assessment and what it implies for the dataset:
[`OMOL25_ASSESSMENT.md`](OMOL25_ASSESSMENT.md).

---

## 2026-08-06 — **Ten second-quantization labels, all failing; and QH9 has zero admissible molecules**

Tests the label class proposed for OMol25 (ionization energies, open-shell
ground states, "purely second-quantization" effects) on data we already have.
`scripts/second_quantization_labels.py` solves other particle-number and spin
sectors of the **stored** active-space Hamiltonians — no QH9 access, no SCF, no
run-file eigenvectors. Gates: independent FCI reproduces the stored spectrum to
**3.9e-11 Ha**; Slater's-rules determinant energies match PySCF's CI energy
evaluation to **exactly 0.0**. 1000/1000 solved.

Held-out accuracy (%), 25 stratified 70/30 splits (`sq_label_screen.json`):

| label | zero for 1 det.? | composition | diagonal (136) | full (248) | coherence only | **Q − C** |
|---|---|---|---|---|---|---|
| correlation correction to the IP | yes | 81.7 | 82.1 | 82.3 | 52.2 | **+0.13** |
| correlation correction to the EA | yes | 71.8 | 74.3 | 71.9 | 50.9 | **−2.37** |
| correlation correction to the S–T gap | yes | 69.3 | 74.7 | 74.8 | 57.9 | **+0.12** |
| quasiparticle pole strength `Z` | yes | 81.6 | 83.4 | 82.1 | 55.9 | **−1.35** |
| unpaired-electron count | yes | 84.3 | 95.4 | 94.6 | 51.7 | **−0.79** |
| double-excitation weight | yes | 84.3 | 95.4 | 94.6 | 51.7 | **−0.79** |
| active-space correlation energy | yes | 84.8 | 95.5 | 94.5 | 52.0 | **−1.00** |
| IP / S–T gap / EA (correlated) | no | 83.3 / 84.9 / 78.0 | 86.1 / 95.6 / 82.5 | 84.7 / 95.1 / 81.1 | 54.3 / 55.7 / 54.9 | **−1.45 / −0.45 / −1.45** |

**"No mean-field analogue" is not sufficient.** The labels are not small
(`IP_corr` σ = 0.49 eV, `ST_corr` σ = 1.07 eV, `Z` down to 0.813) and not all
composition proxies (`ST_corr` vs DoU Spearman **+0.125**, `IP_corr` **+0.328**,
against +0.789 for `N_unpaired`). They fail because in a reference-dominated
basis "how correlated is this molecule" **is** a diagonal readout: `w_double` and
`N_unpaired` are predicted identically to one decimal on all three feature sets,
and the correlated IP/EA/S–T gap are r² = 0.906 / 0.889 / 0.815 explained by
their own mean-field parts.

**Two escape hatches, both closed** (`multireference_stratification.json`):
stratifying by `N_unpaired` quartile gives **19 of 20 negative deltas** with no
trend; and learning curves show the coherence block *costs* sample efficiency
(`ST_corr` Q − C: −3.56 at n=60 → +0.96 at n=700).

**The discriminant, and QH9's position on it.** Because the off-diagonal share is
basis-relative (entry above), the honest instrument is basis-invariant:
`N_unpaired = Σᵢ min(nᵢ, 2−nᵢ)`.

| statistic | min | median | max |
|---|---|---|---|
| weight on the reference determinant | 0.885 | 0.970 | 1.000 |
| **`N_unpaired`** | 0.0003 | **0.113** | **0.4815** |
| quasiparticle weight `Z` | 0.813 | 0.962 | 0.999 |

**Zero of 1000 molecules exceed `N_unpaired` = 0.5**, and QH9's median is *less*
multireference than **planar** ethylene (0.194). QH9's most correlated molecule
sits between 50° and 60° of one ethylene twist.

**Consequence.** Every negative result this project has recorded was measured in
a regime where conditions 2 and 3 of the advantage argument are unreachable in
principle. They are not evidence against the hypothesis; they are evidence that
QH9 cannot test it. → [`OMOL25_ASSESSMENT.md`](OMOL25_ASSESSMENT.md),
[`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) Q14.

---

## 2026-08-06 — **The HOMO–LUMO gap label fails because it is a one-body observable, not because QH9 is too classical**

Full audit of the label that produced the project's first positive result
(`demo_train_curve.py`, 94% held-out). Four candidate explanations, each with a
decisive measurement, on all 1000 molecules of the CAS(8,8), kT = 0.1 run.
Producers: `scripts/gap_diagnosis.py`, `_followup.py`, `_ceiling.py`,
`_controls.py`, `gap_rho_pass.py` → `results/gap_diagnosis*.json`,
`results/gap_rho_pass.npz`. Deck: `Papers/homo_lumo_gap_diagnosis.pptx`.

**The instrument.** On a JW register the 136 Z/ZZ strings are diagonal
operators, so their expectations are functions of `diag(ρ)` alone; the 112
XX/YY strings read only off-diagonal entries. "Quantum − classical" is
therefore an exact feature ablation inside one model class (same logistic
trainer, same 25 stratified 70/30 splits, features standardized), with no
baseline-tuning objection available.

**Held-out accuracy (%), median-split gap, 300 unseen molecules:**

| features | acc |
|---|---|
| composition only (formula, DoU, π-system, quadratic) | 84.9 |
| 8 natural occupations | 92.7 |
| 16 single-qubit occupations `⟨Z⟩` | 92.3 |
| full `diag(ρ)` (4900 populations → 300 PCs) | 89.8 |
| **diagonal Pauli pool, Z+ZZ (136)** | **94.8** |
| ρ spectrum + CI gap (15 numbers, basis-independent) | 96.1 |
| coherence channel alone, XX/YY (112) | 53.0 (AUC 0.555) |
| full quantum pool (248) | 93.5 |

**Quantum − classical = −1.25 ± 1.13 points** (paired t, p = 1.1e-5; McNemar 102
vs 196, p = 6e-8). Adding the 112 coherence features *costs* 1.25 points where
112 Gaussian noise features cost 3.17 — measurably better than noise, nowhere
near paying for its dimensionality.

**Why, in order of decisiveness:**

1. **The label is a one-body mean-field quantity by definition.** `gap_Ha` is
   `ε_LUMO − ε_HOMO` from `eigh(F, S)`, computed before any correlation enters
   the pipeline. No dataset makes a one-body observable quantum.
2. **`diag(ρ)` is nearly sufficient for it.** Out-of-fold `R² = 0.856`.
   Mechanism: at kT = 0.1 the state's mixedness *is* the gap — ground-state
   weight r = +0.834, entropy −0.784, static_corr −0.788, CI gap +0.868.
3. **The coherence that exists is redundant, not absent.** The exact
   off-diagonal share of ρ correlates −0.571 with the gap but **+0.040 with the
   residual** after the diagonal model; static_corr −0.788 → −0.007; entropy
   −0.784 → −0.013. Adding all three to the diagonal pool moves accuracy by
   −0.01 points. Both channels are driven by the same chemistry: off-diagonal
   share vs DoU **Spearman +0.787**. Whatever the diagonal missed is predicted
   better by counting atoms (64.8%) than by coherence (51.9%, chance).
4. **Not the binarisation.** Ridge regression on the continuous gap gives
   diagonal `R² = 0.853` vs quantum `0.849`, **ΔR² = −0.0038**; the coherence
   channel alone reaches `R² = 0.000` (Spearman 0.11). Accuracy by quintile of
   |gap − median| is 75.0 / 95.4 / 99.0 / 100.0 / 98.3 — the median cut costs
   accuracy only where the gap is genuinely ambiguous.
5. **Not a weak apparatus.** On the *same* 1000 states, same pool, same
   trainer, a synthetic purely off-diagonal label separates the pools by
   **+42.0 points** (97.2 vs 55.2).

**Repairs tested and rejected.** Correlated CASCI gap `E₁−E₀`: −0.8. The
correlation correction (the part of the correlated gap the mean-field gap
cannot predict, σ = 1.04 eV): −1.1. Restricting to the *most coherent half* of
QH9: **−3.1**, i.e. worse.

**Headroom in harder chemistry, measured.** Off-diagonal Frobenius share of ρ
(`‖offdiag‖²_F/‖ρ‖²_F`): 1000-molecule set **6.70%** median (p90 13.6%, max
25.3%); the 28 most conjugated molecules **12.4%** at kT = 0.1 and **14.8%** at
kT = 0.25. About 2×, not orders of magnitude.

**Scope caveat, measured.** The weight-≤2 pool is a narrow window on ρ: it sees
**0.46%** of `‖diag(ρ)‖²` and **0.002%** of `‖offdiag(ρ)‖²` (medians), and only
56 of its 112 coherence features are independent (`⟨XX⟩ = ⟨YY⟩` exactly on
these real, S_z-conserving states; max deviation 0.0). So finding 1 is
pool-independent and finding 3 rests on the exact off-diagonal share, which is
not limited by the pool — but "no coherence channel exists in ρ" is *not*
claimed.

**Verdict: retire the raw HOMO–LUMO gap as a label**, and choose the next
dataset for composition-independent variation rather than raw difficulty. See
`OPEN_QUESTIONS.md` Q14 for the OMol25 / OMol_CSH_58k assessment.

---

## 2026-08-06 — **The R± screen does not predict this failure: screen the residual, not the raw label**

Methodological correction found by the audit above. The project's screening
statistic `‖offdiag(ΔR)‖_F / ‖diag(ΔR)‖_F`, computed on the true
class-aggregated `R± = Σ_m ρ_m` over all 1000 molecules (`gap_rho_pass.py`,
~25 min, reads 45 GB), gives **0.1345** for the gap label — sitting *between*
`⟨S²⟩` (0.122) and `c` (0.162), both of which it was used to reject.

Yet the gap label's coherence channel is at chance while a synthetic control on
the same states reaches +42 points. So the screen measures whether the classes
differ off-diagonally; it says nothing about whether that difference is
**redundant with the diagonal**. All three labels are diagonal-saturated in
exactly that way.

Same quantity in the 248-term Pauli feature basis is **0.0069** — the two are
not comparable and must not be quoted interchangeably (cf. the `coh_share`
conflation logged below).

**Change to the procedure:** run the screen on the label *residualized against
an out-of-fold diagonal model*, not on the raw label. Recorded as
[`INVARIANTS.md`](INVARIANTS.md) I16.

---

## 2026-08-06 — **The architecture is worth +26 points over the single neuron, on the identical label** (10-qubit controlled comparison)

The 8-qubit result below could not be compared against the single-neuron
experiment, because its control was a different random operator on a different
register. This run removes that objection: at `n_qubits = 10, seed = 7` the
synthetic control operator is **bit-identical** to the one in
`results/spin_comparison_metrics.json` — verified two ways, the R± screen ratio
(**0.3435 in both files**) and a direct assertion that the stratified split
matches `scripts/train_spin_comparison.py`'s own routine for all three labels.
Same 1000 molecules, same 70/30 split, same label. 500 epochs, T = 4.
`results/hybrid_spin_metrics_10q.json`, `figures/hybrid_spin_10q.png`.

Held-out accuracy (%) on that one label:

| model | **hybrid NETWORK** | single NEURON |
|---|---|---|
| quantum pool | **93.0** | 66.7 |
| Z-only (diagonal) | 77.0 | 66.3 |
| unconstrained diagonal | 77.3 | 69.3 |
| classical descriptors | 67.0 | 67.0 |

**The architecture is worth +26.3 points.** Same data, same label, same split;
the only difference is one layer of quantum neurons feeding a classical MLP
trained by the rule in [`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md), against one
neuron trained on the paper's spectral Fermi-Dirac loss.

**And the single neuron was not reading coherence at all.** Its quantum model
(66.7%) and its diagonal model (66.3%) both sit *at* the classical-descriptor
baseline (67.0%) — a 0.4-point separation that is indistinguishable from
chemistry. The hybrid separates by **+16.0 points** (93.0 vs 77.0) on the same
label. This is the strongest form of the Q11 finding: the earlier failure was
the *objective*, not the label and not the pool.

**Read against descriptors, but do not stop there.** The hybrid's Z-only model
reaches 77.0%, a full 10 points *above* the 67.0% descriptor baseline — so this
label's diagonal correlate is richer than six chemical descriptors capture. The
descriptor baseline is a floor on the confound, not a measurement of it. The
quantity that measures coherence access is the quantum − Z-only gap, and both
models must be run on the same label to get it.

**Confirms the 8-qubit conclusion is not a projection artifact**, and shows the
gap narrows with a more confounded label (+32 points at 8 qubits where the
descriptor baseline was 53.3%; +16 here where it is 67.0%) — exactly as it
should.

**A bug this run exposed and fixed.** `descriptor_baseline` drew its *own*
splits, so under `--labels-subset` the control got the first draw instead of the
third and the baseline came out 62.0% rather than 67.0%. The routine now takes
the already-drawn splits, so the baseline is by construction scored on the same
held-out set as the models. The stored metrics file was recomputed and records
that provenance. Same class of bug as the main-split ordering fixed earlier the
same day — **any routine that redraws a split silently disagrees with every
other number in the run.**

---

## 2026-08-05 — **Hybrid network on 1000 molecules: the coherence ablation lands at chance, the physical labels still do not separate**

First run of the hybrid network on real data. `scripts/train_hybrid_spin.py`,
1000 QH9 molecules, CAS(8,8), kT = 0.1 Ha, 8-qubit register (the 256
most-populated determinants — **97.4% of the population and 95.5% of the
off-diagonal Frobenius weight**, so coherence survives the projection). 8 quantum
neurons (tanh, T = 4) → classical [16, 8] → 1 logit, cross-entropy, full-batch
Adam, 600 epochs, 70/30 stratified split — the **same split** as the
single-neuron experiment. Five models per label, identical except for the one
ingredient each removes.
`results/hybrid_spin_metrics_8q.json`, `figures/hybrid_spin_8q.png`.

Held-out accuracy (%), final epoch:

| label | quantum | identity act. | no depth | **z_only** | diag_full | descriptors | R± screen |
|---|---|---|---|---|---|---|---|
| `⟨S²⟩` | 97.0 | 97.3 | 97.0 | **98.0** | 98.3 | 93.3 | 0.133 |
| `c` | 97.3 | 92.7 | 95.7 | **96.3** | 96.3 | 92.7 | 0.175 |
| **control** | 83.0 | 84.7 | 72.3 | **51.0** | 55.3 | 53.3 | 0.808 |

**The instrument works.** On the synthetic purely-off-diagonal control the
quantum pool reaches 83.0% while `z_only` sits at **51.0%** — chance, and below
that label's own descriptor baseline of 53.3%. The unconstrained diagonal model,
with **4× the parameters** (2337 vs 585), manages only 55.3%. That is the
theorem of [`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md) §5.2 observed on real
molecular states: no depth of classical layers extracts an off-diagonal label
through a commuting pool. Answers [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) Q2.

**Depth is worth ~10 points.** Removing the classical layers and keeping
everything else costs 83.0 → 72.3 on the control, and 97.3 → 95.7 on `c`. This
is the first direct measurement of what the paper's §VII.C architecture buys
over its single neuron, and it is not marginal.

**The nonlinearity cuts both ways, in the direction theory predicts.** Replacing
tanh with the identity — which turns the quantum layer into the raw Pauli
feature map — *helps* on the control (84.7 vs 83.0), because that label is
exactly a linear functional of ρ, and *hurts* on the physical label `c` (92.7 vs
97.3), where it is not. Worth noting the identity run is also visibly unstable:
with no bounded activation the weights grow without limit (spectral radius 75 by
epoch 500 against T = 4) and held-out accuracy oscillates by ±10 points.

**The physical labels still do not separate.** On `⟨S²⟩` the diagonal models are
*better* than the quantum one (98.3 / 98.0 vs 97.0). On `c` the quantum model
leads by 1.0 point — within the run-to-run scatter visible in the curves. Both
sit ~4 points above the classical-descriptor baseline, but so does `z_only`,
which provably cannot read coherence: that margin is composition, not quantum
structure. Consistent with the earlier single-neuron null and with the coherence
confound (2026-07-25). [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) Q1 and Q12 stand.

**Two cautions on reading this.**

- The 8-qubit control is a *different* random operator on a *different* register
  from the single-neuron run's control, and it happens to be far less confounded
  (descriptor baseline 53.3% here versus 67.0% there). So "83.0% vs the shallow
  model's 66.7%" is **not** a controlled comparison. The controlled comparisons
  are the ones within this table. **Resolved 2026-08-06** — the entry above runs
  the identical control label at 10 qubits and the cross-architecture gap is
  +26.3 points.
- **Read every ablation against its own label's descriptor baseline, not against
  50%.** The earlier single-neuron control looked like a failed ablation
  (classical 66.3%) when it was really a confounded label (descriptors 67.0%).

**Hyperparameters were tuned on the control label only** (five configurations,
600 epochs each) and then applied unchanged to all three. The sweep also
produced a clean instance of the dead-neuron failure mode: at `lr = 0.05` with
`l2 = 1e-3` the quantum layer's weights collapsed to zero — spectral radius
0.00, saturation 1.00, accuracy exactly 50.0% — which the per-epoch `sat` and
`rho(B)` diagnostics identified immediately and a loss curve alone would not
have.

---

## 2026-08-05 — **Backpropagation from a classical layer into a quantum neuron, derived and implemented**

The source paper (`Papers/Fermi-Dirac Machines.pdf` §VII.C) proposes exactly the
architecture this project wants — quantum data in, one layer of quantized
neurons, then ordinary classical layers — writes its forward pass (Eqs. 150–151),
and stops:

> "We leave it open to future work to simulate the performance and training of
> hybrid quantum–classical neural networks."

**The gap was real.** "One can take advantage of backpropagation" is an
assertion, not a rule: the chain rule has to terminate, and here it terminates in
the derivative of a **matrix function** with respect to the coefficients of its
argument — which is *not* `φ'(B)`. What the paper does derive (Appendix G) is the
gradient of the *fully quantum* network of §VII.B, a different model whose
gradients compose Fréchet-derivative superoperators and which its own Remarks 5
and 6 flag as not obviously implementable.

**The rule.** Full derivation in [`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md).
With `δ_{m,i} = ∂ℓ_m/∂a_{m,i}` the ordinary backpropagated error:

```
    ∂ℒ/∂Θᵢⱼ = Tr[ Hⱼ · Dφ(Bᵢ)[Rᵢ] ] ,      Rᵢ = (1/M) Σ_m δ_{m,i} ρ_m
```

the classical `δ · φ'(z) · x` with three substitutions: the input vector becomes
an operator pool, the per-sample sum becomes a **matrix-valued aggregate**, and
multiplication by `φ'(z)` becomes the Fréchet derivative of the activation
observable. Four consequences worth recording separately:

- **Reverse mode works only because `Dφ(B)` is self-adjoint** under the
  Hilbert–Schmidt inner product (its divided-difference kernel is symmetric).
  Without it, each of a neuron's `J` parameters would need its own `O(K³)`
  rotation rather than one per neuron — a factor ~150 at our pool size.
- **`R±` does not survive depth**, because the hybrid loss is not linear in
  `ρ_m`. What survives is `Rᵢ`, re-formed each epoch by one GEMM, which keeps the
  eigendecomposition count at `J₁` **independent of dataset size**. Measured at
  K = 1024, `J₁ = 8`: 3.6 s/epoch at M = 60 and 3.4 s at M = 300 — a 5× dataset
  at no measurable cost, the difference being within run-to-run noise.
- **On hardware no new quantum primitive is required.** The forward form
  `∂aᵢ/∂Θᵢⱼ = Tr[Dφ(Bᵢ)[Hⱼ]ρ]` is exactly what the paper's Theorem 1 /
  Algorithm 1 already estimates, and `δ` is classical: `J₁ × J` invocations per
  epoch. The sharp contrast with §VII.B is caused entirely by the scalar
  collapse — because `aᵢ` is a number, no error signal is ever propagated
  backward *through* a quantum layer.
- **A commuting pool reduces the whole network to a classical model on
  `diag(ρ)`, at any depth** — forward pass *and* gradient
  ([`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md) §5.2). This upgrades the Z-only
  ablation from a claim about one neuron's expressivity to one about an entire
  model class, and is recorded as [`INVARIANTS.md`](INVARIANTS.md) I15.

**It also lifts the constraint that made label design hard.** The composite is no
longer linear in ρ, so ratios and other nonlinear functionals come into range
([`QUANTUM_NEURON.md`](QUANTUM_NEURON.md) §2.3) — while a diagonal pool still
sees nothing off-diagonal. Depth expands what is *learnable* without expanding
what is *classically visible*, which is the combination this project needs.

**Implementation.** [`qnn/`](../qnn/README.md) — six modules, all six of the
paper's quantized activations (tanh Eq. 18, sigmoid Eq. 2, softplus Eq. 68, SiLU
Eq. 74, erf §IV A, GReLU Eq. 92, GeLU Eq. 97), 181 tests. The load-bearing ones:
the **composite** gradient finite-differenced across every activation, pool,
depth, loss and temperature; the paper's validated single-neuron trainer
recovered as the shallow special case (§5.4) against both an independent dense
reimplementation and `scripts/train_spin_comparison.py` itself; and the
commuting-pool theorem asserted bit-identically on dephased states.

Two findings that came out of building it:

- **`relu` cannot be an activation observable.** Not `C¹` at 0, so `φ^[1]` is
  undefined wherever an eigenvalue sits there — which, for a pool of traceless
  Pauli strings, is where the spectrum concentrates at initialization. This is
  visibly *why* the paper quantizes softplus and GReLU instead. `qnn` refuses it.
- **A quantized neuron has a failure mode with no classical analogue.** If the
  spectrum of `Bᵢ` is much wider than the activation temperature `T`, every
  divided difference vanishes and the neuron is *dead*, not slow. The fix is a
  spectral-scale initialization `σ = T/√(mean Tr[Hⱼ²]/K)`
  ([`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md) §9); `saturation()` and
  `spectral_radii()` are logged every epoch so the condition is visible rather
  than inferred from a flat loss curve.

---

## 2026-08-06 — **The coherence-confound numbers reproduce; the orphan CSV measures something else**

Recomputed while rebuilding the group-meeting deck, and it resolves the caveat
that has sat on the 2026-07-25 entry since it was written.

**Reproduced, exactly.** From the eigenblocks of all 1000 production states,
without ever forming ρ:

```
    Tr(rho^2)        = sum_k p_k^2
    Tr(Delta(rho)^2) = sum_I d_I^2 ,   d_I = sum_k p_k V_kI^2
    ||offdiag(rho)||_F^2 / Tr(rho^2)  = 1 - Tr(Delta(rho)^2)/Tr(rho^2)
```

| | logged (2026-07-25, unreproduced) | recomputed |
|---|---|---|
| median off-diagonal share | 6.7% | **6.70%** (p10 0.33%, p90 13.58%) |
| Spearman vs degree of unsaturation | 0.79 | **0.787** |

Producer: `scripts/presentation/build_cache.py`. Cost is one pass over the
45 GB run file, ~28 min, dominated by reading `civecs`.

**But the orphan CSV is not what it looked like.** `results/coherence_share_kT0p1.csv`
has a `coh_share` column with median **1.97×10⁻⁴** — three orders of magnitude
below 6.7%. It is the off-diagonal share of the **248-component Pauli feature
vector**, not of the density matrix: `coh_nonzero` is 112, exactly the XX/YY
feature count, and `diag_max` is the largest |Z| or |ZZ| expectation. That
quantity also reproduces (mean 0.0248% logged vs 0.0257% recomputed, from the
feature file).

**Why it matters.** Two different "coherence shares" were circulating under one
name, and the density-matrix one — the one the confound argument rests on — was
the one with no producer. It now has one. `OPEN_QUESTIONS.md` Q6 is
correspondingly narrowed: what remains is packaging, not verification.

Also recomputed on the production set for the deck: the feature-weight split is
**99.58% / 0.391% / 0.0257%** (single-mode occupation and its products /
connected occupation covariance / hopping coherence), against the 99.7 / 0.2 /
0.01 logged 2026-07-14 from a smaller run. Same three decades.

→ `Papers/thermal_states_presentation.pptx` slides 11–12,
`Papers/presentation_references.md` §10.

---

## 2026-08-05 — **The Fermi-Dirac loss does not chase off-diagonal signal on these states** (positive control failed)

The most consequential result of the day, and it is about the *machine*, not the
label.

**Setup.** A synthetic control label `y = sign(Tr(ρ A_od) − θ)` with `A_od` a
random combination of the pool's XX/YY generators — strictly off-diagonal, and
inside the quantum pool's span. Since the identity is also in the pool, an
**exact solution `w* = (A_od − θI)` exists**. The quantum neuron should reach
100%.

**It reached 66.7%.** Diagnosis (`scripts/` diag run, logged in
`spin_comparison_metrics.json`):

| | accuracy | FD loss |
|---|---|---|
| exact solution `w*` | **100.0% train / 100.0% held-out** | **3.5226** |
| what Adam converged to | 68.4% / 66.0% | **1.3714** |

The optimizer's operator has a **lower** Fermi-Dirac loss than the operator that
classifies perfectly. Scaling `w*` up makes the loss worse (16.2, 64.4, 322 at
×5, ×20, ×100), so this is not a normalisation artefact. **The loss genuinely
prefers a worse-accuracy operator.**

**Why.** The decision rule needs only the *mean* `Tr(ρH)` to have the right
sign. The loss `Tr[T ln(I + e^{−yH/T})ρ]` penalises the whole *spectrum* of `H`
on each state's support. Molecular thermal states are highly structured and
strongly overlapping (all 1000 share a dominant reference determinant), so the
loss is dominated by that common mode; the discriminative off-diagonal
directions are ~10× smaller and get shrunk away (converged ‖w‖ ≈ 1.5, versus
‖w*‖ ≈ 9.5).

**This is a property of the states, not of the method.** Running the *paper's
own* Fig. 8 setting — Haar-random pure states, realizable label — through
`figures/quantum_training_impls.py::run_pennylane`, 500 epochs:

| n | accuracy | loss |
|---|---|---|
| 2 | 31.6% → **96.4%** | 1.510 → 1.250 |
| 4 | 54.4% → **93.8%** | 1.459 → 1.348 |
| 6 | 49.0% → **95.4%** | 1.616 → 1.377 |

So the objective works on Haar-random states and fails on molecular thermal
states. **The information is present and linearly accessible**: plain logistic
regression on the *same* quantum feature vector gets 100% train / 91.0%
held-out on the control label.

**Consequence — this is a blocker independent of label choice.** Until the
objective is aligned with the decision rule, the experiment cannot demonstrate a
quantum advantage even with a perfect coherence label. Tracked as
`OPEN_QUESTIONS.md` Q11.

---

## 2026-08-05 — **Singlet / triplet-open-shell character is NOT a usable advantage label** (measured)

Direct test of the `QUANTUM_NEURON.md` §5 candidate #1, on all 1000 molecules of
the production set. Quantum neuron (I, Z, ZZ + XX, YY — 146 params) vs classical
neuron (I, Z, ZZ, strictly diagonal — 56 params), identical Fermi-Dirac loss,
optimizer, temperature, split and epoch budget. Figure:
`figures/spin_quantum_vs_classical.png`; metrics:
`results/spin_comparison_metrics.json`.

| label | quantum | classical | unconstrained diagonal | classical descriptors only |
|---|---|---|---|---|
| `⟨S²⟩` open-shell/triplet character | **99.0%** | **99.0%** | 99.0% | 93.3% |
| `c = Tr(ρ S²_od)` coherence-only | **94.0%** | **94.0%** | 94.0% | 92.7% |

**Quantum − classical = +0.00 points on both. Δloss = −9×10⁻⁵.**

**Why the label fails**, in order of decisiveness:

1. **The observable is ~86% diagonal.** On the S_z = 0 sector,
   `S² = D + S²_od` with `D` the (classical, diagonal) count of singly-occupied
   orbitals. Measured mean `|c|/|⟨S²⟩| = 0.139`.
2. **`corr(⟨S²⟩, D) = 0.9942`, R² = 0.988.** Median splits of `⟨S²⟩` and `D`
   assign **97.4%** of molecules to the same class. The label *is* a diagonal
   label to three significant figures.
3. **Even the coherence-only part is classically determined.**
   `corr(c, D) = 0.9192` (R² = 0.845). Stripping the diagonal from the
   *operator* does not decouple the label from the diagonal at the *dataset*
   level — exactly the per-state-vs-dataset distinction of
   `QUANTUM_NEURON.md` §7, now measured.
4. **Composition confound, again.** `c` vs degree of unsaturation Spearman
   **+0.835**; vs largest π-system +0.798; vs aromatic atoms +0.728; vs gap
   −0.714. Plain chemical descriptors reach 92.7% against the neurons' 94.0% —
   the entire quantum pipeline buys **1.3 points** over counting double bonds.
5. **The R± screen predicted it** before any training:
   `‖offdiag(ΔR)‖/‖diag(ΔR)‖` = **0.122** (`⟨S²⟩`) and **0.162** (`c`). The
   diagonal difference between classes is 6–8× the off-diagonal one.

**Physically:** at kT = 0.1 Ha, thermal triplet population and unpaired-electron
count are both driven by the same thing — small gap / extended π system. There
is no subset of QH9 where spin *coupling* varies at fixed open-shell *count*, so
there is nothing for a coherence-sensitive model to exploit.

**Controls run.** (a) The 10-qubit projection is not the cause: it retains
99.79% (median, 96.41% min) of off-diagonal Frobenius weight, and
`c_proj` vs `c_full` has Pearson 0.99996, max relative error 1.8%. (b) The
positive control (entry above) shows the apparatus additionally lacks power
here. (c) Gradients verified against central differences to 1e-9, and the
classical loss verified *exactly* invariant (|ΔL| = 0.00e+00) under off-diagonal
perturbations of R±.

**Chemistry sanity check passed**: most triplet-like are C₂H₂N₄ tetrazoles
(c = +0.39) and aromatic C₆H₄O / C₆H₅N; least are CF₄, H₂O, saturated esters.

**Verdict: do not use this label for the paper.** Detail and alternatives:
`OPEN_QUESTIONS.md` Q1, Q11, Q12.

---

## 2026-08-05 — The Fermi-Dirac neuron has no bias term, and it matters

While building the comparison: the decision rule is `sign(Tr(ρH))` with **no
intercept**, and every Z/ZZ/XX/YY Pauli string is traceless. So the model is
forced through the origin and *cannot express a thresholded label at all*.

Adding the identity to the pool (`Tr(ρI) = 1` for normalised ρ, so it **is** the
bias) moved held-out accuracy from 78.0% → **99.0%** on `⟨S²⟩` and 81.3% →
**94.0%** on `c`. It is a diagonal operator, so both models get it and it confers
no quantum advantage.

This matters for the paper's own setting only because there the labels are
`sign(Tr(ρ H_target))` with threshold exactly zero — realizable without a bias.
Any label defined by a **median split** (as ours must be, for balance) needs the
identity. Easy to miss; costs ~20 accuracy points silently.

---

## 2026-08-05 — Documentation framework established; two orphan artifacts found

Full repo audit produced `AGENTS.md` and the `docs/` knowledge base. Two
findings worth recording:

- **`results/coherence_share_kT0p1.csv` and
  `figures/qh9_cas8-8_kT0p1_diagnostics_1000mol.png` have no producer script.**
  Repo-wide search finds nothing that writes either. They came from ad-hoc
  analysis sessions. The coherence-confound result (below) rests on the first
  one, so it is currently **unreproducible**. Tracked as `OPEN_QUESTIONS.md` Q6.
- **Several docs carry stale counts.** `docs/thermal_pipeline_report.md` says
  50 molecules and 117 tests (actual: 1000 and 156); `CLAUDE.md` said 152 tests;
  `README.md` said 90 tests in `tests/qthermal/`. Corrected, and the report
  given a status banner.

---

## 2026-08-05 — Conjugated-subset run started at two temperatures

`results/qh9_conjugated_top45.h5` — top conjugated molecules from the full
screen, CAS(8,8), kT ∈ {0.1, 0.25}, `--keep-cap 2450`. **28 complete** at
audit time; resumable.

The log shows `ensemble cap bound at kT_max=0.25` repeatedly, with tails up to
**1.7×10⁻²**. Confirms the standing observation that kT = 0.25 wants
`--keep-cap 0`: at that temperature the thermal window genuinely spans a large
fraction of the 4,900-state sector, so a storage cap of 2,450 discards
percent-level weight. At kT = 0.1 the same run's tails sit at 1e-4…1e-3.

---

## 2026-08-05 — Full-QH9 conjugation screen completed (130,812 molecules)

`scripts/screen_conjugation.py` over the whole database →
`results/qh9_conjugation_screen_full.csv`. Three tiers: degree of unsaturation
(formula, free), frontier gap + near-degenerate frontier DOS (`eigh(F,S)`,
free), largest conjugated π-subsystem (RDKit bond perception from geometry).

This is now the selection instrument: targeted runs pick molecule ids from this
ranking rather than walking the DB in index order. The `--indices` feature in
`qthermal/run.py` (uncommitted) exists to consume it.

---

## ~2026-07-30 — CAS(8,8) is adequate at T = 0 but **truncated for the kT = 0.1 thermal states**

Boundary-occupation check on the production set. Ground states are bracketed
fine even for high-DoU / conjugated molecules (T = 0 edge slack < 0.05). But
the **thermal** states at kT = 0.1 are not: the high-DoU decile has median edge
slack **0.167**, with **92% above 0.1**.

**Interpretation.** The excitable π* ceiling is real — at this temperature the
Boltzmann window reaches states that want orbitals outside the 8-orbital
window. The active space is adequate for ground-state chemistry and marginal
for the thermal ensemble that is actually our input.

**Consequence.** Any result that depends on states pressed against the active
boundary needs a CAS(8,10) or CAS(8,12) confirmation run.
**Still pending** — `OPEN_QUESTIONS.md` Q3.

---

## ~2026-07-28 — Group-meeting deck reconciled to the 1000-molecule dataset

`Papers/thermal_states_presentation_final.pdf`. The editable `.pptx` lives
outside this repo (user's Downloads). LibreOffice rendering is broken in the
sandbox — re-export the PDF with the user's own tool, not here.

---

## ~2026-07-27 — **Blocked beats interleaved for the purification MPS** (~2× smaller χ)

`benchmarks/mps_bond_dimensions.py` on real thermal blocks. The ancilla bond is
the thermal rank *m* and is identical for both layouts, so the comparison is
purely about physical inter-qubit bonds.

**This overturns the prior hypothesis.** The benchmark's own docstring says
"interleaved is expected to win" (same-orbital α/β pairs adjacent), and
`qthermal/README.md`'s Module I row says interleaved "remains the right layout
for future MPS backends". **Both are now wrong** and are flagged in
[`INVARIANTS.md`](INVARIANTS.md) I12; correcting them is `OPEN_QUESTIONS.md` Q7.

**Standing guidance** — wire ordering is per-consumer:
blocked for features and MPS, interleaved for coherence-label operators
(where `S²_od` becomes string-free and 4-local).

---

## ~2026-07-25 — **The coherence confound**: quantumness ≈ composition

From `results/coherence_share_kT0p1.csv` (1000 molecules, CAS(8,8), kT = 0.1):

- median off-diagonal **coherence share 6.7%** — the states are
  diagonal-dominated;
- coherence magnitude vs **degree of unsaturation: Spearman 0.79**.

**Interpretation.** DoU is countable from the chemical formula for free.
So "how quantum is this molecule" is, empirically, "how unsaturated is it".
Any label proportional to coherence *magnitude* is a composition label wearing
a disguise, and a classical model will match it.

**Consequence — this reframed the whole project.** The target is no longer
"label by quantumness" but "label by something only the *structure* of the
coherences determines". See [`QUANTUM_NEURON.md`](QUANTUM_NEURON.md) §3–§5.

⚠️ **Caveat:** the producing script is missing (2026-08-05 entry). Numbers are
carried forward on trust until reproduced.

---

## 2026-07-24 — Julia trainers added (Yao + ITensor, Algorithms 8/9)

`tensor-network-testing/`. Algorithm 9 (Thm. 5 / Eq. C27) is the useful one —
it differentiates the loss actually being minimized, so `train_alg9.jl` is
honest gradient descent on L^log. Algorithm 8 descends a *different* loss.

Yao (statevector) is the training backend; ITensor/MPS is for validation and
larger *n* and is far too slow to drive an optimizer. `convergence_checks.jl`
covers correctness (0–3), convergence sweeps (4–7: Trotter step, bond
dimension, cutoff, Monte Carlo) and cost vs *n* (8), with a fixed seed so
discretization error is isolated from sampling noise.

**Deliberate deviation:** Algorithm 8's MPS variant returns the final MPS
instead of sampling measurement outcomes.

---

## 2026-07-22 — Purification MPS from eigenblocks (Module J)

`qthermal/mps.py`. Because the eigenvectors are orthonormal, the eigenblock
*is already* the Schmidt decomposition across the system|ancilla cut — the
ancilla bond is exactly the thermal rank *m* with singular values `√p_k`, so no
SVD and no truncation happen there. Only the Q qubit sites are TT-SVD'd.

Untruncated the purification is exact; capping bonds gives `truncation_error`,
an upper bound on `‖Ψ − Ψ_trunc‖₂`, and since partial trace contracts the trace
norm, `‖ρ − ρ_trunc‖₁ ≤ 2·truncation_error`.

**Ceiling:** cost is bounded by the encoded block `m · 2^Q` — fine through
ncas = 8 (2¹⁶). Larger active spaces need a sector-basis contraction that never
materializes the full register.

---

## 2026-07-20 — Sector build 34× faster via `pspace` at np = dim

Replaced the `contract_2e`-over-unit-vectors loop in
`diagonalize.build_sector_hamiltonian` with one call to
`fci.direct_spin1.pspace(..., np=dim)` — PySCF's compiled preconditioner-matrix
builder. At `np = dim` the requested subset is the whole sector, so its internal
address list is the identity permutation and the output is already in the same
(na, nb) string order used everywhere else (asserted at runtime).

**Measured: 3.8 s → 0.1 s** on a real CAS(8,8) molecule, bit-identical to
float64 noise.

---

## 2026-07-20 — Closed-form non-interacting solver replaces a dense diagonalization

`NonInteractingSolver`. The g = 0 many-body Hamiltonian is diagonal in the
Slater basis built from `h1eff`'s own eigenvectors, so its spectrum is the
outer sum of alpha-subset and beta-subset energies — an `ncas × ncas`
diagonalization plus combinatorics, never `dim × dim`.

Eigenvectors come back via `fci.addons.transform_ci_for_orbital_rotation(det,
ncas, nelec, U0.T)`. **The transpose is load-bearing**: it is the
new-basis→old-basis map, the reverse of the function's documented forward
convention, and it was confirmed empirically (checking `Hv − Ev` directly)
rather than assumed.

**Verified** against `DenseEDSolver(g=None)`: eigenvalues to ~1e-13,
eigenvectors to `|Hv − Ev|` ~1e-15 with overlap 1.0 to 10 digits, on real QH9
data. **Measured 28 s → 2 s** for the ~350 states a kT = 0.25 run keeps at
CAS(8,8). The win shrinks (never reverses) as `--keep-cap` grows.

Bonus: it carries none of the dense guardrails, so the Gaussian audit stays
cheap at active-space sizes where the interacting solve already needs a
Phase-2 backend.

---

## 2026-07-20/21 — **Production dataset extended to 1000 molecules**

`results/qh9_dense_cas8-8_kT0p1.h5` — 1000 molecules, CAS(8,8), kT = 0.1 Ha,
dense ED, default keep cap (= 1225 at dim 4,900), 45.4 GB.

Run: **47,725 s (13.3 h)**, 950 written + 50 resumed, **0 skipped**. Median
192 s/molecule, max 313 s. Recorded tails at the cap: 1e-4…1e-3.

Zero skips across 1000 real molecules is itself a result — unit detection,
orbital validation, and active-space selection all held.

---

## 2026-07-15 — kT = 0.1 sits in the interesting middle regime (50-molecule stage)

From `docs/thermal_pipeline_report.md`. At kT = 0.1 the ground state keeps
**29–84% of the weight (median 57%)** — strongly mixed, but still sharply
different *between* molecules, which is what a labeled dataset needs.
By contrast kT = 0.25 was uniformly hot and kT = 0.025 uniformly cold.

**Chemistry drives it interpretably.** Mixing tracks the energy gap at
**r = −0.78**: conjugated chains (diacetylene, cyanoacetylene) mix most;
saturated molecules (methane, water) least. Trace distance to the
non-interacting reference splits by chemical family — saturated skeletons
0.34–0.51, N/O π-systems 0.93–0.99.

⚠️ **Caveat on that last split**: the non-interacting reference omits mean-field
electron repulsion, which inflates the distance for compact molecules. Part of
the family split may be that artifact. A mean-field-corrected reference is a
known follow-up (`OPEN_QUESTIONS.md` Q5).

---

## 2026-07-14 — 99.7% of the Pauli feature weight is single-qubit occupation

Measurement behind the extended-Heisenberg basis design. Of the 248 features:
occupation information that factorizes into single-qubit products carries
**99.7%** of the weight; occupation covariances **0.2%**; hopping coherences
**0.01%**.

Two consequences:
1. The basis was restricted to exactly the weight-≤2 strings symmetry does not
   force to zero — Z on every wire, ZZ on every pair, XX/YY within each spin
   block (`4·ncas² − ncas` = 248 at ncas = 8). Cross-spin XX/YY vanish by S_z
   conservation; XY/ZX/single-X vanish by parity superselection and
   time-reversal.
2. **The collective part is tiny but structured** — and earlier measurements
   indicated it is exactly the informative part for classification. This is the
   quantitative root of the label problem.

Also measured: **blocked** ordering reads ~10× more connected-ZZ signal and is
the only layout with nonzero XX/YY under a nearest-neighbour ansatz at
kT = 0.25 (interleaved's adjacent pairs are all spin-flip). Interleaved wins
only the small low-kT same-orbital pairing signal.

---

## 2026-07-13 — 284 GB of corrupted derived data deleted; single-determinant branch retired

`data/groups/qh9_slater_*.h5` — 95 files, 125,013 records — removed. Every file
carried `loader_module='raw SQLite QH9Stable.db'` and 9/9 sampled records
reproduced the AO double transform exactly.

Also retired: `data/active_space_encode.py`, `data/build_state_vectors.py`
(recoverable from git). `data/build_slater.py` is **kept**, bug fixed and
verified, purely as the regeneration route if a mean-field baseline is ever
wanted.

**No downstream contamination.** Repo-wide search confirmed nothing consumed
those files; `data/qh9_scan.jsonl` is geometry-only and clean.

Same day: the matrix-free Chebyshev implementation, accidentally dropped from
`logloss_pennylane.ipynb` in the 2026-06-29 rebase, was restored from commit
`b373b395`.

---

## 2026-07-10 — AO-ordering audit completed

`data/qh9_raw_sqlite_audit.md`. Second site of the bug found and fixed
(`data/build_slater.py::RawQH9SQLiteDataset.__getitem__`) — notably its LMDB
path never transformed, and its CLI even *refuses* `--ham-ordering qh9`, so the
raw-SQLite adapter was silently violating its own contract.

Error magnitudes in the corrupted tree (n = 9 sampled): HOMO error median
0.74 eV / max 3.8 eV; gap error median 0.92 eV / max 3.0 eV; Slater weight
entries shifted by up to 0.93; **binary gap labels flipped near threshold** —
and the stored threshold itself was corrupted (8.107 eV stored vs 9.175 eV
correct for the 48q group).

Fix verified: the corrected builder reproduces water (qh9_index 2) at
homo = −7.8451 eV, gap = 9.1752 eV, matching an independent `eigh(Ham, S)`
recompute to <10⁻⁴ eV and matching qthermal's corrected loader.

`results/qh9_krylov_ncas10.h5` regenerated on the corrected loader the same day.

---

## 2026-07-09 — **Raw QH9 Hamiltonians are already PySCF-ordered** (the founding bug)

Discovered while wiring `qthermal/loader.py`. Untransformed blobs agree with
freshly converged B3LYP/def2-SVP to **≤ 3.7×10⁻³ Ha** across the full spectrum
on records 0–4. Transformed blobs show ~1 Ha spurious core shifts on compact
molecules — **inside the loose physicality windows, so unit detection passed
anyway** — and intruder eigenvalues on linear ones (HCN −57.7 Ha, C₂H₂
−173.7 Ha; true 1s levels −14.4 / −10.2 Ha).

**The lesson that generalizes:** the physicality windows were not tight enough
to catch a ~1 Ha error. Validation gates that pass corrupt data are worse than
no gates, because they manufacture confidence. This is why
[`INVARIANTS.md`](INVARIANTS.md) entries carry verification *commands*, not
just prose.

Now [`INVARIANTS.md`](INVARIANTS.md) I1.

---

## 2026-07-09 — Two deliberate deviations in unit detection

Both evidence-driven, both documented in `qthermal/README.md`:

1. **The synthetic test fixture is B3LYP, not RHF.** The `detect_units`
   physicality window (gap ∈ [0.02, 0.6] Ha) rejects RHF H₂O, whose gap is
   0.67 Ha. QH9 is B3LYP anyway, so the fixture matches the data.
2. **`detect_units` needs a geometric tiebreak.** The spectral test is
   one-sided: misreading Ångström as Bohr *compresses* the molecule and the
   core levels collapse unambiguously — but misreading Bohr as Ångström merely
   *stretches* it, and the `eigh(F,S)` spectrum can stay inside all three
   windows. When both hypotheses pass, the one whose shortest interatomic
   distance lies in the covalent window [0.7, 1.7] Å wins.

---

## 2026-06/07 — Classifier optimization: label aggregation is the whole game

`docs/classifier_optimization.md`, `docs/scaling_comparison.md`.

The decisive realization: **the loss and gradient depend on the training set
only through `R± = Σ_{y=±1} ρ_i`**, so per-state work drops out of the epoch
loop entirely. Measured at n = 4: original 1.34 s → exact-aggregate 0.0077 s at
1,000 samples (**174×**), with runtime approximately flat in sample count while
every other method scales linearly. Static representation fell from 3.99 MiB to
0.0156 MiB (**256×**).

Also landed: matrix-free fused Pauli kernels + adaptive Chebyshev loss
(crossover at **n = 9**; at n = 10, degree 18 and 6.9× faster than exact
diagonalization, with 42× less memory).

**Deliberately not adopted:** QNode/parameter-shift autodiff — the loss is a
*spectral* function of H(ω), not a circuit expectation, so label aggregation was
both simpler and faster. **Deferred:** JAX/GPU (`resolve_array_backend` leaves
the seam open; estimated 5–20×, untested — no CuPy on this machine).

**Corrected:** the paper notebook's `dfj` divides the gradient by T, an
extraneous factor relative to Thm. 5 / Eq. (63). All optimized paths use
`dH/dω_j = H_j`. Now [`INVARIANTS.md`](INVARIANTS.md) I7.

---

## 2026-06 — Figure-8 reproduction needed a *purely quantum* target

`figures/generate_paper_training_curves.py`. With a random quantum target
(XX+YY+ZZ nearest-neighbour + X+Y+Z) on Haar-random states, the **classical**
FCIM reached similar loss to the Heisenberg model — because all-to-all ZZ can
fit the target's ZZ component. The two models were solving statistically
equivalent sub-tasks.

**Fix:** use a purely quantum target — **XX+YY nearest-neighbour only**. The
diagonal FCIM then has near-zero gradient, because ZZ expectations carry no
information about XX+YY labels.

**This is the same lesson as the coherence confound, two months earlier and on
synthetic data:** if the classical model can fit your target, your target was
classical. It should have been the warning for the label problem.
