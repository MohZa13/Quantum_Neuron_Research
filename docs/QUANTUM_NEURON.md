# The quantum neuron — model, hypothesis class, and the label problem

*The project's focus document. Everything else in `docs/` supports this.*

Prerequisites: [`ORIENTATION.md`](ORIENTATION.md) for context,
[`GLOSSARY.md`](GLOSSARY.md) for vocabulary.

---

## 1. The model, exactly

From `Papers/Fermi-Dirac Machines.pdf`, implemented in
`notebooks/pennylane/logloss_pennylane.ipynb` and reproduced verbatim in
`figures/quantum_training_impls.py`.

**Neuron output** (Eq. 16–18, and the clearest statement in
`notebooks/pennylane/sampling_demo.py`):

```
    output(ρ) = Tr[ g_T(H(ω)) · ρ ]        g_T(x) = tanh(x/T)
    H(ω)      = Σ_j ω_j P_j                 P_j Pauli strings, ω trainable
```

**Decision rule** — `figures/quantum_training_impls.py:78`:

```python
energies = [np.trace(H_model @ rho) for rho in states]   # predict sign(·)
```

**Training loss** (Eq. 56) — line 139–140 of the same file:

```python
m_loss_q = evec @ diag(T * log(1 + exp(-y_i * eval / T))) @ evec.conj().T
l_q     += np.real(np.trace(m_loss_q @ training_states[i]))
```

**The paper's own labels** — line 124:

```python
ys = [np.sign(np.real(np.trace(H_target @ rho))) for rho in training_states]
```

That last line matters more than it looks. The paper labels its synthetic data
with `y = sign(Tr(ρ H_target))` for a *random* `H_target`. Any label we design
in that same form is not a new construct — it is the paper's own labeling
scheme with a physically meaningful operator substituted for the random one.

---

## 2. What the model can and cannot express

Four facts, and together they pin down the entire label-design problem.
§2.1–§2.2 are about the **single neuron** of §1; §2.3 says what changes once
that neuron is a layer inside a network, and §2.4 is a property of our data.

### 2.1 The single neuron's hypothesis class is thresholds on linear functionals of ρ

```
    { y = sign(Tr(ρ H))  :  H = Σ_j ω_j P_j , P_j ∈ operator pool }
```

Nothing else. Consequences:

- A label of the form `sign(Tr(ρA) − θ)` with `A` in the pool's span is
  representable **exactly, with zero Bayes error**.
- A label that is a *nonlinear* function of ρ — entropy, purity, negativity,
  a ratio of two expectations — is **not in the class**. The model cannot
  express it; at best it lucks into a correlated linear surrogate.
- Ratios break linearity; **differences do not**. Do not label on
  `Tr(ρA₁)/Tr(ρA₂)`; label on `sign(Tr(ρ(A₁ − cA₂)))`, which is still one
  linear functional. Since `Tr ρ = 1` for our states, normalizing costs
  nothing.

### 2.2 Training sees the dataset only through R±

The loss is linear in each `ρ_i`, so it depends on the training set only
through `R± = Σ_{y=±1} ρ_i`. This is why per-epoch cost is **independent of
dataset size** after one aggregation (measured 174× at 1,000 samples;
`docs/scaling_comparison.md`).

It is also a free screening tool: the entire learning problem is *"find an
operator that tells R₊ from R₋"*. See §6.

### 2.3 A hybrid network breaks the linearity ceiling — but not the coherence theorem

Everything in §2.1 is a statement about **one neuron**. Since 2026-08-05 we also
have the paper's §VII.C architecture — a layer of `J₁` quantum neurons feeding a
classical MLP — implemented in [`qnn/`](../qnn/README.md) and derived in
[`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md). Its hypothesis class is

```
    { y = sign( F( Tr[φ(B₁)ρ], …, Tr[φ(B_{J₁})ρ] ) )  :  Bᵢ ∈ span(pool),
                                                          F realizable by the MLP }
```

Each `aᵢ` is still linear in ρ. But `F` is not, so **the composite is not**.
That changes the label-design rules of §2.1 in three specific ways:

| §2.1 said | With depth |
|---|---|
| ratios of expectations are outside the class | **inside** — `F` can divide; the "use `A₁ − cA₂`, never a ratio" rule is no longer forced |
| nonlinear functionals (purity, entropy) are outside | **inside if they factor through finitely many linear functionals.** E.g. `Tr(ρ²) = Σⱼ Tr(ρPⱼ)²/K` over a complete Pauli basis — expressible, but needs `J₁ = K²` neurons, so exact purity is exponentially wide. Approximations on a sub-basis are cheap |
| thresholds on *one* linear functional | thresholds on any decision surface in `J₁` of them — including non-convex and disconnected regions |

Note also that `φ(B)` for `B` in the pool's span generally leaves that span: the
*reachable observables* are richer than the pool, because functional calculus on
a sum of non-commuting terms generates the algebra. The pool bounds what `B` can
be, not what `φ(B)` is.

**What does not change — and this is the important half.** The
classical-reduction corollary
([`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md) §5.2) holds at *any* depth: for a
mutually commuting pool, both the forward pass and the gradient depend on ρ only
through `diag(ρ)`. So

> depth expands which **labels** are learnable; it does not expand what a
> **diagonal pool** can see.

The §4 construction below is therefore *strengthened*, not weakened, by the
hybrid architecture: the same coherence-only label is now being withheld from a
much larger classical model class, and the Z-only ablation still lands at
chance by theorem rather than by measurement.

### 2.4 Our states are real, so odd-Y Pauli strings vanish

`civecs` are real (solver eigenvectors), so ρ is real symmetric. A Pauli string
with an odd number of Y's has purely imaginary entries, hence
`Tr(ρP) = 0` identically.

Practical: roughly half a general Pauli pool is dead weight here, and no label
may live in that half. The 248-term extended-Heisenberg basis already respects
this — it is exactly the weight-≤2 strings symmetry does not force to zero.

---

## 3. Why the obvious labels fail

`scripts/demo_train_curve.py` does work: 1000 real molecules, labeled by
median-split HOMO–LUMO gap, classified from the 248 Pauli features, generalizes
to 300 held-out molecules. So why is that not the result?

Because of what the features are made of, and what the labels correlate with:

| Measured | Implication |
|---|---|
| **99.7%** of Pauli feature weight is occupation information that factorizes into single-qubit products (occupation covariances 0.2%, hopping coherences 0.01%) | The classifier is mostly reading *which orbitals are filled* — a classical quantity |
| Median off-diagonal **coherence share is 6.7%** across the 1000-molecule set | The states are diagonal-dominated. There is a quantum residue, but it is small |
| Coherence magnitude vs degree of unsaturation: **Spearman 0.79** | "How quantum is this molecule" ≈ "how many double bonds does it have" — countable from the formula, for free |

So a label proportional to *amount of coherence* — coherence share, purity gap
`Tr(ρ²) − Tr(Δ(ρ)²)`, the current `static_corr` default — is a **composition
label in disguise**. Likewise anything that a classical descriptor
(gap, DoU, atom counts, occupation numbers) predicts well.

**Avoid as standalone labels:** coherence share, purity gap, `static_corr`,
entropy, raw HOMO–LUMO gap.

> **The gap case is now fully audited (2026-08-06)** — see
> [`RESEARCH_LOG.md`](RESEARCH_LOG.md). Quantum − classical is **−1.25 points**
> and the coherence channel alone sits at chance (53.0%, AUC 0.555, regression
> `R² = 0.000`), while `diag(ρ)` alone explains `R² = 0.856` of the gap. The
> reason is stronger than "confounded": `gap_Ha` is `ε_LUMO − ε_HOMO` from
> `eigh(F, S)`, a **one-body mean-field eigenvalue difference** defined before
> correlation enters the pipeline. The confound is the mechanism by which
> `diag(ρ)` gets it (off-diagonal share vs DoU, Spearman +0.787), and the
> redundancy is measured directly: coherence correlates −0.571 with the gap but
> **+0.040 with the residual** after a diagonal model.
>
> Three consequences for §5. (a) The correlated CASCI gap and the *correlation
> correction* to the gap were tested as repairs and both fail (−0.8, −1.1), so
> "use a more correlated version of a one-body quantity" is not a route. (b) The
> failure is not binarisation: regression reproduces it, `ΔR² = −0.0038`. (c) The
> R± screen does **not** flag it (0.1345, between `⟨S²⟩` and `c`) — screen the
> residual instead, [`INVARIANTS.md`](INVARIANTS.md) I16.

---

## 4. The construction: strip the diagonal

Let `Δ` be dephasing — erase all off-diagonal entries, keep the diagonal.
That is precisely "what a classical model sees". For any operator `A`:

```
    Tr(ρA) − Tr(Δ(ρ)A)  =  Σ_{i≠j} ρ_ij A_ji  =  Tr(ρ · A_od)
```

where `A_od` is `A` with its diagonal zeroed. One line of algebra, and it gives
the whole recipe:

> **Pick a physically meaningful observable `A`. Zero its diagonal.
> Label `y = sign(Tr(ρ A_od) − θ)`.**

Three properties come for free:

| Property | Why |
|---|---|
| **Exactly learnable** | linear in ρ ⇒ inside the hypothesis class (§2.1), provided `A_od` is in the pool's span |
| **Provably invisible to classical models** | `diag(ρ)` contributes *zero* — not "little", zero |
| **A built-in ablation** | retrain with a **Z-only pool**. Z-only ⇒ `H` diagonal ⇒ `Tr(ρH)` reads only `diag(ρ)` ⇒ chance accuracy, necessarily |

That last row is the real prize. Most quantum-advantage claims are empirical
comparisons someone can always attack ("you undertuned the baseline"). This one
is a *theorem about the model class*, tested by the same code path with a
restricted pool — no separate classical baseline to argue about.

---

## 5. Label candidates, ranked

### ✗ 1. Spin coupling — `A = S²` — **TESTED AND REJECTED (2026-08-05)**

> **Measured on all 1000 molecules: quantum − classical = +0.00 points**, on
> both `⟨S²⟩` (99.0% vs 99.0%) and `c = Tr(ρS²_od)` (94.0% vs 94.0%). Classical
> chemical descriptors alone reach 93.3% / 92.7%.
>
> The theory below is correct — singlet vs triplet *is* 100% coherence per state.
> What it misses is the dataset level: `corr(⟨S²⟩, D) = 0.994` and
> `corr(c, D) = 0.919`, so across QH9 the spin *coupling* and the unpaired-electron
> *count* move together. Per-state invisibility did not survive contact with §7.
>
> A second finding came out of the same experiment: the FD loss does not chase
> off-diagonal signal on these states at all (`OPEN_QUESTIONS.md` Q11), so this
> null result is doubly caveated.
>
> Full numbers, controls and diagnosis: `RESEARCH_LOG.md` 2026-08-05.
> **Kept here because the construction is still the right template** — what
> failed was this particular observable on this particular dataset, not the
> strip-the-diagonal method.

Here the coherence-dependence is structural, not empirically lucky.

Two orbitals, one electron each, `S_z = 0`. Nature builds two states from the
*same two determinants*:

```
    singlet = ( |p↑ q↓⟩ − |p↓ q↑⟩ )/√2      ⟨S²⟩ = 0
    triplet = ( |p↑ q↓⟩ + |p↓ q↑⟩ )/√2      ⟨S²⟩ = 2
```

Identical 50/50 diagonals. **Only the sign of the off-diagonal differs.** The
singlet/triplet distinction is 100% coherence, by construction.

On the `S_z = 0` sector the total-spin operator splits cleanly
(`S² = S₋S₊` there, since `S_z(S_z+1) = 0`):

```
    S²  =  D                    +   S²_od
           │                        │
           │                        └─ spin exchange between orbitals p≠q:
           │                           purely OFF-diagonal. "how are the
           │                           unpaired spins coupled?"
           └─ Σ_p n_{pβ}(1 − n_{pα}) = half the singly-occupied count:
              purely DIAGONAL. "how many unpaired electrons are there?"
```

So the coherence-only label is

```
    c_m = ⟨S²⟩_m − ⟨D⟩_m = Tr(ρ_m S²_od)
```

with an interpretable scale, per open-shell pair: `0` closed shell, `−1`
singlet-coupled, `+1` triplet-coupled. A classical model reading occupations
sees `⟨D⟩` — "two unpaired electrons" — and **cannot** see which coupling.

**Chemically real.** "Thermally accessible diradical character" governs
photochemistry, singlet fission, and reactivity. At kT = 0.1 Ha ≈ 2.7 eV,
singlet–triplet gaps of 3–5 eV give thermal triplet weights ≈ 10–30%.

**Computable from files we already have.**
- `⟨S²⟩ = Σ_k p_k S_k(S_k+1)` — FCI eigenstates are spin-pure; get each root's
  spin from `pyscf.fci.spin_op`.
- `⟨D⟩ = Σ_k p_k Σ_I |c_kI|² D_I`, where `D_I` is an integer read straight off
  the bit pattern of `basis_indices[I]`. Pure NumPy, no PySCF.

**Representable by the ansatz.** Under **interleaved** ordering (which
`export_thermal_training.py` already defaults to), each orbital's α and β wires
are adjacent, so JW parity strings cancel and `S²_od` becomes a sum of
**4-local, string-free, even-Y** Pauli terms on pairs of adjacent wires.
Squarely inside a plausible pool, and manifestly not Z-only.

> No contradiction with the MPS finding that *blocked* gives ~2× smaller χ —
> different consumer, different optimum. Blocked for tensor networks,
> interleaved for this label. See [`INVARIANTS.md`](INVARIANTS.md) I12.

**Caveat to check first.** The ground state dominates the mixture and is
singlet-coupled, so raw `c_m` may be negative for nearly every molecule and
`sign(c)` degenerate. Histogram it before committing; use the median split or
the residual (§7).

### 2. Excitation-channel routing

Sort determinants by excitation rank relative to the reference. The diagonal
tells you *how much* weight sits at each rank; the off-diagonals tell you
*which pairs are phase-locked*.

Label on `reference↔doubles` coherence versus `singles↔singles` coherence —
the classic **static vs dynamic correlation** axis, and a genuinely important
chemical distinction. `A₁ = Σ_{i∈ref, j∈doubles}(|i⟩⟨j| + |j⟩⟨i|)` is Hermitian
and purely off-diagonal; use the linear combination `A₁ − cA₂`, not the ratio
(§2.1).

At fixed total coherence magnitude this is **orthogonal to the DoU confound** —
which is exactly the axis §3 says we need.

### 3. CI sign structure

`y = sign(⟨Φ_ref| ρ |Φ_HOMO→LUMO⟩)` — a single matrix element. Maximally
coherence-pure, and well-defined *only* because of the MO sign canonicalization
([`INVARIANTS.md`](INVARIANTS.md) I4).

**Risk:** leading double-excitation coefficients tend to a fixed sign relative
to the reference, which would collapse the label to one class. Screen the
balance before committing.

### 4. Localized bond-order alternation

Chemically the most legible (benzene's equal bond orders vs a polyene's
alternation), and in a localized basis bond order *is* an off-diagonal object.

**Blocker:** the operator must be the same for every molecule. All molecules
share a 16-qubit CAS(8,8) register, but qubit *k* means "the k-th frontier
orbital of *that* molecule" — a different physical object each time. A
localized-orbital label needs a canonical per-molecule orbital mapping, which
does not exist. Real engineering cost; parked.

### ★ 5. Logarithmic negativity across the α|β cut — **REOPENED 2026-08-05**

Rigorous coherence witness (exactly zero for any diagonal state) and the right
physical axis. It was excluded for one reason: it is a function of the
eigenvalues of a partial transpose — **nonlinear in ρ, hence outside the *single
neuron's* hypothesis class** (§2.1).

**That exclusion no longer applies.** The hybrid network's output is an arbitrary
classical function of `J₁` linear functionals of ρ (§2.3), so nonlinear
functionals are expressible whenever they factor through finitely many of them.

This makes it the best remaining candidate, because it does not share candidate
1's failure mode. `c = Tr(ρS²_od)` failed because `corr(c, D) = 0.919` — spin
coupling tracks unpaired-electron count across QH9, so a diagonal model predicts
it anyway. Negativity is a spectral property of a partial transpose rather than a
weighted count of determinants, so it has no structural reason to inherit that
correlation. Whether it escapes it in practice is a measurement, not an argument.

The old note that "candidate 5 collapses into candidate 1" was true only of its
*linear surrogate* (the spin-exchange operator, i.e. `S²_od`). With the wider
class we no longer need the surrogate. Tracked as
[`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) Q13.

---

## 6. Screen candidates before training anything

Since training only ever sees `R±` (§2.2), rank every candidate in seconds
without a single training run. Form `D = R₊ − R₋` and look at it:

```
    score(label) = ‖ offdiag(D) ‖_F  /  ( ‖ diag(D) ‖_F + ε )
```

- **large numerator** → real off-diagonal signal for the quantum model to grip
- **small denominator** → the diagonal/classical model has nothing
- **both tiny** → the label is unlearnable by *any* model in this class.
  Discard it before spending a training run.

**Cost.** `D` is 4,900 × 4,900 in the CAS(8,8) `S_z = 0` sector (= 70², the
`sector_dim` in the export file) — about 190 MB float64, a few seconds to build
from the eigenblocks. Normalize each `ρ_m` to trace 1 first (stored traces are
`1 − truncation_error`), and use class *means* if the split is not 50/50.

**Confirmation ablation.** Retrain with a Z-only pool. It must land at chance.

---

## 7. Killing the dataset-level confound

Two claims that are easy to blur, and the difference decides whether the result
survives review:

| Claim | Status |
|---|---|
| **A.** "No diagonal-only quantum model can learn this" | **Proven by construction** (§4). Free |
| **B.** "No classical model can learn this" | **Not implied by A.** Needs work |

Why A does not give B: a label can depend strictly on off-diagonals and still be
predictable across 1000 molecules, because *chemistry correlates the two*.
Triplet character tracks conjugation tracks DoU — countable from the formula.
A classical model would not be *reading* the coherence; it would be *knowing
chemistry*. Still gets the right answer.

**The fix**, using columns already in `results/coherence_share_kT0p1.csv`
(`DoU`, `n_aromatic_atoms`, `gap_Ha`, `largest_pi_atoms`,
`n_frontier_within_kT`) plus per-molecule occupation numbers:

1. Compute `c_m = Tr(ρ_m A_od)` for all 1000 molecules.
2. Regress `c_m` on those classical features.
3. Label on the **sign of the residual**, thresholded within composition strata.

The residual is orthogonal-by-construction to everything a classical descriptor
sees. It also makes the task harder — **verify the residual's dynamic range
stays well above the per-block `truncation_error`**, or you are training on
numerical noise.

---

## 8. Implementation roadmap

Ordered by information gained per hour spent.

| # | Step | Where | Notes |
|---|---|---|---|
| 1 | Compute `⟨S²⟩` and `⟨D⟩` per block | new `scripts/`, or a diagnostic in `qthermal/thermal.py` | Derivable post-hoc from `weights`, `amps`, `basis_indices` alone — no rerun needed |
| 2 | Implement the R± screening metric | new `scripts/screen_labels.py` | **Highest value per minute.** Ranks all candidates before any training |
| 3 | Add `--label offdiag:s2` + `--residualize` | `scripts/export_thermal_training.py` | Sits alongside the existing `h5:` / `csv:` label specs |
| 4 | Train on the winning label | `notebooks/pennylane/` or `tensor-network-testing/train_alg9.jl` | Both consume the export format |
| 5 | **Z-only-pool ablation** | same training path, restricted pool | Must land at chance. This is the result |
| 6 | Classical-descriptor baseline | new | Closes claim B (§7) |

**Prerequisite to watch.** The active-space adequacy finding (`RESEARCH_LOG.md`
2026-07-30): CAS(8,8) brackets high-DoU ground states fine (T = 0 edge slack
< 0.05), but the **kT = 0.1 thermal states are truncated** — high-DoU decile
median edge slack 0.167, 92% above 0.1. The excitable π* ceiling is real. If a
spin-coupling label depends on states pressed against the active-space
boundary, a CAS(8,10)/(8,12) confirmation run is needed first
([`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) Q3).

---

## 9. Implementation status

| Component | State |
|---|---|
| R± screening metric | ✅ `scripts/train_spin_comparison.py::screening_score` |
| Quantum-vs-classical harness on real states | ✅ `scripts/train_spin_comparison.py` |
| Spin-coupling label (candidate #1) | ✗ **tested, no advantage** (§5) |
| Objective aligned with the decision rule | ❌ **fails on these states** — `OPEN_QUESTIONS.md` Q11 |
| Neuron + logistic loss, paper-faithful | ✅ `notebooks/paper/logloss.ipynb` |
| Optimized rewrite (R± aggregation, sparse/fused Pauli, Chebyshev) | ✅ `notebooks/pennylane/logloss_pennylane.ipynb` |
| Equivalence between them | ✅ 12 tests, machine precision |
| Scaling to n = 10 | ✅ `docs/scaling_comparison.md` — Chebyshev crossover at n = 9 |
| Julia trainers (Alg. 8/9, Yao + ITensor) | ✅ `tensor-network-testing/` |
| Real thermal states as `{ρ_m, y_m}` | ✅ format done — `scripts/export_thermal_training.py` |
| Trained on real molecular data | ✅ `scripts/train_spin_comparison.py` (one neuron), `scripts/train_hybrid_spin.py` (the network). The *notebooks* still use Haar-random states and remain reference implementations, not the experiment |
| Z-only ablation | ✅ run, both architectures — and now a **theorem** at any depth ([`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md) §5.2, [`INVARIANTS.md`](INVARIANTS.md) I15) |
| Classical-descriptor baseline | ✅ in both harnesses (`descriptor_baseline`) — 93.3% on `⟨S²⟩`, 92.7% on `c`, 53.3% on the control |
| **A defensible label** | ❌ **the open problem** |

### The network (paper §VII.C), added 2026-08-05

| Component | State |
|---|---|
| Backpropagation from a classical layer into a quantum neuron | ✅ **derived** — [`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md). The paper leaves this open |
| All six of the paper's quantized activations + divided differences | ✅ `qnn/activations.py` |
| Quantum layer forward/backward, `J₁` eigendecompositions per epoch independent of M | ✅ `qnn/quantum_layer.py` |
| Composite gradient verified against finite differences | ✅ 44 tests across every activation, pool, depth, loss, temperature |
| Reduction to the paper's single-neuron trainer | ✅ asserted against the validated code, `tests/qnn/test_paper_equivalence.py` |
| Commuting-pool ⇒ `diag(ρ)`-only, at any depth | ✅ proved and asserted bit-identically |
| Trained on the 1000-molecule set | ✅ `scripts/train_hybrid_spin.py` — control 83.0% vs Z-only 51.0%; depth worth +10.7 points; **no advantage on the physical labels** |
| Controlled comparison vs the single neuron | ✅ identical label and split, 10 qubits: **93.0% vs 66.7%**, i.e. +26.3 points from the architecture. The single neuron separated quantum from classical by 0.4 points and sat at the descriptor baseline |
| Hypothesis class beyond linear functionals (§2.3) | ✅ available — **not yet exploited by any label** |

The last row is the actionable one. §5's candidate list was written for a model
that could only threshold a linear functional, so it rules out purity, entropy,
negativity and every ratio. Those constraints are gone (§2.3). Nothing in §5 has
been re-examined under the wider class, and the first candidate to revisit is
§5's candidate 5 — logarithmic negativity across the α|β cut, a rigorous
coherence witness discarded *only* because it is nonlinear in ρ.
Tracked as [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) Q13.
