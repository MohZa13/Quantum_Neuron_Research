# Decision log (ADRs)

*Choices a future reader would otherwise re-litigate. Each entry: the decision,
the alternatives, why, and its current status.*

**Status values:** `ACTIVE` (holds today) · `SUPERSEDED` (replaced — links
forward) · `PROVISIONAL` (working assumption, revisit) · `RETIRED` (no longer
relevant).

**Add an entry when** you make a choice that is not forced by the code and that
someone could reasonably make differently. Reverse-engineering intent from a
diff is expensive; writing three sentences now is not.

Template: [`templates/DECISION.md`](templates/DECISION.md).

---

## D1 — CASCI on stored Kohn–Sham orbitals; no SCF, no geometry relaxation

**Status:** `ACTIVE` · 2026-07 · Enforced by [`INVARIANTS.md`](INVARIANTS.md) I2

**Alternatives:** re-converge RHF orbitals per molecule; run CASSCF; relax
geometries.

**Why.** The dataset's value is being *faithful to QH9's own electronic
structure*. Re-converging orbitals would make each record a different physical
object from the source dataset — silently, and in a way no downstream consumer
could detect. Reproducibility against a public dataset beats variational
quality here.

**Cost accepted.** CASCI on KS orbitals is not variationally optimal. Stated
openly in `qthermal/README.md` rather than hidden.

---

## D2 — Eigenblock (`p`, `civecs`) as the universal state representation

**Status:** `ACTIVE` · Enforced by [`INVARIANTS.md`](INVARIANTS.md) I3

**Alternative:** store dense ρ.

**Why.** Dense is not merely wasteful, it is impossible: 4,900² in the
determinant basis and 65,536² on the qubit register, per molecule per
temperature. The eigenblock is exact, tiny, and every downstream operation
(trace distance, Pauli expectations, MPS construction) has a formulation that
consumes it directly.

**Discipline it forces.** Every new analysis must be expressible as a
contraction over eigenblocks. That constraint has been productive — the 248
Pauli features evaluate in under a second because of it.

---

## D3 — Three solvers behind one `Protocol`, with *different* contracts

**Status:** `ACTIVE` · `qthermal/diagonalize.py`

**Alternative:** one solver; or a uniform interface that pretends the full
spectrum always exists.

**Why.** The regimes are genuinely different — dense is exact but capped at
dim ≈ 70,000; Krylov reaches dim 853,776 but only certifies a low-kT window;
the non-interacting reference has a closed form and needs no diagonalization at
all. Pretending otherwise would mean either lying about `evals` or crippling
the dense path.

**Cost accepted.** Readers must handle `evals` being absent
([`INVARIANTS.md`](INVARIANTS.md) I5). Made explicit in the `TruncatedEnsemble`
docstring and enforced by tests. The seam is what lets a Phase-2 sampling
backend (Q9) land without touching anything upstream.

---

## D4 — Certified truncation everywhere, never silent

**Status:** `ACTIVE` · Enforced by [`INVARIANTS.md`](INVARIANTS.md) I10

**Alternative:** drop negligible weight quietly.

**Why.** Every downstream number should carry its own error bar. Concretely:
the kT = 0.25 runs discard up to 1.7×10⁻² of the thermal weight when capped —
knowing that is the difference between a usable dataset and a misleading one.

Each solver supplies a bound of the appropriate kind: dense exact, Krylov a
rigorous counting bound, and (future) DMRG variational / METTS sampling error
bars.

---

## D5 — Wire ordering is a per-consumer choice, not a global constant

**Status:** `ACTIVE` (revised 2026-07-27) · [`INVARIANTS.md`](INVARIANTS.md) I12

**History.** Interleaved was originally expected to win everywhere
(same-orbital α/β pairs adjacent ⇒ shorter-range structure). Measurement said
otherwise: blocked reads ~10× more connected-ZZ signal *and* gives ~2× smaller
MPS bond dimensions.

**Now.** Blocked for features and MPS; interleaved for coherence-label
operators, where `S²_od` becomes string-free and 4-local. The ordering is
recorded in every output file's `/meta`, and `encode_run` refuses to mix
conventions in one file.

**Debt.** Two docstrings still assert the old claim — Q7.

---

## D6 — Label aggregation (R±) instead of parameter-shift autodiff

**Status:** `ACTIVE` · `docs/classifier_optimization.md`

**Alternative:** the originally proposed PennyLane QNode + parameter-shift
gradients.

**Why.** The loss is a *spectral* function of H(ω), not a circuit expectation —
parameter-shift does not apply naturally. Meanwhile the loss's linearity in
each ρ_i makes `R± = Σ_{y=±1} ρ_i` an exact reduction, giving per-epoch cost
**independent of dataset size**. Measured 174× at 1,000 samples; 256× less
static memory.

**Second-order consequence — this is the important part.** The same linearity
that makes training cheap also *pins down the hypothesis class* to thresholds
on `Tr(ρH)`. That constraint is what turns label design from guesswork into a
solvable problem ([`QUANTUM_NEURON.md`](QUANTUM_NEURON.md) §2).

---

## D7 — The paper notebook is preserved verbatim; corrections live outside it

**Status:** `ACTIVE` · [`INVARIANTS.md`](INVARIANTS.md) I6, I7

**Alternative:** fix the notebook in place.

**Why.** `notebooks/paper/logloss.ipynb` is the *reference* — its value is
being exactly what the paper describes, including the extraneous `1/T` factor
in `dfj`. Corrections live in the optimized implementations and in
`figures/quantum_training_impls.py::run_original`, which applies the fix so
comparisons are matched.

**Cost accepted.** Cell order becomes load-bearing, because
`notebook_test_utils` executes cells by index. Documented as an invariant
because the failure mode looks numerical.

---

## D8 — Two dependency tiers: loose ranges + a lock file

**Status:** `ACTIVE` · `pyproject.toml` / `requirements.lock`

**Why.** `pyproject.toml` says what the code needs to import and run;
`requirements.lock` is a freeze of the environment that produced every
committed result and figure. **Reproducing published numbers means the lock
file.** PennyLane is capped to a single minor because it breaks API across
minors and `pennylane-lightning` must match it exactly.

---

## D9 — Flat layout, explicit package list

**Status:** `ACTIVE` · `pyproject.toml`

**Why.** With this many top-level directories (`tests`, `scripts`, `benchmarks`,
`figures`, `data`, `docs`, `notebooks`, `results`, `Papers`), setuptools
auto-discovery refuses to disambiguate. So `packages = ["qthermal", "qnn"]` and
`py-modules = ["notebook_test_utils"]` are listed explicitly.

**Consequence.** `qthermal` and `qnn` are installed editable, so imports work
from any working directory — **no `sys.path` manipulation in new files**. `scripts/`,
`benchmarks/`, `figures/` run as plain scripts, so sibling imports within those
directories resolve via the script's own directory.

---

## D10 — The single-determinant baseline is retired, not deleted

**Status:** `ACTIVE` · 2026-07-13 · `data/README.md`

**Why.** `data/build_slater.py` produced a mean-field, zero-temperature
representation with no explicit electron interaction — superseded by
`qthermal` for this project's goals, and its 284 GB output was corrupted by the
AO bug and deleted. The **builder** is kept, bug fixed and verified, because it
is the regeneration route if an interacting-vs-mean-field comparison is ever
wanted. Regeneration instructions live in `data/qh9_raw_sqlite_audit.md`.

---

## D11 — Documentation framework: append-only logs + per-directory READMEs

**Status:** `ACTIVE` · 2026-08-05

**Alternative:** one large README; or docs generated from docstrings.

**Why.** This is long-running research read by agents with **no context**. Three
properties matter more than concision: (a) a deterministic entry point
(`AGENTS.md`), (b) provenance for every artifact, and (c) **append-only**
knowledge — findings and decisions accumulate rather than being overwritten, so
history survives context resets. Rewriting a "current status" section destroys
exactly the information a future reader needs.

Docstrings stay the authority on *how code works* — they are excellent here and
are not duplicated. `docs/` covers what docstrings structurally cannot: cross-file
architecture, artifact provenance, negative results, and open questions.

---

## D12 — Build the paper's §VII.C hybrid architecture, not its §VII.B quantum observable network

**Status:** `ACTIVE` · 2026-08-05 · Derivation:
[`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md)

**Alternatives.** (a) The fully quantum observable network of paper §VII.B,
where every layer stays operator-valued. (b) Stay with the single neuron and
add expressivity through a larger Pauli pool.

**Why.** §VII.B's gradients are *compositions* of Fréchet-derivative
superoperators (Eqs. G59–G64). The paper's own Remark 5 notes that the
intermediate `B^(ℓ)` are non-local so standard Hamiltonian simulation does not
apply, and Remark 6 leaves the existence of an efficient reverse-mode algorithm
open. §VII.C collapses each activation observable to a *scalar* before the next
layer, so exactly **one** Fréchet derivative appears per parameter and never a
composition. That is the whole reason it is tractable, and it is why the paper
itself calls it "perhaps the most interesting such instantiation".

Option (b) does not help: the single-neuron hypothesis class is *thresholds on
linear functionals of ρ* no matter how large the pool
([`QUANTUM_NEURON.md`](QUANTUM_NEURON.md) §2.1). Classical depth is what buys a
nonlinear decision boundary in the activation variables.

**Cost accepted.** The loss is no longer applied by functional calculus, so it
is no longer linear in ρ and the `R±` optimization does not survive (D13).

---

## D13 — Re-form the δ-weighted aggregate every epoch rather than abandoning aggregation

**Status:** `ACTIVE` · 2026-08-05

**Alternatives.** Per-sample gradients (the naive route once `R±` fails);
mini-batching.

**Why.** Depth breaks `R±` — the hybrid loss is nonlinear in each ρ_m, so the
dataset cannot be collapsed into two fixed matrices before training. But the
*quantum layer* is still linear in ρ, so the sum over samples still collapses,
just into a per-neuron matrix `Rᵢ = (1/M) Σ_m δ_{m,i} ρ_m` that moves with the
parameters. Re-forming it costs one GEMM per epoch and preserves the property
that actually matters: **`J₁` eigendecompositions per epoch, independent of
dataset size.**

Mini-batching was rejected for the same reason — the expensive part of an epoch
does not scale with how many samples participate, so a mini-batch epoch costs
what a full-batch epoch costs and takes a noisier step.

**Measured** (2026-08-05): at K = 1024, J₁ = 8, an epoch costs 3.6 s at M = 60
and 3.4 s at M = 300 — a 5× dataset at no measurable cost, the difference being
within run-to-run noise.

---

## D14 — Gauss–Legendre divided differences instead of an epsilon-threshold

**Status:** `ACTIVE` · 2026-08-05

**Alternative.** The existing single-neuron code's `if |Δλ| < 1e-10: use φ'(λ)`
(`figures/quantum_training_impls.py:91`, `scripts/train_spin_comparison.py:169`).

**Why.** That patch is safe where it is used — it only ever fires on exact
degeneracy — but it leaves an error band: just above the threshold the quotient
still has almost no significant digits, and just below, the substitution is
wrong by `O(ε·φ'')`. Since the divided difference *is* the nonlinearity of a
quantized neuron, and since the whole point of this package is that its gradient
is a derivation to be checked rather than assumed, the error there should be
below the finite-difference tolerance the tests use, not comparable to it.

The replacement is the paper's own identity, Eq. (A6):
`φ^[1](a,b) = ∫₀¹ φ'(sa+(1−s)b) ds`, evaluated by 8-node Gauss–Legendre below
`|a−b| = 10⁻³` and by the difference quotient above. Both branches then carry
`≤ 10⁻¹¹` absolute error; measured, and asserted in
`tests/qnn/test_activations.py`.

**Cost accepted.** The switch is *tight* (10⁻³, not the ~1 that would maximise
accuracy) because the quadrature branch costs 8 evaluations of `φ'` per matrix
entry: at K = 1024 a wide band would cost ~1 s/epoch against ~5 ms. Accuracy
past 12 digits is not worth an epoch budget.

**Not retrofitted** to the single-neuron code, which is validated against the
paper notebooks by 12 equivalence tests and should not be perturbed. The two
agree to `1e-8`; `tests/qnn/test_paper_equivalence.py` asserts it and attributes
the residual to exactly this difference.

---

## D15 — `relu` is refused as an activation observable

**Status:** `ACTIVE` · 2026-08-05

**Alternative.** Allow it and document a caveat.

**Why.** ReLU is not differentiable at 0, so `φ^[1](a,b)` is undefined whenever
an eigenvalue of the pre-activation operator sits there. For a pool of traceless
Pauli strings that is not a corner case — it is where the spectrum concentrates
at initialization. A caveat in a docstring would be read after the bug.

This is also, in retrospect, why the paper quantizes *softplus* and *GReLU*
rather than ReLU itself: both are the smooth approximations whose derivatives
(`f_T` and `Φ(x/T)`) are the objects its Theorems 6 and its Eq. (96) actually
need. `get_activation("relu")` raises; `get_activation("relu", classical=True)`
succeeds, because a classical layer needs no operator derivative.

---

## D16 — Permanent in-code timers in QThermalMPS, not a one-off profiling script

**Status:** `ACTIVE` · 2026-08-10

**Alternatives:** a throwaway benchmark script replicating the ladder loop;
Julia's sampling `Profile`; no instrumentation (rely on rung wall clocks).

**Why.** A separate script drifts from the real implementation and would have
re-measured a copy, not the pipeline (the rung-level `seconds` field already
existed and still left a wrong hypothesis standing for a month — "expansion
dominates" survived precisely because the split within a rung was never
measured). `@timeit` sections cost ~1 μs against stages that run for seconds,
so they stay on permanently: every future run can be asked where its time went
after the fact (`--profile 1`), including runs that were not launched as
benchmarks. TimerOutputs was already in the dependency closure via ITensors.

**Cost accepted.** One more direct dependency; ~15 sections' worth of noise in
the source. The synthetic `--warmup` (default only when profiling) adds ~3.5
min to profiled runs so the first molecule's numbers are compilation-free.

---

## D17 — Non-stock KrylovKit solver settings are the package default, not opt-in

**Status:** `ACTIVE` · 2026-08-10

**Alternatives:** keep KrylovKit's stock `exponentiate` defaults and expose a
flag; hardcode a fixed tolerance.

**Why.** The stock defaults (Arnoldi, no eager exit, tol 1e-12) cost a
measured 4.9x on the stage that is 90% of the pipeline, for accuracy that
changes at the 1e-12 level against a 1e-8 truncation floor — strictly below
the error budget at any cutoff, because the default tolerance is *derived*
(`clamp(cutoff/10, 1e-12, 1e-8)`), not fixed. A correctness-neutral 5x that
must be remembered per run would be forgotten exactly once per important run.
`--solver-tol none` recovers stock behavior for solver-suspicion debugging;
`updater_kwargs = (;)` does the same in the API.

**Cost accepted.** `issymmetric = true` is an assertion about H_eff (real
symmetric), not a measurement — it holds for every Hamiltonian this package
can build (real `(h1, g)`, real MPS), and would corrupt results silently if a
complex/non-symmetric operator were ever introduced. Guarded by a comment at
the definition and by the dense-reference evolve tests.
