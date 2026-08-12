"""The deck's content: one entry per slide, plus its presenter notes.

Layout lives in ``build_deck.py``; this file is the script of the talk and is
meant to be readable on its own.  A slide is a dict:

    kicker   small capitalised rubric above the title
    title    the slide's claim
    layout   which builder in build_deck.py arranges it
    ...      layout-specific fields
    notes    presenter notes (the detail that does not fit on the slide)

Numbers quoted here are reproduced by ``build_cache.py`` / ``figures.py`` from
``results/``; see ``Papers/presentation_references.md`` for sources.
"""
from __future__ import annotations

TITLE = "Thermal States as Training Data for a Quantum Neuron"
SUBTITLE = ("Building a mixed-state dataset for the Fermi–Dirac machine — "
            "and what it taught us about looking for quantum advantage")
VENUE = "Group meeting  ·  August 2026"

SLIDES: list[dict] = [

    # ---------------------------------------------------------------- title
    dict(
        layout="title",
        title=TITLE,
        subtitle=SUBTITLE,
        venue=VENUE,
        strapline="A sixteen-wire register, half filled: the object the whole "
                  "pipeline delivers.",
        notes="""
Plan for the next twenty minutes. Three parts.

One: the machine we want to feed — a neuron whose input is a quantum state,
not a feature vector — and exactly what it can and cannot express. That
constraint drives everything else.

Two: the data structure. This is the bulk of the engineering and the part I
most want feedback on. The object is an exact interacting Gibbs state, stored
in a form that never materializes a density matrix, and it converts losslessly
into whichever representation a downstream consumer wants.

Three: what happened when we trained on it. The headline is a negative result
that I think is more interesting than the positive one would have been, and it
generalizes beyond our dataset — it is a statement about what it takes for
molecular data to separate a quantum model from a classical one.

I will flag the source dataset only where it matters. Nothing in the pipeline
is specific to it, and we are actively looking at alternatives.
""",
    ),

    # ============================================ PART I — WHAT WE ARE FEEDING
    dict(
        layout="eq_bullets",
        kicker="PART I  ·  THE MACHINE",
        title="A neuron whose input is a quantum state",
        equations=[
            ("neuron",
             "The neuron applies a scalar function to a trainable Hamiltonian by "
             "functional calculus — diagonalise $H(\\theta)$, map the eigenvalues, "
             "rebuild — and reads the result against the input state."),
            ("hypclass",
             "With a hard threshold on the output, the hypothesis class is exactly "
             "this: half-spaces in the operator inner product. Nothing else is "
             "reachable by one neuron, however large the pool."),
        ],
        bullets=[
            "**The state is the input.** There is no encoding circuit and no "
            "feature extraction step: $\\rho$ enters the trace directly. That is "
            "the property we are trying to exploit.",
            "**$\\varphi_T$ is a quantized activation.** Temperature $T$ smooths "
            "a step function into a differentiable one, exactly as a sigmoid does "
            "classically — but applied to a spectrum.",
            "**The class is linear in $\\rho$.** Everything that follows comes "
            "from that one line: what a label may be, why training cost is "
            "independent of dataset size, and a free screening test we use later.",
        ],
        figure="fig_activation.png",
        figure_caption="One knob interpolates between a hard decision and a soft one.",
        notes="""
The model is from He, Liu and Wilde, *Fermi–Dirac machines as quantizations of
neurons* (arXiv:2605.24386). Their construction: take a classical activation,
find its "quantization" as an operator function, and you get a neuron that acts
on density matrices.

Say clearly for a QI audience: phi applied to an operator means the functional
calculus. Diagonalize B, apply phi to each eigenvalue, rotate back. It is not
phi applied entrywise, and that distinction will matter twice later — once for
the gradient, once for the ablation theorem.

Why the linearity matters so much. Tr(rho H) is linear in rho. So:
(a) the decision boundary is a hyperplane in the space of density matrices;
(b) a label that is a *nonlinear* functional of rho — entropy, purity,
    negativity, a ratio of two expectations — is outside the class by
    construction, not by difficulty;
(c) the training loss depends on the dataset only through the two class sums
    R_plus and R_minus, which is why an epoch costs the same at M = 10 and
    M = 1000. Measured 174x speed-up at 1,000 samples.

Point (b) is the single most useful fact for anyone designing a label. Ratios
break linearity; differences do not.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART I  ·  THE INPUT SPECIFICATION",
        title="Why thermal states are the natural training data",
        equations=[
            ("gibbs",
             "The canonical state of a Hamiltonian: the maximum-entropy state at "
             "fixed mean energy. One parameter, $\\beta$, sweeps it from the "
             "ground state to the maximally mixed state."),
        ],
        bullets=[
            "**Mixed states are the interesting case.** A pure-state classifier "
            "can be simulated by sampling amplitudes; the mixedness is what makes "
            "$\\rho$ a genuinely operator-valued input.",
            "**Temperature is a purity dial.** Low $\\beta^{-1}$: one eigenstate "
            "dominates. High: everything is uniform. In between the state has "
            "structure that is neither trivially pure nor trivially flat.",
            "**They are physically meaningful, not synthetic.** The source paper "
            "trains on Haar-random states with random target operators. We wanted "
            "inputs where the label could mean something.",
        ],
        figure="fig_diagnostics.png",
        figure_caption="1,000 production ensembles: mixing tracks the "
                       "electronic gap, and the states sit far from the matched "
                       "free-fermion Gibbs state.",
        notes="""
Justify the temperature choice honestly. We work at kT = 0.1 Hartree. That is
about 3 x 10^4 K — nowhere near a laboratory temperature. It is chosen for the
*structure* of the resulting ensemble, not for physical realism, and I would
rather say that than dress it up.

The reason is the spread. At kT = 0.1 the weight on the lowest level ranges
16-98% with median 50% (the histogram two slides on): strongly mixed, but
sharply different between molecules, which is what a labelled dataset needs.
The left panel here is why - mixing tracks the electronic gap at r = -0.82, so
the ensemble inherits the molecule's own structure. We also ran 0.025
(uniformly cold, nearly pure — no variation) and 0.25 (uniformly hot — also no
variation, and the truncation gets expensive: percent-level discarded weight).

If asked "why not a physical temperature": at 300 K the Gibbs state of an
electronic Hamiltonian is the ground state to twenty digits. There is no
ensemble to learn from. The interesting regime for this experiment is set by
the electronic gap, not by the thermometer.
""",
    ),

    # =========================================== PART II — THE DATA STRUCTURE
    dict(
        layout="dictionary",
        kicker="PART II  ·  THE OBJECT",
        title="What the pipeline is actually doing, in QI terms",
        rows=[
            ("A single-particle orbital", "one fermionic mode  →  one wire"),
            ("A Slater determinant",
             "a computational-basis state of fixed Hamming weight"),
            ("Two-electron integrals", "the two-body coupling tensor of the Hamiltonian"),
            ("An active space",
             "a restriction to a sub-register of modes around the Fermi level"),
            ("A frozen core",
             "spectator modes in a fixed product state: a dressed one-body "
             "term plus a scalar"),
        ],
        panel_title="The input a source must provide",
        panel_bullets=[
            "a converged **one-particle Hamiltonian** and overlap matrix,",
            "**nuclear geometry**, so the two-body tensor can be built,",
            "nothing else.",
        ],
        panel_note="Everything downstream is agnostic to where those came from. "
                   "Our runs use a public dataset of ~130k small organic molecules "
                   "(QH9), but that choice enters only the first stage, and its "
                   "limitations — discussed at the end — are a reason we are "
                   "looking at other sources.",
        equation="sector",
        equation_caption="The register is $Q=2n$ wires, but particle number and "
                         "$S_z$ are conserved, so the state lives in one symmetry "
                         "sector of this dimension. Everything is done inside it.",
        notes="""
This slide is the translation layer. The audience knows quantum information,
not quantum chemistry, and every chemistry word in this talk has a one-line QI
meaning.

Two things worth stressing.

First, "active space" is just a subregister. We keep n modes straddling the
Fermi level and freeze the rest in a filled product state. Freezing is exact
bookkeeping — it contributes a mean-field-like dressing of the one-body term
plus a constant — but the *restriction* is an approximation, and it is the only
one in the construction. Part IV is about what it costs.

Second, we never leave the symmetry sector. Working in the (N, S_z) sector
turns a 2^16 = 65,536-dimensional problem into a 4,900-dimensional one, exactly.
That is a factor of 13 in dimension and 175 in matrix area, for free, and it is
why exact diagonalization is feasible at all here.

If someone asks why not just use the full Fock space: nothing forbids it, but
the Hamiltonian is block diagonal and the blocks we do not need are the
overwhelming majority of the space.
""",
    ),

    dict(
        layout="eq_stack",
        kicker="PART II  ·  THE HAMILTONIAN",
        title="One approximation, made explicit",
        equations=[
            ("hamiltonian",
             "The second-quantized Hamiltonian restricted to the active modes. "
             "The one-body term carries the frozen-core dressing; the two-body "
             "tensor is a Coulomb kernel between mode densities, computed from "
             "the geometry. Within the sector this operator is diagonalized "
             "exactly — no ansatz, no variational error."),
        ],
        cards=[
            ("The only approximation",
             "Restricting to $n$ frontier modes. Nothing else in the construction "
             "is inexact: no self-consistency loop, no basis fitting, no "
             "perturbation theory."),
            ("Why exact, not variational",
             "This is *training data*. A defect in it produces no error message — "
             "it silently degrades the physics the classifier learns. So every "
             "stage is cross-checked against an independent implementation and "
             "every truncation carries a stored bound."),
            ("What that buys",
             "The Hamiltonian is built twice, once through library machinery and "
             "once from a literal transcription of the equation above, and the two "
             "must agree elementwise, to eight decimal places in Hartree, before "
             "any state is produced."),
        ],
        notes="""
Do not spend long on the equation itself — this audience reads second
quantization fluently. Spend the time on the discipline argument in the middle
card, because it is the thing that made this project slow and the thing that
saved it.

The concrete story, if there is time: early in the project a basis-ordering
convention was applied twice. Nothing crashed. Every validation gate passed,
because the gates were physicality windows loose enough to admit a one-Hartree
error. We built 284 GB of derived data on top of it before an independent
recomputation caught it, and all of it was discarded. The lesson is not "be
careful" — it is that a validation gate which passes corrupt data is worse than
no gate at all, because it manufactures confidence. Every invariant in the
repository now carries a verification command rather than a prose assertion.

On "no self-consistency loop": we deliberately run the correlated calculation
on the orbitals the source dataset stores, rather than re-converging our own.
Re-converging would make each record a different physical object from the
public dataset, silently, in a way no downstream consumer could detect.
Reproducibility against the source beats variational quality here. It is a
documented, deliberate deviation, not an oversight.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART II  ·  THE CENTRAL DATA STRUCTURE",
        title="The eigenblock: a state stored as its own spectrum",
        equations=[
            ("eigenblock",
             "The stored object is the $m\\times\\dim$ matrix of retained "
             "eigenvectors together with their Boltzmann weights. The density "
             "matrix is never written down."),
        ],
        bullets=[
            "**Dense is not merely wasteful, it is impossible.** $4{,}900^2$ per "
            "molecule per temperature in the sector basis, $65{,}536^2$ on the "
            "full register. At $m\\approx10^3$ the eigenblock is three orders of "
            "magnitude smaller and *exact*.",
            "**Truncation is certified, never silent.** Levels are kept in energy "
            "order until the cumulative weight reaches $1-10^{-6}$; the discarded "
            "weight is stored per state as its error bar. Retained weights are "
            "not renormalized away.",
            "**It forces a useful discipline.** Every downstream analysis has to "
            "be expressible as a contraction over the block. That constraint is "
            "why the 248 Pauli expectation values evaluate in under a second per "
            "molecule.",
        ],
        figure="fig_eigenblock.png",
        figure_caption="Cumulative Boltzmann weight against levels kept, and the "
                       "resulting spread of ground-state weight.",
        notes="""
This is the slide I would most like reactions to, because everything else is
downstream of it.

The point in QI language: we store the state in its eigenbasis, which is the
Schmidt basis of its own purification. That single choice is what makes
everything else cheap, and it is why the next slide's conversion is free rather
than an approximation.

On the retention rule: we keep the smallest energy prefix whose cumulative
Boltzmann weight is at least 1 - 10^-6, subject to a storage cap. The weights
are *not* renormalized afterwards. That means every stored state has trace
slightly below one — between 0.967 and 1.000 in the production set — and the
deficit is precisely the error bound. Anyone consuming these has to normalize
before training, because the pool contains the identity and the truncation
error would otherwise become a free feature correlated with molecular
complexity. That is a real trap; it is written into the invariants.

Numbers: the production set is 1,000 molecules at CAS(8,8), kT = 0.1 Ha,
13.3 hours of wall time, zero failures across all 1,000. Recorded discarded
weight 10^-4 to 10^-3.
""",
    ),

    dict(
        layout="three_faces",
        kicker="PART II  ·  ONE OBJECT, THREE CONSUMERS",
        title="The same state, in whichever form the consumer wants",
        centre=("The eigenblock", "$(p,\\;V)$", "$m$ weights, $m$ orthonormal rows"),
        faces=[
            ("Pauli feature vector",
             "$\\mathrm{Tr}(\\rho P_j)$ evaluated as a contraction over the block. "
             "For a weight-$\\leq 2$ basis on 16 wires this is 248 numbers per "
             "state — and for a single neuron it is a *sufficient statistic*.",
             "seconds per molecule"),
            ("Purification MPS",
             "The block is already the Schmidt decomposition across the "
             "system–ancilla cut, so the conversion is a tensor-train "
             "factorization of the system leg only.",
             "exact, then truncated with a bound"),
            ("Dense $\\rho$ on a sub-register",
             "Project onto the $K$ most-populated basis states and materialize a "
             "$K\\times K$ matrix, for models that need the operator itself. "
             "$K=1024$ retains 99.8% of the weight (median).",
             "the form the trainers consume"),
        ],
        note="No conversion loses information silently: each carries its own "
             "error term, and the untruncated round trip reproduces $\\rho$ to "
             "machine precision.",
        notes="""
This is the "data structures" heart of the talk. One stored object, three
exact-or-certified views, chosen by the consumer rather than fixed in advance.

Why three and not one. The single neuron only ever sees the state through
Tr(rho P_j), so for that model the 248-number feature vector is not a
compression — it is a sufficient statistic, and the dataset interfaces to
training with no loss at all. The hybrid network is nonlinear in rho, so it
needs the operator; that is the third face. Tensor-network backends want the
MPS; that is the second.

Worth flagging the wire-ordering subtlety if anyone asks. Jordan-Wigner needs
a wire order, and the best order is per-consumer: blocked (all alpha modes,
then all beta) gives about 2x smaller MPS bond dimensions and reads roughly 10x
more connected two-mode signal; interleaved (alpha and beta of the same orbital
adjacent) is what makes the spin-exchange operator string-free and 4-local.
We originally predicted interleaved would win everywhere for the MPS and
measurement said otherwise. It is recorded in the decision log as a reversal.
""",
    ),

    dict(
        layout="steps_eq",
        kicker="PART II  ·  A DESIGN CHOICE, EXPLAINED",
        title="Why we diagonalize exactly to build the purification",
        equations=[
            ("purification",
             "A purification of $\\rho$ on system $\\otimes$ ancilla. Because the "
             "eigenvectors are orthonormal, the $\\sqrt{p}$-weighted eigenblock "
             "*is already* the Schmidt decomposition across that cut."),
            ("mpsbound",
             "The partial trace contracts the trace norm, so the MPS's own "
             "per-bond discarded mass bounds the error in $\\rho$ itself."),
        ],
        steps=[
            ("The ancilla bond is free",
             "It equals the thermal rank $m$ exactly, with singular values "
             "$\\sqrt{p_k}$: no SVD and no truncation on that cut. The expensive "
             "half of a generic purification is handed to us by the "
             "diagonalization we already did."),
            ("Only the system leg is factorized",
             "Successive thin SVDs across the $Q$ wires give left-canonical cores; "
             "bonds are capped and every discarded mass recorded."),
            ("The alternative would be variational",
             "Imaginary-time evolution or purification DMRG reaches larger "
             "registers, but its error is variational rather than certified."),
        ],
        note="**The trade this makes.** Exactness costs reach — the conversion "
             "is bounded by $m\\cdot 2^{Q}$, the wall we hit in Part IV.",
        notes="""
The design-choice slide. Two questions to answer: why exact diagonalization,
and what it costs.

Why. A purification of a general mixed state normally requires you to find a
Schmidt decomposition, which is the expensive part. We get it for free, because
we stored the state in its eigenbasis and the eigenvectors are orthonormal.
So the ancilla bond is exactly m with singular values sqrt(p_k) — no truncation
error is introduced there at all. Only the Q physical wires need tensor-train
SVDs. Untruncated, the round trip reproduces rho to machine precision; we test
that.

What it costs. The conversion materializes the encoded block, which is
m x 2^Q. At Q = 16 that is fine. At Q = 20 it is not, and a sector-basis
contraction that never materializes the full register would be needed — or,
better, a method that produces the MPS directly and never diagonalizes at all.
That is exactly the Part IV proposal.

If asked about bond dimensions: measured on real production blocks, physical
bonds stay well below the ancilla bond, which is what makes the object cheap to
hold. The thermal state satisfies a quasi-area law in this regime — the
purification's cut entropy exceeds the eigenvectors' by at most ln m.
""",
    ),

    dict(
        layout="two_cards_eq",
        kicker="PART II  ·  CERTIFICATION",
        title="Three solvers behind one interface, with different contracts",
        cards=[
            ("Dense diagonalization",
             "Full eigendecomposition of the sector matrix. Exact spectrum, exact "
             "tail weights, about a minute per molecule. Memory ends the method "
             "near dimension $7\\times10^{4}$."),
            ("Matrix-free Krylov",
             "Converges only the low-energy window the temperature needs. The root "
             "count is escalated until a counting bound *certifies* the discarded "
             "weight; demonstrated at dimension 853,776. Correct at any "
             "temperature, but the window widens with $T$ and escalation "
             "eventually caps out."),
        ],
        equations=[
            ("tail",
             "The certificate. Everything above the $m$ retained levels is bounded "
             "by the exactly-known part of the spectrum plus a counting bound on "
             "the rest — so a solver that never sees the full spectrum can still "
             "state a rigorous error."),
        ],
        note="**Why an interface and not one solver.** The regimes are genuinely "
             "different, and pretending otherwise would mean either lying about "
             "what a solver returns or crippling the exact path. A sampling "
             "backend satisfies the same contract without touching anything "
             "upstream — which is the escape route in Part IV.",
        notes="""
Three implementations, one protocol, deliberately different contracts. The
dense solver stores the whole spectrum; the Krylov solver cannot and does not,
so consumers must not assume it exists. That is enforced by tests rather than
by convention.

The third implementation, not on the slide: a non-interacting reference solver.
Setting the two-body tensor to zero makes the many-body Hamiltonian diagonal in
the Slater basis built from the one-body eigenvectors, so its spectrum is an
outer sum and needs an n x n diagonalization rather than a dim x dim one. We
use it for the "how far from free-fermion is this state" audit that appears in
Part III. It also carries a caveat: the reference omits mean-field repulsion,
which inflates the distance for compact molecules, so we do not use that
distance as a label.

The certificate is worth a sentence because it is the thing that makes the
iterative path usable at all. Without it, "we converged 500 roots" is not a
statement about the state; with it, "the discarded Boltzmann weight is below
epsilon" is.
""",
    ),

    # =========================================== PART III — THE LABEL PROBLEM
    dict(
        layout="figure_hero",
        kicker="PART III  ·  THE FIRST EXPERIMENT",
        title="It works: the state's features determine the frontier gap",
        figure="fig_gap_training.png",
        figure_caption="1,000 molecules · 248 thermal-state Pauli features · "
                       "label: below- vs above-median HOMO–LUMO gap, balanced · "
                       "tested on 300 molecules never seen in training.",
        equations=[
            ("features",
             "The input is the vector of Pauli expectation values of the thermal "
             "state — nothing about the molecule's identity, formula or geometry "
             "is provided."),
        ],
        stats=[("94%", "held-out accuracy"),
               ("50%", "chance, by construction"),
               ("300", "molecules never trained on")],
        notes="""
Take this at face value for one slide. It is a real result and it is not
trivial: from the Pauli expectation values of a thermal state, with no
identifying information about the molecule, the classifier recovers which side
of the median that molecule's frontier gap falls on, and it generalizes to
molecules it has never seen. The loss curve is well behaved, there is no
overfitting gap worth mentioning.

The HOMO-LUMO gap is a natural first label. It is the most-used scalar in
molecular machine learning, it is a genuine electronic-structure property, it
is balanced by construction under a median split, and — importantly — it is
*physically upstream* of the thermal state: the gap is what sets how mixed the
ensemble is. So a classifier reading the state ought to be able to recover it.
It did.

Then set up the next slide: the question is not whether it works. The question
is what it is reading.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART III  ·  THE DIAGNOSIS",
        title="What is the classifier actually reading?",
        equations=[
            ("dephasing",
             "Let $\\Delta$ be the dephasing channel — erase every off-diagonal "
             "entry, keep the populations. $\\Delta(\\rho)$ is precisely *what a "
             "classical model sees*. Subtracting the two expectations leaves the "
             "coherences alone, and one line of algebra gives the whole recipe: "
             "the quantum content of any linear label lives in $A_{\\mathrm{od}}$."),
        ],
        bullets=[
            "**Decompose the feature weight.** Of the 248 expectation values, "
            "99.6% of the squared weight is single-mode occupation and its "
            "products — which orbitals are filled. 0.39% is genuine occupation "
            "*covariance*. 0.026% is hopping coherence.",
            "**So the model is reading populations.** The features are dominated "
            "by $\\mathrm{diag}(\\rho)$, and a diagonal — that is, classical — "
            "model has access to all of it.",
            "**This is a property of the data, not of the model.** The same "
            "machine reaches 94–96% on the source paper's Haar-random states, "
            "where there is no dominant diagonal to hide behind.",
        ],
        figure="fig_feature_weight.png",
        figure_caption="Squared weight of the 248-component feature vector, "
                       "averaged over the production set. Note the log scale: "
                       "the coherent part is four decades down.",
        wide_figure=True,
        notes="""
The dephasing identity is the technical core of the whole talk and it is one
line. Tr(rho A) minus Tr(Delta(rho) A) equals the sum over i not equal j of
rho_ij A_ji, which is Tr(rho A_od) where A_od is A with its diagonal zeroed.

Read it twice, because it says two different things:
- forwards: any observable's expectation splits into a classical part read off
  the populations and a coherent part;
- backwards: if you *choose* an observable that is already purely off-diagonal,
  then the populations contribute exactly zero — not "a little", zero.

That backwards reading is the label construction two slides from now.

On the feature decomposition: single-mode occupation and its products means
<Z_w> and the factorized <Z_i><Z_j>. Connected covariance is the ZZ
correlation with the factorized part removed. Hopping coherence is XX and YY.
The numbers are recomputed from the production feature file by the deck's own
script; the earlier logged values from a smaller run were 99.7 / 0.2 / 0.01,
same three decades.

Do not overclaim. Four decades down is not zero. The question the next slide
answers is whether the residue is *usable*.
""",
    ),

    dict(
        layout="figure_hero",
        kicker="PART III  ·  THE FINDING",
        title="The coherence is real — and it is a composition variable",
        figure="fig_coherence_audit.png",
        figure_caption="Off-diagonal Frobenius share "
                       "$\\|\\rho_{\\mathrm{od}}\\|_F^2/\\mathrm{Tr}\\,\\rho^2$, "
                       "computed from the eigenblocks of all 1,000 production "
                       "states, against degree of unsaturation — a number counted "
                       "from the chemical formula at zero cost.",
        callout=("Why this is the interesting failure, not the obvious one.",
                 "The states are *not* classical: they are strongly correlated, "
                 "and their trace distance to the matched free-fermion Gibbs state "
                 "runs up to 0.99. Coherence is present and physically real. The "
                 "problem is that its **magnitude is redundantly encoded in the "
                 "molecule's composition** — so a classical model does not need "
                 "to read the coherence, it only needs to know chemistry. "
                 "Quantumness and classical descriptors are *confounded here*, "
                 "not absent."),
        stats=[("6.7%", "median off-diagonal share"),
               ("0.79", "Spearman vs degree of unsaturation"),
               ("0.93–0.99", "trace distance to free-fermion, for $\\pi$-systems")],
        notes="""
This is the slide the talk is built around. Make the distinction carefully,
because the shallow version of this result sounds obvious and the real one is
not.

The shallow version: "molecular thermal states are nearly diagonal, so of
course a quantum model gains nothing". That would be a statement about
magnitude, and it would be uninteresting — and it is also not what we found.
The states carry a median 6.7% off-diagonal Frobenius share, with a long tail
past 20%, and their trace distance to the *matched non-interacting* Gibbs state
reaches 0.99 for pi-systems. These are strongly correlated states. There is
plenty of coherence.

The real version: how much coherence a molecule has is predicted, at Spearman
0.79, by its degree of unsaturation — which is arithmetic on the chemical
formula. Free. No electronic structure calculation required.

So any label proportional to the *amount* of coherence is a composition label
wearing a disguise, and a classical model will match it without ever touching
an off-diagonal entry. It is not that the quantum information is missing. It is
that it is redundant with something classical.

This is the part I think generalizes. Chemistry correlates coherence with
composition quite generally — conjugation drives both. Any benchmark built by
taking molecules, computing correlated states, and labelling them by an
electronic property inherits this confound. You have to design *against* it,
and the next slide is our attempt.

Also worth flagging as a methodological aside: this number had been carried in
our notes for months from a script that had been lost. Rebuilding it for this
talk reproduced 6.7% and 0.79 exactly, from the eigenblocks. Reproduce your own
orphaned numbers.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART III  ·  THE DESIGN RESPONSE",
        title="Strip the diagonal: a label the populations cannot see",
        equations=[
            ("stripdiag",
             "Take any physically meaningful observable $A$, zero its diagonal, "
             "and label on the sign of the resulting expectation."),
            ("spinsplit",
             "The instance we tested. On the $S_z=0$ sector the total-spin "
             "operator splits exactly: $D$ counts unpaired electrons and is "
             "diagonal; $S^2_{\\mathrm{od}}$ is spin exchange between orbitals and "
             "is purely off-diagonal. Singlet and triplet built from the same two "
             "determinants have *identical populations* and differ only in the "
             "sign of a coherence."),
        ],
        bullets=[
            "**Exactly learnable.** Linear in $\\rho$, so it is inside the "
            "hypothesis class with zero Bayes error — provided $A_{\\mathrm{od}}$ "
            "is in the operator pool's span.",
            "**Provably invisible to a diagonal model.** By the identity on the "
            "previous slide, $\\mathrm{diag}(\\rho)$ contributes exactly zero.",
            "**It comes with its own ablation.** Retrain with a commuting pool. "
            "It *must* land at chance — and if it does not, that is a bug in our "
            "implementation, not a discovery.",
        ],
        note="That last property is the point. Most quantum-advantage claims are "
             "empirical comparisons somebody can always attack — *you undertuned "
             "the baseline*. This one is a theorem about the model class, tested "
             "through the same code path with one argument changed.",
        notes="""
The construction is three lines and it is the most reusable thing in the talk.

Why spin coupling as the first instance. It is rigorous rather than empirically
lucky: two orbitals, one electron each, S_z = 0. Nature builds the singlet and
the triplet out of *the same two determinants* with the same 50/50 populations;
only the relative sign differs. So the singlet-triplet distinction is 100%
coherence by construction, not by measurement. It is also chemically real —
thermally accessible diradical character governs photochemistry and singlet
fission — and it is computable from files we already had, with no rerun.

The ablation deserves emphasis with this audience. A commuting operator pool
means H is diagonal in the common eigenbasis, so Tr(rho H) reads only the
populations. Restricting the pool to Z-strings therefore *must* give chance
accuracy on a purely off-diagonal label. That is not a baseline you tune. It is
an identity. The reason to run it anyway is that a reviewer is entitled to
doubt that the implementation matches the claim.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART III  ·  MAKING THE ABLATION A THEOREM",
        title="Depth changes what is learnable — not what is visible",
        equations=[
            ("backprop",
             "Backpropagation from a classical layer into a quantum neuron — the "
             "rule the source paper leaves open. Beside the classical "
             "$\\delta\\cdot\\varphi'(z)\\cdot x$: the input vector becomes an "
             "operator pool, the per-sample sum a matrix-valued aggregate, and "
             "$\\varphi'$ the Fréchet derivative — which is *not* $\\varphi'(B)$."),
            ("commuting",
             "The corollary that matters. For a mutually commuting pool the "
             "forward pass **and the gradient**, at arbitrary depth, depend on "
             "$\\rho$ only through its populations."),
        ],
        bullets=[
            "**Why this architecture and not the fully quantum one.** Collapsing "
            "each activation observable to a scalar before the next layer means "
            "exactly one Fréchet derivative per parameter, never a composition of "
            "superoperators — which is what makes the gradient implementable.",
            "**No new quantum primitive is needed on hardware.** The forward form "
            "of the same derivative is precisely the quantity the source paper's "
            "existing sampling algorithm estimates; the classical error signal "
            "just multiplies it.",
            "**So the ablation is now a theorem about the whole model class**, "
            "not about one neuron. No depth of classical layers can recover "
            "coherence a commuting pool never admitted — asserted bit-identically "
            "in the test suite.",
        ],
        note="Depth does buy expressivity — ratios of expectations, and nonlinear "
             "functionals that factor through finitely many linear ones, move "
             "inside the class. It buys **no** access to coherence.",
        notes="""
This is the theory contribution and it is worth two minutes.

The source paper writes the forward pass for a hybrid network — quantum first
layer, classical layers on top — and then says one can take advantage of
backpropagation, and leaves the simulation and training open. But "one can use
backprop" is an assertion, not a rule: the chain rule has to terminate
somewhere, and where it terminates here is the derivative of a *matrix
function* with respect to the coefficients of its argument.

The seam is Daleckii-Krein: the Frechet derivative of an operator function is a
Hadamard product with the matrix of first divided differences, in the
eigenbasis of the argument. Two properties do the work. It is self-adjoint,
which is what makes reverse mode possible — otherwise J gradient components
would need J separate cubic-cost rotations instead of one. And it is
well-defined under degeneracy, which matters in practice because our pools
contain many commuting terms and the spectra really are degenerate at
initialization.

The corollary is the payoff. In a common eigenbasis the divided-difference
matrix picks out only diagonal entries of the aggregate, so off-diagonal
entries of rho appear nowhere in the forward pass *or* the gradient, at any
depth. We test it by deleting every off-diagonal entry of every state and
asserting that a commuting-pool network's outputs and all its gradients are
bit-identical — with a companion test that the non-commuting pool's output does
change, so the first assertion is not vacuous.

Cost note if asked: the spectral work is dataset-size independent. J1
eigendecompositions per epoch regardless of M. Measured, an epoch costs 3.6 s
at M = 60 and 4.2 s at M = 1000 — a seventeen-fold dataset for 17% more time.
""",
    ),

    dict(
        layout="figure_hero",
        kicker="PART III  ·  THE MEASUREMENT",
        title="Quantum minus classical: +0.00 points",
        figure="fig_quantum_vs_classical.png",
        figure_caption="All 1,000 molecules. Identical loss, optimizer, "
                       "temperature, split and epoch budget — the models differ "
                       "only in whether the operator pool reaches off the diagonal.",
        callout=("Why the coherence-only label still failed.",
                 "Per *state*, $c=\\mathrm{Tr}(\\rho S^2_{\\mathrm{od}})$ is 100% "
                 "coherence — the construction is correct. Across the *dataset* it "
                 "is not independent: the correlation between spin coupling and "
                 "unpaired-electron count is 0.92, so the label is a diagonal "
                 "label to two significant figures. Per-state invisibility did not "
                 "survive contact with the dataset-level confound."),
        stats=[("+0.00", "points, on both physical labels"),
               ("0.92", "corr(coupling, unpaired count)"),
               ("1.3", "points the whole pipeline buys over counting double bonds")],
        notes="""
The measurement. Quantum pool: identity, Z, ZZ, XX, YY — 146 parameters.
Classical pool: identity, Z, ZZ, strictly diagonal — 56 parameters. Same
everything else. Difference in held-out accuracy: zero to two decimal places.
Difference in loss: 9 x 10^-5.

Four reasons, in order of decisiveness.

One: the observable is about 86% diagonal on this sector — mean |c| / |<S^2>|
is 0.139.

Two: the correlation between <S^2> and the diagonal count D is 0.994. Median
splits of the two assign 97.4% of molecules to the same class. The label *is* a
diagonal label to three significant figures.

Three — and this is the subtle one — even the coherence-only part is
classically determined: corr(c, D) = 0.919. Stripping the diagonal off the
*operator* did not decouple the label from the diagonal at the *dataset* level.
That is exactly the distinction between "no diagonal-only quantum model can
learn this", which we proved, and "no classical model can learn this", which
does not follow.

Four: plain chemical descriptors reach 92.7% against the neurons' 94.0%. The
entire pipeline buys 1.3 points over counting double bonds.

And the screening metric predicted all of it before any training ran — next
slide.

Controls, if challenged: the 10-qubit projection retains 99.79% of off-diagonal
Frobenius weight (median), so the projection is not the cause; gradients verified
against central differences to 1e-9; the classical loss verified *exactly*
invariant under off-diagonal perturbations.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART III  ·  THE INSTRUMENT",
        title="A positive control that failed — and what it revealed",
        equations=[
            ("screen",
             "Because the single-neuron loss is linear in each state, training "
             "sees the dataset only through the two class sums. So the entire "
             "learning problem is *tell $R_+$ from $R_-$*, and this ratio ranks "
             "any candidate label in seconds, with no training run at all."),
        ],
        bullets=[
            "**We built a label that must work.** A synthetic target drawn from "
            "the pool's own off-diagonal generators, so an exact zero-error "
            "solution provably exists in the hypothesis class.",
            "**The optimizer did not find it — and preferred not to.** The exact "
            "operator scores 100% at Fermi–Dirac loss 3.52. The optimizer "
            "converges to loss 1.37 at 66%. Scaling the exact solution up only "
            "makes the loss worse, so it is not a normalisation artefact.",
            "**Diagnosis.** The decision rule needs only the *mean* to have the "
            "right sign; the loss penalises the whole *spectrum* of $H$ on each "
            "state's support. These states are strongly overlapping — all 1,000 "
            "share a dominant configuration — so the loss is dominated by that "
            "common mode and shrinks the discriminative directions away.",
        ],
        figure="fig_control_failure.png",
        figure_caption="The objective is misaligned with the decision rule on "
                       "structured, strongly-overlapping inputs.",
        notes="""
This is the second finding, and it is about the machine rather than the data.

Set the control up properly: we drew a random combination of the pool's own
XX/YY generators, made the label the sign of its expectation minus a threshold,
and noted that since the identity is also in the pool, the exact solution
w* = A_od - theta*I *exists*. The neuron should reach 100%. It reached 66.7%.

Then the diagnostic that makes it a finding rather than a bug report: we
evaluated the loss *at* the exact solution. It is 3.52. The optimizer converged
to 1.37. The loss genuinely prefers an operator that classifies worse. Scaling
w* up by 5, 20, 100 gives 16.2, 64.4, 322 — monotonically worse — so this is
not a normalisation artefact.

Why. The Fermi-Dirac log-loss is Tr[l_y(H) rho], applied by functional
calculus. It penalises the entire spectrum of H on the support of each state,
not just the mean. Our states are highly structured and strongly overlapping —
every one of the thousand has the same dominant reference configuration — so
the common mode dominates the loss while the discriminative off-diagonal
directions are about ten times smaller and get shrunk away. Converged norm 1.5
against the exact solution's 9.5.

Two things rescue this from being a dead end. Plain logistic regression on the
*same* quantum features gets 100% train / 91% held-out, so the information is
there and linearly accessible. And the hybrid network optimizes an ordinary
cross-entropy on a classical scalar, so the pathology cannot occur by
construction — measured, with the diagonal ablation at 50.3% and the quantum
pool above it.

The screening metric on this slide is the reusable tool. On our two physical
labels it reads 0.12 and 0.16; even the synthetic purely-off-diagonal control
only reaches 0.34 on this dataset, which is itself a strong statement about how
diagonal these states are.
""",
    ),

    # ================================================ PART IV — SCALE AND NEXT
    dict(
        layout="figure_hero",
        kicker="PART IV  ·  THE WALL",
        title="Where exact diagonalization stops",
        figure="fig_scaling.png",
        figure_caption="Sector dimension against window size at half filling. "
                       "The qubit register is $Q=2n$; the Hilbert space the solver "
                       "must handle is the symmetry sector, not $2^{Q}$.",
        bullets=[
            "**The window is the approximation, and it is too small for the "
            "molecules we care about.** Extended $\\pi$-systems contribute more "
            "than eight active electrons on their own; measured, the thermal "
            "states of the most unsaturated decile press against the boundary of "
            "the window (median edge occupation slack 0.167).",
            "**Those are exactly the interesting molecules.** They carry the most "
            "coherence — which is the confound of Part III seen from the other "
            "side.",
            "**Diagonalization cannot supply the larger window.** Dense ends near "
            "$7\\times10^{4}$; the certified Krylov path reaches an order of "
            "magnitude further but only at low temperature, and the production "
            "temperature is not low.",
        ],
        notes="""
Be honest about this: it is the strongest internal criticism of the dataset.

The measurement behind the first bullet: we checked how much occupation sits on
the boundary orbitals of the window. At zero temperature the ground states are
bracketed fine even for the most conjugated molecules — edge slack below 0.05.
But the thermal states at kT = 0.1 are not: in the highest-unsaturation decile
the median edge slack is 0.167, with 92% above 0.1. The Boltzmann window
reaches states that want orbitals outside the eight we kept. So the active
space is adequate for ground-state chemistry and marginal for the ensemble that
is actually our input.

Consequence, stated plainly: any result that depends on states pressed against
the active-space boundary needs a larger-window confirmation run before we
would believe it. That is an open item, not a closed one.

On the qubit axis: note the register is 2n wires but the solver works in the
symmetry sector. At n = 8 that is 4,900 rather than 65,536. The sector is what
sets the compute; the register size is what sets the conversion cost.
""",
    ),

    dict(
        layout="steps_eq",
        kicker="PART IV  ·  PAST THE WALL",
        title="Scaling the register: what changes and what does not",
        equations=[
            ("qubits",
             "Wires grow linearly in the number of modes, the sector grows "
             "combinatorially, and the *conversion* cost — the step that "
             "materializes the encoded block — grows as $m\\cdot 2^{Q}$. That last "
             "term is what fails first, and it is a property of our conversion, "
             "not of the physics."),
        ],
        steps=[
            ("Sample instead of diagonalize",
             "Minimally-entangled typical thermal states: collapse into a product "
             "basis, evolve in imaginary time as an MPS, measure, repeat. Every "
             "operation is an MPS operation, no spectrum is ever constructed, and "
             "the output arrives natively in the simulation form. Certificates "
             "become statistical error bars, which satisfies the same solver "
             "contract."),
            ("Keep the interface, replace the backend",
             "The solver protocol is the seam that was designed for this. Nothing "
             "upstream of it and nothing downstream of it has to change."),
            ("Watch the resolution limit",
             "Sampling resolves an expectation only above its statistical error — "
             "and the discriminative coherences here sit four decades below the "
             "leading features. This is the one place where Part III's finding "
             "and Part IV's method interact badly, and it needs quantifying "
             "before we commit."),
        ],
        note="**Also on the table: change the source, not just the solver.** "
             "Nothing above the first pipeline stage knows where the one-particle "
             "Hamiltonian came from. Larger basis sets, transition-metal "
             "complexes, or lattice models with a tunable interaction all satisfy "
             "the same interface — and the last of those would let us vary "
             "quantumness *independently of composition*, which is precisely the "
             "axis Part III says we lack.",
        notes="""
Two escape routes, and I want opinions on which to spend time on.

Route one is computational: replace the solver. METTS gives thermal expectation
values by a Markov chain over minimally-entangled typical thermal states, all
in MPS form, with cost polynomial in bond dimension and chain length. It never
builds a spectrum, so the wall on the previous slide simply does not apply, and
the output is already in the representation the tensor-network trainers want.
The quasi-area-law behaviour of these states at kT = 0.1 is what makes the bond
dimension modest, and we have measured bond profiles on real blocks that
support it.

The caveat in step three is genuine and I do not want to gloss it: a sampling
method resolves an off-diagonal expectation only when it stands above the
Monte Carlo error, and Part III showed those expectations are four decades down.
Trading exactness for reach might trade away precisely the signal we are hunting.

Route two is scientific, and it is the one I have come round to. The confound
in Part III is a property of *this family of systems*: in organic molecules,
conjugation drives coherence and composition together, so they cannot be varied
independently. A model system with a tunable interaction strength at fixed
particle content would break that degeneracy by construction. That is a
different dataset, not a bigger one, and the pipeline already accepts it — the
only thing the first stage needs is a one-particle Hamiltonian and a geometry.
""",
    ),

    dict(
        layout="summary",
        kicker="SUMMARY",
        title="Where this stands",
        columns=[
            ("Built",
             ["An exact interacting thermal-state dataset with certified "
              "truncation at every stage, and a storage form that converts "
              "losslessly into features, purification MPS, or a dense operator.",
              "The training rule for the hybrid quantum–classical network the "
              "source paper leaves open, with the commuting-pool reduction proved "
              "at gradient level and asserted in tests.",
              "A screening metric that ranks a candidate label in seconds without "
              "training anything."]),
            ("Found",
             ["Coherence in these states is real but **redundant with "
              "composition** — Spearman 0.79 against a quantity you count from "
              "the formula. Labels proportional to how quantum a molecule is are "
              "classical labels in disguise.",
              "The spectral Fermi–Dirac objective **does not chase off-diagonal "
              "signal** on strongly-overlapping inputs: it prefers a "
              "worse-accuracy operator. Confirmed against a control with a known "
              "exact solution."]),
            ("Next",
             ["A label whose *structure*, not magnitude, is coherent — "
              "excitation-channel routing is the leading candidate.",
              "A source where quantumness varies **independently of composition**. "
              "This is now the priority over scaling the current one.",
              "A sampling backend for larger registers, with the coherence "
              "resolution limit quantified first."]),
        ],
        strapline="Every stage validated against an independent reference; every "
                  "truncation certified; every negative result written down.",
        notes="""
Close on the reframing rather than on the artifact list.

We set out to build molecular training data for a quantum classifier, and we
built it — the engineering is sound and the object is reusable. What we learned
is that the hard part was never the physics or the compute. It is that
"quantum advantage on real data" requires a label whose *structure* only
coherence determines, and on molecular data the obvious candidates are all
predicted by composition, which is free.

If I had to name the single transferable lesson: check the confound before you
build the dataset, not after. The screening metric on the previous slide costs
seconds and would have told us this in an afternoon.

Questions I would most like: is there a molecular observable whose coherent
part varies at fixed composition? And is anyone convinced by the model-system
route, or does leaving real molecules give up the point of the exercise?
""",
    ),

    dict(
        layout="references",
        kicker="REFERENCES",
        title="Sources",
        groups=[
            ("The model",
             ["A. He, N. Liu, M. M. Wilde, *Fermi–Dirac machines as quantizations "
              "of neurons*, arXiv:2605.24386 (2026) — the neuron, the quantized "
              "activations, the hybrid architecture of §VII.C.",
              "Yu et al., *QH9: a quantum Hamiltonian prediction benchmark*, "
              "NeurIPS Datasets & Benchmarks (2023), arXiv:2306.09549."]),
            ("Methods",
             ["Sun et al., *Recent developments in the PySCF program package*, "
              "J. Chem. Phys. **153**, 024109 (2020).",
              "Davidson, *J. Comput. Phys.* **17**, 87 (1975); Knowles & Handy, "
              "*Chem. Phys. Lett.* **111**, 315 (1984).",
              "Jordan & Wigner, *Z. Phys.* **47**, 631 (1928).",
              "Oseledets, *Tensor-train decomposition*, SIAM J. Sci. Comput. "
              "**33**, 2295 (2011); Schollwöck, *Ann. Phys.* **326**, 96 (2011).",
              "White, *Phys. Rev. Lett.* **102**, 190601 (2009); Stoudenmire & "
              "White, *New J. Phys.* **12**, 055026 (2010) — METTS."]),
            ("The gradient",
             ["Daleckii & Krein (1965); Bhatia, *Matrix Analysis*, Thm. V.3.3; "
              "Higham, *Functions of Matrices*, Thm. 3.11 — the Fréchet "
              "derivative of an operator function."]),
        ],
        note="Full citation list, including everything needed to reproduce the "
             "code, in the companion document **`Papers/presentation_references.md`**. "
             "Code and data provenance: the `Quantum_Neuron_Research` repository — "
             "`qthermal/` (pipeline), `qnn/` (the network), "
             "`docs/HYBRID_BACKPROP.md` (the derivation), "
             "`scripts/presentation/` (this deck and every figure in it).",
        notes="""
The companion references document is the one to hand out: it is organised by
what each citation supports, separates the physics from the numerical methods
from the software, and marks which ones were needed to get the *code* right
rather than only the exposition.

Two worth calling out because they are load-bearing and easy to miss:
Daleckii-Krein, which is the theorem the whole hybrid gradient rests on and
which the source paper only states for tanh; and the METTS pair, which is the
concrete proposal in Part IV rather than background reading.
""",
    ),
]
