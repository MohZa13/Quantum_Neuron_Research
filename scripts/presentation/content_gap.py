"""Four slides: why the HOMO-LUMO gap label shows no quantum advantage.

Same slide model as ``content.py``; layout lives in ``build_deck.py``.  Every
number here is reproduced by ``scripts/gap_diagnosis*.py`` into
``results/gap_diagnosis*.json`` and plotted by ``figures_gap.py``, so nothing is
quoted from memory.
"""
from __future__ import annotations

TITLE = "Why the HOMO–LUMO gap shows no quantum advantage"
SUBTITLE = ("A four-way test on the first 1000 QH9 thermal states, "
            "CAS(8,8), $k_BT$ = 0.1 Ha")
VENUE = "August 2026"

SLIDES: list[dict] = [

    # ------------------------------------------------------------ 1. the null
    dict(
        layout="figure_hero",
        kicker="THE MEASUREMENT  ·  1000 MOLECULES, 25 SPLITS",
        title="The coherence channel carries no gap information",
        figure="fig_ladder.png",
        figure_caption="Logistic classifier, identical trainer and identical "
                       "splits throughout; the only thing that changes is which "
                       "features it may read. Label: below- vs above-median "
                       "mean-field HOMO–LUMO gap, balanced 50/50.",
        callout=(
            "The ablation is exact, not a tuned baseline",
            "On a Jordan–Wigner register the 136 Z and ZZ strings are diagonal "
            "operators, so their expectations are functions of "
            "$\\mathrm{diag}(\\rho)$ alone: provably what a dephased, classical "
            "model may read (HYBRID_BACKPROP §5.2). The 112 XX/YY strings read "
            "*only* off-diagonal entries of $\\rho$.\n\n"
            "So «quantum − classical» here is a feature ablation inside one "
            "model class, with no separate baseline to argue about.\n\n"
            "**Result:** the coherence channel alone lands at chance (53.0%, "
            "AUC 0.555). Adding it to the diagonal pool *costs* 1.25 points "
            "(paired t, p = 1.1e-5); 112 Gaussian noise features cost 3.17. It "
            "sits "
            "closer to noise than to signal.\n\n"
            "Only 56 of the 112 are independent: on real, $S_z$-conserving "
            "states ⟨XX⟩ = ⟨YY⟩ exactly (max deviation 0.0)."),
        stats=[("94.8%", "diagonal pool alone"),
               ("−1.25", "points gained by going quantum"),
               ("53.0%", "coherence channel alone")],
        notes="""
Start here, because it reframes the question. The earlier result (94% held-out
on the gap from thermal-state Pauli features) is real and reproduces. What is
new is the decomposition of that 94%.

The point of the left panel is that the ladder is flat, and flat low down.
Sixteen single-qubit occupation numbers get 92.3%. Eight natural occupation
numbers get 92.7%. Fifteen numbers describing only the eigenvalue spectrum of
rho plus the CI gap get 96.1%, the best model on the slide, and that one is
basis-independent, so by construction it contains no coherence information
whatsoever.

Note the full diag(rho) row at 89.8%: 4900 exact determinant populations
compressed to 300 principal components. It sits *below* the 136-feature
diagonal pool, which says the diagonal pool is a good smoothing of the
populations rather than a weaker view of them. I include it so nobody can call
the classical baseline a strawman: we handed it the exact diagonal and it did
not need it.

The right panel is the honest version of "no advantage". Adding 112 real
coherence features to the diagonal pool costs 1.25 points of held-out accuracy.
Adding 112 pure noise features costs 3.17. So the coherence block is measurably
better than noise, and nowhere near paying for its own dimensionality. McNemar
over the pooled held-out predictions: the quantum pool is uniquely right 102
times, the classical pool 196 times, p = 6e-8.
""",
    ),

    # -------------------------------------------------- 2. not the binarisation
    dict(
        layout="figure_hero",
        kicker="CANDIDATE 1  ·  «IT IS A CLASSIFIER, NOT A REGRESSION»",
        title="Predicting the gap as a number gives the same answer",
        figure="fig_not_binarisation.png",
        figure_caption="Left: ridge regression on the continuous gap, same "
                       "features, same splits. Right: held-out accuracy of the "
                       "binary classifier by quintile of |gap − median|.",
        callout=(
            "Regression reproduces the null",
            "If the median split were discarding the electronic structure and "
            "keeping only a coarse geometric contrast, predicting the gap as a "
            "*number* would restore the quantum channel.\n\n"
            "It stays absent. The diagonal pool reaches $R^2$ = 0.853, the full "
            "quantum pool 0.849, so quantum − classical = **−0.0038**. The "
            "coherence channel alone reaches $R^2$ = **0.000**: on held-out "
            "molecules it does no better than the training mean, on any scale.\n\n"
            "The right panel shows what binarisation actually costs. Accuracy "
            "is 75% in the nearest quintile, within 0.44 eV of the cut, and "
            "95–100% beyond it. That is threshold noise on a continuous "
            "quantity: "
            "a known, bounded, classical cost, and not a hidden coherence "
            "channel."),
        stats=[("0.853", "diagonal pool, $R^2$"),
               ("0.849", "full quantum pool, $R^2$"),
               ("0.000", "coherence channel, $R^2$")],
        notes="""
This is the cheapest hypothesis to test and it dies cleanly.

The regression uses the same estimator family (ridge instead of logistic,
lambda by inner cross-validation) on the same 25 stratified 70/30 splits. If
the story were "molecular geometry decides which side of the median you land
on, while electronic coupling decides the fine structure", then going
continuous should have surfaced the fine structure. The two ladders instead
agree to three decimals.

The coherence-only R^2 of 0.000 deserves a pause. That is zero to three places,
not a small positive number: on held-out molecules the off-diagonal features
predict the gap no better than the training mean. Rank correlation is 0.11.

The quintile panel doubles as a sanity check on label design. Flat accuracy
across quintiles would indicate a model that has learned something crude;
instead it reaches 100% in the fourth quintile, which is what a well-determined
label looks like, with the loss concentrated exactly where the median split is
physically arbitrary.
""",
    ),

    # ------------------------------------------------------ 3. the real reason
    dict(
        layout="figure_hero",
        kicker="CANDIDATE 2  ·  WHERE THE GAP ACTUALLY LIVES",
        title="Coherence tracks the gap, then adds nothing to it",
        figure="fig_where_gap_lives.png",
        figure_caption="Left: |Pearson r| of four coherence measures with the "
                       "gap, and with the gap residual after an out-of-fold "
                       "diagonal model. Right: predicting the sign of that "
                       "residual.",
        callout=(
            "Present, and redundant",
            "The states carry real coherence: the exact off-diagonal share of "
            "$\\rho$ is 6.7% median and reaches 25%, and on these same 1000 "
            "states a synthetic purely off-diagonal label separates the pools "
            "by **+42.0 points**. The instrument works.\n\n"
            "The label is the problem. The gap is $\\varepsilon_{LUMO} - "
            "\\varepsilon_{HOMO}$ from $\\mathrm{eigh}(F,S)$, a **one-body "
            "mean-field eigenvalue difference** defined before correlation "
            "enters, and it is 86% determined by $\\mathrm{diag}(\\rho)$ alone "
            "($R^2$ = 0.856 out-of-fold).\n\n"
            "Coherence correlates with the gap at r = −0.57. Against the "
            "residual that collapses to r = +0.04: both are driven by the same "
            "chemistry, off-diagonal share vs degree of unsaturation at "
            "Spearman **+0.79**.\n\n"
            "Whatever the diagonal missed is predicted better by counting atoms "
            "(64.8%) than by coherence (51.9%, chance)."),
        stats=[("0.856", "$R^2$ of the diagonal alone"),
               ("+0.04", "coherence vs the residual"),
               ("+0.79", "coherence vs degree of unsaturation")],
        notes="""
This is the diagnosis, and it is a redundancy result rather than an absence
result, which changes what to do next.

Take the four probes on the left. Static correlation correlates −0.79 with the
gap; against the residual, −0.007. Entropy: −0.78 becomes −0.013. The exact
off-diagonal Frobenius share of rho: −0.57 becomes +0.04. Adding all three as
extra features to the diagonal pool moves held-out accuracy by −0.01 points.
Every quantum-flavoured scalar we have is a function of something the diagonal
has already reported.

The mechanism: at kT = 0.1 the thermal state's mixedness *is* the gap.
Ground-state weight correlates +0.83 with it, entropy −0.78. So diag(rho) is
close to a sufficient statistic for a label defined by the one-body spectrum.

One caveat for the record. The weight-<=2 pool is a narrow window on rho: it
sees 0.46% of ||diag(rho)||^2 and 0.002% of ||offdiag(rho)||^2. Strictly, then,
the claim is "no coherence channel the extended-Heisenberg pool can read",
rather than "none exists in rho". The residual test is what makes the stronger
claim safe, because the exact off-diagonal share is not limited by the pool and
it explains nothing either.

Second caveat, and a methodological finding in its own right: the rho-level
R+/R- screen for this label is 0.135, comparable to <S^2> at 0.122 and c at
0.162. The screen would not have predicted this failure. It measures whether
the classes differ off-diagonally; it says nothing about whether that
difference is redundant with the diagonal. Screening should be run on the
residual, not on the raw label.
""",
    ),

    # ------------------------------------------------------------- 4. verdict
    dict(
        layout="figure_hero",
        kicker="VERDICT  ·  WHAT TO DO INSTEAD",
        title="Retire the gap, and pick molecules for the right reason",
        figure="fig_headroom.png",
        figure_caption="Left: the same ablation under four labels, identical "
                       "molecules and features. Right: off-diagonal weight of "
                       "$\\rho$, QH9 overall against the most conjugated "
                       "subset, at two temperatures.",
        callout=(
            "Abandon it, for a stronger reason than we assumed",
            "**Retire the label.** The argument is definitional: a mean-field "
            "HOMO–LUMO gap is a *one-body observable*, and no dataset makes a "
            "one-body observable quantum.\n\n"
            "The obvious repairs were tested and fail. The correlated CASCI gap "
            "$E_1-E_0$: **−0.8**. The correlation correction, i.e. the part of "
            "the correlated gap the mean-field gap cannot predict, σ ≈ 1.0 eV: "
            "**−1.1**. Restricting to the *most coherent half* of QH9 makes it "
            "**worse** (−3.1).\n\n"
            "**Harder molecules still pay, for a different label.** The 28 most "
            "conjugated molecules roughly double the off-diagonal weight (6.7% "
            "→ 12.4%); $k_BT$ = 0.25 adds a little more (14.8%). Useful "
            "headroom of ≈2×, available only once the label is one that "
            "coherence determines."),
        stats=[("+42.0", "synthetic off-diagonal control"),
               ("−1.1", "correlation correction to the gap"),
               ("≈2×", "coherence from the hardest molecules")],
        notes="""
The recommendation, and then the OMol25 question.

Retire the label. The decisive argument is definitional rather than empirical:
eigh(F, S) is a one-body eigenproblem, and a property defined by a one-body
operator has no correlation content to withhold from a classical model. Every
empirical result on the previous three slides is downstream of that one fact.

This is not "QH9 is too easy". Two of the three repair attempts moved towards
harder chemistry and made the separation worse, and the third, the correlation
correction, is by construction the part no one-body method can produce, carries
a 1.0 eV spread, and still handed the win to the diagonal pool.

Now the dataset question, which is the right question asked about the wrong
axis. What we need is a label whose value varies at *fixed composition*, rather
than molecules with more correlation. This is the same confound that killed the
spin labels, where corr(c, D) was 0.919; here it is Spearman 0.79 between
coherence and degree of unsaturation.

On OMol25 specifically. It is the natural successor: over 100M calculations at
wB97M-V/def2-TZVPD, 83 elements, systems up to 350 atoms, and crucially
variable charge and spin including transition-metal complexes with genuine
static correlation. QH9 is 130k neutral closed-shell organics of at most nine
heavy atoms, and it has no axis along which correlation varies independently of
composition, which is exactly what we keep colliding with.

Two practical caveats. First, our pipeline consumes the AO-basis Fock matrix,
which is what lets us skip SCF entirely. OMol25 proper ships energies and
forces; the artifact we would actually need is OMol_CSH_58k, the closed-shell
Hamiltonian subset (58 elements, systems up to 150 atoms, def2-TZVPD). Note "closed
shell": it excludes the open-shell metal complexes that carry the most static
correlation, which was half the appeal.

Second, cost, and it does not land where people expect. The CASCI dimension is
set by the active space rather than the molecule: CAS(8,8) is 4900, CAS(10,10)
is 63,504, CAS(12,12) is 853,776. What grows with OMol25 is the integral
transform, since def2-TZVPD on 100 atoms is thousands of AOs against about 120
for def2-SVP on nine heavy atoms, and the active space genuinely *required*: a
metal complex needs the d shell plus ligand frontier orbitals, so CAS(8,8) is
indefensible there. We measured 4.5-6 min/molecule for the iterative solver at
CAS(10,10), kT = 0.025, and the 1000-molecule CAS(8,8) run is 45 GB on disk.

There is also a live warning from our own data: CAS(8,8) already truncates the
thermal states of high-DoU QH9 molecules, median edge slack 0.167 in the top
decile. Moving to more correlated chemistry without widening the active space
means representing the interesting molecules worst.

So: the label first, the dataset second, and the dataset chosen for
composition-independent variation rather than for raw difficulty.
""",
    ),
]
