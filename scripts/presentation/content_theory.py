"""Background lecture: molecular Hamiltonians and thermal states.

One dict per slide, plus its speaker notes.  Layout lives in ``build_deck.py``.

House style for this deck, checked by the build:

* no em dashes, and no en dash outside a hyphenated name or a numeric range;
* direct assertion.  State what holds.  Contrastive constructions of the form
  "not X but Y" are reported by the style lint and should be rewritten;
* the slide carries the statement and the mathematics; the speaker notes carry
  the explanation;
* the subject is the physics.  The project appears as one instance of the
  general construction, never as the topic.
"""
from __future__ import annotations

TITLE = "Molecular Hamiltonians and Thermal States"
SUBTITLE = ("The structure of interacting fermionic systems, and what their "
            "equilibrium states are good for")
VENUE = "Background lecture"

SLIDES: list[dict] = [

    dict(
        layout="title",
        title=TITLE,
        subtitle=SUBTITLE,
        venue=VENUE,
        strapline="Sixteen fermionic modes at half filling.",
        notes="""
This lecture covers the structure of interacting electronic systems and the
equilibrium states built from them.

The audience I have in mind knows quantum information and has had little reason
to look at quantum chemistry. The two fields study the same objects. A molecular
Hamiltonian is a sparse, local, physically constrained operator on a fermionic
register, and a thermal state of one is a mixed state with tunable purity and
entanglement. Both are natural inputs to the algorithms this community designs.

Five parts. The fermionic algebra and the Hamiltonian written in it. The
reductions that make the problem finite. Thermal states and their construction.
The translation to qubits. Finally, and at length, what these objects are used
for, from industrial catalysis to quantum algorithm benchmarks.

The last part carries the argument. Electronic structure is the largest single
application area proposed for quantum computation, and the reason is that these
objects resist classical treatment while determining chemistry that matters.
""",
    ),

    dict(
        layout="flow",
        kicker="ORIENTATION",
        title="The construction, end to end",
        stages=[
            ("Nuclear\ngeometry", "positions and\ncharges"),
            ("Single-particle\nbasis", "orbitals from a\nmean field"),
            ("Second\nquantization", "one matrix and\none tensor"),
            ("Active space", "the correlated\nregion"),
            ("Spectrum", "exact within\nthe region"),
            ("Gibbs state", "equilibrium at\ntemperature $T$"),
        ],
        highlight=(2, 5),
        caption="Each stage is a definite mathematical object. The two "
                "highlighted stages carry the content of this lecture.",
        note="**Two facts frame everything that follows.** The Hamiltonian of "
             "any non-relativistic electronic system contains only one-body and "
             "two-body terms, which constrains its structure severely. The "
             "dimension of the space it acts on grows combinatorially in the "
             "number of orbitals, which constrains what can be computed.",
        notes="""
This diagram is the map. Every later slide fills in one box.

The two facts in the note govern the whole subject and are worth holding onto.

The first is a structural constraint. Coulomb interaction is pairwise, so the
Hamiltonian has at most two-body terms. In the configuration basis this makes
the matrix extremely sparse in a specific, characterisable way, which we
quantify on slide 7. Every efficient method in electronic structure exploits it.

The second is the cost. The dimension of the many-electron space is a binomial
coefficient, so it grows faster than any polynomial in the number of orbitals.
Slide 11 gives the numbers.

The tension between these two facts defines the field. The Hamiltonian is
compactly specified by O(n^4) numbers, and the state it determines lives in a
space of exponential dimension.
""",
    ),

    # ================================ I. FERMIONIC MANY-BODY SYSTEMS
    dict(
        layout="eq_bullets",
        kicker="PART I  ·  THE HAMILTONIAN",
        title="The electronic Hamiltonian",
        equations=[
            ("electronic",
             "Kinetic energy, attraction to the nuclei, and Coulomb repulsion "
             "between electrons, in atomic units. Nuclear positions "
             "$\\mathbf{R}_A$ enter as fixed parameters under the "
             "Born-Oppenheimer approximation."),
        ],
        bullets=[
            "**The first two terms are one-body.** They act on one electron at "
            "a time and are diagonalized by a suitable single-particle basis.",
            "**The third term couples every pair of electrons.** It admits no "
            "separable solution and carries the entire computational "
            "difficulty of the subject.",
            "**The operator itself is exactly known.** The open problems concern "
            "representation and computation.",
        ],
        stats=[("$O(N^{2})$", "interacting pairs"),
               ("2", "highest body order"),
               ("$1.6$ mHa", "accuracy chemistry requires")],
        notes="""
Atomic units throughout. One Hartree is 27.2 electron volts. A covalent bond is
a few tenths of a Hartree, and chemical accuracy, the precision at which
predictions become useful for reaction rates, is about 1.6 millihartree.

Born-Oppenheimer: the proton is roughly two thousand times the electron mass, so
the nuclei are stationary on the timescale of electronic motion. The electronic
energy as a function of nuclear position defines the potential energy surface,
and essentially all of chemistry is the geometry of that surface.

Emphasise the second bullet for this audience. The exponential difficulty
originates in a single term. Delete the electron-electron repulsion and the
ground state is a Slater determinant computable in polynomial time. Restore it
and the problem becomes QMA-hard in the worst case.

The third bullet is the framing to leave them with. Chemistry has no unknown
physical law at this level. Dirac observed in 1929 that the underlying laws were
completely known and the difficulty was that the resulting equations were too
complicated to solve. The subject has been about representation ever since.
""",
    ),

    dict(
        layout="eq_stack",
        kicker="PART I  ·  THE ALGEBRA",
        title="Fock space and the mode operators",
        equations=[
            ("anticommute",
             "The canonical anticommutation relations. These define the algebra "
             "and carry the antisymmetry of identical fermions."),
            ("fock_build",
             "A basis state of Fock space, specified by an occupation string. "
             "The ordering of the creation operators fixes the sign, so a "
             "convention must be adopted and held throughout."),
        ],
        cards=[
            ("Modes, not particles",
             "A basis of $M$ single-particle functions gives $M$ modes. The "
             "state is a distribution of occupations over modes, which removes "
             "the labelling of indistinguishable particles entirely."),
            ("Antisymmetry is algebraic",
             "Setting $p=q$ gives $(a^{\\dagger}_p)^{2}=0$, so no mode holds "
             "two electrons. The Pauli principle follows from the algebra in "
             "one line."),
            ("The register is natural",
             "$\\dim\\mathcal{F}=2^{M}$. Fock space over $M$ modes and an "
             "$M$-qubit register have the same dimension, with basis states in "
             "bijection."),
        ],
        notes="""
This change of language makes everything else possible, and it is the slide to
spend time on with this audience because the payoff is immediate.

In first quantization one writes a function of many coordinates and imposes
antisymmetry under exchange by hand. The bookkeeping is severe and the objects
scale badly.

In second quantization one fixes a set of single-particle functions, called
orbitals, and specifies the state by how many electrons occupy each. The
antisymmetry moves into the algebra of the operators.

Read the anticommutation relations as the definition. Everything else follows.
Setting p equal to q in the third relation gives the creation operator squared
equal to zero, which is the Pauli exclusion principle in one line.

The last card is the observation that connects to this audience. Fock space over
M modes has dimension two to the M, matching M qubits, with basis states in
bijection. The correspondence between states is trivial. The correspondence
between operators is the entire content of part four.

If asked about bosons: the same construction with commutators, and the mode
occupations become unbounded, so the space requires truncation before a register
can hold it.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART I  ·  THE ALGEBRA",
        title="Occupations, phases, and conserved quantities",
        equations=[
            ("creation_phase",
             "Acting on an occupation string, a creation operator contributes a "
             "sign given by the parity of the modes below it. This phase is a "
             "property of the algebra and reappears verbatim in the "
             "Jordan-Wigner transformation on slide 19."),
            ("number_ops",
             "The number operator, the total particle number, and the "
             "quantities the electronic Hamiltonian conserves."),
        ],
        bullets=[
            "**Conservation laws block-diagonalize the Hamiltonian.** Particle "
            "number, spin projection, and total spin are conserved, so the "
            "matrix decomposes into independent sectors labelled by their "
            "eigenvalues.",
            "**Spatial symmetry adds further blocks** whenever the nuclear "
            "framework has a point group.",
            "**Each sector is treated on its own, exactly.** For an "
            "$N$-electron system the relevant sector occupies a small fraction "
            "of Fock space.",
        ],
        note="For eight electrons in eight orbitals, Fock space has dimension "
             "$2^{16}=65{,}536$. Fixing the particle number and the spin "
             "projection leaves 4,900 states, a reduction of more than thirteen "
             "at no cost in accuracy. The same mechanism underlies the qubit "
             "tapering used to shrink registers for hardware experiments.",
        notes="""
Two points, and the second is the practical one.

The phase in the first equation is the sign incurred when a creation operator is
commuted past the modes below it. It is the price of writing an antisymmetric
object in terms of independent occupations, and it becomes the string of Z
operators in the Jordan-Wigner transformation. Flagging the connection now makes
slide 17 land as a consequence rather than a new idea.

The conservation laws are the cheapest structure available. The Coulomb
interaction commutes with the number operator, with the spin projection, and
with the total spin, so the Hamiltonian matrix is block diagonal with respect to
those quantum numbers.

For this audience the relevant framing is that a symmetry sector is a subspace
one restricts to exactly, and that the restriction genuinely shrinks the
problem. The same observation, applied to a qubit register, is the basis of the
tapering techniques that remove qubits by fixing the value of a stabilizer.

The example in the note is the one this project uses, and the factor of thirteen
is free.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART I  ·  THE HAMILTONIAN",
        title="The Hamiltonian in second quantization",
        equations=[
            ("second_quant",
             "The electronic Hamiltonian in the mode algebra. Indices $p,q,r,s$ "
             "run over spatial orbitals and $\\sigma,\\tau$ over spin."),
            ("integrals",
             "The coefficients. $h$ is an $n\\times n$ matrix and $g$ an "
             "$n^{4}$ tensor of Coulomb repulsions between orbital pair "
             "densities."),
        ],
        bullets=[
            "**Two arrays specify the system completely.** Given a basis, the "
            "physics reduces to $O(n^{2})$ and $O(n^{4})$ numbers.",
            "**The coefficients depend on the chosen basis.** Different "
            "orbitals give different arrays and the same spectrum, so the basis "
            "is a representation choice with consequences for cost.",
            "**The form is universal.** Lattice models, nuclear shell models, "
            "and quantum dots share this operator structure with different "
            "coefficients.",
        ],
        stats=[("$n^{2}+n^{4}$", "numbers specifying $H$"),
               ("$e^{\\,\\Theta(n)}$", "dimension of its sector"),
               ("$O(n^{4})$", "Pauli terms after encoding")],
        notes="""
This is the working form of the Hamiltonian and it appears on every subsequent
slide.

The one-body term moves an electron from orbital q to orbital p. The two-body
term moves two electrons at once. Both preserve the electron count, which is the
conservation law from the previous slide.

The contrast between the two statistics at the bottom is the whole subject. The
Hamiltonian is specified by a polynomial number of coefficients, and the space
it acts on has combinatorial dimension. It is a compressed description of an
exponentially large operator.

On the third bullet: the Hubbard model is this Hamiltonian with the one-body
term restricted to neighbouring sites and the two-body tensor reduced to a
single on-site constant. The nuclear shell model is this Hamiltonian with
different coefficients again. Anything proved about the structure here applies
to all of them, which is why methods migrate freely between quantum chemistry
and condensed matter physics.

On basis dependence: the exact spectrum in a complete basis is unique. Every
practical calculation uses a finite basis, where the choice affects both the
accuracy and the compactness of the description. Selecting a good basis is a
substantial part of the craft.
""",
    ),

    dict(
        layout="eq_stack",
        kicker="PART I  ·  STRUCTURE",
        title="Matrix elements between configurations",
        equations=[
            ("sc_diag",
             "Diagonal element for a configuration with occupied set $K$."),
            ("sc_single",
             "Configurations differing by one replacement $p\\to q$."),
            ("sc_double",
             "Configurations differing by two replacements, and the vanishing "
             "of everything beyond. The overall sign follows from the "
             "permutation aligning the two occupation strings."),
        ],
        note="**These are the Slater-Condon rules, and they are the source of "
             "all sparsity in the problem.** A configuration couples only to "
             "those reachable by moving at most two electrons. Each row carries "
             "$O(n^{4})$ non-zero entries while the row length grows "
             "combinatorially, so the matrix becomes arbitrarily sparse as the "
             "space grows.",
        notes="""
This slide is the structural heart of part one, and for an audience used to
thinking about sparse Hamiltonians it is the most directly useful content in the
lecture.

The rules follow from the two-body character of the Hamiltonian by a short
calculation. Since the operator moves at most two electrons, two configurations
differing in three or more occupied orbitals have zero matrix element.

The consequences are quantitative. Take N electrons in M orbitals. The number of
configurations reachable by a single replacement is of order N times M, and by a
double replacement of order N squared times M squared. So each row has O(n^4)
non-zero entries while the dimension is a binomial coefficient. The matrix is
sparse, and the sparsity improves as the system grows.

Three algorithmic facts follow directly, and all three matter here. First, the
matrix never needs to be stored; the action of the Hamiltonian on a vector is
computed from these rules on demand, which is what makes iterative eigensolvers
applicable at dimensions where dense storage is impossible. Second, the same
property places the Hamiltonian in the sparse access model used by quantum
simulation algorithms. Third, the coefficient one-norm controls the cost of
Hamiltonian simulation, and reducing it is an active research area.

The next slide shows the predicted pattern in a real system.
""",
    ),

    dict(
        layout="figure_hero",
        kicker="PART I  ·  STRUCTURE",
        title="The predicted sparsity, in a real system",
        figure="fig_hamiltonian_matrix.png",
        figure_caption="A real active-space Hamiltonian in the configuration "
                       "basis, and the thermal state constructed from its "
                       "spectrum. Magnitudes span many decades, so the colour "
                       "scale is logarithmic.",
        bullets=[
            "**The block pattern follows the excitation structure.** Blocks "
            "collect configurations connected by single and double "
            "replacements; the white regions vanish exactly.",
            "**Fifty-nine percent of the entries are zero** in this "
            "225-dimensional example, and the fraction rises with the size of "
            "the space.",
            "**Diagonalization destroys the sparsity.** The thermal state on "
            "the right is dense, because eigenvectors of an interacting "
            "Hamiltonian are delocalised over the whole space. This is the "
            "central obstacle to storing these states.",
        ],
        notes="""
The left panel confirms the Slater-Condon rules visually. The block structure
comes from grouping configurations by which orbitals they occupy. Blocks near
the diagonal connect by single replacements and the outlying blocks by doubles.
The white regions are exactly zero, by the rule on the previous slide.

The right panel is the thermal state built from the eigenvectors of that matrix,
in the same basis. Every entry is filled. This is generic: a function of a
sparse matrix is dense.

That observation sets up part three. The Hamiltonian is cheap to hold because it
is sparse and structured. Its equilibrium state is expensive to hold because it
is dense. Any workable pipeline keeps the state factored and never forms the
matrix on the right.

The example uses a six-orbital active space, chosen because 225 by 225 is
legible when projected. At eight orbitals the same matrix is 4,900 by 4,900 with
higher sparsity.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART I  ·  THE BASIS",
        title="Obtaining the single-particle basis",
        equations=[
            ("scf",
             "The self-consistent field equation. $F$ is an effective "
             "one-particle Hamiltonian built from the current solution, $S$ is "
             "the overlap of the underlying functions, and the equation is "
             "iterated to a fixed point."),
        ],
        bullets=[
            "**Hartree-Fock replaces the pair repulsion by its mean field** and "
            "recovers about 99 percent of the total electronic energy with a "
            "single determinant.",
            "**Density functional theory takes the same form,** writing the "
            "exchange and correlation contributions as a functional of the "
            "density. Accuracy is set by that choice; B3LYP is the most widely "
            "used one for molecules.",
            "**The remaining one percent carries the chemistry.** Bond "
            "energies, reaction barriers, and excited states live there, which "
            "is why the correlated treatment of part two is required.",
        ],
        figure="fig_orbital_ladder.png",
        figure_caption="Orbital energies of a water molecule. The energy axis "
                       "is broken; the core level lies far below the valence "
                       "region.",
        notes="""
Mean field theory supplies the orbitals used everywhere downstream.

The idea: each electron moves in the average field of the others, which gives a
one-electron equation. The average depends on the solution, so the equation is
iterated to self-consistency. In a finite basis it becomes the generalized
eigenvalue problem shown, with the overlap matrix appearing because the
atom-centred functions are non-orthogonal.

Density functional theory replaces exchange and correlation with a functional of
the density. Its accuracy depends entirely on which functional is chosen, and
the choice is empirical. B3LYP dates from 1993 and remains dominant for
molecules.

The third bullet carries the argument, and the numbers make the case. Hartree-
Fock recovers about ninety-nine percent of the total electronic energy. The
missing one percent, called the correlation energy, contains bond dissociation,
reaction barriers, excited states, and magnetic properties. A method accurate to
one percent of the total energy is useless for chemistry.

The figure shows a real case. Twenty-four orbitals, ten electrons filling the
lowest five. The core level sits five hundred and twenty electron volts below
the rest and never changes occupation. Chemistry happens at the boundary between
filled and empty.
""",
    ),

    dict(
        layout="two_cards_eq",
        kicker="PART I  ·  THE COEFFICIENTS",
        title="Computing the two-electron tensor",
        cards=[
            ("Evaluation",
             "The integrals are definite integrals over atom-centred Gaussian "
             "functions and have closed forms. Cost scales as the fourth power "
             "of the basis size, with screening reducing the prefactor sharply "
             "for extended systems."),
            ("Transformation",
             "Rotating the tensor into the orbital basis costs $O(n^{5})$ when "
             "performed one index at a time. This step dominates the setup for "
             "medium-sized systems."),
        ],
        equations=[
            ("ao2mo",
             "The four-index transformation. Each index is contracted in turn "
             "against the orbital coefficients."),
        ],
        note="**Storage sets the practical limit.** At 200 basis functions the "
             "tensor holds $1.6\\times10^{9}$ entries. Its numerical rank is far "
             "lower than its formal size, and density fitting, Cholesky "
             "decomposition, and tensor factorization all exploit that. The "
             "same low-rank structure has reduced published quantum resource "
             "estimates for chemistry by orders of magnitude.",
        notes="""
This slide covers where the numbers come from.

Gaussian basis functions are used because the product of two Gaussians centred
at different points is a Gaussian centred between them, which turns a
four-centre integral into a closed-form expression. Slater-type functions are
physically better and give integrals with no closed form, so Gaussians won.

The transformation is a change of basis on a four-index object. Performed
naively it costs the eighth power of the basis size. Performed one index at a
time it costs the fifth, which is the standard implementation.

The note is the practical constraint and connects to something this audience
will recognise. The two-electron tensor is formally n to the fourth, and its
numerical rank is roughly linear in the basis size. Density fitting and Cholesky
decomposition write it as a product of smaller factors. The same observation,
pushed further into double factorization and tensor hypercontraction, is the
main source of progress in quantum algorithm resource estimates for chemistry
over the past decade.
""",
    ),

    # ================================ II. REDUCING THE PROBLEM
    dict(
        layout="eq_bullets",
        kicker="PART II  ·  CORRELATION",
        title="Configuration interaction and its cost",
        equations=[
            ("ci_expansion",
             "The exact state in a finite basis is a superposition over all "
             "configurations. Determining the coefficients is a Hermitian "
             "eigenvalue problem in the space they span."),
            ("fock_dim",
             "Fock space decomposes by particle number, and each sector has "
             "binomial dimension."),
        ],
        bullets=[
            "**Full configuration interaction is exact in the given basis** and "
            "is the reference against which every approximate method is "
            "measured.",
            "**Its cost is the binomial coefficient.** Water in a modest basis "
            "gives $1.8\\times10^{9}$ configurations. Benzene in a standard "
            "basis exceeds $10^{44}$.",
            "**Truncating by excitation level fails for strongly correlated "
            "systems,** where many configurations carry comparable weight and "
            "no single reference dominates.",
        ],
        figure="fig_dimension.png",
        figure_caption="Configurations against the number of orbitals treated "
                       "as correlated, at half filling.",
        notes="""
Full configuration interaction is conceptually simple. Write the state as a
superposition over every configuration and diagonalize. It is exact within the
basis and intractable beyond about twenty orbitals.

The numbers deserve saying aloud. Water in a small basis has 1.8 billion
configurations. Benzene in a standard basis exceeds ten to the forty-four, which
is larger than the number of atoms in the observable universe by twenty orders
of magnitude.

The third bullet explains why the obvious remedy fails in exactly the cases one
cares about. The standard approach keeps configurations up to some excitation
level relative to a reference. That works when the reference dominates, which
holds near equilibrium for closed-shell molecules. It fails for stretched bonds,
transition metals, and excited states, where the weight spreads across many
configurations of comparable importance. Those systems are called strongly
correlated or multireference, and they are the interesting ones.

The resolution appears on the next slide: keep the full expansion, and shrink
the space it runs over.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART II  ·  ACTIVE SPACES",
        title="Restricting correlation to the region that requires it",
        equations=[
            ("frozen",
             "The frozen-core reduction. Orbitals held doubly occupied "
             "contribute an exactly computable potential to the one-body matrix "
             "and an additive constant. Under the occupancy assumption the step "
             "is exact."),
        ],
        bullets=[
            "**Partition the orbitals into three sets.** Core orbitals stay "
            "doubly occupied, active orbitals carry every possible occupation, "
            "and virtual orbitals stay empty.",
            "**Diagonalize exactly within the active set.** All correlation "
            "inside the window is recovered; correlation involving the frozen "
            "orbitals is discarded.",
            "**Energetic separation justifies the partition.** Promoting an "
            "electron out of a core orbital costs hundreds of electron volts, "
            "so those configurations carry negligible weight at chemical energy "
            "scales.",
            "**Selecting the window is the central difficulty.** Chemical "
            "intuition, natural orbital occupations, entanglement measures, and "
            "automated procedures based on atomic valence character are all in "
            "use.",
        ],
        note="**Complete active space methods built on this construction are "
             "the reference treatment** for transition metal chemistry, "
             "photochemistry, and bond breaking. The uniform choice in this "
             "project is eight electrons in eight orbitals, applied to every "
             "molecule so that all states share one register.",
        notes="""
The active space concept is the standard resolution and predates any interest
from quantum computing by forty years.

The observation is that correlation is local in energy. Core electrons are
tightly bound and their occupation never changes. High virtual orbitals sit far
above the Fermi level and are never populated. The chemistry occupies a narrow
window at the frontier.

So one treats that window exactly and freezes the rest. Freezing is exact
bookkeeping under the stated assumption: doubly occupied orbitals contribute a
fixed mean-field potential and a constant, both available in closed form.

The fourth bullet is where the craft lies. Choosing the active space is not
automatic. A poor choice gives qualitatively wrong answers, and a good choice
requires knowing which orbitals participate. There is a substantial literature
on automating it, using natural orbital occupation numbers, entanglement
measures taken from tensor network calculations, or projections onto atomic
valence orbitals.

For a dataset the requirement differs from a single calculation. A uniform rule
gives every molecule the same register and makes the states comparable, which is
what a learning problem needs. It is worse for any individual molecule and
correct for the collection.
""",
    ),

    dict(
        layout="two_cards_eq",
        kicker="PART II  ·  WHAT REMAINS",
        title="The reduced problem",
        cards=[
            ("What the reduction achieves",
             "Eight orbitals holding eight electrons give 4,900 configurations "
             "in the relevant spin sector. Direct diagonalization returns the "
             "full spectrum in about a minute."),
            ("What the reduction costs",
             "Correlation involving the discarded orbitals is lost. The error "
             "is systematic and is estimated by enlarging the window until the "
             "quantity of interest stops changing, or by adding a perturbative "
             "correction on top of the active space result."),
        ],
        equations=[
            ("spin_sector",
             "The dimension of the active sector, with the value used here."),
        ],
        note="**Adequacy is testable directly.** Compute the occupation of the "
             "orbitals at the boundary of the window. Appreciable population "
             "there indicates a state extending beyond the window. The test is "
             "more demanding at finite temperature, where excited "
             "configurations are populated by construction.",
        notes="""
This slide closes part two with the accounting.

The reduction takes an intractable problem to a small one. Four thousand nine
hundred configurations is a matrix that fits in memory and diagonalizes in
seconds.

The cost is stated plainly. Everything outside the window is treated at the mean
field level. For quantities dominated by frontier orbitals this suffices. For
quantities depending on dynamic correlation across many orbitals it does not,
and one adds a perturbative correction, which is what second-order perturbation
theory on a complete active space reference provides.

The adequacy test in the note is the honest check and is easy to perform. If the
highest active orbital carries appreciable occupation, the window is too small.
Ground states of ordinary organic molecules pass with an eight-orbital window.
Thermal states at elevated temperature become marginal, because the Boltzmann
weight reaches configurations wanting orbitals above the window. That limits the
temperature range accessible at fixed window size.
""",
    ),

    # ================================ III. THERMAL STATES
    dict(
        layout="eq_bullets",
        kicker="PART III  ·  EQUILIBRIUM",
        title="The Gibbs state",
        equations=[
            ("gibbs_def",
             "The equilibrium state at inverse temperature $\\beta$. In the "
             "energy eigenbasis it is diagonal with Boltzmann weights."),
            ("partition",
             "The partition function determines the thermodynamics. Free "
             "energy, entropy, heat capacity, and every equilibrium "
             "expectation value follow from it."),
        ],
        bullets=[
            "**It maximizes entropy at fixed mean energy,** which is the "
            "information-theoretic characterisation due to Jaynes.",
            "**It carries coherence in the configuration basis.** The energy "
            "eigenbasis of an interacting Hamiltonian bears no simple relation "
            "to the occupation basis a register encodes, so off-diagonal "
            "structure survives there.",
            "**Temperature interpolates continuously** between the ground state "
            "and the maximally mixed state, giving a one-parameter family of "
            "mixed states with controlled purity.",
        ],
        figure="fig_boltzmann.png",
        figure_caption="A real many-body spectrum, and the weights three "
                       "temperatures assign to it. The weight axis is "
                       "logarithmic.",
        notes="""
The Gibbs state arrives by two independent routes, and both deserve a sentence.

Physically, a system in weak contact with a large bath at temperature T
equilibrates to this state. The bath fixes the mean energy and nothing else.

Information-theoretically, it is the maximum entropy state consistent with a
given mean energy. Jaynes made this the foundation of statistical mechanics in
1957, and for this audience it is the more natural statement.

The second bullet is easy to misread. Every density matrix is diagonal in its
own eigenbasis. The question is whether it is diagonal in the basis one can
access. The energy eigenbasis of an interacting Hamiltonian is a complicated
superposition of configurations, so in the occupation basis the thermal state
has off-diagonal entries. Those coherences are the quantum content available to
any protocol acting on the state.

The third bullet is the property that makes these states useful as test objects.
One parameter tunes purity continuously, with the intermediate regime carrying
nontrivial entanglement structure.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART III  ·  THE REGIME",
        title="What temperature controls",
        equations=[],
        bullets=[
            "**Low temperature gives an almost pure state.** The ground state "
            "dominates and the ensemble contains one state to numerical "
            "precision.",
            "**High temperature gives an almost uniform state,** which retains "
            "little information about the specific Hamiltonian.",
            "**The intermediate regime carries structure.** A few hundred "
            "states participate, the number varies strongly between systems, "
            "and the entanglement across a bipartition is neither minimal nor "
            "maximal.",
            "**Electronic excitations are large compared with thermal energy.** "
            "At room temperature the Gibbs state of a molecular electronic "
            "Hamiltonian equals the ground state to many digits, so this regime "
            "requires elevated temperature or systems with small gaps.",
        ],
        figure="fig_effective_rank.png",
        figure_caption="Participation and purity against temperature, across "
                       "250 molecules. The dashed line marks the value used in "
                       "this project.",
        stats=[("206", "states carrying 99% of the weight"),
               ("50%", "median weight on the lowest state"),
               ("16% to 98%", "range across the set")],
        notes="""
This slide answers which temperature to use and states the caveat honestly.

The two limits are uninformative. Cold gives a pure state. Hot gives the
identity. Both look the same for every system, so neither distinguishes one
Hamiltonian from another.

The middle is where the ensemble reflects the spectrum. The measured numbers
below the figure: about two hundred states carry ninety-nine percent of the
weight, the lowest state holds half the weight at the median, and that ranges
from one sixth to almost all depending on the molecule.

The fourth bullet is the physical caveat and should be stated directly.
Electronic excitation energies are electron volts. Room temperature is
twenty-five millielectron volts. The Boltzmann factor for an electronic
excitation at room temperature is around ten to the minus forty. For molecular
electronic structure this regime therefore corresponds to temperatures far above
ambient, and temperature functions as a control parameter on the mixedness of
the state.

Systems with small electronic gaps are the exception, and finite-temperature
methods were developed for them. Correlated materials near a phase transition,
and molecules with near-degenerate frontier orbitals, have thermally accessible
excited states at ordinary temperatures.
""",
    ),

    dict(
        layout="table",
        kicker="PART III  ·  CONSTRUCTION",
        title="Methods for producing a thermal state, by reachable size",
        lede="All five produce the same object. They differ in what is computed "
             "along the way, and therefore in the size of system they reach.",
        columns=[("Method", 1.25), ("What it computes", 1.5),
                 ("Reach", 0.85), ("Properties", 1.5)],
        rows=[
            ("Direct diagonalization",
             "Every eigenvalue and eigenvector",
             "up to $\\sim10^{5}$",
             "Cubic in time, quadratic in memory. The truncation error is "
             "known exactly."),
            ("Iterative subspace methods",
             "The low-energy part of the spectrum",
             "$10^{6}$ to $10^{10}$",
             "Requires only the action of $H$ on a vector, supplied by the "
             "Slater-Condon rules. Applicable where the required window is "
             "narrow."),
            ("Purification with tensor networks",
             "The state as a matrix product operator",
             "no dimension limit",
             "Bounded by entanglement. Imaginary time evolution from the "
             "infinite-temperature state, with a variational error."),
            ("Typical-state sampling",
             "Thermal averages from sampled pure states",
             "no dimension limit",
             "Exploits quantum typicality and avoids the doubled register. "
             "Statistical error bars replace a deterministic bound."),
            ("Quantum hardware",
             "A physical copy of the state",
             "set by the device",
             "Imaginary time evolution, engineered dissipation, or variational "
             "preparation. Presently limited by noise."),
        ],
        note="**The requirement determines the method.** Exact states with "
             "certified error demand the first row and therefore a small active "
             "space. Large systems demand the third or fourth row and give up "
             "the certificate.",
        notes="""
This is the reference slide of part three. Take it slowly.

Direct diagonalization computes everything, at a cost cubic in the dimension. It
stops near a hundred thousand states. In exchange the truncation error is known
exactly, since every weight is available.

Iterative subspace methods, meaning Lanczos and Davidson, never form the matrix.
They require only the ability to apply the Hamiltonian to a vector, which the
Slater-Condon rules supply. They converge the lowest states, which suffices at
low temperature.

Tensor network methods change the question from how many configurations exist to
how entangled the state is. The thermal state is written as a pure state on a
doubled system and evolved in imaginary time from infinite temperature. Thermal
states obey area laws for mutual information, which bounds the resources
required, and this is the route to large systems.

Typical-state sampling uses quantum typicality: a single random state in a large
Hilbert space reproduces thermal averages to a relative accuracy improving
exponentially with system size. Minimally entangled typical thermal states are
the practical algorithm.

Quantum hardware is the eventual route and is not yet competitive at these
sizes. The three listed approaches are under active development.

The closing note is the design principle for anyone building one of these
pipelines.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART III  ·  REPRESENTATION",
        title="Representing the state without materializing it",
        equations=[
            ("storage_eq",
             "The spectral form. Storing the weights and eigenvectors is exact "
             "and requires $m$ rows, where $m$ counts the levels carrying "
             "weight."),
        ],
        bullets=[
            "**A density matrix is quadratic in the dimension.** For a modest "
            "active space that is 192 megabytes in the configuration basis and "
            "34 gigabytes on the qubit register, for one system at one "
            "temperature.",
            "**The spectral form exploits the low rank of the ensemble.** A few "
            "hundred levels carry the weight, so the factored representation is "
            "smaller by orders of magnitude and exact.",
            "**Downstream quantities are contractions over the factors.** "
            "Expectation values, reduced density matrices, entropies, and the "
            "conversion to a matrix product state all consume the factored form "
            "directly.",
        ],
        figure="fig_storage.png",
        figure_caption="Cost of one thermal state at one temperature, by "
                       "representation, on a logarithmic scale.",
        notes="""
Representation determines whether finite-temperature calculations succeed in
practice, and the constraint binds sooner than the arithmetic does.

The numbers: the density matrix in the configuration basis is four thousand nine
hundred squared, which is 192 megabytes. On the full sixteen-qubit register it
would be sixty-five thousand squared, which is 34 gigabytes. Multiply by a
thousand systems and by several temperatures.

The resolution is that diagonalization already produced the state in factored
form. The density matrix is the product of those factors, and forming the
product serves no purpose. Keeping the factors is exact.

The third bullet makes this workable. Everything wanted downstream is a
contraction. An expectation value is a weighted sum over eigenvectors. A reduced
density matrix is a partial trace performed on the factors. The conversion to a
matrix product state uses the factors as the Schmidt decomposition they already
are.

Truncation deserves one sentence. Levels are retained in energy order until the
accumulated weight reaches a threshold, and the discarded weight is recorded, so
each stored state carries its own error bound.
""",
    ),

    # ================================ IV. QUBITS
    dict(
        layout="eq_bullets",
        kicker="PART IV  ·  ENCODING",
        title="Mapping fermionic modes to qubits",
        equations=[
            ("occupation",
             "The occupation number encoding. One qubit per mode, with basis "
             "states in bijection."),
        ],
        bullets=[
            "**The state spaces coincide exactly.** Fock space over $M$ modes "
            "and an $M$-qubit register both have dimension $2^{M}$, requiring "
            "no padding.",
            "**The operator algebras differ.** Fermionic operators on distinct "
            "modes anticommute; qubit operators on distinct wires commute. A "
            "faithful encoding supplies the missing signs.",
            "**The required sign is already known.** It is the parity phase "
            "from slide 5, the parity of the occupied modes below the one being "
            "acted on.",
        ],
        note="Every fermion-to-qubit encoding is a choice of how to store that "
             "parity information. The alternatives differ in the locality of "
             "the resulting operators and in the number of qubits required.",
        notes="""
The state mapping is trivial and the operator mapping is the content.

Take two modes and the state with both occupied. Creating them in one order
differs by a sign from the other order. That sign is physical. It produces
observable interference and is the origin of exchange effects, of the Pauli
principle, and of Fermi statistics generally.

Qubit operators on different wires commute, so a direct substitution loses the
sign and produces a Hamiltonian for distinguishable particles.

The remedy appears already in the algebra. Acting with a creation operator on an
occupation string contributes the parity of the modes below it. An encoding must
reproduce that parity, and the parity of a mode's occupation is read by the Z
operator. So the phase becomes a product of Z operators. Where that product is
placed distinguishes the encodings.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART IV  ·  JORDAN-WIGNER",
        title="The standard encoding",
        equations=[
            ("jw",
             "Annihilation lowers the target qubit and multiplies by the parity "
             "of every earlier one. The number operator becomes a single-qubit "
             "observable."),
            ("bk_weight",
             "Worst-case operator weight. The Bravyi-Kitaev encoding stores "
             "occupation and parity in a binary tree, reducing the weight at "
             "the cost of a less transparent correspondence."),
        ],
        bullets=[
            "**The transformation is exact and invertible.** All commutation "
            "relations, spectra, and expectation values are preserved.",
            "**Diagonal quantities stay local.** Every occupation and every "
            "density correlation becomes a product of $Z$ operators.",
            "**Hopping terms acquire a string.** An operator moving an electron "
            "between two modes acts on every qubit between them, so operator "
            "weight scales with register size.",
        ],
        figure="fig_jw_map.png",
        figure_caption="Schematic. An occupation string on the register, and "
                       "the operator moving an electron between modes 1 and 6.",
        notes="""
Jordan and Wigner published this in 1928 for a one-dimensional spin chain. It
became the default encoding for fermionic quantum simulation.

Read the first equation as two operations. Annihilating mode p lowers qubit p,
which is the combination of X and Y, and multiplies by the parity of all earlier
qubits, which is the string of Z operators. The number operator involves only Z,
so measuring an occupation is a single-qubit measurement.

The locality cost is intrinsic. Fermionic statistics are non-local when
expressed in terms of distinguishable subsystems, and every encoding pays for
that somewhere. Jordan-Wigner pays in operator weight. Bravyi-Kitaev stores
partial sums in a tree so that both occupation and parity are recoverable in
logarithmic weight, at the cost of a less readable correspondence. The parity
encoding sits at the opposite extreme from Jordan-Wigner.

For simulation on a device with limited connectivity, the choice interacts with
hardware topology, and there are encodings designed to match specific lattices
using additional qubits.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART IV  ·  THE PAULI FORM",
        title="A molecular Hamiltonian as a sum of Pauli operators",
        equations=[
            ("pauli_form",
             "After encoding, the Hamiltonian is a weighted sum of Pauli "
             "strings. The term count follows the size of the two-electron "
             "tensor."),
        ],
        bullets=[
            "**A real example.** The eight-orbital Hamiltonian of water gives "
            "3,125 Pauli terms on 16 qubits.",
            "**The weight distribution extends to the register size.** Terms "
            "above weight four come entirely from the parity strings.",
            "**The coefficients are strongly unequal.** Ninety percent of the "
            "total magnitude sits in about a fifth of the terms.",
            "**This structure sets the cost of quantum simulation.** The "
            "coefficient one-norm bounds the gate count for Trotter and "
            "qubitization methods, and reducing it through factorization of the "
            "two-electron tensor has lowered published resource estimates by "
            "several orders of magnitude.",
        ],
        figure="fig_pauli_spectrum.png",
        figure_caption="Jordan-Wigner decomposition of a real active-space "
                       "Hamiltonian, computed for this lecture.",
        notes="""
This is a real decomposition of a real Hamiltonian, computed for the lecture.

Three thousand one hundred and twenty-five terms on sixteen qubits. The count
follows n to the fourth, so doubling the orbital count multiplies the terms by
sixteen.

The left panel shows how many qubits each term acts on. The underlying physics is
one-body and two-body, so without the parity strings the distribution would stop
at weight four. It reaches sixteen. Everything beyond four is the cost of the
encoding.

The right panel matters most for this audience. The coefficients are very
unequal, with ninety percent of the total magnitude reached after about a fifth
of the terms. Three consequences follow. Small terms can be truncated with
controlled error. Commuting terms can be grouped and measured simultaneously,
reducing measurement cost by orders of magnitude. And the one-norm of the
coefficients, which appears directly in simulation cost bounds, can be reduced
substantially by choosing a better representation of the same operator. Double
factorization and tensor hypercontraction do exactly this, and they are the
reason published resource estimates for chemistry on fault-tolerant hardware
have fallen by several orders of magnitude over the last decade.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART IV  ·  PREPARATION",
        title="Preparing a mixed state on a register",
        equations=[
            ("purif_def",
             "The purification. A second register is adjoined and entangled "
             "with the first; tracing it out leaves the thermal state."),
            ("imag_time",
             "The standard construction. Begin from the maximally entangled "
             "state, which purifies the infinite-temperature ensemble, then "
             "evolve in imaginary time on the system register alone."),
        ],
        bullets=[
            "**A quantum computer prepares pure states.** Mixed states arise as "
            "marginals of larger pure states, or from an incoherent process "
            "such as coupling to a bath.",
            "**The doubled state is the thermofield double,** an object with "
            "independent significance in field theory and in holography.",
            "**Imaginary time evolution is non-unitary,** so its "
            "implementation is indirect. Measurement with post-selection, "
            "engineered dissipation, and variational construction are the three "
            "established routes.",
        ],
        figure="fig_purification.png",
        figure_caption="Schematic of the doubled register.",
        notes="""
This construction is standard in both tensor network simulation and quantum
algorithms.

Any mixed state on a register can be written as a pure state on two copies such
that discarding one leaves the mixed state on the other. In the thermal case the
doubled state is called the thermofield double.

The construction in the second equation is the practical recipe. At infinite
temperature the thermal state is proportional to the identity, and its
purification is the maximally entangled state, a product of Bell pairs and
trivial to prepare. Lowering the temperature is imaginary time evolution on the
system half only. In tensor network simulation this is implemented by applying
short-time exponentials of the Hamiltonian.

The thermofield double deserves one extra sentence for this audience because it
appears in a different context entirely. In holographic duality it is the state
dual to an eternal black hole, with the entanglement between the two copies
serving as the geometric connection between two boundaries. The same object,
arrived at from statistical mechanics.

In our own pipeline this step costs nothing, because exact diagonalization
supplies the Schmidt decomposition the purification requires.
""",
    ),

    # ================================ V. USES
    dict(
        layout="eq_stack",
        kicker="PART V  ·  WHAT THEY YIELD",
        title="Quantities determined by a molecular Hamiltonian",
        equations=[
            ("yields",
             "Ground state energy as a function of geometry, its gradient, the "
             "excited spectrum, equilibrium expectation values, the free "
             "energy, and response functions. Each is a distinct experimental "
             "observable."),
        ],
        cards=[
            ("Structure and reactivity",
             "The energy as a function of nuclear position defines the "
             "potential energy surface. Its minima are stable structures, its "
             "saddle points are transition states, and their relative heights "
             "give reaction rates."),
            ("Spectra",
             "Excitation energies and transition intensities give ultraviolet, "
             "visible, X-ray, and magnetic resonance spectra. These are the "
             "primary experimental probes of electronic structure."),
            ("Thermodynamics",
             "The partition function gives free energies, and free energy "
             "differences give equilibrium constants. Whether a reaction "
             "proceeds is a statement about the thermal ensemble."),
        ],
        note="**Predictive accuracy has a sharp threshold.** Reaction rates "
             "depend exponentially on barrier heights, so an error of 4 "
             "kilojoules per mole changes a predicted rate by a factor of five "
             "at room temperature. This is why chemical accuracy is defined at "
             "one part in ten thousand of the total energy.",
        notes="""
This slide answers why anyone computes these objects.

The potential energy surface is the central construct. Every stable molecule is
a minimum on it, every reaction is a path between minima, and every barrier
height is a saddle point. Given the surface one has structures, vibrational
frequencies, and reaction rates.

Spectroscopy is the direct experimental test. Excitation energies and
intensities are measured routinely, and calculations reproducing them are
trusted for quantities that cannot be measured.

Thermodynamics is where the thermal state enters explicitly. Whether a reaction
proceeds at a given temperature is determined by a free energy difference, and
free energy is a property of the ensemble.

The note quantifies the accuracy requirement and explains why this is hard.
Rates depend exponentially on barriers. A factor of five in a rate constant
changes an industrial process from viable to useless. The corresponding energy
error is about one and a half millihartree, roughly one part in ten thousand of
the total electronic energy of a small molecule. Reaching that requires the
correlated treatment described in part two.
""",
    ),

    dict(
        layout="eq_stack",
        kicker="PART V  ·  CHEMISTRY AND INDUSTRY",
        title="Systems whose electronic structure resists calculation",
        equations=[],
        cards=[
            ("Nitrogen fixation",
             "Industrial ammonia synthesis runs at 400 degrees and 200 "
             "atmospheres and consumes one to two percent of world energy "
             "production. The enzyme nitrogenase performs the same conversion "
             "at ambient conditions using an iron-molybdenum cluster. Its "
             "mechanism remains unresolved, because the cluster demands an "
             "active space beyond the reach of exact methods."),
            ("Photochemistry",
             "Singlet fission converts one absorbed photon into two triplet "
             "excitons and offers a route past the efficiency limit of "
             "single-junction photovoltaics. The states involved are "
             "multiconfigurational and require correlated treatment."),
            ("Catalysis and materials",
             "Transition metal centres in homogeneous catalysts, battery "
             "cathode materials, and metalloenzyme active sites all carry "
             "near-degenerate d orbitals. These are the standard failure cases "
             "for single-reference methods."),
        ],
        note="**The pattern is consistent.** Systems of the greatest practical "
             "value are those with many near-degenerate orbitals, and those are "
             "exactly the systems where approximate methods lose reliability "
             "and exact methods become intractable.",
        notes="""
This slide makes the case that these objects matter outside the field.

Nitrogen fixation is the standard example and the numbers are worth stating. The
Haber-Bosch process fixes atmospheric nitrogen into ammonia for fertilizer and
supports roughly half of world food production. It consumes on the order of one
to two percent of global energy and emits over one percent of global carbon
dioxide, because it runs at four hundred degrees and two hundred atmospheres.
Nitrogenase performs the same chemistry at ambient temperature and pressure
using a cluster of seven iron atoms, one molybdenum, and nine sulfurs.
Understanding how would be worth a great deal. The obstacle is that the cluster
has many near-degenerate d orbitals, requiring an active space of order fifty
orbitals, which is ten to the thirty configurations. It is the standard
benchmark target in quantum computing resource estimates for chemistry.

Singlet fission is the photochemistry example. A single photon produces two
triplet excitons, which in principle allows a solar cell to exceed the
Shockley-Queisser limit of about thirty-three percent for a single junction. The
intermediate states are superpositions of several configurations, so no
single-reference method describes them.

The pattern in the note is the argument of this section. Practical importance
and computational difficulty are correlated, because both follow from having
many orbitals close in energy.
""",
    ),

    dict(
        layout="figure_hero",
        kicker="PART V  ·  THE SCALE OF THE PROBLEM",
        title="What the interesting systems demand",
        figure="fig_targets.png",
        figure_caption="Active space sizes typically required for a correlated "
                       "treatment, with the resulting configuration count at "
                       "half filling. Orbital counts are published estimates; "
                       "sources in the companion document.",
        callout=("This gap is the reason for interest in quantum computation.",
                 "An active space of fifty-four orbitals contains $10^{30}$ "
                 "configurations, which exceeds anything that can be stored by "
                 "an enormous margin. A quantum register of one hundred and "
                 "eight qubits holds the corresponding state directly. "
                 "Electronic structure is therefore among the clearest cases "
                 "where a quantum computer addresses a problem of established "
                 "practical value, and it dominates the quantum algorithms "
                 "literature for that reason."),
        stats=[("$10^{30}$", "configurations for nitrogenase"),
               ("108", "qubits for the same space"),
               ("$10^{5}$", "reach of exact diagonalization")],
        notes="""
This slide connects the lecture to this audience's own interests, and it is the
one to linger on.

The vertical axis lists systems by the size of active space a correlated
treatment requires. The horizontal axis is the resulting configuration count on
a logarithmic scale spanning thirty orders of magnitude.

Exact diagonalization reaches the first two rows. Everything below stays out of
reach permanently, because the growth is combinatorial.

The callout is the argument. Fifty-four orbitals gives about ten to the thirty
configurations. A quantum register of one hundred and eight qubits represents
that space directly, because a qubit register is exactly the right kind of
object for the job.

On the resource estimation literature: the first serious estimate for treating
the nitrogenase cofactor on a fault-tolerant quantum computer appeared in 2017
and gave runtimes measured in years. Subsequent work on better representations
of the Hamiltonian, particularly low-rank factorizations of the two-electron
tensor, has reduced that by several orders of magnitude. That progress came from
studying the structure described in part one of this lecture.
""",
    ),

    dict(
        layout="eq_bullets",
        kicker="PART V  ·  CONDENSED MATTER",
        title="The same operator, on a lattice",
        equations=[
            ("hubbard",
             "The Hubbard model. The second-quantized Hamiltonian of slide 6 "
             "with hopping restricted to neighbouring sites and the two-body "
             "tensor reduced to a single on-site term."),
        ],
        bullets=[
            "**It is the standard model of correlated electrons** and is "
            "believed to contain the essential physics of high-temperature "
            "superconductivity in the copper oxides. Its phase diagram remains "
            "unsettled after four decades.",
            "**Finite temperature is essential here.** Phase transitions, "
            "pseudogap behaviour, and transport coefficients are properties of "
            "the thermal ensemble.",
            "**Embedding methods reduce a solid to a small cluster.** Dynamical "
            "mean field theory and its variants require the finite-temperature "
            "properties of an interacting cluster with a handful of orbitals, "
            "solved repeatedly inside a self-consistent loop.",
            "**That inner problem is the object of this lecture,** computed "
            "thousands of times per materials calculation.",
        ],
        note="Methods transfer directly between the two fields. The density "
             "matrix renormalization group originated in condensed matter "
             "physics and became a standard tool in quantum chemistry; coupled "
             "cluster theory travelled in the opposite direction.",
        notes="""
The purpose of this slide is to show that the construction is not
chemistry-specific.

The Hubbard model is the Hamiltonian from part one with a very sparse tensor:
hopping between neighbouring sites, and a repulsion acting only when two
electrons occupy the same site. It was introduced in 1963 and is the standard
model of correlated electrons. Whether it superconducts in the physically
relevant parameter regime remains under debate.

Finite temperature is not optional in this field. A phase diagram is a map of
thermal behaviour. The pseudogap regime of the cuprates is defined by its
temperature dependence. Transport coefficients are thermal averages.

The third and fourth bullets are the connection worth making explicitly.
Dynamical mean field theory treats a solid by identifying a small strongly
correlated region, solving it exactly at finite temperature, and embedding it in
a mean field description of the rest. The inner solver handles an interacting
fermionic Hamiltonian with a handful of orbitals at finite temperature, which is
precisely the object constructed in this lecture.

The note is a historical observation with a practical consequence. Progress in
one field is directly usable in the other, and someone entering from quantum
information can read both literatures as one subject.
""",
    ),

    dict(
        layout="eq_stack",
        kicker="PART V  ·  QUANTUM INFORMATION",
        title="Where these objects enter quantum information science",
        equations=[],
        cards=[
            ("Benchmark and target",
             "Molecular Hamiltonians are the standard testbed for ground state "
             "preparation, phase estimation, and variational algorithms. "
             "Resource estimates for chemistry constitute the most developed "
             "quantitative case for fault-tolerant quantum advantage."),
            ("Gibbs sampling as a primitive",
             "Thermal state preparation is a subroutine in quantum algorithms "
             "for semidefinite programming and for training energy-based "
             "models. Estimating a partition function is classically hard, "
             "which is what makes the primitive valuable."),
            ("Quantum data",
             "Thermal states form a physically meaningful family of mixed "
             "states with tunable purity and entanglement, known exactly at "
             "small size. They are natural test objects for tomography, "
             "classical shadows, and protocols taking quantum states as input."),
        ],
        note="**A quantum Boltzmann machine treats the thermal state as the "
             "model itself,** with the Hamiltonian coefficients as trainable "
             "parameters. Each gradient step requires an expectation value in "
             "that state, placing the preparation cost inside the training "
             "loop.",
        notes="""
This slide states the connections to this audience's own work.

The first card describes the largest application area proposed for quantum
computers. Estimating the ground state energy of a molecular Hamiltonian to
chemical accuracy has a precise statement, a known classical difficulty, and
established practical value. The resource estimation literature for it is more
quantitative than for any other proposed application.

The second card concerns thermal states as a computational tool. Gibbs state
preparation appears as a subroutine in quantum algorithms for semidefinite
programming and in training quantum Boltzmann machines. Computing a partition
function classically is hard in general, so a quantum method that samples from a
Gibbs state without computing the partition function is a genuine primitive.

The third card is the framing relevant to learning problems. States produced by
random circuits carry no meaningful labels. Thermal states of physical
Hamiltonians carry properties one can measure and predict, and at small sizes
they are known exactly, so a protocol can be validated against ground truth.
That combination is uncommon.

The note makes the cost concrete. In a quantum Boltzmann machine the thermal
state is the model, and each gradient step requires an expectation value in it,
so state preparation sits in the inner loop of training.
""",
    ),

    dict(
        layout="two_cards_eq",
        kicker="PART V  ·  ONE INSTANCE",
        title="The construction, applied",
        cards=[
            ("What is built",
             "Exact thermal states of active-space Hamiltonians for a thousand "
             "small organic molecules, on a common eight-orbital register, each "
             "stored in factored form with its truncation error recorded."),
            ("What it supports",
             "Training data for a classifier reading a density matrix directly. "
             "The uniform register makes states comparable, and exact "
             "construction makes the labels trustworthy."),
        ],
        equations=[
            ("spin_sector",
             "The register used, and its dimension."),
        ],
        note="**The open question concerns the label.** Identifying a property "
             "that these states determine, and that a classical description of "
             "the same system fails to recover, is the subject of the companion "
             "talk.",
        notes="""
One slide on the project, kept brief deliberately.

We construct exact thermal states of active-space Hamiltonians for a thousand
small organic molecules, all on the same eight-orbital register, each stored in
the factored form of slide 17 with its discarded weight recorded.

The purpose is training data for a classifier whose input is a density matrix.
The uniform register is what makes examples comparable. Exact construction is
what makes labels trustworthy, since a defect in training data produces no error
signal and degrades a model silently.

The note is the honest closing position. Producing the states is solved.
Identifying a property that requires the quantum description remains open, and
that is the subject of the companion talk.
""",
    ),

    dict(
        layout="summary",
        kicker="SUMMARY",
        title="What to take away",
        columns=[
            ("Structure",
             ["A molecular Hamiltonian contains only one-body and two-body "
              "terms, so $O(n^{4})$ coefficients specify it and the "
              "Slater-Condon rules make it sparse.",
              "The space it acts on has combinatorial dimension. This gap "
              "between description and state defines the subject.",
              "Symmetries block-diagonalize it exactly, reducing both classical "
              "cost and qubit count."]),
            ("Equilibrium",
             ["The Gibbs state follows from the spectrum and maximizes entropy "
              "at fixed mean energy.",
              "Temperature gives a continuous family of mixed states with "
              "controlled purity and entanglement.",
              "The construction method follows from whether an exact error "
              "bound is required."]),
            ("Consequence",
             ["The occupation encoding is exact. Fermionic statistics cost "
              "operator locality and nothing else.",
              "Systems of the greatest practical value are the ones that resist "
              "classical treatment.",
              "That coincidence makes electronic structure the most developed "
              "application area for quantum computation."]),
        ],
        strapline="A compressed operator, an intractable state, and a class of "
                  "problems whose value is established independently.",
        notes="""
Three sentences to close with.

A molecular Hamiltonian is a sparse structured operator specified by a
polynomial number of coefficients, acting on a space of combinatorial dimension.

Its equilibrium state follows immediately from the spectrum, and the difficulty
lies entirely in obtaining and representing that spectrum.

The systems where this difficulty is greatest are the systems of greatest
practical value, and that coincidence is why this field receives the attention
it does from the quantum computing community.

Then take questions. The companion talk covers what we did with a dataset
constructed this way.
""",
    ),

    dict(
        layout="references",
        kicker="REFERENCES",
        title="Principal sources",
        groups=[
            ("Structure",
             ["Helgaker, Jørgensen and Olsen, *Molecular Electronic-Structure "
              "Theory* (2000). Second quantization, the Slater-Condon rules, "
              "and active space methods.",
              "Szabo and Ostlund, *Modern Quantum Chemistry* (1996). The "
              "accessible treatment of the same material."]),
            ("Equilibrium states",
             ["Jaynes, *Phys. Rev.* **106**, 620 (1957). The maximum entropy "
              "characterisation.",
              "Verstraete, García-Ripoll and Cirac, *Phys. Rev. Lett.* **93**, "
              "207204 (2004); White, *Phys. Rev. Lett.* **102**, 190601 "
              "(2009)."]),
            ("Quantum computation",
             ["Jordan and Wigner, *Z. Phys.* **47**, 631 (1928); Bravyi and "
              "Kitaev, *Ann. Phys.* **298**, 210 (2002).",
              "McArdle, Endo, Aspuru-Guzik, Benjamin and Yuan, *Rev. Mod. "
              "Phys.* **92**, 015003 (2020).",
              "Reiher, Wiebe, Svore, Wecker and Troyer, *PNAS* **114**, 7555 "
              "(2017). Resource estimates for nitrogenase."]),
        ],
        note="A complete list, organised by the claim each source supports, is "
             "in the companion document **`Papers/theory_references.md`**. It "
             "marks the sources required to implement the construction "
             "correctly and gives the provenance of every number shown here.",
        notes="""
The companion document is the one to take away. It is organised by what each
source supports and separates the physics from the numerical methods and from
the software.

Three entries to point out. Helgaker, Jørgensen and Olsen for anything in parts
one and two. McArdle and colleagues for the quantum computing side, which covers
encodings and state preparation in more depth than this lecture. Reiher and
colleagues for the resource estimate behind slide 24, which established
nitrogenase as the benchmark target.
""",
    ),
]
