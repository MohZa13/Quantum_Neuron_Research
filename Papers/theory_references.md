# References for *Molecular Hamiltonians and Thermal States*

Companion document to `Papers/molecular_hamiltonians_and_thermal_states.pptx`.

Entries are grouped by the part of the talk they support, and each carries a
short note saying what it is being cited for. Where a source was needed to
implement the construction correctly, rather than only to describe it, the note
says so.

Contents:
[1 Foundations](#1-foundations) ·
[2 Mean field theory and density functional theory](#2-mean-field-theory-and-density-functional-theory) ·
[3 Basis sets and integrals](#3-basis-sets-and-integrals) ·
[4 Active spaces and exact diagonalization](#4-active-spaces-and-exact-diagonalization) ·
[5 Thermal states](#5-thermal-states) ·
[6 Construction methods, by reachable size](#6-construction-methods-by-reachable-size) ·
[7 Qubit encodings](#7-qubit-encodings) ·
[8 Preparing thermal states on quantum hardware](#8-preparing-thermal-states-on-quantum-hardware) ·
[9 Applications in chemistry](#9-applications-in-chemistry) ·
[10 Applications in condensed matter physics](#10-applications-in-condensed-matter-physics) ·
[11 Machine learning on quantum states](#11-machine-learning-on-quantum-states) ·
[12 Datasets](#12-datasets) ·
[13 Software](#13-software) ·
[14 Textbooks and reviews](#14-textbooks-and-reviews) ·
[15 This project](#15-this-project)

---

## 1 Foundations

**[1] M. Born and R. Oppenheimer**, *Zur Quantentheorie der Molekeln*, Annalen
der Physik **389**, 457 (1927).
The separation of nuclear and electronic motion. Slide 3: why the nuclei enter
as fixed parameters and why a molecule has a potential energy surface.

**[2] J. C. Slater**, *The Theory of Complex Spectra*, Physical Review **34**,
1293 (1929).
**[2a] E. U. Condon**, *The Theory of Complex Spectra*, Physical Review **36**,
1121 (1930).
The determinant construction that makes a many-electron wavefunction
antisymmetric, and the rules for matrix elements between two determinants.
Together these are the Slater-Condon rules shown on slide 7. They state that a
two-body operator has zero matrix element between configurations differing in
three or more occupied orbitals, which is the origin of all sparsity in the
configuration basis. **Needed for the implementation** of any determinant-driven
eigensolver.

**[3] V. Fock**, *Konfigurationsraum und zweite Quantelung*, Zeitschrift für
Physik **75**, 622 (1932).
Second quantization for fermions. Slides 4 to 6.

**[4] P. Jordan and E. Wigner**, *Über das Paulische Äquivalenzverbot*,
Zeitschrift für Physik **47**, 631 (1928).
The transformation between fermionic operators and spin operators, introduced
here for a one-dimensional chain and now the default encoding for quantum
simulation of fermions. Slides 18 and 19. **Needed for the implementation:** the
sign convention determines the parity strings, and getting it wrong produces a
Hamiltonian with the correct spectrum on some sectors and not on others.

**[5] P. A. M. Dirac**, *Quantum Mechanics of Many-Electron Systems*,
Proceedings of the Royal Society A **123**, 714 (1929).
Cited for the framing of the field: the underlying laws are known and the
difficulty is entirely computational.

---

## 2 Mean field theory and density functional theory

**[6] D. R. Hartree**, *The Wave Mechanics of an Atom with a Non-Coulomb Central
Field*, Proceedings of the Cambridge Philosophical Society **24**, 89 (1928).
**[7] V. Fock**, *Näherungsmethode zur Lösung des quantenmechanischen
Mehrkörperproblems*, Zeitschrift für Physik **61**, 126 (1930).
The self-consistent field idea, and its antisymmetric form. Slide 9.

**[8] C. C. J. Roothaan**, *New Developments in Molecular Orbital Theory*,
Reviews of Modern Physics **23**, 69 (1951).
**[9] G. G. Hall**, *The Molecular Orbital Theory of Chemical Valency VIII*,
Proceedings of the Royal Society A **205**, 541 (1951).
The matrix form of the self-consistent equations in a finite basis, which is the
equation shown on slide 9. **Needed for the implementation:** the generalized
eigenvalue problem with an overlap matrix is exactly how we recover the orbitals
from the stored one-particle matrix.

**[10] P. Hohenberg and W. Kohn**, *Inhomogeneous Electron Gas*, Physical Review
**136**, B864 (1964).
**[11] W. Kohn and L. J. Sham**, *Self-Consistent Equations Including Exchange
and Correlation Effects*, Physical Review **140**, A1133 (1965).
Density functional theory, and the auxiliary non-interacting system whose
orbitals we use as a basis. Slide 9.

**[12] A. D. Becke**, *Density-functional thermochemistry III: The role of exact
exchange*, Journal of Chemical Physics **98**, 5648 (1993).
**[13] C. Lee, W. Yang and R. G. Parr**, *Development of the Colle-Salvetti
correlation-energy formula into a functional of the electron density*, Physical
Review B **37**, 785 (1988).
**[14] P. J. Stephens, F. J. Devlin, C. F. Chabalowski and M. J. Frisch**, *Ab
Initio Calculation of Vibrational Absorption and Circular Dichroism Spectra
Using Density Functional Force Fields*, Journal of Physical Chemistry **98**,
11623 (1994).
The three components of B3LYP as it is actually implemented. This is the
functional the dataset used, so it defines the orbitals we inherit.

**[15] A. J. Cohen, P. Mori-Sánchez and W. Yang**, *Challenges for Density
Functional Theory*, Chemical Reviews **112**, 289 (2012).
The known failure modes, and the reason a single determinant is inadequate for
the systems discussed on slide 23.

---

## 3 Basis sets and integrals

**[16] S. F. Boys**, *Electronic Wave Functions I: A General Method of
Calculation for the Stationary States of Any Molecular System*, Proceedings of
the Royal Society A **200**, 542 (1950).
Gaussian basis functions, which is why the integrals on slide 6 have closed
forms and can be evaluated quickly.

**[17] F. Weigend and R. Ahlrichs**, *Balanced basis sets of split valence,
triple zeta valence and quadruple zeta valence quality for H to Rn*, Physical
Chemistry Chemical Physics **7**, 3297 (2005).
The def2 family, of which def2-SVP is the member used throughout. **Needed for
the implementation:** the basis must match the one the dataset used exactly, or
the recovered orbitals are not the stored ones.

**[18] S. Obara and A. Saika**, *Efficient recursive computation of molecular
integrals over Cartesian Gaussian functions*, Journal of Chemical Physics
**84**, 3963 (1986).
**[19] M. Head-Gordon and J. A. Pople**, *A method for two-electron Gaussian
integral and integral derivative evaluation using recurrence relations*,
Journal of Chemical Physics **89**, 5777 (1988).
How the two-electron integrals of slide 10 are computed in practice. We use a
library implementation rather than writing our own.

---

## 4 Active spaces and exact diagonalization

**[20] B. O. Roos, P. R. Taylor and P. E. M. Siegbahn**, *A complete active
space SCF method (CASSCF) using a density matrix formulated super-CI approach*,
Chemical Physics **48**, 157 (1980).
The active space construction of slides 12 and 13. Note that the
configuration interaction step is performed without the orbital optimization,
which is the deliberate choice described on slide 10.

**[21] P. J. Knowles and N. C. Handy**, *A new determinant-based full
configuration interaction method*, Chemical Physics Letters **111**, 315 (1984).
**[22] J. Olsen, B. O. Roos, P. Jørgensen and H. J. Aa. Jensen**, *Determinant
based configuration interaction algorithms for complete and restricted
configuration interaction spaces*, Journal of Chemical Physics **89**, 2185
(1988).
The determinant-driven formulation that makes it possible to apply the
Hamiltonian to a vector without storing it. **Needed for the implementation:**
the ordering of determinants in these algorithms is the ordering our stored
eigenvectors use, and it must be respected by every consumer of the data.

**[23] C. Lanczos**, *An iteration method for the solution of the eigenvalue
problem of linear differential and integral operators*, Journal of Research of
the National Bureau of Standards **45**, 255 (1950).
**[24] E. R. Davidson**, *The iterative calculation of a few of the lowest
eigenvalues and corresponding eigenvectors of large real-symmetric matrices*,
Journal of Computational Physics **17**, 87 (1975).
The two standard iterative subspace methods, which are the second row of the
table on slide 16.

**[25] E. R. Sayfutyarova, Q. Sun, G. K.-L. Chan and G. Knizia**, *Automated
Construction of Molecular Active Spaces from Atomic Valence Orbitals*, Journal
of Chemical Theory and Computation **13**, 4063 (2017).
**[26] C. J. Stein and M. Reiher**, *Automated Selection of Active Orbital
Spaces*, Journal of Chemical Theory and Computation **12**, 1760 (2016).
Automated alternatives to the fixed window rule of slide 12. These select a
chemically motivated window per molecule, which is better for any single
calculation and worse for the dataset property described on slide 12, because
the resulting spaces are not commensurate across records.

**[27] D. I. Lyakh, M. Musiał, V. F. Lotrich and R. J. Bartlett**,
*Multireference Nature of Chemistry: The Coupled-Cluster View*, Chemical
Reviews **112**, 182 (2012).
Survey of the situations in which a single configuration fails, which is the
motivation for active space methods given on slide 23.

---

## 5 Thermal states

**[28] J. W. Gibbs**, *Elementary Principles in Statistical Mechanics*, Yale
University Press (1902).
The canonical ensemble.

**[29] J. von Neumann**, *Mathematische Grundlagen der Quantenmechanik*,
Springer (1932); English translation, Princeton University Press (1955).
The density matrix and the quantum entropy that the maximum entropy
characterisation refers to.

**[30] E. T. Jaynes**, *Information Theory and Statistical Mechanics*, Physical
Review **106**, 620 (1957).
The Gibbs state as the maximum entropy state at fixed mean energy. Slide 14,
first bullet.

**[31] M. A. Nielsen and I. L. Chuang**, *Quantum Computation and Quantum
Information*, Cambridge University Press, 10th anniversary edition (2010).
Standard reference for the density matrix formalism, for purification, and for
the properties of the trace distance used in our error accounting. Chapters 2
and 9.

---

## 6 Construction methods, by reachable size

This section supports the table on slide 16, in the order the rows appear.

### Tensor network approaches

**[32] S. R. White**, *Density matrix formulation for quantum renormalization
groups*, Physical Review Letters **69**, 2863 (1992).
The density matrix renormalization group.

**[33] U. Schollwöck**, *The density-matrix renormalization group in the age of
matrix product states*, Annals of Physics **326**, 96 (2011).
The modern formulation, and the reference for canonical forms and truncation
error accounting. **Needed for the implementation** of the matrix product state
conversion.

**[34] I. V. Oseledets**, *Tensor-Train Decomposition*, SIAM Journal on
Scientific Computing **33**, 2295 (2011).
The successive singular value decomposition algorithm used to factorize a state
into a chain of tensors. **Needed for the implementation.**

**[35] G. Vidal**, *Efficient Simulation of One-Dimensional Quantum Many-Body
Systems*, Physical Review Letters **93**, 040502 (2004).
**[36] J. Haegeman, J. I. Cirac, T. J. Osborne, I. Pižorn, H. Verschelde and
F. Verstraete**, *Time-Dependent Variational Principle for Quantum Lattices*,
Physical Review Letters **107**, 070601 (2011).
The two standard ways of evolving a tensor network in time, real or imaginary.
Imaginary time evolution is the third row of the table.

**[37] F. Verstraete, J. J. García-Ripoll and J. I. Cirac**, *Matrix Product
Density Operators: Simulation of Finite-Temperature and Dissipative Systems*,
Physical Review Letters **93**, 207204 (2004).
**[38] M. Zwolak and G. Vidal**, *Mixed-State Dynamics in One-Dimensional
Quantum Lattice Systems*, Physical Review Letters **93**, 207205 (2004).
**[39] A. E. Feiguin and S. R. White**, *Finite-temperature density matrix
renormalization using an enlarged Hilbert space*, Physical Review B **72**,
220401(R) (2005).
The purification approach to finite temperature: represent the thermal state as
a pure state on a doubled system and evolve in imaginary time. Slides 16 and 21.

### Sampling approaches

**[40] S. R. White**, *Minimally Entangled Typical Quantum States at Finite
Temperature*, Physical Review Letters **102**, 190601 (2009).
**[41] E. M. Stoudenmire and S. R. White**, *Minimally entangled typical thermal
state algorithms*, New Journal of Physics **12**, 055026 (2010).
The fourth row of the table. Thermal averages are obtained from a Markov chain
over states that are individually cheap to represent.

**[42] A. Hams and H. De Raedt**, *Fast algorithm for finding the eigenvalue
distribution of very large matrices*, Physical Review E **62**, 4365 (2000).
**[43] S. Sugiura and A. Shimizu**, *Canonical Thermal Pure Quantum State*,
Physical Review Letters **111**, 010401 (2013).
Quantum typicality: a single random state reproduces thermal averages in a large
enough Hilbert space. The underlying justification for the sampling row.

### Why the reach is set by entanglement

**[44] M. B. Hastings**, *An area law for one-dimensional quantum systems*,
Journal of Statistical Mechanics, P08024 (2007).
**[45] M. M. Wolf, F. Verstraete, M. B. Hastings and J. I. Cirac**, *Area Laws
in Quantum Systems: Mutual Information and Correlations*, Physical Review
Letters **100**, 070502 (2008).
**[46] J. Eisert, M. Cramer and M. B. Plenio**, *Colloquium: Area laws for the
entanglement entropy*, Reviews of Modern Physics **82**, 277 (2010).
Why a state with bounded entanglement can be written compactly regardless of the
dimension of the space it lives in. Reference [45] is the finite temperature
statement specifically.

**[47] G. K.-L. Chan and S. Sharma**, *The Density Matrix Renormalization Group
in Quantum Chemistry*, Annual Review of Physical Chemistry **62**, 465 (2011).
The practical experience of applying these methods to molecules, whose
interactions are long ranged once the orbitals are placed on a chain.

---

## 7 Qubit encodings

**[48] G. Ortiz, J. E. Gubernatis, E. Knill and R. Laflamme**, *Quantum
algorithms for fermionic simulations*, Physical Review A **64**, 022319 (2001).
The Jordan-Wigner transformation applied to fermionic simulation on a quantum
computer, including the parity string bookkeeping. **Needed for the
implementation.**

**[49] S. B. Bravyi and A. Yu. Kitaev**, *Fermionic Quantum Computation*,
Annals of Physics **298**, 210 (2002).
The Bravyi-Kitaev encoding, which reduces the worst case operator weight from
the number of modes to its logarithm. Mentioned on slide 19 as the standard
alternative.

**[50] J. T. Seeley, M. J. Richard and P. J. Love**, *The Bravyi-Kitaev
transformation for quantum computation of electronic structure*, Journal of
Chemical Physics **137**, 224109 (2012).
A direct comparison of the encodings for molecular Hamiltonians, with explicit
operator forms.

**[51] K. Setia and J. D. Whitfield**, *Bravyi-Kitaev superfast simulation of
electronic structure on a quantum computer*, Journal of Chemical Physics
**148**, 164104 (2018).
A further encoding that trades qubit count for locality.

**[52] J. D. Whitfield, J. Biamonte and A. Aspuru-Guzik**, *Simulation of
electronic structure Hamiltonians using quantum computers*, Molecular Physics
**109**, 735 (2011).
The full transformation of a molecular Hamiltonian into Pauli operators, which
is the calculation shown on slide 20.

**[53] S. Bravyi, J. M. Gambetta, A. Mezzacapo and K. Temme**, *Tapering off
qubits to simulate fermionic Hamiltonians*, arXiv:1701.08213 (2017).
How conserved quantities remove qubits from the register, which is the reduction
described on slide 5.

---

## 8 Preparing thermal states on quantum hardware

**[54] R. P. Feynman**, *Simulating physics with computers*, International
Journal of Theoretical Physics **21**, 467 (1982).
**[55] S. Lloyd**, *Universal Quantum Simulators*, Science **273**, 1073 (1996).
**[56] D. S. Abrams and S. Lloyd**, *Simulation of Many-Body Fermi Systems on a
Universal Quantum Computer*, Physical Review Letters **79**, 2586 (1997).
The origin of quantum simulation, and its extension to fermions.

**[57] A. Aspuru-Guzik, A. D. Dutoi, P. J. Love and M. Head-Gordon**, *Simulated
Quantum Computation of Molecular Energies*, Science **309**, 1704 (2005).
The paper that established molecular electronic structure as a target
application.

**[58] B. M. Terhal and D. P. DiVincenzo**, *Problem of equilibration and the
computation of correlation functions on a quantum computer*, Physical Review A
**61**, 022301 (2000).
**[59] D. Poulin and P. Wocjan**, *Sampling from the Thermal Quantum Gibbs State
and Evaluating Partition Functions with a Quantum Computer*, Physical Review
Letters **103**, 220502 (2009).
**[60] K. Temme, T. J. Osborne, K. G. Vollbrecht, D. Poulin and F. Verstraete**,
*Quantum Metropolis sampling*, Nature **471**, 87 (2011).
The main line of work on preparing Gibbs states on a quantum computer. Slide 21.

**[61] M. Motta, C. Sun, A. T. K. Tan, M. J. O'Rourke, E. Ye, A. J. Minnich,
F. G. S. L. Brandão and G. K.-L. Chan**, *Determining eigenstates and thermal
states on a quantum computer using quantum imaginary time evolution*, Nature
Physics **16**, 205 (2020).
Imaginary time evolution implemented on hardware, which is the direct analogue
of the classical method in the third row of the table on slide 16.

**[62] J. Wu and T. H. Hsieh**, *Variational Thermal Quantum Simulation via
Thermofield Double States*, Physical Review Letters **123**, 220502 (2019).
**[63] D. Zhu et al.**, *Generation of thermofield double states and critical
ground states with a quantum computer*, Proceedings of the National Academy of
Sciences **117**, 25402 (2020).
The variational route to the doubled state of slide 21, and its experimental
demonstration.

**[64] W. Cottrell, B. Freivogel, D. M. Hofman and S. F. Lokhande**, *How to
Build the Thermofield Double State*, Journal of High Energy Physics **2019**,
058 (2019).
The construction from the high energy physics side, where the same object is
central for a different reason.

**[65] C.-F. Chen, M. J. Kastoryano, F. G. S. L. Brandão and A. Gilyén**,
*Quantum Thermal State Preparation*, arXiv:2303.18224 (2023).
Preparation by engineered dissipation, that is by simulating a coupling to a
bath rather than by evolving a pure state.

**[66] A. N. Chowdhury and R. D. Somma**, *Quantum algorithms for Gibbs sampling
and hitting-time estimation*, Quantum Information and Computation **17**, 41
(2017).
Cost estimates for the preparation step, which is the open question referred to
on slide 26.

---

## 9 Applications in chemistry

**[67] B. O. Roos**, *The Complete Active Space Self-Consistent Field Method and
its Applications in Electronic Structure Calculations*, Advances in Chemical
Physics **69**, 399 (1987).
The standard account of what active space methods are used for.

**[68] K. Andersson, P.-Å. Malmqvist and B. O. Roos**, *Second-order
perturbation theory with a complete active space self-consistent field
reference function*, Journal of Chemical Physics **96**, 1218 (1992).
The usual way of adding the correlation outside the active space, and therefore
a description of what the active space alone leaves out.

**[69] M. B. Smith and J. Michl**, *Singlet Fission*, Chemical Reviews **110**,
6891 (2010).
A process in which the distinction between two ways of coupling a pair of spins
is the whole physics. Cited on slide 23 as an example of a property that
occupations alone cannot express.

**[70] L. Salem and C. Rowland**, *The Electronic Properties of Diradicals*,
Angewandte Chemie International Edition **11**, 92 (1972).
The classic treatment of molecules with two unpaired electrons.

### Sources for the systems named on slides 23 and 24

**[70a] J. W. Erisman, M. A. Sutton, J. Galloway, Z. Klimont and
W. Winiwarter**, *How a century of ammonia synthesis changed the world*, Nature
Geoscience **1**, 636 (2008).
The energy and food-supply figures quoted for the Haber-Bosch process.

**[70b] B. M. Hoffman, D. Lukoyanov, Z.-Y. Yang, D. R. Dean and
L. C. Seefeldt**, *Mechanism of Nitrogen Fixation by Nitrogenase: The Next
Stage*, Chemical Reviews **114**, 4041 (2014).
The state of knowledge on the enzyme, and the reason its mechanism remains
open.

**[70c] M. Reiher, N. Wiebe, K. M. Svore, D. Wecker and M. Troyer**,
*Elucidating reaction mechanisms on quantum computers*, Proceedings of the
National Academy of Sciences **114**, 7555 (2017).
The paper that established the iron-molybdenum cofactor as the benchmark target
for quantum computational chemistry. The active space sizes plotted on slide 24
follow this work and its successors.

**[70d] J. Lee, D. W. Berry, C. Gidney, W. J. Huggins, J. R. McClean, N. Wiebe
and R. Babbush**, *Even More Efficient Quantum Computations of Chemistry
Through Tensor Hypercontraction*, PRX Quantum **2**, 030305 (2021).
**[70e] V. von Burg, G. H. Low, T. Häner, D. S. Steiger, M. Reiher, M. Roetteler
and M. Troyer**, *Quantum computing enhanced computational catalysis*, Physical
Review Research **3**, 033055 (2021).
**[70f] M. Motta, E. Ye, J. R. McClean, Z. Li, A. J. Minnich, R. Babbush and
G. K.-L. Chan**, *Low rank representations for quantum simulation of electronic
structure*, npj Quantum Information **7**, 83 (2021).
Successive reductions in the resource cost of simulating chemistry, obtained by
factorizing the two-electron tensor. These are the works referred to on slides
10 and 20 when the coefficient one-norm is discussed.

**[70g] W. Shockley and H. J. Queisser**, *Detailed Balance Limit of Efficiency
of p-n Junction Solar Cells*, Journal of Applied Physics **32**, 510 (1961).
The single-junction efficiency limit that singlet fission is intended to
circumvent.

**[70h] P. E. M. Siegbahn**, *Structures and Energetics for O2 Formation in
Photosystem II*, Accounts of Chemical Research **42**, 1871 (2009).
The electronic structure of the manganese-calcium cluster that oxidises water in
photosynthesis, and the source for the active space estimate quoted for it.

---

## 10 Applications in condensed matter physics

**[71] J. Hubbard**, *Electron correlations in narrow energy bands*, Proceedings
of the Royal Society A **276**, 238 (1963).
The lattice model referred to on slide 25. It has the same second quantized form
as the molecular Hamiltonian with a much sparser coupling tensor.

**[72] A. Georges, G. Kotliar, W. Krauth and M. J. Rozenberg**, *Dynamical mean
field theory of strongly correlated fermion systems and the limit of infinite
dimensions*, Reviews of Modern Physics **68**, 13 (1996).
**[73] G. Kotliar, S. Y. Savrasov, K. Haule, V. S. Oudovenko, O. Parcollet and
C. A. Marianetti**, *Electronic structure calculations with dynamical mean-field
theory*, Reviews of Modern Physics **78**, 865 (2006).
The embedding framework in which a finite temperature calculation on a small
interacting cluster is performed repeatedly inside a self-consistent loop. This
is the third bullet on slide 25.

**[74] E. Gull, A. J. Millis, A. I. Lichtenstein, A. N. Rubtsov, M. Troyer and
P. Werner**, *Continuous-time Monte Carlo methods for quantum impurity models*,
Reviews of Modern Physics **83**, 349 (2011).
The standard solvers for that inner calculation, and a useful comparison point
for the methods in the table on slide 16.

**[75] D. Zgid and G. K.-L. Chan**, *Dynamical mean-field theory from a quantum
chemical perspective*, Journal of Chemical Physics **134**, 094115 (2011).
**[76] G. Knizia and G. K.-L. Chan**, *Density Matrix Embedding: A Simple
Alternative to Dynamical Mean-Field Theory*, Physical Review Letters **109**,
186404 (2012).
The connection between the chemistry and condensed matter formulations, which is
the reason the same construction serves both.

---

## 11 Machine learning on quantum states

**[77] A. He, N. Liu and M. M. Wilde**, *Fermi-Dirac machines as quantizations of
neurons*, arXiv:2605.24386 (2026).
The model our states are training data for. Slide 27. Repository copy:
`Papers/Fermi-Dirac Machines.pdf`.

**[78] M. Schuld and N. Killoran**, *Quantum Machine Learning in Feature Hilbert
Spaces*, Physical Review Letters **122**, 040504 (2019).
**[79] V. Havlíček, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala,
J. M. Chow and J. M. Gambetta**, *Supervised learning with quantum-enhanced
feature spaces*, Nature **567**, 209 (2019).
The usual framing, in which classical data is encoded into a quantum state by a
circuit. Our setting is the opposite: the data is already quantum, and no
encoding step exists.

**[80] H.-Y. Huang, M. Broughton, J. Cotler, S. Chen, J. Li, M. Mohseni,
H. Neven, R. Babbush, R. Kueng, J. Preskill and J. R. McClean**, *Quantum
advantage in learning from experiments*, Science **376**, 1182 (2022).
The case for learning tasks in which the input is a physical quantum state,
which is the regime this dataset is built for.

**[81] M. H. Amin, E. Andriyash, J. Rolfe, B. Kulchytskyy and R. Melko**,
*Quantum Boltzmann Machine*, Physical Review X **8**, 021050 (2018).
**[82] M. Kieferová and N. Wiebe**, *Tomography and generative training with
quantum Boltzmann machines*, Physical Review A **96**, 062327 (2017).
Models in which a thermal state is the model itself rather than the input.
Slide 26, third card.

**[83] M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii,
J. R. McClean, K. Mitarai, X. Yuan, L. Cincio and P. J. Coles**, *Variational
quantum algorithms*, Nature Reviews Physics **3**, 625 (2021).
Context for the near term algorithms that consume these Hamiltonians.

---

## 12 Datasets

**[84] H. Yu, M. Liu, Y. Luo, A. Strasser, X. Qian, X. Qian and S. Ji**, *QH9: A
Quantum Hamiltonian Prediction Benchmark for QM9 Molecules*, NeurIPS Datasets
and Benchmarks Track (2023); arXiv:2306.09549.
The source of the one-particle Hamiltonians and geometries used in our runs.
**Needed for the implementation:** the storage schema, the basis convention, and
in particular the ordering of atomic orbitals in the stored matrices.

**[85] R. Ramakrishnan, P. O. Dral, M. Rupp and O. A. von Lilienfeld**, *Quantum
chemistry structures and properties of 134 kilo molecules*, Scientific Data
**1**, 140022 (2014).
**[86] L. Ruddigkeit, R. van Deursen, L. C. Blum and J.-L. Reymond**,
*Enumeration of 166 billion organic small molecules in the chemical universe
database GDB-17*, Journal of Chemical Information and Modeling **52**, 2864
(2012).
The geometry set that QH9 is built on, and the enumeration it was drawn from.

**[87] K. T. Schütt, M. Gastegger, A. Tkatchenko, K.-R. Müller and
R. J. Maurer**, *Unifying machine learning and quantum chemistry with a deep
neural network for molecular wavefunctions*, Nature Communications **10**, 5024
(2019).
Prediction of the one-particle Hamiltonian by machine learning, which is the
task QH9 was assembled for. Our use of the dataset is different: we take the
Hamiltonians as given and build states from them.

---

## 13 Software

Versions are pinned in `requirements.lock`.

**[88] Q. Sun et al.**, *Recent developments in the PySCF program package*,
Journal of Chemical Physics **153**, 024109 (2020); and **Q. Sun et al.**,
*PySCF: the Python-based simulations of chemistry framework*, WIREs
Computational Molecular Science **8**, e1340 (2018).
Molecule construction, integral evaluation, the transformation to the orbital
basis, determinant ordering, and the configuration interaction kernels. **Needed
for the implementation** in several places where a convention is not documented
and had to be established by direct numerical check.

**[89] V. Bergholm et al.**, *PennyLane: Automatic differentiation of hybrid
quantum-classical computations*, arXiv:1811.04968 (2018).
The Jordan-Wigner mapping utilities and the qubit tapering used to produce the
Pauli decomposition on slide 20.

**[90] C. R. Harris et al.**, *Array programming with NumPy*, Nature **585**,
357 (2020).
**[91] P. Virtanen et al.**, *SciPy 1.0: fundamental algorithms for scientific
computing in Python*, Nature Methods **17**, 261 (2020).
**[92] J. D. Hunter**, *Matplotlib: A 2D graphics environment*, Computing in
Science and Engineering **9**, 90 (2007).
**[93] A. Collette**, *Python and HDF5*, O'Reilly (2013); The HDF Group,
*Hierarchical Data Format, version 5*.
Numerical, plotting and storage infrastructure. Every figure in the talk is
produced by `scripts/presentation/figures_theory.py`.

**[94] M. Fishman, S. R. White and E. M. Stoudenmire**, *The ITensor Software
Library for Tensor Network Calculations*, SciPost Physics Codebases **4** (2022).
**[95] X.-Z. Luo, J.-G. Liu, P. Zhang and L. Wang**, *Yao.jl: Extensible,
Efficient Framework for Quantum Algorithm Design*, Quantum **4**, 341 (2020).
The tensor network and state vector libraries used for the trainers that consume
this dataset.

**[96] G. M. Machado, M. M. Oliveira and L. A. F. Fernandes**, *A
physiologically-based model for simulation of color vision deficiency*, IEEE
Transactions on Visualization and Computer Graphics **15**, 1291 (2009).
Every categorical colour in this deck was checked against simulated protanopia
and deuteranopia using this model, together with a lightness and contrast check.
The palette and its constraints are recorded in `scripts/presentation/style.py`.

---

## 14 Textbooks and reviews

**[97] T. Helgaker, P. Jørgensen and J. Olsen**, *Molecular Electronic-Structure
Theory*, Wiley (2000).
The reference for parts I and II of the talk. Second quantization in chapter 1,
the self-consistent field equations in chapter 3, and the frozen core reduction
in chapter 10. **Needed for the implementation** of the effective one-body
matrix and the core energy shown on slide 12.

**[98] A. Szabo and N. S. Ostlund**, *Modern Quantum Chemistry*, Dover (1996).
The more accessible treatment of the same material, and the one to recommend to
someone approaching this from physics.

**[99] R. G. Parr and W. Yang**, *Density-Functional Theory of Atoms and
Molecules*, Oxford University Press (1989).
Standard reference for section 2.

**[100] S. McArdle, S. Endo, A. Aspuru-Guzik, S. C. Benjamin and X. Yuan**,
*Quantum computational chemistry*, Reviews of Modern Physics **92**, 015003
(2020).
**[101] Y. Cao, J. Romero, J. P. Olson, M. Degroote, P. D. Johnson, M. Kieferová,
I. D. Kivlichan, T. Menke, B. Peropadre, N. P. D. Sawaya, S. Sim, L. Veis and
A. Aspuru-Guzik**, *Quantum Chemistry in the Age of Quantum Computing*, Chemical
Reviews **119**, 10856 (2019).
**[102] B. Bauer, S. Bravyi, M. Motta and G. K.-L. Chan**, *Quantum Algorithms
for Quantum Chemistry and Quantum Materials Science*, Chemical Reviews **120**,
12685 (2020).
The three standard reviews of the intersection. Reference [100] is the best
single entry point for the material in part IV of the talk.

---

## 15 This project

Not external citations, but the provenance of every number and figure shown.
Paths are relative to the repository root.

| Slide | What is shown | Produced by |
|---|---|---|
| 8 | A real sector Hamiltonian and the thermal state built from it | `figures_theory.py::hamiltonian_matrix`, from `results/qh9_dense_cas8-6_kT0p25.h5` |
| 9 | Orbital energies of water, and the active window | `figures_theory.py::orbital_ladder`, from `data/QH9Stable.db` via `qthermal/orbitals.py` |
| 10 | The integral transformation | `qthermal/hamiltonian.py` (Module D) |
| 11 | Configuration counts | `figures_theory.py::dimension`, combinatorial |
| 12 | The frozen-core reduction and the resulting dimension | `qthermal/active_space.py`, `qthermal/hamiltonian.py` |
| 14 | A real spectrum and its Boltzmann weights | `figures_theory.py::boltzmann`, from `results/qh9_dense_cas8-8_kT0p1.h5` |
| 15 | Participation and purity against temperature | `figures_theory.py::effective_rank`, 250 molecules |
| 16 | The construction methods | this document, sections 4 and 6 |
| 17 | Storage cost by representation | `figures_theory.py::storage`; the pipeline's format is `qthermal/io_hdf5.py` |
| 19 | The Jordan-Wigner map | schematic; the implementation is `qthermal/encode.py` |
| 20 | Pauli decomposition of a real Hamiltonian | `figures_theory.py::pauli_spectrum`, 3,125 terms for water at eight active orbitals |
| 21 | The purification | schematic; the implementation is `qthermal/mps.py` |
| 24 | Active space sizes required by named systems | `figures_theory.py::targets`; orbital counts from [70b], [70c], [70h] |
| 27 | The dataset built here | `qthermal/run.py`, `results/qh9_dense_cas8-8_kT0p1.h5` |

The companion talk on what was done with the resulting dataset is
`Papers/thermal_states_presentation.pptx`, with its own reference list in
`Papers/presentation_references.md`.

Design decisions taken during construction, and the reasoning behind them, are
recorded in `docs/DECISIONS.md`. Measured results, including negative ones, are
in `docs/RESEARCH_LOG.md`.
