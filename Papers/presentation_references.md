# References — *Thermal States as Training Data for a Quantum Neuron*

Companion document to `Papers/thermal_states_presentation.pptx`.

Organised by **what each work supports**, not alphabetically, because the point
of the list is traceability: for any claim, equation, or line of code in the
talk you should be able to find the thing it rests on.

Entries marked **[code]** were needed to implement something *correctly* — a
convention, a sign, an algorithm, an error bound — as opposed to only to
motivate or describe it. Where a choice in the code departs from the cited
source, the departure is stated.

Repository cross-references point into `Quantum_Neuron_Research`.

---

## 1. The model

**[1] A. He, N. Liu, M. M. Wilde**, *Fermi–Dirac machines as quantizations of
neurons*, arXiv:2605.24386v1 [quant-ph] (23 May 2026).
Repository copy: `Papers/Fermi-Dirac Machines.pdf`.

The source of the entire classifier. Specifically used:

| What | Where in [1] | Where in the talk / code |
|---|---|---|
| Canonical quantization of a neuron; the activation *observable* `φ(H(θ))` | §II.A | slide 2 |
| Fermi–Dirac objective and its gradient | §II.B | slides 2, 16 |
| Reduction to a classical neuron for a **commuting** Hamiltonian | §II.A (Eqs. 14–15) | slides 13–14; the ablation |
| Logistic-loss training for binary classification | §II.E.2 | slides 10, 15, 16 |
| Quantized ReLU / softplus, sigmoid, swish | §III | `qnn/activations.py` **[code]** |
| Quantized erf, Gaussian-smoothed ReLU (GReLU), GeLU | §IV | `qnn/activations.py` **[code]** |
| Hybrid quantum–classical gradient-estimation algorithms | §II.C | slide 14, hardware remark |
| Numerical experiments on Haar-random states (the setting we reproduce as a baseline) | §VI | slide 11, "94–96%" |
| **Hybrid network: quantum first layer, classical layers above** | §VII.C, Eqs. 150–151 | `qnn/`, slide 14 |
| Fréchet derivative of `tanh` (Eq. A4) and its integral form (Eqs. A6, A15) | Appendix A | `docs/HYBRID_BACKPROP.md` §4, §6 **[code]** |
| Fully quantum observable network and its gradient superoperators; Remarks 5–6 | §VII.B, Appendix G | `docs/DECISIONS.md` D12 — why we did *not* build this |

**Two deliberate departures from [1]**, both documented in the repository:

- *The training rule for §VII.C is not in [1].* The paper writes the forward
  pass and states that "one can take advantage of […] the standard
  backpropagation algorithm", then leaves simulation and training explicitly
  open. The rule is derived in `docs/HYBRID_BACKPROP.md` §5 from [16]–[18]
  below. **[code]**
- *The paper's `dfj` divides the gradient by `T`*, an extraneous factor relative
  to its own Theorem 5 / Eq. (63). The reference notebook is preserved verbatim
  including this factor; every optimized implementation uses `dH/dω_j = H_j`.
  See `docs/INVARIANTS.md` I7.

---

## 2. The data source

**[2] H. Yu, M. Liu, Y. Luo, A. Strasser, X. Qian, X. Qian, S. Ji**, *QH9: A
Quantum Hamiltonian Prediction Benchmark for QM9 Molecules*, NeurIPS Datasets
and Benchmarks Track (2023); arXiv:2306.09549.

The 130,831-molecule Kohn–Sham Hamiltonian dataset used for every production
run. **[code]** for the raw-SQLite schema `data(id, N, Z, pos, Ham)` and the
basis convention.

> **The single most expensive correctness point in this project.** The raw
> SQLite Hamiltonians are stored in PySCF atomic-orbital ordering already; the
> AO-reordering helpers in the QHBench code apply to *processed* and
> model-output matrices only. Applying them to the raw blobs double-transforms
> and silently corrupts every record. Evidence and the audit that found it:
> `data/qh9_raw_sqlite_audit.md`, `docs/INVARIANTS.md` I1.

**[3] R. Ramakrishnan, P. O. Dral, M. Rupp, O. A. von Lilienfeld**, *Quantum
chemistry structures and properties of 134 kilo molecules*, Scientific Data
**1**, 140022 (2014). — QM9, the geometry set QH9 is built on.

**[4] A. D. Becke**, *Density-functional thermochemistry. III. The role of exact
exchange*, J. Chem. Phys. **98**, 5648 (1993); **C. Lee, W. Yang, R. G. Parr**,
Phys. Rev. B **37**, 785 (1988); **P. J. Stephens, F. J. Devlin,
C. F. Chabalowski, M. J. Frisch**, J. Phys. Chem. **98**, 11623 (1994).
— B3LYP, the functional QH9's stored orbitals were converged with.

**[5] F. Weigend, R. Ahlrichs**, *Balanced basis sets of split valence, triple
zeta valence and quadruple zeta valence quality for H to Rn*, Phys. Chem. Chem.
Phys. **7**, 3297 (2005). — def2-SVP, the basis. **[code]**: the basis name
must match QH9's exactly or the recovered orbitals are not the stored ones.

---

## 3. Electronic structure: what the pipeline computes

**[6] T. Helgaker, P. Jørgensen, J. Olsen**, *Molecular Electronic-Structure
Theory*, Wiley (2000).

The standard reference for everything in Part II of the talk. **[code]** for:
second quantization and the two-electron integral convention (chemists'
notation `(pq|rs)`, Ch. 1); the generalized eigenproblem `FC = SCε` and
Löwdin/overlap handling (Ch. 3); **the frozen-core effective Hamiltonian** —
the dressed one-body term and the core energy — (Ch. 10).

**[7] B. O. Roos, P. R. Taylor, P. E. M. Siegbahn**, *A complete active space
SCF method (CASSCF) using a density matrix formulated super-CI approach*, Chem.
Phys. **48**, 157 (1980). — The active-space construction. We run CASCI (no
orbital optimization) on the *stored* Kohn–Sham orbitals; the deliberate choice
not to re-converge is `docs/DECISIONS.md` D1.

**[8] Q. Sun et al.**, *Recent developments in the PySCF program package*,
J. Chem. Phys. **153**, 024109 (2020); and **Q. Sun et al.**, *PySCF: the
Python-based simulations of chemistry framework*, WIREs Comput. Mol. Sci.
**8**, e1340 (2018).

**[code]**, heavily. Determinant (string) ordering in `pyscf.fci.cistring`,
`fci.direct_spin1.pspace` (used to build the sector matrix — a 34× speed-up
over the naive route), `fci.spin_op` for `⟨S²⟩`, and
`fci.addons.transform_ci_for_orbital_rotation`.

> **A convention that had to be established empirically, not read off:** the
> transpose in `transform_ci_for_orbital_rotation(det, ncas, nelec, U0.T)` is
> load-bearing. It is the new-basis→old-basis map, the *reverse* of the
> function's documented forward convention, and it was confirmed by checking
> `Hv − Ev` directly rather than assumed. `docs/RESEARCH_LOG.md` 2026-07-20.

**[9] E. R. Davidson**, *The iterative calculation of a few of the lowest
eigenvalues and corresponding eigenvectors of large real-symmetric matrices*,
J. Comput. Phys. **17**, 87 (1975).
**[10] P. J. Knowles, N. C. Handy**, *A new determinant-based full
configuration interaction method*, Chem. Phys. Lett. **111**, 315 (1984).

The matrix-free Krylov solver of slide 9. **[code]**: [10] is the
determinant-driven `σ`-vector formulation PySCF implements and is why the
solver never forms the sector matrix.

**[11] G. C. Wick**, *The evaluation of the collision matrix*, Phys. Rev.
**80**, 268 (1950). — Wick's theorem, behind the closed-form non-interacting
(free-fermion) reference solver used for the "distance from free" audit.

> **Caveat stated in the talk's notes:** that reference omits mean-field
> electron repulsion, which inflates the distance for compact molecules. It is
> used as a diagnostic, never as a label. `docs/OPEN_QUESTIONS.md` Q5.

---

## 4. Fermions to qubits

**[12] P. Jordan, E. Wigner**, *Über das Paulische Äquivalenzverbot*,
Z. Phys. **47**, 631 (1928).

**[13] G. Ortiz, J. E. Gubernatis, E. Knill, R. Laflamme**, *Quantum algorithms
for fermionic simulations*, Phys. Rev. A **64**, 022319 (2001).
**[14] J. T. Seeley, M. J. Richard, P. J. Love**, *The Bravyi–Kitaev
transformation for quantum computation of electronic structure*, J. Chem. Phys.
**137**, 224109 (2012).

**[code]** for the parity-string signs. Two facts the implementation depends
on, both asserted in `tests/qthermal/test_encode.py`:

- Under **blocked** wire order (all α modes, then all β) PySCF's determinant
  convention — α creators ascending, then β creators ascending — *is* ascending
  wire order, so every Jordan–Wigner reordering sign is `+1` and the map is a
  pure scatter.
- Under **interleaved** order (α_p, β_p adjacent) each determinant picks up
  `(−1)^Σ_p n_α(p)·|{q<p : n_β(q)=1}|`.

Wire ordering is a per-consumer choice, and the measured optimum reversed our
prediction: blocked gives ≈2× smaller MPS bond dimensions and reads ≈10× more
connected two-mode signal, while interleaved is what makes the spin-exchange
operator string-free and 4-local. `docs/DECISIONS.md` D5.

**[15] N. C. Rubin, R. Babbush, J. McClean**, *Application of fermionic marginal
constraints to hybrid quantum algorithms*, New J. Phys. **20**, 053020 (2018).
— Background for reading fermionic 1- and 2-RDM information out of weight-≤2
Pauli expectation values, which is what the 248-component feature vector is.

---

## 5. The gradient: functions of matrices

These are what `docs/HYBRID_BACKPROP.md` is built on and are **[code]** for
`qnn/quantum_layer.py` and `qnn/activations.py`.

**[16] Ju. L. Daleckii, S. G. Krein**, *Integration and differentiation of
functions of Hermitian operators and applications to the theory of
perturbations*, American Mathematical Society Translations, Series 2, **47**,
1–30 (1965).

**[17] R. Bhatia**, *Matrix Analysis*, Springer GTM 169 (1997) — Theorem V.3.3
and Ch. V generally (the divided-difference / Löwner formulation).

**[18] N. J. Higham**, *Functions of Matrices: Theory and Computation*, SIAM
(2008) — Theorem 3.11 (Fréchet derivative via divided differences), and Ch. 4
on numerically evaluating them.

The three properties the implementation actually uses, proved in
`docs/HYBRID_BACKPROP.md` §4.1: **self-adjointness** of `Dφ(B)[·]` — this is
what makes reverse mode possible and saves a factor of ≈150 in our pool;
**well-definedness under degeneracy**, which matters because our pools contain
many commuting terms and the spectra really are degenerate at initialization;
and **structure preservation**, which keeps the arithmetic real.

**[19] W. Gautschi**, *Orthogonal Polynomials: Computation and Approximation*,
Oxford (2004) — Gauss–Legendre quadrature, used for the divided difference
`φ^[1](a,b) = ∫₀¹ φ'(sa+(1−s)b) ds` when `|a−b| < 10⁻³`, where the difference
quotient loses all significance. `docs/DECISIONS.md` D14. **[code]**

**[20] D. P. Kingma, J. Ba**, *Adam: A Method for Stochastic Optimization*,
ICLR 2015; arXiv:1412.6980. — The optimizer in every training run shown.

**[21] K. He, X. Zhang, S. Ren, J. Sun**, *Delving Deep into Rectifiers*,
ICCV 2015; and **X. Glorot, Y. Bengio**, AISTATS 2010. — The classical
initialization schemes the "spectral-scale" initialization `σ = T/√J` is the
quantized-neuron analogue of (`docs/HYBRID_BACKPROP.md` §9). **[code]**: a
quantized neuron whose spectrum is much wider than `T` is *dead*, not slow,
because the divided differences vanish exponentially.

---

## 6. Tensor networks

**[22] I. V. Oseledets**, *Tensor-train decomposition*, SIAM J. Sci. Comput.
**33**, 2295 (2011). — TT-SVD, the algorithm `qthermal/mps.py` implements to
factorize the system leg. **[code]**

**[23] U. Schollwöck**, *The density-matrix renormalization group in the age of
matrix product states*, Ann. Phys. **326**, 96 (2011). — Canonical forms,
bond truncation, and the standard error accounting. **[code]**

**[24] S. R. White**, *Density matrix formulation for quantum renormalization
groups*, Phys. Rev. Lett. **69**, 2863 (1992). — DMRG.

**[25] F. Verstraete, J. J. García-Ripoll, J. I. Cirac**, *Matrix product
density operators: simulation of finite-temperature and dissipative systems*,
Phys. Rev. Lett. **93**, 207204 (2004).
**[26] A. E. Feiguin, S. R. White**, *Finite-temperature density matrix
renormalization using an enlarged Hilbert space*, Phys. Rev. B **72**,
220401(R) (2005).

Thermal states as purifications on a doubled Hilbert space — the framework
slide 8 sits in. Our construction is the special case where the Schmidt
decomposition is *already known* because the state was produced by exact
diagonalization, so the ancilla bond is the thermal rank `m` exactly and no
truncation occurs on that cut.

**[27] S. R. White**, *Minimally entangled typical quantum states at finite
temperature*, Phys. Rev. Lett. **102**, 190601 (2009).
**[28] E. M. Stoudenmire, S. R. White**, *Minimally entangled typical thermal
state algorithms*, New J. Phys. **12**, 055026 (2010).

METTS — the proposed replacement backend on slide 18.

**[29] M. B. Hastings**, *An area law for one-dimensional quantum systems*,
J. Stat. Mech. (2007) P08024.
**[30] M. M. Wolf, F. Verstraete, M. B. Hastings, J. I. Cirac**, *Area laws in
quantum systems: mutual information and correlations*, Phys. Rev. Lett.
**100**, 070502 (2008).
**[31] J. Eisert, M. Cramer, M. B. Plenio**, *Colloquium: Area laws for the
entanglement entropy*, Rev. Mod. Phys. **82**, 277 (2010).

Why a modest bond dimension suffices at kT = 0.1 — [30] is the finite-
temperature statement specifically. Note this is a *working hypothesis* for our
mapped molecular Hamiltonian, whose Coulomb terms are long-range on the chain;
it is supported by measured bond profiles and by long experience with DMRG in
quantum chemistry [32], not proved for our case.

**[32] G. K.-L. Chan, S. Sharma**, *The density matrix renormalization group in
quantum chemistry*, Annu. Rev. Phys. Chem. **62**, 465 (2011).

---

## 7. Quantum information background

**[33] M. A. Nielsen, I. L. Chuang**, *Quantum Computation and Quantum
Information*, Cambridge (2010).

**[code]** for two specific results the error accounting uses: the trace
distance and its operational meaning, and **monotonicity of the trace distance
under the partial trace** (Thm. 9.2 / §9.2.1), which is what converts the MPS's
per-bond discarded mass into the bound `‖ρ − ρ̃‖₁ ≤ 2‖|Ψ⟩ − |Ψ̃⟩‖₂` on slide 8.

**[34] E. T. Jaynes**, *Information theory and statistical mechanics*,
Phys. Rev. **106**, 620 (1957). — The Gibbs state as the maximum-entropy state
at fixed mean energy (slide 3).

**[35] A. Streltsov, G. Adesso, M. B. Plenio**, *Colloquium: Quantum coherence
as a resource*, Rev. Mod. Phys. **89**, 041003 (2017).
**[36] T. Baumgratz, M. Cramer, M. B. Plenio**, *Quantifying coherence*,
Phys. Rev. Lett. **113**, 140401 (2014).

The framework behind slides 11–12: the **dephasing channel** `Δ` in a fixed
basis as the free operation, and coherence measured relative to it. The
identity the talk turns on — `Tr(ρA) − Tr(Δ(ρ)A) = Tr(ρ A_od)` — is elementary,
but "what a classical model sees is `Δ(ρ)`" is the resource-theoretic statement
these make precise. Our reported quantity, `‖ρ_od‖_F² / Tr ρ²`, is the
normalized `ℓ₂`-coherence of [36] in the determinant basis.

**[37] G. Vidal, R. F. Werner**, *Computable measure of entanglement*,
Phys. Rev. A **65**, 032314 (2002). — Logarithmic negativity, considered and
rejected as a label because it is nonlinear in `ρ` and therefore outside the
single neuron's hypothesis class (`docs/QUANTUM_NEURON.md` §5, candidate 5).

---

## 8. Software

Cited because the numbers in the talk cannot be reproduced without the same
stack. Exact versions: `requirements.lock`.

**[38] C. R. Harris et al.**, *Array programming with NumPy*, Nature **585**,
357 (2020).
**[39] P. Virtanen et al.**, *SciPy 1.0: fundamental algorithms for scientific
computing in Python*, Nature Methods **17**, 261 (2020).
**[40] J. D. Hunter**, *Matplotlib: A 2D graphics environment*, Computing in
Science & Engineering **9**, 90 (2007).
**[41] A. Collette**, *Python and HDF5*, O'Reilly (2013); The HDF Group,
*Hierarchical Data Format, version 5*. — The interchange format for every
`results/*.h5`.
**[42] V. Bergholm et al.**, *PennyLane: Automatic differentiation of hybrid
quantum-classical computations*, arXiv:1811.04968 (2018/2022). — Used for the
Jordan–Wigner mapping utilities, qubit tapering, and the optimized notebook
implementations.
**[43] X.-Z. Luo, J.-G. Liu, P. Zhang, L. Wang**, *Yao.jl: Extensible, Efficient
Framework for Quantum Algorithm Design*, Quantum **4**, 341 (2020).
**[44] M. Fishman, S. R. White, E. M. Stoudenmire**, *The ITensor Software
Library for Tensor Network Calculations*, SciPost Phys. Codebases **4** (2022).
— [43] and [44] back the Julia trainers in `tensor-network-testing/`.
**[45] G. Landrum et al.**, *RDKit: Open-source cheminformatics*,
https://www.rdkit.org. — Bond perception from geometry in the conjugation
screen; the only non-academic source in this list, and it is used for a
triage heuristic, not for any reported physics.

---

## 9. Figures and presentation

**[46] G. M. Machado, M. M. Oliveira, L. A. F. Fernandes**, *A physiologically-
based model for simulation of color vision deficiency*, IEEE Transactions on
Visualization and Computer Graphics **15**, 1291 (2009).

**[code]**, and not a decorative citation. Every categorical colour in the
deck's figures was checked against protanopia and deuteranopia simulated with
this model at severity 1.0, together with an OKLCH lightness-band and
chroma-floor check and a contrast check against the slide surface. Three of the
four hues in the original brand palette failed and were re-stepped. The palette
and its constraints are recorded in `scripts/presentation/style.py`; charts
with unconstrained neighbour adjacency (scatter, small multiples) are capped at
three series because only the first three slots clear the all-pairs test.

**[47] B. Ottosson**, *A perceptual color space for image processing* (2020),
https://bottosson.github.io/posts/oklab/. — The OKLab/OKLCH space the
lightness, chroma and ΔE checks are computed in.

**[48] E. R. Tufte**, *The Visual Display of Quantitative Information*, 2nd ed.,
Graphics Press (2001). — Background for the figure conventions used
throughout: recessive grids, no truncated bar baselines, direct labelling in
preference to legends where it fits.

---

## 10. This project's own artifacts

Not external citations, but the provenance for every number on the slides.
All paths are relative to the repository root.

| Slide | Claim | Produced by |
|---|---|---|
| 3, 6 | ensemble diagnostics; `p₀` spread 16–98%, median 50%; `r(gap, entropy) = −0.82` | `scripts/presentation/build_cache.py` → `results/presentation_cache.npz`, from `results/qh9_dense_cas8-8_kT0p1.h5` |
| 6 | thermal rank, retention rule, certified truncation | `qthermal/thermal.py`, `qthermal/io_hdf5.py` |
| 7 | 248-component feature vector; `K = 1024` retains 99.8% | `qthermal/encode.py`, `scripts/spin_labels.py` |
| 8 | purification construction and its bound | `qthermal/mps.py` (Module J) |
| 9 | certified tail bound; dimension 853,776 | `qthermal/diagonalize.py::IterativeWindowSolver` |
| 10 | 94% held-out on the frontier-gap label | `scripts/presentation/figures.py::gap_training` (same recipe as `scripts/demo_train_curve.py`) |
| 11 | feature weight 99.6 / 0.39 / 0.026% | `scripts/presentation/build_cache.py` |
| 12 | off-diagonal share median **6.7%**; Spearman **0.79** vs degree of unsaturation | `scripts/presentation/build_cache.py`, computed from the eigenblocks |
| 13–15 | `S²` split; quantum − classical = +0.00; descriptor baselines | `scripts/spin_labels.py`, `scripts/train_spin_comparison.py` → `results/spin_comparison_metrics.json` |
| 14 | hybrid gradient; commuting-pool theorem | `docs/HYBRID_BACKPROP.md`, `qnn/`, `tests/qnn/test_pools.py` |
| 16 | positive control: 100% at loss 3.52 vs 66% at loss 1.37 | `results/spin_comparison_metrics.json` (`control.exact_solution`) |
| 17 | active-space adequacy: high-unsaturation decile median edge slack 0.167 | `docs/RESEARCH_LOG.md` ~2026-07-30 |

Narrative context, negative results and open questions:
`docs/RESEARCH_LOG.md`, `docs/OPEN_QUESTIONS.md`, `docs/QUANTUM_NEURON.md`,
`docs/DECISIONS.md`, `docs/INVARIANTS.md`.

> **One reproduction note worth recording.** The 6.7% coherence share and the
> Spearman 0.79 had been carried in the research log since ~2026-07-25 from a
> script that was subsequently lost, and were flagged as unreproduced
> (`docs/OPEN_QUESTIONS.md` Q6). They were recomputed from the eigenblocks for
> this talk and both reproduce exactly. The surviving CSV,
> `results/coherence_share_kT0p1.csv`, turns out to hold a *different*
> quantity — the off-diagonal share of the 248-component **Pauli feature
> vector** (median 0.02%), not of the density matrix — which is also reproduced,
> to four significant figures, by `build_cache.py`.
