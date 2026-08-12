# Glossary

*Written for a reader without a quantum-chemistry or quantum-ML background.
Terms are grouped by area; within each group, ordered so earlier entries
explain later ones.*

---

## Quantum states

**State vector / wavefunction (|ψ⟩)** — a list of complex amplitudes, one per
possible configuration. Squaring an amplitude gives the probability of finding
the system in that configuration.

**Density matrix (ρ)** — the general description of a quantum state, needed
whenever the state is a *mixture* rather than a single wavefunction. A square
matrix whose entries are indexed by pairs of configurations.

- **Diagonal entries `ρ_ii`** — the probability of configuration *i*. This is
  the **classical** part: a probability distribution, nothing more.
- **Off-diagonal entries `ρ_ij` (i≠j)** — the **coherences**. They record that
  configurations *i* and *j* are phase-locked in superposition, not merely
  uncertain.

> The distinction in one image: a coin that has landed but you have not looked
> at it, and a coin in the superposition (|H⟩+|T⟩)/√2, have **identical
> diagonals** (50/50). Everything that separates "quantum" from "merely
> unknown" lives off the diagonal. See [`QUANTUM_NEURON.md`](QUANTUM_NEURON.md) §2.

**Coherence** — in this project, always means the off-diagonal structure of ρ
in the computational (determinant) basis. Measured here as *coherence share*:
the fraction of ρ's total weight sitting off the diagonal. Median 6.7% across
the 1000-molecule set — these states are diagonal-dominated.

**Pure vs mixed state** — pure = one wavefunction (`ρ = |ψ⟩⟨ψ|`); mixed = a
statistical blend of several. Thermal states are mixed.

**Thermal state / Gibbs state** — the equilibrium mixture at temperature T:
each energy eigenstate `|E_k⟩` weighted by the Boltzmann factor
`p_k ∝ exp(−E_k/kT)`. Cold ⇒ one state dominates; hot ⇒ many contribute.

**Eigenblock** — this repo's storage format for a thermal state:
`ρ = Σ_k p_k |ψ_k⟩⟨ψ_k|`, stored as the weights `p` (shape `(m,)`) plus the
eigenvectors `civecs` (shape `(m, dim)`). **Never** expanded into a dense
matrix — at CAS(8,8) that would be 65,536², about 4×10⁹ complex entries.

**Trace distance** — a distance between two quantum states, in [0,1]. 0 =
identical, 1 = perfectly distinguishable. Used here to measure how far an
interacting thermal state is from its non-interacting counterpart.

**Entropy (of a mixture)** — `−Σ p_k ln p_k`. How spread out the mixture is.
0 means a single pure state dominates.

**Purification** — a trick that represents a mixed state as a *pure* state on a
larger system (original + an added "ancilla"). Tracing the ancilla away
recovers ρ exactly. Used by `qthermal/mps.py` so tensor-network methods, which
prefer pure states, can handle thermal states.

---

## Quantum chemistry

**Orbital** — a one-electron wavefunction. Each can hold two electrons (spin
up and spin down).

**MO / AO** — Molecular Orbital (delocalized over the molecule) vs Atomic
Orbital (the basis functions centered on each atom that MOs are built from).

**def2-SVP** — the atomic-orbital basis set QH9 uses. Fixes how many AOs each
element contributes (5 for H, 14 for C/N/O/F here).

**AO ordering** — the convention for which basis function is index 0, 1, 2…
Different codes use different conventions. Getting this wrong silently corrupts
every matrix. **See [`INVARIANTS.md`](INVARIANTS.md) #1 — this cost the project
284 GB of data.**

**HOMO / LUMO / gap** — Highest Occupied and Lowest Unoccupied Molecular
Orbital; the gap between them is the cheapest single indicator of chemical
reactivity and of how easily the molecule is thermally excited.

**Fock matrix (F)** — the effective one-electron Hamiltonian. QH9 stores one
per molecule. Solving `F C = S C ε` recovers the orbitals `C` and their
energies `ε`.

**Overlap matrix (S)** — atomic orbitals are not orthogonal; `S` records how
much each pair overlaps. This is why the eigenproblem is *generalized*.

**B3LYP / Kohn–Sham** — the density-functional method QH9 was computed with.
Its orbitals are not the same thing as Hartree–Fock orbitals. **We use them as
stored, without re-running SCF** — see [`INVARIANTS.md`](INVARIANTS.md) #2.

**SCF** — Self-Consistent Field, the iterative loop that converges orbitals.
The pipeline deliberately never runs it.

**Active space / CAS(N,M)** — N electrons in M orbitals, treated exactly; all
other electrons frozen into an effective background. Production setting is
**CAS(8,8)**: the 4 highest occupied + 4 lowest empty frontier orbitals.

**Frozen core** — the inner electrons excluded from the active space, folded
into a scalar energy (`ecore`) plus an effective one-body potential (`h1eff`).

**CASCI** — Complete Active Space Configuration Interaction: exact
diagonalization *within* the active space, with orbitals held fixed. (CASSCF
would additionally re-optimize the orbitals; we do not.)

**Determinant / configuration** — one specific assignment of electrons to
orbitals. The CAS(8,8) sector has 4,900 of them: C(8,4) alpha strings × C(8,4)
beta strings = 70 × 70.

**Sector** — the subspace with fixed electron count and fixed spin projection
(here `nelecas = 8`, `S_z = 0`). All the work happens inside one sector.

**ERI / two-electron integrals (g)** — the electron–electron repulsion terms.
**QH9 does not contain these** — computing them is the pipeline's main new
physics (module D). Stored as the full `ncas⁴` tensor in *chemist notation*
`(pq|rs)`.

**1-RDM / natural occupations** — the one-particle reduced density matrix, and
its eigenvalues. Each occupation lies in [0,2]. Values near 0 or 2 mean a
well-defined filled/empty orbital; values near 1 signal **static correlation**.

**Static correlation** — the situation where no single determinant dominates;
several configurations are comparably important. Scored here as
`Σ_i min(n_i, 2−n_i)`, which is 0 for a clean closed shell.

**Singlet / triplet** — two electrons in two orbitals can pair their spins
(singlet, total spin 0) or align them (triplet, total spin 1). In the S_z = 0
sector both are built from the *same two determinants* and differ only in the
**sign of the off-diagonal element** — which makes the distinction pure
coherence. This is the basis of the project's best label candidate.

**Degree of unsaturation (DoU)** — countable from the chemical formula:
`(2C + 2 + N − H − F)/2`. Counts rings plus π-bonds. Free to compute, which is
exactly why a label correlated with it is suspect.

**Conjugation** — a connected network of alternating single/double bonds, over
which electrons delocalize. The structural driver of quantum behaviour in these
molecules.

---

## Qubits and encodings

**Qubit** — a two-level quantum system. `n` qubits span a `2^n`-dimensional
space.

**Jordan–Wigner (JW) transformation** — the standard map from fermions
(electrons, which anticommute) to qubits. One qubit per spin-orbital, so
CAS(8,8) → **16 qubits**. The anticommutation is carried by strings of Z
operators.

**Wire ordering — blocked vs interleaved** — two ways to lay spin-orbitals onto
wires:

- **blocked**: wires `0..ncas−1` = all alpha, `ncas..2·ncas−1` = all beta.
  Every JW reordering sign is +1 (a pure scatter). Best measured MPS bond
  dimensions; also reads ~10× more connected-ZZ signal.
- **interleaved**: alpha_p on wire `2p`, beta_p on wire `2p+1` — same-orbital
  pairs adjacent. Better for nearest-neighbour ansätze, and the layout in which
  the spin-exchange operator `S²_od` becomes string-free and 4-local.

  *These are not interchangeable; pick per consumer and record which.*

**Pauli operators (I, X, Y, Z)** — the 2×2 building blocks. A **Pauli string**
is a tensor product over all wires, e.g. `Z0 X3 X5`. Any operator on n qubits
is a weighted sum of the 4ⁿ Pauli strings.

- **Z-only strings are diagonal.** A model built only from them can read *only*
  `diag(ρ)` — i.e. it is a classical model. This is the ablation that proves a
  label needs coherence.
- With **real** ρ (which ours are), any string with an **odd number of Y's**
  has expectation exactly zero.

**Extended-Heisenberg basis** — this project's 248-term operator basis at
CAS(8,8): Z on every wire, ZZ on every wire pair, and XX/YY within each spin
block. Exactly the weight-≤2 strings that symmetry does not force to zero.
Count is `4·ncas² − ncas`.

**Z₂ tapering** — exploiting exact parity symmetries to delete qubits at zero
information cost. Here 16 → 14 qubits; expectations carry over up to a recorded
± sign, and the classifier's core linear algebra drops 16-fold.

**MPS (matrix product state) / bond dimension χ** — a compressed
representation of a quantum state as a chain of small tensors. χ measures how
much entanglement crosses each link; small χ = cheap. **DMRG** is the standard
algorithm for finding low-energy MPS.

---

## The classifier

**Fermi–Dirac machine / quantum neuron** — the model from `Papers/Fermi-Dirac
Machines.pdf`. Its neuron output is

```
    output(ρ) = Tr[ g_T(H(ω)) ρ ] ,   g_T(x) = tanh(x/T),   H(ω) = Σ_j ω_j P_j
```

Trainable weights `ω`, fixed Pauli basis `P_j`, temperature `T`. The decision
rule is `sign(Tr[ρH])`.

**Hypothesis class** — the set of rules a model can express. Here it is exactly
*"threshold on a linear functional of ρ"*. Knowing this precisely is what makes
label design a solvable problem rather than guesswork.

**Logistic / log-loss** — the training objective,
`L = (1/M) Σ_m Tr[ T·ln(I + e^{−y_m H/T}) ρ_m ]`. Crucially **linear in each
ρ_m**.

**R± (label aggregation)** — because the loss is linear in each state, it
depends on the training set only through `R± = Σ_{y=±1} ρ_i`. Sum the states of
each class once, and per-epoch cost becomes **independent of dataset size**.
The project's single largest optimization (measured 174× at 1,000 samples).

**FCIM** — Fully Connected Ising Model: the *classical* comparison model
(all-to-all ZZ + Z, hence diagonal). The natural baseline the quantum model
must beat.

**Parameter counts** — quantum Heisenberg `6n−3`; FCIM `n(n+1)/2`.

**Chebyshev loss** — a matrix-free polynomial approximation to the spectral
loss, avoiding `O(2^{3n})` diagonalization. Takes over above ~n = 9.

**Dephasing (Δ)** — the operation that erases all off-diagonal entries and
keeps the diagonal. Exactly "what a classical model sees". The identity
`Tr(ρA) − Tr(Δ(ρ)A) = Tr(ρ A_od)` is the foundation of the coherence-label
construction.

**A_od** — an operator with its diagonal zeroed. `Tr(ρ A_od)` depends on the
coherences of ρ and on nothing else.

---

## The hybrid network

Full treatment: [`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md); code in
[`qnn/`](../qnn/README.md).

**Activation observable** — the paper's central construct. Apply an activation
function to an *operator* rather than to a number: `A = φ(B)`, meaning
diagonalize `B` and apply `φ` to each eigenvalue. The result is itself an
observable, so it has an expectation value `Tr[Aρ]` in any state. This is what
distinguishes the paper's quantization of a neuron from every earlier proposal.

**Activation variable** — the scalar `a = Tr[φ(B)ρ]` that comes out when the
activation observable is measured against an input state. It is a *number*, so
an ordinary classical neuron can consume it. That collapse is the whole reason
the hybrid architecture is tractable.

**Hybrid quantum–classical network** — paper §VII.C: quantum data in, one layer
of quantized neurons, then ordinary classical layers. The architecture this
project builds. The paper defines its forward pass and leaves training open.

**Quantum observable network** — paper §VII.B: the *other* proposal, where every
layer stays operator-valued. Its gradients compose Fréchet-derivative
superoperators and the paper leaves even their efficient implementation open.
Not what we build; the contrast is why §VII.C is the tractable one.

**Fréchet derivative `Dφ(B)[X]`** — the derivative of a matrix function: how
`φ(B)` changes when `B` moves in direction `X`. **It is not `φ'(B)`.** In the
eigenbasis of `B` it is a Hadamard product with the divided-difference matrix.
Everything hard about training a quantum neuron is here.

**First divided difference `φ^[1](a,b)`** — `(φ(a) − φ(b))/(a − b)`, and `φ'(a)`
when `a = b`. The entries of the matrix above. Computing it accurately near
`a = b` needs care; see [`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md) §6.

**δ-weighted state aggregate `Rᵢ`** — `(1/M) Σ_m δ_{m,i} ρ_m`, the deep
generalization of `R±`. The quantum layer's gradient depends on the dataset only
through one such matrix per neuron, so the number of eigendecompositions per
epoch stays independent of dataset size. Unlike `R±` it must be re-formed each
epoch, because δ moves when the parameters do — that is the cost of depth.

**Operator pool** — the fixed tuple `(H₁,…,H_J)` a quantum neuron's
pre-activation is built from: `Bᵢ = Σⱼ Θᵢⱼ Hⱼ`. It plays the role the *input
vector* plays in a classical neuron.

**Commuting pool** — a pool whose operators all commute (so all diagonal in one
basis). Then the whole network, at any depth, reads only `diag(ρ)` — forward
pass *and* gradient. This is what makes the Z-only ablation a theorem;
[`INVARIANTS.md`](INVARIANTS.md) I15.

**Saturation** — the mean divided difference of a neuron, in units of `φ'(0)`.
Near 1 the neuron is in its linear regime; near 0 its spectrum is far wider than
the temperature `T`, every divided difference has collapsed, and the neuron is
**dead** — not slow, dead. Logged every epoch.

**Spectral-scale initialization** — `σ = T/√(mean Tr[Hⱼ²]/K)`, which puts the
RMS eigenvalue of each `Bᵢ` at the activation's bending scale. The quantized
analogue of He/Glorot initialization; it equalizes the operator's spectral
spread with the nonlinearity's scale rather than fan-in with output variance.

---

## This repo's own jargon

| Term | Meaning |
|---|---|
| **Module A…J** | The pipeline stages, lettered in each `qthermal/` module docstring. `qnn/` re-uses the letters A…F for its own six stages |
| **Half 1 / Half 2** | The thermal-state factory vs the classifiers. They now meet on real data; what is missing is a defensible label |
| **The pool** | `qnn`'s operator basis `{Hⱼ}`. `quantum` reaches off the diagonal; `z_only` and `diagonal_full` provably cannot, at any depth — the ablation |
| **The seam** | Where the classical chain rule meets the quantum layer: `∂ℒ/∂Θᵢⱼ = Tr[Hⱼ·Dφ(Bᵢ)[Rᵢ]]`. The thing the source paper leaves open |
| **sat / rho(B)** | Per-epoch quantum-layer diagnostics in the training logs: mean divided difference in units of `φ'(0)`, and largest eigenvalue of the pre-activation operator. `sat → 0` with `rho(B) ≫ T` means the layer is going dead |
| **The bridge** | `scripts/export_thermal_training.py` — where real states become training records, and where the label is chosen |
| **kT tag** | HDF5 group naming: `kT = 0.1` → `kT_0p1000` |
| **keep cap** | Ceiling on stored eigenvectors per ensemble. Default `max(1024, dim//4)` = **1225** at CAS(8,8). `--keep-cap 0` lifts it |
| **cap_hit** | Flag meaning the cap, not the weight cutoff, bounded the truncation — the recorded tail then exceeds the requested cutoff |
| **truncation_error** | Boltzmann weight that fell outside the stored states. **Always recorded, never silently dropped** |
| **Quantumness audit** | The per-block diagnostics: entropy, natural occupations, static correlation, leading-determinant weight, trace distance to the non-interacting reference |
| **Gaussian reference** | The g = 0 (non-interacting) state, solved in closed form, used as the "no electron interaction" baseline |
| **Certified tail bound** | The Krylov solver's rigorous upper bound on discarded Boltzmann weight — what makes an incomplete spectrum trustworthy |
| **Resume safety** | `complete=True` written last; incomplete groups deleted and rewritten on rerun |
