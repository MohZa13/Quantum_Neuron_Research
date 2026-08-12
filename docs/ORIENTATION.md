# Orientation — what this project is, in plain language

*For a reader with no prior context. No quantum background assumed.
Vocabulary: [`GLOSSARY.md`](GLOSSARY.md).*

---

## 1. The goal

Build a **quantum neuron** that classifies molecules by looking directly at
their quantum state — and demonstrate that it can do something no classical
model can.

That last clause is the whole point. A quantum machine-learning model that
merely reproduces what logistic regression already does is a curiosity. The
research question is whether there is a task where the *quantum* structure of
the input is doing real work.

## 2. Why molecules, and why "thermal states"

A molecule's electrons occupy a quantum state. At absolute zero they sit in the
single lowest-energy configuration (the *ground state*). Heat it up and the
electrons spread over many configurations at once, weighted by a Boltzmann
factor — this mixture is a **thermal state** (a Gibbs state).

Thermal states are the right input for this project because a single
temperature knob, kT, dials the state between two extremes:

- **cold** → one dominant electron configuration → essentially classical
- **hot** → many configurations superposed and interfering → genuinely quantum

We work at kT = 0.1 Hartree, which is about 32,000 K. That is not a laboratory
scenario. It is chosen because it is the regime where the state is a rich
quantum mixture but still differs sharply *between molecules* — which is
exactly what a labeled dataset needs. (At kT = 0.25 every molecule was
uniformly hot; at 0.025, uniformly cold. See `RESEARCH_LOG.md` 2026-07-15.)

## 3. Where the molecules come from

**QH9** is a public dataset of 130,831 small organic molecules (H, C, N, O, F;
up to ~20 atoms) with their density-functional Hamiltonians precomputed at the
B3LYP/def2-SVP level. We take the molecules at their stored equilibrium
geometries, exactly as given.

QH9 provides only *one-electron* data — how a single electron feels the
molecule's average field. It contains **no electron–electron interaction**.
Supplying that missing physics is the pipeline's main new computation, and it
is what makes the resulting states genuinely correlated rather than a product
of independent orbitals.

## 4. How a molecule becomes a thermal state

Solving for all electrons exactly is impossible — the state space grows
exponentially. So we do what quantum chemistry always does: pick an **active
space**, the handful of electrons and orbitals nearest the chemical action,
solve *that* exactly, and freeze the rest into an effective background.

Production setting is **CAS(8,8)**: 8 electrons in 8 orbitals — the 4 highest
occupied and 4 lowest empty frontier orbitals. Inside that window the problem
is a 4,900 × 4,900 matrix per molecule (every way of arranging 4 spin-up and 4
spin-down electrons among 8 orbitals), which we diagonalize exactly.

Stage by stage — each letter is one module in `qthermal/`:

| | Stage | What happens |
|---|---|---|
| **A** | `loader` | Read one molecule from the SQLite database; empirically verify its coordinate units (Ångström vs Bohr is never assumed) |
| **B** | `orbitals` | Rebuild the molecular orbitals by solving `H C = S C ε`; fix their arbitrary sign gauge |
| **C** | `active_space` | Choose the frontier window; derive every downstream dimension from it |
| **D** | `hamiltonian` | **The new physics**: compute the electron–electron repulsion integrals and fold the frozen core into an effective potential |
| **E** | `diagonalize` | Solve for the energy eigenstates — dense, or matrix-free Krylov for larger spaces |
| **F** | `thermal` | Form the Boltzmann mixture at each temperature; run the "quantumness audit" |
| **G** | `io_hdf5` | Write it all to a resume-safe HDF5 file |
| **H** | `run` | Command-line driver, multiprocessing across molecules |
| **I** | `encode`/`encode_run` | Map states onto qubits (Jordan–Wigner) and extract 248 Pauli features |
| **J** | `mps` | Convert eigenblocks into purification matrix-product states for the tensor-network trainers |

Roughly 190 seconds per molecule at production settings on this machine.

## 5. The classifier — a Fermi–Dirac machine

The model comes from the paper *Fermi-Dirac Machines* (`Papers/`). Its neuron
outputs

```
    output(ρ) = Tr[ g_T(H(ω)) · ρ ]        with  g_T(x) = tanh(x/T)
    H(ω)      = Σ_j ω_j P_j                 (P_j = Pauli operators, ω trainable)
```

In words: build a Hamiltonian out of trainable weights, pass it through a
Fermi–Dirac-like function, and read off its expectation on the input state.
The decision is `sign(Tr[ρ H])`.

Two consequences drive everything downstream:

1. **The output is linear in ρ.** So the model can express exactly one kind of
   rule: a threshold on a linear functional of the state. That is a *narrow*
   hypothesis class — and knowing it precisely is what makes label design
   tractable rather than guesswork.
2. **Training therefore only ever sees `R± = Σ_{y=±1} ρ_i`** — the summed state
   of each class. This is what makes training cost independent of dataset size
   after a one-time aggregation (the project's main optimization, measured up
   to 174× at 1,000 samples).

Details, and why this matters for choosing labels: [`QUANTUM_NEURON.md`](QUANTUM_NEURON.md).

## 5b. …and a network built out of them

Since 2026-08-05 we also have the paper's other proposal (its §VII.C): **one
layer of these quantum neurons reading ρ directly, then ordinary classical
layers on top**. The paper defines it and stops there —

> "We leave it open to future work to simulate the performance and training of
> hybrid quantum–classical neural networks."

The reason it stops is that the chain rule has to cross from a classical layer
into an *operator-valued* nonlinearity, and the derivative of a matrix function
`φ(B)` is not `φ'(B)`. Working that out is
[`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md); the code is [`qnn/`](../qnn/README.md).
The rule turns out to be the classical one with three substitutions:

```
    classical   dL/dw_ij = mean_m  δ_{m,i} · φ'(z_{m,i}) · x_{m,j}
    hybrid      dL/dΘ_ij =         Tr[ H_j · Dφ(B_i)[R_i] ]
```

the input vector becomes an *operator pool*, the sum over samples becomes a
matrix-valued aggregate `R_i`, and multiplication by `φ'(z)` becomes the Fréchet
derivative of the activation observable.

Two things change and one does not. Consequence 1 above is **lifted** — the
composite is no longer linear in ρ, so ratios and other nonlinear labels come
into range. Consequence 2 is **weakened** — `R±` no longer collapses the dataset
once and for all, though one aggregate per neuron per epoch still keeps the
expensive part independent of dataset size. What does *not* change is the thing
the project rests on: a diagonal operator pool still reads only `diag(ρ)`, now
provably at any depth.

## 6. Where the project actually stands

**Half 1 — the thermal-state factory — works and has produced data.**
1000 molecules at CAS(8,8), kT = 0.1 Ha, exact, with recorded truncation
errors; plus 248 Pauli features per molecule, a Z₂-tapered 14-qubit basis, and
a certified Krylov backend demonstrated up to dimension 853,776.

**Half 2 — the classifiers — work, and now on real data.**
The paper reproduction and its optimized rewrite agree to machine precision and
are benchmarked to 10 qubits; they still train on Haar-random states exactly as
in the source paper's Fig. 8, and are the reference implementations. The
experiments run on real molecular thermal states:
`scripts/train_spin_comparison.py` (one neuron) and
`scripts/train_hybrid_spin.py` (the network).

**The label is the open work item.** `scripts/spin_labels.py` produces real
`{ρ_m, y_m}` and both models train on it. What we do *not* have is a defensible
`y_m` — one determined by coherence and not predictable from chemistry. On the
two physical candidates tried so far, a **diagonal** model that provably cannot
see coherence ties or beats the quantum one (`⟨S²⟩`: 98.0 vs 97.0; `c`: 96.3 vs
97.3), and both sit only ~4 points above plain chemical descriptors. Whatever
those models are reading, it is not quantum structure.

The machinery is not the blocker: on a synthetic label built to be
coherence-only, the same code reaches 83.0% while the diagonal ablation sits at
chance. What is missing is a *physical* label with that property.

## 7. Why the label is the hard part

The first instinct — label molecules by HOMO–LUMO gap, or by how much
correlation they have — fails a specific test. A quick demonstration
(`scripts/demo_train_curve.py`) does classify 1000 real molecules by
median-split HOMO–LUMO gap from the 248 Pauli features, and it generalizes to
held-out molecules. But 99.7% of that feature weight is **occupation
information** — which orbitals are filled — and a classical model reads that
just as well. The quantum machinery is not earning its keep.

Two measured findings sharpen the problem:

- The states are **diagonal-dominated**: median off-diagonal coherence is 6.7%
  across the 1000-molecule set.
- Coherence *magnitude* is **confounded with composition**: it correlates with
  degree of unsaturation at Spearman 0.79, and degree of unsaturation is
  countable from the chemical formula for free.

So "label by how quantum the molecule is" is, empirically, "label by how many
double bonds it has" — a classical label wearing a disguise.

The way out is to construct a label that is a **linear functional of ρ with the
diagonal stripped out**: provably invisible to any classical or Z-only model,
yet exactly inside the classifier's hypothesis class. That construction, the
candidate labels, and how to screen them before training are the subject of
[`QUANTUM_NEURON.md`](QUANTUM_NEURON.md) §4–§6. It is the project's live front.

## 8. What to read next

- The research problem in depth → [`QUANTUM_NEURON.md`](QUANTUM_NEURON.md)
- What to work on → [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md)
- What is already known → [`RESEARCH_LOG.md`](RESEARCH_LOG.md)
- How to run anything → [`WORKFLOWS.md`](WORKFLOWS.md)
- What every file is → [`REPO_MAP.md`](REPO_MAP.md)
