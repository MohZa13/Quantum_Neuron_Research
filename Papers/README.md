# Papers/ — source literature

## `Fermi-Dirac Machines.pdf` — **the source paper**

Defines everything the classifier implements. The equations referenced
throughout this repo's code and docs:

| Reference | Content |
|---|---|
| Sec. II.A | Single-qubit Pauli building blocks |
| **Eq. 16–18** | The neuron: `output = Tr[g_T(H(ω))ρ]`, `g_T(x) = tanh(x/T)`, `H(ω) = Σ_j ω_j H_j` |
| **Eq. 56** | The logistic loss `L^log_T(ω)` — the objective we minimize |
| Sec. VI.C | Training protocol |
| **Thm. 5 / Eq. 63** | The gradient. ⚠️ The paper notebook divides by an extra `T`; corrected everywhere else ([`../docs/INVARIANTS.md`](../docs/INVARIANTS.md) I7) |
| **Sec. VI.D.2 / Fig. 8** | The reproduction target — 2×3 grid, n = 2…7. Image at `../docs/fig_8.png`, digitized into `../results/digitized/` |
| Eq. 113 | 2-qubit TFIM, used by `../notebooks/pennylane/sampling_demo.py` |
| Eq. 115 / 116 | Quantum (Heisenberg) and classical (FCIM) Hamiltonian structures |
| **App. B, p. 40–41 / Eq. B4, B9** | Algorithm 8 — squared-loss gradient, two Hadamard tests (Fig. 10) |
| **App. C.2, p. 45 / Eq. C27** | Algorithm 9 — logistic-loss gradient (Fig. 11) |
| Thm. 13 / Eq. C44, C77 | Algorithm 10 — the loss *value* from the same quantities |

Implemented in `../notebooks/`, `../figures/quantum_training_impls.py`, and
`../tensor-network-testing/`.

> **The most important thing in this paper for our purposes** is the shape of
> the model, not any single equation: the output is **linear in ρ**, which
> fixes the hypothesis class to thresholds on `Tr(ρH)`. That constraint drives
> the entire label-design problem
> ([`../docs/QUANTUM_NEURON.md`](../docs/QUANTUM_NEURON.md) §2).
>
> Note also that the paper's own Fig. 8 labels are `y = sign(Tr(ρ H_target))`
> for a random `H_target` — so a physically meaningful operator substituted for
> the random one is not a new construct, it is the paper's own scheme.

## `QBM Learning of Ground-State Energies.pdf`

Quantum Boltzmann machine background — thermal states as trainable models,
adjacent to the Fermi–Dirac formulation.

## The two decks

Both are **generated** from [`../scripts/presentation/`](../scripts/presentation/)
and share one visual system. Build either with:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=scripts/presentation \
    .venv/bin/python scripts/presentation/build_deck.py {results|theory}
```

| Deck | Audience | Source |
|---|---|---|
| `molecular_hamiltonians_and_thermal_states.pptx` | Conference background lecture. Assumes quantum information, assumes no chemistry. 29 slides | `content_theory.py` |
| `thermal_states_presentation.pptx` | Group meeting. What we built and what it found. 20 slides | `content.py` |

Read the background talk first; it is the prerequisite for the other.

---

## `molecular_hamiltonians_and_thermal_states.pptx` — **the background talk**

Five parts. The fermionic algebra and the Hamiltonian written in it, including
the anticommutation relations, the Fock space construction, the parity phase,
and the Slater-Condon rules. The reductions that make the problem finite. Thermal
states, their construction ordered by reachable size, and their representation.
The translation to qubits. Finally, at length, what these objects are used for,
from industrial nitrogen fixation to quantum algorithm resource estimates.
About 5,000 words of speaker notes.

**House style, checked by the build.** No em dashes. Direct assertion:
contrastive constructions of the form "not X but Y" are reported by the style
lint and rewritten. The slide carries the statement and the mathematics; the
notes carry the explanation. The project appears as one instance of the general
construction, never as the topic. A clean build prints nothing; keep it that way.

Companion citation list: `theory_references.md`, organised by the claim each
source supports.

## `thermal_states_presentation.pptx` — the results deck

20 slides, rebuilt **2026-08-06**. Presenter notes on every slide (~4,000
words). Restructured around the *data structure* rather than the source
dataset, and carrying two results the previous version predates: the
coherence-vs-composition confound, and the positive-control failure of the
Fermi–Dirac objective.

**It is generated, not hand-edited.** `content.py` is the script of the talk — edit prose there, not in PowerPoint,
or the next rebuild discards it. Companion files:

| File | Role |
|---|---|
| `thermal_states_presentation.html` | Pixel-mirror of the same slide model; what the build is checked against |
| `thermal_states_presentation.pdf` | Chrome render of that HTML. **Fonts are substituted** — re-export from the `.pptx` for brand-faithful output |
| `presentation_references.md` | The citation list, organised by what each work supports and marking which were needed to get the *code* right |

## `thermal_states_presentation_final.pdf` — the previous version

18 slides, reconciled to the 1000-molecule dataset 2026-07-28. Kept because it
is the only copy of that version; its `.pptx` source never lived in this repo.
Superseded by the deck above.

---

Not tracked in git by default policy elsewhere, but these PDFs *are* committed —
they are small and the project needs its references pinned.
