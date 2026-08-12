# scripts/ — analysis and bridge tools

Standalone tools that sit *around* the `qthermal/` pipeline: selecting which
molecules to run, converting pipeline output into training data, and
demonstrating results.

Not a package. Run as `python -m scripts.<name>` (works because the repo root
is on the path) or `python scripts/<name>.py`. Always `.venv/bin/python`.

Full command lines: [`../docs/WORKFLOWS.md`](../docs/WORKFLOWS.md).

---

## `screen_conjugation.py` — pick molecules before spending compute

Streams the whole QH9 database and scores every molecule on three tiers, so a
targeted thermal run costs hours instead of weeks.

| Tier | Metric | Cost | What it does and does not tell you |
|---|---|---|---|
| 1 | Degree of unsaturation, from the formula | free | Counts rings + π-bonds. **Does not distinguish conjugated from isolated unsaturation** |
| 2 | Frontier gap + count of levels within kT of mid-gap, from `eigh(F,S)` | free | The *direct* proxy for thermal coherence. Gap tracks mixing at r = −0.78 |
| 3 | Largest connected conjugated π-subsystem, via RDKit bond perception | moderate | The true structural conjugation metric. **Optional** — degrades to Tiers 1–2 if RDKit is absent, with a warning |

Output is sorted ascending by gap (most quantum first). **Resumable** via a
`.partial` append-file, fsync'd every 500 rows — a killed run continues where
it stopped.

Full-database output: `results/qh9_conjugation_screen_full.csv` (130,812 rows).
This is the selection instrument for targeted runs — feed ids to
`qthermal.run --indices`.

## `export_thermal_training.py` — ★ the bridge

Turns a Module-H run file into `{ρ_m, y_m}` training records for the Julia
trainers (`../tensor-network-testing/`) and any downstream classifier.

States are exported in **eigenblock** form — weights `p` plus JW amplitudes
`amps`, with parity signs folded in and the shared `basis_indices` map stored
once. A dense ρ is never built (65,536² at CAS(8,8)). Recover a state vector
with `v = zeros(2**Q); v[basis_indices] = amps[k]`.

**The label is the point of this file.** `--label` accepts:
- `h5:<key>` — a per-block diagnostic from the run file
- `csv:<path>:<col>` — join an external table on molecule idx

> ⚠️ **The default `h5:static_corr` is an acknowledged placeholder** — a
> one-body quantity a classical model reads directly. Choosing a defensible
> label is the project's open question:
> [`../docs/QUANTUM_NEURON.md`](../docs/QUANTUM_NEURON.md) §4–§5.

Planned additions ([`../docs/QUANTUM_NEURON.md`](../docs/QUANTUM_NEURON.md) §8):
`--label offdiag:s2` (coherence-only spin-coupling label) and `--residualize`
(strip the classical-descriptor component).

## `spin_labels.py` — singlet / triplet-open-shell observables

Computes, for every molecule in a run file, the thermal expectation of total
spin and splits it into the part a classical model can see and the part only
coherence carries:

```
    S² = D + S²_od          on the S_z = 0 sector (where S² = S₋S₊)

    D       diagonal    "HOW MANY unpaired electrons"  — readable from diag(ρ)
    S²_od   off-diagonal "HOW those spins are COUPLED" — singlet vs triplet,
                         100% coherence by construction
```

The sector `S²` matrix is molecule-independent, so it is built **once** from
PySCF's `contract_ss` (its diagonal is cross-checked against an independently
derived open-shell count) and everything after is a sparse apply against the
stored eigenblock — exact, no per-root loop, no dense ρ. Optionally writes the
qubit-projected ρ stack the classifier comparison consumes.

## `train_spin_comparison.py` — quantum vs classical neuron

The comparison itself. Both models are the *same* Fermi-Dirac machine — same
loss, optimizer, temperature, split, epoch budget — differing **only** in
whether the operator pool reaches off the diagonal:

| model | pool | H |
|---|---|---|
| classical | I, Z, ZZ (the paper's FCIM) | strictly diagonal ⇒ sees only `diag(ρ)` |
| quantum | + XX, YY | reaches off the diagonal |
| classical_full | unconstrained diagonal | the strongest diagonal model there is |

Also computes the **R± screening metric** `‖offdiag(R₊−R₋)‖/‖diag(R₊−R₋)‖`,
which ranks a label's quantum-visibility in seconds without training, a
classical-descriptor baseline for the composition confound, and a **positive
control** (synthetic purely off-diagonal label with a known exact solution) that
tests whether the apparatus can detect an advantage at all.

`--replot` re-renders the figure from the stored JSON — no retraining, no ρ stack.

> **Result (2026-08-05): quantum − classical = +0.00 points** on both `⟨S²⟩` and
> `c`, and the positive control revealed that the Fermi-Dirac loss does not chase
> off-diagonal signal on these states. See `../docs/RESEARCH_LOG.md` and
> `../docs/OPEN_QUESTIONS.md` Q11.

## `train_hybrid_spin.py` — ★ the hybrid network on real states

The architecture of `Papers/Fermi-Dirac Machines.pdf` §VII.C — one layer of
Fermi–Dirac quantum neurons reading ρ directly, then a classical MLP — trained
with the rule derived in
[`../docs/HYBRID_BACKPROP.md`](../docs/HYBRID_BACKPROP.md) and implemented in
[`../qnn/`](../qnn/README.md). The paper defines this model and says training it
is left to future work; this is that experiment, on real molecules.

Same 1000 molecules, same labels, same seed and the **same stratified split** as
`train_spin_comparison.py`, so the deep and shallow results read side by side.

Four models per label, identical except for the quantum layer's operator pool:

| model | pool | what it tests |
|---|---|---|
| `quantum` | I, Z, ZZ + XX, YY | reaches off the diagonal |
| `z_only` | I, Z, ZZ | **the ablation** — provably reads only `diag(ρ)`, at any depth |
| `diagonal_full` | all `K` basis projectors | the strongest diagonal model that exists, so a quantum win cannot be blamed on under-parameterisation |
| `quantum_shallow` | I, Z, ZZ + XX, YY | quantum pool, *no classical layers* — **the depth ablation** |

`--project-qubits k` restricts the register to the 2ᵏ most-populated
determinants of the loaded stack. The quantum layer costs `O(K³)` per neuron and
is independent of dataset size, so this is the only lever that makes a sweep
affordable: at K = 1024 an epoch is 3.6 s, at K = 256 it is 0.14 s.
`--replot` re-renders from the stored JSON without retraining or a ρ stack.

> **The ablation is a theorem, not a baseline.** A commuting pool makes the whole
> network — forward pass *and* gradient, at any depth — a function of `diag(ρ)`
> ([`../docs/HYBRID_BACKPROP.md`](../docs/HYBRID_BACKPROP.md) §5.2, asserted in
> `tests/qnn/test_pools.py`). On a coherence-only label it must sit at chance.
> Do not "improve" it ([`../docs/INVARIANTS.md`](../docs/INVARIANTS.md) I15).

> **Result (2026-08-05, 8-qubit register).** On the purely off-diagonal control:
> quantum **83.0%** held-out, `z_only` **51.0%** (chance), `diagonal_full` 55.3%
> with 4× the parameters. Removing the classical layers costs 10.7 points.
> On the *physical* labels `⟨S²⟩` and `c` there is still **no** coherence
> advantage — the diagonal models tie or beat the quantum one. See
> `../docs/RESEARCH_LOG.md` and `../docs/OPEN_QUESTIONS.md` Q1, Q12.
>
> **Controlled against the single neuron (2026-08-06, 10-qubit register).** On
> the *bit-identical* control label and split used by
> `train_spin_comparison.py`: hybrid **93.0%** vs single-neuron **66.7%** —
> **+26.3 points from the architecture alone**. The single neuron's quantum and
> classical models were separated by 0.4 points and both sat at the 67.0%
> descriptor baseline; the hybrid separates by 16.0 points.
>
> **Always read an ablation against its own label's descriptor baseline, not
> against 50%.** The earlier single-neuron control looked like a failed ablation
> (classical 66.3%) when it was really a confounded label (descriptors 67.0%).

## `demo_train_curve.py` — demonstration figure

Trains a logistic decision rule on the 248 Pauli features of the 1000-molecule
set, labeled by median HOMO–LUMO gap, 70/30 stratified split. Writes
`figures/qh9_quantum_neuron_training.png`.

**What it shows:** the features carry molecular information that generalizes to
unseen molecules. **What it does not show:** any quantum advantage — 99.7% of
that feature weight is occupation information a classical model reads too.
Presentation asset, not a result. Input paths are hardcoded.

Audited in full by the four `gap_diagnosis*` scripts below; read that section
before quoting its 94%.

---

## `gap_diagnosis*.py` — the audit of the HOMO–LUMO gap label

Four scripts plus one streaming pass, answering why the label above shows no
quantum advantage. Findings: [`../docs/RESEARCH_LOG.md`](../docs/RESEARCH_LOG.md)
2026-08-06. Deck: `Papers/homo_lumo_gap_diagnosis.pptx`.

**The instrument.** On a Jordan-Wigner register the 136 Z/ZZ strings of the
248-term extended-Heisenberg basis are diagonal operators, so their expectations
are functions of `diag(ρ)` alone — provably what a dephased model may read
([`../docs/HYBRID_BACKPROP.md`](../docs/HYBRID_BACKPROP.md) §5.2). The 112 XX/YY
strings read only off-diagonal entries. So "quantum − classical" is an exact
feature ablation inside one model class, with the same trainer and the same
splits, and there is no baseline-tuning objection to answer.

| Script | What it settles | Cost |
|---|---|---|
| `gap_diagnosis.py` | The accuracy and regression ladders, the threshold-distance breakdown, the residual test. Also the shared estimator library the other three import | ~15 min |
| `gap_rho_pass.py` | Everything that needs ρ itself: the exact `R±` screen, the full 4900-entry `diag(ρ)` per molecule, per-molecule Frobenius norms | **~25 min**, reads 45 GB |
| `gap_diagnosis_followup.py` | Whether a *different label* fixes it (correlated CASCI gap, correlation correction, synthetic control) or *different molecules* do (coherence-stratified split, conjugated subset at two temperatures) | ~10 min |
| `gap_diagnosis_ceiling.py` | Whether the classical baseline was under-powered: trains on `diag(ρ)` itself rather than its 136-feature summary | ~5 min |
| `gap_diagnosis_controls.py` | Coherence features against equal-count Gaussian noise; how much of `‖offdiag ρ‖²` the weight-≤2 pool can even see; global coherence against the residual | ~5 min |

```bash
PYTHONPATH=scripts .venv/bin/python scripts/gap_diagnosis.py            # first
.venv/bin/python scripts/gap_rho_pass.py                                # long, independent
PYTHONPATH=scripts .venv/bin/python scripts/gap_diagnosis_followup.py   # writes the feature cache
PYTHONPATH=scripts .venv/bin/python scripts/gap_diagnosis_ceiling.py    # needs gap_rho_pass.npz
PYTHONPATH=scripts .venv/bin/python scripts/gap_diagnosis_controls.py
```

`PYTHONPATH=scripts` is needed because the follow-ups import `gap_diagnosis` as
a top-level module. Everything lands in `results/gap_diagnosis*.json`, which is
the only thing `presentation/figures_gap.py` reads.

> ⚠️ The ρ-space screen ratio (**0.1345** here) and the Pauli-feature-space one
> (**0.0069**) are different quantities that differ by ~20×. Do not quote them
> interchangeably, and screen the *residual* rather than the raw label —
> [`../docs/INVARIANTS.md`](../docs/INVARIANTS.md) I16.

---

## `second_quantization_labels.py` and the basis audit — the OMol25 evidence base

Written for [`../docs/OMOL25_ASSESSMENT.md`](../docs/OMOL25_ASSESSMENT.md).
Everything here runs off the **stored** `h1eff`/`g` of the existing run file, so
none of it needs QH9, a new SCF, or the 45 GB eigenvector blocks — except
`spin_ladder_pilot.py`, which is deliberately different (see below).

| Script | What it settles |
|---|---|
| `second_quantization_labels.py` | Solves the cation, anion and triplet sectors of each stored active-space Hamiltonian, plus each sector's single-determinant energy by Slater's rules. Produces labels that are **identically zero for any single determinant**: correlation corrections to IP / EA / S–T gap, quasiparticle pole strength `Z`, Head-Gordon `N_unpaired`. Gates itself against the stored spectrum (4e-11 Ha) and against PySCF's own CI energy evaluation (exactly 0.0) |
| `sq_label_screen.py` | Runs the gap audit's diagnostic over those ten labels. **All ten fail** (Q − C from −2.37 to +0.13) |
| `localized_basis_experiment.py` | ★ Canonical vs full Edmiston-Ruedenberg vs block-ER, each with a **zero-correlation single determinant carried through the same rotation**. This control is the whole point: full ER makes an uncorrelated determinant look *more* coherent (0.935) than the correlated state (0.929) |
| `basis_dependence_probe.py` | C₂H₄ torsion with SCF continuation and an FCI-invariance gate: the same singlet/triplet label goes from screen ratio 1.06 to 2451 under a physics-preserving rotation, as `N_unpaired` goes 0.19 → 2.00 |
| `multireference_stratification.py` | Does the ablation improve with multireference character (no: 19/20 quartile deltas negative) and do coherence features help at small *n* (no: they cost sample efficiency) |
| `plot_omol25_assessment.py` | The three-panel summary figure |

```bash
.venv/bin/python scripts/second_quantization_labels.py --limit 1000 --workers 10
PYTHONPATH=scripts .venv/bin/python scripts/sq_label_screen.py
.venv/bin/python scripts/localized_basis_experiment.py --limit 1000 --workers 8
PYTHONPATH=scripts .venv/bin/python scripts/basis_dependence_probe.py
PYTHONPATH=scripts .venv/bin/python scripts/multireference_stratification.py
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/plot_omol25_assessment.py
```

> ⚠️ `spin_ladder_pilot.py` (which `basis_dependence_probe.py` imports for its
> geometry and CASCI builder) **runs its own SCF on its own geometries**. That is
> forbidden inside the Phase-1 pipeline
> ([`../docs/INVARIANTS.md`](../docs/INVARIANTS.md) I2) and permitted here only
> because this is a standalone probe that writes nothing the pipeline reads. It
> uses B3LYP/def2-SVP so its numbers are comparable with the production run, and
> it scans with **SCF continuation** because the closed-shell solution for
> twisted ethylene has more than one branch.

> ⚠️ Never quote an off-diagonal share without naming the basis and carrying the
> zero-correlation control — [`../docs/INVARIANTS.md`](../docs/INVARIANTS.md) I17.

---

## `presentation/` — the talks, as code

**Three decks**, all generated from this directory and sharing one visual system.
Editing the slides in PowerPoint works until the next rebuild discards it; edit
the content module instead.

| Deck | Content module | Figures | Output |
|---|---|---|---|
| `theory` | `content_theory.py` | `figures_theory.py` → `figures/deck_theory/` | `Papers/molecular_hamiltonians_and_thermal_states.*` |
| `results` | `content.py` | `figures.py` → `figures/deck/` | `Papers/thermal_states_presentation.*` |
| `gap` | `content_gap.py` | `figures_gap.py` → `figures/deck_gap/` | `Papers/homo_lumo_gap_diagnosis.*` |

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=scripts/presentation \
    .venv/bin/python scripts/presentation/build_cache.py       # results deck, ~28 min, once
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=scripts/presentation \
    .venv/bin/python scripts/presentation/figures_theory.py    # theory figures (+ its own cache)
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=scripts/presentation \
    .venv/bin/python scripts/presentation/figures.py           # results figures
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=scripts/presentation \
    .venv/bin/python scripts/presentation/build_deck.py theory   # .pptx + .html
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=scripts/presentation \
    .venv/bin/python scripts/presentation/build_deck.py results
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=scripts/presentation \
    .venv/bin/python scripts/presentation/figures_gap.py      # gap figures (needs results/gap_diagnosis*.json)
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=scripts/presentation \
    .venv/bin/python scripts/presentation/build_deck.py gap
```

> **The theory deck declares a punctuation style and the build enforces it**
> (`lint=True` in `build_deck.DECKS`): no em dashes, and no en dash outside a
> hyphenated name. A clean build prints nothing. Do not silence it with
> `--no-lint`; fix the text.

`PYTHONPATH` is required — these import each other as top-level modules rather
than as a package, matching how `figures/` and `benchmarks/` already work.

| File | |
|---|---|
| `content.py`, `content_theory.py` | The scripts of the talks: one dict per slide, plus speaker notes. No layout code |
| `build_deck.py` | Layout engine, the deck registry, and two renderers (`.pptx` via python-pptx, `.html` mirror) over one block list. Reports any block crossing the footer, any punctuation violation, and nudges under-full slides down for optical balance |
| `style.py` | Palette, geometry, matplotlib defaults. **The series colours are validated, not chosen** — see the module docstring before changing one |
| `equations.py` | Display equations → transparent PNGs sized in slide inches (mathtext, Computer Modern), cached by hash |
| `inline_math.py` | The LaTeX subset used in slide prose → bold / italic / super- / subscript runs |
| `figures.py` | The results deck's figures. `mps_bonds` is opt-in (minutes per molecule) |
| `figures_theory.py` | The background talk's figures. `build_cache` inside it needs the QH9 database and PennyLane; the rest read `results/theory_cache.npz` |
| `build_cache.py` | One pass over the 45 GB run file → `results/presentation_cache.npz`. Currently also the only producer of the density-matrix coherence share ([`../docs/OPEN_QUESTIONS.md`](../docs/OPEN_QUESTIONS.md) Q6) |

Rendering the PDF: `google-chrome --headless --print-to-pdf` on the HTML gives a
correct layout with **substituted fonts** (Calibri/Cambria are not installed
here). Re-export from the `.pptx` for brand-faithful output. LibreOffice is
present but cannot write output in this sandbox — do not spend time on it.

---

## Adding a script here

Belongs in `scripts/` if it is an analysis or a bridge; in `qthermal/` if it is
a pipeline stage; in `benchmarks/` if it measures performance.

Checklist: module docstring saying *why* and showing the usage line · `argparse`
CLI with no hardcoded paths (`demo_train_curve.py` is the exception, not the
model) · resumable if it can run for more than a few minutes · a row in
[`../docs/DATA_CATALOG.md`](../docs/DATA_CATALOG.md) for anything it writes ·
an entry in this file and in [`../docs/REPO_MAP.md`](../docs/REPO_MAP.md).

**Wanted** (see [`../docs/OPEN_QUESTIONS.md`](../docs/OPEN_QUESTIONS.md) Q1, Q6):
`screen_labels.py` (the R± screening metric — highest value per hour in the
project), `coherence_audit.py` and `plot_run_diagnostics.py` (to reproduce the
two orphan artifacts).
