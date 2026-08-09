# Repository map — every file, one line each

*Complete inventory. If a file is not here, it did not exist at the last audit
(2026-08-05) — add it when you create one.*

Legend: **[core]** load-bearing code · **[entry]** runnable entry point ·
**[ref]** reference/reading · **[gen]** generated artifact ·
**[stale]** known out of date · **[orphan]** no known producer

---

## Root

| File | |
|---|---|
| `AGENTS.md` | **[entry]** Master entry point for agents: routes, repo diagram, non-negotiables. **Start here.** |
| `CLAUDE.md` | **[core]** Auto-loaded operational rules: environment, commands, architecture summary, invariants |
| `README.md` | **[ref]** Human-facing project README: pipeline table, layout, quick start |
| `pyproject.toml` | **[core]** Package metadata. Flat layout — `packages = ["qthermal", "qnn"]` is explicit because auto-discovery cannot disambiguate this many top-level dirs. PennyLane pinned to one minor (breaks API across minors; lightning must match) |
| `requirements.lock` | **[core]** `pip freeze` of the environment that produced every committed result and figure. **Reproducing published numbers means this file, not `pyproject.toml`'s ranges** |
| `notebook_test_utils.py` | **[core]** Executes notebook cells **by index** into a namespace; shared by equivalence tests and benchmarks. Moved here from `tests/` (staged, uncommitted) so it is importable as a top-level module |
| `.gitignore` | `*.h5`, `*.db`, `*.pt`, `*.partial`, venv, caches. Result files exist only locally |
| `.claude/settings.local.json` | Two pre-approved Bash commands (git status, notebook equivalence test) |

## `qthermal/` — HALF 1: QH9 → thermal states

A strict linear pipeline, one module per stage, lettered in the docstrings.
See [`../qthermal/README.md`](../qthermal/README.md) for physics conventions
and solver contracts.

| File | Stage | |
|---|---|---|
| `__init__.py` | — | Package docstring, `__version__ = "0.1.0"` |
| `README.md` | — | **[ref]** Physics conventions, solver contracts, six evidence-driven deviations, measured performance |
| `loader.py` | **A** | **[core]** `MoleculeRecord` dataclass, raw SQLite adapter, empirical Ångström/Bohr detection with geometric tiebreak. Holds the QH9→PySCF AO transform helpers that **must not** be applied to raw DB blobs |
| `orbitals.py` | **B** | **[core]** PySCF `Mole` construction, overlap, MO recovery via `eigh(F,S)`, sign-gauge canonicalization, validation gates |
| `active_space.py` | **C** | **[core]** `ActiveSpace` frozen dataclass — frontier window plus every derived dimension (`ncas`, `dim`, string counts). Nothing downstream may hardcode 8 or 4900 |
| `hamiltonian.py` | **D** | **[core]** Frozen-core CASCI `(ecore, h1eff, g)` from injected orbitals. `g` is full `ncas⁴` chemist notation. Never calls `mf.kernel()` |
| `diagonalize.py` | **E** | **[core]** The `SpectralSolver` seam (a `Protocol`) plus three implementations with **different contracts**: `DenseEDSolver`, `IterativeWindowSolver` (certified Krylov), `NonInteractingSolver` (closed-form g=0 reference) |
| `thermal.py` | **F** | **[core]** Boltzmann weights (stable shifted softmax), truncation with exact recorded tail, `ThermalBlock`, quantumness diagnostics, projected trace distance to the Gaussian reference |
| `io_hdf5.py` | **G** | **[core]** Resume-safe gzip HDF5 writer. `complete=True` written **last**; incomplete groups deleted and rewritten. Layout documented at the top of the file |
| `run.py` | **H** | **[entry]** CLI + multiprocessing orchestration. `SOLVERS` registry lives here. Uncommitted: `--indices` targeted-subset selection |
| `encode.py` | **I** | **[core]** Jordan–Wigner encoding (blocked / interleaved wire layouts + parity signs), sector compression, the 248-term extended-Heisenberg Pauli basis and its determinant-basis expectations, Z₂ tapering, PennyLane JW Hamiltonian builder |
| `encode_run.py` | **I-CLI** | **[entry]** Batch run-file → Pauli-coefficient file, resume-safe, optional `--taper` |
| `mps.py` | **J** | **[core]** Eigenblock → purification MPS via TT-SVD. Ancilla bond = thermal rank; only physical bonds depend on wire ordering |

## `qnn/` — HALF 2: the hybrid quantum-classical network

One layer of Fermi–Dirac quantum neurons reading ρ directly, then a classical
MLP: the architecture of `Papers/Fermi-Dirac Machines.pdf` §VII.C, whose
training the paper leaves as an open problem. The rule is derived in
[`HYBRID_BACKPROP.md`](HYBRID_BACKPROP.md); **read it before touching
`quantum_layer.py`**.

| File | Stage | |
|---|---|---|
| `__init__.py` | — | Package docstring, quick start, `__version__ = "0.1.0"` |
| `README.md` | — | **[ref]** The rule in one line, layout, cost table, deviations, and why the ablation is a theorem rather than a baseline |
| `activations.py` | **A** | **[core]** The paper's six quantized activations (tanh Eq. 18, sigmoid Eq. 2, softplus Eq. 68, SiLU Eq. 74, erf §IV A, GReLU Eq. 92, GeLU Eq. 97) plus the identity control. Numerically stable first divided differences — difference quotient above `\|a−b\| = 10⁻³`, 8-node Gauss–Legendre on the paper's Eq. (A6) below — the matrix function, and the Fréchet derivative. `relu` is **refused** for the quantum layer (not C¹ at 0, which is where a traceless pool puts the spectrum) |
| `pools.py` | **B** | **[core]** Operator pools in structured diagonal/single-mask form, never dense: `quantum` (I, Z, ZZ + XX, YY), `z_only` (the ablation), `diagonal_full` (the classical ceiling), `xy_only`. Spectral-scale initialization `σ = T/√(mean Tr[Hⱼ²]/K)` |
| `states.py` | **C** | **[core]** `StateBatch` — the only two places the data is touched: `expectations` (forward) and `aggregate` (the δ-weighted state aggregate `Rᵢ`), one GEMM each. Accepts a memory-mapped stack in place |
| `quantum_layer.py` | **D** | **[core]** **The new math.** Forward `aᵢ = Tr[φ(Bᵢ)ρ]`, backward `∂L/∂Θᵢⱼ = Tr[Hⱼ·Dφ(Bᵢ)[Rᵢ]]`. `J₁` eigendecompositions per epoch, independent of dataset size. A commuting pool takes an exact `O(K)` path (`U is None`) instead of `O(K³)` |
| `classical.py` | **E** | **[core]** Textbook dense layers, hand-written so the chain rule is visible at the seam |
| `network.py` | **F** | **[core]** Composition, logistic/squared losses, full-batch Adam, optional `shots` sampling with the true measurement variance |

## `scripts/` — analysis and bridge tools

Run as plain scripts (`python -m scripts.foo` or `python scripts/foo.py`).

| File | |
|---|---|
| `screen_conjugation.py` | **[entry]** Cheap three-tier triage over the whole QH9 DB: Tier 1 degree of unsaturation (formula only), Tier 2 frontier gap + near-degenerate frontier DOS (`eigh(F,S)`), Tier 3 largest conjugated π-subsystem (RDKit, optional — degrades gracefully). Resumable via `.partial` checkpoint |
| `export_thermal_training.py` | **[entry]** ★ **The bridge.** Run file → `{ρ_m, y_m}` training HDF5 in eigenblock form for the Julia trainers. **The label is pluggable and the default (`h5:static_corr`) is an acknowledged placeholder** |
| `spin_labels.py` | **[entry]** Thermal `⟨S²⟩` split into its diagonal part `D` (how many unpaired electrons — classical) and coherence-only part `c = Tr(ρ S²_od)` (how they are coupled). Sector `S²` built once from `contract_ss`; optionally emits the qubit-projected ρ stack |
| `train_spin_comparison.py` | **[entry]** Quantum vs classical Fermi-Dirac neuron on those labels — identical loss/optimizer/split, differing only in off-diagonal pool reach. Includes the R± screening metric, a classical-descriptor baseline, and a positive control. `--replot` re-renders from JSON |
| `train_hybrid_spin.py` | **[entry]** ★ The **hybrid** network (paper §VII.C) on the same 1000 molecules and the same labels and split as `train_spin_comparison.py`, so the deep and shallow results read side by side. Four models per label: quantum pool, `z_only`, `diagonal_full`, and a depth ablation (quantum pool, no classical layers). `--project-qubits k` restricts the register to the 2ᵏ most-populated determinants — the only lever that makes a sweep affordable, since the quantum layer's cost is `O(K³)` and M-independent. `--replot` re-renders from JSON |
| `demo_train_curve.py` | **[entry]** Demonstration: trains a logistic decision rule on the 248 Pauli features of 1000 molecules, labeled by median HOMO–LUMO gap, with a 70/30 split. Produces `figures/qh9_quantum_neuron_training.png`. Hardcoded input paths |
| `gap_diagnosis.py` | **[entry]** ★ The audit of that label (2026-08-06). Splits the 248 features into their exact diagonal (Z/ZZ, 136) and off-diagonal (XX/YY, 112) blocks and runs the accuracy ladder, the regression ladder, the threshold-distance breakdown and the residual test. Also the estimator library (`fit_logistic`, `eval_classifier`, `eval_regressor`, `oof_predictions`) the other three import |
| `gap_rho_pass.py` | **[entry]** One streaming pass over the 45 GB run file (~25 min): class-aggregated `R±` for the exact screening ratio, the full 4900-entry `diag(ρ)` per molecule, and per-molecule diagonal/off-diagonal Frobenius norms → `results/gap_rho_pass.npz` |
| `gap_diagnosis_followup.py` | **[entry]** Alternative labels under the same ablation (correlated CASCI gap, correlation correction, synthetic off-diagonal control), the coherence-stratified split, and the conjugated-subset headroom. Owns the `results/gap_diagnosis_data.npz` feature cache |
| `gap_diagnosis_ceiling.py` | **[entry]** The honest classical ceiling: trains on `diag(ρ)` itself (300 PCs) rather than its 136-feature summary, and reports the ρ-space R± screen |
| `gap_diagnosis_controls.py` | **[entry]** Two controls: coherence features vs an equal number of Gaussian noise features, and how much of `‖offdiag(ρ)‖²` the weight-≤2 pool can even see (0.002%). Plus the global-coherence-vs-residual probes |
| `second_quantization_labels.py` | **[entry]** ★ Solves *other* particle-number and spin sectors of the stored active-space Hamiltonians — cation, anion, triplet — plus each sector's single-determinant energy by Slater's rules. Yields labels that vanish identically for any single determinant: correlation corrections to IP/EA/S–T gap, the quasiparticle pole strength `Z = Σ_p \|⟨Ψ⁺\|a_p\|Ψ⟩\|²`, Head-Gordon `N_unpaired`. Self-gating: reproduces the stored spectrum to 4e-11 Ha |
| `sq_label_screen.py` | **[entry]** Runs the gap audit's diagnostic over those ten labels: ladder, regression, residual test, composition confound. All ten fail |
| `localized_basis_experiment.py` | **[entry]** ★ **The basis question, settled.** Canonical vs full Edmiston-Ruedenberg vs block-ER (reference-preserving), each with a **zero-correlation single determinant carried through the rotation** as the control. ER localization uses only the stored ERIs — no AO data, no SCF. [`INVARIANTS.md`](INVARIANTS.md) I17 |
| `basis_dependence_probe.py` | **[entry]** C₂H₄ torsion scan with SCF continuation: the same singlet/triplet label moves from a purely diagonal readout to a purely coherent one as `N_unpaired` goes 0.19 → 2.00. FCI-invariance gate at 1e-8 Ha |
| `spin_ladder_pilot.py` | **[core]** The ethylene geometry and the B3LYP/def2-SVP → CASCI(8,8) builder the probe imports. **Deliberately runs its own SCF and its own geometries**, which the Phase-1 pipeline must not ([`INVARIANTS.md`](INVARIANTS.md) I2) |
| `multireference_stratification.py` | **[entry]** Does the ablation improve with multireference character, and do coherence features pay off at small *n*? Neither: 19 of 20 quartile deltas negative, and the coherence block costs sample efficiency |

### `scripts/presentation/` — the group-meeting deck, as code

The deck in `Papers/` is **generated**. Edit here, then rebuild; edits made in
PowerPoint are discarded by the next build. Run with
`PYTHONPATH=scripts/presentation` (these import each other as top-level
modules, not as a package).

| File | |
|---|---|
| `content.py` | **[entry]** ★ **The results talk.** One dict per slide — kicker, title, layout, content, presenter notes. Readable on its own; contains no layout code |
| `content_theory.py` | **[entry]** ★ **The background lecture** (29 slides): the fermionic algebra and the Slater-Condon rules, active spaces, thermal states and their construction by reachable size, representation, Jordan-Wigner, qubit preparation, and an extended treatment of applications. Declares a punctuation and register style the build lints |
| `content_gap.py` | **[entry]** ★ **The gap-label diagnosis**, 4 slides: the exact diagonal/off-diagonal ablation and its null, regression as the control for binarisation, the redundancy result, and the verdict plus the OMol25 assessment. Every number traces to `results/gap_diagnosis*.json` |
| `build_deck.py` | **[core]** Deck registry + layout engine + two renderers over one absolutely-positioned block list: `.pptx` (python-pptx, real text runs and notes) and `.html` (pixel-mirror, the review and PDF route). `build_deck.py {results\|theory\|gap}`. Reports footer overruns and punctuation violations; balances under-full slides |
| `style.py` | **[core]** The single source of palette, geometry and matplotlib defaults. Series colours are snapped to steps that clear an OKLCH lightness/chroma band, Machado-2009 protanopia/deuteranopia separation, and contrast — **do not add a fifth slot**, and cap all-pairs forms (scatter, small multiples) at three series |
| `equations.py` | **[core]** Display equations → transparent PNGs sized in slide inches, matplotlib mathtext with the Computer Modern font set, cached by content hash. Named in one dict so the maths is reviewable without reading layout code. mathtext rejects `\tfrac`, `\big`, `\le`, `\stackrel` |
| `inline_math.py` | **[core]** The small LaTeX subset used in slide *prose* → runs with bold / italic / super- / subscript, understood by both renderers |
| `figures.py` | **[entry]** The results deck's figures, from the project's own data. `mps_bonds` is excluded from the default run (minutes per molecule) |
| `figures_gap.py` | **[entry]** The gap-diagnosis deck's four figures. Reads only the four `results/gap_diagnosis*.json` files, so it never recomputes a number it plots |
| `figures_theory.py` | **[entry]** The background talk's figures: orbital ladder, configuration counting, a real sector Hamiltonian and thermal state, Boltzmann weights, participation against temperature, construction methods by reach, storage cost, the Jordan-Wigner map, a real Pauli decomposition, and the purification. Its `build_cache` needs the QH9 database and PennyLane |
| `build_cache.py` | **[entry]** One pass over the 45 GB run file → `results/presentation_cache.npz`. **~28 min.** Also the current producer of the density-matrix coherence share and the feature-weight split ([`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) Q6) |

### Bridge to Module K

`scripts/train_mps_thermal.py` — trains the hybrid network on MPS-produced
thermal states (`results/qh9_mps_ncas10.h5`), handling the register-endianness
conversion; conventions pinned by `tests/test_mps_bridge.py`.

## `notebooks/` — the classifier

| File | |
|---|---|
| `paper/logloss.ipynb` | **[ref]** Paper-faithful reference implementation. 9 cells. **Cell order is load-bearing** — `notebook_test_utils` executes cells 0–3 by index |
| `pennylane/logloss_pennylane.ipynb` | **[core]** The optimized production implementation. 15 cells; tests execute 1–7. Contains `optimize_phase2` (R± label aggregation), sparse/fused Pauli kernels, adaptive Chebyshev loss, backend resolution |
| `pennylane/phase1_optimization.ipynb` | **[ref]** Exploratory autodiff/profiling notebook. **Cells are in reverse narrative order** (section 7 first, section 1 last) — an artifact of how it was assembled |
| `pennylane/sampling_demo.py` | **[ref]** Minimal 2-qubit demo of the neuron output `Tr[g_T(H(ω))ρ]` with `g_T = tanh(·/T)`. The clearest single statement of the model |

## `figures/` — figure generation and rendered output

| File | |
|---|---|
| `quantum_training_impls.py` | **[core]** Both training passes reproduced verbatim side by side (`run_original`, `run_pennylane`, `run_pennylane_matched`). Same seed ⇒ bit-identical states/labels/init. **This is the code of record for benchmarking** |
| `run_pennylane_vs_original.py` | **[entry]** Produces the 500-epoch PennyLane curves and the 10-epoch same-seed equivalence check |
| `run_matched_pennylane.py` | **[entry]** Trains quantum + classical FCIM together per n — a genuinely matched single-trial comparison |
| `benchmark_notebook_comparison.py` | **[entry]** 3×3 grid: loss, validation accuracy, ms/epoch for n ∈ {2,4,7} |
| `generate_paper_training_curves.py` | **[entry]** Regenerates the training-curve figure with a **purely quantum target** (XX+YY nearest-neighbour only) — the fix for an earlier convergence artifact where the classical FCIM matched the quantum model because all-to-all ZZ could fit the target's diagonal part |
| `digitize_fig8_classical.py` | **[entry]** Digitizes the classical curves out of `docs/fig_8.png` into `results/digitized/`. Pixel↔data calibration recorded as fixed constants |
| `plot_fig8_pennylane.py` | **[entry]** Recreates the paper's Fig. 8 2×3 grid |
| `plot_efficiency_comparison.py` | **[entry]** Two-panel efficiency figure from `results/equivalence_check.csv` |
| `*.png` (9 files) | **[gen]** Rendered figures — see [`DATA_CATALOG.md`](DATA_CATALOG.md) for which script produces each |
| `deck/*.png` (9), `deck/eq/*.png` (38) | **[gen]** Both decks' figures and display equations. Produced by `scripts/presentation/figures.py` and `equations.py`; do not edit by hand |
| `deck_theory/*.png` (11) | **[gen]** The background lecture's figures. Produced by `scripts/presentation/figures_theory.py` |
| `deck_gap/*.png` (4) | **[gen]** The gap-diagnosis deck's figures. Produced by `scripts/presentation/figures_gap.py` |

## `benchmarks/` — measurement harnesses and their CSVs

| File | |
|---|---|
| `benchmark_scaling.py` | **[entry]** Times one loss-and-gradient pass across methods, qubit counts and sample counts; setup measured separately |
| `benchmark_paper_comparison.py` | **[entry]** Paper-style 2/4/7-qubit comparison. Gates the expensive original dense path behind a timing probe and a 5-minute threshold |
| `plot_paper_comparison.py` | **[entry]** Renders the three paper-comparison figures from the CSVs |
| `mps_bond_dimensions.py` | **[entry]** Blocked vs interleaved JW ordering for the purification MPS. **Docstring's stated expectation ("interleaved is expected to win") was overturned by measurement** — see `RESEARCH_LOG.md` 2026-07-27 |
| `scaling_results.csv`, `scaling_results_quick.csv` | **[gen]** Scaling benchmark output |
| `paper_training_curves_2_4_7.csv`, `paper_efficiency_summary_2_4_7.csv`, `paper_sampling_efficiency_2_4_7.csv` | **[gen]** Paper-comparison output |

## `tests/` — 337 tests (156 qthermal/notebook + 181 qnn)

| File | |
|---|---|
| `__init__.py`, `.gitignore` | Package marker; cache ignore |
| `qthermal/conftest.py` | Synthetic **B3LYP** H₂O record built end-to-end with PySCF. B3LYP not RHF — RHF H₂O has a 0.67 Ha gap and would fail `detect_units`' own physicality window |
| `qthermal/test_loader.py` (11) | Record round-trip, SQLite adapter, unit detection, `--indices` selection (uncommitted additions) |
| `qthermal/test_orbitals.py` (9) | `build_mol`, overlap, MO recovery, sign canonicalization, validation gates |
| `qthermal/test_active_space.py` (4) | Frontier window and derived dimensions |
| `qthermal/test_hamiltonian.py` (3) | CASCI Hamiltonian vs a **fully manual** frozen-core construction implemented in the test |
| `qthermal/test_diagonalize.py` (29) | **The correctness gate**: lowest dense eigenvalue + `ecore` reproduces PySCF's CASCI energy to 1e-8 Ha, at both ncas=6 and 8. Plus Krylov certification, guardrails, the non-interacting solver |
| `qthermal/test_thermal.py` (16) | Weights, truncation, diagnostics, Gaussian audit |
| `qthermal/test_io_hdf5.py` (6) | Layout, dtypes, resume safety, kT tags |
| `qthermal/test_encode.py` (10) | JW-encoded eigenvectors must be exact eigenvectors of an **independently constructed** PennyLane JW Hamiltonian — fermionic signs, integral conventions and bit order must all be simultaneously right |
| `qthermal/test_encode_run.py` (3) | Run file → extended-Heisenberg mapping file, taper round-trip |
| `qthermal/test_mps.py` (6) | Purification MPS traced back to the density matrix — exact untruncated, within bound when capped |
| `qthermal/test_run.py` (14) | End-to-end CLI on a synthetic single-molecule DB |
| `test_notebook_equivalence.py` (12) | Original vs optimized notebook parity: dense/CSR/fused Pauli equivalence, label aggregation, Chebyshev accuracy, analytic vs finite-difference gradients, complex64 agreement, matrix-free routing at n=11 |
| `qnn/conftest.py` | Synthetic random states — deliberately **rank-deficient**, because real thermal blocks are, and a full-rank random ρ hides bugs that only appear when ρ has a null space |
| `qnn/test_activations.py` (102) | All six quantized activations plus the identity control: derivative vs finite differences, divided difference vs the definition / the derivative on the diagonal / an independent `scipy.integrate.quad` of the paper's Eq. (A6), accuracy across the quotient↔quadrature switch, tanh's closed form vs the generic path at arguments where `cosh` would overflow, **Fréchet derivative vs finite differences of the matrix function**, self-adjointness, and gauge-invariance under degenerate eigenvalues |
| `qnn/test_pools.py` (23) | Structured vs dense operators, the little-endian Pauli convention, term counts, spectral-scale init. **The theorem**: deleting every off-diagonal entry of every state leaves a commuting-pool network's outputs and *all* gradients bit-identical — and provably changes the quantum pool's |
| `qnn/test_states.py` (8) | Both contractions against the loops they replace, including the transpose convention on a deliberately non-symmetric input (real data never is, so nothing else would catch it) |
| `qnn/test_gradients.py` (44) | **The decisive test.** Central finite differences of the *composite* loss, per parameter, across every activation, pool, depth, loss, temperature and classical activation; plus the degenerate-spectrum case (B = 0), the commuting fast path vs a dense construction, held-out weighting vs a subset batch, and shot-noise scaling |
| `qnn/test_paper_equivalence.py` (4) | The paper's single-neuron trainer as the shallow special case (`HYBRID_BACKPROP.md` §5.4) — against both an independent dense reimplementation and `scripts/train_spin_comparison.py` itself |

## `QThermalMPS/` — Module K: thermal states as purification MPS (Julia)

Standalone Julia package; the pipeline stage beyond dense reach. Reads Module G
run files, builds the Jordan-Wigner MPO from `(h1eff, g)`, purifies the CASCI
sector at beta = 0, evolves in imaginary time by TDVP, and exports per-kT
snapshots (MPS + dense reduced density matrices) that `qnn/` consumes.

| file | role |
|---|---|
| `src/layout.jl` | wire conventions — a contract with `qthermal/encode.py` |
| `src/sites.jl` | site types; the bosonic-ancilla decision (load-bearing) |
| `src/purification.jl` | the beta = 0 sector-maximally-entangled state |
| `src/hamiltonian.jl` | `(h1eff, g)` → OpSum → MPO (ITensorMPOConstruction) |
| `src/evolve.jl` | `thermal_ladder`: graded-step TDVP + subspace expansion |
| `src/observables.jl` | `physical_rdm`, `pauli_expect`, `sector_mean_energy` |
| `src/io.jl` | Module G readers; ladder writer (resume-safe, per-mol attrs) |
| `bin/thermal.jl` | CLI |
| `test/` | 730 assertions vs by-hand Hamiltonians and stored `evals` |

Read `QThermalMPS/README.md` before touching: four documented silent-failure
modes (ancilla statistics, frozen TDVP manifold, `"Link"`-tag priming, the
4-QNVal compile limit).

## `tensor-network-testing/` — Julia trainers (Yao / ITensor)

Reference and pretraining implementations of the paper's algorithms. Not
integrated with the Python pipeline; consume exported `{ρ_m, y_m}` files.

| File | |
|---|---|
| `algorithm9_yao.jl` | **[core]** Algorithm 9 (Thm. 5 / Eq. C27): logistic-loss gradient via the Fig. 11 circuit, statevector backend. **The one to use for training** |
| `algorithm9.jl` | Same algorithm on ITensorMPS — validation and larger *n*; far too slow to drive an optimizer |
| `algorithm8_yao.jl` | Algorithm 8 (Eq. B4): squared-loss gradient via two Hadamard tests, plus Algorithm 10 for the loss value |
| `algorithm8.jl` | Algorithm 8 on ITensorMPS; returns final MPS instead of sampling outcomes (deliberate deviation) |
| `train_alg9.jl` | **[entry]** Optimizers.jl loop driven by the Algorithm 9 gradient — honest gradient descent on L^log |
| `convergence_checks.jl` | **[entry]** Checks 0–3 correctness, 4–7 convergence sweeps (Trotter step, bond dim, cutoff, Monte Carlo), 8 cost vs *n*. Fixed seed isolates discretization from sampling noise |
| `MPS_construction.jl`, `dmrg_tutorial.jl` | ITensor tutorials, kept as reference |

## `data/`

| File | |
|---|---|
| `QH9Stable.db` | **[gen]** ~30 GB raw SQLite, untracked. Table `data(id, N, Z, pos, Ham)`. **Ham blobs are already PySCF-ordered — apply no reorder** |
| `qh9_scan.jsonl` | **[gen]** 15 MB geometry-derived size index of every record (nao, nelec, spin-orbital count). Hamiltonian-independent, therefore clean and reusable |
| `build_slater.py` | **[ref]** Retired single-determinant baseline builder (72 KB). Kept, with the AO bug fixed, purely as the regeneration route if a mean-field baseline is ever wanted |
| `qh9_raw_sqlite_audit.md` | **[ref]** The 2026-07-10 AO-ordering audit: mechanism, scope, error magnitudes, regeneration. **Read before touching the loader** |
| `README.md` | **[ref]** Directory guide and the retired-baseline explanation |
| `.tmp/` | Empty scratch directory |

## `results/`

Full catalog with provenance, schema and status: [`DATA_CATALOG.md`](DATA_CATALOG.md).
Summary: 12 HDF5 files (`*.h5`, gitignored, ~49 GB total), the conjugation
screen CSVs, classifier benchmark CSVs, digitized Fig. 8 curves, and three run
logs.

## `Papers/`

| File | |
|---|---|
| `Fermi-Dirac Machines.pdf` | **[ref]** The source paper. Defines the neuron (Eq. 16–18), the logistic loss (Eq. 56), Theorem 5 / Eq. 63 gradients, the Fig. 8 experiment, and Algorithms 8–10 |
| `QBM Learning of Ground-State Energies.pdf` | **[ref]** Quantum Boltzmann machine background |
| `thermal_states_presentation_final.pdf` | **[ref]** Group-meeting deck, reconciled to the 1000-molecule dataset 2026-07-28. Editable source is the `.pptx` outside this repo |
