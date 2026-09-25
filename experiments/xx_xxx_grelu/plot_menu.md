# GReLU neuron on XX vs XXX thermal states: setup and plot menu

> The pre-run planning document (22 plot options), kept for reference. The write-up of the runs is **[REPORT.md](REPORT.md)**. Paths below are relative to `experiments/xx_xxx_grelu/`, except `data/`, which is at the repository root.

Plan for this week's group meeting. The pipeline is built and tested. The six
suggested plots were made on 2026-09-23 and are in `figures/`,
rendered by `src/plot_xx_xxx_grelu.jl`.

## Results (2026-09-23)

- **Cross-validation.** Both models reached mean validation AUC 1.000. Chosen settings: Alg 9 T = 1, l2 = 1e-3; Alg 8 T = 1, l2 = 1e-4.
- **Exact gradients (B1, C).** Both reach 100% accuracy and AUC 1.000 on the 16 held-out test chains, with 0 of 160 states misclassified. The decision margin never closes at any kT.
- **Learned H (D1).** θ_XX ≈ θ_YY on every bond, θ_ZZ has the opposite sign, and θ_Z ≈ 0 (at most 0.015). This is the SU(2)-breaking "ZZ against XX+YY" direction from A2: summed over bonds, −θ_ZZ / θ_XX = 2.6 for Alg 9 and 1.1 for Alg 8. Alg 8's ‖θ‖₁ grows to 33, against 2.4 for Alg 9, and it uses a bias of −0.15.
- **Temperature transfer (F1).** Training on hot rungs and testing on cold ones gives 0.96 (Alg 9) and 1.00 (Alg 8). Cold to hot gives 1.00 (Alg 9) and 0.99 (Alg 8). The only misses are one chain at the coldest three rungs (Alg 9) and one chain at kT = 2 (Alg 8).
- **Classical control (2026-09-24, `src/ffnn_xx_xxx.jl`).** A feed-forward network (37 local expectation values → 10 ReLU → 1; 391 parameters) on the same split gets test accuracy 1.000 and AUC 1.000 on all 5 seeds. Retraining with the training chains' labels shuffled drops it to chance: over 50 shuffles, test accuracy is 0.497 ± 0.121 and AUC 0.51. So the split does not leak, and 100% reflects how easy the task is (A2), not a bug.
- **Finite shots (E). Training on single-shot gradients fails at these budgets.** At the initial θ, Alg 9 needs about 1,200 circuit runs (extrapolated from N ≤ 512) before the gradient error drops below ‖g‖. Alg 8 is about 10× noisier: its variance carries ‖θ‖₁³/T², and ‖θ‖₁ grows during training. With Adam at lr 0.05, the noise-driven steps inflate ‖θ‖₁, which raises the variance further. Alg 9 with N = 256 touches 100% test accuracy but does not hold it, and Alg 8 never gets off chance. The optimiser was tuned for exact gradients. A noise-aware setup (lower or decaying learning rate, a ‖θ‖₁ cap, larger N) is the obvious next run.

## What was built

| file | role |
|---|---|
| `src/grelu_neuron.jl` | GReLU neuron: exact loss and gradient (Daleckii–Krein), and Monte-Carlo simulations of the GReLU versions of **Algorithm 9** and **Algorithm 8** (Hadamard tests, single shots) |
| `src/xx_xxx_data.jl` | reads `data/xx_xxx_thermal_states/xx_xxx_n10.h5`, rebuilds each ρ exactly from its couplings J, checks against the MPOs, makes the split |
| `src/test_grelu_neuron.jl` | 17 checks on the n = 4 fixture (all pass): reader vs the file, Pauli algebra, gradients vs finite differences, both estimators unbiased |
| `src/train_xx_xxx_grelu.jl` | experiments → `results/*.csv` |

**The model.** H(θ) = Σ_j θ_j H_j on 10 qubits, using the same 37 terms as `Alg9Yao`: Z_i, Z_iZ_{i+1}, X_iX_{i+1} and Y_iY_{i+1}. The neuron's output is Tr[GReLU_T(H(θ)) ρ], where GReLU_T(x) = xΦ(x/T) + Tφ(x/T) (paper Eq. 92). The two training routes are:

- **Algorithm 9 (margin loss).** L = (1/M) Σ Tr[GReLU_T(−y_m H) ρ_m]. This is the GReLU version of the logistic loss, because the logistic loss is softplus(−yH). Its gradient is Theorem 17 (Eq. F84), which is the Algorithm 9 circuit with a Gaussian time distribution. The prediction is sign Tr[H ρ]. There is no identity term.
- **Algorithm 8 (squared loss).** L = (1/M) Σ (Tr[GReLU_T(H) ρ_m] − ỹ_m)², with ỹ ∈ {0, 1}. The targets can't be ±1 because GReLU ≥ 0, so the model needs an identity (bias) term. Each run is 2·b1·b2 from two independent circuit blocks, as in Appendix B. The prediction is output > ½.

**The split** (`results/split.csv`, seed 20260923). Whole chains (`model:draw`, with all 10 kT rungs) are assigned to one side:

- **Test:** 8 XX + 8 XXX chains = 160 states.
- **Train:** 34 + 34 chains = 680 states, in 5 stratified group folds of 7 + 7 chains (the last fold has 6 + 6).

Hyperparameters (T and the L2 strength) are chosen by cross-validation on the training folds only.

**Things already checked** (from the `split` step):

- The rebuilt energies and log Z match the file to 1e-13.
- On the fixture, contracting the MPOs reproduces the file's stored trace distances to 1e-15, so site order, J order and the Pauli convention all match.
- On the full data, the χ64 MPO is off from the exact ρ by a trace distance of 1.6e-2 (max) at kT = 0.1 and 7e-5 at kT = 2, and has eigenvalues down to −2.9e-3. That is why the neuron is fed exact ρ rather than the MPO.
- The bond correlators show what separates the classes. XXX has ⟨ZZ⟩ = ⟨XX⟩ = ⟨YY⟩ (SU(2) symmetry). XX has |⟨ZZ⟩| well below |⟨XX⟩|: −0.18 vs −0.41 at kT = 2, and −0.46 vs −0.66 at kT = 0.1.

**Expectation.** A 4-step smoke run of the Algorithm 9 model already reached 100% test accuracy. The spectra baseline in `classify_n10.json` is AUC 1.000. Accuracy will probably saturate, so the more interesting plots are about why it works, what it costs in shots and circuit runs, and how it generalises.

## Plot menu

Each entry gives the plot, what it would show, and where its data comes from. ★ marks my suggested shortlist for the meeting.

### A. Data and setup

| # | plot | shows | data |
|---|---|---|---|
| A1 | Split map: 84 chains × 10 kT grid, cells coloured by train fold / test | that no chain leaks across the split | `split.csv` |
| A2 ★ | Scatter of ⟨ZZ⟩ vs ⟨(XX+YY)/2⟩ for every state, coloured by class, shaded by kT, with the line y = x | XXX lies exactly on the diagonal and XX lies off it; a neuron with no bias can separate them at every temperature | `correlators.csv` |
| A3 | MPO-vs-ED trace distance against kT (log y), with the MPO's minimum eigenvalue | why we rebuild ρ from J instead of using the χ64 MPO | `mpo_check.csv` |

### B. Training dynamics (exact gradient, i.e. infinite shots)

| # | plot | shows | data |
|---|---|---|---|
| B1 ★ | Loss and accuracy against step, Algorithm 9 vs Algorithm 8, train (solid) and test (dashed) | convergence, and how the two losses behave differently | `final_history.csv` |
| B2 | θ against step, one line per term, coloured by type (Z / ZZ / XX / YY / I) | which terms the optimiser drives first | `final_history.csv` |
| B3 | CV grid: validation AUC heatmap over (T, l2) for Algorithm 8, a line over l2 for Algorithm 9 | the model selection is visible and was done without the test set | `cv.csv` |

### C. Test performance

| # | plot | shows | data |
|---|---|---|---|
| C1 ★ | Test-set decision score (Tr[Hρ] for Algorithm 9, output − ½ for Algorithm 8) as histogram or violin by class, with the threshold marked | the margin, not just the accuracy | `final_predictions.csv` |
| C2 ★ | Score against kT, one line per test chain, coloured by class | margins shrink as states heat up; shows where errors would happen | `final_predictions.csv` |
| C3 | Accuracy against kT: Algorithm 9, Algorithm 8, classical ansatz, and the `classify_n10.json` baselines | performance at each temperature compared with the classical models | predictions + `classify_n10.json` |
| C4 | ROC curves and confusion matrices for the test set | the standard summaries | `final_predictions.csv` |
| C5 ★ | AUC against number of trainable parameters (log x): GReLU neurons (37–38) vs logistic regression / MLP on scalars (16), spectra (576) and tensors (163,840) | the honest comparison, framed as parameter efficiency | predictions + `classify_n10.json` |

### D. What the neuron learned

| # | plot | shows | data |
|---|---|---|---|
| D1 ★ | Final θ along the chain, grouped bars per bond: θ_ZZ,i, θ_XX,i, θ_YY,i (and θ_Z,i) | whether it learned "ZZ against XX+YY", i.e. the SU(2)-breaking direction; we expect opposite signs and θ_Z ≈ 0 | `final_history.csv` (last step) |
| D2 | Spectrum of the learned H(θ) with GReLU applied, overlaid with the eigen-populations of one XX and one XXX test state at low and high kT | the activation acting on quantum states; where the non-commutativity matters | needs a small extra dump |

### E. Algorithm costs (the main quantum-algorithm result)

| # | plot | shows | data |
|---|---|---|---|
| E1 ★ | Gradient-estimate relative error (and cosine to the exact gradient) against circuit runs N, log-log with a 1/√N guide. Algorithm 9 vs Algorithm 8, single shots vs exact circuit expectations, at θ₀ and at the trained θ | Algorithm 8 is far noisier: it multiplies two estimators and carries a ‖θ‖₁³/T² prefactor. The fixture tests already showed it (relative error 3.5 vs 0.2 at equal N) | `gradvar.csv` |
| E2 ★ | Test accuracy / loss against step for N = 16, 64, 256 runs per step, with exact gradients as reference | how much quantum sampling training actually needs | `shots_*_history.csv` |
| E3 | Total circuit runs (N × steps) to reach 95% test accuracy, for each budget | the end-to-end quantum cost | `shots_*_history.csv` |
| E4 | Test accuracy against measurements per state at inference, split into hot and cold states | inference cost; hot states need more shots because their margins are smaller | `readout.csv` |
| E5 | ‖θ‖₁/T (the Eq. C42 sample-complexity prefactor) against step | why budgets that are fine early can fail late, as ‖θ‖₁ grows | any `*_history.csv` (`theta_l1`) |

### F. Generalisation and ablations

| # | plot | shows | data |
|---|---|---|---|
| F1 ★ | Temperature transfer: accuracy for each test kT, training on hot → testing on cold and the reverse | the open question from the dataset handoff: does it transfer across temperature? | `kT_transfer_*.csv` |
| F2 | Quantum ansatz (Z, ZZ, XX, YY) vs classical ansatz (Z, ZZ only): accuracy against kT | whether XX/YY terms help at all. A classical ansatz sees only the diagonal of ρ, so it can use only ⟨ZZ⟩; the question is whether that is enough | `classical_*.csv` |

### Suggested meeting story (about 6 figures)

A2 (why it's separable) → B1 (training) → C1/C2 (margins at each temperature) → D1 (learned H) → E1 + E2 (Algorithm 8 vs 9 costs) → F1 (temperature transfer). C5 is the backup slide if someone asks about the classical baselines.

## Runtime estimates (M1 Pro, measured per step)

| experiment | what it runs | est. time |
|---|---|---|
| `cv` | Algorithm 9: 4 l2 × 5 folds; Algorithm 8: 9 (T, l2) × 5 folds; 150 exact steps each | ~45 min |
| `final`, `classical` | 2 + 2 runs × 200 exact steps | ~10 min |
| `kT` | 4 runs × 200 exact steps | ~8 min |
| `shots` | Algorithm 9 N = 16/64/256; Algorithm 8 N = 64/256; 200 steps each | ~45 min |
| `gradvar` | N = 8…512, 20 repeats, both algorithms, two θ | ~30 min |
| `readout` | up to 4096 shots per state, 10 repeats | ~5 min |

Everything runs with `julia experiments/xx_xxx_grelu/src/train_xx_xxx_grelu.jl cv final classical kT gradvar readout shots`. Run `cv` first, because every later experiment reads the hyperparameters it chooses from `cv_choice.csv`.
