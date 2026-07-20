# The QH9 Thermal-State Pipeline: A Plain-Language Report

*Status as of 2026-07-15.*

## What we are building, and why

The goal is a labeled dataset of **quantum thermal states of real molecules**,
to serve as training data for the group's quantum machine-learning classifier
(the Fermi-Dirac / Heisenberg-neuron line of work).

A "thermal state" here is a snapshot of a molecule's electrons in equilibrium
at some temperature: not a single wavefunction, but a statistical mixture of
the molecule's energy eigenstates, each weighted by the Boltzmann factor for
that temperature. Cold means the molecule sits almost entirely in its ground
state; hot means many excited states contribute at once. These mixed states
are interesting classifier inputs precisely because they interpolate between
"essentially classical" (one dominant electron configuration) and "genuinely
quantum" (many configurations interfering), and we can dial that knob with a
single temperature parameter.

The raw material is **QH9**, a public dataset of 130,831 small organic
molecules (H, C, N, O, F; up to ~20 atoms) with their quantum-chemistry
Hamiltonians precomputed at the B3LYP/def2-SVP level. We use the molecules at
their equilibrium geometries, exactly as stored.

The temperatures are deliberately extreme by laboratory standards — kT = 0.1
Hartree is roughly 32,000 K. That is not a physical scenario we expect in a
flask; it is the regime where the electronic state becomes a rich quantum
mixture, which is the property the classifier project needs.

## How a molecule becomes a thermal state

Treating all electrons of even a small molecule exactly is impossible (the
state space grows exponentially), so the pipeline follows standard quantum
chemistry practice and focuses on an **active space**: the handful of
electrons and orbitals nearest the chemical action. Our production setting is
"CAS(8,8)" — 8 electrons in 8 orbitals, four occupied and four empty frontier
orbitals. Inside that window the problem is solved *exactly*; the remaining
(core) electrons are frozen and folded into an effective potential.

For CAS(8,8) the exact problem is a 4,900 × 4,900 matrix per molecule (all
the ways of arranging 4 spin-up and 4 spin-down electrons among 8 orbitals).
The pipeline:

1. reads a molecule from the QH9 database and empirically verifies its
   coordinate units and orbital data;
2. builds the effective active-space Hamiltonian on the stored B3LYP orbitals
   (intentionally *without* re-running any self-consistent field step, so the
   dataset stays faithful to QH9's own electronic structure);
3. diagonalizes that Hamiltonian — every eigenstate and energy;
4. forms the Boltzmann mixture at each requested temperature, discarding only
   states whose combined weight is below a set cutoff (one part in a million
   by default), with the discarded amount recorded exactly;
5. computes a "quantumness audit" for each state: entropy (how mixed it is),
   natural-orbital occupations, a static-correlation score, the weight of the
   single dominant electron configuration, and the trace distance to a
   matched *non-interacting* reference state (how much electron-electron
   interaction actually changes the state);
6. writes everything to a resume-safe HDF5 file — a crashed or interrupted
   run picks up exactly where it left off.

## What has been implemented

The pipeline lives in `qthermal/` as nine small modules, one per stage
(loader, orbitals, active space, Hamiltonian, solvers, thermal ensembles,
storage, command-line runner, qubit encoding), with 117 automated tests. The
tests are not superficial: every stage is checked against an independent
reference (PySCF's own CASCI energies, PennyLane's independently constructed
qubit Hamiltonians, brute-force expectation values, explicitly built
symmetry transformations).

**Two solver backends.** The workhorse is dense exact diagonalization — exact
answers, the full spectrum, about a minute per molecule at CAS(8,8) on this
machine. For larger active spaces it refuses politely (memory guardrails)
and a second, matrix-free Krylov backend takes over: it converges only the
low-energy window that low temperatures actually need, together with a
*certified* mathematical bound on everything it left out. That backend has
handled active spaces up to dimension 853,776 (12 orbitals, HCN) on a
desktop. Its certificate makes it a low-temperature tool by design; hot
ensembles at large sizes await a Phase-2 sampling method.

**Honest truncation everywhere.** Storage caps how many eigenvectors are
kept (configurable via `--keep-cap`, including "keep everything"). Whenever
any cap or cutoff bites, the discarded Boltzmann weight is recorded in the
output — never silently dropped — so every downstream number carries its own
error bar.

**Qubit encoding (the classifier bridge).** Each thermal state can be mapped
onto a register of qubits by the Jordan-Wigner transformation (one qubit per
spin-orbital: 16 qubits for CAS(8,8)). On top of that sits the
**extended-Heisenberg feature map**: the 248 two-qubit-or-smaller Pauli
measurements that symmetry allows to be nonzero for these states. Evaluating
all 248 for a molecule takes under a second because the computation happens
in the compact determinant basis — the 65,536-dimensional qubit space is
never actually built. A further **Z₂ tapering** step exploits two exact
parity symmetries to shrink the register from 16 to 14 qubits at zero cost:
the stored feature values carry over unchanged (up to recorded ± signs), and
the classifier's core linear-algebra cost drops 16-fold.

**One data-integrity save worth recording.** While wiring up the loader we
discovered that the raw QH9 Hamiltonian blobs are *already* in PySCF orbital
order, and that applying the published reordering convention to them —
as an earlier script did — silently corrupts the physics (spectra shift by
up to an Hartree). An audit confirmed 284 GB of previously derived data was
affected; it was removed, the bug fixed at both sites, and the finding
documented (`data/qh9_raw_sqlite_audit.md`).

## What has been produced so far

| dataset | contents |
|---|---|
| `results/qh9_dense_cas8-8_kT0p1.h5` | **50 molecules**, CAS(8,8), kT = 0.1 Ha — the current production set (2.1 GB) |
| `results/qh9_dense_cas8-8_kT0p1_extheis.h5` | 248 Pauli features per molecule, plus the tapered 14-qubit basis |
| `results/qh9_dense_cas8-8_kT0p25.h5` | 4 molecules at kT = 0.25 Ha (run interrupted; resumable) |
| `results/qh9_krylov_ncas12_hcn.h5` | Krylov milestone: HCN at dimension 853,776, certified |
| `figures/qh9_cas8-8_kT0p1_diagnostics.png` | two-panel summary of the 50-molecule run |

Findings from the 50-molecule kT = 0.1 set, in plain terms:

- **The temperature sits in the interesting middle regime.** The ground state
  keeps 29–84% of the weight (median 57%); ensembles are strongly mixed but
  still differ sharply *between* molecules — which is what a labeled dataset
  wants. (At kT = 0.25 everything was uniformly hot; at 0.025, uniformly cold.)
- **Chemistry drives the numbers in interpretable ways.** How mixed a
  molecule gets tracks its energy gap (correlation −0.78): conjugated chains
  like diacetylene and cyanoacetylene mix the most, saturated molecules like
  methane and water the least. The distance to the non-interacting reference
  splits cleanly by chemical family — saturated skeletons cluster low
  (0.34–0.51), nitrogen/oxygen π-systems saturate high (0.93–0.99) — a
  promising, physically meaningful classification target.
- **The states are "classical plus a quantum residue."** 99.7% of the Pauli
  feature weight is occupation information that factorizes into single-qubit
  products; the genuinely collective part (occupation covariances at 0.2%,
  hopping coherences at 0.01%) is small but structured — and earlier
  measurements showed it is exactly the informative part for classification.
- **Truncation is negligible here.** Worst case, two parts in ten thousand
  of a molecule's thermal weight fell outside the stored states, and the
  exact deficit is recorded per molecule.

## Known caveats

- The non-interacting reference used in the trace-distance audit omits
  mean-field electron repulsion, which inflates the distance for compact
  molecules; part of the family split above may reflect that artifact. A
  mean-field-corrected reference is a known possible follow-up.
- CASCI on stored B3LYP (Kohn-Sham) orbitals is a deliberate Phase-1 choice —
  faithful to QH9, not variationally optimal.
- The kT = 0.25 CAS(8,8) dataset is 4/100 complete and, at that temperature,
  wants the `--keep-cap 0` setting introduced since (its capped molecules
  are missing 0.1–4% of their thermal weight — recorded, but not ideal).

## Natural next steps

1. Finish or rerun the kT = 0.25 set (uncapped) and encode it, giving each
   molecule a feature vector at two temperatures.
2. Train the classifier on the 14-qubit tapered features (or the 13-qubit
   sector encoding where fermionic structure is not needed).
3. Phase-2 solver work when bigger active spaces or hotter ensembles are
   needed: a sampling backend (TPQ/METTS), tensor-network states, and a
   warm-started Krylov escalation (the known 40% waste in the current
   escalation loop).
