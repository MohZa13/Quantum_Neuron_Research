# QThermalMPS — thermal states as purification MPS, by imaginary-time evolution

Module K. Takes the `(ecore, h1eff, g)` an active-space run already stores and
produces `rho(kT)` as a **matrix product state**, without diagonalizing
anything.

```
|Psi(0)>  = (1/sqrt(dim)) sum_{n in sector} |n>_phys |n>_anc      exactly rho(0) = P/dim
|Psi(b)>  = e^{-b H / 2} |Psi(0)> / norm                          by two-site TDVP
rho(b)    = Tr_anc |Psi(b)><Psi(b)|                               a plain qubit partial trace
```

**Why this exists.** The Python pipeline forms the thermal state from an
explicit eigenbasis, so it stops where dense `eigh` does: CAS(8,8) is a
sector of dimension 4900, CAS(8,10) is 63,504, CAS(8,12) is 853,776. And
`qthermal/mps.py` (Module J), which does build a purification MPS, gets there
through a `2^Q` scatter that dies at `ncas = 8`. Nothing here is ever
diagonalized and no `2^Q` object is ever formed.

**One pass gives every temperature.** Imaginary-time evolution sweeps *through*
every intermediate `beta`, so each requested `kT` is a snapshot. The whole
ladder costs what its coldest rung costs.

**The norm is the partition function.** Because `<Psi(0)|Psi(0)> = 1` and
`rho(0) = P/dim`,

```
Z(beta) = Tr e^{-beta H} = dim * || e^{-beta H/2} |Psi(0)> ||^2
```

so `logZ`, the free energy and the entropy come out of the evolution for free,
provided the norm is not thrown away. It isn't: the state is renormalized in
bounded chunks with the logarithm accumulated.

## Quick start

```julia
using QThermalMPS
enable_threading!()                       # start julia with -t 8

c = read_case("results/qh9_dense_cas88_5mols.h5", "mol_0")
L, sites, H, snaps = thermal_ladder(c, [1.0, 0.5, 0.25, 0.1]; maxdim = 300)

snaps[end]                                # ThermalSnapshot(kT=0.1 ... chi=214 ...)
rho = physical_rdm(snaps[end].psi, L, 0:9)   # dense 1024x1024, for qnn/
pauli_expect(snaps[end].psi, L, "ZZ" * "I"^14)
```

or from the shell:

```bash
julia -t 8 --project=QThermalMPS QThermalMPS/bin/thermal.jl \
    --in results/qh9_dense_cas88_5mols.h5 --out results/qh9_cas88_mps.h5 \
    --kT 1.0,0.5,0.25,0.1 --maxdim 300 --rho-wires 0:9
```

## Layout

| file | role |
|---|---|
| `layout.jl` | wire conventions — a **contract** with `qthermal/encode.py` |
| `sites.jl` | site indices and local operators; the ancilla-statistics decision |
| `purification.jl` | the `beta = 0` sector state; lifting operators onto the chain |
| `hamiltonian.jl` | `(h1eff, g)` -> OpSum -> MPO |
| `snapshot.jl` | `ThermalSnapshot` |
| `io.jl` | run files in, ladders out |
| `observables.jl` | `rho`, reduced density matrices, Pauli expectations |
| `evolve.jl` | the imaginary-time ladder |
| `fused.jl` | the fused wire+ancilla backend (below) |

## Four things that are easy to get wrong

**1. Ancillas are bosonic, and that is load-bearing.** They carry their own
U(1) charges (`Na`, `Sza`) with no fermion-parity flag, so a Jordan-Wigner
string running between two physical wires passes through the ancillas between
them as the identity. That is what makes `Tr_anc` — a *plain qubit partial
trace* — equal the Module I encoded `rho`.

Make the ancillas fermionic instead and every JW string picks up a `Z` on each
intervening ancilla. For two wires at one electron with `H = -t(c+_0c_1 +
h.c.)`, first order in `beta`:

| | `<10|rho|01>` |
|---|---|
| fermionic ancillas, naive `Tr_anc` | **0** |
| bosonic ancillas, naive `Tr_anc` | `beta*t/2` (correct) |

`test/test_sites.jl` runs exactly this. The price is that the honest pair
creator `c+_p b+_a` is *odd*, so distinct wires anticommute and
`(sum_w X_w)^2 = 0` — the `beta = 0` state cannot be built by exponentiating
it. It is built from parity-free `"Raise"` operators instead.

**2. TDVP silently converges to the wrong state unless the bond space is
opened first.** TDVP moves *within* the manifold of its current bond
dimension. The `beta = 0` purification starts at `chi = O(ncas^2)`; under
U(1)xU(1) whole quantum-number blocks are missing, and two-site TDVP often
cannot grow into them at all. Measured on `h2o` CAS(8,6): `chi` stayed at 6
from `beta = 0` to `beta = 40`, and `logZ` was wrong by 0.4.

The failure is invisible to the two checks you would reach for first:

- **thermodynamic consistency proves nothing** — `d logZ / d beta = -<H>` holds
  *exactly* even on the wrong manifold, because the tangent-space projection
  preserves `<Psi|H|Psi>`. `logZ` and `energy` stay perfectly consistent with
  each other and both are wrong.
- **a step-size study proves nothing** — halving `dbeta` reproduces the same
  wrong number to every digit, because the manifold is exactly invariant.

So a global Krylov subspace expansion runs before each chunk while `chi` is
still growing, adding directions at zero amplitude. It is on by default. The
one symptom that *does* show it is `ThermalSnapshot.maxlinkdim` never rising
above its `beta = 0` value; `bin/thermal.jl` warns on it.

**3. Prime the bra by index, not by tag.** `prime(dag(psi), "Link")` is the
idiomatic spelling and it is wrong here: `expand` returns bonds tagged `"sum"`,
which that call misses, so ket and bra contract straight through them and
`tr(rho)` comes out a power of two too large. `observables.jl` primes
everything that is not the site index.

**4. Use `:blocked` wire ordering.** Measured exact bond dimension of the
thermal purification — no TDVP, dense diagonalization then TT-SVD:

| case | `:blocked` | `:interleaved` |
|---|---|---|
| random CAS(4,4), any `beta > 0` | 36 (= sector dim) | **256** (no compression at all) |
| `h2o` CAS(8,6), `kT = 0.25`, cutoff 1e-10 | 221 | **821** |
| `h2o` CAS(8,6), `kT = 0.25`, cutoff 1e-6 | 118 | **582** |

Interleaved is 4–5x more expensive *intrinsically*, not as a numerical
accident. This confirms and sharpens the standing guidance in
`docs/RESEARCH_LOG.md` (2026-07-27) and `INVARIANTS.md` I12 — which was
measured for the eigenblock purification, and now holds for the
imaginary-time one too.

## The fused backend, and the geometry measurement behind it

`FusedLayout` puts each wire and its own ancilla on ONE dim-4 site: half the
chain at measured-identical bond dimension. The measurement (dense TT-SVD of
the same thermal purification, h2o CAS(8,6), kT = 0.25):

| grouping | sites | χ@1e-10 |
|---|---|---|
| qubit-blocked (split backend) | 24 | 221 |
| **fused wire+ancilla** | **12** | **222** |
| orbital-fused (ITensor "Electron" sites) | 12 | **664** |

The obvious chain-halving — Electron sites, as in standard QC-DMRG — forces
the orbital geometry these thermal states pay 3× χ for (~30× at χ³ per
update), so it is deliberately not implemented. The fused backend measures
1.37× end-to-end on a real ladder (setup 97 s → 7 s, ladder 1057 s → 837 s,
h2o CAS(8,6), same accuracy), with two structural bonuses: JW strings cannot
dress ancillas (the string operator is `(-1)^{n_phys}` of the site), and the
MPO compiles directly on the chain (3 QNVals used, one left for the compiler).

**Trap 5, found here: `ITensorMPOConstruction.MPO_new` mis-signs fermionic
operators on the fused site type** — hopping terms acquire an extra
`(-1)^{n_anc}` on the operator site. It is a sign-gauge conjugation of the
right Hamiltonian, so `<H>`, `<H^2>`, and every thermodynamic self-consistency
check pass; only an evolved state exported to the fixed register (or the
element-by-element dense comparison in `test_fused.jl`) shows it. The fused
backend therefore compiles with ITensor's own `MPO(os, sites)` (`alg =
:itensor` default) — once per molecule, cost irrelevant.

## RDM export: meet-in-the-middle

Sweeping one environment across the chain carries all `k` open wires through
every bond — `Σ_k 4^k χ³` work, ~15 min per cold ncas = 10 state in the first
production run. `physical_rdm` now builds left and right half-environments
that each carry half the open register and joins them once
(`O(4^{k/2} χ³ + 4^k χ²)`), with ket/bra opens folded into combined QN
indices to keep block counts flat. Measured on the stored production states,
bit-exact against the old path:

| state | old | combine only | meet+combine |
|---|---|---|---|
| mol_3 kT = 0.1, χ = 256 | ~900 s | 393 s | **5.0 s** |
| mol_3 kT = 1.0, χ = 64 | 637 s | 246 s | 173 s |

The warm state stays slower because it is block-bound, not FLOP-bound: near
ln(dim) entropy spreads weight over many QN sectors per bond. (The fused
backend drops the `Sza` charge and coarsens blocks, which helps exactly
there.)

## Cost

`ncas = 6` is a *validation* size, not a payoff size: the sector is only 225
states, the exact purification needs `chi ~ 200`, and dense diagonalization
wins. The method earns its keep where dense cannot go at all — CAS(8,10) and
beyond, where the sector is 63,504+ but `chi` need not be.

Two levers dominate:

- `dbeta` — graded, `min(dbeta_max, dbeta*(1 + beta/ramp))`. Reaching
  `kT = 0.025` means `beta = 40`; on a uniform grid at `dbeta = 0.05` that is
  800 steps and the last 700 do nothing, because the state stops moving as the
  thermal weight concentrates. Halving `dbeta` still scales the whole schedule,
  so it remains a one-knob convergence study.
- `maxdim` / `cutoff` — see the table above for what is actually needed.

`enable_threading!()` turns on ITensor's threaded block-sparse contractions
(and serialises BLAS and Strided, as ITensor recommends — the QN blocks here
are small and numerous, so threading at both levels oversubscribes). Requires
`julia -t N`.

## Tests

```bash
julia -t 8 --project=QThermalMPS QThermalMPS/test/runtests.jl
```

**730 assertions, all passing** (layout 233, sites 22, hamiltonian 40,
purification 164, observables 169, evolve 102). Budget ~45 min single-threaded;
`evolve` is 30 of it, because most of its assertions are against a dense
reference and so have to actually run ladders.

Gate order: `layout` -> `sites` -> `hamiltonian` -> `purification` ->
`observables` -> `evolve`. Two independent references are used throughout, and
a bug would have to fool both:

1. a **by-hand** second-quantized Hamiltonian and dense `exp(-beta H)`, built
   with explicit JW parity signs and no tensor network anywhere — this pins the
   *basis*, not just the spectrum, which is what `rho` has to agree with Module
   I about;
2. Boltzmann sums over the `evals` the Python pipeline wrote — cross-language,
   and the only check that touches real molecular data.

Tests needing `results/*.h5` skip themselves with a message when those files
are absent (they are gitignored and must be regenerated).

## Production run, ncas = 10

QH9 `mol_3` (acetylene), CAS(10,10), CI sector **63,504** — past dense `eigh`
and past Module J's `2^Q` scatter. Full ladder in one pass, 88 min on 8
threads. Artifact: `results/qh9_mps_ncas10.h5`.

No spectrum exists at this size, so the run is bracketed at both ends instead:

| end | quantity | run | reference | error |
|---|---|---|---|---|
| hot (β=0) | `<Psi(0)\|H\|Psi(0)>` | −21.663909237992 | −21.663909237992 (`sector_mean_energy`) | **1.1e-14** |
| cold | ground state | −25.2277762909 (DMRG) | −25.2277763085 (Python Krylov) | **1.8e-08** |

| kT | β | χ | E | S | log Z | wall |
|---|---|---|---|---|---|---|
| 4.00 | 0.25 | 19 | −21.94845447 | 11.023741 | 16.5109 | 104 s |
| 2.00 | 0.50 | 38 | −22.21847859 | 10.922979 | 22.0322 | 96 s |
| 1.00 | 1.00 | 64 | −22.70637369 | 10.561502 | 33.2679 | 329 s |
| 0.50 | 2.00 | **246** | −23.44480442 | 9.477765 | 56.3674 | 799 s |
| 0.25 | 4.00 | **256** | −24.27225250 | 7.100572 | 104.1896 | 1726 s |
| 0.10 | 10.00 | **256** | −25.04565217 | 2.321989 | 252.7785 | 2248 s |

`S` starts at `ln(63504) = 11.0589` and falls monotonically; `E` falls toward
the DMRG ground state, reaching `E − E0 = 0.182` at kT = 0.1 against a 0.207
gap.

**Read the artifact's rungs at kT ≤ 0.5 as upper bounds (~1–2e-2 Ha), but the
method converges fast.** χ pinned at the cap from kT = 0.5 down. The full sweep
at β = 2 (`dbeta = 0.05`, `cutoff = 1e-10`):

| maxdim | χ | E | shift | ratio |
|---|---|---|---|---|
| 64 | 64 | −23.41979142 | — | |
| 128 | 128 | −23.44191070 | −2.21e-02 | |
| 256 | 256 | −23.45596820 | −1.41e-02 | 0.64 |
| 512 | 512 | −23.45950851 | **−3.54e-03** | **0.25** |

Convergence *accelerates* — geometric tails give ~5e-3 residual at χ=256 and
~1e-3 at χ=512. The production artifact's larger error is dominated by its
looser `cutoff = 1e-8` (−23.44480 vs −23.45597 at identical maxdim, a 1.1e-2
gap). Chemical accuracy at kT = 0.5 therefore needs χ ~ 512–1024 at
`cutoff ≤ 1e-10` — a 2–4× rerun, not thousands. Warm rungs are better but not
immune: independent re-runs at `cutoff 1e-9`/`maxdim 400` move kT = 4/2/1 by
5.8e-4 / 6.4e-3 / 6.9e-3, so only kT = 4 is sub-1e-3. Caveat for sweeps:
`expand_every` defaults to `10*dbeta`, so a step study silently varies the
expansion schedule too — pin it explicitly.

Each temperature also carries a dense 1024×1024 `rho` over JW wires 0–9, the
form `qnn/states.py` consumes. Read back in Python: `tr(rho) = 1.0000000`
exactly, symmetric to 1e-16, PSD to 1e-17. The subsystem RDM is unit-trace *by
construction*, so unlike the top-determinants projection it carries no
truncation error for a threshold model to mistake for signal.

## Where the time goes (measured 2026-08-10)

Every stage runs inside a `TimerOutputs` section (`QThermalMPS.TIMER`);
`bin/thermal.jl --profile 1` prints the per-molecule table below plus a
per-chunk `chi`/time trace, and `--warmup` (on by default when profiling)
absorbs JIT into a tiny synthetic ladder first. The production mol_3 ladder,
re-run at its exact settings (reproduces every rung energy digit for digit;
8 threads, Julia 1.12.6):

| stage | wall | share |
|---|---|---|
| read + setup (opsum, compile, inflate, psi0) + guard | 5.9 s | 0.1% |
| ladder: **tdvp** | **5620 s** | **90.3%** |
| ladder: expand | 455 s | 7.3% |
| rho export (6 temperatures) | 141 s | 2.3% |
| everything else (writes, renorms, snapshot energies) | ~6 s | 0.1% |
| package load + JIT warmup, once per process | 211 s | — |

**An earlier version of this README claimed the expansion dominated; that was
an unmeasured hypothesis and the profile refutes it** — TDVP dominates every
rung. Acting on that profile (second 2026-08-10 RESEARCH_LOG entry), the
local-solver defaults were replaced: KrylovKit's stock `exponentiate` ran
Arnoldi at tol 1e-12 with no early exit, and the instituted default
(Lanczos + `eager` + tol = cutoff/10) is a **certified 4.4x end-to-end**
(6227 → 1411 s on the production mol_3 ladder, energies within truncation
noise, 798/798 tests) — `--solver-tol none` recovers the old behavior.
Opt-in `--nsite-capped 1` adds 2.36x on capped chunks at +2e-4/beta drift;
`rho` exports are now shuffle+deflate compressed (57 → ~10 MB per molecule,
transparent to readers). **`--backend fused` is EXPERIMENTAL at ncas = 10**:
it is no faster there (dim-4 sites are ~1.5x/step slower at the cap) and its
warm rungs misconverge by 4e-3..4e-2 with the trap-2 signature — see the
2026-08-10 RESEARCH_LOG entry before using it beyond ncas = 6. Three measured
facts that steer further optimisation
(`docs/RESEARCH_LOG.md` 2026-08-10 for the full analysis):

- Per-TDVP-step cost is strongly *sublinear* in `chi` here (24 s/step at
  chi = 19, 64 s at 64, ~103 s at 256): block-count-bound, not FLOP-bound, so
  the chi = 512-1024 convergence rerun costs far less than chi^3 fear
  suggests, and per-step overhead beats `maxdim` as a target.
- Expansion's footprint is mostly *indirect*: the tdvp chunk right after an
  expansion sweeps at the directsum-inflated bond (202 s/step vs 103 steady).
- 19 of the 57 TDVP steps cover beta 4→10 where the state barely moves — the
  `ramp`/`dbeta_max` grading is the cheapest big lever; `converge_dbeta`
  certifies it. The Lanczos `exponentiate` also runs at ~1e-12 tolerance
  against a 1e-8 truncation cutoff (`updater_kwargs` is plumbed, untested).

## Open shell, and scaling out

**Open-shell sectors (S_z != 0, odd electron counts) are supported**
(2026-08-11): the MPS layer was always `(nalpha, nbeta)`-general — layouts,
QN flux, the beta = 0 state, the closed form, both backends — and the one
blocker, `read_case`'s S_z = 0 file contract, now honors `nalpha`/`nbeta`
meta attributes (legacy files read unchanged). Validated against dense
exp(-beta H) in odd-electron sectors (`test/test_openshell.jl`) and
end-to-end on the allyl radical (ROHF-CASCI(3,3), doublet: exact-reference
agreement at the dbeta^2 level). Upstream caveat: QH9/Phase 1 only produce
closed-shell inputs (RHF-CASCI); open-shell files come from custom ROHF
exporters. Only S_z is conserved, not S^2 — the ladder prepares the
S_z-canonical ensemble. Expect larger chi for radicals: multiplet
degeneracies make flat Schmidt blocks.

**Parallel structure**: per-molecule and per-setting runs are independent
processes with zero cross-talk (job-array shaped; ~10-55 GB RSS each by
chi); a single ladder's temperatures are inherently sequential (one
imaginary-time flow); in-process threading saturates ~4x on 8 cores. Scale
out at the process level.

## Status

Validated end to end against exact references for `ncas <= 6`, and bracketed at
both ends at `ncas = 10`. The method **reaches** ncas = 10; converging it there
costs more bond dimension than this run spent. Open: a run at χ in the low
thousands to converge kT ≤ 0.5, and the second molecule in the file.
