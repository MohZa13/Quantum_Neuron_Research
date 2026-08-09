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

## Status

Validated end to end against exact references for `ncas <= 6`, and bracketed at
both ends at `ncas = 10`. The method **reaches** ncas = 10; converging it there
costs more bond dimension than this run spent. Open: a run at χ in the low
thousands to converge kT ≤ 0.5, and the second molecule in the file.

Most of the wall time is the subspace expansion, not TDVP — `expand` builds
`H|psi>` at bond `chi * 103` before truncating. Bounding that intermediate is
the obvious next optimisation.
