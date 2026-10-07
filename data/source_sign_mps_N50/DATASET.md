# Source-sign quench: full 50-qubit MPS states — how to read them

**What these are.** The complete state of a 50-qubit XX or XXX chain, stored as
a matrix product state, at every Trotter step after the middle qubit (qubit 25)
was given a quarter turn about z in one of two directions. Each file is one
state with three labels: **Hamiltonian** (XX or XXX), **rotation direction**
`s = ±1`, and **cumulative evolution time `t`**. Nothing has been traced out.
A training region R (k qubits at offset d from qubit 25) is cut out later with
the reader shipped beside the data, so any region can be chosen after the fact.

> This file is copied next to the data as `DATASET.md`. Beside it,
> `README.md` is the hands-on guide to cutting regions from (k, d)
> (`docs/SOURCE_SIGN_REGIONS.md`), with `make_regions.py` and
> `source_sign_mps_reader.py` (numpy + h5py only) as its executable half.
> The folder that is sent is assembled by `scripts/source_sign_package.py`.
> The source copies are `docs/SOURCE_SIGN_MPS.md`,
> `scripts/source_sign_make_regions.py` and
> `scripts/source_sign_mps_reader.py`. The generator is
> `scripts/source_sign_full_mps.jl`, the validation
> `scripts/source_sign_mps_validate.py`, and the specification is
> `bit_encoding/` (paper Eqs. 1, 4, 5 and 8 plus the whiteboard protocol).
> The sibling reduced-state datasets (fixed windows, already traced) are
> described in `docs/SOURCE_SIGN_DATASETS.md`; the physics is identical.

**Contents:** 1 quick start · 2 physics · 3 files and labels · 4 schema ·
5 conventions · 6 cutting out a region · 7 accuracy · 8 regenerating

---

## 1. Quick start

```python
import source_sign_mps_reader as smr

rows = smr.manifest(".")                     # one dict per state: hamiltonian, s, t, file, ...
psi  = smr.load("XXX/XXX_s+1_t06.400.h5")   # one state
psi.hamiltonian, psi.s, psi.t                # ('XXX', 1, 6.4) -- the three labels

rho = smr.region_rho(psi, k=5, d=2)          # 5 qubits centred on qubit 25 + 2 = 27
rho.shape                                    # (32, 32) complex128, unit trace

rho = smr.reduced_density_matrix(psi, [24, 25, 26])   # any explicit qubit list (1-based)
for d, sites, rho in smr.all_regions(psi, k=3):        # every 3-qubit region, one sweep
    ...
```

A training set in the whiteboard's sense, `{(s_i, ρ_{t_i,R_i}(s_i))}`, for one
Hamiltonian and region size, over chosen offsets and times:

```python
ts = smr.training_set(".", "XXX", k=3, ds=range(-5, 6), times=[0.5, 1.0, 1.5])
ts["rho"].shape, ts["s"], ts["t"], ts["d"]   # (66, 8, 8), labels, times, offsets
```

The two labels of one (t, d) cell are adjacent and are exact spin flips of
each other; keep them on the same side of a train/test split.

`python source_sign_mps_reader.py .` re-verifies every file (§7) and rewrites
`manifest.csv`.

## 2. The physics

Open chain, `N = 50` qubits (spins-1/2), `J = 1`, so times are in units of `1/J`:

```
H_eps = 2J Σ_{j=1}^{49} ( Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + eps Sz_j Sz_{j+1} ),   S = σ/2
      = (J/2) Σ_j ( X_j X_{j+1} + Y_j Y_{j+1} + eps Z_j Z_{j+1} )
```

- **XX**: eps = 0. Free fermions under Jordan–Wigner.
- **XXX**: eps = 1, the isotropic Heisenberg antiferromagnet.

**Ground state.** `|g⟩` is the ground state in the zero-total-Sᶻ sector
(unique for even N in both models), found by DMRG with Sᶻ conservation.

**The rotation (the label).** On qubit `j0 = 25`, one of the two central
qubits:

```
V_s = exp(-i s π σᶻ_25 / 4) = (I − i s σᶻ_25)/√2,        s = +1 ("+")  or  s = −1 ("-")
```

**Time evolution.** `|ψ_s(t)⟩ = e^{−iHt} V_s |g⟩`, by TEBD with
nearest-neighbour two-qubit gates over the whole chain. **One Trotter step** is
one 4th-order Suzuki step of length `dt = 0.1`:

```
S2(τ) = Π_{b=1}^{49} G_b(τ/2) · Π_{b=49}^{1} G_b(τ/2),      G_b(τ) = exp(−iτ h_{b,b+1})
S4(dt) = S2(p dt) S2(p dt) S2((1−4p) dt) S2(p dt) S2(p dt),  p = 1/(4 − 4^{1/3})
```

The state after **every** step is stored, at cumulative time `t = step × dt`,
from `t = 0` (step 0, which is `V_s|g⟩` before any evolution) to `t = 13`
(step 130). After each step the MPS is truncated (singular values below a
relative discarded weight of 1e-10, bond dimension at most 1024) and
renormalised.

**How far the signal has travelled.** The kick spreads at a finite speed:
`v = 2J` for XX and `v ≈ πJ` for XXX (sites per unit time). The front reaches
the nearer chain end (qubit 1, 24 sites away) at t ≈ 12 for XX and t ≈ 7.6 for
XXX, so **XXX states after t ≈ 7.6 and XX states after t ≈ 12 contain
reflections from the chain ends**. Regions further than `v t` from qubit 25
have not yet been reached and carry almost no record of `s` (§7 of
`SOURCE_SIGN_DATASETS.md` has the measured falloff).

**Why the task is well posed.** The two global states are orthogonal at every
t, `⟨ψ_+|ψ_−⟩ = i⟨g|σᶻ_25|g⟩ = 0`, so with the whole chain `s` can always be
read perfectly. Only the restriction to a region makes it hard.

## 3. Files and labels

```
<root>/
  README.md                       how to make region states from (k, d): start there
  DATASET.md                      this document
  CLAUDE.md                       orientation for an AI agent working in the folder
  make_regions.py                 command-line region extractor
  source_sign_mps_reader.py       the reader library (numpy + h5py)
  requirements.txt, SHA256SUMS    dependencies; checksums of every file
  manifest.csv                    one row per state
  validation.json                 the numbers in §7
  XX/   XX_s+1_t00.000.h5 … XX_s+1_t13.000.h5,  XX_s-1_t00.000.h5 … XX_s-1_t13.000.h5
  XXX/  XXX_s+1_t00.000.h5 …                     XXX_s-1_t00.000.h5 …
```

**524 files, 4.0 GB:** 131 Trotter steps (t = 0.0, 0.1, …, 13.0) × 2 rotation
directions × 2 Hamiltonians. XX files are at most 4.1 MB (bond dimension
χ ≤ 172); XXX files grow to 25 MB (χ ≤ 448, reached at t ≈ 11).

**The filename carries all three labels:** `<HAMILTONIAN>_s<+1|-1>_t<t>.h5`, with
t to three decimals. The same labels are file attributes (§4), and
`manifest.csv` lists them with the run diagnostics:

| column | meaning |
|---|---|
| `hamiltonian` | `XX` or `XXX` |
| `s` | +1 or −1, the rotation direction (the classification label) |
| `t` | cumulative evolution time, units of 1/J |
| `step` | Trotter step index, `t = step × 0.1` |
| `file` | path relative to the root |
| `maxlinkdim` | largest bond dimension of this state |
| `energy` | ⟨H⟩, conserved by the exact dynamics (drift = Trotter + truncation error) |
| `norm_loss_cumulative` | total discarded norm from truncation up to this step |

**The two labels were simulated separately**, from the same ground state.
They are related by an exact symmetry, `|ψ_−(t)⟩ = F|ψ_+(t)⟩` with
`F = Π_j σˣ_j` (flip every qubit), and the files satisfy it to the precision in
§7. A consequence for training: any statistic that is symmetric under flipping
every qubit of the region is identical for the two labels.

## 4. Schema (one file = one state)

| object | shape, dtype (as numpy/h5py sees it) | meaning |
|---|---|---|
| `mps/site_01` … `mps/site_50` | `(χ_{j−1}, 2, χ_j)` complex128 | the site tensors, `χ_0 = χ_50 = 1` |
| `bond_dims` | `(51,)` int | `χ_0 … χ_50` |
| `checks/sz` | `(50,)` float64 | ⟨σᶻ_j⟩ for j = 1…50, computed by the generator |
| `checks/rho_k3_d0` | `(8, 8)` complex128 | ρ on qubits 24, 25, 26 (§5 convention), computed by the generator |

**Attributes** (`f.attrs`): the labels `hamiltonian`, `s`, `rotation` (`"+"` or
`"-"`), `t`, `step`; the model `eps`, `J`, `n_qubits`, `j0`, `rotation_gate`;
the method `dt`, `trotter`, `cutoff`, `maxdim`, `E0` (ground-state energy),
`gs_variance`, `gs_dmrg_maxdim`; the diagnostics `energy`,
`norm_loss_cumulative`, `maxlinkdim`; the conventions `basis`,
`tensor_layout`, `canonical_form`; and the provenance `generator`,
`generator_git`, `created`, `schema_version = 1`.

Tensors are gzip-compressed. They are stored dense but are block-sparse
(the dynamics conserves total Sᶻ), so most entries are exact zeros and
compress away.

## 5. Conventions that fail silently if you get them wrong

**Site and physical index.** Qubit `j` is site `j`, 1-based, `j = 1 … 50`;
qubit 25 is the rotated one. The physical index of every tensor is
**0 = spin up (σᶻ = +1), 1 = spin down**. The amplitude of a basis state is

```
ψ(b_1, …, b_50) = A_1[:, b_1, :] · A_2[:, b_2, :] · … · A_50[:, b_50, :]
```

**Gauge.** Files are right-canonical (sites 2…50 right-orthonormal, the norm
on site 1, `⟨ψ|ψ⟩ = 1`). The reader does not rely on it: it computes both
environments explicitly, so it is also correct for any MPS you build yourself.

**Region density matrices are little-endian.** For a region with qubits
`sites[0] < sites[1] < …`, qubit `q` (site `sites[q]`) is **bit q** of the
row index: `r = Σ_q b_q 2^q`, and `rho[r, r'] = ⟨r|ρ|r'⟩` (ket = row). This is
the convention of `qnn/pools.py` and of the reduced-state datasets, so a
region goes to `qnn` unrelabelled.
- `np.kron(A, B)` puts `A` in the **most** significant position, so a
  kron-built operator on the region must list its factors **reversed**:
  `kron(A_{k−1}, …, A_0)`.
- Pass `little_endian=False` to get the textbook kron order instead
  (first qubit most significant).

**Keep the states complex.** At t = 0 the two labels' region states are
exact complex conjugates (`ρ_− = ρ_+*`, because H and |g⟩ are real). Dropping
the imaginary part — or transposing ρ, which is the same thing for a Hermitian
matrix — makes every t = 0 pair identical, or swaps the labels.

**Julia readers:** HDF5.jl reads column-major, so every array appears with its
axes reversed: a site tensor is `(χ_j, 2, χ_{j−1})` and `checks/rho_k3_d0` is
`[bra, ket]`. Use `permutedims(A, (3, 2, 1))` and `transpose` to recover the
shapes above. Physical index 1 (Julia) = up.

## 6. Cutting out a region

A region of `k` qubits at offset `d` (the paper's notation) is the `k`
contiguous qubits centred on qubit `25 + d`:

```
sites = 25 + d − (k−1)/2, …, 25 + d + (k−1)/2           (k odd)
```

`smr.region_sites(k, d)` returns them and refuses a region that runs off the
chain; for k = 7, `d` runs from −21 to +22. `dist = max(0, |d| − (k−1)/2)` is
the number of qubits between qubit 25 and the region's nearest qubit (0 when
the region contains it).

The reader contracts `ρ_R = Tr_{R̄} |ψ⟩⟨ψ|` exactly from the MPS: the qubits
left and right of R are traced through their transfer matrices, and the region
block is contracted with its physical legs open. Cost grows as `χ³ · 2^k`.
Measured on 8 cores, for the largest XXX state (χ = 448), one region takes
0.4 s (k = 1) to 2.7 s (k = 7), and all 44 seven-qubit regions take 41 s with
`all_regions`, which shares the environments across the regions of one
state. XX states (χ ≤ 172) are about 10× faster. Peak memory for one k = 7
region at χ = 448 is 1.6 GB (measured). A non-contiguous qubit list is also accepted (the gaps
are traced out of the contiguous span, so keep the span ≲ 10).

The same (k, d) at t a multiple of 0.2 is exactly a cell of the reduced-state
datasets in `SOURCE_SIGN_DATASETS.md`, which were made by a different run of
the same physics; §7 compares the two.

## 7. Accuracy

All numbers are from `validation.json` (`scripts/source_sign_mps_validate.py`).
Distances between region states are **trace distances** ½‖ρ − ρ'‖₁, maximised
over every region position, both labels and the stated times.

**The reader and the files agree with the generator.** For every one of the
524 files, the norm, ⟨σᶻ_j⟩ on all 50 qubits and ρ on qubits 24–26,
recomputed by `source_sign_mps_reader.py`, match the values ITensor computed
when it wrote the file to **≤ 4e-15**. This is what certifies the on-disk
layout, the site order, the physical-index order and the ket/bra order.

**The two labels are each other's spin flip, as they must be.** Although
simulated separately, `⟨Fψ_+(t)|ψ_−(t)⟩ = −1` to **9e-14** at every step in both
models (−1 is the ground state's spin-flip parity at N = 50, so
`|ψ_−⟩ = −F|ψ_+⟩`). `⟨ψ_+|ψ_−⟩`, exactly 0 in theory, stays below 9e-6.

**XX against the exact answer.** The XX chain is free-fermionic, so every
region's ρ is known exactly at N = 50 (`scripts/source_sign_data.ff_rdms`, no
shared code). Every region of every k, every one of the 131 times, both labels:

| k | max over t ≤ 4 | median over t | max over all t ≤ 13 |
|---|---|---|---|
| 1 | 4.9e-6 | 6.5e-6 | 2.2e-5 |
| 3 | 2.4e-5 | 5.2e-5 | 1.3e-4 |
| 5 | 7.0e-5 | 9.1e-5 | 1.8e-4 |
| 7 | 1.6e-4 | 1.6e-4 | 2.7e-4 |

**XXX against two earlier, independent runs** of the same physics
(`results/source_sign/raw/`, which stored only region states), every region
on the shared 0.2 grid, both labels (k = 7 at integer t):

| k | vs dt = 0.025, cutoff 1e-12 (t ≤ 7.2) | vs dt = 0.05, cutoff 1e-10 (t ≤ 13) |
|---|---|---|
| 1 | 2.1e-5 | 4.2e-5 |
| 3 | 4.3e-5 | 9.0e-5 |
| 5 | 8.2e-5 | 1.8e-4 |
| 7 | 1.4e-4 | 3.4e-4 |

**Run diagnostics.** The energy ⟨H⟩ (exactly conserved) drifts by at most
6.4e-6 (XX) and 1.8e-5 (XXX) over the run, out of |E| ≈ 30 and 43. The
total norm discarded by truncation is 3.2e-6 (XX) and 3.9e-6 (XXX). The
bond-dimension cap of 1024 was never reached.

**What the errors mean for training.** The error grows with k and t, and
stays **≤ 3.4e-4 for every region up to k = 7 at every time**. Inside the
light cone the true distance between the two labels' region states is
between ~0.01 and ~0.99, so the error is at least 30× smaller than the
signal. Far outside the cone the true signal itself decays below this level,
and such regions should be read as carrying no record of `s`.

**Choice of step.** One Trotter step is dt = 0.1 of a 4th-order scheme. At
dt = 0.2 the Trotter error of XXX alone is already ~1e-5 by t = 1. At dt = 0.1
the error is no larger than at dt = 0.05 through t = 4: truncation, not the
Trotter step, sets the error budget above (`docs/DECISIONS.md` D26).

## 8. Regenerating

From the repository root (Julia project `SpinThermalMPS`; the full procedure
is in `docs/WORKFLOWS.md`, "Full-chain source-sign MPS"):

```bash
for e in 0 1; do   # ground states, ~2 min each
  julia --project=SpinThermalMPS scripts/source_sign_full_mps.jl ground --eps $e
done
for e in 0 1; do for s in +1 -1; do   # the four runs, in parallel
  nohup julia --project=SpinThermalMPS scripts/source_sign_full_mps.jl evolve --eps $e --sign $s \
      --dt 0.1 --tmax 13 --cutoff 1e-10 --maxdim 1024 > evolve_eps${e}_s${s}.log 2>&1 &
done; done
.venv/bin/python scripts/source_sign_mps_validate.py --root results/source_sign_mps
.venv/bin/python scripts/source_sign_mps_reader.py results/source_sign_mps   # check + manifest
```

A killed `evolve` run resumes with `--resume true` from its last completed
step.
