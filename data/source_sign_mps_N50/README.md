# Source-sign MPS dataset: making R-region states from k and d

This folder holds the **complete 50-qubit quantum state**, stored as a matrix
product state (MPS), of a spin chain at every time step after one qubit was
rotated in one of two directions. Nothing has been traced out. This README
explains how to turn those states into **region states**: the reduced
density matrix ρ_R of a block R of `k` adjacent qubits whose centre sits `d`
qubits from the rotated one. Each region state, paired with the rotation
direction `s = ±1`, is one training sample `(s, ρ_R)` for a classifier.

> **Files in this folder.** `README.md` (this guide), `DATASET.md` (the full
> physics, file schema and accuracy report), `make_regions.py` (command-line
> region extractor), `source_sign_mps_reader.py` (the Python library behind
> it), `manifest.csv` (one row per stored state), `validation.json`,
> `SHA256SUMS`, `requirements.txt`, `CLAUDE.md` (orientation for an AI
> agent), and the states in `XX/` and `XXX/`.

**Contents:** 1 quick start · 2 the experiment · 3 region R from (k, d) ·
4 producing region states · 5 conventions · 6 which cells carry signal ·
7 notes for training · 8 cost · 9 verifying

---

## 1. Quick start

Python ≥ 3.9 with numpy and h5py (`pip install -r requirements.txt`).
Run everything from inside this folder.

```bash
sha256sum -c SHA256SUMS --quiet       # transfer intact? (silent = yes; macOS: shasum -a 256 -c)
python make_regions.py --selftest     # must end with "SELFTEST PASSED" (~30 s)

# 3-qubit regions at offsets d = -5 ... +5, every time step, both labels
python make_regions.py --hamiltonian XXX --k 3 --d -5:5 --out regions/XXX_k3.h5
```

```python
import h5py
with h5py.File("regions/XXX_k3.h5") as f:
    rho = f["rho"][()]        # (M, 8, 8) complex128 region states
    s   = f["s"][()]          # (M,) labels +1 / -1
    t, d = f["t"][()], f["d"][()]
```

## 2. The experiment, and the three labels of every stored state

An open chain of `N = 50` qubits (spins-1/2), numbered **1 to 50**:

```
H_eps = (J/2) Σ_{j=1}^{49} ( X_j X_{j+1} + Y_j Y_{j+1} + eps Z_j Z_{j+1} ),   J = 1
```

1. **Hamiltonian** (`hamiltonian`, the folder name): `XX` is eps = 0 (free
   fermions), and `XXX` is eps = 1 (Heisenberg antiferromagnet).
2. Start in the ground state |g⟩ (zero total magnetisation).
3. **Rotate qubit 25** (`j0`) by a quarter turn about z. The direction is the
   **label** `s`:
   `V_s = exp(−i s π Z_25 / 4) = (I − i s Z_25)/√2`, with `s = +1` or `s = −1`.
4. Evolve under `e^{−iHt}` with 4th-order Trotter steps of `dt = 0.1`. The
   state after every step is stored, so **time** `t` = 0.0, 0.1, …, 13.0
   (131 values). `t = 0` is the rotated ground state before any evolution.

So there are 2 Hamiltonians × 2 labels × 131 times = **524 states**, one file
each: `XXX/XXX_s+1_t06.400.h5` is XXX, s = +1, t = 6.4. With the whole chain
the two labels are perfectly distinguishable at any t (the global states are
orthogonal). The question the data poses is how much of `s` a small region
still reveals.

## 3. The region R, from k and d

`k` is the number of qubits in R (odd), and `d` is the offset of R's
**centre** from the rotated qubit 25:

```
R(k, d) = { 25 + d − (k−1)/2,  …,  25 + d + (k−1)/2 }          (1-based qubit numbers)

qubit:  1  2 …  19 20 21 22 23 24 [25] 26 27 28 29 … 50
                                    ^ rotated qubit (j0)
k = 1, d = 0    ->  {25}
k = 3, d = 0    ->  {24, 25, 26}
k = 3, d = +2   ->  {26, 27, 28}
k = 5, d = −4   ->  {19, 20, 21, 22, 23}
k = 7, d = +22  ->  {44, …, 50}                  (the right end of the chain)
```

The region must fit on the chain, which bounds `d`:

| k | valid d | number of regions |
|---|---|---|
| 1 | −24 … +25 | 50 |
| 3 | −23 … +24 | 48 |
| 5 | −22 … +23 | 46 |
| 7 | −21 … +22 | 44 |
| general odd k | (k−1)/2 − 24 … 25 − (k−1)/2 | 51 − k |

`dist = max(0, |d| − (k−1)/2)` is the number of qubits between qubit 25 and
the nearest qubit of R; it is 0 when R contains qubit 25.

**Even k, or any other qubit set:** there is no centre, so `(k, d)` does not
apply. Pass the qubits explicitly (§4b, `reduced_density_matrix`).

## 4. Producing region states

The region state is `ρ_R = Tr_{qubits not in R} |ψ⟩⟨ψ|`, a `2^k × 2^k`
complex Hermitian matrix with unit trace. It is computed exactly from the
MPS. Nothing is sampled and nothing is approximated beyond the stored state.

### 4a. Command line: `make_regions.py` (recommended)

```bash
python make_regions.py --hamiltonian {XX|XXX} --k K --out FILE.h5 \
    [--d SPEC] [--t SPEC] [--signs +1,-1] [--workers W] [--big-endian]
```

| option | values | default |
|---|---|---|
| `--d` | `all`, a range `-5:5` (inclusive), a stepped range `-10:10:2`, a list `0,2,4`, or a mix `-3:3,10` | `all` |
| `--t` | `all`, a range `0:4` (every stored step in it), a stepped range `0:13:0.5`, or a list `1.0,2.5` | `all` |
| `--signs` | `+1,-1`, `+1`, `-1` | both |
| `--workers` | parallel processes, one state each | 1 |
| `--threads` | BLAS threads per process | cores / workers |
| `--big-endian` | kron qubit order instead of little-endian (§5) | off |

Examples:

```bash
python make_regions.py --hamiltonian XX  --k 1 --out regions/XX_k1.h5                 # every d, every t
python make_regions.py --hamiltonian XXX --k 5 --d 0 --t 0:13:0.5 --out regions/XXX_k5_d0.h5
python make_regions.py --hamiltonian XXX --k 7 --d -6:6 --t 0:6 --workers 4 --out regions/XXX_k7_core.h5
```

Every requested `t` must be a stored time (a multiple of 0.1 from 0 to 13),
and every `d` must fit (§3); otherwise the tool stops with a message. The
output is written to `FILE.h5.part` and renamed when complete.

**Output file**, one sample per (state, d), with `M = (#t) × (#d) × (#signs)`:

| dataset | shape, dtype | meaning |
|---|---|---|
| `rho` | (M, 2^k, 2^k) complex128 | the region state (§5 for the index convention) |
| `s` | (M,) int8 | label: +1 or −1 |
| `t` | (M,) float64 | evolution time |
| `step` | (M,) int16 | Trotter step, `t = 0.1 × step` |
| `d` | (M,) int16 | region offset |
| `first_site` | (M,) int16 | first qubit of R (1-based); qubit q of R is `first_site + q` |
| `dist` | (M,) int16 | qubits between qubit 25 and R |
| `pair_id` | (M,) int32 | the (t, d) cell; the two labels of a cell share it |
| `in_strict_cone` | (M,) bool | `dist ≤ v t`, v = 2 (XX) or π (XXX), see §6 |
| `trace_distance` | (M,) float64 | ½‖ρ₊ − ρ₋‖₁ of the cell (when both signs are written) |
| `helstrom` | (M,) float64 | ½(1 + trace_distance): best possible single-shot accuracy for the cell |

Attributes record `hamiltonian`, `k`, `j0`, `d_list`, `t_list`, `signs`,
`basis` and `region` (the conventions in words), and provenance.

**Sample order:** by t, then d, then `s = +1` before `s = −1`. Samples `2c`
and `2c+1` are the two labels of cell `c`.

### 4b. Python: `source_sign_mps_reader.py`

```python
import source_sign_mps_reader as smr

psi = smr.load("XXX/XXX_s+1_t06.400.h5")           # one stored state
psi.hamiltonian, psi.s, psi.t                       # ('XXX', 1, 6.4)

smr.region_sites(5, -4)                             # [19, 20, 21, 22, 23]
rho = smr.region_rho(psi, k=5, d=-4)                # (32, 32) complex128

for d, sites, rho in smr.all_regions(psi, k=3, ds=range(-5, 6)):   # many d, one sweep
    ...

rho = smr.reduced_density_matrix(psi, [20, 25, 30])  # any qubit set (not contiguous: span ≲ 10)

ts = smr.training_set(".", "XX", k=3, ds=range(-5, 6), times=[0.0, 0.5, 1.0])
ts["rho"], ts["s"], ts["t"], ts["d"]                 # same ordering as make_regions.py
```

`all_regions` shares the left and right environments across regions, so
cutting many `d` from one state is much cheaper than calling `region_rho`
once per d (§8).

### 4c. Doing the contraction yourself

Use this if you write your own code, for example in another language. Each
file stores 50 tensors `A_j = mps/site_jj` (`site_01` … `site_50`), each
complex128 with numpy shape `(χ_{j−1}, 2, χ_j)`, where `χ_0 = χ_50 = 1`. The
middle index is the qubit's state, **0 = spin up (Z = +1), 1 = spin down**.
The amplitude of a basis state is:

```
ψ(b_1, …, b_50) = A_1[:, b_1, :] · A_2[:, b_2, :] · … · A_50[:, b_50, :]
```

For R = {a, …, b}:

```
L    = Σ over qubits 1..a−1 of the transfer matrices   (χ_{a−1} × χ_{a−1}; L = [[1]] if a = 1)
       L ← Σ_{x,y,s} L[x, y] A_j[x, s, :] ⊗ conj(A_j[y, s, :])
Rt   = the same from the right over b+1..50             (= identity: files are right-canonical)
T    = A_a A_{a+1} … A_b, contracted over the bonds, shape (χ_{a−1}, 2, …, 2, χ_b)
ρ[i, i'] = Σ L[l, l'] T[l, i, r] conj(T[l', i', r']) Rt[r, r']
```

Then divide by the trace (it is 1 to ~1e-15 anyway). In
`T[l, i_a, …, i_b, r]` the first qubit index belongs to qubit a. Reshaping
in C order therefore makes **qubit a the most significant bit**: that is the
big-endian (kron) order. Reverse the bits to get the little-endian convention
of §5. Check your result against the reference values in
`make_regions.py` (`REFERENCE`), or against `region_rho`.

**From Julia:** HDF5.jl reads arrays column-major, so each tensor appears as
`(χ_j, 2, χ_{j−1})`. Use `permutedims(A, (3, 2, 1))`. Index 1 (Julia) = up.

## 5. Conventions that fail silently

1. **Little-endian qubit order (the default).** Qubit `q` of the region,
   q = 0 … k−1, is qubit `first_site + q` of the chain and **bit q** of the
   matrix index: `r = Σ_q b_q 2^q`, with `b = 0` for up. `rho[r, r'] =
   ⟨r|ρ|r'⟩` (row = ket). Consequence: an operator built with `np.kron`
   must list its factors **reversed**, `kron(A_{k−1}, …, A_1, A_0)`.
   `--big-endian` (or `little_endian=False`) gives the kron order instead
   (`first_site` most significant). Mixing the two silently reverses the
   region.
2. **Keep ρ complex, and never transpose it.** At t = 0 the two labels'
   region states are exact complex conjugates (`ρ₋ = ρ₊*`). Dropping the
   imaginary part makes them identical, and `ρᵀ = ρ*` swaps the labels. Both
   errors pass every check of Hermiticity, trace and positivity.
3. **The labels are each other's spin flip.** In every cell,
   `ρ₋ = X^{⊗k} ρ₊ X^{⊗k}` (the index permutation `r → r XOR (2^k−1)`),
   verified to ≤ 1e-10 by `--selftest`. So any statistic that is symmetric
   under flipping every qubit of the region is identical for the two
   classes.
4. **Times are exact multiples of 0.1.** Compare times with a tolerance
   (`abs(t − 6.4) < 1e-6`), not `==` on computed floats.

## 6. Which (t, d) cells carry signal

The rotation's effect spreads from qubit 25 at a finite speed: `v = 2` qubits
per unit time for XX, and `v = π ≈ 3.14` for XXX. A region with
`dist > v·t` has not yet been reached and has almost no information about `s`.
- The `trace_distance` column measures this per cell: it runs from 0 (the
  labels are indistinguishable from R) to 1 (perfectly distinguishable).
- Measured on these states: every cell inside the strict cone (`dist ≤ v t`) has
  trace distance ≥ 0.09 (XX, k = 7, all cells) and ≥ 0.02 (XXX, k = 3). The
  front is not sharp: one qubit beyond it the trace distance can still reach
  0.35–0.45. Four qubits beyond it, it is at most 0.006 (XX, k = 7) and 0.013
  (XXX, k = 3). Selecting `dist ≤ v t + 4` keeps essentially every
  informative cell; `in_strict_cone` marks the strict cone.
- **Reflections:** the front reaches the chain end (qubit 1, 24 qubits away)
  at t ≈ 12 for XX and t ≈ 7.6 for XXX. Later states contain reflections from
  the ends, which is physical, but it is a different regime.

Reference trace distances (the exact numbers `--selftest` checks):

| | k = 3, d = 0, t = 0 | k = 3, d = +1, t = 1 | k = 5, d = −4, t = 3 | k = 7, d = +10, t = 6 | k = 3, d = +20, t = 2 |
|---|---|---|---|---|---|
| XX | 0.9009 | 0.7290 | 0.4322 | 0.3174 | 2e-10 |
| XXX | 0.8402 | 0.5855 | 0.3042 | 0.1770 | 7e-9 |

## 7. Notes for training a quantum neuron on these states

These findings come from the reduced-state analysis of the same physics
(N = 12 exactly, confirmed at N = 50).
- **Keep each cell's two labels on the same side of a train/test split**
  (`pair_id`). Otherwise the test set contains the exact spin flip of a
  training sample.
- **A pool of real operators is blind at t = 0.** Because `ρ₋ = ρ₊*` at
  t = 0, `Tr[ρ₊ A] = Tr[ρ₋ A]` for every real symmetric `A`. A neuron whose
  operators are all real (Z, ZZ, XX, YY, …) cannot separate the t = 0 pairs,
  even though their trace distance is ~0.9 when R contains qubit 25. The
  imaginary operator that carries that signal is the **spin current**
  `X_p Y_q − Y_p X_q`.
- **Only operators that conserve total Z can see anything.** The dynamics
  conserves total magnetisation, so every ρ_R is block-diagonal in R's
  magnetisation, and e.g. a lone `X_p` has expectation exactly 0. The 1- and
  2-body operators that conserve it are `Z_p`, `Z_p Z_q`, `X_p X_q + Y_p Y_q`
  and `X_p Y_q − Y_p X_q`.
- **The sign of the signal is not uniform across cells.** For XXX the leading
  signal `⟨Z_{25+d}⟩₊ − ⟨Z_{25+d}⟩₋` alternates in sign with d, as
  `(−1)^{d+1}`. For XX it flips in both d and t. A single neuron is linear in
  ρ, so training one model on many d (or many t) at once can fail even when
  every individual cell is distinguishable. Compare "all cells pooled"
  against "one d, all t" and "one t, all d".

## 8. Cost

Cost grows with the bond dimension χ of the state (XX ≤ 172; XXX grows from
134 to 448 by t ≈ 11) and as `2^k` with region size. Measured on an 8-core
machine:

| job | samples | wall time (4 workers) | output file |
|---|---|---|---|
| XX, k = 7, every d, every t | 11,528 | 9 min | 0.61 GB |
| XXX, k = 3, every d, every t | 12,576 | 7 min | 4 MB |
| XXX, k = 7, every d, t = 12.0 … 13.0 (χ = 448) | 968 | 16 min | 52 MB |
| XXX, k = 7, every d, every t | 11,528 | ~1.5–2 h (estimated from the row above, cost ∝ χ³) | ~0.6 GB |
| one region of the largest state, no workers | 1 | 0.4 s (k = 1) … 2.7 s (k = 7) | – |

The tool shares the environment work across the d of one state. For the
largest state, all 48 k = 3 regions take 3.1 s against 0.5 s for one, and
all 44 k = 7 regions take 41 s against 2.7 s for one (one process, 8
threads).

- Memory per process: up to 1.6 GB for one k = 7 region of a late XXX state,
  well below that otherwise. With `--workers W`, budget W × 1.6 GB for k = 7.
- Output size: a k = 7 sample is 256 KB uncompressed. The `rho` dataset is
  gzip-compressed (a k = 7 XX file stores 53 KB per sample, about 5× smaller).

## 9. Verifying, and where the numbers come from

- `python make_regions.py --selftest` checks the following, in ~30 s:
  - four files against values ITensor stored in them when it wrote them;
  - twelve region states against reference numbers that pin down the qubit
    order, the label assignment and the complex conjugation;
  - Hermiticity, trace, positivity and the spin-flip relation between labels.
- The XX reference values agree with the exact free-fermion solution to
  ~1e-6.
- `python source_sign_mps_reader.py .` re-checks every one of the 524 files
  and rewrites `manifest.csv` (slower: minutes).
- Accuracy of the states themselves (`DATASET.md` §7): every region up to
  k = 7, at every time, is within trace distance 2.7e-4 of the exact answer
  (XX) and 3.4e-4 of an independent simulation (XXX).
