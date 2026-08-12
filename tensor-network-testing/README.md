# tensor-network-testing/ — Julia trainers (Yao / ITensor)

Implementations of the source paper's **Algorithms 8, 9 and 10** — the
hybrid quantum-classical gradient estimators — in Julia, with both a
statevector backend (Yao.jl) and a tensor-network backend (ITensorMPS).

These are reference and pretraining implementations. They are **not** wired
into the Python pipeline; they consume `{ρ_m, y_m}` files exported by
[`../scripts/export_thermal_training.py`](../scripts/export_thermal_training.py),
or generate their own DMRG training states.

```bash
julia
julia> import Pkg; Pkg.add(["Yao", "ITensors", "ITensorMPS", "Optimisers"])
julia> include("train_alg9.jl")
```

---

## Which file to use

**Use Algorithm 9.** It differentiates the loss actually being minimized
(the logistic loss), so `train_alg9.jl` is honest gradient descent on `L^log`.
Algorithm 8 descends the *squared* loss — a different objective.

**Use the Yao backend for training.** ITensor/MPS is for validation and for
reaching larger *n* on a single gradient evaluation; it is far too slow to drive
an optimizer loop. Both expose the same
`setup` / `logloss_gradient` / `exact_logloss` interface, so `train_alg9.jl`
takes either via `backend=`.

| File | |
|---|---|
| **`algorithm9_yao.jl`** | ★ Algorithm 9 (Thm. 5 / Eq. C27), statevector. Plus Algorithm 10 / Eq. C44 for the loss *value* |
| **`train_alg9.jl`** | ★ Optimisers.jl loop driven by that gradient. θ is a plain `Vector{Float64}`, a valid Optimisers parameter tree — no glue needed |
| `algorithm9.jl` | Same algorithm on ITensorMPS, training states prepared by DMRG, every circuit carried as an MPS |
| `algorithm8_yao.jl` | Algorithm 8 (Eq. B4): squared-loss gradient from two independent Hadamard tests (Fig. 10), combined by Eq. B9 |
| `algorithm8.jl` | Algorithm 8 on ITensorMPS. **Deliberate deviation:** each block returns its final MPS rather than sampling measurement outcomes |
| `convergence_checks.jl` | The validation suite — see below |
| `MPS_construction.jl`, `dmrg_tutorial.jl` | ITensor tutorials, kept as reference |

## Model convention (shared by all four algorithm files)

1D spin-½ chain, **ancilla on site 1**, system on sites 2…N+1.
`J = 4N − 3` terms in this order: `Z_i` (i = 1…N), then `Z_iZ_{i+1}`,
`X_iX_{i+1}`, `Y_iY_{i+1}` (i = 1…N−1). Labels `y_m ∈ {−1, +1}`,
`g_T(H) = tanh(H/T)`.

At these sizes `L^(2)` and the logistic loss both have exact classical values,
which each module checks at the bottom of the file.

> Note this is **not** the same operator basis as the Python side's
> extended-Heisenberg set (Z + all-pair ZZ + within-spin-block XX/YY,
> `4·ncas² − ncas` terms). Nearest-neighbour here, all-pair there. Do not
> compare parameter vectors across the two without translating.

## `convergence_checks.jl`

```julia
julia> include("convergence_checks.jl")
julia> run_all(; n=4)             # correctness + convergence
julia> cost_vs_n(; ns=4:2:12)     # check 8 alone, slower
```

| # | Check | Expected |
|---:|---|---|
| 0 | DMRG training states vs exact diagonalization | agreement |
| 1 | Ancilla-free path: `:overlap` == `:hadamard` | ~machine ε (identity) |
| 2 | Algorithm 9 vs central differences on `L^log` | agreement |
| 3 | `Alg9ITensor` vs `Alg9Yao` on identical draws | agreement |
| 4 | Trotter step `dtmax` sweep | 2nd order: error ~ dtmax² |
| 5 | Bond dimension `maxdim` sweep | a plateau |
| 6 | Truncation `cutoff` sweep | a plateau |
| 7 | Monte Carlo `nsamples` sweep | error ~ 1/√nsamples |
| 8 | Cost vs *n* | bond dimension and wall time growth |

Checks 4–7 reuse **one fixed seed**, so the `(s, t, k)` draws are identical
across a sweep — that is what isolates discretization error from Monte Carlo
noise. Preserve that when adding sweeps.

---

## How this connects to the rest of the project

The export format is documented in
[`../docs/DATA_CATALOG.md`](../docs/DATA_CATALOG.md) §3. States arrive as
eigenblocks — weights plus sparse amplitudes over a fixed particle-number
sector — which feeds both Julia paths: sample `k ~ p_k` and run the pure-state
circuit on `|v_k⟩`, or build ρ explicitly.

Reconstruct a state vector in Julia:

```julia
v = zeros(2^Q)
v[basis_indices .+ 1] = amps[k, :]     # note the +1: HDF5 indices are 0-based
```

**Before training on real data**, read
[`../docs/QUANTUM_NEURON.md`](../docs/QUANTUM_NEURON.md) — the label in the
exported file is currently a placeholder, and training on it will produce a
result that a classical model matches.
