# XX vs XXX thermal states — data handoff

Thermal states `rho(beta) = e^{-beta H}/Z` of two random spin-1/2 chains at
`n = 10` sites, each produced as a **matrix product operator** and certified
against dense exact diagonalisation. The intended task is a binary classifier
that reads the tensors and recovers which Hamiltonian produced them.

```
H_XX  = sum_i J_i ( X_i X_{i+1} + Y_i Y_{i+1} )              label 0
H_XXX = sum_i J_i ( X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1} ) label 1
```

Open boundary, no field, 9 couplings per chain, `J_i ~ U[0.5, 1.5]` i.i.d.
**drawn from the same law for both classes** — the couplings carry no label, so
a classifier cannot win by reading them off.

Pauli convention, not spin: the two-site XXX chain gives a triplet at `+1` and
a singlet at `-3` (spin operators would give a quarter of that).

## What you are receiving

| file | size | what |
|---|---|---|
| `xx_xxx_n10.h5` | 2.9 GB | **the dataset** — 840 samples |
| `fixture_n4.h5` | 0.7 MB | 24 samples at `n = 4`, same schema — use it to debug your reader. Note its MPO levels are `chi4`/`chi8` (a 4-site chain cannot reach 64) and it has 3 `kT` rungs, not 10 |
| `classify_n10.json` | 26 KB | results already obtained, so you do not re-derive them |
| `HANDOFF.md` | this file | |
| `SHA256SUMS` | | verify after transfer: `sha256sum -c SHA256SUMS` |

840 samples = **84 coupling draws x 10 temperature rungs**, balanced 42 XX / 42
XXX. One draw contributes 10 samples — the same Hamiltonian at 10 temperatures.

`kT = 2.0, 1.41, 1.0, 0.71, 0.5, 0.35, 0.25, 0.18, 0.13, 0.1` (`beta = 1/kT`).

> The generation run asked for 48 draws per class in 8 shards; shard 1 (draws
> 7–12) crashed in Julia and was not rerun, so draw ids 7–12 are absent. The
> loss is symmetric across the two classes, so the dataset stays balanced —
> it is 84 draws rather than 96, not a skewed 96.

## How it was made

Imaginary-time TEBD on a purification (ancilla per site, interleaved), then the
ancillas are traced out into an MPO, canonicalised and sign-fixed.

| setting | value |
|---|---|
| truncation `cutoff` | `1e-16` — **this**, not `chi`, is the accuracy knob |
| `chi_mps` cap | 1024 (never binds; realised 28–667, see below) |
| Trotter | 4th order, `dbeta = 0.02`, `dbeta_max = 0.04` |
| seed | 20260916 |

Every sample carries its own dense-ED reference, computed independently.

## The certificate

`td_mps` (a per-sample attribute) is the trace distance of the **untruncated**
MPS state against dense ED. Across all 840 samples:

```
worst 3.78e-07      best 2.46e-08
```

That is the number that means "the MPS converged". It is the only one.

The bond dimension the state actually needed, and the certificate, by rung:

| kT | chi_mps min / med / max | worst td_mps |
|---|---|---|
| 2.00 | 28 / 43 / 77 | 2.9e-07 |
| 1.00 | 48 / 86 / 176 | 3.4e-07 |
| 0.50 | 84 / 170 / 372 | 3.6e-07 |
| 0.25 | 159 / 284 / 583 | 3.8e-07 |
| 0.10 | 218 / 369 / 667 | 3.1e-07 |

## HDF5 layout

```
/meta                              (attributes only)
    n = 10                         sites
    kT = [2.0 ... 0.1]             the 10 rungs
    chi_mpo_levels = [8,16,32,64]  the ablation levels present
    n_samples = 840,  n_draws = 84
    label_names = ["XX", "XXX"]    index 0 and 1 of `label`
    cutoff, dbeta, dbeta_max, order, jlo, jhi, seed, chi_mps_cap, created
    complete = True                <- if this is missing the file is torn

/samples/sample_00000 ... sample_00839
    @label      0 = XX, 1 = XXX          <-- THE LABEL
    @model      "XX" | "XXX"             <-- the same label as a string
    @draw       which coupling draw
    @kT, @beta  the temperature
    @n          10
    @chi_mps    bond dimension the purification reached
    @td_mps     THE CERTIFICATE (vs dense ED)
    @energy_mps, @energy_ed              agreement check, ~1e-7
    @logZ_mps,  @logZ_ed                 agreement check
    J           (9,)  the couplings of this chain

    mpo/chi8  mpo/chi16  mpo/chi32  mpo/chi64
        @chimax             8 | 16 | 32 | 64
        @trace_distance     THIS level vs ED — ablation metadata, NOT a certificate
        @trace_mpo          trace before renormalisation (the truncation loss)
        @degenerate_bonds   MISNAMED: counts singular VALUES in a numerically
                            degenerate multiplet (~1e-9 rel.), summed over bonds
        tensors        (C, 2, 2, C, 10) as h5py sees it  <-- REVERSED, see below
        spectra        (C, 9)           bond singular values
        chi            (11,)            true bond dims before zero-padding
        tensors_gauge  same state, degenerate Schmidt bases re-randomised
        spectra_gauge  same
```

The **four `chi_mpo` levels are the same state at four feature resolutions**,
not four different states. Pick one; `chi64` is the most faithful.

`tensors_gauge` / `spectra_gauge` are the **gauge control**: the identical
physical state with an arbitrary basis chosen inside each degenerate Schmidt
subspace. Train on one, test on the other — a classifier reading the physics is
unaffected; one reading the basis collapses. At `chi64`, 28% of the nonzero
singular values sit in a degenerate multiplet (~1e-9 relative tolerance) across
roughly 70% of bonds, so this control is not optional.

## Four ways to read this file wrong

**1. The axis order is reversed.** HDF5.jl writes a Julia array with its
dimensions flipped, so the `(n, chi, 2, 2, chi)` array the generator wrote
arrives in h5py as `(chi, 2, 2, chi, n)`. A full axis reversal (`arr.T`) undoes
it exactly. **This does not raise** — the array is finite, the right total size,
and the bond index is silently transposed against the site index. Check your
reader against `fixture_n4.h5` before touching the big file.

**2. Group your splits by `(model, draw)`, never by `draw` — and stratify.**
Use `StratifiedGroupKFold`: every group is entirely one class, so an
unstratified split hands a fold 70/30 class proportions by accident. Draw ids are
per-class: `draw = 3` exists once as XX and once as XXX, and they are different
chains. Also, one draw's 10 temperature rungs are the same Hamiltonian, so a
random per-sample split puts `kT = 0.5` of a chain in train and `kT = 0.35` of
that same chain in test — the score then measures interpolation in temperature,
not generalisation to unseen couplings. Use `f"{model}:{draw}"` as the group.

**3. Tensors are zero-padded to `chimax`.** The `chi` dataset holds the real
bond dimensions (11 entries = `n+1`, including the trivial edge bonds of 1).
At `chi64` roughly 55% of the entries are structural zeros. Fine as features;
not fine if you contract without slicing.

**4. The low `chi_mpo` levels are NOT density matrices.** MPO truncation is not
positivity preserving. Reconstructed densely, the worst `chi8` sample has a
minimum eigenvalue of `-0.27` and `||rho||_1 = 8.0` — which is why its
"trace distance" reads 3.58, a value impossible between two real states. At
`chi64` it is nearly clean (min eigenvalue `-1.8e-3`, `||rho||_1 = 1.02`).
Every level is renormalised to unit **trace**, so this is invisible unless you
look for it. If anything downstream assumes a valid state — an entropy, a
purity, `Tr[phi(B) rho]` — use `chi64`, and check the spectrum.

## Minimal reader

Self-contained; no dependency on our packages.

```python
import h5py, numpy as np

def load(path, level=None, gauge=False):     # level=None -> the largest present
    X, y, groups, kT = [], [], [], []
    with h5py.File(path, "r") as f:
        assert f["meta"].attrs.get("complete", False), "torn file, regenerate"
        if level is None:                            # chi64 here, chi8 in the fixture
            level = "chi%d" % max(f["meta"].attrs["chi_mpo_levels"])
        for name in sorted(f["samples"]):
            g = f["samples"][name]
            t = np.asarray(g["mpo"][level]["tensors_gauge" if gauge else "tensors"]).T
            X.append(t.reshape(-1))                  # (n, C, 2, 2, C) flattened
            y.append(int(g.attrs["label"]))          # 0 = XX, 1 = XXX
            model = g.attrs["model"]
            model = model.decode() if isinstance(model, bytes) else model
            groups.append(f"{model}:{int(g.attrs['draw'])}")
            kT.append(float(g.attrs["kT"]))
    return np.stack(X), np.array(y), np.array(groups), np.array(kT)

X, y, groups, kT = load("xx_xxx_n10.h5")
# -> X (840, 163840), y (840,), 84 distinct groups
# then: StratifiedGroupKFold(5).split(X, y, groups)   <- stratified, see below
```

Sanity checks that should pass: `X.shape == (840, 163840)`, `y.sum() == 420`,
`len(set(groups)) == 84`, and for the fixture, the ED spectrum you rebuild from
the stored `J` must match the state to the quoted `td_mps`.

## What is already known — read before spending compute

From `classify_n10.json`: 5-fold `StratifiedGroupKFold` on `(model, draw)`, the
regularisation strength chosen per outer fold by an inner 3-fold grouped
`GridSearchCV`. Plain `GroupKFold` with a fixed `C` scores 0.96 on `spectra`
where this protocol scores 1.000, so the split matters as much as the model:

| features | dim | logistic AUC | MLP AUC |
|---|---|---|---|
| `scalars` (energy, logZ, kT, bond entropies) | 16 | 0.997 | 0.993 |
| `spectra` (bond singular values, `chi64`) | 576 | **1.000** | **1.000** |
| `tensors` (`chi64`) | 163840 | 0.927 | 0.967 |

**The baseline is not 50%, and it is not `scalars` either — it is `spectra` at
AUC 1.000.** XXX carries an extra `ZZ` on every bond, so its energy scale and
entanglement structure differ from XX by construction; the gauge-invariant
singular values already separate the classes perfectly at `n = 10`. A tensor
model scoring 0.97 has not beaten the task, it has lost to a 576-dimensional
one. If tensors are to be worth their `chi^2` dimensions, that is the number to
beat, and the honest report says so.

The cross-gauge scores (train on `tensors`, test on `tensors_gauge`) sit at
0.918–0.982, close to the same-gauge scores, so the tensor models are mostly
not reading the arbitrary basis — but "mostly" is why the control ships.

## Not included

- **Dense `rho`.** 8 MB per sample at `n = 10`; regenerable from the stored `J`
  by direct diagonalisation in under a second per sample.
- **The purification tensors.** Deliberately: any unitary on the ancillas leaves
  `rho` invariant while changing every tensor, so purification tensors encode
  the evolution path, and a classifier fed them can score well by learning the
  trajectory rather than the state. Tracing the ancillas out removes that
  freedom; what you have is the result.
- `n != 10`, other coupling laws, fields, periodic boundaries.
