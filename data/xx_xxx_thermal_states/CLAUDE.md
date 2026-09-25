# CLAUDE.md — XX vs XXX thermal-state dataset

Guidance for an agent working with the data in this folder. Everything below was
verified against the shipped files, not inferred from the generator.

## What the data is

Thermal states `rho(beta) = e^{-beta H} / Z` of random spin-1/2 chains at `n = 10`
sites, stored as **matrix product operators** and certified against dense exact
diagonalisation. Two Hamiltonians, open boundary, no field:

```
H_XX  = sum_i J_i ( X_i X_{i+1} + Y_i Y_{i+1} )              label 0
H_XXX = sum_i J_i ( X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1} ) label 1
```

Pauli convention, not spin: a two-site XXX chain gives a triplet at `+1` and a
singlet at `-3`. Nine couplings per chain, `J_i ~ U[0.5, 1.5]` i.i.d., **drawn
from the same law for both classes**, so the couplings carry no label and no
model can win by reading them off.

**The task:** binary classification of `label` from the stored tensors.

Each state was made by imaginary-time TEBD on a purification, then the ancillas
were traced out into an MPO, canonicalised and sign-fixed. Every sample carries
`td_mps`, its trace distance against an independently computed dense-ED
reference: **worst 3.78e-07 across all 840 samples.** That is the number that
means "the MPS converged", and it is the only one. Do not treat the per-level
`trace_distance` attribute as a convergence certificate; see rule 4.

## Files

| file | what |
|---|---|
| `xx_xxx_n10.h5` | 2.9 GB — the dataset, 840 samples |
| `fixture_n4.h5` | 675 KB — 24 samples at `n = 4`, same schema. **Debug your reader here first.** Its MPO levels are `chi4`/`chi8` and it has 3 `kT` rungs, not 10 |
| `load_spin_data.py` | standalone reader, `h5py` + `numpy` only. Handles the axis order and the group keys correctly |
| `classify_n10.json` | baselines already measured — read before spending compute |
| `HANDOFF.md` | the same material written for a human, with the generation settings |
| `SHA256SUMS` | `sha256sum -c SHA256SUMS` after transfer |

Start by running the reader's self-check, which asserts every structural claim
in this document:

```bash
python load_spin_data.py fixture_n4.h5
python load_spin_data.py xx_xxx_n10.h5      # ~4 s, ~1.2 GB resident
```

## Shape of the dataset

840 samples = **84 coupling draws x 10 temperature rungs**, balanced 420 XX /
420 XXX. One draw contributes 10 samples: the same Hamiltonian at 10
temperatures.

```
kT   = 2.0, 1.41, 1.0, 0.71, 0.5, 0.35, 0.25, 0.18, 0.13, 0.1        (beta = 1/kT)
```

The generation run asked for 48 draws per class in 8 shards; shard 1 (draws
7-12) crashed and was not rerun, so draw ids 7-12 are absent. The loss is
symmetric across the classes, so this is 84 clean draws, not a skewed 96.

## HDF5 layout

```
/meta                              attributes only
    n = 10                         sites
    kT = [2.0 ... 0.1]             the 10 rungs
    chi_mpo_levels = [8,16,32,64]  the levels present
    n_samples = 840, n_draws = 84
    label_names = ["XX", "XXX"]    index 0 and 1 of `label`
    cutoff = 1e-16, dbeta = 0.02, dbeta_max = 0.04, order = 4
    jlo = 0.5, jhi = 1.5, seed = 20260916, chi_mps_cap = 1024, created
    complete = True                <- if absent or false the file is torn; stop

/samples/sample_00000 ... sample_00839
    @label      0 = XX, 1 = XXX         <-- THE TARGET
    @model      "XX" | "XXX"            the same label as a string
    @draw       which coupling draw     per class, see rule 2
    @kT, @beta  the temperature
    @n          10
    @chi_mps    bond dim the purification reached (28-667; the 1024 cap never bound)
    @td_mps     THE CERTIFICATE, vs dense ED
    @energy_mps, @energy_ed             agreement check, ~1e-7
    @logZ_mps,  @logZ_ed                agreement check
    J           (9,)   the couplings of this chain

    mpo/chi8  mpo/chi16  mpo/chi32  mpo/chi64
        @chimax             8 | 16 | 32 | 64
        @trace_distance     THIS level vs ED — ablation metadata, NOT a certificate
        @trace_mpo          trace before renormalisation (the truncation loss)
        @degenerate_bonds   MISNAMED: counts singular VALUES in a numerically
                            degenerate multiplet (~1e-9 rel.), summed over bonds
        tensors        (C, 2, 2, C, 10) as h5py sees it   <-- REVERSED, rule 1
        spectra        (C, 9)     bond singular values, zero-padded
        chi            (11,)      true bond dims, n+1 entries, edges are 1
        tensors_gauge  same state, degenerate Schmidt bases re-randomised
        spectra_gauge  same
```

The four `chi_mpo` levels are **the same state at four feature resolutions**, not
four different states. Use `chi64` unless you are deliberately running the
resolution ablation.

After the axis reversal that `load_spin_data.py` applies:

| array | shape at `chi64` | meaning |
|---|---|---|
| `tensors` | `(840, 10, 64, 2, 2, 64)` | `[m, site, left bond, bra, ket, right bond]` |
| `spectra` | `(840, 9, 64)` | `[m, bond, singular values]`, padded to `chimax` |
| `chi` | `(840, 11)` | true bond dims; `chi[m, j]` nonzeros in `spectra[m, j-1]` |
| `J` | `(840, 9)` | couplings |

Flattened feature sets, matching the ones the baselines used:
`tensors` 163840 dims, `spectra` 576, `scalars` 16 (energy, logZ, kT, beta,
summed bond purity, the 9 bond entropies, their mean and max).

## Five ways to get this wrong

**1. The axis order is reversed.** HDF5.jl writes a Julia array with its
dimensions flipped, so the `(n, chi, 2, 2, chi)` array the generator wrote
arrives in h5py as `(chi, 2, 2, chi, n)`. A full axis reversal (`arr.T`) undoes
it exactly. **This does not raise** — the array is finite, the right total size,
and the bond index is silently transposed against the site index. `chi` is the
check: the nonzero bond support of site `s` must equal `chi[s]` and `chi[s+1]`.
`load_spin_data.py` asserts this.

**2. Group the splits by `(model, draw)`, never by `draw` alone, and never per
sample.** Draw ids are per class: `draw = 3` exists once as XX and once as XXX
and they are different chains. And one draw's 10 temperature rungs are the same
Hamiltonian, so a random per-sample split puts `kT = 0.5` of a chain in train
and `kT = 0.35` of that same chain in test — the score then measures
interpolation in temperature, not generalisation to unseen couplings. Use
`f"{model}:{draw}"`, which the reader builds as `ds.groups`, with
**`StratifiedGroupKFold`**, not plain `GroupKFold`. Every group is entirely one
class, so an unstratified split hands a fold 70/30 class proportions by
accident, and accuracy against a shifting base rate is not comparable across
folds. This is not cosmetic: on `spectra`, plain `GroupKFold` with a fixed `C`
scores 0.96 where the stored protocol scores 1.000.

**3. Tensors are zero-padded to `chimax`.** `chi` holds the real bond
dimensions. At `chi64` roughly 55% of the entries are structural zeros. Fine as
features; wrong if you contract the MPO without slicing to `chi` first.

**4. The low `chi_mpo` levels are not density matrices.** MPO truncation does
not preserve positivity. Reconstructed densely, the worst `chi8` sample has a
minimum eigenvalue of `-0.27` and `||rho||_1 = 8.0`, which is why its
`trace_distance` attribute reads 3.58 — a value impossible between two real
states. At `chi64` it is nearly clean (min eigenvalue `-1.8e-3`,
`||rho||_1 = 1.02`). Every level is renormalised to unit **trace**, so this is
invisible unless you look. Anything downstream that assumes a valid state — an
entropy, a purity, `Tr[O rho]` — must use `chi64` and check the spectrum.

**5. `tensors_gauge` is a control, not extra data.** It is the identical
physical state with an arbitrary basis chosen inside each degenerate Schmidt
subspace. Measured at `chi64`: **28% of the nonzero singular values sit in a
degenerate multiplet** at ~1e-9 relative tolerance, spread over ~70% of bonds
(at a strict 1e-12 it is 4%, so the degeneracy is near, not bit-exact). That is
how much of a tensor feature vector is ill-defined — large, but not most of it.
Train on `tensors`, test on `tensors_gauge`: a
model reading the physics is unaffected, one reading the basis collapses. Never
pool the two as augmentation without saying so — they are the same states.

## The baseline is 1.000 AUC, not 50%

From `classify_n10.json`, at `chi64`. Protocol: 5-fold outer
`StratifiedGroupKFold` on `(model, draw)`, with the regularisation strength
chosen per outer fold by an inner 3-fold `StratifiedGroupKFold` `GridSearchCV`
(`C` over `1e-4 .. 1.0`, MLP `alpha` over `0.1, 1, 10`, hidden layers `(64, 16)`,
features standardised). With `d >> M` on the tensor features the regularisation
strength is not a detail, it is the model, so it must not be tuned on the outer
folds or fixed by hand:

| features | dim | logistic AUC | MLP AUC | cross-gauge AUC |
|---|---|---|---|---|
| `scalars` | 16 | 0.997 | 0.993 | — |
| `spectra` | 576 | **1.000** | **1.000** | — |
| `tensors` | 163840 | 0.927 | 0.967 | 0.918 / 0.959 |

XXX carries an extra `ZZ` on every bond, so its energy scale and entanglement
structure differ from XX by construction, and the gauge-invariant singular
values already separate the classes perfectly at `n = 10`. **A tensor model
scoring 0.97 has not beaten the task, it has lost to a 576-dimensional one.**

Two consequences for any report:

- Quote `spectra` as the baseline. A tensor model is only interesting if it
  reaches 1.000 as well, and then the claim is about parameter efficiency or
  transfer, not accuracy.
- The interesting questions here are not "can it classify". They are: does a
  tensor model generalise across gauge (cross-gauge AUC vs same-gauge), across
  temperature (hold out whole `kT` rungs), and to `n != 10` — which this dataset
  cannot answer, because it has only `n = 10`.

`classify_n10.json` carries per-`kT` accuracy and the per-fold hyperparameters
for all 24 (level, feature set, model) combinations, so those need no rerun.

These reproduce exactly from this folder — verified, including the per-fold `C`:

```python
from load_spin_data import load_dataset, design_matrix
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ds = load_dataset("xx_xxx_n10.h5")
X = design_matrix(ds, "spectra")
p = np.zeros(len(ds))
for tr, te in StratifiedGroupKFold(5).split(X, ds.y, ds.groups):
    gs = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        {"logisticregression__C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
        cv=StratifiedGroupKFold(3), scoring="roc_auc", n_jobs=-1)
    gs.fit(X[tr], ds.y[tr], groups=ds.groups[tr])
    p[te] = gs.predict_proba(X[te])[:, 1]
roc_auc_score(ds.y, p)        # -> 1.0000
```

## Not included

- **Dense `rho`.** 8 MB per sample at `n = 10`. Regenerable from the stored `J`
  by direct diagonalisation in under a second per sample: build
  `H = sum_i J_i (XX + YY [+ ZZ])` by `kron`, one `eigh`, then
  `rho = V diag(softmax(-beta E)) V^T`.
- **The purification tensors.** Withheld deliberately: any unitary on the
  ancillas leaves `rho` invariant while changing every tensor, so purification
  tensors encode the TEBD trajectory, and a classifier fed them can score well
  by learning the path rather than the state. Tracing the ancillas out removes
  that freedom.
- `n != 10`, other coupling laws, fields, periodic boundaries.

Regenerating any of that means going back to the source repo; it cannot be done
from this folder.
