"""Reader for the XX vs XXX thermal-state dataset.

Standalone: needs only ``h5py`` and ``numpy``. Nothing from the package that
produced the file is required, or available.

    from load_spin_data import load_dataset, design_matrix

    ds = load_dataset("xx_xxx_n10.h5")            # chi64, the faithful level
    X  = design_matrix(ds, "spectra")             # (840, 576)
    # then: sklearn.model_selection.GroupKFold(5).split(X, ds.y, ds.groups)

Run it directly to verify a file against every invariant in CLAUDE.md:

    python load_spin_data.py fixture_n4.h5
    python load_spin_data.py xx_xxx_n10.h5
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

FEATURE_SETS = ("tensors", "spectra", "scalars")


@dataclass
class SpinData:
    """One MPO level of the dataset, fully in memory.

    ``tensors`` is ``(M, n, C, 2, 2, C)`` and ``spectra`` is ``(M, n-1, C)`` --
    both already un-reversed from the Julia axis order on read.
    """

    tensors: np.ndarray      # (M, n, C, 2, 2, C)
    spectra: np.ndarray      # (M, n-1, C)   spectra[m, j] = bond j, zero-padded
    chi: np.ndarray          # (M, n+1)      true bond dims, edges included
    y: np.ndarray            # (M,)          0 = XX, 1 = XXX
    groups: np.ndarray       # (M,)          "XX:3" -- SPLIT ON THIS, not on draw
    kT: np.ndarray
    beta: np.ndarray
    energy: np.ndarray       # energy_mps
    logZ: np.ndarray         # logZ_mps
    td_mps: np.ndarray       # the convergence certificate, vs dense ED
    J: np.ndarray            # (M, n-1)      couplings of each chain
    level: str
    n: int

    def __len__(self) -> int:
        return len(self.y)


def _text(v) -> str:
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)


def available_levels(path: str) -> list[int]:
    """The chi_mpo levels present, ascending."""
    with h5py.File(path, "r") as f:
        return sorted(int(c) for c in np.asarray(f["meta"].attrs["chi_mpo_levels"]))


def load_dataset(path: str, level: int | str | None = None, gauge: bool = False,
                 dtype=np.float64) -> SpinData:
    """Load one MPO level into memory.

    ``level=None`` takes the largest level present -- ``chi64`` for the n=10
    dataset, ``chi8`` for the n=4 fixture. Prefer the largest: the lower levels
    are the same state truncated, and truncation is not positivity preserving.

    ``gauge=True`` reads ``tensors_gauge``/``spectra_gauge`` instead: the
    identical physical state with an arbitrary basis inside each degenerate
    Schmidt subspace. Train on one, test on the other to check a model is
    reading the state and not the basis.

    ``dtype=np.float32`` halves the footprint; chi64 tensors are 1.1 GB in
    float64.
    """
    with h5py.File(path, "r") as f:
        meta = f["meta"].attrs
        if not bool(meta.get("complete", False)):
            raise ValueError(f"{path}: meta/complete is not set -- torn file, regenerate")
        if level is None:
            level = max(int(c) for c in np.asarray(meta["chi_mpo_levels"]))
        key = level if isinstance(level, str) else f"chi{int(level)}"
        n = int(meta["n"])

        tkey = "tensors_gauge" if gauge else "tensors"
        skey = "spectra_gauge" if gauge else "spectra"
        cols: dict[str, list] = {k: [] for k in
                                 ("t", "s", "chi", "y", "grp", "kT", "beta",
                                  "E", "logZ", "td", "J")}

        for name in sorted(f["samples"]):
            g = f["samples"][name]
            a = g.attrs
            if key not in g["mpo"]:
                raise KeyError(f"{path}: no level {key!r}; present: {sorted(g['mpo'])}")
            lg = g["mpo"][key]
            # Julia writes (n, C, 2, 2, C); HDF5.jl reverses the dimensions, so
            # h5py sees (C, 2, 2, C, n). A full axis reversal undoes it exactly.
            # This does NOT raise if skipped -- it silently transposes the bond
            # index against the site index.
            cols["t"].append(np.ascontiguousarray(np.asarray(lg[tkey]).T, dtype=dtype))
            cols["s"].append(np.ascontiguousarray(np.asarray(lg[skey]).T, dtype=dtype))
            cols["chi"].append(np.asarray(lg["chi"]))
            cols["J"].append(np.asarray(g["J"]))
            cols["y"].append(int(a["label"]))
            # draw ids are per class: "XX:3" and "XXX:3" are different chains,
            # and one draw's kT rungs are all the same Hamiltonian.
            cols["grp"].append(f"{_text(a['model'])}:{int(a['draw'])}")
            cols["kT"].append(float(a["kT"]))
            cols["beta"].append(float(a["beta"]))
            cols["E"].append(float(a["energy_mps"]))
            cols["logZ"].append(float(a["logZ_mps"]))
            cols["td"].append(float(a["td_mps"]))

    return SpinData(
        tensors=np.stack(cols["t"]), spectra=np.stack(cols["s"]),
        chi=np.stack(cols["chi"]), y=np.array(cols["y"]),
        groups=np.array(cols["grp"]), kT=np.array(cols["kT"]),
        beta=np.array(cols["beta"]), energy=np.array(cols["E"]),
        logZ=np.array(cols["logZ"]), td_mps=np.array(cols["td"]),
        J=np.stack(cols["J"]), level=key, n=n,
    )


def bond_entropies(spectra: np.ndarray) -> np.ndarray:
    """Von Neumann entropy of each bond's normalised spectrum, ``(M, n-1)``.

    These are MPO singular values, so this is the operator-space entanglement of
    rho -- how hard rho is to compress -- not the state entanglement.
    """
    s2 = spectra ** 2
    total = s2.sum(axis=-1, keepdims=True)
    p = np.divide(s2, total, out=np.zeros_like(s2), where=total > 0)
    return -(p * np.log(p, out=np.zeros_like(p), where=p > 0)).sum(axis=-1)


def design_matrix(ds: SpinData, which: str) -> np.ndarray:
    """Flatten one feature set to ``(M, d)`` -- the three the baselines used."""
    if which == "tensors":
        return ds.tensors.reshape(len(ds), -1)          # n * C^2 * 4
    if which == "spectra":
        return ds.spectra.reshape(len(ds), -1)          # (n-1) * C
    if which == "scalars":
        S = bond_entropies(ds.spectra)
        return np.column_stack([
            ds.energy, ds.logZ, ds.kT, ds.beta,
            (ds.spectra ** 2).sum(axis=(1, 2)),
            S, S.mean(axis=1), S.max(axis=1),
        ])
    raise ValueError(f"which must be one of {FEATURE_SETS}, got {which!r}")


def self_check(path: str) -> None:
    """Assert every structural invariant the docs claim. Raises on the first miss."""
    levels = available_levels(path)
    ds = load_dataset(path)
    M, n, C = len(ds), ds.n, levels[-1]

    assert ds.tensors.shape == (M, n, C, 2, 2, C), ds.tensors.shape
    assert ds.spectra.shape == (M, n - 1, C), ds.spectra.shape
    assert ds.chi.shape == (M, n + 1), ds.chi.shape
    assert ds.J.shape == (M, n - 1), ds.J.shape
    assert set(np.unique(ds.y)) <= {0, 1}
    assert 2 * ds.y.sum() == M, f"not balanced: {ds.y.sum()} of {M}"
    assert (ds.chi[:, 0] == 1).all() and (ds.chi[:, -1] == 1).all(), "edge bonds not 1"
    assert (ds.chi <= C).all(), "a bond exceeds chimax"
    assert np.allclose(ds.kT, 1.0 / ds.beta), "kT and beta disagree"

    # the axis order is right iff the tensor's bond support matches `chi`
    sup_l = (np.abs(ds.tensors).sum(axis=(3, 4, 5)) > 0).sum(axis=2)
    sup_r = (np.abs(ds.tensors).sum(axis=(2, 3, 4)) > 0).sum(axis=2)
    assert (sup_l == ds.chi[:, :-1]).all(), "AXIS ORDER WRONG (left bond vs chi)"
    assert (sup_r == ds.chi[:, 1:]).all(), "AXIS ORDER WRONG (right bond vs chi)"

    # each spectra row is one bond, zero-padded to chimax
    nz = (np.abs(ds.spectra) > 0).sum(axis=2)
    assert (nz == ds.chi[:, 1:-1]).all(), "spectra rows are not bonds"

    per_group = {g: (ds.groups == g).sum() for g in np.unique(ds.groups)}
    assert len(set(per_group.values())) == 1, f"ragged groups: {set(per_group.values())}"

    print(f"{path}")
    print(f"  n = {n}   samples = {M}   groups = {len(per_group)} "
          f"x {next(iter(per_group.values()))} kT rungs   levels = {levels}")
    print(f"  label balance  {M - ds.y.sum()} XX / {ds.y.sum()} XXX")
    print(f"  kT rungs       {sorted(set(np.round(ds.kT, 4)), reverse=True)}")
    print(f"  td_mps         worst {ds.td_mps.max():.2e}   best {ds.td_mps.min():.2e}")
    print(f"  mpo bonds      min {ds.chi[:, 1:-1].min()}  max {ds.chi[:, 1:-1].max()} (capped at chimax)")
    for w in FEATURE_SETS:
        print(f"  {w:<8} -> {design_matrix(ds, w).shape}")
    print("  all structural invariants hold")


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:] or ["fixture_n4.h5"]:
        self_check(arg)
