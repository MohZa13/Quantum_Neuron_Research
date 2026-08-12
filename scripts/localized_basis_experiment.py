"""Is the "these states are 93% diagonal" result an artifact of the orbital basis?

`basis_dependence_probe.py` showed on twisted ethylene that an orbital rotation
inside the active space -- which changes no energy and no observable -- can move
the singlet/triplet label from a purely diagonal readout to a purely coherent
one.  Every coherence number in this project was measured in one particular
basis (canonical, delocalized Kohn-Sham), so the question is whether the whole
negative result is a basis artifact.

This script settles it, per molecule, from the stored `h1eff` and `g` alone.
Three bases, and -- crucially -- a **control state that has no correlation at
all**, the single reference determinant, carried through each rotation with
`fci.addons.transform_ci_for_orbital_rotation`:

  canonical   the pipeline's basis.
  full ER     Edmiston-Ruedenberg over all eight active orbitals: maximize
              sum_p (pp|pp).  This mixes occupied with virtual.
  block ER    ER within the four occupied-like and the four virtual-like
              orbitals separately.  This is the subgroup that **preserves the
              reference determinant**, so any off-diagonal weight it leaves is
              correlation rather than bookkeeping.

The control is what makes the answer unambiguous.  If a zero-correlation
determinant also picks up off-diagonal weight under a rotation, that rotation is
manufacturing coherence, not revealing it.

Writes ``results/localized_basis.npz``.

    .venv/bin/python scripts/localized_basis_experiment.py --limit 1000 --workers 8
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "results/qh9_dense_cas8-8_kT0p1.h5"
OUT = REPO / "results/localized_basis.npz"
INV_TOL = 1e-8

os.environ.setdefault("OMP_NUM_THREADS", "1")


def transform(h1, g, U):
    return U.T @ h1 @ U, np.einsum("pi,qj,rk,sl,pqrs->ijkl", U, U, U, U, g,
                                   optimize=True)


def er_objective(g):
    return float(np.einsum("pppp->", g))


def er_localize(h1, g, sweeps=8, tol=1e-10):
    """Edmiston-Ruedenberg by Jacobi pair rotations; uses only the ERIs.

    The 2x2 sub-objective is a pure second harmonic in the rotation angle
    (Edmiston-Ruedenberg 1963):

        Delta(theta) = A + A cos(4 theta) + B sin(4 theta)
        A = (pq|pq) - (1/4)[(pp|pp) + (qq|qq) - 2 (pp|qq)]
        B = (pp|pq) - (qq|pq)

    maximised at 4 theta = atan2(B, -A).  Every accepted rotation must raise the
    objective, which is checked rather than assumed.
    """
    n = h1.shape[0]
    U = np.eye(n)
    gg = g.copy()
    obj = er_objective(gg)
    for _ in range(sweeps):
        start = obj
        for p in range(n):
            for q in range(p + 1, n):
                A = (gg[p, q, p, q]
                     - 0.25 * (gg[p, p, p, p] + gg[q, q, q, q]
                               - 2.0 * gg[p, p, q, q]))
                B = gg[p, p, p, q] - gg[q, q, p, q]
                if abs(A) < 1e-14 and abs(B) < 1e-14:
                    continue
                theta = 0.25 * np.arctan2(B, -A)
                c, s = np.cos(theta), np.sin(theta)
                if abs(s) < 1e-12:
                    continue
                R = np.eye(n)
                R[p, p] = R[q, q] = c
                R[p, q], R[q, p] = -s, s
                _, gg_new = transform(np.zeros_like(h1), gg, R)
                if er_objective(gg_new) > obj:
                    gg, obj = gg_new, er_objective(gg_new)
                    U = U @ R
        if obj - start < tol:
            break
    return U


def block_er(h1, g, nocc):
    """ER inside the occupied block and inside the virtual block, separately."""
    n = h1.shape[0]
    Uo = er_localize(h1[:nocc, :nocc], g[:nocc, :nocc, :nocc, :nocc])
    Uv = er_localize(h1[nocc:, nocc:], g[nocc:, nocc:, nocc:, nocc:])
    U = np.zeros((n, n))
    U[:nocc, :nocc] = Uo
    U[nocc:, nocc:] = Uv
    return U


def ground_state(h1, g, ncas, nelec):
    from pyscf import fci
    s = fci.direct_spin1.FCI()
    s.conv_tol = 1e-12
    e, c = s.kernel(h1, g, ncas, nelec)
    return float(e), np.asarray(c)


def offdiag_share(v):
    """||offdiag rho||_F^2 / ||rho||_F^2 for the pure state |v><v|."""
    v = np.asarray(v).ravel()
    v = v / np.linalg.norm(v)
    return float(1.0 - np.sum(v ** 4))


def features(c, ncas, na, nb):
    from qthermal.encode import extended_heisenberg_expectations
    return extended_heisenberg_expectations(
        np.asarray(c).ravel()[None, :], np.array([1.0]), ncas, na, nb, "blocked")


def one_molecule(payload):
    from pyscf import fci

    idx, h1, g, ncas, na, nb = payload
    out = {"idx": idx, "ok": 0.0}
    try:
        nocc = na
        e_c, c_can = ground_state(h1, g, ncas, (na, nb))

        # zero-correlation control: the reference determinant as a CI vector
        ref = np.zeros_like(c_can)
        addr = fci.cistring.str2addr(ncas, na, (1 << nocc) - 1)
        ref[addr, addr] = 1.0

        res = {"canonical": (np.eye(ncas), c_can, ref)}
        for tag, U in (("full_er", er_localize(h1, g)),
                       ("block_er", block_er(h1, g, nocc))):
            h1r, gr = transform(h1, g, U)
            e_r, c_r = ground_state(h1r, gr, ncas, (na, nb))
            ref_r = fci.addons.transform_ci_for_orbital_rotation(
                ref, ncas, (na, nb), U)
            out[f"inv_dev_{tag}"] = abs(e_r - e_c)
            res[tag] = (U, c_r, ref_r)

        for tag, (U, c_r, ref_r) in res.items():
            out[f"share_{tag}"] = offdiag_share(c_r)
            out[f"share_ctrl_{tag}"] = offdiag_share(ref_r)
            out[f"wmax_{tag}"] = float(np.max(np.asarray(c_r).ravel() ** 2)
                                       / np.sum(np.asarray(c_r) ** 2))
        out["feat_canonical"] = features(c_can, ncas, na, nb)
        out["feat_block_er"] = features(res["block_er"][1], ncas, na, nb)
        out["E_canonical"] = e_c
        out["ok"] = 1.0
    except Exception as exc:                                  # noqa: BLE001
        out["error"] = repr(exc)[:200]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    with h5py.File(RUN, "r") as f:
        names = sorted((k for k in f if k.startswith("mol_")),
                       key=lambda s: int(s.split("_")[1]))
        if a.limit:
            names = names[:a.limit]
        meta = dict(f["meta"].attrs)
        ncas, nel = int(meta["ncas"]), int(meta["nelecas"])
        na = nb = nel // 2
        jobs = [(int(nm.split("_")[1]), f[nm]["h1eff"][:], f[nm]["g"][:],
                 ncas, na, nb) for nm in names]

    print(f"{len(jobs)} molecules, CAS({nel},{ncas}), {a.workers} workers",
          flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, r in enumerate(ex.map(one_molecule, jobs, chunksize=4)):
            rows.append(r)
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(jobs)}", flush=True)

    ok = np.array([bool(r["ok"]) for r in rows])
    bad = [r for r in rows if not r["ok"]]
    print(f"solved {ok.sum()}/{len(rows)}"
          + (f"   first error: {bad[0].get('error')}" if bad else ""))

    def col(key):
        return np.array([r.get(key, np.nan) for r in rows], dtype=float)

    for tag in ("full_er", "block_er"):
        d = col(f"inv_dev_{tag}")[ok]
        print(f"FCI invariance, {tag:9s}: max |dE| = {np.nanmax(d):.3e} Ha")
        assert np.nanmax(d) < INV_TOL, f"{tag} rotation changed the FCI energy"

    print("\noff-diagonal share of the ground state (median over molecules):")
    print(f"{'basis':10s} {'correlated state':>18s} {'zero-correlation control':>26s}")
    for tag in ("canonical", "full_er", "block_er"):
        print(f"{tag:10s} {np.nanmedian(col('share_' + tag)[ok]):18.4f}"
              f" {np.nanmedian(col('share_ctrl_' + tag)[ok]):26.4f}")

    sc, sb = col("share_canonical")[ok], col("share_block_er")[ok]
    print(f"\ncanonical -> block-ER, per molecule: max |change| = "
          f"{np.nanmax(np.abs(sc - sb)):.2e}, median |change| = "
          f"{np.nanmedian(np.abs(sc - sb)):.2e}")

    np.savez_compressed(
        OUT, idx=np.array([r["idx"] for r in rows]), ok=ok,
        **{k: col(k) for k in
           ("E_canonical", "inv_dev_full_er", "inv_dev_block_er",
            "share_canonical", "share_full_er", "share_block_er",
            "share_ctrl_canonical", "share_ctrl_full_er", "share_ctrl_block_er",
            "wmax_canonical", "wmax_full_er", "wmax_block_er")},
        feat_canonical=np.array([r.get("feat_canonical", np.full(248, np.nan))
                                 for r in rows]),
        feat_block_er=np.array([r.get("feat_block_er", np.full(248, np.nan))
                                for r in rows]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
