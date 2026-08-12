"""Second-quantization labels the stored active-space Hamiltonians already permit.

The gap audit (RESEARCH_LOG 2026-08-06) closed the HOMO-LUMO gap because it is a
*one-body* observable: `eigh(F, S)` is a one-body eigenproblem, so there is no
correlation content for a classical model to miss.  The obvious repair is a
label with **no mean-field analogue at all**.  Every molecule's run group
carries `ecore`, `h1eff` and the full `ncas^4` `g`, which is the complete
active-space Hamiltonian -- so we can solve *other particle-number and spin
sectors* of it without touching QH9, PySCF orbitals, or the 45 GB run file's
eigenvectors.

For each molecule, in the same CAS(8,8) active space and on the same orbitals:

    E_N      FCI ground state, (4a, 4b)      -- the neutral, already stored
    E_cat    FCI ground state, (4a, 3b)      -- the cation
    E_ani    FCI ground state, (5a, 4b)      -- the anion
    E_trip   FCI ground state, (5a, 3b)      -- the S_z = 1 triplet

and the *single-determinant* energy of the corresponding reference determinant
in each sector, by Slater's rules on the same integrals.  The differences are
the labels:

    IP_corr  = (E_cat - E_N)   - (E_det_cat - E_det_N)
    EA_corr  = ...                                     analogous
    ST_corr  = (E_trip - E_N)  - (E_det_trip - E_det_N)

Each is **identically zero for any single-determinant state**: it is the part of
an ionization energy or a spin-state splitting that exists only in the
many-body description.  Orbital relaxation cannot contaminate it, because both
terms use the same frozen orbitals.

Two more with no mean-field analogue:

    Z_pole   sum_p |<Psi_cat| a_p |Psi_N>|^2 -- the quasiparticle pole strength.
             Exactly 1 for a single determinant; below 1 exactly to the extent
             the two states are correlated.  This is the photoemission satellite
             weight, and it is a genuine overlap between sectors.
    N_unpaired  sum_i min(n_i, 2 - n_i) over natural occupations (Head-Gordon):
             0 for any closed-shell determinant.

Writes ``results/second_quantization_labels.npz``.

    .venv/bin/python scripts/second_quantization_labels.py --limit 1000 --workers 8
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
OUT = REPO / "results/second_quantization_labels.npz"
KT = "kT_0p1000"

os.environ.setdefault("OMP_NUM_THREADS", "1")


def det_energy(h1: np.ndarray, g: np.ndarray, occ_a, occ_b) -> float:
    """<D|H|D> for the determinant with these alpha/beta occupations.

    Slater's rules on chemist-notation integrals: g[p,q,r,s] = (pq|rs), so the
    Coulomb term is g[p,p,q,q] and the (same-spin only) exchange is g[p,q,q,p].
    """
    a, b = list(occ_a), list(occ_b)
    e = sum(h1[p, p] for p in a) + sum(h1[p, p] for p in b)
    for s in (a, b):
        for p in s:
            for q in s:
                e += 0.5 * (g[p, p, q, q] - g[p, q, q, p])
    for p in a:
        for q in b:
            e += g[p, p, q, q]
    return float(e)


def solve(h1, g, ncas, nelec, nroots=1):
    from pyscf import fci
    solver = fci.direct_spin1.FCI()
    solver.max_cycle = 200
    solver.conv_tol = 1e-10
    e, c = solver.kernel(h1, g, ncas, nelec, nroots=nroots)
    if nroots == 1:
        return float(e), c
    return np.asarray(e), c


def one_molecule(payload):
    """All second-quantization labels for one molecule. Returns a dict."""
    from pyscf import fci

    idx, ecore, h1, g, ncas, na, nb, nat_occs = payload
    out = {"idx": idx, "ok": 0.0}
    try:
        e_n, c_n = solve(h1, g, ncas, (na, nb))
        e_cat, c_cat = solve(h1, g, ncas, (na, nb - 1))
        e_ani, _ = solve(h1, g, ncas, (na + 1, nb))
        e_tri, _ = solve(h1, g, ncas, (na + 1, nb - 1))

        nocc = na                                  # doubly occupied reference
        ref_a = list(range(nocc))
        ed_n = det_energy(h1, g, ref_a, ref_a)
        # independent check of Slater's rules: energy of the one-hot CI vector
        c_ref = np.zeros_like(c_n)
        addr = fci.cistring.str2addr(ncas, na, (1 << nocc) - 1)
        c_ref[addr, addr] = 1.0
        out["det_energy_dev"] = abs(
            ed_n - float(fci.direct_spin1.energy(h1, g, c_ref, ncas, (na, nb))))
        ed_cat = det_energy(h1, g, ref_a, list(range(nocc - 1)))
        ed_ani = det_energy(h1, g, list(range(nocc + 1)), ref_a)
        ed_tri = det_energy(h1, g, list(range(nocc + 1)), list(range(nocc - 1)))

        out.update(
            E_N=e_n, E_cat=e_cat, E_ani=e_ani, E_trip=e_tri,
            IP_cas=e_cat - e_n, EA_cas=e_ani - e_n, ST_cas=e_tri - e_n,
            IP_mf=ed_cat - ed_n, EA_mf=ed_ani - ed_n, ST_mf=ed_tri - ed_n,
            IP_corr=(e_cat - e_n) - (ed_cat - ed_n),
            EA_corr=(e_ani - e_n) - (ed_ani - ed_n),
            ST_corr=(e_tri - e_n) - (ed_tri - ed_n),
            E_corr_neutral=e_n - ed_n,
        )

        # quasiparticle pole strength: total weight of a_p|Psi_N> on |Psi_cat>
        z = 0.0
        for p in range(ncas):
            d = fci.addons.des_b(c_n, ncas, (na, nb), p)
            z += float(np.tensordot(c_cat, d, axes=([0, 1], [0, 1]))) ** 2
        out["Z_pole"] = z

        # correlation diagnostics of the neutral ground state
        dm1 = fci.direct_spin1.make_rdm1(c_n, ncas, (na, nb))
        n_i = np.linalg.eigvalsh(dm1)[::-1]
        out["N_unpaired"] = float(np.sum(np.minimum(n_i, 2.0 - n_i)))
        p_i = np.clip(n_i / 2.0, 1e-12, 1 - 1e-12)
        out["S_orb"] = float(-np.sum(p_i * np.log(p_i)
                                     + (1 - p_i) * np.log(1 - p_i)))
        out["nat_occ_dev"] = float(np.sum(np.minimum(nat_occs, 2 - nat_occs)))

        # ground-state weight on doubles and higher (rank vs the reference)
        na_str = fci.cistring.make_strings(range(ncas), na)
        nb_str = fci.cistring.make_strings(range(ncas), nb)
        ref = (1 << nocc) - 1
        rank_a = np.array([bin(s & ~ref).count("1") for s in na_str])
        rank_b = np.array([bin(s & ~ref).count("1") for s in nb_str])
        rank = rank_a[:, None] + rank_b[None, :]
        w = c_n ** 2
        out["w_ref"] = float(w[rank == 0].sum())
        out["w_single"] = float(w[rank == 1].sum())
        out["w_double"] = float(w[rank == 2].sum())
        out["w_higher"] = float(w[rank >= 3].sum())
        out["ok"] = 1.0
    except Exception as exc:                              # noqa: BLE001
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
        meta = dict(f["meta"].attrs) if "meta" in f else {}
        ncas = int(meta.get("ncas", 8))
        nel = int(meta.get("nelecas", 8))
        na = nb = nel // 2
        jobs = []
        for nm in names:
            grp = f[nm]
            jobs.append((int(nm.split("_")[1]), float(grp["ecore"][()]),
                         grp["h1eff"][:], grp["g"][:], ncas, na, nb,
                         grp[KT]["nat_occs"][:]))
        evals0 = np.array([float(f[nm]["evals"][0]) for nm in names])
        ecores = np.array([float(f[nm]["ecore"][()]) for nm in names])

    print(f"{len(jobs)} molecules, CAS({nel},{ncas}), {a.workers} workers",
          flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, r in enumerate(ex.map(one_molecule, jobs, chunksize=4)):
            rows.append(r)
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(jobs)}", flush=True)

    ok = np.array([r["ok"] for r in rows]) > 0
    print(f"solved {ok.sum()}/{len(rows)}")
    for r in rows[:len(rows)]:
        if not r["ok"]:
            print("  FAILED", r["idx"], r.get("error"))
            break

    keys = [k for k in rows[0] if k not in ("ok", "error")]
    data = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
            for k in keys}
    data["idx"] = np.array([r["idx"] for r in rows], dtype=np.int64)
    data["ok"] = ok

    # correctness check: `evals` holds the ACTIVE-SPACE CI eigenvalues, ecore
    # excluded (verified against `qthermal.diagonalize`), so the independent FCI
    # here must reproduce evals[0] directly.
    dev = np.abs(data["E_N"][ok] - evals0[ok])
    print(f"max |E_N - evals[0]| = {dev.max():.3e} Ha   "
          f"(ecore is excluded from `evals`; median |ecore| "
          f"{np.median(np.abs(ecores)):.1f} Ha)")
    data["energy_check_max_dev"] = np.array(dev.max())

    np.savez_compressed(OUT, **data)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
