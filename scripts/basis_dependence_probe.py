"""Is "coherence" a property of the state, or of the orbital basis we store it in?

Every coherence number this project has measured -- the 6.7% off-diagonal share,
the R+/R- screen ratios, the Z/ZZ-vs-XX/YY ablation -- is computed in the
determinant basis built from **canonical, delocalized Kohn-Sham orbitals**.
The determinant basis is not physically privileged: an orthogonal rotation
inside the active space leaves every energy, every eigenvalue of rho, and every
observable invariant, while completely redistributing weight between diag(rho)
and offdiag(rho).

Twisted ethylene makes the stakes concrete.  At 90 degrees the pi bond is
broken and the ground state is a singlet diradical.  In the canonical
(delocalized) basis that singlet is the two-configuration mixture
(pi^2 - pi*^2)/sqrt(2), while the triplet is the open-shell (pi pi*) --
**different determinants, so a classical model reading only populations can
tell them apart**.  Rotate the frontier pair by 45 degrees onto the two carbons
and the same two states become (a b_bar +/- b a_bar)/sqrt(2): identical
populations, opposite off-diagonal sign.  Same physics, same energies, and the
singlet/triplet label moves from the diagonal channel to the coherence channel.

This script measures that, with an FCI-invariance gate (the energies must not
move) so the rotation cannot be silently wrong.

    .venv/bin/python scripts/basis_dependence_probe.py

Writes ``results/basis_dependence_probe.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spin_ladder_pilot import NCAS, NELECAS, build

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/basis_dependence_probe.json"
ANGLES = [0, 10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90]
HA_EV = 27.211386
TOL = 1e-8


def transform(h1, g, U):
    """Rotate the active-space Hamiltonian by the orthogonal U (columns = new)."""
    h1p = U.T @ h1 @ U
    gp = np.einsum("pi,qj,rk,sl,pqrs->ijkl", U, U, U, U, g, optimize=True)
    return h1p, gp


def lowest_of_each_spin(h1, g, na, nb, nroots=12):
    """Lowest S=0 and lowest S=1 root of the S_z = 0 sector, as flat vectors."""
    from pyscf import fci
    s = fci.direct_spin1.FCI()
    s.conv_tol = 1e-12
    s.max_space = 30
    e, cs = s.kernel(h1, g, NCAS, (na, nb), nroots=nroots)
    sing = trip = None
    for k, c in enumerate(cs):
        ss = float(fci.spin_op.spin_square(c, NCAS, (na, nb))[0])
        v = np.asarray(c).ravel()
        v = v / np.linalg.norm(v)
        if ss < 0.2 and sing is None:
            sing = (float(e[k]), v, np.asarray(c))
        if 1.8 < ss < 2.2 and trip is None:
            trip = (float(e[k]), v, np.asarray(c))
        if sing and trip:
            break
    return sing, trip


def pure_state_split(v0, v1):
    """(cos of the two diagonals, ||offdiag(d rho)||_F / ||diag(d rho)||_F).

    For pure states ||rho0 - rho1||_F^2 = 2(1 - <v0|v1>^2), so the dense
    4900 x 4900 difference is never formed.
    """
    d0, d1 = v0 ** 2, v1 ** 2
    cos = float(d0 @ d1 / (np.linalg.norm(d0) * np.linalg.norm(d1)))
    dd = float(np.linalg.norm(d0 - d1))
    fro2 = 2.0 * (1.0 - float(v0 @ v1) ** 2)
    off = float(np.sqrt(max(fro2 - dd ** 2, 0.0)))
    return cos, off / dd, dd, off


def main() -> None:
    from pyscf import fci

    rows = []
    dm0 = None                      # SCF continuation: stay on one RKS branch
    for ang in ANGLES:
        mol, mf, ecore, h1, g, conv, dm0, e_scf = build(ang, dm0=dm0)
        na = nb = NELECAS // 2

        # --- basis 1: canonical delocalized KS orbitals (what the pipeline uses)
        sing, trip = lowest_of_each_spin(h1, g, na, nb)
        if sing is None or trip is None:
            print(f"  {ang}: missing a spin state, skipped")
            continue
        cos_c, ratio_c, dd_c, off_c = pure_state_split(sing[1], trip[1])

        # --- basis 2: natural orbitals of that singlet
        dm1 = fci.direct_spin1.make_rdm1(sing[2], NCAS, (na, nb))
        occ, U_no = np.linalg.eigh(dm1)
        order = np.argsort(-occ)
        occ, U_no = occ[order], U_no[:, order]
        h1_no, g_no = transform(h1, g, U_no)
        sing_no, trip_no = lowest_of_each_spin(h1_no, g_no, na, nb)
        inv1 = max(abs(sing_no[0] - sing[0]), abs(trip_no[0] - trip[0]))
        cos_n, ratio_n, _, _ = pure_state_split(sing_no[1], trip_no[1])

        # --- basis 3: localize the two most-open-shell natural orbitals
        # (the pair whose occupations are closest to 1) by a 45 degree rotation
        pair = np.argsort(np.abs(occ - 1.0))[:2]
        i, j = int(min(pair)), int(max(pair))
        R = np.eye(NCAS)
        c45 = s45 = np.sqrt(0.5)
        R[i, i] = R[j, j] = c45
        R[i, j], R[j, i] = -s45, s45
        h1_l, g_l = transform(h1_no, g_no, R)
        sing_l, trip_l = lowest_of_each_spin(h1_l, g_l, na, nb)
        inv2 = max(abs(sing_l[0] - sing[0]), abs(trip_l[0] - trip[0]))
        cos_l, ratio_l, _, _ = pure_state_split(sing_l[1], trip_l[1])

        row = dict(
            angle=ang, scf_converged=bool(conv), E_scf=e_scf,
            ST_gap_eV=(trip[0] - sing[0]) * HA_EV,
            nat_occ=[float(x) for x in occ],
            localized_pair=[i, j], pair_occ=[float(occ[i]), float(occ[j])],
            fci_invariance_max_dev_Ha=[float(inv1), float(inv2)],
            canonical=dict(diag_cosine=cos_c, screen_ratio=ratio_c),
            natural=dict(diag_cosine=cos_n, screen_ratio=ratio_n),
            localized=dict(diag_cosine=cos_l, screen_ratio=ratio_l),
        )
        rows.append(row)
        ok = "OK " if max(inv1, inv2) < TOL else "!! "
        print(f"  {ok}{ang:3d} deg  S-T {row['ST_gap_eV']:+7.3f} eV | "
              f"canonical: cos {cos_c:.4f} screen {ratio_c:7.3f} | "
              f"localized: cos {cos_l:.4f} screen {ratio_l:9.3f} | "
              f"inv {max(inv1, inv2):.1e}", flush=True)

    out = dict(
        system="C2H4 torsion, B3LYP/def2-SVP orbitals, CASCI(8,8), lowest S=0 vs S=1",
        definition=("screen_ratio = ||offdiag(rho_S - rho_T)||_F / "
                    "||diag(rho_S - rho_T)||_F. Large means the singlet/triplet "
                    "label lives in the coherences; small means a model reading "
                    "only determinant populations can already separate them."),
        invariance_gate_Ha=TOL, rows=rows)
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
