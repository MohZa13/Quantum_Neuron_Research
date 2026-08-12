"""Pilot: a label the diagonal is *structurally* blind to, measured on real chemistry.

Every label tested so far failed the same way -- not because coherence was
absent per state, but because `diag(rho)` and molecular composition already
carried the answer at the dataset level (RESEARCH_LOG 2026-08-06).  The
structural escape is a label where two states have **the same determinant
populations and differ only in the sign structure**: singlet versus triplet
coupling of the same two open shells.

Twisted ethylene is the minimal real system that does this, and it is the
textbook model of the retinal/rhodopsin photoisomerisation coordinate.  At a
planar geometry the ground state is a closed-shell singlet far below the
triplet.  At 90 degrees the pi bond is broken, the two electrons localise on
separate carbons, and the singlet and triplet become degenerate diradicals
built from *the same two determinants* -- identical diagonals, opposite
off-diagonal sign.  Composition is constant along the whole scan by
construction, so the composition confound cannot operate at all.

This is the fixed-composition, varying-correlation axis that QH9 lacks and that
OMol25 supplies three ways: spin-state ladders on one geometry, AFIR reaction
paths, and conformers.

**This script deliberately runs its own SCF and its own geometries.**  That is
outside the Phase-1 pipeline contract (INVARIANTS I2, which forbids SCF and
geometry changes in `qthermal`), and nothing here writes to the pipeline; it is
a standalone probe, at the same level of theory QH9 used (B3LYP/def2-SVP) so the
numbers are comparable with the production run.

    MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/spin_ladder_pilot.py

Writes ``results/spin_ladder_pilot.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/spin_ladder_pilot.json"

NCAS, NELECAS = 8, 8
ANGLES = [0, 15, 30, 45, 60, 70, 80, 85, 90]
HA_EV = 27.211386


def ethylene(twist_deg: float, r_cc=1.339, r_ch=1.087, ang=121.3):
    """C2H4 with one CH2 group rotated by `twist_deg` about the C=C axis."""
    a = np.deg2rad(ang)
    t = np.deg2rad(twist_deg)
    # Carbons on z. `ang` is the H-C=C angle, so the C->H vector makes that
    # angle with the C->C direction: its z-component is r_ch*cos(ang) < 0,
    # i.e. the hydrogens point AWAY from the other carbon.
    dz, dy = r_ch * np.cos(a), r_ch * np.sin(a)
    atoms = [("C", (0.0, 0.0, -r_cc / 2)), ("C", (0.0, 0.0, r_cc / 2))]
    for sy in (+1, -1):                                    # bottom CH2, fixed
        atoms.append(("H", (0.0, sy * dy, -r_cc / 2 + dz)))
    for sy in (+1, -1):                                    # top CH2, rotated
        y, x = sy * dy * np.cos(t), sy * dy * np.sin(t)
        atoms.append(("H", (x, y, r_cc / 2 - dz)))
    return atoms


def build(twist_deg, dm0=None):
    """B3LYP/def2-SVP RKS, then the CAS(8,8) Hamiltonian on those orbitals.

    The closed-shell RKS solution for twisted ethylene has more than one branch
    (this is exactly why OMol25 runs such systems in UKS with a broken-symmetry
    guess), so callers should scan with **continuation**: pass the previous
    angle's converged density as ``dm0`` to stay on one branch.  Without it the
    scan jumps between solutions and the CASCI numbers are not comparable.
    """
    from pyscf import dft, gto, mcscf

    mol = gto.M(atom=[(s, tuple(map(float, r))) for s, r in ethylene(twist_deg)],
                basis="def2-svp", unit="Angstrom", spin=0, charge=0, verbose=0)
    mf = dft.RKS(mol, xc="b3lyp")
    mf.conv_tol = 1e-11
    mf.max_cycle = 200
    mf.kernel(dm0=dm0)
    # canonicalise MO signs the way qthermal.orbitals does (INVARIANTS I4)
    C = mf.mo_coeff.copy()
    for k in range(C.shape[1]):
        if C[np.argmax(np.abs(C[:, k])), k] < 0:
            C[:, k] *= -1
    mf.mo_coeff = C

    mc = mcscf.CASCI(mf, NCAS, NELECAS)
    h1, ecore = mc.get_h1eff()
    g = mc.get_h2eff()
    from pyscf import ao2mo
    g = ao2mo.restore(1, np.asarray(g), NCAS)
    return (mol, mf, float(ecore), np.asarray(h1), g, mf.converged,
            mf.make_rdm1(), float(mf.e_tot))


def sector_state(h1, g, nelec, nroots=1):
    from pyscf import fci
    s = fci.direct_spin1.FCI()
    s.conv_tol = 1e-11
    e, c = s.kernel(h1, g, NCAS, nelec, nroots=nroots)
    return (np.atleast_1d(e), c if nroots > 1 else [c])


def rho_from_vec(c, dim):
    v = np.asarray(c).ravel()
    v = v / np.linalg.norm(v)
    return np.outer(v, v)


def split_norms(M):
    d = np.diag(M)
    off2 = float(np.sum(M * M) - np.sum(d ** 2))
    return float(np.linalg.norm(d)), float(np.sqrt(max(off2, 0.0)))


def main() -> None:
    from pyscf import fci

    rows = []
    for ang in ANGLES:
        mol, mf, ecore, h1, g, conv = build(ang)
        na = nb = NELECAS // 2

        # lowest states of each spin sector, on the SAME Hamiltonian
        e_s, cs = sector_state(h1, g, (na, nb), nroots=3)
        e_t, ct = sector_state(h1, g, (na + 1, nb - 1), nroots=1)

        # pick the lowest true singlet out of the S_z = 0 roots
        pick, s2v = None, []
        for k, c in enumerate(cs):
            ss = fci.spin_op.spin_square(c, NCAS, (na, nb))[0]
            s2v.append(float(ss))
            if pick is None and ss < 0.2:
                pick = k
        pick = 0 if pick is None else pick
        c_sing = cs[pick]
        e_sing = float(e_s[pick])

        dim = c_sing.size
        rho_s = rho_from_vec(c_sing, dim)

        # the triplet in the SAME S_z = 0 determinant sector, so the two states
        # live in one space and their diagonals are directly comparable
        c_t0 = None
        for k, c in enumerate(cs):
            if s2v[k] > 1.8:
                c_t0 = c
                break
        rho_t = rho_from_vec(c_t0, dim) if c_t0 is not None else None

        row = dict(angle=ang, scf_converged=bool(conv),
                   E_singlet=e_sing + ecore, E_triplet=float(e_t[0]) + ecore,
                   ST_gap_eV=(float(e_t[0]) - e_sing) * HA_EV,
                   s2_roots=s2v, root_used=int(pick))

        if rho_t is not None:
            d_s, o_s = split_norms(rho_s)
            d_t, o_t = split_norms(rho_t)
            dd, do = split_norms(rho_s - rho_t)
            row.update(
                ST_gap_same_sector_eV=(float(e_s[[i for i, v in enumerate(s2v)
                                                 if v > 1.8][0]]) - e_sing) * HA_EV,
                offdiag_share_singlet=o_s ** 2 / (o_s ** 2 + d_s ** 2),
                offdiag_share_triplet=o_t ** 2 / (o_t ** 2 + d_t ** 2),
                # the screening statistic, between the two classes
                screen_ratio=do / dd,
                diag_overlap=float(
                    np.dot(np.diag(rho_s), np.diag(rho_t))
                    / (np.linalg.norm(np.diag(rho_s))
                       * np.linalg.norm(np.diag(rho_t)))),
                leading_weights=sorted(np.diag(rho_s))[-3:][::-1],
            )
        rows.append(row)
        print(f"  {ang:3d} deg  S-T {row['ST_gap_eV']:+7.3f} eV  "
              f"screen {row.get('screen_ratio', float('nan')):8.3f}  "
              f"diag-overlap {row.get('diag_overlap', float('nan')):.5f}",
              flush=True)

    out = dict(
        system="C2H4 torsion scan, B3LYP/def2-SVP orbitals, CASCI(8,8)",
        note=("`screen_ratio` is ||offdiag(rho_S - rho_T)||_F / "
              "||diag(rho_S - rho_T)||_F -- the same statistic as the R+/R- "
              "screen, here between the singlet and triplet classes at one "
              "fixed composition. QH9's median-split HOMO-LUMO gap gives "
              "0.1345 over 1000 molecules."),
        rows=rows)
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
