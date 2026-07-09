"""Module D — ERIs and the frozen-core CASCI Hamiltonian.

This is the primary new computation: QH9 contains no two-electron data, so the
electron-repulsion integrals are computed here and transformed to the active
space, with the frozen core folded into an effective one-body matrix and a
scalar core energy.

Method note: the orbitals are B3LYP Kohn-Sham orbitals, but the active-space
Hamiltonian uses bare Coulomb integrals with an HF-style frozen-core potential
("CASCI on KS orbitals"). This is intentional — do not "fix" it by running
SCF, and never call ``mf.kernel()``, which would overwrite the QH9-consistent
orbitals.

Two-electron integrals ``g`` are returned as the full ncas^4 tensor in
**chemist notation** ``(pq|rs)`` (``ao2mo.restore(1, ...)`` applied — no
packed s8/s4 symmetry). OpenFermion-style physicist notation is a Phase-2
concern; no conversion happens here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import ao2mo, mcscf, scf

from qthermal.active_space import ActiveSpace

_SYM_TOL = 1e-9


@dataclass
class CASHamiltonian:
    """Frozen-core active-space Hamiltonian (energies in Hartree)."""

    ecore: float           # scalar core energy incl. nuclear repulsion
    h1eff: np.ndarray      # (ncas, ncas) effective one-body matrix
    g: np.ndarray          # (ncas,)*4 chemist-notation ERIs (pq|rs)


def make_injected_rhf(mol, C: np.ndarray, nocc: int) -> scf.hf.RHF:
    """Non-iterated RHF wrapper carrying injected orbitals. Never run kernel()."""
    mf = scf.RHF(mol)
    mf.mo_coeff = np.asarray(C, dtype=np.float64)
    mf.mo_occ = np.zeros(C.shape[1])
    mf.mo_occ[:nocc] = 2.0
    return mf


def make_casci(mf, aspace: ActiveSpace) -> mcscf.casci.CASCI:
    """CASCI object over the frontier window of the injected orbitals.

    PySCF's CASCI takes the ``ncas`` orbitals following the first
    ``(nelectron - nelecas) // 2`` doubly-occupied ones — exactly the
    contiguous frontier window Module C selects; the assertions prove the
    alignment for the actual parameters.
    """
    mc = mcscf.CASCI(mf, ncas=aspace.ncas, nelecas=aspace.nelecas)
    assert mc.ncore == aspace.ncore, (mc.ncore, aspace.ncore)
    np.testing.assert_array_equal(
        aspace.active_idx, mc.ncore + np.arange(aspace.ncas))
    return mc


def assert_g_8fold_symmetry(g: np.ndarray, tol: float = _SYM_TOL) -> None:
    """Assert invariance under the 8-fold permutational symmetry of real
    chemist-notation integrals: (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)."""
    for perm in [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)]:
        dev = float(np.abs(g - g.transpose(perm)).max())
        assert dev < tol, f"g violates permutation {perm}: max dev {dev:.3e}"


def build_cas_hamiltonian(mol, C: np.ndarray, nocc: int,
                          aspace: ActiveSpace) -> CASHamiltonian:
    """Frozen-core (ecore, h1eff, g) from injected orbitals via PySCF CASCI."""
    mf = make_injected_rhf(mol, C, nocc)
    mc = make_casci(mf, aspace)

    h1eff, ecore = mc.get_h1eff()
    h1eff = np.asarray(h1eff, dtype=np.float64)
    g = np.asarray(ao2mo.restore(1, mc.get_h2eff(), aspace.ncas),
                   dtype=np.float64)

    assert h1eff.shape == (aspace.ncas, aspace.ncas)
    assert g.shape == (aspace.ncas,) * 4
    h_asym = float(np.abs(h1eff - h1eff.T).max())
    assert h_asym < _SYM_TOL, f"h1eff asymmetry {h_asym:.3e}"
    assert_g_8fold_symmetry(g)

    return CASHamiltonian(ecore=float(ecore), h1eff=h1eff, g=g)
