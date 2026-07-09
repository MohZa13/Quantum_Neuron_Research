"""Module D tests: CASCI Hamiltonian vs fully manual frozen-core construction.

The manual route is deliberately implemented here (test-only, per spec):
h_core + HF-style core Coulomb/exchange potential from the frozen-core
density, transformed to the active window, plus explicitly transformed ERIs.
"""

import numpy as np
import pytest

from qthermal.active_space import select_active
from qthermal.hamiltonian import build_cas_hamiltonian
from qthermal.orbitals import build_mol, orbitals, overlap


@pytest.fixture(scope="module", params=[(3, 3), (4, 4)],
                ids=["ncas6", "ncas8"])
def cas_setup(request, h2o_record):
    n_act_occ, n_act_virt = request.param
    mol = build_mol(h2o_record, "Angstrom")
    S = overlap(mol)
    C, eps, nocc = orbitals(h2o_record, mol, S)
    aspace = select_active(eps, nocc, n_act_occ=n_act_occ, n_act_virt=n_act_virt)
    return mol, C, nocc, aspace


def manual_cas_hamiltonian(mol, C, aspace):
    """Spec Module D.2 cross-validation route (test-only, not production)."""
    from pyscf import ao2mo, scf

    C_core = C[:, aspace.core_idx]
    C_act = C[:, aspace.active_idx]

    h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    D_core = 2.0 * C_core @ C_core.T
    J, K = scf.hf.get_jk(mol, D_core)
    veff = J - 0.5 * K

    h1eff_manual = C_act.T @ (h_core + veff) @ C_act
    ecore_manual = mol.energy_nuc() + np.einsum(
        "ij,ji->", D_core, h_core + 0.5 * veff)
    g_manual = ao2mo.restore(1, ao2mo.kernel(mol, C_act), aspace.ncas)
    return float(ecore_manual), h1eff_manual, g_manual


def test_shapes_and_symmetry(cas_setup):
    mol, C, nocc, aspace = cas_setup
    ham = build_cas_hamiltonian(mol, C, nocc, aspace)
    assert ham.h1eff.shape == (aspace.ncas, aspace.ncas)
    assert ham.g.shape == (aspace.ncas,) * 4
    assert ham.h1eff.dtype == np.float64 and ham.g.dtype == np.float64
    assert np.isfinite(ham.ecore)
    # 8-fold permutational symmetry of real chemist-notation integrals.
    for perm in [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)]:
        assert np.abs(ham.g - ham.g.transpose(perm)).max() < 1e-9


def test_manual_cross_validation(cas_setup):
    mol, C, nocc, aspace = cas_setup
    ham = build_cas_hamiltonian(mol, C, nocc, aspace)
    ecore_m, h1eff_m, g_m = manual_cas_hamiltonian(mol, C, aspace)
    assert abs(ham.ecore - ecore_m) < 1e-8
    np.testing.assert_allclose(ham.h1eff, h1eff_m, atol=1e-8)
    np.testing.assert_allclose(ham.g, g_m, atol=1e-8)


def test_orbitals_not_mutated_by_construction(cas_setup):
    """The injected-orbital route must never trigger an SCF that alters C."""
    mol, C, nocc, aspace = cas_setup
    C_before = C.copy()
    build_cas_hamiltonian(mol, C, nocc, aspace)
    np.testing.assert_array_equal(C, C_before)
