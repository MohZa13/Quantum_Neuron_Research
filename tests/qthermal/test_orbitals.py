"""Module B tests: build_mol, overlap, orbital recovery and validation."""

import numpy as np
import pytest

from qthermal.loader import MoleculeRecord
from qthermal.orbitals import (
    MoleculeValidationError,
    _canonicalize_signs,
    build_mol,
    orbitals,
    overlap,
)


def test_build_mol_matches_reference(h2o_scf, h2o_record):
    mol_ref, _ = h2o_scf
    mol = build_mol(h2o_record, "Angstrom")
    assert mol.nao == mol_ref.nao
    assert mol.nelectron == 10
    assert mol.charge == 0 and mol.spin == 0
    assert not mol.cart  # spherical AOs, PySCF default
    np.testing.assert_allclose(mol.atom_coords(), mol_ref.atom_coords(), atol=1e-10)


def test_overlap_matches_pyscf(h2o_scf, h2o_record):
    mol_ref, _ = h2o_scf
    mol = build_mol(h2o_record, "Angstrom")
    S = overlap(mol)
    np.testing.assert_allclose(S, mol_ref.intor("int1e_ovlp"), atol=1e-10)
    assert S.dtype == np.float64


def test_orbitals_validates_provided_C(h2o_record):
    mol = build_mol(h2o_record, "Angstrom")
    S = overlap(mol)
    C, eps, nocc = orbitals(h2o_record, mol, S)
    assert nocc == 5
    np.testing.assert_array_equal(C, h2o_record.C)
    np.testing.assert_array_equal(eps, h2o_record.eps)


def test_orbitals_recovers_from_fock(h2o_record, h2o_record_no_orbitals):
    mol = build_mol(h2o_record_no_orbitals, "Angstrom")
    S = overlap(mol)
    C, eps, nocc = orbitals(h2o_record_no_orbitals, mol, S)
    assert nocc == 5
    np.testing.assert_allclose(eps, h2o_record.eps, atol=1e-7)
    # Recovered orbitals equal converged ones up to per-column sign.
    ovl = C.T @ S @ h2o_record.C
    np.testing.assert_allclose(np.abs(np.diag(ovl)), 1.0, atol=1e-6)
    assert np.abs(C.T @ S @ C - np.eye(C.shape[1])).max() < 1e-6


def test_orbitals_rejects_bad_C(h2o_record):
    mol = build_mol(h2o_record, "Angstrom")
    S = overlap(mol)
    bad = MoleculeRecord(idx=9, Z=h2o_record.Z, R=h2o_record.R, F=h2o_record.F,
                         C=h2o_record.C * 1.01, eps=h2o_record.eps)
    with pytest.raises(MoleculeValidationError, match="orthonormal"):
        orbitals(bad, mol, S)


def test_orbitals_rejects_unsorted_eps(h2o_record):
    mol = build_mol(h2o_record, "Angstrom")
    S = overlap(mol)
    eps_bad = h2o_record.eps.copy()
    eps_bad[[0, 1]] = eps_bad[[1, 0]]
    bad = MoleculeRecord(idx=9, Z=h2o_record.Z, R=h2o_record.R, F=h2o_record.F,
                         C=h2o_record.C, eps=eps_bad)
    with pytest.raises(MoleculeValidationError, match="sorted"):
        orbitals(bad, mol, S)


def test_sign_gauge_is_fixed(h2o_record):
    """Two orbital sets that differ only by arbitrary per-column sign flips
    (the physically meaningless MO gauge freedom) must canonicalize to the
    exact same matrix, so h1eff/g/civecs built downstream are reproducible."""
    mol = build_mol(h2o_record, "Angstrom")
    S = overlap(mol)
    rng = np.random.default_rng(0)
    flips = rng.choice([-1.0, 1.0], size=h2o_record.C.shape[1])

    flipped = MoleculeRecord(idx=h2o_record.idx, Z=h2o_record.Z, R=h2o_record.R,
                             F=h2o_record.F, C=h2o_record.C * flips,
                             eps=h2o_record.eps)
    C_ref, _, _ = orbitals(h2o_record, mol, S)
    C_flipped, _, _ = orbitals(flipped, mol, S)
    np.testing.assert_array_equal(C_ref, C_flipped)


def test_canonicalize_signs_convention():
    C = np.array([[0.5, -0.2], [-0.9, 0.1]])
    out = _canonicalize_signs(C)
    # column 0's largest-magnitude entry (row 1, -0.9) must become positive
    assert out[1, 0] > 0
    # column 1's largest-magnitude entry (row 0, -0.2) must become positive
    assert out[0, 1] > 0
    np.testing.assert_array_equal(np.abs(out), np.abs(C))


def test_orbitals_rejects_tiny_gap(h2o_record):
    mol = build_mol(h2o_record, "Angstrom")
    S = overlap(mol)
    eps_bad = h2o_record.eps.copy()
    eps_bad[5] = eps_bad[4] + 1e-3  # collapse the gap, keep ordering
    bad = MoleculeRecord(idx=9, Z=h2o_record.Z, R=h2o_record.R, F=h2o_record.F,
                         C=h2o_record.C, eps=eps_bad)
    with pytest.raises(MoleculeValidationError, match="gap"):
        orbitals(bad, mol, S)
