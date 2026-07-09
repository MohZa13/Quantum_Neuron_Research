"""Shared fixtures for the qthermal test suite.

The synthetic reference record is H2O at B3LYP/def2-SVP, generated end-to-end
with PySCF (build mol, run SCF, feed the converged Fock matrix and coordinates
back in).  B3LYP is used rather than RHF because QH9 Hamiltonians are B3LYP
Kohn-Sham matrices and the `detect_units` physicality window (HOMO-LUMO gap in
[0.02, 0.6] Ha) is calibrated for KS spectra: RHF H2O/def2-SVP has a 0.67 Ha
gap and would fail its own criteria even in the correct unit.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

H2O_ATOM_ANGSTROM = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"


@pytest.fixture(scope="session")
def h2o_scf():
    """Converged B3LYP/def2-SVP H2O calculation (session-cached)."""
    from pyscf import dft, gto

    mol = gto.M(atom=H2O_ATOM_ANGSTROM, basis="def2-svp", charge=0, spin=0,
                unit="Angstrom", verbose=0)
    mf = dft.RKS(mol, xc="b3lyp")
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    return mol, mf


@pytest.fixture(scope="session")
def h2o_record(h2o_scf):
    """Synthetic MoleculeRecord for H2O with Angstrom coordinates."""
    from qthermal.loader import MoleculeRecord

    mol, mf = h2o_scf
    F = mf.get_fock()
    R = mol.atom_coords(unit="Angstrom")
    Z = np.array([mol.atom_charge(i) for i in range(mol.natm)], dtype=np.int64)
    return MoleculeRecord(idx=0, Z=Z, R=R, F=np.asarray(F, dtype=np.float64),
                          C=np.asarray(mf.mo_coeff, dtype=np.float64),
                          eps=np.asarray(mf.mo_energy, dtype=np.float64))


@pytest.fixture(scope="session")
def h2o_record_no_orbitals(h2o_record):
    """Same record with C/eps withheld, forcing recovery from eigh(F, S)."""
    from qthermal.loader import MoleculeRecord

    r = h2o_record
    return MoleculeRecord(idx=r.idx, Z=r.Z, R=r.R, F=r.F, C=None, eps=None)
