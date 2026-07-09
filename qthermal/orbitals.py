"""Module B — molecule construction and orbital validation.

Builds neutral closed-shell PySCF molecules at def2-SVP with spherical AOs
(PySCF default — QH9 was generated with PySCF defaults, so AO ordering matches
as long as atom order is preserved), and validates or recovers MO coefficients.

Validation failures raise :class:`MoleculeValidationError`; the orchestrator
(Module H) catches these, logs the reason, and skips the molecule — validation
must never crash a run.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.linalg
from pyscf import gto

logger = logging.getLogger(__name__)

MIN_GAP_HA = 0.02


class MoleculeValidationError(RuntimeError):
    """A molecule failed an acceptance check; skip it and record the reason."""


def _canonicalize_signs(C: np.ndarray) -> np.ndarray:
    """Fix each orbital's sign gauge: largest-magnitude AO coefficient > 0.

    MO coefficients are only defined up to a per-column sign; without this,
    h1eff/g/civecs pick up arbitrary sign flips between otherwise-identical
    molecules, which is noise to any model trained on the raw tensors.
    """
    signs = np.sign(C[np.argmax(np.abs(C), axis=0), np.arange(C.shape[1])])
    return C * np.where(signs == 0, 1.0, signs)


def build_mol(record, unit: str) -> gto.Mole:
    """PySCF Mole for a record: def2-SVP, neutral, singlet, spherical AOs."""
    atom = [[int(z), tuple(map(float, xyz))]
            for z, xyz in zip(record.Z, record.R)]
    return gto.M(atom=atom, basis="def2-svp", charge=0, spin=0,
                 unit=unit, verbose=0)


def overlap(mol: gto.Mole) -> np.ndarray:
    """AO overlap matrix S."""
    return np.asarray(mol.intor("int1e_ovlp"), dtype=np.float64)


def orbitals(record, mol: gto.Mole, S: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return validated (C, eps, nocc); recover C, eps from eigh(F, S) if absent.

    Acceptance checks (any failure raises MoleculeValidationError):
      - mol.nelectron even (closed shell),
      - C.T @ S @ C == I to 1e-6,
      - eps sorted ascending,
      - HOMO-LUMO gap > 0.02 Ha.
    """
    if mol.nelectron % 2 != 0:
        raise MoleculeValidationError(
            f"odd electron count {mol.nelectron} (idx={record.idx})")
    nocc = mol.nelectron // 2

    if record.C is not None:
        C = np.asarray(record.C, dtype=np.float64)
        if record.eps is not None:
            eps = np.asarray(record.eps, dtype=np.float64)
        else:
            eps = np.einsum("pi,pq,qi->i", C, record.F, C)
    else:
        eps, C = scipy.linalg.eigh(record.F, S)
        eps = np.asarray(eps, dtype=np.float64)
        C = np.asarray(C, dtype=np.float64)

    C = _canonicalize_signs(C)

    ortho_err = float(np.abs(C.T @ S @ C - np.eye(C.shape[1])).max())
    if ortho_err > 1e-6:
        raise MoleculeValidationError(
            f"orbitals not S-orthonormal, max |C^T S C - I| = {ortho_err:.3e} "
            f"(idx={record.idx})")

    if np.any(np.diff(eps) < 0):
        raise MoleculeValidationError(
            f"orbital energies not sorted ascending (idx={record.idx})")

    if nocc >= len(eps):
        raise MoleculeValidationError(
            f"nocc={nocc} exceeds orbital count {len(eps)} (idx={record.idx})")
    gap = float(eps[nocc] - eps[nocc - 1])
    if gap <= MIN_GAP_HA:
        raise MoleculeValidationError(
            f"HOMO-LUMO gap {gap:.4f} Ha <= {MIN_GAP_HA} (idx={record.idx})")

    return C, eps, nocc
