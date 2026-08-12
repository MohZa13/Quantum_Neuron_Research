"""Module A — data interface for QH9 records.

Provides the :class:`MoleculeRecord` dataclass every downstream module consumes,
an adapter that streams records from the raw ``QH9Stable.db`` SQLite file, and
an empirical coordinate-unit detector (Angstrom vs Bohr must never be assumed).

The SQLite schema follows ``data/build_slater.py`` (``RawQH9SQLiteDataset``):
table ``data(id, N, Z, pos, Ham)`` with ``Z`` an int32 blob, ``pos`` a float64
``(N, 3)`` blob, and ``Ham`` a float64 ``(nao, nao)`` blob.

AO ordering (empirical, 2026-07-09): the raw ``QH9Stable.db`` ``Ham`` blobs
are **already in PySCF def2-SVP AO ordering** — no reordering is applied here.
Validated against fresh B3LYP/def2-SVP spectra on records 0-4: raw blobs agree
to <= 3.7e-3 Ha across the full spectrum (B3LYP-variant-level), while applying
the QHBench ``PYSCF_DEF2SVP_CONVENTION`` reorder (as ``data/build_slater.py``
line ~145 does) corrupts the matrices — ~1 Ha spurious core shifts on compact
molecules that still slip through the physicality windows, and catastrophic
intruder eigenvalues on linear molecules (HCN -57.7 Ha, C2H2 -173.7 Ha). The
transform helpers below are kept only for QHBench *processed/model-output*
matrices, which do use the native QH9 ordering.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import scipy.linalg

logger = logging.getLogger(__name__)

_SYMMETRY_TOL = 1e-9


@dataclass
class MoleculeRecord:
    """One QH9 molecule: geometry plus converged B3LYP/def2-SVP Fock matrix.

    ``C`` (MO coefficients) and ``eps`` (orbital energies) are optional; when
    absent, Module B recovers them from ``eigh(F, S)``.
    """

    idx: int
    Z: np.ndarray
    R: np.ndarray          # (n_atoms, 3), unit determined by detect_units
    F: np.ndarray          # (nao, nao), AO basis, PySCF ordering
    C: np.ndarray | None = None
    eps: np.ndarray | None = None

    def __post_init__(self):
        self.Z = np.asarray(self.Z, dtype=np.int64)
        self.R = np.asarray(self.R, dtype=np.float64)
        self.F = np.asarray(self.F, dtype=np.float64)
        if self.C is not None:
            self.C = np.asarray(self.C, dtype=np.float64)
        if self.eps is not None:
            self.eps = np.asarray(self.eps, dtype=np.float64)

        assert self.R.shape == (self.Z.shape[0], 3), \
            f"R shape {self.R.shape} inconsistent with {self.Z.shape[0]} atoms"
        nao = self.F.shape[0]
        assert self.F.shape == (nao, nao), f"F must be square, got {self.F.shape}"
        asym = float(np.abs(self.F - self.F.T).max())
        assert asym < _SYMMETRY_TOL, f"Fock matrix asymmetry {asym:.3e} > {_SYMMETRY_TOL}"


# --- QH9 -> PySCF AO reordering (def2-SVP) ---------------------------------
# Copied from data/build_slater.py (PYSCF_DEF2SVP_CONVENTION); kept here so the
# qthermal package stays self-contained and import-side-effect free.
# NOT applied to raw QH9Stable.db blobs (already PySCF-ordered — see module
# docstring); use these only for QHBench processed/model-output matrices.
_ATOM_TO_ORBITALS = {1: "ssp", 6: "sssppd", 7: "sssppd", 8: "sssppd", 9: "sssppd"}
_ORBITAL_IDX_MAP = {"s": [0], "p": [1, 2, 0], "d": [0, 1, 2, 3, 4]}
_ORBITAL_SIGN_MAP = {"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]}
_ORBITAL_ORDER_MAP = {
    1: [0, 1, 2],
    6: [0, 1, 2, 3, 4, 5],
    7: [0, 1, 2, 3, 4, 5],
    8: [0, 1, 2, 3, 4, 5],
    9: [0, 1, 2, 3, 4, 5],
}


def qh9_def2svp_ao_count(atomic_number: int) -> int:
    """AO count per atom in the QH9 def2-SVP Hamiltonian matrices."""
    return 5 if int(atomic_number) <= 2 else 14


def qh9_to_pyscf_transform(Z) -> tuple[np.ndarray, np.ndarray]:
    """Index permutation and sign flips mapping QH9 AO order to PySCF order.

    ``M_pyscf = (M_qh9[idx][:, idx]) * signs[:, None] * signs[None, :]``.
    """
    orbitals = ""
    orbitals_order: list[int] = []
    for z in np.asarray(Z, dtype=np.int64):
        z = int(z)
        offset = len(orbitals_order)
        orbitals += _ATOM_TO_ORBITALS[z]
        orbitals_order += [i + offset for i in _ORBITAL_ORDER_MAP[z]]

    indices: list[np.ndarray] = []
    signs: list[np.ndarray] = []
    for orb in orbitals:
        offset = sum(len(ix) for ix in indices)
        indices.append(np.asarray(_ORBITAL_IDX_MAP[orb]) + offset)
        signs.append(np.asarray(_ORBITAL_SIGN_MAP[orb], dtype=np.float64))

    indices = [indices[i] for i in orbitals_order]
    signs = [signs[i] for i in orbitals_order]
    return (np.concatenate(indices).astype(np.int64), np.concatenate(signs))


def qh9_ham_to_pyscf(ham: np.ndarray, Z) -> np.ndarray:
    """Reorder a raw QH9 def2-SVP Hamiltonian into PySCF AO ordering."""
    idx, signs = qh9_to_pyscf_transform(Z)
    out = ham[np.ix_(idx, idx)] * signs[:, None] * signs[None, :]
    return np.ascontiguousarray(out, dtype=np.float64)


# --- Record iteration --------------------------------------------------------

def _resolve_db_path(path) -> Path:
    # TODO(user): adapt to local storage. This adapter reads the raw
    # QH9Stable SQLite database (schema per data/build_slater.py). If your
    # records live elsewhere (e.g. with precomputed C/eps), replace or extend
    # this resolution plus `_record_from_row` below; everything downstream
    # depends only on the MoleculeRecord interface.
    p = Path(path)
    if p.is_dir():
        candidate = p / "QH9Stable" / "raw" / "QH9Stable.db"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"No QH9Stable/raw/QH9Stable.db under {p}; pass the .db file directly")
    if not p.exists():
        raise FileNotFoundError(f"QH9 database not found: {p}")
    return p


def _record_from_row(row) -> MoleculeRecord:
    idx, num_nodes, atoms_blob, pos_blob, ham_blob = row
    Z = np.frombuffer(atoms_blob, dtype=np.int32).astype(np.int64)
    R = np.frombuffer(pos_blob, dtype=np.float64).reshape(int(num_nodes), 3)
    nao = int(sum(qh9_def2svp_ao_count(z) for z in Z))
    # Raw QH9Stable.db Hamiltonians are already PySCF-ordered (see module
    # docstring) — reordering here would corrupt them.
    F = np.ascontiguousarray(
        np.frombuffer(ham_blob, dtype=np.float64).reshape(nao, nao))
    return MoleculeRecord(idx=int(idx), Z=Z, R=R.copy(), F=F, C=None, eps=None)


_SQLITE_MAX_VARS = 500          # well under SQLITE_MAX_VARIABLE_NUMBER


def iter_records(path, limit: int | None = None,
                 indices=None) -> Iterator[MoleculeRecord]:
    """Stream MoleculeRecords from the raw QH9Stable SQLite database.

    ``path`` is either the ``.db`` file or a QH9 root directory containing
    ``QH9Stable/raw/QH9Stable.db``. Yields at most ``limit`` records when
    ``limit`` is not None.

    ``indices`` restricts the stream to those QH9 record ids (``MoleculeRecord
    .idx``), still in ascending id order — the way to run a targeted subset,
    e.g. the head of a screening ranking, without walking the whole table.
    Ids absent from the database are silently skipped.
    """
    if limit is not None and limit <= 0:
        return
    db_path = _resolve_db_path(path)
    n_yielded = 0
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        if indices is None:
            cursor = conn.execute(
                "select id, N, Z, pos, Ham from data order by id")
            for row in cursor:
                yield _record_from_row(row)
                n_yielded += 1
                if limit is not None and n_yielded >= limit:
                    return
            return

        ids = sorted({int(i) for i in indices})
        for start in range(0, len(ids), _SQLITE_MAX_VARS):
            chunk = ids[start:start + _SQLITE_MAX_VARS]
            placeholders = ",".join("?" * len(chunk))
            cursor = conn.execute(
                "select id, N, Z, pos, Ham from data "
                f"where id in ({placeholders}) order by id", chunk)
            for row in cursor:
                yield _record_from_row(row)
                n_yielded += 1
                if limit is not None and n_yielded >= limit:
                    return


# --- Empirical unit detection ------------------------------------------------

class UnitDetectionError(RuntimeError):
    """Neither (or both) coordinate-unit hypotheses produced a physical spectrum."""


# Physicality window for eigh(F, S) spectra of neutral closed-shell CHONF
# molecules with B3LYP Kohn-Sham Fock matrices. The wrong unit yields wildly
# unphysical values, so the classification is unambiguous. Note: the core
# window is calibrated for 1s levels of C/N/O (F 1s sits near -24 Ha and may
# need a wider window if fluorine-rich records ever fail detection).
_CORE_WINDOW = (-21.0, -9.0)
_HOMO_WINDOW = (-0.5, -0.1)
_GAP_WINDOW = (0.02, 0.6)


def _spectrum_is_physical(record: MoleculeRecord, unit: str) -> bool:
    from qthermal.orbitals import build_mol  # local import: orbitals never imports loader

    try:
        mol = build_mol(record, unit)
    except Exception as exc:  # e.g. nuclei overlapping under the wrong unit
        logger.debug("build_mol failed for unit=%s: %s", unit, exc)
        return False
    S = mol.intor("int1e_ovlp")
    if S.shape[0] != record.F.shape[0]:
        raise ValueError(
            f"Fock dimension {record.F.shape[0]} != def2-SVP nao {S.shape[0]}; "
            "basis or AO-ordering mismatch")
    try:
        eps = scipy.linalg.eigh(record.F, S, eigvals_only=True)
    except scipy.linalg.LinAlgError:
        return False
    nocc = mol.nelectron // 2
    lowest = float(eps[0])
    homo = float(eps[nocc - 1])
    gap = float(eps[nocc] - eps[nocc - 1])
    ok = (_CORE_WINDOW[0] <= lowest <= _CORE_WINDOW[1]
          and _HOMO_WINDOW[0] <= homo <= _HOMO_WINDOW[1]
          and _GAP_WINDOW[0] <= gap <= _GAP_WINDOW[1])
    logger.debug("unit=%s lowest=%.3f homo=%.3f gap=%.3f -> %s",
                 unit, lowest, homo, gap, ok)
    return ok


_ANGSTROM_PER_BOHR = 0.529177210903
# Shortest interatomic distance window (Angstrom) for covalent CHONF
# molecules: X-H bonds ~0.96-1.12 A, multiple heavy-heavy bonds >= ~1.1 A.
_MIN_DIST_WINDOW = (0.7, 1.7)


def _min_distance_angstrom(record: MoleculeRecord, unit: str) -> float:
    R = record.R if unit == "Angstrom" else record.R * _ANGSTROM_PER_BOHR
    diff = R[:, None, :] - R[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    return float(d[np.triu_indices(len(R), k=1)].min())


def detect_units(record: MoleculeRecord) -> str:
    """Return 'Angstrom' or 'Bohr' by testing which yields a physical spectrum.

    Builds the PySCF molecule under both unit hypotheses, solves the
    generalized eigenproblem eigh(F, S), and accepts the unit whose spectrum
    has a 1s core level in [-21, -9] Ha, HOMO in [-0.5, -0.1] Ha, and a
    HOMO-LUMO gap in [0.02, 0.6] Ha.

    The spectral test is one-sided in practice: misreading Angstrom as Bohr
    compresses the molecule 1.89x and the core levels collapse (unambiguous
    failure), but misreading Bohr as Angstrom merely stretches it and the
    eigh(F, S) spectrum can stay inside all three windows. When both
    hypotheses pass spectrally, a geometric tiebreak picks the hypothesis
    whose shortest interatomic distance falls in the covalent-bond window
    [0.7, 1.7] Angstrom.
    """
    verdict = {u: _spectrum_is_physical(record, u) for u in ("Angstrom", "Bohr")}
    passing = [u for u, ok in verdict.items() if ok]
    if len(passing) == 1:
        return passing[0]
    if len(passing) == 2:
        lo, hi = _MIN_DIST_WINDOW
        geometric = [u for u in passing
                     if lo <= _min_distance_angstrom(record, u) <= hi]
        if len(geometric) == 1:
            logger.info("Unit detection tiebreak via bond lengths: %s "
                        "(record idx=%d)", geometric[0], record.idx)
            return geometric[0]
    raise UnitDetectionError(
        f"Unit detection ambiguous for record idx={record.idx}: "
        f"spectral verdict {verdict}, min distances "
        f"{ {u: round(_min_distance_angstrom(record, u), 3) for u in ('Angstrom', 'Bohr')} }")


class CachedUnitDetector:
    """Detect units on the first record, cache, re-verify every Nth record."""

    def __init__(self, recheck_every: int = 100):
        self.recheck_every = int(recheck_every)
        self._unit: str | None = None

    def unit_for(self, record: MoleculeRecord, ordinal: int) -> str:
        if self._unit is None or ordinal % self.recheck_every == 0:
            unit = detect_units(record)
            if self._unit is None:
                self._unit = unit
                logger.info("Coordinate unit detected: %s (record idx=%d)",
                            unit, record.idx)
            elif unit != self._unit:
                raise UnitDetectionError(
                    f"Unit flipped from {self._unit} to {unit} at ordinal "
                    f"{ordinal} (record idx={record.idx})")
        return self._unit
