"""Module A tests: MoleculeRecord round-trip, QH9 SQLite adapter, detect_units."""

import sqlite3

import numpy as np
import pytest

from qthermal.loader import (
    CachedUnitDetector,
    MoleculeRecord,
    UnitDetectionError,
    detect_units,
    iter_records,
    qh9_to_pyscf_transform,
)

ANGSTROM_PER_BOHR = 0.529177210903


def test_record_roundtrip(h2o_record):
    r = h2o_record
    r2 = MoleculeRecord(idx=r.idx, Z=r.Z.tolist(), R=r.R.tolist(),
                        F=r.F, C=r.C, eps=r.eps)
    assert r2.idx == 0
    assert r2.Z.dtype == np.int64
    for a, b in ((r2.Z, r.Z), (r2.R, r.R), (r2.F, r.F), (r2.C, r.C), (r2.eps, r.eps)):
        np.testing.assert_array_equal(np.asarray(a, dtype=np.float64),
                                      np.asarray(b, dtype=np.float64))
    assert r2.F.dtype == np.float64 and r2.R.dtype == np.float64


def test_record_rejects_asymmetric_fock(h2o_record):
    F_bad = h2o_record.F.copy()
    F_bad[0, 1] += 1e-6
    with pytest.raises(AssertionError):
        MoleculeRecord(idx=1, Z=h2o_record.Z, R=h2o_record.R, F=F_bad)


def test_detect_units_angstrom(h2o_record):
    assert detect_units(h2o_record) == "Angstrom"


def test_detect_units_bohr(h2o_record):
    r = h2o_record
    bohr = MoleculeRecord(idx=0, Z=r.Z, R=r.R / ANGSTROM_PER_BOHR, F=r.F)
    assert detect_units(bohr) == "Bohr"


def test_detect_units_rejects_garbage(h2o_record):
    r = h2o_record
    silly = MoleculeRecord(idx=0, Z=r.Z, R=r.R * 40.0, F=r.F)
    with pytest.raises(UnitDetectionError):
        detect_units(silly)


def test_cached_unit_detector(h2o_record):
    det = CachedUnitDetector(recheck_every=100)
    assert det.unit_for(h2o_record, ordinal=0) == "Angstrom"
    # Ordinals 1..99 must not re-run detection; feed a record that would
    # fail detection to prove the cache is used.
    r = h2o_record
    garbage = MoleculeRecord(idx=1, Z=r.Z, R=r.R * 40.0, F=r.F)
    assert det.unit_for(garbage, ordinal=1) == "Angstrom"
    # Ordinal 100 re-runs detection; an inconsistent record must raise.
    bohr = MoleculeRecord(idx=2, Z=r.Z, R=r.R / ANGSTROM_PER_BOHR, F=r.F)
    with pytest.raises(UnitDetectionError):
        det.unit_for(bohr, ordinal=100)


def _make_synthetic_qh9_db(path, record):
    """Write a one-row SQLite DB in the raw QH9Stable schema.

    The stored Hamiltonian must be in QH9's native AO ordering, so the
    PySCF-ordered Fock matrix is inverse-transformed before storage.
    """
    idx_map, signs = qh9_to_pyscf_transform(record.Z)
    inv = np.argsort(idx_map)
    F_signed = record.F * signs[:, None] * signs[None, :]
    F_qh9 = F_signed[np.ix_(inv, inv)]
    # Round-trip sanity for the test helper itself.
    F_back = F_qh9[np.ix_(idx_map, idx_map)] * signs[:, None] * signs[None, :]
    np.testing.assert_allclose(F_back, record.F, atol=1e-14)

    with sqlite3.connect(path) as conn:
        conn.execute("create table data (id integer primary key, N integer, "
                     "Z blob, pos blob, Ham blob)")
        conn.execute(
            "insert into data (id, N, Z, pos, Ham) values (?, ?, ?, ?, ?)",
            (0, len(record.Z),
             record.Z.astype(np.int32).tobytes(),
             record.R.astype(np.float64).tobytes(),
             F_qh9.astype(np.float64).tobytes()))
        conn.commit()


def test_iter_records_sqlite_adapter(tmp_path, h2o_record):
    db = tmp_path / "QH9Stable.db"
    _make_synthetic_qh9_db(db, h2o_record)

    records = list(iter_records(db, limit=5))
    assert len(records) == 1
    rec = records[0]
    assert rec.idx == 0
    np.testing.assert_array_equal(rec.Z, h2o_record.Z)
    np.testing.assert_allclose(rec.R, h2o_record.R, atol=1e-14)
    np.testing.assert_allclose(rec.F, h2o_record.F, atol=1e-12)
    assert rec.C is None and rec.eps is None
    assert rec.F.dtype == np.float64


def test_iter_records_limit(tmp_path, h2o_record):
    db = tmp_path / "QH9Stable.db"
    _make_synthetic_qh9_db(db, h2o_record)
    assert list(iter_records(db, limit=0)) == []
