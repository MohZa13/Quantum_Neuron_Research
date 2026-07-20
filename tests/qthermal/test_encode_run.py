"""Module I CLI tests: Module-H run file -> extended-Heisenberg mapping file."""

import h5py
import numpy as np
import pytest

from qthermal.encode import (
    extended_heisenberg_expectations,
    extended_heisenberg_labels,
    taper_extended_heisenberg,
)
from qthermal.encode_run import main as encode_main
from qthermal.run import main as run_main

from tests.qthermal.test_loader import _make_synthetic_qh9_db

pytest.importorskip("pennylane")

NCAS = 6                                   # 3 occ + 3 virt, dim 400
N_TERMS = 4 * NCAS ** 2 - NCAS


@pytest.fixture(scope="module")
def run_file(tmp_path_factory, h2o_record):
    tmp = tmp_path_factory.mktemp("encode_run")
    db = tmp / "QH9Stable.db"
    _make_synthetic_qh9_db(db, h2o_record)
    out = tmp / "run.h5"
    assert run_main(["--qh9-path", str(db), "--out", str(out), "--limit", "1",
                     "--n-act-occ", "3", "--n-act-virt", "3",
                     "--kT-list", "0.05,0.25", "--log-level", "WARNING"]) == 0
    return out


def test_end_to_end_coefficients(run_file, tmp_path):
    out = tmp_path / "extheis.h5"
    assert encode_main(["--in", str(run_file), "--out", str(out),
                        "--log-level", "WARNING"]) == 0
    with h5py.File(out, "r") as f, h5py.File(run_file, "r") as src:
        meta = dict(f["meta"].attrs)
        assert meta["ansatz"] == "extended_heisenberg"
        assert meta["ordering"] == "blocked"
        assert meta["n_terms"] == N_TERMS
        assert meta["nalpha"] == meta["nbeta"] == 3
        labels = [s.decode() for s in f["pauli_labels"][:]]
        assert labels == extended_heisenberg_labels(NCAS)

        g = f["mol_0"]
        assert g.attrs["complete"]
        for tag in ("kT_0p0500", "kT_0p2500"):
            blk, ref = g[tag], src["mol_0"][tag]
            coeffs = blk["coeffs"][:]
            assert coeffs.shape == (N_TERMS,)
            expected = extended_heisenberg_expectations(
                ref["civecs"][:], ref["p"][:], NCAS, 3, 3)
            np.testing.assert_allclose(coeffs, expected, atol=1e-14)
            assert blk.attrs["trace"] == pytest.approx(ref["p"][:].sum())


def test_resume_skips_complete(run_file, tmp_path):
    out = tmp_path / "extheis.h5"
    assert encode_main(["--in", str(run_file), "--out", str(out),
                        "--log-level", "WARNING"]) == 0
    with h5py.File(out, "r") as f:
        raw = f["mol_0"]["kT_0p0500"]["coeffs"][:]
    assert encode_main(["--in", str(run_file), "--out", str(out),
                        "--log-level", "WARNING"]) == 0
    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(f["mol_0"]["kT_0p0500"]["coeffs"][:],
                                      raw)


def test_taper_datasets(run_file, tmp_path):
    """--taper stores the relabeled basis; adding it to an existing file
    fills in the taper datasets without touching molecule groups."""
    out = tmp_path / "extheis.h5"
    assert encode_main(["--in", str(run_file), "--out", str(out),
                        "--log-level", "WARNING"]) == 0
    with h5py.File(out, "r") as f:
        assert "pauli_labels_tapered" not in f

    assert encode_main(["--in", str(run_file), "--out", str(out), "--taper",
                        "--log-level", "WARNING"]) == 0
    ref_labels, ref_signs, ref_kept = taper_extended_heisenberg(NCAS, 3, 3)
    with h5py.File(out, "r") as f:
        assert f["meta"].attrs["n_qubits_tapered"] == 2 * NCAS - 2
        # nalpha = nbeta = 3: both block parities are odd
        np.testing.assert_array_equal(f["meta"].attrs["taper_sector"],
                                      [-1, -1])
        assert [s.decode() for s in f["pauli_labels_tapered"][:]] == ref_labels
        np.testing.assert_array_equal(f["taper_signs"][:], ref_signs)
        np.testing.assert_array_equal(f["taper_kept_wires"][:], ref_kept)
        assert f["mol_0"].attrs["complete"]          # untouched by the rerun
        tapered_wires = {int(t[1:]) for lbl in ref_labels
                         for t in lbl.split()}
        assert tapered_wires <= set(range(2 * NCAS - 2))
