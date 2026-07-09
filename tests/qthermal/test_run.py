"""Module H tests: end-to-end CLI on a synthetic single-molecule QH9 database."""

import h5py
import numpy as np
import pytest

from qthermal.io_hdf5 import kT_tag
from qthermal.run import build_parser, main

from tests.qthermal.test_loader import _make_synthetic_qh9_db


@pytest.fixture()
def synthetic_db(tmp_path, h2o_record):
    db = tmp_path / "QH9Stable.db"
    _make_synthetic_qh9_db(db, h2o_record)
    return db


def _cli(db, out, extra=()):
    return ["--qh9-path", str(db), "--out", str(out), "--limit", "5",
            "--n-act-occ", "3", "--n-act-virt", "3",
            "--kT-list", "0.05,0.25", "--log-level", "WARNING", *extra]


def test_parser_defaults():
    args = build_parser().parse_args(["--qh9-path", "x.db", "--out", "y.h5"])
    assert args.limit == 100
    assert args.n_act_occ == 4 and args.n_act_virt == 4
    assert args.solver == "dense" and not args.allow_large_dense
    assert args.kT_list == "0.05,0.1,0.25" and not args.kT_relative
    assert args.workers == 1


def test_end_to_end_single_molecule(synthetic_db, tmp_path):
    out = tmp_path / "run.h5"
    assert main(_cli(synthetic_db, out)) == 0

    with h5py.File(out, "r") as f:
        meta = dict(f["meta"].attrs)
        assert meta["unit"] == "Angstrom"          # detected, not assumed
        assert meta["ncas"] == 6 and meta["nelecas"] == 6
        assert meta["solver_name"] == "dense_ed"
        assert meta["kT_convention"] == "absolute_hartree"

        g = f["mol_0"]
        assert g.attrs["complete"]
        np.testing.assert_array_equal(g["Z"][:], [8, 1, 1])
        np.testing.assert_array_equal(g["active_idx"][:], np.arange(2, 8))
        assert g["evals"].shape == (400,)
        assert g["g"].shape == (6, 6, 6, 6)
        for kT in (0.05, 0.25):
            blk = g[kT_tag(kT)]
            m = blk["E"].shape[0]
            assert blk["civecs"].shape == (m, 400)
            assert 0.0 <= blk["tracedist_gaussian"][()] <= 1.0 + 1e-10
        # Ground state must match the stored spectrum's bottom.
        assert g[kT_tag(0.05)]["E"][0] == pytest.approx(g["evals"][0])


def test_resume_is_noop(synthetic_db, tmp_path):
    out = tmp_path / "run.h5"
    assert main(_cli(synthetic_db, out)) == 0
    with h5py.File(out, "r") as f:
        raw = f["mol_0"]["h1eff"][:]
    assert main(_cli(synthetic_db, out)) == 0      # resume: skips complete group
    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(f["mol_0"]["h1eff"][:], raw)


def test_relative_kT_convention(synthetic_db, tmp_path):
    out = tmp_path / "rel.h5"
    assert main(_cli(synthetic_db, out,
                     extra=["--kT-relative", "--kT-list", "0.002,0.01"])) == 0
    with h5py.File(out, "r") as f:
        assert f["meta"].attrs["kT_convention"] == "relative_spectral_width"
        width = f["mol_0"]["evals"][-1] - f["mol_0"]["evals"][0]
        assert kT_tag(0.002 * width) in f["mol_0"]


def test_bad_kT_list_rejected(synthetic_db, tmp_path):
    assert main(_cli(synthetic_db, tmp_path / "z.h5",
                     extra=["--kT-list", "-0.1"])) == 2


def test_workers_pool(synthetic_db, tmp_path):
    """The spawn-based Pool path produces the same output file."""
    out = tmp_path / "mp.h5"
    assert main(_cli(synthetic_db, out, extra=["--workers", "2"])) == 0
    with h5py.File(out, "r") as f:
        assert f["mol_0"].attrs["complete"]
