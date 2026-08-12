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
    assert args.keep_cap is None


def test_end_to_end_single_molecule(synthetic_db, tmp_path):
    out = tmp_path / "run.h5"
    assert main(_cli(synthetic_db, out)) == 0

    with h5py.File(out, "r") as f:
        meta = dict(f["meta"].attrs)
        assert meta["unit"] == "Angstrom"          # detected, not assumed
        assert meta["ncas"] == 6 and meta["nelecas"] == 6
        assert meta["solver_name"] == "dense_ed"
        assert meta["kT_convention"] == "absolute_hartree"
        assert meta["keep_cap"] == -1             # default max(1024, dim // 4)

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


def test_negative_keep_cap_rejected(synthetic_db, tmp_path):
    assert main(_cli(synthetic_db, tmp_path / "z.h5",
                     extra=["--keep-cap", "-3"])) == 2


def test_keep_cap_end_to_end(synthetic_db, tmp_path):
    """--keep-cap N caps stored civecs per kT block (cap_hit, honest
    truncation_error, full spectrum untouched); --keep-cap 0 lifts the cap so
    the weight cutoff is honored."""
    out = tmp_path / "cap.h5"
    assert main(_cli(synthetic_db, out, extra=["--keep-cap", "2"])) == 0
    with h5py.File(out, "r") as f:
        assert f["meta"].attrs["keep_cap"] == 2
        assert f["mol_0"]["evals"].shape == (400,)     # spectrum stays exact
        blk = f["mol_0"][kT_tag(0.25)]
        assert blk["civecs"].shape == (2, 400)
        assert blk.attrs["cap_hit"]
        assert blk["truncation_error"][()] > 1e-6

    out0 = tmp_path / "nocap.h5"
    assert main(_cli(synthetic_db, out0, extra=["--keep-cap", "0"])) == 0
    with h5py.File(out0, "r") as f:
        assert f["meta"].attrs["keep_cap"] == 0
        blk = f["mol_0"][kT_tag(0.25)]
        assert not blk.attrs["cap_hit"]
        assert blk["truncation_error"][()] <= 1e-6 * 1.001


def test_workers_pool(synthetic_db, tmp_path):
    """The spawn-based Pool path produces the same output file."""
    out = tmp_path / "mp.h5"
    assert main(_cli(synthetic_db, out, extra=["--workers", "2"])) == 0
    with h5py.File(out, "r") as f:
        assert f["mol_0"].attrs["complete"]


def test_iterative_solver_end_to_end(synthetic_db, tmp_path):
    """--solver iterative writes a complete group with no full spectrum."""
    out = tmp_path / "it.h5"
    assert main(_cli(synthetic_db, out,
                     extra=["--solver", "iterative",
                            "--kT-list", "0.025,0.05"])) == 0
    with h5py.File(out, "r") as f:
        assert f["meta"].attrs["solver_name"] == "iterative_krylov"
        g = f["mol_0"]
        assert g.attrs["complete"]
        assert "evals" not in g            # Krylov never promises the spectrum
        for kT in (0.025, 0.05):
            blk = g[kT_tag(kT)]
            m = blk["E"].shape[0]
            assert blk["civecs"].shape == (m, 400)
            assert blk["truncation_error"][()] < 1e-5
            assert 0.0 <= blk["tracedist_gaussian"][()] <= 1.0 + 1e-10


def test_iterative_matches_dense_end_to_end(synthetic_db, tmp_path):
    """Stored per-kT diagnostics agree between the two solvers."""
    dense_out, it_out = tmp_path / "d.h5", tmp_path / "i.h5"
    kts = ["--kT-list", "0.025,0.05"]
    assert main(_cli(synthetic_db, dense_out, extra=kts)) == 0
    assert main(_cli(synthetic_db, it_out,
                     extra=["--solver", "iterative", *kts])) == 0
    with h5py.File(dense_out, "r") as fd, h5py.File(it_out, "r") as fi:
        for kT in (0.025, 0.05):
            bd, bi = fd["mol_0"][kT_tag(kT)], fi["mol_0"][kT_tag(kT)]
            n = min(bd["E"].shape[0], bi["E"].shape[0])
            np.testing.assert_allclose(bi["E"][:n], bd["E"][:n], atol=1e-7)
            # kept sets may differ by cutoff-level states (see the parity
            # test in test_diagonalize) — tolerances are O(cutoff), not 0.
            np.testing.assert_allclose(bi["nat_occs"][:], bd["nat_occs"][:],
                                       atol=1e-5)
            assert abs(bi["entropy"][()] - bd["entropy"][()]) < 5e-5
            assert abs(bi["tracedist_gaussian"][()]
                       - bd["tracedist_gaussian"][()]) < 1e-5


def test_relative_kT_requires_dense(synthetic_db, tmp_path):
    assert main(_cli(synthetic_db, tmp_path / "rel.h5",
                     extra=["--solver", "iterative", "--kT-relative"])) == 2


def test_max_nroots_parser_default():
    args = build_parser().parse_args(["--qh9-path", "x.db", "--out", "y.h5"])
    assert args.max_nroots == 512


def test_parse_indices_forms():
    from qthermal.run import parse_indices
    assert parse_indices(None) is None
    assert parse_indices("3,1,2") == [1, 2, 3]
    assert parse_indices("0-4") == [0, 1, 2, 3, 4]
    assert parse_indices("7, 0-2 ,7") == [0, 1, 2, 7]


def test_parse_indices_from_file(tmp_path):
    from qthermal.run import parse_indices
    path = tmp_path / "ids.txt"
    path.write_text("5\n3\n9,11\n")
    assert parse_indices(f"@{path}") == [3, 5, 9, 11]
