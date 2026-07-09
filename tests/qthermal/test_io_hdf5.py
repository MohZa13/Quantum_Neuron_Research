"""Module G tests: layout, dtypes, resume-safety, kT tags."""

import h5py
import numpy as np
import pytest

from qthermal.io_hdf5 import RunWriter, kT_tag
from qthermal.thermal import ThermalBlock


def _fake_block(kT, m=4, dim=36, ncas=4):
    rng = np.random.default_rng(int(kT * 1000))
    p = rng.random(m); p /= p.sum() * 1.000001
    return ThermalBlock(
        kT=kT, E=np.sort(rng.standard_normal(m)), p=p,
        civecs=rng.standard_normal((m, dim)),
        truncation_error=1e-7, entropy=0.3,
        nat_occs=np.linspace(2.0, 0.0, ncas), static_corr=0.1,
        c_max_sq=0.9, tracedist_gaussian=0.05, tracedist_bound=1e-7,
        cap_hit=False)


META = {"basis": "def2-svp", "n_act_occ": 2, "n_act_virt": 2, "ncas": 4,
        "nelecas": 4, "solver_name": "dense_ed", "kT_list": [0.05, 0.25],
        "kT_convention": "absolute_hartree", "code_version": "0.1.0",
        "pyscf_version": "2.13.1", "unit": "Angstrom"}


def _write_one(writer, idx=7, evals=None):
    return writer.write_molecule(
        idx, Z=[8, 1, 1], R=np.zeros((3, 3)), active_idx=np.arange(2, 6),
        nocc=5, ecore=-52.5, h1eff=np.eye(4), g=np.zeros((4, 4, 4, 4)),
        evals_full=evals, blocks=[_fake_block(0.05), _fake_block(0.25)])


def test_kT_tag_stability():
    assert kT_tag(0.05) == "kT_0p0500"
    assert kT_tag(0.1) == "kT_0p1000"
    assert kT_tag(0.25) == kT_tag(0.2500000001) == "kT_0p2500"


def test_write_and_read_roundtrip(tmp_path):
    path = tmp_path / "run.h5"
    with RunWriter(path, META) as w:
        assert _write_one(w, evals=np.arange(36.0))

    with h5py.File(path, "r") as f:
        assert f["meta"].attrs["basis"] == "def2-svp"
        np.testing.assert_allclose(f["meta"].attrs["kT_list"], [0.05, 0.25])
        g = f["mol_7"]
        assert g.attrs["complete"]
        assert g["Z"].dtype == np.int64
        for name in ("R", "h1eff", "g", "evals", "ecore"):
            assert g[name].dtype == np.float64, name
        assert g["ecore"][()] == -52.5
        blk = g[kT_tag(0.05)]
        assert blk["civecs"].dtype == np.float64
        assert blk["civecs"].shape == (4, 36)
        assert blk["civecs"].compression == "gzip"
        assert blk["truncation_error"][()] == pytest.approx(1e-7)
        assert not blk.attrs["cap_hit"]


def test_evals_optional(tmp_path):
    path = tmp_path / "run.h5"
    with RunWriter(path, META) as w:
        _write_one(w, evals=None)
    with h5py.File(path, "r") as f:
        assert "evals" not in f["mol_7"]     # readers must not rely on it


def test_resume_skips_complete(tmp_path):
    path = tmp_path / "run.h5"
    with RunWriter(path, META) as w:
        assert _write_one(w)
    with RunWriter(path, META) as w:
        assert w.is_complete(7)
        assert not _write_one(w)             # second write is a no-op skip
        assert not w.is_complete(8)


def test_incomplete_group_rewritten(tmp_path):
    path = tmp_path / "run.h5"
    with RunWriter(path, META) as w:
        _write_one(w)
    with h5py.File(path, "a") as f:          # simulate a crash mid-write
        del f["mol_7"].attrs["complete"]
        del f["mol_7"]["h1eff"]
    with RunWriter(path, META) as w:
        assert not w.is_complete(7)
        assert _write_one(w)
        assert w.is_complete(7)
    with h5py.File(path, "r") as f:
        np.testing.assert_allclose(f["mol_7"]["h1eff"][:], np.eye(4))


def test_meta_written_once(tmp_path):
    path = tmp_path / "run.h5"
    with RunWriter(path, META) as w:
        pass
    meta2 = dict(META, basis="OVERWRITE-ATTEMPT")
    with RunWriter(path, meta2) as w:
        assert w.meta["basis"] == "def2-svp"
