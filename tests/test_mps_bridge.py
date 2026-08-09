"""The Julia->Python bridge: MPS-produced thermal states into qnn.

These tests pin the register conventions the bridge depends on — the exact
place a silent bug would live (a mis-oriented bit order produces states that
train fine and mean nothing).  They need the local ncas=10 artifact, which is
gitignored; without it they skip.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MPS_H5 = ROOT / "results" / "qh9_mps_ncas10.h5"

pytestmark = pytest.mark.skipif(
    not MPS_H5.exists(), reason="results/qh9_mps_ncas10.h5 not generated locally")


@pytest.fixture(scope="module")
def bridge():
    spec = importlib.util.spec_from_file_location(
        "train_mps_thermal", ROOT / "scripts" / "train_mps_thermal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def loaded(bridge):
    return bridge.load_states(str(MPS_H5), ["mol_3"])


def test_bitrev_perm_is_involution(bridge):
    p = bridge.bitrev_perm(10)
    assert np.array_equal(p[p], np.arange(1024))
    assert p[1] == 512 and p[512] == 1          # LSB <-> MSB
    assert p[0] == 0 and p[1023] == 1023


def test_states_are_unit_trace_symmetric_psd(loaded):
    rho, records, meta = loaded
    assert rho.shape[1:] == (1024, 1024)
    for m in range(rho.shape[0]):
        a = rho[m]
        assert abs(np.trace(a) - 1.0) < 1e-8
        assert np.abs(a - a.T).max() < 1e-10
        assert np.linalg.eigvalsh(a).min() > -1e-8


def test_alpha_sector_is_sharp(loaded):
    # N_alpha = 5 exactly: every state's diagonal lives on popcount-5 rows,
    # and total occupation is 5 electrons.  Both are invariant under any bit
    # permutation, so they test the export, not the loader's orientation.
    rho, records, _ = loaded
    pc = np.array([bin(i).count("1") for i in range(1024)])
    for m in range(rho.shape[0]):
        d = np.diag(rho[m])
        assert d[pc != 5].sum() < 1e-9
        assert abs(d[pc == 5].sum() - 1.0) < 1e-8
    for r in records:
        assert abs(r["nelec_from_diag"] - 5.0) < 1e-6


def test_orientation_hf_string(loaded):
    # The orientation razor: after the little-endian conversion the coldest
    # state's dominant determinant must be orbitals 0-4 occupied = row 31.
    # A loader that forgot the bit reversal would put it at 992 instead.
    rho, records, _ = loaded
    cold = min(range(len(records)), key=lambda i: records[i]["kT"])
    assert records[cold]["kT"] == pytest.approx(0.1)
    assert int(np.argmax(np.diag(rho[cold]))) == 31


def test_hot_state_is_near_uniform_on_sector(loaded):
    # At kT = 4 (beta = 0.25) the state is close to P/dim: all 252 sector
    # rows populated, spread bounded by e^{+-beta*spread}.  A state
    # accidentally re-diagonalized or projected would fail the count.
    rho, records, _ = loaded
    hot = max(range(len(records)), key=lambda i: records[i]["kT"])
    assert records[hot]["kT"] == pytest.approx(4.0)
    pc = np.array([bin(i).count("1") for i in range(1024)])
    d = np.diag(rho[hot])[pc == 5]
    assert d.shape[0] == 252
    assert d.min() > 0.1 / 252
    assert d.max() < 10.0 / 252


def test_projection_is_lossless_here(bridge, loaded):
    # Only C(10,5) = 252 of 1024 rows are populated, so keeping the top 256
    # loses nothing: the projected stack IS the alpha-sector restriction.
    rho, _, _ = loaded
    _, k, retention = bridge.project_register(rho, 8)
    assert k == 8
    assert retention["population_kept"] > 1 - 1e-9
    assert retention["offdiag_frobenius_kept"] > 1 - 1e-6


def test_training_smoke(bridge, loaded):
    # End to end: hotcold on the projected register must train well below its
    # initial loss within a few dozen epochs.  This is the "does it train"
    # bit executed as a test, small enough to run every time.
    from qnn import HybridNetwork, StateBatch, build_pool
    rho, records, _ = loaded
    rho_p, n, _ = bridge.project_register(rho, 8)
    batch = StateBatch(rho_p, normalise=True)
    y = np.where([r["kT"] >= 1.0 for r in records], 1.0, -1.0)
    net = HybridNetwork(build_pool(n, "quantum"), n_quantum=4, hidden=(8,),
                        activation="tanh", loss="logistic", seed=11)
    h = net.fit(batch, y, epochs=60, lr=0.05)
    assert h.loss_tr[-1] < 0.2 * h.loss_tr[0]
    assert h.acc_tr[-1] == 1.0
