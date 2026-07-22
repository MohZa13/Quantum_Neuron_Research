"""Module J (mps) tests: the purification MPS reproduces the thermal state.

The gold-standard check diagonalises a random CAS(4,4) sector with the dense
solver, builds the purification MPS from the eigenblock, traces out the ancilla,
and requires the result to equal the density matrix built independently from the
JW-encoded vectors (:func:`thermal_density_matrix`) — exactly when untruncated,
and within the certified ``truncation_error`` bound when bonds are capped.
"""

import numpy as np
import pytest

from qthermal.active_space import ActiveSpace
from qthermal.diagonalize import DenseEDSolver
from qthermal.encode import encode_jw, thermal_density_matrix
from qthermal.mps import (purification_mps, reduced_density_matrix, to_dense_ket,
                          tt_svd)
from qthermal.thermal import boltzmann_weights

ORDERINGS = ["blocked", "interleaved"]


def _random_cas44(seed=7):
    rng = np.random.default_rng(seed)
    n = 4
    h1 = rng.normal(size=(n, n))
    h1 = 0.5 * (h1 + h1.T)
    g = rng.normal(size=(n,) * 4)
    g = g + g.transpose(1, 0, 2, 3)
    g = g + g.transpose(0, 1, 3, 2)
    g = g + g.transpose(2, 3, 0, 1)              # chemist (pq|rs) 8-fold symmetry
    aspace = ActiveSpace(active_idx=np.arange(4, dtype=np.int64),
                         core_idx=np.arange(0, dtype=np.int64),
                         n_act_occ=2, n_act_virt=2)
    return h1, g / 8.0, aspace


@pytest.fixture(scope="module")
def cas44_block():
    """(civecs, p) for a random CAS(4,4) thermal state; full spectrum kept."""
    h1, g, aspace = _random_cas44()
    ens = DenseEDSolver().solve(h1, g, aspace, kT_max=5.0)
    p = boltzmann_weights(ens.E, kT=0.5)         # full spectrum -> exact, normalised
    return ens.vecs, p, aspace


def test_tt_svd_reconstructs_random_tensor():
    rng = np.random.default_rng(0)
    tensor = rng.normal(size=(1, 2, 3, 2, 4))    # leading axis = trivial left bond
    cores, err = tt_svd(tensor)
    assert err == 0.0
    psi = cores[0]
    for core in cores[1:]:
        psi = np.tensordot(psi, core, axes=([-1], [0]))
    np.testing.assert_allclose(psi.reshape(tensor.shape), tensor, atol=1e-12)


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_exact_purification_reconstructs_rho(cas44_block, ordering):
    civecs, p, _ = cas44_block
    mps = purification_mps(civecs, p, ncas=4, nalpha=2, nbeta=2, ordering=ordering)

    rho_mps = reduced_density_matrix(mps)
    rho_ref = thermal_density_matrix(encode_jw(civecs, 4, 2, 2, ordering), p)

    assert mps.truncation_error == 0.0
    np.testing.assert_allclose(rho_mps, rho_ref, atol=1e-10)


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_ancilla_bond_is_thermal_rank_and_norm_is_trace(cas44_block, ordering):
    civecs, p, _ = cas44_block
    mps = purification_mps(civecs, p, ncas=4, nalpha=2, nbeta=2, ordering=ordering)

    assert mps.ancilla_bond() == len(p)
    assert len(mps.cores) == 2 * 4 + 1
    ket = to_dense_ket(mps)
    np.testing.assert_allclose(float(np.vdot(ket, ket)), p.sum(), atol=1e-12)


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_rho_eigenvalues_are_boltzmann_weights(cas44_block, ordering):
    """rho = sum_k p_k |orthonormal_k><.| so its spectrum is exactly {p_k}."""
    civecs, p, _ = cas44_block
    mps = purification_mps(civecs, p, ncas=4, nalpha=2, nbeta=2, ordering=ordering)

    eig = np.linalg.eigvalsh(reduced_density_matrix(mps))
    nonzero = np.sort(eig[eig > 1e-9])[::-1]
    np.testing.assert_allclose(nonzero, np.sort(p[p > 1e-9])[::-1], atol=1e-9)


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_truncation_error_bounds_trace_norm(cas44_block, ordering):
    civecs, p, _ = cas44_block
    rho_ref = thermal_density_matrix(encode_jw(civecs, 4, 2, 2, ordering), p)

    mps = purification_mps(civecs, p, ncas=4, nalpha=2, nbeta=2,
                           ordering=ordering, chi_max=3)
    rho_t = reduced_density_matrix(mps)

    trace_norm = float(np.abs(np.linalg.eigvalsh(rho_t - rho_ref)).sum())
    assert mps.truncation_error > 0.0                       # cap actually bit
    # partial trace is a trace-norm contraction: ||rho - rho_t||_1 <= 2||dPsi||_2
    assert trace_norm <= 2.0 * mps.truncation_error + 1e-9


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_dim_mismatch_raises(cas44_block, ordering):
    civecs, p, _ = cas44_block
    with pytest.raises(ValueError, match="sector expects"):
        purification_mps(civecs[:, :-1], p, ncas=4, nalpha=2, nbeta=2,
                         ordering=ordering)
