"""Module F tests: weights, truncation, diagnostics, and the Gaussian audit."""

import numpy as np
import pytest

from qthermal.active_space import select_active
from qthermal.diagonalize import DenseEDSolver
from qthermal.hamiltonian import build_cas_hamiltonian
from qthermal.orbitals import build_mol, orbitals, overlap
from qthermal.thermal import (
    boltzmann_weights,
    build_thermal_block,
    ensemble_entropy,
    gaussian_reference_ensemble,
    leading_determinant_weight,
    natural_occupations,
    resolve_kT_list,
    static_correlation_score,
    thermal_rdm1,
    trace_distance_projected,
    truncate_prefix,
)

KT_MAX = 0.25
CUTOFF = 1e-6


# --- pure-function tests -----------------------------------------------------

def test_boltzmann_weights_stability():
    """Raw energies of ~ -1e6 Ha must not overflow: shifted softmax only."""
    evals = np.array([-1.0e6, -1.0e6 + 0.05, -1.0e6 + 5.0])
    p = boltzmann_weights(evals, kT=0.05)
    assert np.all(np.isfinite(p)) and abs(p.sum() - 1.0) < 1e-12
    # rtol limited by float64 spacing (~1e-10) of the huge raw energies.
    np.testing.assert_allclose(p[1] / p[0], np.exp(-1.0), rtol=1e-8)


def test_boltzmann_weights_rejects_nonpositive_kT():
    with pytest.raises(ValueError):
        boltzmann_weights(np.array([0.0, 1.0]), kT=0.0)


def test_truncate_prefix_analytic():
    evals = np.array([0.0, 1.0, 2.0, 50.0])
    m, p, tail, capped = truncate_prefix(evals, kT=0.5, weight_cutoff=1e-6)
    # exp(-100) is far below the cutoff, exp(-4) is not.
    assert m == 3 and not capped
    assert abs(tail - p[3]) < 1e-15 and tail < 1e-6


def test_truncate_prefix_cap_binds():
    evals = np.linspace(0.0, 0.1, 100)   # nearly uniform weights at high kT
    m, p, tail, capped = truncate_prefix(evals, kT=10.0, weight_cutoff=1e-6,
                                         cap=10)
    assert capped and m == 10
    assert tail > 1e-6                    # cap forced a larger discarded tail


def test_ensemble_entropy_limits():
    assert ensemble_entropy(np.array([1.0])) == 0.0
    n = 64
    assert abs(ensemble_entropy(np.full(n, 1.0 / n)) - np.log(n)) < 1e-12


def test_static_correlation_score():
    assert static_correlation_score(np.array([2.0, 2.0, 0.0])) == 0.0
    assert abs(static_correlation_score(np.array([1.0, 1.0])) - 2.0) < 1e-15


def test_trace_distance_identical_and_orthogonal():
    dim = 6
    e = np.eye(dim)
    v1, p1 = e[:1], np.array([1.0])
    v2, p2 = e[1:2], np.array([1.0])
    assert trace_distance_projected(v1, p1, v1, p1) < 1e-12
    assert abs(trace_distance_projected(v1, p1, v2, p2) - 1.0) < 1e-12
    # Mixed vs itself with permuted member order is still zero.
    v3 = e[:2]; p3 = np.array([0.7, 0.3])
    v4 = e[[1, 0]]; p4 = np.array([0.3, 0.7])
    assert trace_distance_projected(v3, p3, v4, p4) < 1e-12


def test_resolve_kT_list():
    kT, tag = resolve_kT_list([0.05, 0.25], relative=False)
    np.testing.assert_allclose(kT, [0.05, 0.25])
    assert tag == "absolute_hartree"
    kT, tag = resolve_kT_list([0.01], relative=True, spectral_width=10.0)
    np.testing.assert_allclose(kT, [0.1])
    assert tag == "relative_spectral_width"
    with pytest.raises(ValueError):
        resolve_kT_list([0.01], relative=True)


# --- end-to-end on H2O, ncas=6 (dim=400, fast) -------------------------------

@pytest.fixture(scope="module")
def h2o_solved(h2o_record):
    mol = build_mol(h2o_record, "Angstrom")
    S = overlap(mol)
    C, eps, nocc = orbitals(h2o_record, mol, S)
    aspace = select_active(eps, nocc, n_act_occ=3, n_act_virt=3)
    ham = build_cas_hamiltonian(mol, C, nocc, aspace)
    solver = DenseEDSolver()
    ens = solver.solve(ham.h1eff, ham.g, aspace, kT_max=KT_MAX,
                       weight_cutoff=CUTOFF)
    ens_g = gaussian_reference_ensemble(ham.h1eff, aspace, solver,
                                        kT_max=KT_MAX, weight_cutoff=CUTOFF)
    return aspace, ham, ens, ens_g


@pytest.mark.parametrize("kT", [0.05, 0.10, 0.25])
def test_thermal_block_h2o(h2o_solved, kT):
    aspace, ham, ens, ens_g = h2o_solved
    blk = build_thermal_block(ens, ens_g, kT, aspace, weight_cutoff=CUTOFF)

    m = len(blk.E)
    assert blk.p.shape == (m,) and blk.civecs.shape == (m, aspace.dim)
    assert abs(blk.p.sum() + blk.truncation_error - 1.0) < 1e-12
    assert np.all(np.diff(blk.E) >= -1e-12)
    assert blk.entropy >= 0.0
    assert 0.0 < blk.c_max_sq <= 1.0
    assert np.all(blk.nat_occs >= 0.0) and np.all(blk.nat_occs <= 2.0)
    # trace of the thermal 1-RDM = nelecas * (kept weight)
    assert abs(blk.nat_occs.sum() - aspace.nelecas * blk.p.sum()) < 1e-8
    assert 0.0 <= blk.tracedist_gaussian <= 1.0 + 1e-10
    assert blk.tracedist_bound >= 0.0
    assert blk.civecs.dtype == np.float64


def test_low_kT_expectations(h2o_solved):
    """Equilibrium closed-shell molecule at kT=0.05: nearly pure ground state,
    small static correlation, tiny truncated ensemble (m ~ O(10))."""
    aspace, ham, ens, ens_g = h2o_solved
    blk = build_thermal_block(ens, ens_g, 0.05, aspace, weight_cutoff=CUTOFF)
    assert len(blk.E) <= 50
    assert blk.p[0] > 0.9
    assert blk.c_max_sq > 0.8          # single-reference ground state
    assert blk.static_corr < 0.5       # small at equilibrium (expected, not a bug)


def test_kT_monotonicity(h2o_solved):
    aspace, ham, ens, ens_g = h2o_solved
    blocks = [build_thermal_block(ens, ens_g, kT, aspace, weight_cutoff=CUTOFF)
              for kT in (0.05, 0.10, 0.25)]
    entropies = [b.entropy for b in blocks]
    sizes = [len(b.E) for b in blocks]
    assert entropies == sorted(entropies)
    assert sizes == sorted(sizes)


def test_kT_above_kT_max_clamps(h2o_solved, caplog):
    """Asking for kT > kT_max cannot crash; it clamps to the kept states and
    reports the honest (larger) truncation error."""
    aspace, ham, ens, ens_g = h2o_solved
    ens_small = DenseEDSolver().solve(ham.h1eff, ham.g, aspace, kT_max=0.01,
                                      weight_cutoff=CUTOFF)
    ens_g_small = DenseEDSolver().solve(ham.h1eff, None, aspace, kT_max=0.01,
                                        weight_cutoff=CUTOFF)
    blk = build_thermal_block(ens_small, ens_g_small, 0.25, aspace,
                              weight_cutoff=CUTOFF)
    assert len(blk.E) <= len(ens_small.E)
    assert blk.truncation_error > CUTOFF


def test_thermal_rdm1_ground_state_only(h2o_solved):
    """With all weight on the ground state the thermal RDM is the pure-state
    RDM; occupations of a dominantly closed-shell state are near 2 and 0."""
    aspace, ham, ens, ens_g = h2o_solved
    dm = thermal_rdm1(ens.vecs[:1], np.array([1.0]), aspace)
    n = natural_occupations(dm)
    assert abs(n.sum() - aspace.nelecas) < 1e-10
    assert n[0] > 1.9 and n[-1] < 0.1
    assert leading_determinant_weight(ens.vecs[0]) > 0.8
