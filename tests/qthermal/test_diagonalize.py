"""Module E tests.

Includes the mandatory correctness gate: the lowest dense-ED eigenvalue plus
ecore must reproduce PySCF's CASCI energy (same injected orbitals) to 1e-8 Ha,
and the second eigenvalue must match ``fcisolver`` with ``nroots=2`` — at both
ncas=6 and ncas=8 to prove the active space is genuinely parametric.
"""

import numpy as np
import pytest

from qthermal.active_space import ActiveSpace, select_active
from qthermal.diagonalize import (
    DenseEDSolver,
    SectorTooLargeError,
    SpectralSolver,
    TruncatedEnsemble,
    build_sector_hamiltonian,
)
from qthermal.hamiltonian import build_cas_hamiltonian, make_casci, make_injected_rhf
from qthermal.orbitals import build_mol, orbitals, overlap

KT_MAX = 0.25
CUTOFF = 1e-6


@pytest.fixture(scope="module", params=[(3, 3), (4, 4)], ids=["ncas6", "ncas8"])
def solved(request, h2o_record):
    """Hamiltonian + dense-ED ensemble + CASCI reference, per active space."""
    n_act_occ, n_act_virt = request.param
    mol = build_mol(h2o_record, "Angstrom")
    S = overlap(mol)
    C, eps, nocc = orbitals(h2o_record, mol, S)
    aspace = select_active(eps, nocc, n_act_occ=n_act_occ, n_act_virt=n_act_virt)
    ham = build_cas_hamiltonian(mol, C, nocc, aspace)

    ensemble = DenseEDSolver().solve(ham.h1eff, ham.g, aspace,
                                     kT_max=KT_MAX, weight_cutoff=CUTOFF)

    mf = make_injected_rhf(mol, C, nocc)
    mc = make_casci(mf, aspace)
    mc.fcisolver.nroots = 2
    # Davidson at its default tolerance leaves the second root ~2e-8 Ha off
    # the exact eigenvalue at dim=4900; tighten so the 1e-8 gate tests the
    # dense solver, not the reference's convergence.
    mc.fcisolver.conv_tol = 1e-12
    mc.kernel()
    e_casci = np.atleast_1d(mc.e_tot)
    return aspace, ham, ensemble, e_casci


def test_correctness_gate_ground_state(solved):
    aspace, ham, ensemble, e_casci = solved
    assert abs((ensemble.E[0] + ham.ecore) - e_casci[0]) < 1e-8


def test_correctness_gate_second_root(solved):
    aspace, ham, ensemble, e_casci = solved
    assert len(e_casci) == 2
    assert abs((ensemble.E[1] + ham.ecore) - e_casci[1]) < 1e-8


def test_ensemble_container(solved):
    aspace, ham, ensemble, _ = solved
    assert isinstance(ensemble, TruncatedEnsemble)
    assert isinstance(DenseEDSolver(), SpectralSolver)
    m = len(ensemble.E)
    assert ensemble.vecs.shape == (m, aspace.dim)
    assert np.all(np.diff(ensemble.E) >= -1e-12)          # ascending
    assert ensemble.evals_full is not None and len(ensemble.evals_full) == aspace.dim
    np.testing.assert_allclose(np.linalg.norm(ensemble.vecs, axis=1), 1.0,
                               atol=1e-10)
    if ensemble.cap_hit:
        # Cap max(1024, dim//4) bound before the cutoff was reached; the
        # (exact) tail weight may then legitimately exceed the cutoff.
        assert m == max(1024, aspace.dim // 4)
        assert 0.0 <= ensemble.tail_weight < 1e-2
    else:
        assert 0.0 <= ensemble.tail_weight <= CUTOFF * 1.001
    assert ensemble.solver_name == "dense_ed"


def test_eigenvector_residual(solved):
    """H v = E v for the ground vector, via the same contraction kernel."""
    from pyscf import fci

    aspace, ham, ensemble, _ = solved
    na, nb = aspace.na_strings, aspace.nb_strings
    nelec = (aspace.nalpha, aspace.nbeta)
    h2e = fci.direct_spin1.absorb_h1e(ham.h1eff, ham.g, aspace.ncas, nelec, 0.5)
    v = ensemble.vecs[0]
    Hv = fci.direct_spin1.contract_2e(h2e, v.reshape(na, nb), aspace.ncas,
                                      nelec).ravel()
    assert np.abs(Hv - ensemble.E[0] * v).max() < 1e-8


def test_gaussian_reference_solvable(solved):
    """g=None solves the one-body-only sector through the same code path."""
    aspace, ham, ensemble, _ = solved
    ens_g = DenseEDSolver().solve(ham.h1eff, None, aspace,
                                  kT_max=KT_MAX, weight_cutoff=CUTOFF)
    assert len(ens_g.evals_full) == aspace.dim
    # Interacting ground state is below the g=0 spectrum shifted by the
    # mean-field-like repulsion, so just check both are finite and distinct.
    assert not np.allclose(ens_g.E[0], ensemble.E[0])


def _fake_aspace(half: int) -> ActiveSpace:
    return ActiveSpace(active_idx=np.arange(half, 3 * half),
                       core_idx=np.arange(half),
                       n_act_occ=half, n_act_virt=half)


def test_guardrail_requires_flag_above_5000():
    aspace = _fake_aspace(5)          # ncas=10 -> dim 63504
    solver = DenseEDSolver(allow_large_dense=False)
    with pytest.raises(SectorTooLargeError, match="allow-large-dense"):
        solver.solve(np.zeros((10, 10)), None, aspace, kT_max=0.1)


def test_guardrail_refuses_above_70000():
    aspace = _fake_aspace(6)          # ncas=12 -> dim 853776
    solver = DenseEDSolver(allow_large_dense=True)
    with pytest.raises(SectorTooLargeError, match="tensor-network"):
        solver.solve(np.zeros((12, 12)), None, aspace, kT_max=0.1)


def test_sector_hamiltonian_matches_one_body_theory():
    """For g=0 the sector spectrum is sums of orbital energies: check the
    ground state equals the sum of the nalpha+nbeta lowest one-body levels."""
    rng = np.random.default_rng(7)
    half = 2                          # ncas=4, (2, 2), dim 36
    aspace = _fake_aspace(half)
    ncas = aspace.ncas
    A = rng.standard_normal((ncas, ncas))
    h1 = 0.5 * (A + A.T)
    H = build_sector_hamiltonian(h1, None, aspace)
    evals = np.linalg.eigvalsh(H)
    orb = np.linalg.eigvalsh(h1)
    expected_ground = 2.0 * orb[:aspace.nalpha].sum()
    assert abs(evals[0] - expected_ground) < 1e-10
