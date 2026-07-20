"""Module E tests.

Includes the mandatory correctness gate: the lowest dense-ED eigenvalue plus
ecore must reproduce PySCF's CASCI energy (same injected orbitals) to 1e-8 Ha,
and the second eigenvalue must match ``fcisolver`` with ``nroots=2`` — at both
ncas=6 and ncas=8 to prove the active space is genuinely parametric.
"""

import numpy as np
import pytest

import qthermal.diagonalize as dmod
from qthermal.active_space import ActiveSpace, select_active
from qthermal.diagonalize import (
    DenseEDSolver,
    IterativeWindowSolver,
    NonInteractingSolver,
    SectorTooLargeError,
    SpectralSolver,
    TruncatedEnsemble,
    build_sector_hamiltonian,
    krylov_tail_bound,
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


def test_dense_keep_cap_override():
    """keep_cap=N caps the stored prefix; keep_cap=0 lifts the cap so the
    weight cutoff alone decides. The full spectrum is kept either way."""
    rng = np.random.default_rng(7)
    aspace = _fake_aspace(2)              # ncas=4, dim=36
    A = rng.standard_normal((aspace.ncas, aspace.ncas))
    h1 = 0.5 * (A + A.T)
    # kT far above the spectral width: all 36 states carry ~equal weight,
    # so the cutoff wants the whole sector.
    kw = dict(kT_max=50.0, weight_cutoff=1e-9)

    ens = DenseEDSolver(keep_cap=5).solve(h1, None, aspace, **kw)
    assert ens.cap_hit and ens.vecs.shape == (5, aspace.dim)
    assert ens.tail_weight > 0.5          # most of the mass discarded
    assert len(ens.evals_full) == aspace.dim

    ens0 = DenseEDSolver(keep_cap=0).solve(h1, None, aspace, **kw)
    assert not ens0.cap_hit and ens0.vecs.shape == (aspace.dim, aspace.dim)
    assert ens0.tail_weight <= 1e-12


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


# --- NonInteractingSolver: the closed-form Gaussian-reference shortcut ------
#
# The mandatory correctness gate for the 2026-07-20 optimization: every
# quantity NonInteractingSolver produces must reproduce DenseEDSolver(g=None)
# (the previous, general-purpose code path, still available and used here
# purely as ground truth) to float64 noise -- on the real H2O molecule the
# `solved` fixture builds, at both ncas=6 and ncas=8.

@pytest.fixture(scope="module")
def gaussian_solved(solved):
    """Closed-form vs dense-reference g=0 ensembles, same Hamiltonian."""
    aspace, ham, _, _ = solved
    new = NonInteractingSolver().solve(ham.h1eff, None, aspace,
                                       kT_max=KT_MAX, weight_cutoff=CUTOFF)
    old = DenseEDSolver().solve(ham.h1eff, None, aspace,
                                kT_max=KT_MAX, weight_cutoff=CUTOFF)
    return aspace, ham, new, old


def test_noninteracting_protocol_and_container(gaussian_solved):
    assert isinstance(NonInteractingSolver(), SpectralSolver)
    aspace, ham, new, old = gaussian_solved
    assert isinstance(new, TruncatedEnsemble)
    assert new.solver_name == "noninteracting_closed_form"
    assert new.evals_full is not None and len(new.evals_full) == aspace.dim
    m = len(new.E)
    assert new.vecs.shape == (m, aspace.dim)
    assert np.all(np.diff(new.E) >= -1e-12)               # ascending
    np.testing.assert_allclose(np.linalg.norm(new.vecs, axis=1), 1.0,
                               atol=1e-10)


def test_noninteracting_matches_dense_reference(gaussian_solved):
    """Full spectrum, kept count, and truncation accounting must match the
    previous (dense, general-purpose) code path exactly, not approximately."""
    aspace, ham, new, old = gaussian_solved
    np.testing.assert_allclose(new.evals_full, old.evals_full, atol=1e-9)
    assert len(new.E) == len(old.E)
    np.testing.assert_allclose(new.E, old.E, atol=1e-9)
    assert abs(new.tail_weight - old.tail_weight) < 1e-12
    assert new.cap_hit == old.cap_hit


def test_noninteracting_eigenvector_residuals(gaussian_solved):
    """H v = E v for every kept vector, against the true (dense) sector
    matrix -- basis-independent, so it holds even under energy degeneracy."""
    aspace, ham, new, old = gaussian_solved
    H = build_sector_hamiltonian(ham.h1eff, None, aspace)
    for E, v in zip(new.E, new.vecs):
        assert np.abs(H @ v - E * v).max() < 1e-8


def test_noninteracting_rejects_interacting_g():
    aspace = _fake_aspace(2)
    h1 = np.eye(aspace.ncas)
    g = np.zeros((aspace.ncas,) * 4)
    with pytest.raises(ValueError, match="g=0 reference"):
        NonInteractingSolver().solve(h1, g, aspace, kT_max=0.1)


def test_noninteracting_matches_one_body_theory():
    """Same analytic check as test_sector_hamiltonian_matches_one_body_theory,
    through the closed-form solver directly."""
    rng = np.random.default_rng(11)
    aspace = _fake_aspace(2)                  # ncas=4, (2, 2), dim=36
    A = rng.standard_normal((aspace.ncas, aspace.ncas))
    h1 = 0.5 * (A + A.T)
    ens = NonInteractingSolver().solve(h1, None, aspace, kT_max=50.0,
                                      weight_cutoff=1e-9)
    orb = np.linalg.eigvalsh(h1)
    expected_ground = 2.0 * orb[:aspace.nalpha].sum()
    assert abs(ens.E[0] - expected_ground) < 1e-10
    H = build_sector_hamiltonian(h1, None, aspace)
    evals_ref = np.linalg.eigvalsh(H)
    np.testing.assert_allclose(ens.evals_full, evals_ref, atol=1e-9)


def test_noninteracting_keep_cap_override():
    """keep_cap plumbing mirrors DenseEDSolver's (test_dense_keep_cap_override)."""
    rng = np.random.default_rng(7)
    aspace = _fake_aspace(2)                  # ncas=4, dim=36
    A = rng.standard_normal((aspace.ncas, aspace.ncas))
    h1 = 0.5 * (A + A.T)
    kw = dict(kT_max=50.0, weight_cutoff=1e-9)

    ens = NonInteractingSolver(keep_cap=5).solve(h1, None, aspace, **kw)
    assert ens.cap_hit and ens.vecs.shape == (5, aspace.dim)
    assert ens.tail_weight > 0.5

    ens0 = NonInteractingSolver(keep_cap=0).solve(h1, None, aspace, **kw)
    assert not ens0.cap_hit and ens0.vecs.shape == (aspace.dim, aspace.dim)
    assert ens0.tail_weight <= 1e-12


# --- IterativeWindowSolver ---------------------------------------------------

KT_IT = 0.025


def _random_one_body_sector(seed=7, half=2):
    """(h1, aspace, exact sector spectrum) for a small g=0 problem."""
    rng = np.random.default_rng(seed)
    aspace = _fake_aspace(half)
    A = rng.standard_normal((aspace.ncas, aspace.ncas))
    h1 = 0.5 * (A + A.T)
    evals = np.linalg.eigvalsh(build_sector_hamiltonian(h1, None, aspace))
    return h1, aspace, evals


def _exact_tail(evals, kT, m):
    p = np.exp(-(evals - evals[0]) / kT)
    return float(p[m:].sum() / p.sum())


def test_tail_bound_dominates_exact_tail():
    """The counting bound must upper-bound the exact softmax tail for every
    truncation point, and shrink monotonically as more states are kept."""
    rng = np.random.default_rng(3)
    evals = np.sort(rng.normal(scale=2.0, size=500))
    E, dim, kT = evals[:40], len(evals), 0.3
    bounds = [krylov_tail_bound(E, m, dim, kT) for m in range(1, 41)]
    for m, b in zip(range(1, 41), bounds):
        assert b >= _exact_tail(evals, kT, m)
    assert np.all(np.diff(bounds) <= 1e-15)


def test_tail_bound_validates_arguments():
    E = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="1 <= m"):
        krylov_tail_bound(E, 0, 10, 0.1)
    with pytest.raises(ValueError, match="1 <= m"):
        krylov_tail_bound(E, 3, 10, 0.1)
    with pytest.raises(ValueError, match="kT"):
        krylov_tail_bound(E, 1, 10, -0.1)


@pytest.fixture(scope="module")
def iterative_solved(solved):
    """Krylov ensemble on the same Hamiltonian as the dense fixture."""
    aspace, ham, dense_ens, _ = solved
    ens = IterativeWindowSolver().solve(ham.h1eff, ham.g, aspace,
                                        kT_max=KT_IT, weight_cutoff=CUTOFF)
    return aspace, ham, dense_ens, ens


def test_krylov_container_and_protocol(iterative_solved):
    aspace, ham, dense_ens, ens = iterative_solved
    assert isinstance(IterativeWindowSolver(), SpectralSolver)
    assert isinstance(ens, TruncatedEnsemble)
    m = len(ens.E)
    assert 1 <= m and ens.vecs.shape == (m, aspace.dim)
    assert np.all(np.diff(ens.E) >= -1e-12)
    assert ens.evals_full is None                 # never promised by Krylov
    assert ens.solver_name == "iterative_krylov"
    np.testing.assert_allclose(np.linalg.norm(ens.vecs, axis=1), 1.0,
                               atol=1e-8)


def test_krylov_matches_dense_prefix(iterative_solved):
    """Kept energies equal the dense spectrum's bottom, and the certified
    bound really dominates the exact discarded weight (known from dense)."""
    aspace, ham, dense_ens, ens = iterative_solved
    m = len(ens.E)
    np.testing.assert_allclose(ens.E, dense_ens.evals_full[:m], atol=1e-7)
    assert not ens.cap_hit
    assert _exact_tail(dense_ens.evals_full, KT_IT, m) <= ens.tail_weight
    assert ens.tail_weight <= CUTOFF


def test_krylov_eigenvector_residuals(iterative_solved):
    """H v = E v through the same contraction kernel, for every kept vector."""
    from pyscf import fci

    aspace, ham, _, ens = iterative_solved
    na, nb = aspace.na_strings, aspace.nb_strings
    nelec = (aspace.nalpha, aspace.nbeta)
    h2e = fci.direct_spin1.absorb_h1e(ham.h1eff, ham.g, aspace.ncas, nelec, 0.5)
    for E, v in zip(ens.E, ens.vecs):
        Hv = fci.direct_spin1.contract_2e(h2e, v.reshape(na, nb), aspace.ncas,
                                          nelec).ravel()
        assert np.abs(Hv - E * v).max() < 1e-4    # Davidson tol_residual 1e-5


def test_krylov_thermal_block_parity(iterative_solved):
    """End-to-end Module F parity: blocks built from Krylov and dense
    ensembles agree on every stored diagnostic."""
    from qthermal.thermal import build_thermal_block, gaussian_reference_ensemble

    aspace, ham, dense_ens, ens = iterative_solved
    dense_g = DenseEDSolver().solve(ham.h1eff, None, aspace,
                                    kT_max=KT_MAX, weight_cutoff=CUTOFF)
    it_g = gaussian_reference_ensemble(ham.h1eff, aspace,
                                       kT_max=KT_IT, weight_cutoff=CUTOFF)
    blk_d = build_thermal_block(dense_ens, dense_g, KT_IT, aspace)
    blk_i = build_thermal_block(ens, it_g, KT_IT, aspace)

    n = min(len(blk_d.E), len(blk_i.E))
    np.testing.assert_allclose(blk_i.E[:n], blk_d.E[:n], atol=1e-7)
    np.testing.assert_allclose(blk_i.p[:n], blk_d.p[:n], atol=1e-6)
    # The kept sets may differ by states whose weight sits at the 1e-6
    # cutoff (dense truncates against the full spectrum, Krylov against its
    # window), so thermally averaged quantities legitimately differ by
    # O(cutoff) — those tolerances bound truncation-set differences, not
    # solver error. c_max_sq uses only the ground vector: its tolerance is
    # the Davidson residual accuracy (~1e-5), not the truncation.
    np.testing.assert_allclose(blk_i.nat_occs, blk_d.nat_occs, atol=1e-5)
    assert abs(blk_i.entropy - blk_d.entropy) < 5e-5
    assert abs(blk_i.static_corr - blk_d.static_corr) < 2e-5
    assert abs(blk_i.c_max_sq - blk_d.c_max_sq) < 1e-6
    assert abs(blk_i.tracedist_gaussian - blk_d.tracedist_gaussian) < 1e-5
    assert blk_i.truncation_error < 2e-6


def test_krylov_gaussian_matches_one_body_theory():
    """g=None through the Krylov seam reproduces the exact one-body sector."""
    h1, aspace, evals = _random_one_body_sector()
    ens = IterativeWindowSolver(init_nroots=4).solve(h1, None, aspace,
                                                     kT_max=0.05)
    orb = np.linalg.eigvalsh(h1)
    assert abs(ens.E[0] - 2.0 * orb[:aspace.nalpha].sum()) < 1e-9
    np.testing.assert_allclose(ens.E, evals[:len(ens.E)], atol=1e-8)


def test_krylov_escalates_and_certifies(monkeypatch):
    """init_nroots too small: k doubles until the bound certifies, and the
    result is still exact against the dense reference."""
    h1, aspace, evals = _random_one_body_sector()
    kT = float(evals[5] - evals[0]) / 20.0

    ks = []
    orig = IterativeWindowSolver._solve_roots

    def spy(self, h1eff, g, asp, k):
        ks.append(k)
        return orig(self, h1eff, g, asp, k)

    monkeypatch.setattr(IterativeWindowSolver, "_solve_roots", spy)
    ens = IterativeWindowSolver(init_nroots=2).solve(h1, None, aspace,
                                                     kT_max=kT)
    assert len(ks) >= 2 and ks == sorted(ks)      # escalation happened
    assert not ens.cap_hit and ens.tail_weight <= CUTOFF
    np.testing.assert_allclose(ens.E, evals[:len(ens.E)], atol=1e-8)
    assert _exact_tail(evals, kT, len(ens.E)) <= ens.tail_weight


def test_krylov_root_cap():
    """Unreachable cutoff: solver caps at max_nroots with one buffer root and
    reports the honest (large) certified bound."""
    h1, aspace, evals = _random_one_body_sector()
    ens = IterativeWindowSolver(init_nroots=2, max_nroots=4).solve(
        h1, None, aspace, kT_max=1e3)
    assert ens.cap_hit
    assert len(ens.E) == 3                        # k_ceiling=4 minus buffer
    assert ens.tail_weight > CUTOFF
    np.testing.assert_allclose(ens.E, evals[:3], atol=1e-8)


def test_krylov_unconverged_escalates(monkeypatch):
    """A degenerate-multiplet-style unconverged solve widens the window
    instead of raising, and the final result is still exact + certified."""
    h1, aspace, evals = _random_one_body_sector()
    orig = IterativeWindowSolver._solve_roots
    calls = []

    def flaky(self, h1eff, g, asp, k):
        E, vecs, _ = orig(self, h1eff, g, asp, k)
        calls.append(k)
        return E, vecs, len(calls) > 1        # first solve "stalls"

    monkeypatch.setattr(IterativeWindowSolver, "_solve_roots", flaky)
    ens = IterativeWindowSolver(init_nroots=4).solve(h1, None, aspace,
                                                     kT_max=0.05)
    assert calls[:2] == [4, 8]                # escalated past the stall
    assert not ens.cap_hit and ens.tail_weight <= CUTOFF
    np.testing.assert_allclose(ens.E, evals[:len(ens.E)], atol=1e-8)


def test_krylov_unconverged_at_ceiling_raises(monkeypatch):
    h1, aspace, _ = _random_one_body_sector()
    orig = IterativeWindowSolver._solve_roots

    def never(self, h1eff, g, asp, k):
        E, vecs, _ = orig(self, h1eff, g, asp, k)
        return E, vecs, False

    monkeypatch.setattr(IterativeWindowSolver, "_solve_roots", never)
    with pytest.raises(RuntimeError, match="unconverged at the root ceiling"):
        IterativeWindowSolver(init_nroots=2, max_nroots=2).solve(
            h1, None, aspace, kT_max=0.05)


def test_krylov_unconverged_at_ceiling_caps_on_prior(monkeypatch):
    """Ceiling stall with an earlier converged solve: cap there, honestly."""
    h1, aspace, evals = _random_one_body_sector()
    orig = IterativeWindowSolver._solve_roots

    def flaky(self, h1eff, g, asp, k):
        E, vecs, _ = orig(self, h1eff, g, asp, k)
        return E, vecs, k == 2                # only the k=2 solve converges

    monkeypatch.setattr(IterativeWindowSolver, "_solve_roots", flaky)
    ens = IterativeWindowSolver(init_nroots=2, max_nroots=4).solve(
        h1, None, aspace, kT_max=1e3)         # uncertifiable -> escalates
    assert ens.cap_hit and len(ens.E) == 1
    assert ens.tail_weight > CUTOFF
    np.testing.assert_allclose(ens.E, evals[:1], atol=1e-8)


def test_krylov_ram_guardrail(monkeypatch):
    h1, aspace, _ = _random_one_body_sector()
    monkeypatch.setattr(dmod, "_available_ram_bytes", lambda: 1000)
    with pytest.raises(SectorTooLargeError, match="free RAM"):
        IterativeWindowSolver().solve(h1, None, aspace, kT_max=0.05)


def test_krylov_ram_caps_escalation(monkeypatch):
    """RAM sufficient for k=2 but not k=4: keeps the k=2 solution, capped."""
    h1, aspace, _ = _random_one_body_sector()
    dim = aspace.dim
    budget = (IterativeWindowSolver._bytes_needed(2, dim)
              + IterativeWindowSolver._bytes_needed(4, dim)) // 2
    monkeypatch.setattr(dmod, "_available_ram_bytes", lambda: budget)
    ens = IterativeWindowSolver(init_nroots=2).solve(h1, None, aspace,
                                                     kT_max=1e3)
    assert ens.cap_hit and len(ens.E) == 1
    assert ens.tail_weight > CUTOFF
