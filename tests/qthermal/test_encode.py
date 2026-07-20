"""Module G (encode) tests: encoders checked against independent references.

The central test diagonalizes a random CAS(4,4) problem with the dense solver
(PySCF contract_2e path) and requires the JW-encoded eigenvectors to be exact
eigenvectors of the *independently constructed* PennyLane Jordan-Wigner
Hamiltonian. Fermionic sign conventions, integral conventions, and bit
ordering all have to be simultaneously right for that to hold.
"""

import numpy as np
import pytest
from pyscf import fci

from qthermal.active_space import ActiveSpace
from qthermal.diagonalize import DenseEDSolver
from qthermal.encode import (encode_jw, encode_sector, jw_basis_indices, jw_wire,
                             jw_hamiltonian, pauli_components, pauli_label,
                             thermal_density_matrix)

pennylane = pytest.importorskip("pennylane")


def test_pauli_components_matches_pennylane_decompose():
    rng = np.random.default_rng(3)
    n = 3
    m = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    m = m + m.conj().T
    mine = {pauli_label(i, n): c for i, c in enumerate(pauli_components(m))}
    ref = pennylane.pauli_decompose(m, wire_order=range(n), pauli=True)
    for word, coeff in ref.items():
        label = " ".join(f"{p}{w}" for w, p in sorted(word.items())) or "I"
        np.testing.assert_allclose(mine[label], coeff, atol=1e-12)
    assert len(mine) == 4 ** n


def _random_cas44(seed=7):
    rng = np.random.default_rng(seed)
    n = 4
    h1 = rng.normal(size=(n, n))
    h1 = 0.5 * (h1 + h1.T)
    g = rng.normal(size=(n,) * 4)
    g = g + g.transpose(1, 0, 2, 3)
    g = g + g.transpose(0, 1, 3, 2)
    g = g + g.transpose(2, 3, 0, 1)          # chemist (pq|rs) 8-fold symmetry
    aspace = ActiveSpace(active_idx=np.arange(4, dtype=np.int64),
                         core_idx=np.arange(0, dtype=np.int64),
                         n_act_occ=2, n_act_virt=2)
    return h1, g / 8.0, aspace


@pytest.fixture(scope="module")
def cas44_ensemble():
    h1, g, aspace = _random_cas44()
    ens = DenseEDSolver().solve(h1, g, aspace, kT_max=5.0)
    assert len(ens.E) >= 3
    return h1, g, aspace, ens


@pytest.mark.parametrize("ordering", ["blocked", "interleaved"])
def test_jw_index_map_is_bijective_with_correct_occupations(ordering):
    ncas, na, nb = 4, 2, 2
    idx = jw_basis_indices(ncas, na, nb, ordering)
    assert len(np.unique(idx)) == idx.shape[0] == 36
    q = 2 * ncas
    for basis_int in idx:
        bits = [(basis_int >> (q - 1 - w)) & 1 for w in range(q)]
        alpha = [bits[jw_wire(p, 0, ncas, ordering)] for p in range(ncas)]
        beta = [bits[jw_wire(p, 1, ncas, ordering)] for p in range(ncas)]
        assert sum(alpha) == na and sum(beta) == nb


@pytest.mark.parametrize("ordering", ["blocked", "interleaved"])
def test_jw_encoded_eigenvectors_diagonalize_jw_hamiltonian(cas44_ensemble,
                                                            ordering):
    h1, g, aspace, ens = cas44_ensemble
    H = jw_hamiltonian(h1, g, aspace.ncas, ordering=ordering)
    Hmat = pennylane.matrix(H, wire_order=range(2 * aspace.ncas))
    V = encode_jw(ens.vecs, aspace.ncas, aspace.nalpha, aspace.nbeta,
                  ordering)
    for energy, vec in zip(ens.E, V):
        assert np.linalg.norm(Hmat @ vec - energy * vec) < 1e-8


@pytest.mark.parametrize("ordering", ["blocked", "interleaved"])
def test_jw_occupations_match_pyscf_rdm1(cas44_ensemble, ordering):
    _, _, aspace, ens = cas44_ensemble
    civec = ens.vecs[0].reshape(aspace.na_strings, aspace.nb_strings)
    dm1 = fci.direct_spin1.make_rdm1(civec, aspace.ncas,
                                     (aspace.nalpha, aspace.nbeta))
    vec = encode_jw(ens.vecs[0], aspace.ncas, aspace.nalpha, aspace.nbeta,
                    ordering)[0]
    prob = np.abs(vec) ** 2
    q = 2 * aspace.ncas
    basis = np.arange(prob.shape[0])
    for orb in range(aspace.ncas):
        occ = sum(prob[(basis & (1 << (q - 1 - jw_wire(orb, s, aspace.ncas,
                                                       ordering)))) > 0].sum()
                  for s in (0, 1))
        np.testing.assert_allclose(occ, dm1[orb, orb], atol=1e-10)


def test_interleaved_parity_signs_nontrivial():
    """CAS(4,4) has crossings: the interleaved sign vector must mix +/-1."""
    from qthermal.encode import jw_parity_signs
    signs = jw_parity_signs(4, 2, 2, "interleaved")
    assert set(np.unique(signs)) == {-1.0, 1.0}
    np.testing.assert_array_equal(jw_parity_signs(4, 2, 2, "blocked"), 1.0)


def test_encodings_preserve_geometry(cas44_ensemble):
    _, _, aspace, ens = cas44_ensemble
    gram = ens.vecs @ ens.vecs.T
    for states in (encode_jw(ens.vecs, aspace.ncas, aspace.nalpha,
                             aspace.nbeta),
                   encode_jw(ens.vecs, aspace.ncas, aspace.nalpha,
                             aspace.nbeta, "interleaved"),
                   encode_sector(ens.vecs)):
        np.testing.assert_allclose(np.real(states @ states.conj().T), gram,
                                   atol=1e-12)
    assert encode_sector(ens.vecs).shape[1] == 64     # 36 -> 2**6


@pytest.mark.parametrize("ncas", [3, 4, 6])
def test_extended_heisenberg_paulis_count_and_structure(ncas):
    from qthermal.encode import extended_heisenberg_paulis, jw_wire
    ops = extended_heisenberg_paulis(ncas)
    assert len(ops) == 4 * ncas ** 2 - ncas
    alpha = {jw_wire(p, 0, ncas, "blocked") for p in range(ncas)}
    for op in ops:
        (word, _), = op.pauli_rep.items()
        assert 1 <= len(word) <= 2                    # quadratic neurons
        kinds = set(word.values())
        if kinds & {"X", "Y"}:
            assert len(kinds) == 1                    # XX or YY, never mixed
            wires = set(word.keys())
            assert wires <= alpha or wires.isdisjoint(alpha)


@pytest.mark.parametrize("ordering", ["blocked", "interleaved"])
def test_extended_heisenberg_expectations_match_pennylane(cas44_ensemble,
                                                          ordering):
    """Determinant-basis Tr(rho P) == brute-force expval on the encoded
    register, string by string, in the generator's order — for both wire
    layouts (interleaved exercises nontrivial parity signs)."""
    from qthermal.encode import (extended_heisenberg_expectations,
                                 extended_heisenberg_labels,
                                 extended_heisenberg_paulis)

    _, _, aspace, ens = cas44_ensemble
    w = np.exp(-(ens.E - ens.E[0]) / 0.5)
    w /= w.sum()
    vals = extended_heisenberg_expectations(
        ens.vecs, w, aspace.ncas, aspace.nalpha, aspace.nbeta, ordering)
    ops = extended_heisenberg_paulis(aspace.ncas, ordering)
    labels = extended_heisenberg_labels(aspace.ncas, ordering)
    assert len(vals) == len(ops) == len(labels) == 4 * aspace.ncas ** 2 - aspace.ncas

    V = encode_jw(ens.vecs, aspace.ncas, aspace.nalpha, aspace.nbeta, ordering)
    for op, label, mine in zip(ops, labels, vals):
        M = pennylane.matrix(op, wire_order=range(2 * aspace.ncas))
        ref = sum(wk * np.real(vk.conj() @ M @ vk) for wk, vk in zip(w, V))
        np.testing.assert_allclose(mine, ref, atol=1e-10, err_msg=label)
        (word, _), = op.pauli_rep.items()
        assert label == " ".join(f"{p}{wi}" for wi, p in sorted(word.items()))
    # particle-number sum rule: sum <Z_w> = n_qubits - 2 * nelec
    n_qubits = 2 * aspace.ncas
    np.testing.assert_allclose(
        vals[:n_qubits].sum(),
        n_qubits - 2 * (aspace.nalpha + aspace.nbeta), atol=1e-10)


@pytest.fixture(scope="module")
def cas31_ensemble():
    """ncas=3, (1, 1) electrons: odd block parities exercise sector = -1."""
    rng = np.random.default_rng(11)
    h1 = rng.normal(size=(3, 3))
    h1 = 0.5 * (h1 + h1.T)
    g = rng.normal(size=(3,) * 4)
    g = g + g.transpose(1, 0, 2, 3)
    g = g + g.transpose(0, 1, 3, 2)
    g = g + g.transpose(2, 3, 0, 1)
    aspace = ActiveSpace(active_idx=np.arange(3, dtype=np.int64),
                         core_idx=np.arange(0, dtype=np.int64),
                         n_act_occ=1, n_act_virt=2)
    return aspace, DenseEDSolver().solve(h1, g / 8.0, aspace, kT_max=5.0)


def _taper_state(vec, ncas, ordering, removed, sector):
    """Apply U = prod_s (X_{q_s} + S_s)/sqrt(2), then project each removed
    qubit onto its sector X-eigenstate and drop it (wire 0 = MSB)."""
    n_qubits = 2 * ncas
    b = np.arange(vec.shape[0])
    psi = vec.astype(np.complex128)
    for s, q in enumerate(removed):
        par = np.zeros_like(b)
        for p in range(ncas):
            w = jw_wire(p, s, ncas, ordering)
            par += (b >> (n_qubits - 1 - w)) & 1
        sval = np.where(par % 2 == 0, 1.0, -1.0)
        flip = b ^ (1 << (n_qubits - 1 - q))
        psi = (psi[flip] + sval * psi) / np.sqrt(2.0)
    t = psi.reshape((2,) * n_qubits)
    for s, q in sorted(enumerate(removed), key=lambda e: -e[1]):
        x_eig = np.array([1.0, sector[s]]) / np.sqrt(2.0)
        t = np.tensordot(t, x_eig, axes=([q], [0]))
    return t.reshape(-1)


@pytest.mark.parametrize("ordering", ["blocked", "interleaved"])
@pytest.mark.parametrize("case", ["cas44", "cas31"])
def test_taper_preserves_expectations(cas44_ensemble, cas31_ensemble, case,
                                      ordering):
    """Tr(rho P) == sign * Tr(rho_tapered P') for every string, against an
    explicitly constructed Clifford + sector projection."""
    from qthermal.encode import (extended_heisenberg_expectations,
                                 taper_extended_heisenberg)

    if case == "cas44":
        _, _, aspace, ens = cas44_ensemble
    else:
        aspace, ens = cas31_ensemble
    ncas, nal, nbe = aspace.ncas, aspace.nalpha, aspace.nbeta
    labels, signs, kept = taper_extended_heisenberg(ncas, nal, nbe, ordering)
    assert len(labels) == len(signs) == 4 * ncas ** 2 - ncas
    assert len(kept) == 2 * ncas - 2
    assert set(np.unique(signs)) <= {-1, 1}

    w = np.exp(-(ens.E - ens.E[0]) / 0.5)
    w /= w.sum()
    vals = extended_heisenberg_expectations(ens.vecs, w, ncas, nal, nbe,
                                            ordering)

    removed = [max(jw_wire(p, s, ncas, ordering) for p in range(ncas))
               for s in (0, 1)]
    sector = [(-1) ** nal, (-1) ** nbe]
    V = encode_jw(ens.vecs, ncas, nal, nbe, ordering)
    T = np.stack([_taper_state(v, ncas, ordering, removed, sector)
                  for v in V])
    # states live entirely in the symmetry sector: the projection is lossless
    np.testing.assert_allclose(np.linalg.norm(T, axis=1), 1.0, atol=1e-12)

    P = {"X": pennylane.PauliX, "Y": pennylane.PauliY, "Z": pennylane.PauliZ}
    for label, sign, val in zip(labels, signs, vals):
        factors = [P[t[0]](int(t[1:])) for t in label.split()]
        op = factors[0]
        for f in factors[1:]:
            op = op @ f
        M = pennylane.matrix(op, wire_order=range(2 * ncas - 2))
        ref = sum(wk * np.real(tk.conj() @ M @ tk) for wk, tk in zip(w, T))
        np.testing.assert_allclose(val, sign * ref, atol=1e-10,
                                   err_msg=f"{label} (sign {sign})")


def test_thermal_density_matrix_spectrum_is_the_weights(cas44_ensemble):
    _, _, aspace, ens = cas44_ensemble
    w = np.exp(-(ens.E - ens.E[0]) / 0.05)
    w /= w.sum()
    states = encode_jw(ens.vecs, aspace.ncas, aspace.nalpha, aspace.nbeta)
    rho = thermal_density_matrix(states, w)
    np.testing.assert_allclose(np.trace(rho).real, 1.0, atol=1e-12)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
    evals = np.linalg.eigvalsh(rho)
    np.testing.assert_allclose(np.sort(evals)[::-1][:len(w)],
                               np.sort(w)[::-1], atol=1e-12)
