"""Module I: encode eigenblock thermal states as qubit inputs.

Turns the (m, dim) CI eigenvector block + Boltzmann weights produced by the
solvers into qubit-register states for the Fermi-Dirac classifier, two ways:

- Jordan-Wigner (`encode_jw`): one qubit per active spin-orbital,
  Q = 2*ncas, two wire layouts:
  * `ordering="blocked"` — wires 0..ncas-1 are the alpha spin-orbitals
    (ascending active-orbital index), wires ncas..2*ncas-1 the beta.
    PySCF's determinant convention applies alpha creators (ascending
    orbital) before beta creators (ascending), which is exactly
    ascending-wire creation order under this layout — so every
    Jordan-Wigner reordering sign is +1 and the map is a pure scatter.
  * `ordering="interleaved"` — alpha_p on wire 2p, beta_p on wire 2p+1
    (same-orbital pairs adjacent: better for nearest-neighbor classifier
    ansaetze and MPS bond dimension). Re-sorting creators to ascending-wire
    order moves each occupied beta_q past every occupied alpha_p with
    p > q, so each determinant picks up the parity
    (-1)**sum_p n_alpha(p) * |{q < p : n_beta(q) = 1}|.
  Both sign conventions are enforced by `tests/qthermal/test_encode.py`,
  which checks the encoded vectors are eigenvectors of the independently
  JW-mapped Hamiltonian in the matching wire convention.
- Sector compression (`encode_sector`): determinants are relabeled
  0..dim-1 and padded to Q = ceil(log2(dim)) qubits. Loses fermionic
  locality (irrelevant to the abstract-wire classifier), saves ~4 qubits.

Bit convention matches PennyLane: wire 0 is the most significant bit of the
computational-basis index.

`jw_hamiltonian` builds the active-space Hamiltonian as a PennyLane operator
via qml.jordan_wigner, for validation and for inspecting the Pauli strings.
PennyLane is imported lazily so the encoders stay dependency-light.

`extended_heisenberg_paulis` / `_labels` / `_expectations` define the
weight-<=2 classifier operator basis (Z, all-pair ZZ, same-spin XX/YY) and
evaluate Tr(rho P) over it directly in the determinant basis; the batch CLI
over a run file is `qthermal.encode_run`.
"""

from __future__ import annotations

import numpy as np
from pyscf import fci


def _occupation_strings(ncas: int, nelec: int) -> np.ndarray:
    """PySCF alpha/beta strings as bitmask ints (bit p = orbital p occupied)."""
    return np.asarray(fci.cistring.make_strings(range(ncas), nelec),
                      dtype=np.int64)


def jw_wire(orbital: int, spin: int, ncas: int, ordering: str) -> int:
    """Wire index of spin-orbital (orbital, spin); spin 0 = alpha, 1 = beta."""
    if ordering == "blocked":
        return orbital + spin * ncas
    if ordering == "interleaved":
        return 2 * orbital + spin
    raise ValueError(f"unknown ordering {ordering!r}")


def jw_basis_indices(ncas: int, nalpha: int, nbeta: int,
                     ordering: str = "blocked") -> np.ndarray:
    """Computational-basis index for every determinant, PySCF sector order.

    Returns shape (na_strings * nb_strings,), aligned with `civec.ravel()`.
    Computed once per (ncas, nalpha, nbeta) — shared by all molecules.
    """
    n_qubits = 2 * ncas

    def block(strings: np.ndarray, spin: int) -> np.ndarray:
        idx = np.zeros(len(strings), dtype=np.int64)
        for p in range(ncas):
            wire = jw_wire(p, spin, ncas, ordering)
            idx |= ((strings >> p) & 1) << (n_qubits - 1 - wire)
        return idx

    idx_a = block(_occupation_strings(ncas, nalpha), 0)
    idx_b = block(_occupation_strings(ncas, nbeta), 1)
    return (idx_a[:, None] | idx_b[None, :]).ravel()


def jw_parity_signs(ncas: int, nalpha: int, nbeta: int,
                    ordering: str = "blocked") -> np.ndarray:
    """Fermionic reordering sign per determinant, PySCF sector order.

    PySCF creates alpha (ascending orbital) before beta (ascending); the
    encoded basis state is built by creators in ascending *wire* order.
    Blocked ordering preserves the creator order (all signs +1); interleaved
    moves occupied beta_q past occupied alpha_p for every p > q.
    """
    stra = _occupation_strings(ncas, nalpha)
    strb = _occupation_strings(ncas, nbeta)
    if ordering == "blocked":
        return np.ones(len(stra) * len(strb))
    # crossings[ia, ib] = sum_p n_alpha(p) * popcount(strb & ((1<<p)-1))
    alpha_bits = np.array([[(a >> p) & 1 for p in range(ncas)] for a in stra])
    beta_prefix = np.array([[bin(int(b) & ((1 << p) - 1)).count("1")
                             for p in range(ncas)] for b in strb])
    crossings = alpha_bits @ beta_prefix.T
    return np.where(crossings % 2 == 0, 1.0, -1.0).ravel()


def encode_jw(vecs: np.ndarray, ncas: int, nalpha: int, nbeta: int,
              ordering: str = "blocked") -> np.ndarray:
    """Scatter (m, dim) CI vectors into (m, 2**(2*ncas)) JW state vectors."""
    vecs = np.atleast_2d(vecs)
    idx = jw_basis_indices(ncas, nalpha, nbeta, ordering)
    if vecs.shape[1] != idx.shape[0]:
        raise ValueError(
            f"civecs have dim {vecs.shape[1]}, sector expects {idx.shape[0]}")
    signs = jw_parity_signs(ncas, nalpha, nbeta, ordering)
    out = np.zeros((vecs.shape[0], 1 << (2 * ncas)), dtype=np.complex128)
    out[:, idx] = vecs * signs
    return out


def encode_sector(vecs: np.ndarray) -> np.ndarray:
    """Pad (m, dim) CI vectors to (m, 2**ceil(log2 dim)) qubit states."""
    vecs = np.atleast_2d(vecs)
    dim = vecs.shape[1]
    n_qubits = max(1, int(np.ceil(np.log2(dim))))
    out = np.zeros((vecs.shape[0], 1 << n_qubits), dtype=np.complex128)
    out[:, :dim] = vecs
    return out


def thermal_density_matrix(states: np.ndarray,
                           weights: np.ndarray) -> np.ndarray:
    """rho = sum_i w_i |psi_i><psi_i| from encoded state vectors."""
    states = np.atleast_2d(states)
    weights = np.asarray(weights, dtype=np.float64)
    if states.shape[0] != weights.shape[0]:
        raise ValueError("one weight per state required")
    return np.einsum("i,ip,iq->pq", weights, states, states.conj(),
                     optimize=True)


# c_p = sum_{rc} T[p, r, c] m[r, c] for one qubit: the (I, X, Y, Z)
# coefficients of a 2x2 block, normalized so m = sum_p c_p sigma_p.
_PAULI_T = np.zeros((4, 2, 2), dtype=np.complex128)
_PAULI_T[0, 0, 0] = _PAULI_T[0, 1, 1] = 0.5          # I
_PAULI_T[1, 0, 1] = _PAULI_T[1, 1, 0] = 0.5          # X
_PAULI_T[2, 0, 1], _PAULI_T[2, 1, 0] = 0.5j, -0.5j   # Y
_PAULI_T[3, 0, 0], _PAULI_T[3, 1, 1] = 0.5, -0.5     # Z

_PAULI_CHARS = "IXYZ"


def pauli_components(matrix: np.ndarray) -> np.ndarray:
    """All 4**Q Pauli coefficients of a 2**Q x 2**Q matrix, O(Q 4**Q).

    Returns c with matrix = sum_P c[P] * P; expectation Tr[P rho] is
    c[P] * 2**Q. Index P is base-4 (digits I=0, X=1, Y=2, Z=3), wire 0 the
    most significant digit — consistent with `encode_jw`'s bit convention.
    Exact per-qubit transform, no enumeration of Pauli words.
    """
    dim = matrix.shape[0]
    n_qubits = int(np.log2(dim))
    if matrix.shape != (dim, dim) or 1 << n_qubits != dim:
        raise ValueError("matrix must be square with power-of-two dimension")
    a = matrix.astype(np.complex128).reshape(1, dim, dim)
    for k in range(n_qubits):
        half = a.shape[1] // 2
        a = a.reshape(a.shape[0], 2, half, 2, half)
        a = np.einsum("prc,arxcy->apxy", _PAULI_T, a, optimize=True)
        a = a.reshape(a.shape[0] * 4, half, half)
    return a.reshape(-1)


def pauli_label(index: int, n_qubits: int) -> str:
    """Human-readable label ('Z0 X3 ...') for a `pauli_components` index."""
    digits = []
    for wire in range(n_qubits - 1, -1, -1):
        digits.append(_PAULI_CHARS[(index >> (2 * wire)) & 3])
    parts = [f"{p}{w}" for w, p in enumerate(digits) if p != "I"]
    return " ".join(parts) or "I"


def extended_heisenberg_paulis(ncas: int, ordering: str = "blocked"):
    """Symmetry-complete quadratic ansatz for JW-encoded thermal states.

    The Fermi-Dirac classifier's trainable operator H(w) = sum_j w_j P_j,
    restricted to the weight-<=2 strings that are not identically zero on
    number-conserving, real, S_z = 0 thermal states (see 2026-07-14
    measurement): Z on every wire, ZZ on every wire pair (no distance
    cutoff — connected correlations do not decay with wire distance), and
    XX/YY within each spin block only (cross-spin pairs vanish by S_z
    conservation; XY/ZX/single-X classes vanish by parity superselection
    and time-reversal). 4*ncas**2 - ncas terms; PennyLane operators on
    wires 0..2*ncas-1.
    """
    import pennylane as qml

    n_qubits = 2 * ncas
    ops = [qml.PauliZ(w) for w in range(n_qubits)]
    ops += [qml.PauliZ(i) @ qml.PauliZ(j)
            for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    for pauli in (qml.PauliX, qml.PauliY):
        for spin in (0, 1):
            wires = sorted(jw_wire(p, spin, ncas, ordering)
                           for p in range(ncas))
            ops += [pauli(wires[a]) @ pauli(wires[b])
                    for a in range(ncas) for b in range(a + 1, ncas)]
    return ops


def extended_heisenberg_labels(ncas: int, ordering: str = "blocked") -> list[str]:
    """`pauli_label`-style labels for `extended_heisenberg_paulis`, same order."""
    n_qubits = 2 * ncas
    labels = [f"Z{w}" for w in range(n_qubits)]
    labels += [f"Z{i} Z{j}"
               for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    for p in "XY":
        for spin in (0, 1):
            wires = sorted(jw_wire(q, spin, ncas, ordering)
                           for q in range(ncas))
            labels += [f"{p}{wires[a]} {p}{wires[b]}"
                       for a in range(ncas) for b in range(a + 1, ncas)]
    return labels


def extended_heisenberg_expectations(civecs: np.ndarray, weights: np.ndarray,
                                     ncas: int, nalpha: int, nbeta: int,
                                     ordering: str = "blocked") -> np.ndarray:
    """Tr(rho P) for every `extended_heisenberg_paulis` string, same order.

    rho = sum_k weights[k] |psi_k><psi_k| for the JW-encoded (real) CI vectors;
    weights are used as given, so a truncated block yields Tr rho = sum(weights)
    and the caller keeps the trace deficit explicit. Everything is contracted
    in the determinant basis — the 2**(2*ncas) register is never materialized:

    - Z / ZZ strings are diagonal: they read the thermal determinant
      distribution W[Ia, Ib] (JW parity signs square away).
    - Bare same-spin XX (= YY on a fixed-(nalpha, nbeta), real sector: their
      matrix elements agree on the number-conserving pairs and the
      pair-creating ones annihilate the sector) hops one electron between two
      orbitals of one spin, so it contracts the alpha/beta Gram matrices
      G_a = sum_k w_k D_k D_k^T, G_b = sum_k w_k D_k^T D_k over string pairs
      differing by that hop. Parity signs are folded into D_k, so both wire
      orderings are exact.
    """
    civecs = np.atleast_2d(np.asarray(civecs))
    if np.iscomplexobj(civecs):
        raise ValueError("extended_heisenberg_expectations expects real CI "
                         "vectors (solver eigenvectors)")
    civecs = civecs.astype(np.float64, copy=False)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (civecs.shape[0],):
        raise ValueError("one weight per CI vector required")

    stra = _occupation_strings(ncas, nalpha)
    strb = _occupation_strings(ncas, nbeta)
    na, nb = len(stra), len(strb)
    if civecs.shape[1] != na * nb:
        raise ValueError(
            f"civecs have dim {civecs.shape[1]}, sector expects {na * nb}")

    signs = jw_parity_signs(ncas, nalpha, nbeta, ordering).reshape(na, nb)
    D = civecs.reshape(-1, na, nb) * signs        # qubit-basis amplitudes
    W = np.einsum("k,kab->ab", weights, D * D)    # thermal det distribution
    Ga = np.einsum("k,kab,kcb->ac", weights, D, D, optimize=True)
    Gb = np.einsum("k,kab,kac->bc", weights, D, D, optimize=True)

    # z[I, p] = 1 - 2 n_p(string I), the Z eigenvalue on orbital p's wire
    za = 1.0 - 2.0 * ((stra[:, None] >> np.arange(ncas)) & 1)
    zb = 1.0 - 2.0 * ((strb[:, None] >> np.arange(ncas)) & 1)
    Wa, Wb = W.sum(axis=1), W.sum(axis=0)

    n_qubits = 2 * ncas
    spin_orb = {jw_wire(p, s, ncas, ordering): (s, p)
                for p in range(ncas) for s in (0, 1)}

    vals = []
    for w in range(n_qubits):                     # Z singles, wire order
        s, p = spin_orb[w]
        vals.append(za[:, p] @ Wa if s == 0 else zb[:, p] @ Wb)
    for i in range(n_qubits):                     # ZZ, lexicographic pairs
        si, pi = spin_orb[i]
        for j in range(i + 1, n_qubits):
            sj, pj = spin_orb[j]
            if si == sj:
                z, Ws = (za, Wa) if si == 0 else (zb, Wb)
                vals.append((z[:, pi] * z[:, pj]) @ Ws)
            elif si == 0:
                vals.append(za[:, pi] @ W @ zb[:, pj])
            else:
                vals.append(za[:, pj] @ W @ zb[:, pi])

    hops = {}                                     # (spin, a, b) -> 2 sum G
    for s in (0, 1):
        G, strings = (Ga, stra) if s == 0 else (Gb, strb)
        addr = {int(v): i for i, v in enumerate(strings)}
        for a in range(ncas):
            for b in range(a + 1, ncas):
                mask = (1 << a) | (1 << b)
                hops[s, a, b] = 2.0 * sum(
                    G[i, addr[int(v) ^ mask]]
                    for i, v in enumerate(strings)
                    if (v >> a) & 1 and not (v >> b) & 1)
    for _ in ("X", "Y"):                          # XX block, then YY = XX
        for s in (0, 1):
            vals.extend(hops[s, a, b] for a in range(ncas)
                        for b in range(a + 1, ncas))
    return np.asarray(vals)


def taper_extended_heisenberg(ncas: int, nalpha: int, nbeta: int,
                              ordering: str = "blocked"):
    """Z2-taper the extended-Heisenberg basis onto 2*ncas - 2 qubits.

    The JW register carries two Pauli symmetries — the alpha- and beta-block
    parities prod_p Z on that block's wires — whose eigenvalue on the fixed
    (nalpha, nbeta) sector is (-1)**n_s. Every extended-Heisenberg string
    commutes with both (Z-type trivially; XX/YY flip two occupations within
    one block), so the basis tapers term for term with expectations invariant:

        Tr(rho P_k) = signs[k] * Tr(rho_tapered P'_k)

    One qubit per spin block is removed — the highest wire of each. A string
    touching a removed wire inherits that block's remaining Z-chain (in the
    sector, Z on the removed wire equals the parity of the rest of its
    block, up to the sector sign), so tapered words can reach weight ncas;
    the term count never changes and no two strings collide.

    Returns ``(labels, signs, kept_wires)``: labels in `pauli_label` style on
    contiguous new wires 0..2*ncas-3 (new wire i is original wire
    ``kept_wires[i]``), signs an int8 array of +/-1.
    """
    import pennylane as qml

    n_qubits = 2 * ncas
    removed = [max(jw_wire(p, s, ncas, ordering) for p in range(ncas))
               for s in (0, 1)]
    generators = [
        qml.Hamiltonian([1.0], [qml.prod(*(qml.PauliZ(jw_wire(p, s, ncas,
                                                              ordering))
                                           for p in range(ncas)))])
        for s in (0, 1)]
    paulixops = [qml.PauliX(w) for w in removed]
    sector = [(-1) ** nalpha, (-1) ** nbeta]

    kept = [w for w in range(n_qubits) if w not in removed]

    labels, signs = [], []
    for op in extended_heisenberg_paulis(ncas, ordering):
        tapered = qml.simplify(qml.taper(qml.Hamiltonian([1.0], [op]),
                                         generators, paulixops, sector))
        # qml.taper already relabels the kept wires contiguously (ascending
        # original order), i.e. tapered wire i is original wire kept[i].
        (word, coeff), = tapered.pauli_rep.items()
        coeff = float(np.real(coeff))
        if abs(abs(coeff) - 1.0) > 1e-9 or not len(word):
            raise RuntimeError(f"unexpected taper of {op}: {tapered}")
        labels.append(" ".join(f"{p}{w}" for w, p in sorted(word.items())))
        signs.append(1 if coeff > 0 else -1)
    return labels, np.asarray(signs, dtype=np.int8), kept


def jw_hamiltonian(h1eff: np.ndarray, g: np.ndarray, ncas: int,
                   ecore: float = 0.0, ordering: str = "blocked"):
    """Active-space Hamiltonian as a PennyLane operator via Jordan-Wigner.

    H = sum_{pq,s} h1eff[p,q] a+_{ps} a_{qs}
      + 1/2 sum_{pqrs,st} g[p,q,r,s] a+_{ps} a+_{rt} a_{st} a_{qs'}   (chemist (pq|rs))
      + ecore

    Spin-orbital -> wire map matches `encode_jw` for the same `ordering`.
    """
    import pennylane as qml
    from pennylane.fermi import FermiA, FermiC, FermiSentence, FermiWord

    def wire(p, spin):
        return jw_wire(p, spin, ncas, ordering)

    sentence = FermiSentence({FermiWord({}): ecore})
    for p in range(ncas):
        for q in range(ncas):
            if h1eff[p, q] == 0.0:
                continue
            for s in (0, 1):
                sentence += h1eff[p, q] * (FermiC(wire(p, s))
                                           * FermiA(wire(q, s)))
    for p in range(ncas):
        for q in range(ncas):
            for r in range(ncas):
                for s in range(ncas):
                    coeff = 0.5 * g[p, q, r, s]
                    if coeff == 0.0:
                        continue
                    for s1 in (0, 1):
                        for s2 in (0, 1):
                            sentence += coeff * (FermiC(wire(p, s1))
                                                 * FermiC(wire(r, s2))
                                                 * FermiA(wire(s, s2))
                                                 * FermiA(wire(q, s1)))
    sentence.simplify()
    op = qml.jordan_wigner(sentence)
    return qml.simplify(op)
