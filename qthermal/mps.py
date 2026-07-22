"""Module J — purification matrix-product states from thermal eigenblocks.

A thermal block (Module F) stores the state rho = sum_k p_k |psi_k><psi_k| as an
``(m, dim)`` CI eigenvector block ``civecs`` V plus Boltzmann weights ``p`` — a
rank-m factorization ``rho = V^T diag(p) V``.  This module turns that directly
into a *purification* MPS without ever forming the dim x dim (or 2^Q x 2^Q)
density matrix:

    |Psi> = sum_k sqrt(p_k) |psi_k>_sys (x) |k>_anc,   Tr_anc |Psi><Psi| = rho.

Because the eigenvectors are orthonormal, this is already the Schmidt
decomposition across the system|ancilla cut: the ancilla bond has dimension m
with singular values sqrt(p_k).  The system half is exactly the Jordan-Wigner
encoded block (Module I), so the MPS follows from a plain TT-SVD of

    Psi[k, s_0, ..., s_{Q-1}] = sqrt(p_k) * <s_0...s_{Q-1} | encode_jw(psi_k)>,

with the ancilla as leg 0.  Q = 2*ncas qubit sites (local dim 2), ``blocked`` or
``interleaved`` wire order (Module I).  Site j after the ancilla is JW wire j
(wire 0 = MSB, PennyLane convention), so the alpha|beta cut sits at bond ncas
under ``blocked`` and same-orbital pairs are adjacent under ``interleaved``.

The ancilla bond equals the thermal rank m (temperature / keep-cap), independent
of ordering; only the physical inter-qubit bonds depend on the wire layout.
Untruncated the purification is *exact*; capping physical bonds at ``chi_max`` or
dropping per-bond singular weight below ``tol`` yields ``truncation_error``, an
upper bound on ||Psi - Psi_trunc||_2 (triangle sum of discarded 2-norms).  Since
the partial trace is a contraction under the trace norm,
||rho - rho_trunc||_1 <= 2 * truncation_error.

Cost is bounded by the encoded block (m * 2^Q), never dim^2.  The 2^Q scatter is
the current ceiling — fine through ncas=8 (2^16); larger active spaces need a
sector-basis contraction that never materialises the full qubit register.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qthermal.encode import jw_basis_indices, jw_parity_signs


def _encode_real(civecs: np.ndarray, ncas: int, nalpha: int, nbeta: int,
                 ordering: str) -> np.ndarray:
    """Real (m, 2**(2*ncas)) JW scatter of the CI block (signs folded in).

    Mirrors :func:`qthermal.encode.encode_jw` but stays float64: solver
    eigenvectors are real, so the purification is real and needs half the
    memory of the complex encoder.
    """
    civecs = np.atleast_2d(np.asarray(civecs, dtype=np.float64))
    idx = jw_basis_indices(ncas, nalpha, nbeta, ordering)
    if civecs.shape[1] != idx.shape[0]:
        raise ValueError(
            f"civecs have dim {civecs.shape[1]}, sector expects {idx.shape[0]}")
    signs = jw_parity_signs(ncas, nalpha, nbeta, ordering)
    out = np.zeros((civecs.shape[0], 1 << (2 * ncas)), dtype=np.float64)
    out[:, idx] = civecs * signs
    return out


def _truncation_rank(s: np.ndarray, chi_max: int | None, tol: float) -> int:
    """Kept singular count: drop the tail with 2-norm <= tol, then cap at chi_max."""
    keep = len(s)
    if tol and tol > 0.0:
        tail_sq = np.cumsum(s[::-1] ** 2)
        drop = int(np.searchsorted(tail_sq, tol ** 2, side="right"))
        keep = len(s) - drop
    if chi_max is not None:
        keep = min(keep, int(chi_max))
    return max(1, keep)


def tt_svd(tensor: np.ndarray, chi_max: int | None = None,
           tol: float = 0.0) -> tuple[list[np.ndarray], float]:
    """Left-to-right TT-SVD of a dense tensor into MPS cores.

    ``tensor`` axes are ``(left_bond, site_0, ..., site_{n-1})``: axis 0 is an
    incoming bond (use size 1 for an ordinary open MPS), the rest are physical
    sites.  Returns ``(cores, error)`` where ``cores[i]`` has shape
    ``(chi_left, site_i, chi_right)`` — ``cores[0]``'s left bond is
    ``tensor.shape[0]`` and the right boundary bond is 1 — and ``error`` is the
    triangle-inequality sum of discarded singular 2-norms, an upper bound on
    ``||tensor - reconstruct(cores)||_2``.
    """
    chi_left = tensor.shape[0]
    site_dims = tensor.shape[1:]
    cores: list[np.ndarray] = []
    carry = tensor.reshape(chi_left, -1)
    error = 0.0
    for d in site_dims:
        rest = carry.shape[1] // d
        mat = carry.reshape(chi_left * d, rest)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        keep = _truncation_rank(s, chi_max, tol)
        if keep < len(s):
            error += float(np.linalg.norm(s[keep:]))
        u, s, vh = u[:, :keep], s[:keep], vh[:keep, :]
        cores.append(u.reshape(chi_left, d, keep))
        carry = s[:, None] * vh
        chi_left = keep
    # carry is now (chi_left, 1); fold the residual scalar into the last core.
    cores[-1] = np.tensordot(cores[-1], carry.reshape(chi_left, 1), axes=([2], [0]))
    return cores, error


@dataclass
class PurificationMPS:
    """Purification of a thermal block as an MPS: cores[0] is the ancilla (open
    leg of dimension m = thermal rank), cores[1:] are the 2*ncas qubit sites in
    JW-wire order.  Tracing the ancilla leg gives rho exactly (up to
    ``truncation_error``)."""

    cores: list[np.ndarray]        # length 2*ncas + 1; cores[0] = ancilla
    ordering: str
    ncas: int
    nalpha: int
    nbeta: int
    weights: np.ndarray            # (m,) Boltzmann weights p (unnormalised trace)
    truncation_error: float        # upper bound on ||Psi - Psi_trunc||_2

    @property
    def num_qubits(self) -> int:
        return 2 * self.ncas

    def bond_dims(self) -> list[int]:
        """Right-bond dimension of every core, ancilla bond first."""
        return [c.shape[2] for c in self.cores]

    def physical_bond_dims(self) -> list[int]:
        """Inter-qubit bond dimensions (excludes the ancilla and the right
        boundary bond) — the part the wire ordering actually moves."""
        return [c.shape[2] for c in self.cores[1:-1]]

    def ancilla_bond(self) -> int:
        return self.cores[0].shape[2]

    def max_physical_bond(self) -> int:
        phys = self.physical_bond_dims()
        return max(phys) if phys else 1


def purification_mps(civecs: np.ndarray, p: np.ndarray, ncas: int,
                     nalpha: int, nbeta: int, ordering: str = "blocked",
                     chi_max: int | None = None,
                     tol: float = 0.0) -> PurificationMPS:
    """Build the purification MPS of rho = sum_k p_k |psi_k><psi_k|.

    ``civecs`` is the ``(m, dim)`` eigenblock, ``p`` the ``(m,)`` weights (used
    as given, so a truncated block yields a sub-normalised purification with
    <Psi|Psi> = sum p).  ``chi_max`` / ``tol`` cap the physical bonds; the
    ancilla bond is always the thermal rank m.  The largest object built is the
    m x 2^Q encoded block, never dim x dim.
    """
    p = np.asarray(p, dtype=np.float64)
    civecs = np.atleast_2d(np.asarray(civecs, dtype=np.float64))
    if p.shape != (civecs.shape[0],):
        raise ValueError("one weight per CI vector required")

    enc = _encode_real(civecs, ncas, nalpha, nbeta, ordering)   # (m, 2^Q)
    weighted = np.sqrt(p)[:, None] * enc                         # ancilla-weighted
    m = weighted.shape[0]

    # The ancilla|system Schmidt form is already exact (orthonormal eigenvectors,
    # weights sqrt(p)), so the ancilla core is a pure identity injection and its
    # bond stays at the full thermal rank m — no SVD, no truncation there. Only
    # the Q qubit sites are TT-SVD'd, with the ancilla as their incoming bond.
    tensor = weighted.reshape([m] + [2] * (2 * ncas))           # (anc bond, wires)
    phys_cores, error = tt_svd(tensor, chi_max=chi_max, tol=tol)
    ancilla_core = np.eye(m).reshape(1, m, m)
    cores = [ancilla_core, *phys_cores]
    return PurificationMPS(cores=cores, ordering=ordering, ncas=ncas,
                           nalpha=nalpha, nbeta=nbeta, weights=p.copy(),
                           truncation_error=float(error))


def to_dense_ket(mps: PurificationMPS) -> np.ndarray:
    """Contract the MPS back to the dense purification, shape (m, 2**(2*ncas)).

    The leading axis is the ancilla leg.  For validation / small sectors only —
    materialises the full 2^Q register.
    """
    cores = mps.cores
    psi = cores[0]                                    # (1, m, chi0)
    psi = psi.reshape(cores[0].shape[1], cores[0].shape[2])   # (m, chi0), drop left bound
    for core in cores[1:]:
        psi = np.tensordot(psi, core, axes=([-1], [0]))       # (..., d, chi_right)
    return psi.reshape(cores[0].shape[1], -1)                 # (m, 2^Q)


def reduced_density_matrix(mps: PurificationMPS) -> np.ndarray:
    """rho = Tr_anc |Psi><Psi| in the qubit basis, shape (2^Q, 2^Q).

    Validation helper — materialises the dense register, small sectors only.
    """
    ket = to_dense_ket(mps)                            # (m, 2^Q)
    return ket.T @ ket                                 # sum_k |sys_k><sys_k|
