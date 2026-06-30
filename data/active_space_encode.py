#!/usr/bin/env python3
"""
Active-space Slater determinant encoding for QH9 molecular data.

For each molecule in a QH9 Slater group file:
  1. Select n_qubits Löwdin AO positions with the largest average |HOMO|^2 weight.
  2. Extract the top n_occ_active occupied MOs (HOMO-k ... HOMO) restricted to
     those AO positions → matrix shape (n_qubits, n_occ_active).
  3. Build the Slater determinant in the 2^n_qubits-dimensional Fock space:
         psi[I] = det(D_frontier[I, :])  for every n_occ_active-element
                  subset I of {0,...,n_qubits-1}.
  4. Normalise.

The resulting state lives in C^(2^n_qubits) with C(n_qubits, n_occ_active)
non-zero amplitudes.  Different molecules have different D_occ coefficients,
so the states vary even though they all have the same occupation pattern in
the full MO basis.
"""

from __future__ import annotations
from itertools import combinations

import h5py
import numpy as np


def build_active_space_states(
    h5_path: str,
    n_qubits: int,
    n_occ_active: int | None = None,
    dtype=np.complex128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Parameters
    ----------
    h5_path      : path to a QH9 Slater group HDF5 file
    n_qubits     : number of active AO slots  →  state dimension 2^n_qubits
    n_occ_active : HOMO orbitals in active space (default n_qubits // 2)
    dtype        : complex dtype for state vectors

    Returns
    -------
    states       : (n_valid, 2^n_qubits) normalised complex state vectors
    labels       : (n_valid,)  ±1  gap-binary labels
    selected_aos : (n_qubits,) AO indices selected (sorted)
    n_skipped    : molecules dropped because the projected state had near-zero norm
    """
    if n_occ_active is None:
        n_occ_active = n_qubits // 2

    with h5py.File(h5_path, "r") as h:
        D_occ        = h["D_occ"][:]           # (n_mol, nao, nao_padded)
        n_occ_arr    = h["n_occ_spatial"][:]   # (n_mol,)  occupied spatial MOs
        y_raw        = h["y_gap_binary"][:]    # (n_mol,)  0 / 1

    n_mol, nao, _ = D_occ.shape
    dim = 1 << n_qubits

    if n_occ_active > n_qubits:
        raise ValueError(f"n_occ_active={n_occ_active} > n_qubits={n_qubits}")
    if n_qubits > nao:
        raise ValueError(f"n_qubits={n_qubits} > nao={nao}")

    # ── AO selection ──────────────────────────────────────────────────────────
    # Pick the n_qubits Löwdin AOs with the largest average |HOMO|^2 weight.
    homo_weights = np.zeros(nao, dtype=float)
    for s in range(n_mol):
        n_occ = int(n_occ_arr[s])
        homo  = D_occ[s, :, n_occ - 1]        # HOMO column
        homo_weights += np.abs(homo) ** 2
    homo_weights /= n_mol
    selected_aos = np.sort(np.argsort(-homo_weights)[:n_qubits])

    # ── Extract frontier MO block for all molecules ───────────────────────────
    # D_frontier[s] = D_occ[s, selected_aos, HOMO-(n_occ_active-1):HOMO+1]
    # shape (n_mol, n_qubits, n_occ_active)
    D_frontier = np.zeros((n_mol, n_qubits, n_occ_active), dtype=dtype)
    for s in range(n_mol):
        n_occ  = int(n_occ_arr[s])
        hi     = n_occ
        lo     = max(0, n_occ - n_occ_active)
        actual = hi - lo          # may be < n_occ_active for very small molecules
        D_frontier[s, :, :actual] = D_occ[s, selected_aos, :][:, lo:hi]

    # ── Slater determinant amplitudes (vectorised over molecules) ─────────────
    occ_combos   = list(combinations(range(n_qubits), n_occ_active))
    basis_indices = np.array([sum(1 << i for i in I) for I in occ_combos])
    combo_array   = np.array([list(I) for I in occ_combos])  # (n_combos, n_occ_active)

    psi = np.zeros((n_mol, dim), dtype=dtype)
    for combo, bidx in zip(combo_array, basis_indices):
        # submatrices: (n_mol, n_occ_active, n_occ_active)
        sub = D_frontier[:, combo, :]
        psi[:, bidx] = np.linalg.det(sub)

    # ── Normalise ─────────────────────────────────────────────────────────────
    norms     = np.linalg.norm(psi, axis=1)
    valid     = norms > 1e-10
    n_skipped = int((~valid).sum())

    states = psi[valid] / norms[valid, np.newaxis]
    labels = np.where(y_raw[valid] >= 0.5, +1, -1)

    return states, labels, selected_aos, n_skipped


def build_homo_states(
    h5_path: str,
    n_qubits: int,
    dtype=np.complex128,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Use each molecule's HOMO orbital vector directly as an n-qubit state.

    For a group with nao Löwdin AOs:
      - If nao == 2^n_qubits: use all AOs directly (no loss of information).
      - If nao  > 2^n_qubits: select the 2^n_qubits AOs with largest average
        HOMO weight (same selection as active-space encoding).
      - If nao  < 2^n_qubits: zero-pad to 2^n_qubits.

    The HOMO column D_occ[:, n_occ-1] is already normalised in the Löwdin
    basis, so no renormalisation is needed (unless AO selection or padding
    alters the norm).

    Returns
    -------
    states  : (n_mol, 2^n_qubits) normalised complex state vectors
    labels  : (n_mol,)  ±1  gap-binary labels
    nao     : original AO count (for reference)
    """
    with h5py.File(h5_path, "r") as h:
        D_occ     = h["D_occ"][:]
        n_occ_arr = h["n_occ_spatial"][:]
        y_raw     = h["y_gap_binary"][:]

    n_mol, nao, _ = D_occ.shape
    dim = 1 << n_qubits

    homos = np.array(
        [D_occ[s, :, int(n_occ_arr[s]) - 1] for s in range(n_mol)],
        dtype=dtype,
    )

    if nao > dim:
        hw  = np.mean(np.abs(homos) ** 2, axis=0)
        sel = np.sort(np.argsort(-hw)[:dim])
        homos = homos[:, sel]
    elif nao < dim:
        pad   = np.zeros((n_mol, dim - nao), dtype=dtype)
        homos = np.hstack([homos, pad])

    norms  = np.linalg.norm(homos, axis=1, keepdims=True)
    states = homos / np.where(norms > 1e-10, norms, 1.0)
    labels = np.where(y_raw >= 0.5, +1, -1)

    return states, labels, nao


def describe_encoding(h5_path: str, n_qubits: int, n_occ_active: int | None = None) -> None:
    """Print a summary of the encoding without building states."""
    if n_occ_active is None:
        n_occ_active = n_qubits // 2

    with h5py.File(h5_path, "r") as h:
        n_mol     = h["D_occ"].shape[0]
        nao       = h["D_occ"].shape[1]
        n_occ_arr = h["n_occ_spatial"][:]
        gaps      = h["gap_qh9_ev"][:]

    from math import comb
    n_combos = comb(n_qubits, n_occ_active)
    dim      = 1 << n_qubits

    print(f"Group file       : {h5_path}")
    print(f"Molecules        : {n_mol}")
    print(f"AO space (nao)   : {nao}")
    print(f"n_occ range      : {n_occ_arr.min()}–{n_occ_arr.max()}")
    print(f"Gap (eV)         : {gaps.min():.2f}–{gaps.max():.2f}  median {np.median(gaps):.2f}")
    print(f"n_qubits         : {n_qubits}  → state dim {dim}")
    print(f"n_occ_active     : {n_occ_active}  (top {n_occ_active} HOMO orbitals)")
    print(f"Non-zero amps    : C({n_qubits},{n_occ_active}) = {n_combos} of {dim}")
    print(f"AOs selected     : {n_qubits} of {nao} (by avg HOMO weight)")
