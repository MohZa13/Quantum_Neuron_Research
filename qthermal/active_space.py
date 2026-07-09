"""Module C — fixed frontier-window active-space selection (no AVAS in Phase 1).

The active space is the contiguous window of ``n_act_occ`` highest occupied
plus ``n_act_virt`` lowest virtual orbitals. All downstream dimensions must be
derived from the :class:`ActiveSpace` object — nothing may hardcode ncas=8,
(4, 4), or dim=4900.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyscf import fci


@dataclass(frozen=True)
class ActiveSpace:
    """Frontier active window: indices plus all derived sector dimensions."""

    active_idx: np.ndarray   # ncas orbital indices, nocc-n_act_occ ... nocc+n_act_virt-1
    core_idx: np.ndarray     # frozen doubly-occupied orbitals, 0 ... nocc-n_act_occ-1
    n_act_occ: int
    n_act_virt: int

    @property
    def ncas(self) -> int:
        return len(self.active_idx)

    @property
    def ncore(self) -> int:
        return len(self.core_idx)

    @property
    def nelecas(self) -> int:
        return 2 * self.n_act_occ

    @property
    def nalpha(self) -> int:
        return self.nelecas // 2

    @property
    def nbeta(self) -> int:
        return self.nelecas // 2

    @property
    def na_strings(self) -> int:
        return fci.cistring.num_strings(self.ncas, self.nalpha)

    @property
    def nb_strings(self) -> int:
        return fci.cistring.num_strings(self.ncas, self.nbeta)

    @property
    def dim(self) -> int:
        """(nelecas, S_z = 0) determinant-sector dimension."""
        return self.na_strings * self.nb_strings


def select_active(eps: np.ndarray, nocc: int,
                  n_act_occ: int = 4, n_act_virt: int = 4) -> ActiveSpace:
    """Select the frontier window around the HOMO-LUMO gap.

    Requires nocc > n_act_occ (at least one core orbital remains) and at
    least n_act_virt virtual orbitals. Phase 1 supports only even nelecas
    with S_z = 0, which the closed-shell window guarantees by construction.
    """
    n_act_occ = int(n_act_occ)
    n_act_virt = int(n_act_virt)
    nmo = len(eps)
    nvirt = nmo - nocc

    if n_act_occ < 1 or n_act_virt < 1:
        raise ValueError(f"n_act_occ={n_act_occ}, n_act_virt={n_act_virt} must be >= 1")
    if nocc <= n_act_occ:
        raise ValueError(
            f"nocc={nocc} <= n_act_occ={n_act_occ}: no core orbitals would remain")
    if nvirt < n_act_virt:
        raise ValueError(
            f"only {nvirt} virtual orbitals available, need n_act_virt={n_act_virt}")

    aspace = ActiveSpace(
        active_idx=np.arange(nocc - n_act_occ, nocc + n_act_virt, dtype=np.int64),
        core_idx=np.arange(nocc - n_act_occ, dtype=np.int64),
        n_act_occ=n_act_occ,
        n_act_virt=n_act_virt,
    )
    assert aspace.nelecas % 2 == 0 and aspace.nalpha == aspace.nbeta  # S_z = 0
    assert aspace.ncas == n_act_occ + n_act_virt
    return aspace
