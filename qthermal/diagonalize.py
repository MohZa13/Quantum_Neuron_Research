"""Module E — sector Hamiltonian construction and the spectral-solver seam.

Builds the active-space Hamiltonian in the (nelecas, S_z = 0) determinant
sector and produces a truncated thermal-ready spectrum.

Solver seam
-----------
Phase 1 ships only :class:`DenseEDSolver`, but the :class:`SpectralSolver`
protocol is the module boundary: tensor-network backends (DMRG low-energy
spectra, METTS/purification thermal sampling, e.g. via ``block2`` or
``quimb``) slot in for active spaces where dense ED is infeasible. The
interface therefore never assumes a full spectrum exists. Every backend must
return a :class:`TruncatedEnsemble` with a *certified* ``tail_weight`` bound
on the discarded Boltzmann weight at ``kT_max``:

- dense ED: exact (computed from the full spectrum);
- DMRG (Phase 2): a variational bound valid at low kT;
- METTS (Phase 2): sampling error bars standing in for the bound.

CI-vector layout is PySCF's opaque ``(na, nb)`` string ordering; only
``pyscf.fci`` kernels (``absorb_h1e``, ``contract_2e``, ``make_rdm1``) touch
it — determinant indexing is never hand-rolled.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from pyscf import fci

from qthermal.active_space import ActiveSpace
from qthermal.thermal import truncate_prefix

logger = logging.getLogger(__name__)

DIM_SILENT_MAX = 5_000
DIM_HARD_MAX = 70_000
_ASYM_TOL = 1e-8


class SectorTooLargeError(RuntimeError):
    """Sector dimension beyond what a solver supports."""


@dataclass
class TruncatedEnsemble:
    """Thermal-ready truncated spectrum — the only object crossing the
    Module E boundary.

    Energies ``E`` exclude ``ecore`` (Module D holds that scalar; both
    conventions are stored downstream). ``evals_full`` is a dense-backend
    extra for provenance; readers must not rely on it — tensor-network
    backends will not provide it.
    """

    E: np.ndarray                       # (m,) ascending, ecore NOT included
    vecs: np.ndarray | None             # (m, dim) CI vectors, or equivalent state repr
    tail_weight: float                  # certified discarded Boltzmann weight at kT_max
    solver_name: str
    kT_max: float
    weight_cutoff: float
    cap_hit: bool = False               # hard cap bound instead of the cutoff
    evals_full: np.ndarray | None = None


@runtime_checkable
class SpectralSolver(Protocol):
    def solve(self, h1eff: np.ndarray, g: np.ndarray | None,
              aspace: ActiveSpace, kT_max: float,
              weight_cutoff: float) -> TruncatedEnsemble: ...


def _available_ram_bytes() -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    import os
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def build_sector_hamiltonian(h1eff: np.ndarray, g: np.ndarray | None,
                             aspace: ActiveSpace) -> np.ndarray:
    """Dense (dim, dim) Hamiltonian in the (nelecas, S_z=0) sector.

    ``g=None`` means the non-interacting (Gaussian) reference: two-electron
    integrals identically zero, one-body ``h1eff`` only.
    """
    ncas, nelec = aspace.ncas, (aspace.nalpha, aspace.nbeta)
    na, nb, dim = aspace.na_strings, aspace.nb_strings, aspace.dim
    if g is None:
        g = np.zeros((ncas,) * 4)
    h2e = fci.direct_spin1.absorb_h1e(h1eff, g, ncas, nelec, 0.5)

    H = np.empty((dim, dim), dtype=np.float64)
    v = np.zeros(dim)
    for j in range(dim):
        v[j] = 1.0
        H[:, j] = fci.direct_spin1.contract_2e(
            h2e, v.reshape(na, nb), ncas, nelec).ravel()
        v[j] = 0.0

    asym = float(np.abs(H - H.T).max())
    assert asym < _ASYM_TOL, f"sector Hamiltonian asymmetry {asym:.3e} > {_ASYM_TOL}"
    return 0.5 * (H + H.T)


class DenseEDSolver:
    """Phase-1 exact-diagonalization backend (full ``numpy.linalg.eigh``).

    Guardrails: silent up to dim=5,000; requires ``allow_large_dense`` plus a
    RAM precheck (>= 3 * dim^2 * 8 bytes available) up to 70,000; refuses
    beyond that — use the Phase-2 tensor-network backend (DMRG/METTS via
    block2 or quimb) for larger sectors.
    """

    name = "dense_ed"

    def __init__(self, allow_large_dense: bool = False):
        self.allow_large_dense = allow_large_dense

    def _check_guardrails(self, dim: int) -> None:
        if dim <= DIM_SILENT_MAX:
            return
        if dim > DIM_HARD_MAX:
            raise SectorTooLargeError(
                f"sector dimension {dim} > {DIM_HARD_MAX}: dense ED refused; "
                "this active space needs the Phase-2 tensor-network solver "
                "(DMRG/METTS)")
        if not self.allow_large_dense:
            raise SectorTooLargeError(
                f"sector dimension {dim} > {DIM_SILENT_MAX}: pass "
                "--allow-large-dense to proceed with dense ED")
        needed = 3 * dim * dim * 8
        avail = _available_ram_bytes()
        if avail < needed:
            raise SectorTooLargeError(
                f"dense ED at dim={dim} needs >= {needed / 1e9:.1f} GB free "
                f"RAM, only {avail / 1e9:.1f} GB available")

    def solve(self, h1eff: np.ndarray, g: np.ndarray | None,
              aspace: ActiveSpace, kT_max: float,
              weight_cutoff: float = 1e-6) -> TruncatedEnsemble:
        dim = aspace.dim
        self._check_guardrails(dim)

        t0 = time.perf_counter()
        H = build_sector_hamiltonian(h1eff, g, aspace)
        t1 = time.perf_counter()
        evals, evecs = np.linalg.eigh(H)
        t2 = time.perf_counter()
        logger.info("dense ED dim=%d: build %.1fs, eigh %.1fs",
                    dim, t1 - t0, t2 - t1)

        m, _, tail, capped = truncate_prefix(
            evals, kT_max, weight_cutoff, cap=max(1024, dim // 4))
        if capped:
            logger.warning(
                "ensemble cap bound at kT_max=%.3g: kept %d states, "
                "tail weight %.3e exceeds cutoff %.1e", kT_max, m, tail,
                weight_cutoff)

        return TruncatedEnsemble(
            E=evals[:m].copy(),
            vecs=np.ascontiguousarray(evecs[:, :m].T),
            tail_weight=float(tail),
            solver_name=self.name,
            kT_max=float(kT_max),
            weight_cutoff=float(weight_cutoff),
            cap_hit=bool(capped),
            evals_full=evals,
        )
