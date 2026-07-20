"""Module E — sector Hamiltonian construction and the spectral-solver seam.

Builds the active-space Hamiltonian in the (nelecas, S_z = 0) determinant
sector and produces a truncated thermal-ready spectrum.

Solver seam
-----------
Phase 1 ships :class:`DenseEDSolver` and the matrix-free
:class:`IterativeWindowSolver`; the :class:`SpectralSolver` protocol is the
module boundary: tensor-network backends (DMRG low-energy spectra,
METTS/purification thermal sampling, e.g. via ``block2`` or ``quimb``) slot
in for active spaces where neither is feasible. The interface therefore never
assumes a full spectrum exists. Every backend must return a
:class:`TruncatedEnsemble` with a *certified* ``tail_weight`` bound on the
discarded Boltzmann weight at ``kT_max``:

- dense ED: exact (computed from the full spectrum);
- iterative Krylov: rigorous counting bound (:func:`krylov_tail_bound`);
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
from qthermal.thermal import resolve_keep_cap, truncate_prefix

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

    Built via ``fci.direct_spin1.pspace`` at ``np=dim`` (the whole sector) —
    PySCF's compiled preconditioner-matrix builder (one C call for the entire
    matrix) rather than looping ``contract_2e`` over dim unit vectors. At
    ``np=dim`` the requested subset is the whole sector, so ``pspace``'s
    internal address list is the identity permutation and its output already
    sits in the same (na, nb) string order as every other module here: ~34x
    faster on a real CAS(8,8) molecule (3.8s -> 0.1s), bit-identical to the
    previous contract_2e-loop implementation to float64 noise (verified
    2026-07-20).
    """
    ncas, nelec = aspace.ncas, (aspace.nalpha, aspace.nbeta)
    dim = aspace.dim
    if g is None:
        g = np.zeros((ncas,) * 4)

    addr, H = fci.direct_spin1.pspace(h1eff, g, ncas, nelec, np=dim)
    assert np.array_equal(addr, np.arange(dim)), \
        "pspace address permutation was not the identity at np=dim"

    asym = float(np.abs(H - H.T).max())
    assert asym < _ASYM_TOL, f"sector Hamiltonian asymmetry {asym:.3e} > {_ASYM_TOL}"
    return 0.5 * (H + H.T)


class DenseEDSolver:
    """Phase-1 exact-diagonalization backend (full ``numpy.linalg.eigh``).

    Guardrails: silent up to dim=5,000; requires ``allow_large_dense`` plus a
    RAM precheck (>= 3 * dim^2 * 8 bytes available) up to 70,000; refuses
    beyond that — use the Phase-2 tensor-network backend (DMRG/METTS via
    block2 or quimb) for larger sectors.

    ``keep_cap`` bounds how many eigenvectors the returned ensemble stores
    (:func:`qthermal.thermal.resolve_keep_cap` semantics: None = default
    ``max(1024, dim // 4)``, 0 = uncapped); the spectrum itself is always
    solved and returned in full via ``evals_full``.
    """

    name = "dense_ed"

    def __init__(self, allow_large_dense: bool = False,
                 keep_cap: int | None = None):
        self.allow_large_dense = allow_large_dense
        self.keep_cap = keep_cap

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
            evals, kT_max, weight_cutoff,
            cap=resolve_keep_cap(self.keep_cap, dim))
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


class NonInteractingSolver:
    """Closed-form solver for the g = 0 (non-interacting) reference sector.

    Used only for the Gaussian-reference audit (Module F) — ``solve`` raises
    if ``g is not None``, and this is deliberately not registered as a
    general ``--solver`` choice (Module H). The non-interacting many-body
    Hamiltonian is diagonal in the basis of Slater determinants built from
    ``h1eff``'s own eigenvectors, so its spectrum is exactly the outer sum of
    every nalpha-subset energy and every nbeta-subset energy of h1eff's
    eigenvalues — an ncas x ncas diagonalization plus combinatorics, never a
    dim x dim matrix. Eigenvectors are recovered in the working determinant
    basis via PySCF's orbital-rotation CI transform,
    ``fci.addons.transform_ci_for_orbital_rotation(det, ncas, nelec, U0.T)``,
    where ``U0`` diagonalizes h1eff. The transpose is load-bearing: it is the
    new-basis -> old-basis map this solver needs, the reverse of what the
    function's docstring states as its forward convention, and was confirmed
    empirically (checking ``H v - E v`` directly) rather than assumed.

    Verified against ``DenseEDSolver(g=None)``: eigenvalues to float64 noise
    (~1e-13) across several (ncas, nelecas) shapes including production
    CAS(8,8), and eigenvectors directly against the sector matrix
    (``|H v - E v|`` ~1e-15, overlap with the true state 1.0 to 10 digits) on
    real QH9 data (2026-07-20). On that same CAS(8,8) system, building the
    ~350 states a kT_max=0.25 run actually keeps takes ~2s versus ~28s for
    the dense build+eigh it replaces; per-state construction cost is linear
    in how many states are kept, so the win shrinks (but never reverses into
    a regression) if ``--keep-cap`` is large enough to need most of the
    sector.

    Always exact and full-spectrum (``evals_full`` is always populated), and
    carries none of the dense solver's dim guardrails — there is no dim x dim
    matrix here to refuse, so the Gaussian audit stays cheap at active-space
    sizes where the interacting solve already needs a Phase-2 backend.
    """

    name = "noninteracting_closed_form"

    def __init__(self, keep_cap: int | None = None):
        self.keep_cap = keep_cap

    def solve(self, h1eff: np.ndarray, g: np.ndarray | None,
              aspace: ActiveSpace, kT_max: float,
              weight_cutoff: float = 1e-6) -> TruncatedEnsemble:
        if g is not None:
            raise ValueError(
                "NonInteractingSolver only solves the g=0 reference sector "
                "(got g is not None); use DenseEDSolver or "
                "IterativeWindowSolver for the interacting problem")

        ncas = aspace.ncas
        nelec = (aspace.nalpha, aspace.nbeta)
        na, nb, dim = aspace.na_strings, aspace.nb_strings, aspace.dim

        eps0, U0 = np.linalg.eigh(h1eff)

        alpha_strings = np.asarray(
            fci.cistring.make_strings(range(ncas), aspace.nalpha), dtype=np.int64)
        beta_strings = np.asarray(
            fci.cistring.make_strings(range(ncas), aspace.nbeta), dtype=np.int64)
        occ_a = (alpha_strings[:, None] >> np.arange(ncas)) & 1
        occ_b = (beta_strings[:, None] >> np.arange(ncas)) & 1
        alpha_energy = occ_a @ eps0
        beta_energy = occ_b @ eps0

        # Flat index ia * nb + ib is the same (na, nb) address convention
        # used throughout (e.g. the old build_sector_hamiltonian's
        # v.reshape(na, nb)), so no reordering is needed downstream.
        pair_energy = (alpha_energy[:, None] + beta_energy[None, :]).ravel()
        order = np.argsort(pair_energy, kind="stable")
        evals_full = pair_energy[order]

        cap = resolve_keep_cap(self.keep_cap, dim)
        m, _, tail, capped = truncate_prefix(evals_full, kT_max, weight_cutoff,
                                             cap=cap)
        if capped:
            logger.warning(
                "Gaussian-reference cap bound at kT_max=%.3g: kept %d states, "
                "tail weight %.3e exceeds cutoff %.1e", kT_max, m, tail,
                weight_cutoff)

        u_inv = U0.T   # new (h1eff-eigenbasis) -> old (working) basis; see
                       # class docstring for the empirical direction check
        det = np.zeros((na, nb))
        vecs = np.empty((m, dim), dtype=np.float64)
        for k in range(m):
            idx = divmod(int(order[k]), nb)
            det[idx] = 1.0
            vecs[k] = np.asarray(fci.addons.transform_ci_for_orbital_rotation(
                det, ncas, nelec, u_inv)).ravel()
            det[idx] = 0.0

        return TruncatedEnsemble(
            E=evals_full[:m].copy(),
            vecs=vecs,
            tail_weight=float(tail),
            solver_name=self.name,
            kT_max=float(kT_max),
            weight_cutoff=float(weight_cutoff),
            cap_hit=bool(capped),
            evals_full=evals_full,
        )


KRYLOV_INIT_NROOTS = 16
KRYLOV_MAX_NROOTS = 512


def krylov_tail_bound(E: np.ndarray, m: int, dim: int, kT: float) -> float:
    """Rigorous upper bound on the discarded Boltzmann weight when only the
    ``m`` lowest of ``k = len(E)`` converged eigenvalues are kept.

    Valid whenever ``E`` holds the ``k`` lowest eigenvalues (ascending) of a
    ``dim``-dimensional sector: every unconverged state lies at or above
    ``E[-1]`` and the partition function is >= the ground-state term, so the
    normalized discarded weight obeys

        tail <= sum_{i=m}^{k-1} e^{-(E_i-E_0)/kT} + (dim-k) e^{-(E_{k-1}-E_0)/kT}
    """
    E = np.asarray(E, dtype=np.float64)
    k = len(E)
    if not 1 <= m <= k <= dim:
        raise ValueError(f"need 1 <= m={m} <= k={k} <= dim={dim}")
    if kT <= 0:
        raise ValueError(f"kT must be positive, got {kT}")
    w = np.exp(-(E - E[0]) / kT)
    return float(w[m:].sum() + (dim - k) * w[-1])


class IterativeWindowSolver:
    """Matrix-free Krylov backend: converges only the low-energy window.

    PySCF's FCI Davidson (``nroots=k``) supplies the k lowest eigenpairs via
    ``contract_2e`` matvecs — the dense (dim, dim) sector matrix is never
    formed, so memory is O(k * dim) and the dense guardrails do not apply.
    ``k`` doubles until :func:`krylov_tail_bound` certifies ``weight_cutoff``
    at ``kT_max`` with at least one converged-but-unkept buffer root, or until
    ``max_nroots`` / available RAM stops escalation (then ``cap_hit=True`` and
    ``tail_weight`` is the certified-but-uncut bound, mirroring the dense cap).

    The certificate stays deterministic (counting bound + Davidson residual
    tolerance) but assumes no eigenvalue below ``E[k-1]`` was missed;
    mitigations are PySCF's pspace-seeded initial guesses, the buffer root,
    and a prefix-consistency check across escalations. ``evals_full`` is never
    provided: per-kT weights downstream normalize over the kept window and
    fold ``tail_weight`` into the truncation error (Module F contract). This
    is the low-kT backend — once the thermal window holds thousands of states
    (e.g. the 0.25 Ha rung at large ncas), escalation will hit the cap and a
    sampling backend (TPQ/METTS, Phase 2) is the right tool instead.
    """

    name = "iterative_krylov"

    def __init__(self, init_nroots: int = KRYLOV_INIT_NROOTS,
                 max_nroots: int = KRYLOV_MAX_NROOTS,
                 conv_tol: float = 1e-10):
        if init_nroots < 2 or max_nroots < 2:
            raise ValueError("need init_nroots >= 2 and max_nroots >= 2 "
                             "(one kept state plus the buffer root)")
        self.init_nroots = int(init_nroots)
        self.max_nroots = int(max_nroots)
        self.conv_tol = float(conv_tol)

    @staticmethod
    def _bytes_needed(k: int, dim: int) -> int:
        # davidson1 subspace is 2*(max_space + 4(k-1)) vectors for xs/ax plus
        # 2k for xt/axt (linalg_helper), k for the returned CI vectors; 1.25x
        # slack for link tables and hdiag.
        return int(1.25 * (11 * k + 32) * dim * 8)

    def _solve_roots(self, h1eff: np.ndarray, g: np.ndarray | None,
                     aspace: ActiveSpace,
                     k: int) -> tuple[np.ndarray, np.ndarray, bool]:
        """k lowest eigenpairs, ascending, plus an all-roots-converged flag.

        Unconverged solves are reported, not raised: exactly degenerate
        multiplets straddling the window edge stall Davidson (common in the
        g=0 reference sector of high-symmetry molecules), and the caller's
        remedy is to widen the window (escalate k) so the multiplet fits.
        """
        ncas, nelec = aspace.ncas, (aspace.nalpha, aspace.nbeta)
        if g is None:
            g = np.zeros((ncas,) * 4)
        cis = fci.direct_spin1.FCISolver()
        cis.verbose = 0
        cis.nroots = k
        cis.conv_tol = self.conv_tol
        cis.max_cycle = 300
        # dim <= pspace_size short-circuits to an exact dense subspace diag,
        # which seeds guesses / catches clustered roots; keep it >= 2k.
        cis.pspace_size = max(400, min(aspace.dim, 2 * k))
        cis.max_memory = max(2000, int(_available_ram_bytes() / 1e6))
        e, ci = cis.kernel(h1eff, g, ncas, nelec)
        e = np.atleast_1d(np.asarray(e, dtype=np.float64))
        if not isinstance(ci, (list, tuple)):
            ci = [ci]
        if len(e) != k:
            raise RuntimeError(f"asked for {k} roots, got {len(e)}")
        converged = bool(np.all(np.atleast_1d(cis.converged)))
        order = np.argsort(e, kind="stable")
        vecs = np.ascontiguousarray(
            np.stack([np.asarray(ci[i], dtype=np.float64).ravel()
                      for i in order]))
        return e[order], vecs, converged

    @staticmethod
    def _certified_m(E: np.ndarray, dim: int, kT: float,
                     cutoff: float) -> int | None:
        """Smallest m <= k-1 whose tail bound meets the cutoff, else None."""
        w = np.exp(-(E - E[0]) / kT)
        suffix = np.cumsum(w[::-1])[::-1]        # suffix[m] = sum_{i>=m} w_i
        bounds = suffix[1:] + (dim - len(E)) * w[-1]   # m = 1 .. k-1
        ok = np.nonzero(bounds <= cutoff)[0]
        return int(ok[0]) + 1 if ok.size else None

    def solve(self, h1eff: np.ndarray, g: np.ndarray | None,
              aspace: ActiveSpace, kT_max: float,
              weight_cutoff: float = 1e-6) -> TruncatedEnsemble:
        dim = aspace.dim
        k_ceiling = max(2, min(self.max_nroots, dim - 1))
        k = max(2, min(self.init_nroots, k_ceiling))
        E = vecs = None
        capped = False
        while True:
            needed, avail = self._bytes_needed(k, dim), _available_ram_bytes()
            if avail < needed:
                if E is None:
                    raise SectorTooLargeError(
                        f"Krylov at dim={dim} needs >= {needed / 1e9:.1f} GB "
                        f"free RAM for k={k} roots, only {avail / 1e9:.1f} GB "
                        "available; lower max_nroots or use a Phase-2 backend")
                logger.warning(
                    "RAM stops escalation beyond k=%d at dim=%d "
                    "(need %.1f GB, have %.1f GB); capping", len(E), dim,
                    needed / 1e9, avail / 1e9)
                m, capped = len(E) - 1, True
                break

            t0 = time.perf_counter()
            E_new, vecs_new, converged = self._solve_roots(h1eff, g, aspace, k)
            logger.info("krylov dim=%d k=%d: %.1fs%s", dim, k,
                        time.perf_counter() - t0,
                        "" if converged else " (unconverged)")
            if not converged:
                # Degenerate multiplet straddling the window edge: widen the
                # window instead of certifying from an invalid solve.
                if k < k_ceiling:
                    logger.warning(
                        "Davidson left roots unconverged at k=%d (likely a "
                        "degenerate multiplet at the window edge); "
                        "escalating", k)
                    k = min(2 * k, k_ceiling)
                    continue
                if E is None:
                    raise RuntimeError(
                        f"Davidson unconverged at the root ceiling k={k} "
                        f"(dim={dim}) with no smaller converged solve to "
                        "fall back on; certified bound unavailable")
                logger.warning(
                    "Davidson unconverged at the root ceiling k=%d; capping "
                    "at the last converged solve (k=%d)", k, len(E))
                m, capped = len(E) - 1, True
                break
            if E is not None:
                # Threshold sits above Davidson reordering jitter on
                # quasi-degenerate roots (~sqrt(conv_tol)) and far below any
                # genuine missed state (which shifts by an inter-state gap).
                n = min(len(E), len(E_new))
                if float(np.abs(E[:n] - E_new[:n]).max()) > 1e-4:
                    logger.warning(
                        "eigenvalue prefix shifted at k=%d (max dev %.2e): "
                        "previously missed low state(s) surfaced", k,
                        float(np.abs(E[:n] - E_new[:n]).max()))
            E, vecs = E_new, vecs_new

            m = self._certified_m(E, dim, kT_max, weight_cutoff)
            if m is not None:
                break
            if k >= k_ceiling:
                m, capped = len(E) - 1, True
                logger.warning(
                    "root cap %d bound at kT_max=%.3g: kept %d states, tail "
                    "bound %.3e exceeds cutoff %.1e — this window wants a "
                    "sampling backend", k_ceiling, kT_max, m,
                    krylov_tail_bound(E, m, dim, kT_max), weight_cutoff)
                break
            k = min(2 * k, k_ceiling)

        tail = krylov_tail_bound(E, m, dim, kT_max)
        return TruncatedEnsemble(
            E=E[:m].copy(),
            vecs=np.ascontiguousarray(vecs[:m]),
            tail_weight=float(tail),
            solver_name=self.name,
            kT_max=float(kT_max),
            weight_cutoff=float(weight_cutoff),
            cap_hit=bool(capped),
            evals_full=None,
        )
