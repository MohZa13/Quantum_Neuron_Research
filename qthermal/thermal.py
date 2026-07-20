"""Module F — thermal ensembles, truncation, and quantumness diagnostics.

All Boltzmann weights come from a numerically stable shifted softmax — raw
energies are never exponentiated. Weights are normalized over the *full*
spectrum when it is available (dense backend) so kept probabilities are exact
and the discarded tail is exact; otherwise the certified ``tail_weight`` of
the ensemble bounds the normalization error.

The Gaussian-reference audit re-solves the sector with the two-electron
integrals zeroed (one-body ``h1eff`` only) through the *same* solver seam, and
reports a subspace-projected trace distance between the interacting and
Gaussian thermal states — no dim x dim density matrix is ever materialized, so
the audit scales to any ncas a solver can handle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from pyscf import fci

from qthermal.active_space import ActiveSpace

logger = logging.getLogger(__name__)

DEFAULT_KT_LIST = (0.05, 0.10, 0.25)          # Hartree
DEFAULT_WEIGHT_CUTOFF = 1e-6


def boltzmann_weights(evals: np.ndarray, kT: float) -> np.ndarray:
    """Stable softmax weights exp(-(E - E0)/kT) / Z over the given energies."""
    if kT <= 0:
        raise ValueError(f"kT must be positive, got {kT}")
    evals = np.asarray(evals, dtype=np.float64)
    p = np.exp(-(evals - evals[0]) / kT)
    return p / p.sum()


def truncate_prefix(evals: np.ndarray, kT: float, weight_cutoff: float,
                    cap: int | None = None) -> tuple[int, np.ndarray, float, bool]:
    """Smallest ascending-energy prefix with cumulative weight >= 1 - cutoff.

    Returns ``(m, weights, tail_weight, capped)`` where ``weights`` is the
    full-spectrum softmax (kept entries are ``weights[:m]``, NOT renormalized)
    and ``tail_weight = 1 - sum(weights[:m])`` is the exact discarded weight.
    ``capped`` reports whether the hard cap bound instead of the cutoff.
    """
    p = boltzmann_weights(evals, kT)
    cum = np.cumsum(p)
    m = int(np.searchsorted(cum, 1.0 - weight_cutoff) + 1)
    m = min(m, len(p))
    capped = cap is not None and m > cap
    if capped:
        m = int(cap)
    tail = float(max(0.0, 1.0 - cum[m - 1]))
    return m, p, tail, capped


def resolve_keep_cap(keep_cap: int | None, dim: int) -> int | None:
    """Effective cap on stored eigenvectors (per ensemble and per kT block).

    ``None`` -> the default ``max(1024, dim // 4)`` storage heuristic;
    ``0`` -> uncapped (the weight cutoff alone decides, up to the full
    sector — mind the m x dim float64 storage); ``N > 0`` -> exactly N.
    """
    if keep_cap is None:
        return max(1024, dim // 4)
    if keep_cap < 0:
        raise ValueError(f"keep_cap must be >= 0, got {keep_cap}")
    if keep_cap == 0:
        return None
    return int(keep_cap)


@dataclass
class ThermalBlock:
    """Everything stored per (molecule, kT): the truncated ensemble plus the
    quantumness audit. ``p`` sums to 1 - truncation_error (not renormalized)."""

    kT: float
    E: np.ndarray                 # (m,) kept energies, ecore not included
    p: np.ndarray                 # (m,) Boltzmann weights
    civecs: np.ndarray            # (m, dim) float64
    truncation_error: float       # discarded Boltzmann weight
    entropy: float                # ensemble (diagonal) entropy -sum p ln p
    nat_occs: np.ndarray          # (ncas,) natural occupations in [0, 2]
    static_corr: float            # sum_i min(n_i, 2 - n_i)
    c_max_sq: float               # leading-determinant weight of the ground state
    tracedist_gaussian: float     # subspace-projected trace distance
    tracedist_bound: float        # additive error bound from both tails
    cap_hit: bool = False


def ensemble_entropy(p: np.ndarray) -> float:
    """Diagonal entropy -sum p ln p (natural log) over the kept weights."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def thermal_rdm1(vecs: np.ndarray, p: np.ndarray, aspace: ActiveSpace) -> np.ndarray:
    """Thermally averaged spin-traced 1-RDM, sum_k p_k rdm1(|E_k>)."""
    na, nb = aspace.na_strings, aspace.nb_strings
    nelec = (aspace.nalpha, aspace.nbeta)
    dm = np.zeros((aspace.ncas, aspace.ncas))
    for w, v in zip(p, vecs):
        dm += w * fci.direct_spin1.make_rdm1(
            np.ascontiguousarray(v.reshape(na, nb)), aspace.ncas, nelec)
    return dm


def natural_occupations(dm1: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of the (spin-traced) 1-RDM, each in [0, 2]."""
    n = np.linalg.eigvalsh(dm1)[::-1]
    assert n.min() > -1e-7 and n.max() < 2.0 + 1e-7, \
        f"natural occupations outside [0, 2]: [{n.min():.3e}, {n.max():.3e}]"
    return np.clip(n, 0.0, 2.0)


def static_correlation_score(nat_occs: np.ndarray) -> float:
    """sum_i min(n_i, 2 - n_i): 0 for a closed-shell determinant."""
    n = np.asarray(nat_occs)
    return float(np.minimum(n, 2.0 - n).sum())


def leading_determinant_weight(ground_vec: np.ndarray) -> float:
    """|c_max|^2 of the ground eigenvector in the determinant basis."""
    return float(np.max(np.abs(ground_vec)) ** 2)


def trace_distance_projected(vecs1: np.ndarray, p1: np.ndarray,
                             vecs2: np.ndarray, p2: np.ndarray) -> float:
    """Trace distance between two low-rank thermal states, computed in the
    joint span of their kept eigenvectors.

    Both density matrices rho = sum_k p_k |v_k><v_k| live entirely inside
    span(vecs1 U vecs2), so projecting there is exact for the kept parts; the
    caller adds 0.5 * (tail_1 + tail_2) as the additive bound for the
    discarded weight. Rank <= m1 + m2; no dim x dim matrix is built.
    """
    V = np.vstack([vecs1, vecs2])            # (m1 + m2, dim)
    Q, _ = np.linalg.qr(V.T)                 # (dim, r) orthonormal columns
    B1 = vecs1 @ Q                           # rows are Q^T v_k
    B2 = vecs2 @ Q
    delta = B1.T @ (p1[:, None] * B1) - B2.T @ (p2[:, None] * B2)
    lam = np.linalg.eigvalsh(delta)
    return float(0.5 * np.abs(lam).sum())


def gaussian_reference_ensemble(h1eff: np.ndarray, aspace: ActiveSpace,
                                kT_max: float,
                                weight_cutoff: float = DEFAULT_WEIGHT_CUTOFF,
                                keep_cap: int | None = None):
    """Solve the g = 0 (non-interacting) sector via the closed-form solver.

    Always uses ``qthermal.diagonalize.NonInteractingSolver``, independent of
    whichever backend solves the interacting problem: the non-interacting
    spectrum has closed form, so this is always exact and never performs a
    dim x dim diagonalization (verified against the previous
    ``solver.solve(h1eff, None, ...)`` path to float64 noise — see
    NonInteractingSolver's docstring). Local import: diagonalize.py already
    imports resolve_keep_cap/truncate_prefix from this module, so a top-level
    import here would be circular.
    """
    from qthermal.diagonalize import NonInteractingSolver
    return NonInteractingSolver(keep_cap=keep_cap).solve(
        h1eff, None, aspace, kT_max=kT_max, weight_cutoff=weight_cutoff)


def _weights_for_kT(ensemble, kT: float, weight_cutoff: float,
                    cap: int) -> tuple[int, np.ndarray, float, bool]:
    """Per-kT truncation of an already-solved ensemble.

    Uses the full spectrum when the backend provided it (weights and tail are
    then exact); otherwise normalizes over the kept energies and folds the
    ensemble's certified tail_weight into the truncation error.
    """
    if ensemble.evals_full is not None:
        m, p, tail, capped = truncate_prefix(ensemble.evals_full, kT,
                                             weight_cutoff, cap=cap)
        if m > len(ensemble.E):
            logger.warning(
                "kT=%.3g needs %d states but only %d kept at kT_max=%.3g; "
                "clamping", kT, m, len(ensemble.E), ensemble.kT_max)
            m = len(ensemble.E)
            tail = float(max(0.0, 1.0 - np.cumsum(p)[m - 1]))
        return m, p[:m], tail, capped

    m, p, tail, capped = truncate_prefix(ensemble.E, kT, weight_cutoff, cap=cap)
    return m, p[:m], tail + ensemble.tail_weight, capped


def build_thermal_block(ensemble, gaussian_ensemble, kT: float,
                        aspace: ActiveSpace,
                        weight_cutoff: float = DEFAULT_WEIGHT_CUTOFF,
                        keep_cap: int | None = None) -> ThermalBlock:
    """Assemble the per-(molecule, kT) dataset entry: truncated ensemble,
    diagnostics, and the Gaussian-reference trace-distance audit.

    ``keep_cap`` follows :func:`resolve_keep_cap` semantics and must match the
    solver's, or the block is silently re-capped to what the ensemble kept."""
    cap = resolve_keep_cap(keep_cap, aspace.dim)

    m, p, tail, capped = _weights_for_kT(ensemble, kT, weight_cutoff, cap)
    E, vecs = ensemble.E[:m], ensemble.vecs[:m]

    mg, pg, tail_g, _ = _weights_for_kT(gaussian_ensemble, kT, weight_cutoff, cap)
    vecs_g = gaussian_ensemble.vecs[:mg]

    dm = thermal_rdm1(vecs, p, aspace)
    nat = natural_occupations(dm)

    return ThermalBlock(
        kT=float(kT),
        E=E.copy(),
        p=p.copy(),
        civecs=np.ascontiguousarray(vecs, dtype=np.float64),
        truncation_error=float(tail),
        entropy=ensemble_entropy(p),
        nat_occs=nat,
        static_corr=static_correlation_score(nat),
        c_max_sq=leading_determinant_weight(ensemble.vecs[0]),
        tracedist_gaussian=trace_distance_projected(vecs, p, vecs_g, pg),
        tracedist_bound=float(0.5 * (tail + tail_g)),
        cap_hit=bool(capped),
    )


def resolve_kT_list(kT_values, relative: bool,
                    spectral_width: float | None = None) -> tuple[np.ndarray, str]:
    """Return (absolute kT values in Hartree, convention tag).

    With ``relative=True`` each value is a fraction of the sector spectral
    width ``evals[-1] - evals[0]`` (dense backend only, where the width is
    known)."""
    kT = np.asarray(list(kT_values), dtype=np.float64)
    if not relative:
        return kT, "absolute_hartree"
    if spectral_width is None:
        raise ValueError("relative kT requires the sector spectral width")
    return kT * float(spectral_width), "relative_spectral_width"
