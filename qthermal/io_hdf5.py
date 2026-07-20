"""Module G — resume-safe HDF5 storage.

One file per run, gzip level 4 on all array datasets. Layout:

    /meta                         run-level attrs (basis, active-space params,
                                  solver, kT ladder + convention, versions, unit)
    /mol_{idx}/
        Z, R, active_idx, nocc    provenance
        ecore, h1eff, g           the Hamiltonian; g is the full ncas^4
                                  chemist-notation (pq|rs) tensor (NOT s8-packed;
                                  gzip makes the 8-fold redundancy cheap and
                                  readers need no unpacking step)
        evals                     full sector spectrum — present only when the
                                  dense solver ran; readers must not rely on it
        kT_{tag}/                 E, p, civecs, truncation_error, entropy,
                                  nat_occs, static_corr, c_max_sq,
                                  tracedist_gaussian, tracedist_bound, cap_hit

Resume safety: ``complete=True`` is written *last*; existing complete groups
are skipped, incomplete ones (crashed runs) are deleted and rewritten. Stored
ensembles are truncated to the run's keep cap (``--keep-cap``; default
``max(1024, dim // 4)``, 0 = uncapped, recorded in ``/meta`` as -1 = default).

Everything numeric is float64 on disk; readers get float64 back — no implicit
float32 anywhere.
"""

from __future__ import annotations

import logging

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def kT_tag(kT: float) -> str:
    """Stable group tag for a temperature, e.g. 0.05 -> 'kT_0p0500'."""
    return "kT_" + f"{float(kT):.4f}".replace(".", "p")


def _write_array(grp, name: str, data, dtype=None) -> None:
    arr = np.asarray(data, dtype=dtype)
    if arr.ndim == 0:
        grp.create_dataset(name, data=arr)          # scalars can't be chunked
    else:
        grp.create_dataset(name, data=arr, compression="gzip",
                           compression_opts=4)


class RunWriter:
    """Append-mode writer for one pipeline run (single-process use only)."""

    def __init__(self, path, meta: dict | None = None):
        self._f = h5py.File(path, "a")
        if meta is not None and "meta" not in self._f:
            grp = self._f.create_group("meta")
            for key, val in meta.items():
                if isinstance(val, (list, tuple, np.ndarray)):
                    grp.attrs[key] = np.asarray(val)
                else:
                    grp.attrs[key] = val
            self._f.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._f.close()

    @property
    def meta(self) -> dict:
        return dict(self._f["meta"].attrs) if "meta" in self._f else {}

    def update_meta(self, key: str, value) -> None:
        self._f["meta"].attrs[key] = value
        self._f.flush()

    def is_complete(self, idx: int) -> bool:
        name = f"mol_{idx}"
        return name in self._f and bool(self._f[name].attrs.get("complete", False))

    def write_molecule(self, idx: int, *, Z, R, active_idx, nocc, ecore,
                       h1eff, g, evals_full, blocks) -> bool:
        """Write one molecule group; returns False if already complete.

        ``blocks`` is an iterable of :class:`qthermal.thermal.ThermalBlock`.
        ``evals_full`` may be None (non-dense backends).
        """
        name = f"mol_{idx}"
        if self.is_complete(idx):
            logger.info("%s already complete; skipping", name)
            return False
        if name in self._f:
            logger.warning("%s exists but incomplete; rewriting", name)
            del self._f[name]

        grp = self._f.create_group(name)
        _write_array(grp, "Z", Z, dtype=np.int64)
        _write_array(grp, "R", R, dtype=np.float64)
        _write_array(grp, "active_idx", active_idx, dtype=np.int64)
        _write_array(grp, "nocc", int(nocc))
        _write_array(grp, "ecore", float(ecore))
        _write_array(grp, "h1eff", h1eff, dtype=np.float64)
        _write_array(grp, "g", g, dtype=np.float64)
        if evals_full is not None:
            _write_array(grp, "evals", evals_full, dtype=np.float64)

        for blk in blocks:
            sub = grp.create_group(kT_tag(blk.kT))
            sub.attrs["kT"] = float(blk.kT)
            _write_array(sub, "E", blk.E, dtype=np.float64)
            _write_array(sub, "p", blk.p, dtype=np.float64)
            _write_array(sub, "civecs", blk.civecs, dtype=np.float64)
            _write_array(sub, "truncation_error", float(blk.truncation_error))
            _write_array(sub, "entropy", float(blk.entropy))
            _write_array(sub, "nat_occs", blk.nat_occs, dtype=np.float64)
            _write_array(sub, "static_corr", float(blk.static_corr))
            _write_array(sub, "c_max_sq", float(blk.c_max_sq))
            _write_array(sub, "tracedist_gaussian", float(blk.tracedist_gaussian))
            _write_array(sub, "tracedist_bound", float(blk.tracedist_bound))
            sub.attrs["cap_hit"] = bool(blk.cap_hit)

        self._f.flush()
        grp.attrs["complete"] = True      # written last: the resume marker
        self._f.flush()
        return True
