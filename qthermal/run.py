"""Module H — CLI orchestration.

    python -m qthermal.run --qh9-path <QH9Stable.db or QH9 root> --out run.h5

One molecule per worker task (the problem is embarrassingly parallel).
Workers rebuild their own PySCF objects from the plain-array MoleculeRecord —
nothing PySCF-owned crosses a process boundary. PySCF threads through BLAS
already, so parallelism happens only across molecules and each worker's
OpenMP/BLAS thread count is capped to cpu_count // workers.

Coordinate units are detected empirically on the first record and re-verified
every 100th (Module A); validation failures are logged and skipped, never
fatal.
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from qthermal import __version__
from qthermal.active_space import select_active
from qthermal.diagonalize import DenseEDSolver, SectorTooLargeError
from qthermal.hamiltonian import build_cas_hamiltonian
from qthermal.io_hdf5 import RunWriter
from qthermal.loader import CachedUnitDetector, MoleculeRecord, iter_records
from qthermal.orbitals import MoleculeValidationError, build_mol, orbitals, overlap
from qthermal.thermal import (
    DEFAULT_KT_LIST,
    DEFAULT_WEIGHT_CUTOFF,
    build_thermal_block,
    gaussian_reference_ensemble,
    resolve_kT_list,
)

logger = logging.getLogger("qthermal.run")

# Relative-kT mode needs the spectral width, which is unknown before the
# solve; the dense backend keeps the full spectrum anyway, so solving with an
# effectively infinite kT_max just means the memory cap decides the kept
# prefix and every per-kT truncation downstream is still exact.
_KT_MAX_UNBOUNDED = 1e12

SOLVERS = {"dense": DenseEDSolver}   # Phase-2 backends slot in here


@dataclass(frozen=True)
class RunConfig:
    n_act_occ: int
    n_act_virt: int
    solver: str
    allow_large_dense: bool
    kT_values: tuple
    kT_relative: bool
    weight_cutoff: float = DEFAULT_WEIGHT_CUTOFF


def process_molecule(task: tuple[MoleculeRecord, str, RunConfig]) -> dict:
    """Worker entry point: full pipeline for one molecule.

    Returns either ``{"status": "ok", ...arrays for the writer...}`` or
    ``{"status": "skip", "reason": ...}``; exceptions never propagate out of
    a worker.
    """
    record, unit, cfg = task
    t0 = time.perf_counter()
    try:
        mol = build_mol(record, unit)
        S = overlap(mol)
        C, eps, nocc = orbitals(record, mol, S)
        aspace = select_active(eps, nocc, cfg.n_act_occ, cfg.n_act_virt)
        ham = build_cas_hamiltonian(mol, C, nocc, aspace)

        solver = SOLVERS[cfg.solver](allow_large_dense=cfg.allow_large_dense)
        kT_max = (_KT_MAX_UNBOUNDED if cfg.kT_relative
                  else float(max(cfg.kT_values)))
        ens = solver.solve(ham.h1eff, ham.g, aspace, kT_max=kT_max,
                           weight_cutoff=cfg.weight_cutoff)
        ens_g = gaussian_reference_ensemble(ham.h1eff, aspace, solver,
                                            kT_max=kT_max,
                                            weight_cutoff=cfg.weight_cutoff)

        width = (float(ens.evals_full[-1] - ens.evals_full[0])
                 if ens.evals_full is not None else None)
        kT_list, _ = resolve_kT_list(cfg.kT_values, cfg.kT_relative,
                                     spectral_width=width)
        blocks = [build_thermal_block(ens, ens_g, kT, aspace,
                                      weight_cutoff=cfg.weight_cutoff)
                  for kT in kT_list]

        return {
            "status": "ok", "idx": record.idx,
            "Z": record.Z, "R": record.R,
            "active_idx": aspace.active_idx, "nocc": nocc,
            "ecore": ham.ecore, "h1eff": ham.h1eff, "g": ham.g,
            "evals_full": ens.evals_full, "blocks": blocks,
            "wall_s": time.perf_counter() - t0,
        }
    except (MoleculeValidationError, SectorTooLargeError, ValueError) as exc:
        return {"status": "skip", "idx": record.idx,
                "reason": f"{type(exc).__name__}: {exc}",
                "wall_s": time.perf_counter() - t0}
    except Exception as exc:  # never crash the run on one molecule
        logger.exception("unexpected failure on idx=%d", record.idx)
        return {"status": "skip", "idx": record.idx,
                "reason": f"unexpected {type(exc).__name__}: {exc}",
                "wall_s": time.perf_counter() - t0}


def _worker_init(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    from pyscf import lib
    lib.num_threads(threads)


def _iter_results(tasks, workers: int):
    if workers <= 1:
        _worker_init(int(os.environ.get("OMP_NUM_THREADS",
                                        os.cpu_count() or 1)))
        for task in tasks:
            yield process_molecule(task)
        return
    import multiprocessing as mp

    threads = max(1, (os.cpu_count() or 1) // workers)
    ctx = mp.get_context("spawn")   # no PySCF/BLAS state crosses the boundary
    with ctx.Pool(workers, initializer=_worker_init,
                  initargs=(threads,)) as pool:
        yield from pool.imap_unordered(process_molecule, tasks, chunksize=1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qthermal.run",
        description="QH9 -> truncated CASCI thermal ensembles (Phase 1)")
    p.add_argument("--qh9-path", required=True,
                   help="QH9Stable.db file or QH9 root directory")
    p.add_argument("--out", required=True, help="output HDF5 file")
    p.add_argument("--limit", type=int, default=100,
                   help="max molecules to process (default 100)")
    p.add_argument("--n-act-occ", type=int, default=4)
    p.add_argument("--n-act-virt", type=int, default=4)
    p.add_argument("--solver", choices=sorted(SOLVERS), default="dense")
    p.add_argument("--allow-large-dense", action="store_true",
                   help="permit dense ED for 5,000 < dim <= 70,000 (RAM-checked)")
    p.add_argument("--kT-list", default=",".join(str(k) for k in DEFAULT_KT_LIST),
                   help="comma-separated temperatures (default 0.05,0.10,0.25)")
    p.add_argument("--kT-relative", action="store_true",
                   help="interpret --kT-list as fractions of the sector "
                        "spectral width instead of Hartree")
    p.add_argument("--weight-cutoff", type=float, default=DEFAULT_WEIGHT_CUTOFF)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    kT_values = tuple(float(x) for x in args.kT_list.split(",") if x.strip())
    if not kT_values or any(k <= 0 for k in kT_values):
        logger.error("invalid --kT-list %r", args.kT_list)
        return 2
    cfg = RunConfig(n_act_occ=args.n_act_occ, n_act_virt=args.n_act_virt,
                    solver=args.solver, allow_large_dense=args.allow_large_dense,
                    kT_values=kT_values, kT_relative=args.kT_relative,
                    weight_cutoff=args.weight_cutoff)

    import pyscf
    meta = {
        "basis": "def2-svp",
        "n_act_occ": cfg.n_act_occ, "n_act_virt": cfg.n_act_virt,
        "ncas": cfg.n_act_occ + cfg.n_act_virt,
        "nelecas": 2 * cfg.n_act_occ,
        "solver_name": SOLVERS[cfg.solver].name,
        "kT_list": list(kT_values),
        "kT_convention": ("relative_spectral_width" if cfg.kT_relative
                          else "absolute_hartree"),
        "weight_cutoff": cfg.weight_cutoff,
        "code_version": __version__,
        "pyscf_version": pyscf.__version__,
        "unit": "pending",
    }

    detector = CachedUnitDetector(recheck_every=100)
    skip_reasons: Counter[str] = Counter()
    wall_times: list[float] = []
    n_done = n_skipped = n_resumed = 0
    t_run = time.perf_counter()

    with RunWriter(args.out, meta) as writer:

        def tasks():
            nonlocal n_resumed
            for ordinal, record in enumerate(
                    iter_records(args.qh9_path, limit=args.limit)):
                unit = detector.unit_for(record, ordinal)
                if writer.meta.get("unit") == "pending":
                    writer.update_meta("unit", unit)
                if writer.is_complete(record.idx):
                    n_resumed += 1
                    continue
                yield record, unit, cfg

        for res in _iter_results(tasks(), args.workers):
            wall_times.append(res["wall_s"])
            if res["status"] == "ok":
                writer.write_molecule(
                    res["idx"], Z=res["Z"], R=res["R"],
                    active_idx=res["active_idx"], nocc=res["nocc"],
                    ecore=res["ecore"], h1eff=res["h1eff"], g=res["g"],
                    evals_full=res["evals_full"], blocks=res["blocks"])
                n_done += 1
                logger.info("mol %d done in %.1f s", res["idx"], res["wall_s"])
            else:
                n_skipped += 1
                skip_reasons[res["reason"]] += 1
                logger.warning("mol %d skipped: %s", res["idx"], res["reason"])

    logger.info("run finished in %.1f s: %d written, %d skipped, %d resumed",
                time.perf_counter() - t_run, n_done, n_skipped, n_resumed)
    if wall_times:
        logger.info("wall time per molecule: median %.1f s, max %.1f s",
                    statistics.median(wall_times), max(wall_times))
    for reason, count in skip_reasons.most_common():
        logger.info("  skip x%d: %s", count, reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
