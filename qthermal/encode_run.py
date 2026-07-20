"""Module I CLI — batch extended-Heisenberg Pauli mappings from a run file.

    python -m qthermal.encode_run --in run.h5 --out run_extheis.h5

For every complete molecule group and every kT block in a Module-H run file,
evaluates Tr(rho P) over `extended_heisenberg_paulis(ncas)` (Z on every JW
wire, ZZ on every wire pair, XX/YY within each spin block — the weight-<=2
strings that survive number/parity/time-reversal symmetry) and stores the
coefficient vector. The contraction happens in the determinant basis
(`extended_heisenberg_expectations`), so cost is set by the sector dimension,
never by 2**(2*ncas).

Weights are used exactly as stored — they sum to 1 - truncation_error — and
the deficit is copied onto each output block as `trace`, so a reader can
renormalize (or reject) truncated blocks explicitly.

Output layout:

    /meta                 ordering, ncas/nalpha/nbeta, n_terms, source file +
                          copied run provenance, versions
    /pauli_labels         (n_terms,) bytes, `extended_heisenberg_labels` order
    /mol_{idx}/
        Z                 provenance (atomic numbers)
        kT_{tag}/coeffs   (n_terms,) float64, attrs kT / trace /
                          truncation_error / cap_hit

``--taper`` additionally stores the Z2-tapered form of the basis (alpha/beta
block parities removed, 2*ncas - 2 qubits; `taper_extended_heisenberg`):

    /pauli_labels_tapered (n_terms,) bytes, contiguous tapered wires
    /taper_signs          (n_terms,) int8, +/-1
    /taper_kept_wires     (2*ncas - 2,) original wire of each tapered wire

Expectations are invariant under the taper, so `coeffs` need no second copy:
the coefficient of tapered string k on the tapered register is
``taper_signs[k] * coeffs[k]``. The flag can also be added to an existing
output file after the fact — molecules resume, the taper datasets are filled
in.

Resume safety mirrors Module G: ``complete=True`` is written last, complete
groups are skipped on rerun, incomplete ones are rewritten.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import h5py
import numpy as np

from qthermal import __version__
from qthermal.encode import (
    extended_heisenberg_expectations,
    extended_heisenberg_labels,
    taper_extended_heisenberg,
)

logger = logging.getLogger("qthermal.encode_run")

_COPIED_META = ("basis", "n_act_occ", "n_act_virt", "kT_list", "kT_convention",
                "solver_name", "weight_cutoff", "keep_cap", "unit")


def encode_molecule(mol_grp, out_grp, ncas: int, nalpha: int, nbeta: int,
                    ordering: str) -> int:
    """Write one molecule's per-kT coefficient blocks; returns block count."""
    out_grp.create_dataset("Z", data=mol_grp["Z"][:].astype(np.int64))
    n_blocks = 0
    for key in sorted(k for k in mol_grp if k.startswith("kT_")):
        blk = mol_grp[key]
        p = blk["p"][:]
        coeffs = extended_heisenberg_expectations(
            blk["civecs"][:], p, ncas, nalpha, nbeta, ordering)
        sub = out_grp.create_group(key)
        sub.create_dataset("coeffs", data=coeffs)
        sub.attrs["kT"] = float(blk.attrs["kT"])
        sub.attrs["trace"] = float(p.sum())
        sub.attrs["truncation_error"] = float(blk["truncation_error"][()])
        sub.attrs["cap_hit"] = bool(blk.attrs.get("cap_hit", False))
        n_blocks += 1
    return n_blocks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qthermal.encode_run",
        description="run HDF5 -> extended-Heisenberg Pauli coefficients")
    p.add_argument("--in", dest="src", required=True,
                   help="Module-H run file (dense or iterative)")
    p.add_argument("--out", required=True, help="output HDF5 file")
    p.add_argument("--ordering", choices=("blocked", "interleaved"),
                   default="blocked",
                   help="JW wire layout (default blocked — the classifier "
                        "bridge convention)")
    p.add_argument("--taper", action="store_true",
                   help="also store the Z2-tapered basis (block parities "
                        "removed, 2*ncas - 2 qubits): tapered labels, signs, "
                        "and the wire map; coefficients are unchanged")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    t_run = time.perf_counter()
    n_done = n_resumed = n_skipped = 0
    with h5py.File(args.src, "r") as src, h5py.File(args.out, "a") as out:
        meta = dict(src["meta"].attrs)
        ncas = int(meta["ncas"])
        nalpha = nbeta = int(meta["nelecas"]) // 2
        labels = extended_heisenberg_labels(ncas, args.ordering)

        if "meta" not in out:
            grp = out.create_group("meta")
            grp.attrs.update(
                source_file=str(args.src), ordering=args.ordering,
                ansatz="extended_heisenberg", n_terms=len(labels),
                ncas=ncas, nalpha=nalpha, nbeta=nbeta,
                code_version=__version__)
            for key in _COPIED_META:
                if key in meta:
                    grp.attrs[key] = meta[key]
            out.create_dataset(
                "pauli_labels", data=np.array(labels, dtype="S"))
        elif (out["meta"].attrs["ordering"] != args.ordering
              or out["meta"].attrs["ncas"] != ncas):
            logger.error("existing --out was built with ordering=%s, ncas=%d;"
                         " refusing to mix conventions in one file",
                         out["meta"].attrs["ordering"],
                         out["meta"].attrs["ncas"])
            return 2

        if args.taper and "pauli_labels_tapered" not in out:
            t_labels, t_signs, kept = taper_extended_heisenberg(
                ncas, nalpha, nbeta, args.ordering)
            out.create_dataset("pauli_labels_tapered",
                               data=np.array(t_labels, dtype="S"))
            out.create_dataset("taper_signs", data=t_signs)
            out.create_dataset("taper_kept_wires",
                               data=np.asarray(kept, dtype=np.int64))
            out["meta"].attrs.update(
                n_qubits_tapered=2 * ncas - 2,
                taper_removed_wires=np.asarray(
                    sorted(set(range(2 * ncas)) - set(kept)), dtype=np.int64),
                taper_sector=np.asarray(
                    [(-1) ** nalpha, (-1) ** nbeta], dtype=np.int64))
            logger.info("tapered basis stored: %d qubits -> %d",
                        2 * ncas, 2 * ncas - 2)

        for name in sorted((k for k in src if k.startswith("mol_")),
                           key=lambda s: int(s.split("_")[1])):
            if not src[name].attrs.get("complete", False):
                logger.warning("%s incomplete in source; skipping", name)
                n_skipped += 1
                continue
            if name in out and out[name].attrs.get("complete", False):
                n_resumed += 1
                continue
            if name in out:
                del out[name]
            t0 = time.perf_counter()
            grp = out.create_group(name)
            n_blocks = encode_molecule(src[name], grp, ncas, nalpha, nbeta,
                                       args.ordering)
            out.flush()
            grp.attrs["complete"] = True
            out.flush()
            n_done += 1
            logger.info("%s: %d kT block(s) in %.1f s", name, n_blocks,
                        time.perf_counter() - t0)

    logger.info("encode run finished in %.1f s: %d written, %d resumed, "
                "%d skipped", time.perf_counter() - t_run, n_done, n_resumed,
                n_skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
