"""Compare blocked vs interleaved JW wire orderings for the purification MPS.

For every (molecule, kT) block in a run file we build the purification MPS
(:mod:`qthermal.mps`) in both orderings and report the physical (inter-qubit)
bond dimensions.  The ancilla bond is the thermal rank m and is identical for
both orderings, so the ordering question is entirely about the physical bonds.

We report the exact physical bond profile and, at a few per-bond truncation
tolerances, the max physical bond needed to hold rho to that accuracy (with the
implied ||rho - rho_trunc||_1 <= 2*error bound).  Interleaved is expected to win
because same-orbital alpha/beta pairs sit adjacent.

    python -m benchmarks.mps_bond_dimensions --file results/qh9_dense_cas8-6_kT0p25.h5
"""

from __future__ import annotations

import argparse
import statistics

import h5py
import numpy as np

from qthermal.mps import purification_mps

ORDERINGS = ["blocked", "interleaved"]


def _iter_blocks(f):
    """Yield (mol, kT, m, civecs_dataset, p_dataset) — m read from the dataset
    shape without materialising the block, so callers can size-filter first."""
    for mol in sorted(k for k in f if k.startswith("mol_")):
        g = f[mol]
        for kt in sorted(k for k in g if k.startswith("kT_")):
            sub = g[kt]
            yield mol, sub.attrs["kT"], sub["civecs"].shape[0], sub["civecs"], sub["p"]


def analyse_block(civecs, p, ncas, nalpha, nbeta, tols):
    """Return {ordering: {'m', 'exact_max', 'profile', tol: max_chi/err ...}}."""
    out = {}
    for ordering in ORDERINGS:
        exact = purification_mps(civecs, p, ncas, nalpha, nbeta, ordering)
        rec = {"m": exact.ancilla_bond(),
               "exact_max": exact.max_physical_bond(),
               "profile": exact.physical_bond_dims()}
        for tol in tols:
            t = purification_mps(civecs, p, ncas, nalpha, nbeta, ordering, tol=tol)
            rec[f"chi@{tol:g}"] = t.max_physical_bond()
            rec[f"err@{tol:g}"] = t.truncation_error
        out[ordering] = rec
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", default="results/qh9_dense_cas8-6_kT0p25.h5")
    ap.add_argument("--tols", type=float, nargs="+", default=[1e-6, 1e-3])
    ap.add_argument("--max-m", type=int, default=0,
                    help="skip blocks with thermal rank m above this (0 = no limit)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most this many blocks (0 = all)")
    args = ap.parse_args()

    with h5py.File(args.file, "r") as f:
        meta = dict(f["meta"].attrs)
        ncas = int(meta["ncas"])
        nelecas = int(meta["nelecas"])
        nalpha = nbeta = nelecas // 2
        print(f"# {args.file}")
        print(f"# ncas={ncas} nelec=({nalpha},{nbeta})  Q={2*ncas} qubits  "
              f"tols={args.tols}\n")

        header = (f"{'block':<16} {'kT':>5} {'m':>5} | "
                  + " | ".join(f"{o[:6]:>6}: exact " +
                               " ".join(f"{t:g}" for t in args.tols)
                               for o in ORDERINGS))
        print(header)
        print("-" * len(header))

        agg = {o: {"exact": [], **{t: [] for t in args.tols}} for o in ORDERINGS}
        n = 0
        for mol, kT, m, civecs_ds, p_ds in _iter_blocks(f):
            if args.max_m and m > args.max_m:
                continue
            civecs, p = civecs_ds[()], p_ds[()]
            res = analyse_block(civecs, p, ncas, nalpha, nbeta, args.tols)
            cells = []
            for o in ORDERINGS:
                r = res[o]
                agg[o]["exact"].append(r["exact_max"])
                chi_str = " ".join(f"{r[f'chi@{t:g}']:>4d}" for t in args.tols)
                for t in args.tols:
                    agg[o][t].append(r[f"chi@{t:g}"])
                cells.append(f"{r['exact_max']:>5d} {chi_str}")
            print(f"{mol:<16} {kT:>5.2f} {m:>5d} | " + " | ".join(cells))
            n += 1
            if args.limit and n >= args.limit:
                break

        print("\n# summary (max physical bond over blocks; median in parens)")
        for o in ORDERINGS:
            ex = agg[o]["exact"]
            parts = [f"exact {max(ex):>4d} ({int(statistics.median(ex))})"]
            for t in args.tols:
                v = agg[o][t]
                parts.append(f"tol{t:g} {max(v):>4d} ({int(statistics.median(v))})")
            print(f"  {o:<12} " + "   ".join(parts))

        best = {}
        for t in ["exact"] + args.tols:
            b = agg["blocked"][t]
            i = agg["interleaved"][t]
            best[t] = (max(b), max(i))
        print("\n# blocked vs interleaved (worst-case max physical bond):")
        for t, (b, i) in best.items():
            tag = "exact" if t == "exact" else f"tol {t:g}"
            winner = "interleaved" if i < b else ("blocked" if b < i else "tie")
            print(f"  {tag:<10}  blocked {b:>4d}   interleaved {i:>4d}   -> {winner}")


if __name__ == "__main__":
    main()
