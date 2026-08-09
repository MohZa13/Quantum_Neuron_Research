"""Few-shot molecule identification across temperature — the learnability half.

`scripts/pair_screen.py` established that molecular identity at matched kT is
an off-diagonal-dominated label (isomer pairs: median screen ratio 2.76).  A
high screen does not guarantee a model can exploit it — the single-neuron
program already produced one counterexample — so this script tests
learnability with real generalization on the only multi-temperature CAS(8,8)
set available: `qh9_conjugated_top45.h5`, 28 conjugated molecules at
kT in {0.1, 0.25}.

Protocol, per molecule pair (A, B) and direction: train the hybrid network on
the pair's two states at one temperature, test on the two states at the other.
Two training samples is deliberately extreme few-shot — the question is whether
the identity features a pool can grip are stable across temperature, and the
quantum-vs-z_only gap aggregated over pairs and directions is the measurement.
All pairs share one batch (one aggregate register, states selected by masks),
so runs differ only in masks and pool.

Usage:
    .venv/bin/python scripts/pair_transfer_top45.py \\
        --h5 results/qh9_conjugated_top45.h5 --json-out results/pair_transfer_top45.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np

from qnn import HybridNetwork, StateBatch, build_pool

logger = logging.getLogger("pair_transfer")

KT_TAGS = {"0.1": "kT_0p1000", "0.25": "kT_0p2500"}


def load_all(h5path, n_keep=1024):
    """All (molecule, kT) states on one aggregate top-`n_keep` register."""
    with h5py.File(h5path, "r") as f:
        mols = sorted([k for k in f if k.startswith("mol_") and
                       f[k].attrs.get("complete", False)],
                      key=lambda s: int(s.split("_")[1]))
        dim = f[mols[0]][KT_TAGS["0.1"]]["civecs"].shape[1]
        W = np.zeros(dim)
        blocks = {}
        for m in mols:
            for kT, tag in KT_TAGS.items():
                g = f[m][tag]
                p = np.asarray(g["p"][:], np.float64)
                V = np.asarray(g["civecs"][:], np.float64)
                p = p / p.sum()
                blocks[(m, kT)] = (p, V)
                W += (p[:, None] * V ** 2).sum(0)
        keep = np.sort(np.argsort(W)[::-1][:n_keep])

        states, recs = [], []
        for m in mols:
            nelec = int(np.asarray(f[m]["Z"][:]).sum())
            for kT in KT_TAGS:
                p, V = blocks[(m, kT)]
                R = V[:, keep]
                rho = (R * p[:, None]).T @ R
                tr = float(np.trace(rho))
                states.append(rho / tr)
                recs.append({"mol": m, "kT": float(kT), "nelec": nelec,
                             "trace_kept": tr})
    return np.stack(states), recs, mols


def project(rho, k):
    """Nested top-2^k projection (same construction as train_hybrid_spin)."""
    W = sum(np.diag(r) for r in rho)
    keep = np.sort(np.argsort(W)[::-1][:1 << k])
    out = np.stack([r[np.ix_(keep, keep)] for r in rho])
    kept_pop = float(W[keep].sum() / W.sum())
    return out, kept_pop


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default="results/qh9_conjugated_top45.h5")
    ap.add_argument("--project-qubits", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--n-quantum", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json-out", default="results/pair_transfer_top45.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rho, recs, mols = load_all(args.h5)
    logger.info("loaded %d states (%d molecules x 2 kT)", len(recs), len(mols))
    rho_p, kept = project(rho, args.project_qubits)
    logger.info("projected to K=%d (%.2f%% population kept)",
                rho_p.shape[1], 100 * kept)
    batch = StateBatch(rho_p, normalise=True)
    n = args.project_qubits

    # Pair inventory: isoelectronic pairs first (the composition-matched
    # class), padded with the closest-|dN| pairs up to --max-pairs.
    nel = {m: next(r["nelec"] for r in recs if r["mol"] == m) for m in mols}
    pairs = sorted(combinations(mols, 2),
                   key=lambda ab: (abs(nel[ab[0]] - nel[ab[1]]), ab))
    pairs = pairs[:args.max_pairs]
    n_isoel = sum(1 for a, b in pairs if nel[a] == nel[b])
    logger.info("%d pairs (%d isoelectronic)", len(pairs), n_isoel)

    row_of = {(r["mol"], r["kT"]): i for i, r in enumerate(recs)}
    results, t0 = [], time.perf_counter()
    for kp, (a, b) in enumerate(pairs):
        entry = {"a": a, "b": b, "nelec_a": nel[a], "nelec_b": nel[b],
                 "isoelectronic": nel[a] == nel[b], "runs": []}
        y = np.zeros(len(recs))
        for r_i, r in enumerate(recs):
            if r["mol"] == a:
                y[r_i] = 1.0
            elif r["mol"] == b:
                y[r_i] = -1.0
        for train_kT, test_kT in ((0.1, 0.25), (0.25, 0.1)):
            tr_mask = np.zeros(len(recs), bool)
            te_mask = np.zeros(len(recs), bool)
            for m, s in ((a, 1), (b, -1)):
                tr_mask[row_of[(m, train_kT)]] = True
                te_mask[row_of[(m, test_kT)]] = True
            for kind in ("quantum", "z_only"):
                net = HybridNetwork(build_pool(n, kind),
                                    n_quantum=args.n_quantum, hidden=(8,),
                                    activation="tanh", loss="logistic",
                                    seed=args.seed * 100 + kp)
                h = net.fit(batch, y, epochs=args.epochs, lr=args.lr,
                            train_mask=tr_mask, test_mask=te_mask)
                out = net.forward(batch)[:, 0]
                correct = int(((out[te_mask] > 0) ==
                               (y[te_mask] > 0)).sum())
                entry["runs"].append({
                    "train_kT": train_kT, "pool": kind,
                    "test_correct": correct, "test_total": int(te_mask.sum()),
                    "loss_tr_final": h.loss_tr[-1]})
        results.append(entry)
        if (kp + 1) % 10 == 0:
            logger.info("%d/%d pairs (%.0fs)", kp + 1, len(pairs),
                        time.perf_counter() - t0)

    def acc(kind, only_isoel=None):
        c = t = 0
        for e in results:
            if only_isoel is not None and e["isoelectronic"] != only_isoel:
                continue
            for r in e["runs"]:
                if r["pool"] == kind:
                    c += r["test_correct"]
                    t += r["test_total"]
        return {"correct": c, "total": t, "acc": c / t if t else float("nan")}

    summary = {}
    for kind in ("quantum", "z_only"):
        summary[kind] = {"all": acc(kind),
                         "isoelectronic": acc(kind, True),
                         "non_isoelectronic": acc(kind, False)}
    logger.info("summary: %s", json.dumps(summary, indent=1))

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as fh:
        json.dump({"summary": summary, "pairs": results,
                   "config": dict(vars(args)),
                   "population_kept": kept}, fh, indent=1)
    logger.info("wrote %s (%.0fs)", args.json_out, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
