"""Bridge test: the hybrid network trained on MPS-produced thermal states.

This is the first time the two halves of the repo actually meet: states made
by ``QThermalMPS`` (imaginary-time TDVP at CAS(10,10), sector dimension
63,504 — beyond every dense route) are fed to the ``qnn`` hybrid network as
plain density matrices.  The question it answers is deliberately modest:

    does the Module K artifact plug into the qnn training loop, at its native
    K = 1024, and does the network learn from these states?

WHAT THE LABELS ARE — AND ARE NOT.  With one or two molecules at six
temperatures, every label available (hot vs cold, which molecule, kT
regression) is visible in diag(rho): Boltzmann weights are diagonal data.  So
the z_only ablation is expected to succeed here too, and nothing in this
script bears on the coherence-label program (OPEN_QUESTIONS.md Q1).  This is
a plumbing test with real states, not a physics result; the JSON says so.

REGISTER CONVENTIONS — the one place a silent bug could live:

  * The h5 ``rho`` is reduced over Jordan-Wigner wires 0..9 — the ALPHA spin
    orbitals under ``blocked`` ordering — with wire 0 the MOST significant
    bit of the row index (Module I convention, `qthermal/mps.py`).
  * qnn pools are little-endian: qubit q is bit q, LEAST significant first
    (`qnn/pools.py`).
  * The loader therefore bit-reverses the register so that qubit q = wire q =
    alpha spin-orbital q.  Two razor checks pin the orientation: every state
    lives on popcount-5 rows exactly (N_alpha = 5 is sharp), and the coldest
    state's dominant determinant is the HF string — orbitals 0-4 occupied —
    which is row index 0b0000011111 = 31 after the reversal (and 992 before:
    a reversed loader cannot pass).

The alpha-RDM is unit trace BY CONSTRUCTION (a partial trace of a normalised
purification), so INVARIANTS.md I8's trace-leak concern does not arise; the
states are normalised anyway because StateBatch does it unconditionally.

Usage:
    .venv/bin/python scripts/train_mps_thermal.py \\
        --mps results/qh9_mps_ncas10.h5 --mols mol_3 \\
        --tasks hotcold,kT-regress --json-out results/mps_thermal_training.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import time
from pathlib import Path

import h5py
import numpy as np

from qnn import HybridNetwork, StateBatch, build_pool

logger = logging.getLogger("train_mps_thermal")

# Sibling script, imported by file path so this works both as a plain script
# and under pytest, without touching sys.path (CLAUDE.md rule).
_spec = importlib.util.spec_from_file_location(
    "_train_hybrid_spin", Path(__file__).resolve().parent / "train_hybrid_spin.py")
_ths = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ths)
project_register = _ths.project_register
screening_score = _ths.screening_score


# --------------------------------------------------------------- loading

def bitrev_perm(nbits: int) -> np.ndarray:
    """Permutation p with p[r] = bit-reversal of r, as int64 indices."""
    r = np.arange(1 << nbits, dtype=np.int64)
    out = np.zeros_like(r)
    for b in range(nbits):
        out |= ((r >> b) & 1) << (nbits - 1 - b)
    return out


def load_states(path: str, mols: list[str]):
    """(rho, records, meta): little-endian states plus per-state provenance.

    Verifies, per state: unit trace, symmetry, and that ALL diagonal weight
    sits on popcount-nalpha rows (the alpha electron number is sharp, so any
    weight elsewhere means a broken export, not an approximation).
    """
    records, stack = [], []
    with h5py.File(path, "r") as f:
        meta = {k: (v.decode() if isinstance(v, bytes) else v)
                for k, v in f["meta"].attrs.items()}
        n = int(meta["nwires"]) // 2               # qubits = alpha wires
        nalpha = int(meta["nalpha"])
        perm = bitrev_perm(n)
        pc = np.array([bin(i).count("1") for i in range(1 << n)])
        onsector = pc == nalpha

        for mol in mols:
            if mol not in f:
                raise KeyError(f"{mol} not in {path} (have "
                               f"{[k for k in f if k.startswith('mol_')]})")
            g = f[mol]
            tags = sorted([k for k in g if k.startswith("kT_")])
            for t in tags:
                b = g[t]
                wires = list(b["rho_wires"][:])
                if wires != list(range(n)):
                    raise ValueError(f"{mol}/{t}: rho_wires {wires} != 0..{n-1}")
                a = np.asarray(b["rho"][:], np.float64)
                a = a[np.ix_(perm, perm)]          # -> little-endian, qubit q = wire q
                tr = float(np.trace(a))
                asym = float(np.abs(a - a.T).max())
                offsec = float(np.abs(np.diag(a)[~onsector]).max())
                if abs(tr - 1) > 1e-6 or asym > 1e-9 or offsec > 1e-9:
                    raise ValueError(
                        f"{mol}/{t}: trace {tr}, asym {asym}, off-sector {offsec}")
                stack.append(a)
                records.append({
                    "mol": mol, "tag": t, "kT": float(b.attrs["kT"]),
                    "beta": float(b.attrs["beta"]),
                    "maxlinkdim": int(b["maxlinkdim"][()]),
                    "energy": float(b["energy"][()]),
                    "entropy": float(b["entropy"][()]),
                    "argmax_diag": int(np.argmax(np.diag(a))),
                    "nelec_from_diag": float(
                        sum(np.diag(a) @ ((np.arange(1 << n) >> q) & 1)
                            for q in range(n))),
                })
    rho = np.stack(stack)
    hf = (1 << nalpha) - 1                        # 0b11111: orbitals 0..4 occupied
    for rec in records:
        if rec["kT"] <= 0.25 and rec["argmax_diag"] != hf:
            logger.warning("%s/%s: dominant determinant %d != HF string %d "
                           "(multireference state, or an orientation bug)",
                           rec["mol"], rec["tag"], rec["argmax_diag"], hf)
    return rho, records, meta


# --------------------------------------------------------------- labels

def make_labels(task: str, records):
    """(y, kind): y in {-1,+1} for classification, standardized for regression."""
    if task == "hotcold":
        return np.where([r["kT"] >= 1.0 for r in records], 1.0, -1.0), "logistic"
    if task == "molecule":
        mols = sorted({r["mol"] for r in records})
        if len(mols) != 2:
            raise ValueError(f"molecule task needs exactly 2 molecules, have {mols}")
        return np.where([r["mol"] == mols[0] for r in records], 1.0, -1.0), "logistic"
    if task == "kT-regress":
        v = np.log10([r["kT"] for r in records])
        return (v - v.mean()) / v.std(), "squared"
    raise ValueError(f"unknown task {task}")


# --------------------------------------------------------------- training

def run_loo(batch, y, loss, pool_kind, n, args, seed):
    """Leave-one-out: one fresh network per fold, masks on one shared batch."""
    M = batch.M
    folds = []
    for i in range(M):
        tr = np.ones(M, bool)
        tr[i] = False
        net = HybridNetwork(build_pool(n, pool_kind), n_quantum=args.n_quantum,
                            hidden=args.hidden_t, activation="tanh", T=args.temperature,
                            loss=loss, seed=seed * 1000 + i)
        h = net.fit(batch, y, epochs=args.epochs, lr=args.lr,
                    train_mask=tr, test_mask=~tr)
        out = float(net.forward(batch)[i, 0])
        folds.append({"held_out": i, "y": float(y[i]), "out": out,
                      "loss_tr": h.loss_tr[-1], "loss_tr0": h.loss_tr[0]})
    if loss == "logistic":
        correct = sum((f["out"] > 0) == (f["y"] > 0) for f in folds)
        summary = {"loo_accuracy": correct / M, "loo_correct": int(correct), "M": M}
    else:
        mae = float(np.mean([abs(f["out"] - f["y"]) for f in folds]))
        summary = {"loo_mae_std_units": mae, "M": M}
    summary["median_final_train_loss"] = float(np.median([f["loss_tr"] for f in folds]))
    summary["median_initial_train_loss"] = float(np.median([f["loss_tr0"] for f in folds]))
    return folds, summary


def full_fit(batch, y, loss, pool_kind, n, args, seed):
    """All states in the training set: does the loss actually go to ~0?"""
    net = HybridNetwork(build_pool(n, pool_kind), n_quantum=args.n_quantum,
                        hidden=args.hidden_t, activation="tanh", T=args.temperature,
                        loss=loss, seed=seed)
    t0 = time.perf_counter()
    h = net.fit(batch, y, epochs=args.epochs, lr=args.lr)
    dt = time.perf_counter() - t0
    return {"loss_first": h.loss_tr[0], "loss_final": h.loss_tr[-1],
            "acc_final": h.acc_tr[-1], "epochs": args.epochs,
            "seconds": dt, "sec_per_epoch": dt / args.epochs}


def fullK_timing(rho, y, n_full, args):
    """A few epochs at the native register size — the K^3 reality check."""
    batch = StateBatch(rho, normalise=True)
    net = HybridNetwork(build_pool(n_full, "quantum"), n_quantum=args.n_quantum,
                        hidden=args.hidden_t, activation="tanh",
                        T=args.temperature, loss="logistic", seed=args.seed)
    t0 = time.perf_counter()
    h = net.fit(batch, y, epochs=args.fullK_epochs, lr=args.lr)
    dt = time.perf_counter() - t0
    return {"K": 1 << n_full, "epochs": args.fullK_epochs,
            "sec_per_epoch": dt / args.fullK_epochs,
            "loss_first": h.loss_tr[0], "loss_final": h.loss_tr[-1]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mps", default="results/qh9_mps_ncas10.h5")
    ap.add_argument("--mols", default="mol_3", help="comma-separated mol_* groups")
    ap.add_argument("--tasks", default="hotcold,kT-regress",
                    help="any of hotcold,molecule,kT-regress")
    ap.add_argument("--pools", default="quantum,z_only")
    ap.add_argument("--project-qubits", type=int, default=8,
                    help="keep the 2^k most-populated rows. At k=8 this is "
                         "lossless here: only C(10,5)=252 rows are populated.")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-quantum", type=int, default=8)
    ap.add_argument("--hidden", default="8")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fullK-epochs", type=int, default=8,
                    help="epochs for the native-K timing run (0 disables)")
    ap.add_argument("--json-out", default="results/mps_thermal_training.json")
    args = ap.parse_args(argv)
    args.hidden_t = tuple(int(t) for t in args.hidden.split(",") if t.strip())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    mols = [m for m in args.mols.split(",") if m]
    rho, records, meta = load_states(args.mps, mols)
    n_full = int(meta["nwires"]) // 2
    logger.info("loaded %d states from %s (K=%d): %s", len(records), args.mps,
                1 << n_full,
                ", ".join(f"{r['mol']}@kT={r['kT']:g}(chi={r['maxlinkdim']})"
                          for r in records))

    retention = {"population_kept": 1.0, "offdiag_frobenius_kept": 1.0}
    n = n_full
    rho_p = rho
    if args.project_qubits and args.project_qubits < n_full:
        rho_p, n, retention = project_register(rho, args.project_qubits)
        logger.info("projected to K=%d: %.4f%% population, %.4f%% offdiag kept",
                    1 << n, 100 * retention["population_kept"],
                    100 * retention["offdiag_frobenius_kept"])
    batch = StateBatch(rho_p, normalise=True)

    results = {"tasks": {}}
    for task in [t for t in args.tasks.split(",") if t]:
        y, loss = make_labels(task, records)
        entry = {"loss": loss, "y": [float(v) for v in y]}
        if loss == "logistic":
            od, dg, ratio = screening_score(batch, y, np.ones(len(y), bool))
            entry["screen"] = {"offdiag": od, "diag": dg, "ratio": ratio}
            logger.info("task %-10s screen ||offdiag||/||diag|| = %.4f", task, ratio)
        for kind in [p for p in args.pools.split(",") if p]:
            t0 = time.perf_counter()
            folds, summary = run_loo(batch, y, loss, kind, n, args, args.seed)
            summary["fit_all"] = full_fit(batch, y, loss, kind, n, args, args.seed)
            summary["seconds"] = time.perf_counter() - t0
            entry[kind] = {"summary": summary, "folds": folds}
            logger.info("task %-10s pool %-8s %s  (%.0fs)", task, kind,
                        {k: v for k, v in summary.items()
                         if k.startswith(("loo_", "median"))},
                        summary["seconds"])
        results["tasks"][task] = entry

    if args.fullK_epochs:
        y, _ = make_labels("hotcold", records)
        results["fullK_timing"] = fullK_timing(rho, y, n_full, args)
        logger.info("native K=%d: %.2f s/epoch, loss %.4f -> %.4f over %d epochs",
                    results["fullK_timing"]["K"],
                    results["fullK_timing"]["sec_per_epoch"],
                    results["fullK_timing"]["loss_first"],
                    results["fullK_timing"]["loss_final"], args.fullK_epochs)

    results["states"] = records
    results["projection"] = retention
    results["config"] = {k: v for k, v in vars(args).items() if k != "hidden_t"}
    results["config"]["hidden"] = list(args.hidden_t)
    results["caveat"] = (
        "Plumbing test: every label here is visible in diag(rho) (Boltzmann "
        "weights are diagonal data), so z_only succeeding is expected and "
        "says nothing about the coherence program (OPEN_QUESTIONS.md Q1).")

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as fh:
        json.dump(results, fh, indent=1)
    logger.info("wrote %s", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
