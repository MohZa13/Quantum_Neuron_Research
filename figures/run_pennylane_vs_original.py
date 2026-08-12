"""
Produces the data behind the logloss_pennylane vs logloss comparison:

1. Full 500-epoch PennyLane (logloss_pennylane) quantum training curve for
   n=2..7 -- this is the "new" curve plotted against the digitized classical
   FCIM reference from Fig. 8.
2. A short (10-epoch) same-seed run of BOTH the original dense-loop
   implementation and PennyLane, to prove they are numerically identical.
   The original nested-loop `dfj` gradient does not scale (measured
   0.7s/epoch at n=2 up to 76s/epoch at n=7 -- a full 500-epoch run at n=7
   would take about 10.5 hours), so only a short window is run for the
   equivalence check and timing measurement; the full curve is only produced
   with PennyLane.

Outputs (results/):
  - pennylane_quantum_{n}qubit.csv        : full 500-epoch loss history
  - equivalence_check.csv                 : per-n max abs diff, original vs
                                             pennylane over the shared 10-epoch
                                             window, plus per-epoch timing for
                                             the efficiency figure
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantum_training_impls import run_original, run_pennylane

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

SEED = 42
FULL_EPOCHS = 500
CHECK_EPOCHS = 10
QUBITS = range(2, 8)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    equivalence_rows = []

    for n in QUBITS:
        print(f"=== n={n} ===")

        print(f"  running logloss_pennylane, {FULL_EPOCHS} epochs...")
        full = run_pennylane(n, epochs=FULL_EPOCHS, seed=SEED, log_every=20)
        pd.DataFrame({
            "Epoch": np.arange(FULL_EPOCHS),
            "Quantum_Loss_PennyLane": full["history"],
        }).to_csv(RESULTS_DIR / f"pennylane_quantum_{n}qubit.csv", index=False)
        print(f"    total time: {full['elapsed']:.3f}s "
              f"({full['elapsed']/FULL_EPOCHS*1000:.2f} ms/epoch), "
              f"final loss={full['history'][-1]:.5f}")

        print(f"  running logloss (original), {CHECK_EPOCHS} epochs "
              f"(equivalence check + timing only)...")
        short_orig = run_original(n, epochs=CHECK_EPOCHS, seed=SEED, log_every=CHECK_EPOCHS)
        short_penny = run_pennylane(n, epochs=CHECK_EPOCHS, seed=SEED, log_every=CHECK_EPOCHS)

        diff = np.max(np.abs(np.array(short_orig["history"]) - np.array(short_penny["history"])))
        orig_per_epoch = short_orig["elapsed"] / CHECK_EPOCHS
        penny_per_epoch = short_penny["elapsed"] / CHECK_EPOCHS

        print(f"    max |logloss - logloss_pennylane| over {CHECK_EPOCHS} epochs: {diff:.3e}")
        print(f"    original: {orig_per_epoch*1000:.1f} ms/epoch  "
              f"pennylane: {penny_per_epoch*1000:.2f} ms/epoch  "
              f"speedup: {orig_per_epoch/penny_per_epoch:.1f}x")

        equivalence_rows.append({
            "n_qubits": n,
            "max_abs_loss_diff": diff,
            "check_epochs": CHECK_EPOCHS,
            "original_ms_per_epoch": orig_per_epoch * 1000,
            "pennylane_ms_per_epoch": penny_per_epoch * 1000,
            "speedup": orig_per_epoch / penny_per_epoch,
            "original_est_500epoch_minutes": orig_per_epoch * 500 / 60,
            "pennylane_actual_500epoch_seconds": full["elapsed"],
        })

    df = pd.DataFrame(equivalence_rows)
    df.to_csv(RESULTS_DIR / "equivalence_check.csv", index=False)
    print("\n=== summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
