"""
Train quantum Heisenberg AND classical FCIM together with logloss_pennylane
(optimize_phase2), on the same target/states/labels per n -- a genuinely
matched single-trial comparison, unlike the earlier version of this figure
which paired a freshly-run quantum curve against a classical curve digitized
from the paper's image (a different random trial).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantum_training_impls import run_pennylane_matched

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

SEED = 42
FULL_EPOCHS = 500
QUBITS = range(2, 8)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    for n in QUBITS:
        print(f"n={n}: running matched quantum+classical logloss_pennylane, {FULL_EPOCHS} epochs...")
        result = run_pennylane_matched(n, epochs=FULL_EPOCHS, seed=SEED, log_every=20)
        pd.DataFrame({
            "Epoch": np.arange(FULL_EPOCHS),
            "Quantum_Loss_PennyLane": result["history_q"],
            "Classical_Loss_PennyLane": result["history_c"],
        }).to_csv(RESULTS_DIR / f"pennylane_matched_{n}qubit.csv", index=False)
        print(f"    elapsed={result['elapsed']:.3f}s  "
              f"final Q={result['history_q'][-1]:.5f}  final C={result['history_c'][-1]:.5f}")


if __name__ == "__main__":
    main()
