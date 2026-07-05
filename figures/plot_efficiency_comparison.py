"""
Efficiency comparison: logloss (original dense-loop) vs logloss_pennylane.

Per-epoch timing for both implementations is measured on the SAME data (same
seed) over a short 10-epoch window (see run_pennylane_vs_original.py); the
500-epoch time for the original implementation is an extrapolation from that
per-epoch cost, since a full run is impractical at n=6/7 (see
equivalence_check.csv for the measured max loss difference, which is what
justifies extrapolating rather than re-measuring the full run).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = Path(__file__).resolve().parent


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "equivalence_check.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(df["n_qubits"], df["original_est_500epoch_minutes"] * 60,
             "o--", color="#D62728", label="logloss (original), extrapolated to 500 epochs")
    ax.plot(df["n_qubits"], df["pennylane_actual_500epoch_seconds"],
             "o-", color="#0057B8", label="logloss_pennylane, measured, 500 epochs")
    ax.set_yscale("log")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Wall-clock time for 500 epochs (s, log scale)")
    ax.set_title("Training time: original vs PennyLane")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    ax2 = axes[1]
    ax2.bar(df["n_qubits"], df["speedup"], color="#2CA02C")
    ax2.set_xlabel("Number of qubits")
    ax2.set_ylabel("Speedup (original ms/epoch ÷ pennylane ms/epoch)")
    ax2.set_title("Per-epoch speedup of logloss_pennylane")
    for x, y in zip(df["n_qubits"], df["speedup"]):
        ax2.text(x, y, f"{y:.0f}x", ha="center", va="bottom", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "logloss_pennylane matches logloss numerically (max loss diff per n in "
        "equivalence_check.csv) while running orders of magnitude faster",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = FIGURES_DIR / "efficiency_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
