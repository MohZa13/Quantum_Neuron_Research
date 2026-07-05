"""
Recreate paper Fig. 8 (2x3 grid, n=2..7) using logloss_pennylane for both the
quantum Heisenberg curve and the classical FCIM curve, trained together on
the same target/states/labels per n (see run_matched_pennylane.py) -- a
matched single-trial comparison, plotted alongside the classical FCIM curve
digitized from the original docs/fig_8.png (see digitize_fig8_classical.py)
for reference against the paper's actual result.
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

PANEL_LAYOUT = [(2, "a"), (3, "b"), (4, "c"), (5, "d"), (6, "e"), (7, "f")]


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ax, (n, label) in zip(axes.flat, PANEL_LAYOUT):
        matched = pd.read_csv(RESULTS_DIR / f"pennylane_matched_{n}qubit.csv")
        digitized = pd.read_csv(RESULTS_DIR / "digitized" / f"fig8_classical_{n}qubit_digitized.csv")

        ax.plot(digitized["Epoch"], digitized["Classical_Loss_Digitized"],
                label=r"Classical $H_{\mathrm{FCIM}}$ (paper Fig. 8, digitized, different trial)",
                color="gray", linewidth=1.5, linestyle=":")
        ax.plot(matched["Epoch"], matched["Classical_Loss_PennyLane"],
                label=r"Classical Model ($H_{\mathrm{FCIM}}$, logloss_pennylane, matched trial)",
                color="red", linewidth=2, linestyle="--")
        ax.plot(matched["Epoch"], matched["Quantum_Loss_PennyLane"],
                label=r"Quantum Model ($H_{\mathrm{Heis}}$, logloss_pennylane, matched trial)",
                color="blue", linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Logistic Loss")
        ax.set_title(f"({label})  {n} Qubits, Heisenberg Model")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Fig. 8 reproduction using logloss_pennylane\n"
        "Quantum + classical FCIM trained together on the same data (matched trial).\n"
        "Gray dotted curve: paper's original classical result (different random trial), shown for reference only.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out = FIGURES_DIR / "fig8_pennylane_reproduction.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
