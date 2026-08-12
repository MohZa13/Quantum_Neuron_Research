"""The three panels the OMol25 assessment turns on.

    MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=scripts/presentation \\
        .venv/bin/python scripts/plot_omol25_assessment.py

Reads only ``results/localized_basis.npz``, ``second_quantization_labels.npz``,
``sq_label_screen.json`` and ``basis_dependence_probe.json``; writes
``figures/omol25_assessment.png``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "presentation"))
import style as S  # noqa: E402

plt.rcParams.update(S.mpl_rc())
OUT = REPO / "figures" / "omol25_assessment.png"


def main() -> None:
    L = np.load(REPO / "results/localized_basis.npz")
    Z = np.load(REPO / "results/second_quantization_labels.npz")
    P = json.loads((REPO / "results/basis_dependence_probe.json").read_text())
    K = json.loads((REPO / "results/sq_label_screen.json").read_text())
    ok = L["ok"].astype(bool)

    fig, (a1, a2, a3) = plt.subplots(
        1, 3, figsize=(15.0, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))

    # --- 1. the basis control ------------------------------------------------
    bases = [("canonical", "canonical\n(the pipeline's)"),
             ("full_er", "full Edmiston–\nRuedenberg"),
             ("block_er", "block ER\n(reference-preserving)")]
    xs = np.arange(3)
    w = 0.36
    corr = [np.nanmedian(L[f"share_{k}"][ok]) for k, _ in bases]
    ctrl = [np.nanmedian(L[f"share_ctrl_{k}"][ok]) for k, _ in bases]
    a1.bar(xs - w / 2, corr, width=w, color=S.SERIES[0], zorder=3,
           label="correlated ground state")
    a1.bar(xs + w / 2, ctrl, width=w, color=S.SERIES[1], zorder=3,
           label="zero-correlation determinant")
    for x, v in zip(xs - w / 2, corr):
        a1.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=9.6,
                fontweight="bold", color=S.INK)
    for x, v in zip(xs + w / 2, ctrl):
        a1.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=9.6,
                fontweight="bold", color=S.INK)
    a1.set_xticks(xs, [n for _, n in bases], fontsize=9.2)
    a1.set_ylabel(r"median  $\|\mathrm{offdiag}\,\rho\|_F^2\,/\,\|\rho\|_F^2$")
    a1.set_ylim(0, 1.30)
    a1.legend(frameon=False, fontsize=9.0, loc="upper left")
    a1.set_title("Coherence is basis-relative;\nthe control says which part is real",
                 loc="left", fontsize=11.5, color=S.NAVY, fontweight="bold", pad=8)
    a1.text(0.02, 0.40, "1000 molecules\nFCI-invariant\nto 2.6e-12 Ha",
            transform=a1.transAxes, ha="left", fontsize=8.8, color=S.SLATE)

    # --- 2. the invariant discriminant --------------------------------------
    nu = Z["N_unpaired"][Z["ok"].astype(bool)]
    eth = [(r["angle"], sum(min(o, 2 - o) for o in r["nat_occ"]))
           for r in P["rows"]]
    a2.axvspan(1.0, 2.2, color=S.SERIES[2], alpha=0.12, zorder=1)
    a2.hist(nu, bins=45, color=S.SERIES[0], zorder=3, label="QH9, 1000 molecules")
    top = a2.get_ylim()[1] * 1.30
    a2.set_ylim(0, top)
    ex = [v for _, v in eth]
    ey = np.full(len(ex), top * 0.80)
    a2.plot(ex, ey, "-o", color=S.SERIES[1], lw=1.6, ms=5, zorder=5,
            label="C₂H₄ torsion, 0° → 90°")
    a2.annotate("planar", xy=(ex[0], ey[0]), xytext=(0, 9),
                textcoords="offset points", fontsize=8.8, color=S.SERIES[1],
                ha="center")
    a2.annotate("90°: perfect diradical", xy=(ex[-1], ey[-1]), xytext=(-6, 9),
                textcoords="offset points", fontsize=8.8, color=S.SERIES[1],
                ha="right")
    a2.text(1.6, top * 0.46, "admissible\nregime", ha="center", fontsize=9.6,
            color=S.SERIES[2], fontweight="bold")
    a2.text(1.6, top * 0.30,
            f"QH9 max {nu.max():.2f}\nzero molecules\nabove 0.5",
            ha="center", fontsize=9.0, color=S.SLATE)
    a2.set_xlim(0, 2.2)
    a2.set_xlabel(r"$N_{\mathrm{unpaired}} = \sum_i \min(n_i,\,2-n_i)$")
    a2.set_ylabel("molecules")
    a2.legend(frameon=False, fontsize=9.0, loc="upper left")
    a2.set_title("QH9 cannot reach the regime\nthe program requires",
                 loc="left", fontsize=11.5, color=S.NAVY, fontweight="bold", pad=8)

    # --- 3. the ten labels ---------------------------------------------------
    lab = K["labels"]
    rows = [(k, v["quantum_minus_classical"]["mean"],
             v["quantum_minus_classical"]["std"],
             bool(v.get("vanishes_for_single_determinant", False)))
            for k, v in lab.items()]
    rows.sort(key=lambda r: r[1])
    ypos = np.arange(len(rows))[::-1]
    for yp, (name, m, sd, vanish) in zip(ypos, rows):
        c = S.SERIES[1] if vanish else S.GRAY
        a3.barh(yp, m, height=0.62, color=c, zorder=3)
        a3.errorbar(m, yp, xerr=sd, color=S.INK, lw=1.0, capsize=2.5, zorder=4)
    a3.axvline(0, color=S.INK, lw=1.1, zorder=5)
    short = {"mean-field HOMO-LUMO gap (reference)": "mean-field HOMO–LUMO gap",
             "correlation correction to the IP": "corr. correction to the IP",
             "correlation correction to the EA": "corr. correction to the EA",
             "correlation correction to the S-T gap": "corr. correction to the S–T gap",
             "quasiparticle pole strength Z": "quasiparticle pole strength $Z$",
             "unpaired-electron count (Head-Gordon)": "unpaired-electron count",
             "double-excitation weight of the ground state": "double-excitation weight",
             "active-space correlation energy": "correlation energy",
             "ionization energy (correlated)": "ionization energy",
             "singlet-triplet gap (correlated)": "singlet–triplet gap",
             "electron affinity (correlated)": "electron affinity"}
    a3.set_yticks(ypos, [short.get(r[0], r[0]) for r in rows], fontsize=9.0)
    a3.set_xlabel("quantum − classical (points), held-out, 25 splits")
    a3.set_xlim(-4.2, 2.2)
    a3.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    a3.legend(handles=[Patch(color=S.SERIES[1],
                             label="zero for any single determinant"),
                       Patch(color=S.GRAY, label="has a mean-field part")],
              frameon=False, fontsize=9.0, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.16))
    a3.set_xlim(-4.6, 2.6)
    a3.set_title("Ten second-quantization labels,\nnone of them separating the pools",
                 loc="left", fontsize=11.5, color=S.NAVY, fontweight="bold", pad=8)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03, facecolor=S.WHITE)
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
