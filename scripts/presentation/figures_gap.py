"""Figures for the HOMO-LUMO gap diagnosis deck.

Inputs are the four JSON files written by the audit scripts -- nothing here
recomputes or invents a number:

    results/gap_diagnosis.json           the ladder, regression, residual test
    results/gap_diagnosis_followup.json  alternative labels, coherence split
    results/gap_diagnosis_ceiling.json   full-diag(rho) baseline, R+/R- screen
    results/gap_diagnosis_controls.json  noise control, pool coverage

    MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=scripts/presentation \\
        .venv/bin/python scripts/presentation/figures_gap.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import style as S  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures" / "deck_gap"
plt.rcParams.update(S.mpl_rc())

A = json.loads((REPO / "results/gap_diagnosis.json").read_text())
B = json.loads((REPO / "results/gap_diagnosis_followup.json").read_text())
C = json.loads((REPO / "results/gap_diagnosis_ceiling.json").read_text())
E = json.loads((REPO / "results/gap_diagnosis_controls.json").read_text())


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor=S.WHITE)
    plt.close(fig)
    print(f"  wrote {p.relative_to(REPO)}")
    return p


def _title(ax, text, size=11.5):
    ax.set_title(text, loc="left", fontsize=size, color=S.NAVY,
                 fontweight="bold", pad=8)


def _note(ax, text, x=0.98, y=0.05, ha="right", va="bottom", size=9.0):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size,
            color=S.SLATE)


# --------------------------------------------------------------- 1. ladder
def ladder() -> Path:
    """Held-out accuracy by feature set, and what the coherence block adds."""
    lad = A["ladder_classification"]
    cei = C["ceiling_classification"]

    def g(d, k):
        return d[k]["acc"]["mean"], d[k]["acc"]["std"]

    rows = [
        ("chance (label is balanced)", 50.0, 0.0, "chance"),
        ("composition only\nformula, DoU, π-system", *g(lad, "composition, quadratic"), "cl"),
        ("8 natural occupations", *g(lad, "natural occupations (8)"), "cl"),
        ("16 orbital occupations\n$\\langle Z\\rangle$ only", *g(lad, "Z only (16)"), "cl"),
        ("full diag(ρ)\n4900 populations → 300 PCs", *g(cei, "full diag(rho), 300 PCs"), "cl"),
        ("diagonal Pauli pool\nZ + ZZ (136)", *g(lad, "diagonal Pauli, Z+ZZ (136)"), "cl"),
        ("ρ spectrum + CI gap\nbasis-independent, 15 numbers",
         *g(lad, "rho spectrum + CI gap"), "cl"),
        ("coherence channel alone\nXX / YY (112)",
         *g(lad, "off-diagonal Pauli, XX/YY (112)"), "qu"),
        ("full quantum pool\n248 features", *g(lad, "full Pauli pool (248)"), "qu"),
    ]
    col = {"chance": S.GRAY, "cl": S.SERIES[0], "qu": S.SERIES[1]}

    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(9.4, 4.95), gridspec_kw=dict(width_ratios=[1.9, 1]))

    ypos = np.arange(len(rows))[::-1]
    for yp, (lab, m, sd, kind) in zip(ypos, rows):
        a1.barh(yp, m - 45, left=45, height=0.66, color=col[kind],
                alpha=0.32 if kind == "chance" else 1.0, zorder=3)
        if sd:
            a1.errorbar(m, yp, xerr=sd, color=S.INK, lw=1.1, capsize=3, zorder=4)
        a1.text(m + sd + 1.2, yp, f"{m:.1f}", va="center", fontsize=10.4,
                color=S.INK, fontweight="bold", zorder=5)
    a1.set_yticks(ypos, [r[0] for r in rows], fontsize=9.3)
    a1.set_xlim(45, 102)
    a1.set_xlabel("held-out accuracy (%) · 300 unseen molecules · 25 random splits")
    a1.grid(axis="x", color=S.HAIR, lw=0.8)
    a1.grid(axis="y", visible=False)
    _title(a1, "Every classical readout already has the label")

    nc = E["noise_control"]
    base = nc["diagonal_only"]["mean"]
    bars = [("+112 coherence\nfeatures (XX/YY)", nc["delta_coherence"], S.SERIES[1]),
            ("+112 Gaussian\nnoise features", nc["delta_noise"], S.GRAY)]
    xs = np.arange(2)
    a2.axhline(0, color=S.INK, lw=1.1, zorder=4)
    a2.bar(xs, [b[1] for b in bars], width=0.5,
           color=[b[2] for b in bars], zorder=3)
    for x, b in zip(xs, bars):
        a2.text(x, b[1] - 0.30, f"{b[1]:+.2f}", ha="center", va="top",
                fontsize=11.5, fontweight="bold", color=S.INK)
    a2.set_xticks(xs, [b[0] for b in bars], fontsize=9.6)
    a2.set_ylabel("accuracy change (points)")
    a2.set_ylim(-4.4, 1.6)
    _title(a2, "Added to the diagonal pool")
    _note(a2, f"baseline: the diagonal pool alone, {base:.1f}%\n"
              f"paired over the same 25 splits", x=0.5, ha="center", y=0.86)

    fig.tight_layout()
    return _save(fig, "fig_ladder.png")


# ------------------------------------------------- 2. not the binarisation
def not_binarisation() -> Path:
    """Regression reproduces the null; the errors sit at the median cut."""
    reg = A["ladder_regression"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.95))

    names = [("composition (formula + DoU)", "composition\nonly"),
             ("rho spectrum + CI gap", "ρ spectrum\n+ CI gap"),
             ("diagonal Pauli, Z+ZZ (136)", "diagonal Pauli\n(classical)"),
             ("off-diagonal Pauli, XX/YY (112)", "coherence\nchannel"),
             ("full Pauli pool (248)", "full quantum\npool")]
    vals = [reg[k]["r2"]["mean"] for k, _ in names]
    errs = [reg[k]["r2"]["std"] for k, _ in names]
    cols = [S.SERIES[0]] * 3 + [S.SERIES[1]] * 2
    xs = np.arange(len(names))
    a1.bar(xs, vals, width=0.62, color=cols, zorder=3)
    a1.errorbar(xs, vals, yerr=errs, fmt="none", color=S.INK, lw=1.1,
                capsize=3, zorder=4)
    for x, v in zip(xs, vals):
        a1.text(x, max(v, 0) + 0.03, f"{v:.3f}", ha="center", fontsize=10.2,
                fontweight="bold", color=S.INK)
    a1.set_xticks(xs, [n for _, n in names], fontsize=9.4)
    a1.set_ylabel("held-out $R^2$ on the continuous gap")
    a1.set_ylim(0, max(vals) + 0.17)
    _title(a1, "Predict the gap as a number — same answer")
    dq = A["regression_quantum_minus_classical_r2"]
    _note(a1, f"quantum − classical:  $\\Delta R^2$ = {dq['mean']:+.4f}", y=0.90,
          size=9.8)

    tv = A["accuracy_vs_threshold_distance"]
    acc, edges = tv["acc"], tv["bin_edges_eV"]
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(5)]
    a2.plot(range(5), acc, "-o", color=S.SERIES[0], lw=2.3, ms=8, zorder=4)
    for i, v in enumerate(acc):
        a2.text(i, v + 1.4, f"{v:.1f}", ha="center", fontsize=10.2,
                color=S.INK, fontweight="bold")
    a2.set_xticks(range(5), [f"{c:.2f}" for c in centres[:-1]] + ["far"],
                  fontsize=9.6)
    a2.set_xlabel("distance from the median split (eV), quintiles")
    a2.set_ylabel("held-out accuracy (%)")
    a2.set_ylim(45, 106)
    a2.axhline(50, color=S.GRAY, lw=1.0, ls=(0, (4, 3)), zorder=2)
    _title(a2, "Where the binary classifier's errors are")
    _note(a2, "the median cut costs accuracy only where the gap\n"
              "is genuinely ambiguous — it hides no coherence channel",
          x=0.03, ha="left", y=0.22)

    fig.tight_layout()
    return _save(fig, "fig_not_binarisation.png")


# ----------------------------------------------------- 3. where the gap lives
def where_gap_lives() -> Path:
    """Coherence tracks the gap — and adds nothing once the diagonal is known."""
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(9.4, 4.95), gridspec_kw=dict(width_ratios=[1.25, 1]))

    gc = E["global_coherence_vs_residual"]
    keys = [("rho_offdiag_share", "off-diagonal share\nof ρ (exact)"),
            ("static_corr", "static correlation"),
            ("entropy", "von Neumann\nentropy"),
            ("c_max_sq", "leading CI\nweight $|c_{max}|^2$")]
    xs = np.arange(len(keys))
    w = 0.36
    with_gap = [abs(gc[k]["pearson_with_gap"]) for k, _ in keys]
    with_res = [abs(gc[k]["pearson"]) for k, _ in keys]
    a1.bar(xs - w / 2, with_gap, width=w, color=S.SERIES[0], zorder=3,
           label="with the gap itself")
    a1.bar(xs + w / 2, with_res, width=w, color=S.SERIES[1], zorder=3,
           label="with what the diagonal model left over")
    for x, v in zip(xs - w / 2, with_gap):
        a1.text(x, v + 0.018, f"{v:.2f}", ha="center", fontsize=9.8,
                fontweight="bold", color=S.INK)
    for x, v in zip(xs + w / 2, with_res):
        a1.text(x, v + 0.018, f"{v:.2f}", ha="center", fontsize=9.8,
                fontweight="bold", color=S.INK)
    a1.set_xticks(xs, [n for _, n in keys], fontsize=9.4)
    a1.set_ylabel("|Pearson correlation|")
    a1.set_ylim(0, 1.0)
    a1.legend(frameon=False, fontsize=9.4, loc="upper right")
    _title(a1, "Coherence tracks the gap — redundantly")

    rl = A["residual_label_classification"]
    bars = [("composition", rl["composition"]["acc"]["mean"], S.SERIES[0]),
            ("diagonal pool", rl["diagonal"]["acc"]["mean"], S.SERIES[0]),
            ("coherence channel", rl["off-diagonal"]["acc"]["mean"], S.SERIES[1])]
    xs = np.arange(3)
    a2.bar(xs, [b[1] - 45 for b in bars], bottom=45, width=0.55,
           color=[b[2] for b in bars], zorder=3)
    for x, b in zip(xs, bars):
        a2.text(x, b[1] + 0.7, f"{b[1]:.1f}", ha="center", fontsize=10.6,
                fontweight="bold", color=S.INK)
    a2.axhline(50, color=S.GRAY, lw=1.1, ls=(0, (4, 3)), zorder=4)
    a2.text(-0.46, 50.5, "chance", fontsize=9, color=S.SLATE, ha="left")
    a2.set_xticks(xs, [b[0] for b in bars], fontsize=9.6)
    a2.set_ylabel("held-out accuracy (%)")
    a2.set_ylim(45, 72)
    _title(a2, "Predicting the leftover")
    a2.set_xlabel("label: sign of the gap residual after the diagonal model",
                  fontsize=9.4)

    fig.tight_layout()
    return _save(fig, "fig_where_gap_lives.png")


# ------------------------------------------------------------- 4. headroom
def headroom() -> Path:
    """Which label lets coherence matter, and how much harder molecules buy."""
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(9.4, 4.95), gridspec_kw=dict(width_ratios=[1.35, 1]))

    keys = [("synthetic off-diagonal control",
             "synthetic off-diagonal control\n(exactly representable)"),
            ("correlation correction (CI gap | KS gap)",
             "correlation correction\nto the gap"),
            ("CASCI frontier gap E1-E0", "correlated CASCI gap  $E_1-E_0$"),
            ("KS HOMO-LUMO gap (the label under test)",
             "mean-field HOMO–LUMO gap\n(the label under test)")]
    ypos = np.arange(len(keys))[::-1]
    for yp, (k, lab) in zip(ypos, keys):
        q, c = B[k]["quantum"]["mean"], B[k]["classical"]["mean"]
        a1.plot([c, q], [yp, yp], color=S.HAIR, lw=3.4, zorder=2,
                solid_capstyle="round")
        a1.scatter([c], [yp], s=70, color=S.SERIES[0], zorder=4)
        a1.scatter([q], [yp], s=70, color=S.SERIES[1], zorder=4)
        a1.text(107, yp, f"{q - c:+.1f}", va="center", ha="right", fontsize=10.6,
                fontweight="bold", color=S.INK)
    a1.set_yticks(ypos, [lab for _, lab in keys], fontsize=9.4)
    a1.set_xlabel("held-out accuracy (%)")
    a1.set_xlim(46, 107)
    a1.set_xticks([50, 60, 70, 80, 90, 100])
    a1.axvline(50, color=S.GRAY, lw=1.0, ls=(0, (4, 3)), zorder=1)
    a1.grid(axis="y", visible=False)
    _title(a1, "Which label lets the coherence channel matter?")
    a1.scatter([], [], s=70, color=S.SERIES[0], label="classical (diagonal pool)")
    a1.scatter([], [], s=70, color=S.SERIES[1], label="quantum (full pool)")
    a1.legend(frameon=False, fontsize=9.2, loc="lower left",
              bbox_to_anchor=(0.0, -0.03))

    cj = B["conjugated_subset"]
    names = ["all 1000\n$k_BT$ = 0.1", "top 28 conj.\n$k_BT$ = 0.1",
             "top 28 conj.\n$k_BT$ = 0.25"]
    vals = [E["rho_offdiag_share"]["median"] * 100,
            cj["kT_0p1000"]["median"] * 100, cj["kT_0p2500"]["median"] * 100]
    xs = np.arange(3)
    a2.bar(xs, vals, width=0.58, color=[S.SERIES[0], S.SERIES[2], S.SERIES[1]],
           zorder=3)
    for x, v in zip(xs, vals):
        a2.text(x, v + 0.4, f"{v:.1f}%", ha="center", fontsize=10.8,
                fontweight="bold", color=S.INK)
    a2.set_xticks(xs, names, fontsize=9.4)
    a2.set_ylabel(r"$\|\mathrm{offdiag}\,\rho\|_F^2\ /\ \|\rho\|_F^2$   (median)")
    a2.set_ylim(0, max(vals) * 1.3)
    _title(a2, "How much coherence is on the table?")
    _note(a2, "harder molecules and hotter states\nbuy about 2× — not 100×",
          x=0.03, ha="left", y=0.80, size=9.2)

    fig.tight_layout()
    return _save(fig, "fig_headroom.png")


def main() -> None:
    for fn in (ladder, not_binarisation, where_gap_lives, headroom):
        fn()


if __name__ == "__main__":
    main()
