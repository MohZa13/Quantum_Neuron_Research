"""Every figure in the group-meeting deck, from the project's own data.

Run after ``build_cache.py``:

    MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/presentation/figures.py

Each function returns the path it wrote.  Nothing here invents a number: the
inputs are ``results/presentation_cache.npz`` (per-molecule summaries of the
production run), ``results/spin_comparison_metrics.json`` (the training runs)
and the extended-Heisenberg feature file.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import style as S  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures" / "deck"
CACHE = REPO / "results" / "presentation_cache.npz"
METRICS = REPO / "results" / "spin_comparison_metrics.json"
FEAT = REPO / "results" / "qh9_dense_cas8-8_kT0p1_extheis.h5"
SCREEN = REPO / "results" / "qh9_conjugation_screen_full.csv"

plt.rcParams.update(S.mpl_rc())


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor=S.WHITE)
    plt.close(fig)
    print(f"  wrote {p.relative_to(REPO)}")
    return p


def _tag(ax, text, x=0.98, y=0.04, ha="right", color=None, size=None):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="bottom",
            fontsize=size or 9.0, color=color or S.SLATE)


def _panel_title(ax, text, size=11.5):
    ax.set_title(text, loc="left", fontsize=size, color=S.NAVY,
                 fontweight="bold", pad=8)


# ---------------------------------------------------------------- I. machine
def activation() -> Path:
    """The quantized activation: one knob interpolating step -> smooth."""
    fig, ax = plt.subplots(figsize=(5.0, 2.75))
    x = np.linspace(-4, 4, 800)
    ax.plot(x, np.sign(x), color=S.GRAY, lw=1.3, ls=(0, (2, 2)), zorder=2,
            label="$\\mathrm{sign}(x)$   ($T\\to 0$)")
    for T, c in zip((2.0, 1.0, 0.4), S.SERIES[:3]):
        ax.plot(x, np.tanh(x / T), color=c, lw=2.0, zorder=3, label=f"$T = {T}$")
    ax.legend(fontsize=10, loc="lower right", labelcolor=S.SLATE,
              borderpad=0.2, labelspacing=0.35)
    ax.axhline(0, color=S.HAIR, lw=0.9, zorder=1)
    ax.axvline(0, color=S.HAIR, lw=0.9, zorder=1)
    ax.set_xlabel("eigenvalue  $x$  of  $H(\\theta)$")
    ax.set_ylabel("$\\varphi_T(x)$")
    ax.set_ylim(-1.35, 1.35)
    ax.grid(False)
    return _save(fig, "fig_activation.png")


# -------------------------------------------------------- II. data structure
def eigenblock() -> Path:
    """The state is low rank: a handful of levels carry all the weight."""
    d = np.load(CACHE, allow_pickle=True)
    cum, p0, rank = d["cum"], d["p0"], d["rank"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.85),
                                 gridspec_kw={"wspace": 0.32})

    k = np.arange(1, cum.shape[1] + 1)
    for row in cum:
        a1.plot(k, row, color=S.BLUE, lw=0.7, alpha=0.18, zorder=2)
    a1.plot(k, np.median(cum, axis=0), color=S.BLUE, lw=2.4, zorder=4,
            label="median over 1,000 molecules")
    a1.axhline(1 - 1e-6, color=S.RUST, lw=1.3, ls=(0, (4, 3)), zorder=3,
               label="retention target  $1-10^{-6}$")
    a1.set_xscale("log")
    a1.set_xlim(1, 4900)
    a1.set_ylim(0, 1.045)
    a1.set_xlabel("eigenstates retained,  $k$")
    a1.set_ylabel("cumulative Boltzmann weight")
    a1.legend(fontsize=8.6, loc="center right", labelcolor=S.SLATE)
    _panel_title(a1, "A few hundred levels carry the state")
    a1.annotate(f"4,900 states in the sector;\nmedian stored rank $m$ = {int(np.median(rank)):,}",
                xy=(0.045, 0.035), xycoords="axes fraction", fontsize=9,
                color=S.SLATE, va="bottom")

    a2.hist(100 * p0, bins=32, color=S.BLUE, alpha=0.85, edgecolor=S.WHITE, lw=0.6)
    a2.axvline(100 * np.median(p0), color=S.RUST, lw=1.6)
    a2.annotate(f"median {100*np.median(p0):.0f}%", (100 * np.median(p0), 0.94),
                xycoords=("data", "axes fraction"), color=S.RUST, fontsize=9.5,
                ha="left", xytext=(5, 0), textcoords="offset points")
    a2.set_xlabel("weight on the lowest level,  $p_0$   (%)")
    a2.set_ylabel("molecules")
    _panel_title(a2, "Neither pure nor maximally mixed")
    _tag(a2, f"range {100*p0.min():.0f}–{100*p0.max():.0f}%", y=0.86, x=0.97)
    return _save(fig, "fig_eigenblock.png")


def scaling() -> Path:
    """Sector dimension against window size, with the two solver ceilings."""
    n = np.array([8, 10, 12, 14, 16])
    from math import comb
    dim = np.array([comb(k, k // 2) ** 2 for k in n], dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 3.1))
    bars = ax.bar(np.arange(len(n)), dim, width=0.6, color=S.BLUE, zorder=3)
    for b, v in zip(bars, dim):
        ax.annotate(f"{v:,.0f}" if v < 1e6 else f"{v:.1e}".replace("e+0", r"$\times10^{") + "}$",
                    (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                    fontsize=9.2, color=S.INK, xytext=(0, 3), textcoords="offset points")
    ax.axhline(7e4, color=S.RUST, lw=1.4, ls=(0, (5, 3)), zorder=4,
               label="dense diagonalization ends  ($\\sim7\\times10^{4}$)")
    ax.axhline(853776, color=S.CYAN, lw=1.4, ls=(0, (1.5, 2.5)), zorder=4,
               label="certified Krylov, demonstrated  (853,776, low $T$)")
    ax.legend(fontsize=9.2, loc="upper left", labelcolor=S.SLATE,
              borderpad=0.2, labelspacing=0.4)
    ax.set_yscale("log")
    ax.set_ylim(2e3, 4e9)
    ax.set_xlim(-0.55, len(n) - 0.45)
    ax.set_xticks(np.arange(len(n)))
    ax.set_xticklabels([f"{k}\n({2*k} qubits)" for k in n])
    ax.set_xlabel("active modes  $n$   (half filling, $S_z=0$)")
    ax.set_ylabel("sector dimension")
    ax.grid(axis="x", visible=False)
    return _save(fig, "fig_scaling.png")


def mps_bonds(mols=("mol_2", "mol_1", "mol_7")) -> Path:
    """Measured physical bond dimensions of the purification MPS."""
    import h5py
    from qthermal.mps import purification_mps

    run = REPO / "results" / "qh9_dense_cas8-8_kT0p1.h5"
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ncas, nalpha, nbeta = 8, 4, 4
    shown = {}
    with h5py.File(run, "r") as f:
        for mol in mols:
            b = f[mol]["kT_0p1000"]
            p, V = b["p"][:], b["civecs"][:]
            for ordering, c, ls in (("blocked", S.BLUE, "-"),
                                    ("interleaved", S.RUST, "--")):
                mps = purification_mps(V, p, ncas, nalpha, nbeta, ordering)
                prof = mps.physical_bond_dims()
                ax.plot(np.arange(1, len(prof) + 1), prof, color=c, ls=ls, lw=1.9,
                        alpha=0.9 if mol == mols[0] else 0.35, zorder=3,
                        label=ordering if mol == mols[0] else None)
                shown.setdefault(ordering, []).append(max(prof))
            m = len(p)
    ax.axhline(m, color=S.SLATE, lw=1.2, ls=(0, (1.5, 2.5)), zorder=2)
    ax.annotate(f"ancilla bond $=m$ (exact, no truncation)", (16, m), ha="right",
                va="bottom", color=S.SLATE, fontsize=9, xytext=(0, 3),
                textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel("cut position along the 16-wire chain")
    ax.set_ylabel("physical bond dimension  $\\chi$")
    ax.legend(fontsize=9.5, loc="upper left", labelcolor=S.SLATE)
    _tag(ax, "3 production molecules, kT = 0.1 Ha", y=0.03)
    print("    max physical bond:",
          {k: [int(x) for x in v] for k, v in shown.items()})
    return _save(fig, "fig_mps_bonds.png")


# ---------------------------------------------------------- III. the label
def gap_training() -> Path:
    """The frontier-gap classifier: it learns and it generalizes."""
    import h5py
    rng = np.random.default_rng(7)
    with h5py.File(FEAT, "r") as f:
        names = sorted((k for k in f if k.startswith("mol_")),
                       key=lambda s: int(s.split("_")[1]))
        idx = np.array([int(n.split("_")[1]) for n in names])
        X = np.array([f[n]["kT_0p1000"]["coeffs"][:] for n in names])
    gaps = {int(r["idx"]): float(r["gap_Ha"]) for r in csv.DictReader(open(SCREEN))}
    g = np.array([gaps[i] for i in idx])
    y = (g <= np.median(g)).astype(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    X = (X - mu) / sd

    te = []
    for c in (0, 1):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        te.append(ci[: int(0.3 * len(ci))])
    mask = np.zeros(len(y), bool)
    mask[np.concatenate(te)] = True
    tr, te = ~mask, mask

    w, b, hist = np.zeros(X.shape[1]), 0.0, {k: [] for k in
                                             ("trl", "tel", "tra", "tea")}
    sig = lambda z: 1 / (1 + np.exp(-z))
    bce = lambda p, t: float(-np.mean(t * np.log(p + 1e-12) + (1 - t) * np.log(1 - p + 1e-12)))
    for _ in range(300):
        p = sig(X[tr] @ w + b)
        w -= 0.35 * (X[tr].T @ (p - y[tr]) / tr.sum() + 2e-3 * w)
        b -= 0.35 * float(np.mean(p - y[tr]))
        ptr, pte = sig(X[tr] @ w + b), sig(X[te] @ w + b)
        hist["trl"].append(bce(ptr, y[tr]))
        hist["tel"].append(bce(pte, y[te]))
        hist["tra"].append(100 * float(np.mean((ptr > .5) == y[tr])))
        hist["tea"].append(100 * float(np.mean((pte > .5) == y[te])))

    it = np.arange(1, 301)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.75),
                                 gridspec_kw={"wspace": 0.3})
    a1.plot(it, hist["trl"], color=S.GRAY, lw=1.8, label="training molecules")
    a1.plot(it, hist["tel"], color=S.BLUE, lw=2.3, label="held-out molecules")
    a1.set_xlabel("iteration")
    a1.set_ylabel("logistic loss")
    a1.legend(fontsize=9, labelcolor=S.SLATE)
    _panel_title(a1, "Loss")

    a2.axhline(50, color=S.GRAY, lw=1.2, ls=(0, (4, 3)))
    a2.annotate("chance (balanced split)", (300, 50), ha="right", va="bottom",
                color=S.SLATE, fontsize=9, xytext=(0, 3), textcoords="offset points")
    a2.plot(it, hist["tra"], color=S.GRAY, lw=1.8)
    a2.plot(it, hist["tea"], color=S.BLUE, lw=2.3)
    a2.annotate(f"{hist['tea'][-1]:.0f}%", (300, hist["tea"][-1]), color=S.BLUE,
                fontsize=13, fontweight="bold", ha="right", va="bottom",
                xytext=(-2, 4), textcoords="offset points")
    a2.set_ylim(45, 102)
    a2.set_xlabel("iteration")
    a2.set_ylabel("classification accuracy  (%)")
    _panel_title(a2, "Held-out accuracy")
    print(f"    gap classifier: train {hist['tra'][-1]:.1f}%  held-out {hist['tea'][-1]:.1f}%")
    return _save(fig, "fig_gap_training.png")


def feature_weight() -> Path:
    """Where the 248 features keep their weight."""
    d = np.load(CACHE, allow_pickle=True)
    tot = d["w_single"] + d["w_conn"] + d["w_xy"]
    shares = [100 * np.mean(d[k] / tot) for k in ("w_single", "w_conn", "w_xy")]
    names = ["single-mode occupation\n$\\langle Z_w\\rangle$ and its products",
             "occupation covariance\n$\\langle Z_iZ_j\\rangle-\\langle Z_i\\rangle\\langle Z_j\\rangle$",
             "hopping coherence\n$\\langle X_iX_j\\rangle,\\ \\langle Y_iY_j\\rangle$"]
    fig, ax = plt.subplots(figsize=(5.6, 2.35))
    ypos = np.arange(3)[::-1]
    ax.barh(ypos, shares, height=0.52, color=[S.BLUE, S.BLUE, S.RUST], zorder=3)
    for yy, v in zip(ypos, shares):
        ax.annotate(f"{v:.3g}%", (v, yy), va="center", ha="left", fontsize=10.5,
                    color=S.INK, fontweight="bold", xytext=(6, 0),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlim(5e-3, 420)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9.5, color=S.INK)
    ax.set_xlabel("share of the feature vector's squared weight  (%, log scale)")
    ax.grid(axis="y", visible=False)
    print(f"    feature shares: {shares}")
    return _save(fig, "fig_feature_weight.png")


def coherence_audit() -> Path:
    """The finding: coherence is present, but it is a composition variable."""
    d = np.load(CACHE, allow_pickle=True)
    off = (d["purity"] - d["dpurity"]) / d["purity"]
    dou, pi = d["dou"], d["pi_atoms"]
    ok = np.isfinite(dou)
    from scipy.stats import spearmanr
    rho = spearmanr(off[ok], dou[ok]).statistic

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.85),
                                 gridspec_kw={"wspace": 0.34})
    a1.hist(100 * off, bins=40, color=S.BLUE, alpha=0.85, edgecolor=S.WHITE, lw=0.5)
    med = 100 * np.median(off)
    a1.axvline(med, color=S.RUST, lw=1.6)
    a1.annotate(f"median {med:.1f}%", (med, 0.93), xycoords=("data", "axes fraction"),
                color=S.RUST, fontsize=9.5, ha="left", xytext=(6, 0),
                textcoords="offset points")
    a1.set_xlabel("off-diagonal share of $\\rho$   (%)")
    a1.set_ylabel("molecules")
    _panel_title(a1, "Coherence is real, and it is small")

    a2.scatter(dou[ok], 100 * off[ok], s=13, color=S.BLUE, alpha=0.30,
               linewidths=0, zorder=3)
    lv = np.unique(dou[ok])
    lv = lv[lv <= np.nanpercentile(dou, 99.5)]
    medline = [np.median(100 * off[ok][dou[ok] == v]) for v in lv]
    a2.plot(lv, medline, color=S.RUST, lw=2.2, zorder=4, marker="o", ms=4.5,
            markeredgecolor=S.WHITE, markeredgewidth=0.8, label="median per class")
    a2.set_xlabel("degree of unsaturation")
    a2.set_ylabel("off-diagonal share  (%)")
    a2.legend(fontsize=9, loc="upper left", labelcolor=S.SLATE)
    _tag(a2, f"Spearman $\\rho$ = {rho:.2f}", y=0.05, size=10.5, color=S.INK)
    _panel_title(a2, "…and it is predicted for free")
    print(f"    off-diagonal share median {med:.2f}%,  Spearman vs DoU {rho:.3f}")
    return _save(fig, "fig_coherence_audit.png")


def quantum_vs_classical() -> Path:
    """Held-out accuracy: the pools are indistinguishable on real labels."""
    m = json.load(open(METRICS))["summary"]
    labels = [("S2", "$\\langle S^2\\rangle$\nopen-shell character"),
              ("c", "$c=\\mathrm{Tr}(\\rho S^2_{\\mathrm{od}})$\ncoherence-only"),
              ("control", "synthetic control\npurely off-diagonal")]
    series = [("quantum", "quantum pool  (I, Z, ZZ, XX, YY)", S.BLUE),
              ("classical", "diagonal pool  (I, Z, ZZ)", S.RUST)]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    x = np.arange(len(labels))
    w = 0.26
    for i, (key, name, c) in enumerate(series):
        v = [100 * m[k][key]["final_acc_te"] for k, _ in labels]
        bars = ax.bar(x + (i - 1) * w, v, width=w - 0.02, color=c, zorder=3, label=name)
        for b, vv in zip(bars, v):
            ax.annotate(f"{vv:.1f}", (b.get_x() + b.get_width() / 2, vv), ha="center",
                        va="bottom", fontsize=9.5, color=S.INK,
                        xytext=(0, 2), textcoords="offset points")
    desc = [100 * json.load(open(METRICS))["summary"]["descriptor_baseline"][k]["acc_te"]
            for k, _ in labels]
    for xi, v in zip(x, desc):
        ax.plot([xi - 1.55 * w, xi + 1.55 * w], [v, v], color=S.CYAN, lw=2.2,
                zorder=4, solid_capstyle="butt")
    ax.plot([], [], color=S.CYAN, lw=2.2,
            label="classical descriptors only  (DoU, gap, aromaticity …)")
    ax.axhline(50, color=S.GRAY, lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.annotate("chance", (len(labels) - 0.55, 50), ha="right", va="bottom",
                color=S.SLATE, fontsize=9, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in labels], fontsize=9.5, color=S.INK)
    ax.set_ylabel("held-out accuracy  (%)")
    ax.set_ylim(0, 116)          # bars start at zero: the point is that they tie
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.8, loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=3,
              labelcolor=S.SLATE, columnspacing=1.2, handletextpad=0.5)
    return _save(fig, "fig_quantum_vs_classical.png")


def control_failure() -> Path:
    """The objective prefers a worse-accuracy operator."""
    m = json.load(open(METRICS))["summary"]["control"]
    acc = [100 * m["exact_solution"]["acc_te"], 100 * m["quantum"]["final_acc_te"]]
    loss = [m["exact_solution"]["fd_loss_tr"], m["quantum"]["final_loss_tr"]]
    names = ["exact solution\n$w^{*}=A_{\\mathrm{od}}-\\theta I$",
             "what Adam\nconverges to"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 2.9),
                                 gridspec_kw={"wspace": 0.45})
    for ax, vals, lab, ttl, lim in (
            (a1, acc, "held-out accuracy  (%)", "The operator that classifies", (0, 118)),
            (a2, loss, "Fermi–Dirac loss", "…is the one the loss rejects", (0, 4.3))):
        bars = ax.bar([0, 1], vals, width=0.5, color=[S.CYAN, S.BLUE], zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}" if max(vals) < 10 else f"{v:.0f}%",
                        (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color=S.INK,
                        xytext=(0, 3), textcoords="offset points")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(names, fontsize=9.2, color=S.INK)
        ax.set_ylabel(lab)
        ax.set_ylim(*lim)
        ax.grid(axis="x", visible=False)
        _panel_title(ax, ttl, size=10.5)
    # Label the bar from inside it: an arrow into this region would have to
    # cross the value label.
    a2.text(1.0, loss[1] * 0.48, "what the\noptimizer\nminimizes", ha="center",
            va="center", color=S.WHITE, fontsize=8.8, linespacing=1.25)
    return _save(fig, "fig_control_failure.png")


def diagnostics() -> Path:
    """The ensembles do sit in the interesting regime — chemistry drives them."""
    d = np.load(CACHE, allow_pickle=True)
    ent, gap, td = d["entropy"], d["gap"], d["tdist"]
    ok = np.isfinite(gap) & np.isfinite(ent) & np.isfinite(td)
    r1 = np.corrcoef(gap[ok], ent[ok])[0, 1]
    r2 = np.corrcoef(ent[ok], td[ok])[0, 1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.8),
                                 gridspec_kw={"wspace": 0.30})
    a1.scatter(gap[ok], ent[ok], s=12, color=S.BLUE, alpha=0.28, linewidths=0)
    a1.set_xlabel("spectral gap  $E_1-E_0$   (Ha)")
    a1.set_ylabel("ensemble entropy  (nats)")
    _panel_title(a1, "Mixing follows the gap")
    _tag(a1, f"$r$ = {r1:.2f}", y=0.86, size=10.5, color=S.INK)

    a2.scatter(ent[ok], td[ok], s=12, color=S.BLUE, alpha=0.28, linewidths=0)
    a2.set_xlabel("ensemble entropy  (nats)")
    a2.set_ylabel("trace distance to the\nfree-fermion Gibbs state")
    _panel_title(a2, "The states are far from free")
    _tag(a2, f"$r$ = {r2:.2f}", y=0.05, size=10.5, color=S.INK)
    print(f"    diagnostics r(gap,entropy)={r1:.2f}  r(entropy,tracedist)={r2:.2f}")
    return _save(fig, "fig_diagnostics.png")


# `mps_bonds` is excluded from the default run: the conversion materializes an
# m x 2^16 block per molecule and takes minutes each.  Run it by name when the
# bond profile is wanted.
ALL = [activation, eigenblock, scaling, gap_training, feature_weight,
       coherence_audit, quantum_vs_classical, control_failure, diagnostics]


def main(only: list[str] | None = None) -> None:
    for fn in ALL:
        if only and fn.__name__ not in only:
            continue
        print(fn.__name__)
        fn()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
