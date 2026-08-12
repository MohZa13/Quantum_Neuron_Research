"""Figures for the background talk on molecular Hamiltonians and thermal states.

    MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=scripts/presentation \\
        .venv/bin/python scripts/presentation/figures_theory.py [name ...]

Most panels are computed from the project's own data rather than drawn by hand.
The two exceptions, the Jordan-Wigner map and the purification diagram, are
labelled as schematics in their captions.

Inputs: the QH9 database (orbital energies for one molecule), the production
run file (spectra and thermal weights), a small run file (a sector Hamiltonian
small enough to display), and a cached Pauli decomposition.
"""
from __future__ import annotations

import re
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

import style as S  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures" / "deck_theory"
CACHE = REPO / "results" / "theory_cache.npz"
RUN = REPO / "results" / "qh9_dense_cas8-8_kT0p1.h5"
SMALL = REPO / "results" / "qh9_dense_cas8-6_kT0p25.h5"

plt.rcParams.update(S.mpl_rc())

HA_TO_EV = 27.211386


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor=S.WHITE)
    plt.close(fig)
    print(f"  wrote {p.relative_to(REPO)}")
    return p


def _panel_title(ax, text, size=11.5):
    ax.set_title(text, loc="left", fontsize=size, color=S.NAVY,
                 fontweight="bold", pad=8)


# ------------------------------------------------------------------- cache
def build_cache() -> Path:
    """Everything the figures need that requires PySCF or the 45 GB run file."""
    import h5py
    from qthermal.active_space import select_active
    from qthermal import orbitals as orb
    from qthermal.loader import detect_units, iter_records

    rec = next(iter(iter_records(REPO / "data" / "QH9Stable.db", indices=[2])))
    mol = orb.build_mol(rec, detect_units(rec))
    C, eps, nocc = orb.orbitals(rec, mol, orb.overlap(mol))
    aspace = select_active(eps, nocc, 4, 4)

    with h5py.File(RUN, "r") as f:
        # every fourth record, so the sample spans the whole set rather than
        # the small saturated molecules that sit at the front of it
        names = sorted((k for k in f if k.startswith("mol_")),
                       key=lambda s: int(s.split("_")[1]))[::4]
        spectra = [f[n]["kT_0p1000"]["E"][:] for n in names]
        h1 = f["mol_2"]["h1eff"][:]
        g2 = f["mol_2"]["g"][:]
    L = max(len(e) for e in spectra)
    E = np.full((len(spectra), L), np.nan)
    for i, e in enumerate(spectra):
        E[i, :len(e)] = e - e[0]

    # Pauli decomposition of one real active-space Hamiltonian.
    from qthermal.encode import jw_hamiltonian
    op = jw_hamiltonian(h1, g2, 8, ecore=0.0, ordering="blocked")
    coeffs, ops = op.terms()
    pc = np.abs(np.asarray(coeffs))
    pw = np.array([len(re.findall(r"[XYZ]\(", str(o))) for o in ops])

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, eps=eps, nocc=nocc,
                        active_idx=aspace.active_idx, core_idx=aspace.core_idx,
                        E=E, pauli_coeff=pc, pauli_weight=pw)
    print(f"wrote {CACHE.relative_to(REPO)}: {len(eps)} orbitals, "
          f"{len(spectra)} spectra, {len(pc)} Pauli terms")
    return CACHE


# ------------------------------------------------- I. the molecular problem
def orbital_ladder() -> Path:
    """One molecule's orbital energies, and the window kept for correlation."""
    d = np.load(CACHE)
    eps, nocc = d["eps"] * HA_TO_EV, int(d["nocc"])
    act = sorted(int(i) for i in d["active_idx"])

    fig, (top, bot) = plt.subplots(2, 1, figsize=(5.4, 3.3), sharex=True,
                                   gridspec_kw={"height_ratios": [7, 1],
                                                "hspace": 0.10})
    lo_e, hi_e = eps[act[0]], eps[act[-1]]
    top.add_patch(Rectangle((0.06, lo_e - 3), 0.62, hi_e - lo_e + 6,
                            facecolor=S.RUST, alpha=0.08, zorder=1))
    for ax in (top, bot):
        for i, e in enumerate(eps):
            occupied, inside = i < nocc, i in act
            c = S.RUST if inside else (S.BLUE if occupied else S.GRAY)
            ax.hlines(e, 0.08, 0.66, color=c, lw=2.4 if inside else 1.6,
                      alpha=1.0 if inside else 0.6, zorder=3)
            if occupied:
                ax.plot([0.30, 0.44], [e, e], marker="o", ms=3.6, lw=0, color=c,
                        alpha=1.0 if inside else 0.6, zorder=4)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
    top.set_ylim(-32, 60)
    bot.set_ylim(eps[0] - 7, eps[0] + 7)
    bot.set_yticks([round(eps[0])])
    top.spines["bottom"].set_visible(False)
    for ax, y in ((top, 0.0), (bot, 1.0)):        # break marks
        ax.plot([-0.012, 0.012], [y - 0.012, y + 0.012], transform=ax.transAxes,
                color=S.HAIR, lw=1.2, clip_on=False)

    top.annotate("virtual orbitals, discarded", (0.70, 40), fontsize=9.4,
                 color=S.SLATE, va="center")
    top.annotate("active window\n8 orbitals, 8 electrons", (0.70, -20),
                 fontsize=9.8, color=S.RUST, va="center")
    bot.annotate("frozen core", (0.70, eps[0]), fontsize=9.4, color=S.BLUE,
                 va="center")
    top.axhline((eps[nocc - 1] + eps[nocc]) / 2, color=S.SLATE, lw=0.9,
                ls=(0, (3, 3)), zorder=2)
    top.annotate("highest occupied level", (0.02, (eps[nocc - 1] + eps[nocc]) / 2),
                 fontsize=8.6, color=S.SLATE, va="bottom", ha="left",
                 xytext=(0, 3), textcoords="offset points")
    fig.supylabel("orbital energy  (eV)", fontsize=11, color=S.SLATE, x=0.02)
    return _save(fig, "fig_orbital_ladder.png")


def dimension() -> Path:
    """How the number of configurations grows with the number of modes."""
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    n = np.arange(2, 21, 2)
    dim = np.array([comb(k, k // 2) ** 2 for k in n], float)
    ax.plot(n, dim, marker="o", ms=6, color=S.BLUE, zorder=4,
            markeredgecolor=S.WHITE, markeredgewidth=1.0,
            label="active window of $n$ orbitals, half filled")
    ax.plot(n, 4.0 ** n, color=S.GRAY, lw=1.6, ls=(0, (4, 3)), zorder=3,
            label="full Fock space of $n$ orbitals,  $4^{n}$")
    full = comb(24, 5) ** 2
    ax.scatter([24], [full], s=70, color=S.RUST, zorder=5,
               marker="D", edgecolor=S.WHITE, linewidth=1.0)
    ax.annotate("water in a standard basis:\n24 orbitals, 10 electrons,\n"
                f"{full:.1e}".replace("e+09", r"$\times10^{9}$") + " configurations",
                (25.6, 1.2e8), ha="right", va="top", fontsize=9.2, color=S.RUST)
    ax.axhspan(1, 7e4, color=S.BLUE, alpha=0.06, zorder=1)
    ax.annotate("reachable by direct\ndiagonalization", (25.6, 8e2),
                fontsize=9.2, color=S.SLATE, va="center", ha="right")
    ax.set_yscale("log")
    ax.set_xlim(1, 26)
    ax.set_ylim(1, 1e13)
    ax.set_xlabel("orbitals treated as active,  $n$")
    ax.set_ylabel("number of configurations")
    ax.legend(fontsize=9.2, loc="upper left", labelcolor=S.SLATE,
              bbox_to_anchor=(0.0, 0.99))
    return _save(fig, "fig_dimension.png")


def hamiltonian_matrix() -> Path:
    """A real sector Hamiltonian beside the thermal state built from it.

    Magnitudes span many decades, so both panels use a logarithmic single-hue
    scale; the point is the pattern of which entries are non-zero, not the
    sign of any one of them.
    """
    import h5py
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from qthermal.active_space import ActiveSpace
    from qthermal.diagonalize import build_sector_hamiltonian

    with h5py.File(SMALL, "r") as f:
        mol = sorted(k for k in f if k.startswith("mol_"))[0]
        gmol = f[mol]
        h1, g2 = gmol["h1eff"][:], gmol["g"][:]
        kt = sorted(k for k in gmol if k.startswith("kT_"))[0]
        p, V = gmol[kt]["p"][:], gmol[kt]["civecs"][:]
        kT = float(gmol[kt].attrs["kT"])
    ncas = h1.shape[0]
    aspace = ActiveSpace(active_idx=np.arange(ncas), core_idx=np.array([], int),
                         n_act_occ=4, n_act_virt=ncas - 4)
    H = np.abs(build_sector_hamiltonian(h1, g2, aspace))
    rho = np.abs((V.T * p) @ V)
    cmap = LinearSegmentedColormap.from_list("seq", [S.WHITE] + S.SEQ)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9),
                             gridspec_kw={"wspace": 0.26})
    for ax, M, title, floor in (
            (axes[0], H, "The Hamiltonian", 1e-6),
            (axes[1], rho, f"The thermal state at $k_{{\\mathrm{{B}}}}T={kT}$ Ha",
             1e-8)):
        M = np.maximum(M, floor)
        im = ax.imshow(M, cmap=cmap, norm=LogNorm(vmin=floor, vmax=M.max()),
                       interpolation="nearest")
        ax.set_xlabel("configuration")
        ax.grid(False)
        _panel_title(ax, title, size=11.0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8, colors=S.SLATE)
        cb.outline.set_visible(False)
    axes[0].set_ylabel("configuration")
    frac = float((H > 1e-10).sum()) / H.size
    print(f"    dim {H.shape[0]}, {100*frac:.1f}% of H entries non-zero")
    return _save(fig, "fig_hamiltonian_matrix.png")


# ------------------------------------------------------- II. thermal states
def boltzmann() -> Path:
    """One spectrum, three temperatures: how a spectrum becomes an ensemble."""
    d = np.load(CACHE)
    E = d["E"][2]
    E = E[np.isfinite(E)][:40]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.9),
                                 gridspec_kw={"width_ratios": [1, 1.55],
                                              "wspace": 0.28})
    for e in E:
        a1.hlines(e, 0.15, 0.85, color=S.BLUE, lw=1.4, alpha=0.75)
    a1.set_ylim(-0.05, 1.35)
    a1.set_xlim(0, 1)
    a1.set_xticks([])
    a1.set_ylabel("$E_k-E_0$   (Ha)")
    a1.grid(False)
    a1.spines["bottom"].set_visible(False)
    _panel_title(a1, "The many-body spectrum")

    for kT, c in zip((0.025, 0.10, 0.25), S.SERIES[:3]):
        w = np.exp(-E / kT)
        w /= w.sum()
        a2.plot(np.arange(len(E)), w, marker="o", ms=3.6, lw=1.8, color=c,
                label=f"$k_{{\\mathrm{{B}}}}T = {kT}$ Ha", zorder=3)
    a2.set_yscale("log")
    a2.set_ylim(1e-7, 2.0)
    a2.set_xlabel("eigenstate index  $k$")
    a2.set_ylabel("weight  $p_k$")
    a2.legend(fontsize=9.4, labelcolor=S.SLATE, loc="upper right")
    _panel_title(a2, "Weights the temperature assigns to it")
    return _save(fig, "fig_boltzmann.png")


def effective_rank() -> Path:
    """What temperature does to the ensemble, in terms that matter for storage."""
    d = np.load(CACHE)
    E = d["E"]
    kTs = np.geomspace(0.01, 0.15, 32)
    n99 = np.zeros((E.shape[0], len(kTs)))
    p0 = np.zeros_like(n99)
    for i, row in enumerate(E):
        e = row[np.isfinite(row)]
        for j, kT in enumerate(kTs):
            w = np.exp(-e / kT)
            w /= w.sum()
            n99[i, j] = np.searchsorted(np.cumsum(w), 0.99) + 1
            p0[i, j] = w[0]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.85),
                                 gridspec_kw={"wspace": 0.32})
    for ax, data, ylab, title, logy in (
            (a1, n99, "states carrying 99% of the weight",
             "How many states participate", True),
            (a2, 100 * p0, "weight on the lowest level  (%)",
             "How pure the state remains", False)):
        lo, med, hi = np.percentile(data, [10, 50, 90], axis=0)
        ax.fill_between(kTs, lo, hi, color=S.BLUE, alpha=0.16, lw=0)
        ax.plot(kTs, med, color=S.BLUE, lw=2.4, zorder=4)
        ax.axvline(0.10, color=S.RUST, lw=1.3, ls=(0, (4, 3)), zorder=3)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("temperature  $k_{\\mathrm{B}}T$   (Ha)")
        ax.set_ylabel(ylab)
        _panel_title(ax, title, size=11.0)
    a1.annotate("the value used here", (0.10, 1.6), rotation=90, fontsize=9,
                color=S.RUST, va="bottom", ha="right",
                xytext=(-3, 0), textcoords="offset points")
    a2.set_ylim(0, 105)
    a2.annotate("shaded band: 10th to 90th\npercentile over 250 molecules",
                (0.011, 4), fontsize=8.6, color=S.SLATE, va="bottom")
    med99 = np.median(n99, axis=0)
    print(f"    median states for 99% at kT=0.1: {np.interp(0.1, kTs, med99):.0f}")
    return _save(fig, "fig_effective_rank.png")


def methods_ladder() -> Path:
    """Construction methods, ordered by the Hilbert space they can reach."""
    rows = [
        ("Direct diagonalization", 1e2, 7e4, S.BLUE),
        ("Iterative subspace methods", 1e3, 1e8, S.BLUE),
        ("Purification with tensor networks", 1e4, 1e13, S.CYAN),
        ("Typical-state sampling", 1e4, 1e13, S.CYAN),
        ("Preparation on quantum hardware", 1e2, 1e13, S.RUST),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for i, (name, lo, hi, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], lw=13, color=c, solid_capstyle="butt",
                alpha=0.92, zorder=3)
        if hi >= 1e13:
            ax.annotate("", xy=(2.2e13, y), xytext=(1e13, y),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6,
                                        mutation_scale=13), zorder=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=10.2, color=S.INK)
    ax.set_xscale("log")
    ax.set_xlim(50, 6e13)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlabel("configurations the method can handle")
    ax.grid(axis="y", visible=False)
    ax.axvline(4900, color=S.NAVY, lw=1.2, ls=(0, (2, 2)), zorder=5)
    ax.annotate("the size used here", (4900, len(rows) - 0.80), fontsize=9.2,
                color=S.NAVY, ha="center", va="bottom")
    ax.annotate("arrow: bounded by entanglement or by hardware, not by dimension",
                (0.99, -0.34), xycoords="axes fraction", fontsize=8.8,
                color=S.SLATE, ha="right", va="top")
    return _save(fig, "fig_methods_ladder.png")


def storage() -> Path:
    """What one thermal state costs to store, by representation."""
    dim, Q, m, chi, npauli = 4900, 16, 1225, 64, 248
    items = [
        ("Dense matrix on the\nqubit register", (2 ** Q) ** 2 * 8, S.RUST),
        ("Dense matrix in the\nconfiguration basis", dim ** 2 * 8, S.RUST),
        ("Eigenblock\n$(p,\\;V)$", m * dim * 8 + m * 8, S.BLUE),
        ("Purification as a\nmatrix product state", 2 * Q * chi ** 2 * 2 * 8, S.CYAN),
        ("Pauli expectation\nvalues", npauli * 8, S.CYAN),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    y = np.arange(len(items))[::-1]
    vals = [v for _, v, _ in items]
    ax.barh(y, vals, height=0.55, color=[c for _, _, c in items], zorder=3)

    def human(v):
        for u, s in ((1e9, "GB"), (1e6, "MB"), (1e3, "kB")):
            if v >= u:
                return f"{v/u:.3g} {s}"
        return f"{v:.0f} B"

    for yy, v in zip(y, vals):
        ax.annotate(human(v), (v, yy), va="center", ha="left", fontsize=10,
                    fontweight="bold", color=S.INK, xytext=(7, 0),
                    textcoords="offset points")
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in items], fontsize=9.4, color=S.INK)
    ax.set_xscale("log")
    ax.set_xlim(1e2, 4e11)
    ax.set_xlabel("bytes for one state, one temperature  (log scale)")
    ax.grid(axis="y", visible=False)
    return _save(fig, "fig_storage.png")


# ------------------------------------------------------------- III. qubits
def jw_map() -> Path:
    """Schematic: occupations to qubits, and the string an operator carries."""
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    n = 8
    occ = [1, 1, 1, 0, 1, 0, 0, 0]
    for i, o in enumerate(occ):
        ax.add_patch(Rectangle((i, 1.55), 0.72, 0.5, facecolor=S.NAVY if o else S.PANEL,
                               edgecolor=S.WHITE, lw=1.2, zorder=3))
        ax.text(i + 0.36, 1.80, "1" if o else "0", ha="center", va="center",
                fontsize=12, color=S.WHITE if o else S.SLATE, zorder=4)
        ax.text(i + 0.36, 2.22, f"$q_{i}$", ha="center", va="center",
                fontsize=10, color=S.SLATE)
    ax.text(-0.35, 1.80, "mode\noccupations", ha="right", va="center",
            fontsize=10, color=S.INK)

    p, q = 1, 6
    for i in range(n):
        if i == p or i == q:
            lab, c = ("$X\\!/\\!Y$", S.RUST)
        elif p < i < q:
            lab, c = ("$Z$", S.CYAN)
        else:
            lab, c = ("$I$", S.HAIR)
        ax.add_patch(Rectangle((i, 0.45), 0.72, 0.5, facecolor=c,
                               edgecolor=S.WHITE, lw=1.2, zorder=3))
        ax.text(i + 0.36, 0.70, lab, ha="center", va="center", fontsize=9.5,
                color=S.WHITE if c != S.HAIR else S.SLATE, zorder=4)
    ax.text(-0.35, 0.70, "the operator\n$a^{\\dagger}_1 a_6$", ha="right",
            va="center", fontsize=10, color=S.INK)
    ax.annotate("", xy=(p + 0.36, 1.45), xytext=(p + 0.36, 1.05),
                arrowprops=dict(arrowstyle="-", color=S.HAIR, lw=1.0))
    ax.add_patch(FancyArrowPatch((p + 0.72, 0.25), (q, 0.25),
                                 arrowstyle="<->", color=S.CYAN, lw=1.4,
                                 mutation_scale=9))
    ax.text((p + q) / 2 + 0.36, 0.05, "the string of $Z$ operators that carries the sign",
            ha="center", va="top", fontsize=9.2, color=S.CYAN)
    ax.set_xlim(-2.6, n + 0.2)
    ax.set_ylim(-0.35, 2.55)
    ax.axis("off")
    return _save(fig, "fig_jw_map.png")


def pauli_spectrum() -> Path:
    """The Pauli decomposition of one real active-space Hamiltonian."""
    d = np.load(CACHE)
    c, w = d["pauli_coeff"], d["pauli_weight"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.8),
                                 gridspec_kw={"wspace": 0.34})
    ws = np.arange(0, w.max() + 1)
    counts = np.array([(w == k).sum() for k in ws])
    a1.bar(ws, counts, width=0.72, color=S.BLUE, zorder=3)
    a1.set_xlabel("number of non-identity factors")
    a1.set_ylabel("Pauli terms")
    _panel_title(a1, f"{len(c):,} terms on 16 qubits")

    order = np.argsort(-c)
    cum = np.cumsum(c[order]) / c.sum()
    a2.plot(np.arange(1, len(c) + 1), 100 * cum, color=S.BLUE, lw=2.2, zorder=3)
    a2.set_xscale("log")
    a2.set_xlabel("terms retained, largest first")
    a2.set_ylabel("share of total magnitude  (%)")
    a2.set_ylim(0, 104)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    a2.axhline(90, color=S.RUST, lw=1.2, ls=(0, (4, 3)), zorder=2)
    a2.annotate(f"90% of the total\nfrom {k90} terms", (1.4, 12), color=S.RUST,
                fontsize=9.4, ha="left", va="bottom")
    _panel_title(a2, "Most of the weight sits in few terms")
    print(f"    {len(c)} terms, max weight {w.max()}, "
          f"90% of magnitude in {k90} terms")
    return _save(fig, "fig_pauli_spectrum.png")


def purification() -> Path:
    """Schematic: a mixed state on n qubits as a pure state on 2n."""
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    ax.add_patch(Rectangle((0.25, 0.98), 8.3, 1.22, facecolor="none",
                           edgecolor=S.SLATE, lw=1.1, ls=(0, (4, 3)), zorder=2))
    for x0, lab, c in ((0.75, "system register", S.NAVY),
                       (5.30, "ancilla register", S.CYAN)):
        for i in range(4):
            ax.add_patch(Rectangle((x0 + i * 0.72, 1.16), 0.54, 0.42,
                                   facecolor=c, edgecolor=S.WHITE, lw=1.2,
                                   zorder=3))
        ax.text(x0 + 1.35, 1.78, lab, ha="center", fontsize=10.2, color=c)
    ax.text(4.40, 2.42, "a single pure state on the doubled register",
            ha="center", fontsize=11.0, color=S.INK)
    ax.add_patch(FancyArrowPatch((6.65, 0.92), (6.65, 0.56), arrowstyle="-|>",
                                 color=S.RUST, lw=1.5, mutation_scale=12,
                                 zorder=4))
    ax.text(6.90, 0.68, "trace out the ancilla", fontsize=9.8, color=S.RUST,
            va="center")
    ax.text(0.75, 0.20, "the system is left in the mixed thermal state",
            fontsize=9.8, color=S.SLATE, va="center")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(-0.05, 2.72)
    ax.axis("off")
    return _save(fig, "fig_purification.png")


def targets() -> Path:
    """Active space sizes that scientifically interesting systems demand.

    Orbital counts are typical published estimates for a correlated treatment
    of each system, collected in `Papers/theory_references.md` section 9.
    """
    rows = [
        ("Small organic molecule\n(this dataset)", 8, S.BLUE),
        ("Bond dissociation,\nexcited states", 12, S.BLUE),
        ("Transition metal complex,\nsingle centre", 24, S.CYAN),
        ("Photosystem II oxygen\nevolving complex", 40, S.CYAN),
        ("FeMo cofactor of\nnitrogenase", 54, S.RUST),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(rows))[::-1]
    dims = [float(comb(n, n // 2)) ** 2 for _, n, _ in rows]
    ax.barh(y, dims, height=0.56, color=[c for _, _, c in rows], zorder=3)
    for yy, (name, n, _), d in zip(y, rows, dims):
        ax.annotate(f"{n} orbitals", (d, yy), va="center", ha="left",
                    fontsize=9.6, color=S.INK, fontweight="bold",
                    xytext=(7, 0), textcoords="offset points")
    ax.axvline(7e4, color=S.NAVY, lw=1.4, ls=(0, (4, 3)), zorder=4)
    ax.annotate("reach of direct diagonalization", (7e4, 4.72), fontsize=9.2,
                color=S.NAVY, ha="center", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.4, color=S.INK)
    ax.set_xscale("log")
    ax.set_xlim(1, 1e36)
    ax.set_ylim(-0.6, 5.05)
    ax.set_xticks([1e0, 1e10, 1e20, 1e30])
    ax.set_xlabel("configurations in the active space, at half filling")
    ax.grid(axis="y", visible=False)
    print("    dims:", [f"{d:.1e}" for d in dims])
    return _save(fig, "fig_targets.png")


ALL = [orbital_ladder, dimension, hamiltonian_matrix, boltzmann, targets,
       effective_rank, methods_ladder, storage, jw_map, pauli_spectrum,
       purification]


def main(only=None) -> None:
    if only and only[0] == "cache":
        build_cache()
        return
    if not CACHE.exists():
        build_cache()
    for fn in ALL:
        if only and fn.__name__ not in only:
            continue
        print(fn.__name__)
        fn()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
