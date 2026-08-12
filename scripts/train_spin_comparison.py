"""Quantum vs classical Fermi-Dirac neuron on singlet/triplet-open-shell labels.

Both models are the *same* machine -- the paper's Fermi-Dirac neuron,

    predict  sign(Tr[rho H(omega)]),      H(omega) = sum_j omega_j P_j
    loss     (1/M) sum_m Tr[ T ln(I + e^{-y_m H/T}) rho_m ]        (Eq. 56)

trained with the same optimizer, temperature, initialisation, split and epoch
budget.  The ONLY difference is which Pauli strings the pool contains:

  classical  {Z_i} + {Z_i Z_j}          n(n+1)/2 params, H strictly DIAGONAL
             (the paper's fully connected Ising model, FCIM)
  quantum    the same + {X_i X_j} + {Y_i Y_j}     H reaches OFF the diagonal
  classical_full   an unconstrained diagonal H (2**n free entries) -- the
             strongest diagonal model that exists, included as a ceiling so a
             quantum win cannot be blamed on classical under-parameterisation

A diagonal H makes Tr(rho H) -- and the whole spectral loss -- a function of
diag(rho) alone.  So the classical models see exactly the classical part of the
state and nothing else; the gap between the curves is attributable to
off-diagonal (coherence) access and to nothing else.  This is the Z-only
ablation of docs/QUANTUM_NEURON.md section 4, run as the comparison itself.

Labels (from scripts/spin_labels.py), both median-split so both are balanced:

    S2   Tr(rho S^2)      total thermal open-shell / triplet character
    c    Tr(rho S2_od)    its coherence-only part, c = S2 - <D>

Qubit register: the 2**n most populated determinants of the shared CAS sector,
relabelled 0..2**n-1 (the sector-compression encoding of qthermal.encode).
Relabelling permutes basis states, so it moves *which* coherences a given Pauli
string reaches, but it can never move weight between the diagonal and the
off-diagonal -- the classical/quantum distinction is exactly preserved.

Usage:
    python -m scripts.train_spin_comparison --labels results/spin_labels_kT0p1.npz \
        --rho /path/rho_10q.npy --out figures/spin_quantum_vs_classical.png
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("train_spin")

# label key -> (plot title, placeholder). Shared by the training and --replot paths.
LABEL_TITLES = {
    "S2": ("$\\langle S^2\\rangle$   thermal open-shell / triplet character", None),
    "c":  ("$c=\\mathrm{Tr}(\\rho\\,S^2_{\\mathrm{od}})$   coherence-only spin coupling", None),
    "control": ("POSITIVE CONTROL — synthetic purely off-diagonal label", None),
}


# --- operator pools ----------------------------------------------------------

def z_table(n: int) -> np.ndarray:
    """z[q, r] = (-1)**bit_q(r), the Z eigenvalue of basis state r on qubit q."""
    r = np.arange(1 << n)
    return np.stack([1.0 - 2.0 * ((r >> q) & 1) for q in range(n)])


def build_pool(n: int, kind: str):
    """Return (diag_terms, offdiag_terms, labels).

    diag_terms:    (P, K) array of diagonal vectors.
    offdiag_terms: list of (mask, value_vector) with P|r> = value[r] |r ^ mask>.
    """
    K = 1 << n
    z = z_table(n)
    diag, off, labels = [], [], []

    if kind == "classical_full":
        # Unconstrained diagonal operator: one parameter per basis state.
        # This span already contains the identity.
        return np.eye(K), [], [f"|{r}><{r}|" for r in range(K)]

    # Identity: the decision rule is sign(Tr(rho H)) with no separate bias, and
    # Tr(rho I) = 1 for every normalised state, so this term *is* the bias.
    # Every Z/ZZ/XX/YY string is traceless, so without it both models are forced
    # through the origin and cannot express a thresholded label at all. It is a
    # diagonal operator, so it is available to the classical pool too and confers
    # no off-diagonal advantage.
    diag.append(np.ones(K)); labels.append("I")

    for q in range(n):                                   # Z_q
        diag.append(z[q]); labels.append(f"Z{q}")
    for p in range(n):                                   # Z_p Z_q
        for q in range(p + 1, n):
            diag.append(z[p] * z[q]); labels.append(f"Z{p} Z{q}")

    if kind == "quantum":
        for p in range(n):
            for q in range(p + 1, n):
                mask = (1 << p) | (1 << q)
                # X_p X_q |r> = |r ^ mask>
                off.append((mask, np.ones(K))); labels.append(f"X{p} X{q}")
                # Y_p Y_q |r> = -z_p(r) z_q(r) |r ^ mask>
                off.append((mask, -z[p] * z[q])); labels.append(f"Y{p} Y{q}")
    elif kind != "classical":
        raise ValueError(kind)

    return np.asarray(diag), off, labels


def features(mat: np.ndarray, diag: np.ndarray, off) -> np.ndarray:
    """Tr(mat @ P_j) for every operator in the pool, for symmetric real `mat`."""
    d = np.diag(mat)
    vals = [diag @ d] if len(diag) else [np.zeros(0)]
    r = np.arange(mat.shape[0])
    vals += [np.array([mat[r, r ^ mask] @ v]) for mask, v in off]
    return np.concatenate(vals)


def assemble(weights: np.ndarray, diag: np.ndarray, off, K: int) -> np.ndarray:
    """H(omega) as a dense (K, K) real symmetric matrix."""
    nd = len(diag)
    H = np.diag(weights[:nd] @ diag) if nd else np.zeros((K, K))
    r = np.arange(K)
    for j, (mask, v) in enumerate(off):
        H[r, r ^ mask] += weights[nd + j] * v
    return H


# --- the Fermi-Dirac logistic loss (label-aggregated) ------------------------

def spectrum(w, diag, off, K):
    """(eigenvalues, eigenvectors) of H(omega); eigenvectors is None if diagonal."""
    if not off:
        return w[:len(diag)] @ diag, None
    lam, U = np.linalg.eigh(assemble(w, diag, off, K))
    return lam, U


def rotate(R, U):
    """R in the eigenbasis of H; the diagonal of rho is all a diagonal H needs."""
    return np.diag(R).copy() if U is None else U.T @ R @ U


def fd_loss(lam, Rpt, Rmt, M, T):
    """Eq. 56 given the spectrum and the rotated label aggregates."""
    fp = T * np.logaddexp(0.0, -lam / T)
    fm = T * np.logaddexp(0.0, lam / T)
    dp = Rpt if Rpt.ndim == 1 else np.diag(Rpt)
    dm = Rmt if Rmt.ndim == 1 else np.diag(Rmt)
    return float(fp @ dp + fm @ dm) / M


def fd_grad(lam, U, Rpt, Rmt, diag, off, M, T):
    """dL/domega via the Daleckii-Krein divided-difference formula.

    For a diagonal pool U is None: H is already diagonal, so no
    eigendecomposition is involved and the result is exact and cheap.
    """
    dfp = -1.0 / (1.0 + np.exp(lam / T))          # f'_+ = -sigmoid(lam/T)
    dfm = 1.0 / (1.0 + np.exp(-lam / T))
    if U is None:
        gvec = dfp * Rpt + dfm * Rmt              # dL/dlam_r
        return (diag @ gvec) / M
    fp = T * np.logaddexp(0.0, -lam / T)
    fm = T * np.logaddexp(0.0, lam / T)
    dl = lam[:, None] - lam[None, :]
    near = np.abs(dl) < 1e-10
    safe = np.where(near, 1.0, dl)
    G = np.zeros_like(dl)
    for f, df, R in ((fp, dfp, Rpt), (fm, dfm, Rmt)):
        term = (f[:, None] - f[None, :]) / safe
        np.copyto(term, np.broadcast_to(df[:, None], dl.shape), where=near)
        G += term * R
    return features(U @ G @ U.T, diag, off) / M


def train(X, y, Rp, Rm, diag, off, *, epochs, T, lr, l2, seed, tr, te):
    """Adam on the Fermi-Dirac loss; records train/test loss and accuracy.

    One eigendecomposition per epoch, shared between the training loss/gradient
    and the held-out loss (identical H, different aggregates).
    """
    rng = np.random.default_rng(seed)
    K = Rp["tr"].shape[0]
    P = len(diag) + len(off)
    w = rng.normal(0.0, 0.05, P)
    mom = np.zeros(P); vel = np.zeros(P)
    b1, b2, eps = 0.9, 0.999, 1e-8
    Mtr, Mte = int(tr.sum()), int(te.sum())
    hist = {k: [] for k in ("loss_tr", "loss_te", "acc_tr", "acc_te")}

    for ep in range(1, epochs + 1):
        lam, U = spectrum(w, diag, off, K)
        rp_tr, rm_tr = rotate(Rp["tr"], U), rotate(Rm["tr"], U)
        loss_tr = fd_loss(lam, rp_tr, rm_tr, Mtr, T) + l2 * float(w @ w)
        loss_te = fd_loss(lam, rotate(Rp["te"], U), rotate(Rm["te"], U), Mte, T)
        pred = np.where(X @ w >= 0, 1, -1)
        hist["loss_tr"].append(loss_tr)
        hist["loss_te"].append(loss_te)
        hist["acc_tr"].append(float((pred[tr] == y[tr]).mean()))
        hist["acc_te"].append(float((pred[te] == y[te]).mean()))

        g = fd_grad(lam, U, rp_tr, rm_tr, diag, off, Mtr, T) + 2 * l2 * w
        mom = b1 * mom + (1 - b1) * g
        vel = b2 * vel + (1 - b2) * g * g
        w -= lr * (mom / (1 - b1 ** ep)) / (np.sqrt(vel / (1 - b2 ** ep)) + eps)
    return w, hist


# --- driver ------------------------------------------------------------------

def stratified_split(y, frac, rng):
    te = np.zeros(len(y), bool)
    for cls in (-1, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        te[idx[:int(round(len(idx) * frac))]] = True
    return ~te, te


def aggregates(rho, y, mask):
    """R+ and R- over the selected molecules."""
    K = rho.shape[1]
    Rp = np.zeros((K, K)); Rm = np.zeros((K, K))
    for i in np.where(mask)[0]:
        (Rp if y[i] > 0 else Rm)[:] += rho[i]
    return Rp, Rm


def screening_score(Rp, Rm, npos, nneg):
    """||offdiag(D)||_F / ||diag(D)||_F for D = mean(rho | y=+1) - mean(rho | y=-1).

    Class *means*, not sums, so an unbalanced split cannot manufacture signal.
    Large numerator: real off-diagonal structure for a quantum pool to grip.
    Small denominator: a diagonal pool has nothing. Both tiny: unlearnable by
    any model whose decision is a threshold on Tr(rho H).
    """
    D = Rp / npos - Rm / nneg
    dg = np.linalg.norm(np.diag(D))
    od = np.sqrt(max(np.linalg.norm(D) ** 2 - dg ** 2, 0.0))
    return float(od), float(dg), float(od / (dg + 1e-12))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--rho", required=True)
    ap.add_argument("--out", default="figures/spin_quantum_vs_classical.png")
    ap.add_argument("--json-out", default="results/spin_comparison_metrics.json")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--screen", default="results/qh9_conjugation_screen_full.csv",
                    help="classical-descriptor table for the confound check")
    ap.add_argument("--replot", action="store_true",
                    help="re-render the figure from an existing --json-out, no retraining")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.replot:
        with open(args.json_out) as fh:
            d = json.load(fh)
        args.epochs = len(d["history"]["S2"]["quantum"]["loss_te"])
        plot(d["history"], LABEL_TITLES, d["summary"], args)
        return 0

    L = np.load(args.labels, allow_pickle=False)
    rho = np.load(args.rho, mmap_mode="r")
    n = int(L["n_qubits"]); K = 1 << n
    N = rho.shape[0]
    assert rho.shape == (N, K, K), rho.shape
    logger.info("%d molecules, %d qubits (K=%d)", N, n, K)

    # normalise every state to unit trace (the projection loses ~0.3%)
    logger.info("loading and normalising rho stack ...")
    R = np.empty((N, K, K), np.float64)
    for i in range(N):
        m = np.asarray(rho[i], np.float64)
        R[i] = m / np.trace(m)

    pools = {kind: build_pool(n, kind)
             for kind in ("quantum", "classical", "classical_full")}
    for kind, (d, o, lab) in pools.items():
        logger.info("pool %-16s %4d params, %s",
                    kind, len(d) + len(o), "diagonal" if not o else "off-diagonal reach")

    feats = {kind: np.stack([features(R[i], d, o) for i in range(N)])
             for kind, (d, o, _) in pools.items()}

    # Positive control: a label built to be coherence-only by construction.
    # A_od is a random combination of the pool's XX/YY generators -- strictly
    # off-diagonal, and inside the quantum pool's span by construction. The
    # quantum neuron must therefore be able to fit it near-perfectly, while a
    # diagonal pool provably cannot see it at all. This is what proves the
    # apparatus can detect an advantage when one exists, so that a null result
    # on the physical labels is informative rather than an artefact.
    nd_q = len(pools["quantum"][0])
    coef = np.random.default_rng(args.seed).normal(size=len(pools["quantum"][1]))
    control = feats["quantum"][:, nd_q:] @ coef

    label_defs = {k: (title, L[k] if k in L.files else control)
                  for k, (title, _) in LABEL_TITLES.items()}

    # The control's exact solution is known in closed form: H = A_od - theta*I,
    # i.e. w* = coef on the XX/YY block and -median on the identity. Evaluating
    # its accuracy and its Fermi-Dirac loss separates "the signal is not there"
    # from "the objective does not chase it".
    wstar = np.zeros(nd_q + len(pools["quantum"][1]))
    wstar[0] = -float(np.median(control))
    wstar[nd_q:] = coef
    exact_pred = np.where(feats["quantum"] @ wstar >= 0, 1, -1)
    exact_y = np.where(control > np.median(control), 1, -1)

    rng = np.random.default_rng(args.seed)
    results, summary = {}, {}
    for lname, (pretty, values) in label_defs.items():
        y = np.where(values > np.median(values), 1, -1)
        tr, te = stratified_split(y, args.test_frac, rng)
        Rp_tr, Rm_tr = aggregates(R, y, tr)
        Rp_te, Rm_te = aggregates(R, y, te)
        od, dg, score = screening_score(Rp_tr, Rm_tr,
                                        int((y[tr] > 0).sum()), int((y[tr] < 0).sum()))
        logger.info("label %s: %+d/%+d, screen ||offdiag||=%.4g ||diag||=%.4g ratio=%.4f",
                    lname, int((y > 0).sum()), int((y < 0).sum()), od, dg, score)
        summary[lname] = {"screen_offdiag": od, "screen_diag": dg,
                          "screen_ratio": score,
                          "n_pos": int((y > 0).sum()), "n_neg": int((y < 0).sum())}
        results[lname] = {}
        for kind, (d, o, _) in pools.items():
            t0 = time.perf_counter()
            w, hist = train(feats[kind], y, {"tr": Rp_tr, "te": Rp_te},
                            {"tr": Rm_tr, "te": Rm_te}, d, o,
                            epochs=args.epochs, T=args.temperature, lr=args.lr,
                            l2=args.l2, seed=args.seed, tr=tr, te=te)
            results[lname][kind] = hist
            summary[lname][kind] = {
                "params": len(d) + len(o),
                "final_loss_tr": hist["loss_tr"][-1], "final_loss_te": hist["loss_te"][-1],
                "final_acc_tr": hist["acc_tr"][-1], "final_acc_te": hist["acc_te"][-1],
                "best_acc_te": max(hist["acc_te"]),
            }
            logger.info("  %-16s acc train %.3f / test %.3f   loss test %.4f   (%.0fs)",
                        kind, hist["acc_tr"][-1], hist["acc_te"][-1],
                        hist["loss_te"][-1], time.perf_counter() - t0)
        if lname == "control":
            d, o, _ = pools["quantum"]
            lam, U = spectrum(wstar, d, o, 1 << n)
            summary[lname]["exact_solution"] = {
                "acc_tr": float((exact_pred[tr] == exact_y[tr]).mean()),
                "acc_te": float((exact_pred[te] == exact_y[te]).mean()),
                "fd_loss_tr": fd_loss(lam, rotate(Rp_tr, U), rotate(Rm_tr, U),
                                      int(tr.sum()), args.temperature)}
            ex = summary[lname]["exact_solution"]
            logger.info("  %-16s acc train %.3f / test %.3f   FD loss %.4f  "
                        "<- the KNOWN optimum of the decision rule",
                        "EXACT solution", ex["acc_tr"], ex["acc_te"], ex["fd_loss_tr"])

    # --- confound check: can plain classical descriptors predict the label? ---
    summary["descriptor_baseline"] = descriptor_baseline(args.screen, L, label_defs,
                                                         args.test_frac, args.seed)
    summary["config"] = vars(args)
    summary["label_stats"] = {
        k: {"corr_S2_D": float(np.corrcoef(L["S2"], L["D"])[0, 1]),
            "corr_S2_c": float(np.corrcoef(L["S2"], L["c"])[0, 1]),
            "corr_D_c": float(np.corrcoef(L["D"], L["c"])[0, 1])}
        for k in ("all",)}
    summary["rho_trace_kept"] = {"min": float(L["rho_trace_kept"].min()),
                                 "median": float(np.median(L["rho_trace_kept"]))}

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as fh:
        json.dump({"summary": summary,
                   "history": {k: {m: v for m, v in d.items()}
                               for k, d in results.items()}}, fh, indent=1)
    logger.info("wrote %s", args.json_out)

    plot(results, label_defs, summary, args)
    return 0


def descriptor_baseline(screen_path, L, label_defs, test_frac, seed):
    """Logistic regression on formula/spectrum descriptors -- the Q4 confound check."""
    import csv
    try:
        table = {int(r["idx"]): r for r in csv.DictReader(open(screen_path))}
    except OSError:
        return {"error": f"missing {screen_path}"}
    idx = L["idx"]
    cols = ["DoU", "gap_Ha", "n_frontier_within_kT", "largest_pi_atoms",
            "n_aromatic_atoms", "n_heavy"]
    rows = []
    for i in idx:
        r = table.get(int(i))
        if r is None:
            return {"error": "molecule missing from screen table"}
        rows.append([float(r[c]) if r[c] not in ("", None) else 0.0 for c in cols])
    Xd = np.asarray(rows)
    Xd = np.column_stack([Xd, Xd ** 2])                    # mild nonlinearity
    Xd = (Xd - Xd.mean(0)) / np.where(Xd.std(0) < 1e-12, 1.0, Xd.std(0))
    Xd = np.column_stack([Xd, np.ones(len(Xd))])

    rng = np.random.default_rng(seed)
    out = {}
    for lname, (_, values) in label_defs.items():
        y01 = (values > np.median(values)).astype(float)
        tr, te = stratified_split(np.where(y01 > 0, 1, -1), test_frac, rng)
        w = np.zeros(Xd.shape[1])
        for _ in range(4000):                              # plain GD, ridge
            p = 1.0 / (1.0 + np.exp(-Xd[tr] @ w))
            w -= 0.5 * (Xd[tr].T @ (p - y01[tr]) / tr.sum() + 1e-3 * w)
        pred = (Xd @ w > 0).astype(float)
        out[lname] = {"acc_tr": float((pred[tr] == y01[tr]).mean()),
                      "acc_te": float((pred[te] == y01[te]).mean()),
                      "features": cols}
    return out


def plot(results, label_defs, summary, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUT, HAIR = "#191A24", "#565873", "#E2E2EC"
    QC, CC, FC = "#6C5CE0", "#D9822B", "#9A9CB4"
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": HAIR,
                         "axes.linewidth": 1, "text.color": INK,
                         "axes.labelcolor": MUT, "xtick.color": MUT,
                         "ytick.color": MUT, "figure.dpi": 200})
    # Drawn back-to-front so the overlap is visible: the wide classical band
    # underneath, the quantum curve dashed on top of it. Where the dashes track
    # the band exactly, the two models are doing the same thing.
    order = [("classical_full", FC, 4.2, "-", 0.55,
              "Classical, unconstrained diagonal (1024 params)"),
             ("classical", CC, 3.4, "-", 1.0,
              "Classical neuron — I, Z, ZZ (strictly diagonal)"),
             ("quantum", QC, 1.5, (0, (4, 3)), 1.0,
              "Quantum neuron — I, Z, ZZ + XX, YY (off-diagonal reach)")]

    nrow = len(label_defs)
    fig, axes = plt.subplots(nrow, 2, figsize=(12.4, 4.2 * nrow))
    ep = np.arange(1, args.epochs + 1)
    for row, (lname, (pretty, _)) in enumerate(label_defs.items()):
        a_loss, a_acc = axes[row, 0], axes[row, 1]
        for kind, col, lw, ls, alpha, lab in order:
            h = results[lname][kind]
            a_loss.plot(ep, h["loss_te"], color=col, lw=lw, ls=ls, alpha=alpha, label=lab)
            a_acc.plot(ep, 100 * np.array(h["acc_te"]), color=col, lw=lw, ls=ls,
                       alpha=alpha, label=lab)

        base = 100 * summary["descriptor_baseline"][lname]["acc_te"]
        a_acc.axhline(base, color="#2E9E6B", lw=1.5, ls=(0, (6, 3)))
        a_acc.annotate(f"classical chemical descriptors only "
                       f"(DoU, gap, aromaticity …): {base:.1f}%",
                       (args.epochs, base), fontsize=8.4, color="#2E9E6B",
                       ha="right", va="top", xytext=(-3, -4), textcoords="offset points")
        a_acc.axhline(50, color=INK, lw=1.1, ls=(0, (5, 3)), alpha=0.5)
        a_acc.annotate("chance", (5, 50), fontsize=8.4, color=INK, ha="left",
                       va="bottom", xytext=(0, 3), textcoords="offset points", alpha=0.7)
        a_acc.set_ylim(45, 103)

        q, c = summary[lname]["quantum"], summary[lname]["classical"]
        gap = 100 * (q["final_acc_te"] - c["final_acc_te"])
        # keep the box clear of the descriptor-baseline line, wherever it falls
        y_box = min(0.33, (base - 45) / 58 - 0.17)
        both = (f"both {100*q['final_acc_te']:.1f}%" if abs(gap) < 0.05 else
                f"{100*q['final_acc_te']:.1f}% vs {100*c['final_acc_te']:.1f}%")
        a_acc.annotate(f"quantum − classical  =  {gap:+.2f} points\n{both} held-out",
                       (0.97, y_box), xycoords="axes fraction", ha="right", va="center",
                       fontsize=10, color=INK, weight="bold", linespacing=1.5)
        a_loss.annotate(f"Δloss  =  {q['final_loss_te'] - c['final_loss_te']:+.5f}",
                        (0.97, 0.9), xycoords="axes fraction", ha="right",
                        va="top", fontsize=9.5, color=INK, weight="bold")

        ex = summary[lname].get("exact_solution")
        if ex:
            a_acc.annotate(
                f"An exact solution EXISTS in the quantum pool:\n"
                f"{100*ex['acc_te']:.0f}% held-out — but its Fermi–Dirac loss is "
                f"{ex['fd_loss_tr']:.2f},\nfar above the {q['final_loss_tr']:.2f} the optimizer "
                f"converges to.\nThe loss prefers a worse-accuracy operator.",
                (0.5, 0.62), xycoords="axes fraction", ha="center", va="center",
                fontsize=9, color="#B3261E", linespacing=1.6)

        a_loss.set_ylabel("Fermi–Dirac log-loss (held-out)")
        a_acc.set_ylabel("classification accuracy (%, held-out)")
        for ax in (a_loss, a_acc):
            ax.set_xlabel("epoch")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color=HAIR, lw=0.8, alpha=0.7)
            ax.set_axisbelow(True)
        ratio = summary[lname]["screen_ratio"]
        a_loss.set_title(f"label: {pretty}", fontsize=12, color=INK,
                         weight="bold", loc="left", pad=9)
        a_acc.set_title(f"R± screen   ‖offdiag(R₊−R₋)‖ / ‖diag(R₊−R₋)‖  =  {ratio:.3f}",
                        fontsize=9.5, color=MUT, loc="left", pad=9)
    h_in = fig.get_size_inches()[1]
    legend_y, foot_y, top_y = 0.45 / h_in, 0.10 / h_in, 1 - 0.30 / h_in
    handles, labs = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labs, frameon=False, fontsize=9.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, legend_y))

    fig.suptitle("Quantum vs classical Fermi–Dirac neuron on molecular thermal states",
                 fontsize=15, weight="bold", color=INK, x=0.055, ha="left",
                 y=1 - 0.06 / h_in)
    kept = summary["rho_trace_kept"]["min"]
    fig.text(0.055, foot_y * 0.35,
             "1000 QH9 molecules · CAS(8,8) · kT = 0.1 Ha · thermal states projected onto the "
             f"1024 most-populated determinants (10 qubits, ≥{100*kept:.1f}% of the thermal "
             "weight retained)\nIdentical Fermi–Dirac loss, optimizer, temperature, split and "
             "epoch budget — the models differ only in whether the operator pool reaches off "
             "the diagonal.",
             fontsize=8.2, color=MUT, ha="left")
    fig.tight_layout(rect=(0, legend_y + 0.30 / h_in, 1, top_y))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
