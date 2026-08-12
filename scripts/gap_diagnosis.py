"""Why does the HOMO-LUMO gap label show no quantum advantage?

Four candidate explanations, each with a decisive measurement:

  H1  the states carry too little coherence for any label to exploit
  H2  binarising a continuous property is what destroys the signal
      (i.e. the failure is classification-vs-regression, not physics)
  H3  the gap is a classically determined quantity -- a mean-field, one-body
      eigenvalue difference -- so a dephased model loses nothing by construction
  H4  something else

The instrument is the exact diagonal/off-diagonal split of the 248-term
extended-Heisenberg feature vector.  On a Jordan-Wigner register the Z and ZZ
strings (136 of them) are diagonal operators: their expectation values are
functions of ``diag(rho)`` alone -- exactly what a classical/dephased model may
read (HYBRID_BACKPROP.md §5.2).  The XX/YY strings (112) read *only* off-diagonal
entries of rho.  So "quantum minus classical" is a within-model-class feature
ablation with no baseline-tuning argument available to a critic.

Writes ``results/gap_diagnosis.json``.  Cheap: seconds, no run-file access
(``scripts/gap_rho_pass.py`` covers the parts that need rho itself).

    .venv/bin/python scripts/gap_diagnosis.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
EXTHEIS = REPO / "results/qh9_dense_cas8-8_kT0p1_extheis.h5"
RUN = REPO / "results/qh9_dense_cas8-8_kT0p1.h5"
SCREEN = REPO / "results/qh9_conjugation_screen_full.csv"
CACHE = REPO / "results/presentation_cache.npz"
OUT = REPO / "results/gap_diagnosis.json"
KT = "kT_0p1000"

NCAS, NQ = 8, 16
N_DIAG = NQ + NQ * (NQ - 1) // 2                       # 16 Z + 120 ZZ = 136
N_SEEDS = 25
TEST_FRAC = 0.30
HA_EV = 27.211386


# ----------------------------------------------------------------- estimators
def standardize(tr: np.ndarray, te: np.ndarray):
    mu, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (tr - mu) / sd, (te - mu) / sd


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """L2-penalised logistic regression by Newton-Raphson (bias unpenalised)."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    P = np.eye(d + 1) * lam
    P[-1, -1] = 0.0
    w = np.zeros(d + 1)
    for _ in range(60):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        grad = Xb.T @ (p - y) + P @ w
        W = np.clip(p * (1 - p), 1e-8, None)
        H = (Xb * W[:, None]).T @ Xb + P
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def logistic_score(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w[:-1] + w[-1]


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    n, d = X.shape
    ym = y.mean()
    A = X.T @ X + lam * np.eye(d)
    w = np.linalg.solve(A, X.T @ (y - ym))
    return w, ym


def auc(score: np.ndarray, y: np.ndarray) -> float:
    r = stats.rankdata(score)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stratified_split(y: np.ndarray, rng: np.random.Generator, frac=TEST_FRAC):
    te = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        te.append(ci[:int(round(len(ci) * frac))])
    te = np.concatenate(te)
    mask = np.zeros(len(y), bool)
    mask[te] = True
    return ~mask, mask


def cv_lambda(X, y, lams, rng, kind="logistic", folds=4):
    """Pick lambda by k-fold CV on the training block."""
    n = len(y)
    order = rng.permutation(n)
    fold = np.array_split(order, folds)
    best, best_s = lams[0], -np.inf
    for lam in lams:
        s = 0.0
        for k in range(folds):
            va = fold[k]
            tr = np.concatenate([fold[j] for j in range(folds) if j != k])
            Xtr, Xva = standardize(X[tr], X[va])
            if kind == "logistic":
                w = fit_logistic(Xtr, y[tr], lam)
                s += float(np.mean((logistic_score(Xva, w) > 0) == y[va]))
            else:
                w, ym = fit_ridge(Xtr, y[tr], lam)
                pred = Xva @ w + ym
                s += -float(np.mean((pred - y[va]) ** 2))
        if s > best_s:
            best_s, best = s, lam
    return best


LAMS_C = [0.03, 0.3, 3.0, 30.0, 300.0, 3000.0]
LAMS_R = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def eval_classifier(X, y, seeds=N_SEEDS):
    """Repeated stratified hold-out; returns per-seed accuracy, AUC, predictions."""
    accs, aucs, preds = [], [], []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        tr, te = stratified_split(y, rng)
        lam = cv_lambda(X[tr], y[tr], LAMS_C, np.random.default_rng(7000 + s))
        Xtr, Xte = standardize(X[tr], X[te])
        w = fit_logistic(Xtr, y[tr], lam)
        sc = logistic_score(Xte, w)
        accs.append(float(np.mean((sc > 0) == y[te])))
        aucs.append(auc(sc, y[te]))
        preds.append(((sc > 0).astype(int), y[te], np.where(te)[0]))
    return np.array(accs), np.array(aucs), preds


def eval_regressor(X, t, seeds=N_SEEDS):
    """Repeated hold-out ridge; returns per-seed test R^2 and Spearman rho."""
    r2s, rhos = [], []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        tr, te = stratified_split((t <= np.median(t)).astype(int), rng)
        lam = cv_lambda(X[tr], t[tr], LAMS_R, np.random.default_rng(7000 + s),
                        kind="ridge")
        Xtr, Xte = standardize(X[tr], X[te])
        w, ym = fit_ridge(Xtr, t[tr], lam)
        pred = Xte @ w + ym
        ss_res = float(np.sum((t[te] - pred) ** 2))
        ss_tot = float(np.sum((t[te] - t[tr].mean()) ** 2))
        r2s.append(1.0 - ss_res / ss_tot)
        rhos.append(float(stats.spearmanr(pred, t[te]).statistic))
    return np.array(r2s), np.array(rhos)


def oof_predictions(X, t, folds=10, seed=0):
    """Out-of-fold ridge predictions of a continuous target (for residualising)."""
    n = len(t)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    parts = np.array_split(order, folds)
    pred = np.zeros(n)
    for k in range(folds):
        va = parts[k]
        tr = np.concatenate([parts[j] for j in range(folds) if j != k])
        lam = cv_lambda(X[tr], t[tr], LAMS_R, np.random.default_rng(99 + k),
                        kind="ridge")
        Xtr, Xva = standardize(X[tr], X[va])
        w, ym = fit_ridge(Xtr, t[tr], lam)
        pred[va] = Xva @ w + ym
    return pred


def summ(a: np.ndarray) -> dict:
    return dict(mean=float(np.mean(a)), std=float(np.std(a, ddof=1)),
                lo=float(np.percentile(a, 2.5)), hi=float(np.percentile(a, 97.5)))


# ---------------------------------------------------------------------- data
def load() -> dict:
    with h5py.File(EXTHEIS, "r") as f:
        names = sorted((k for k in f if k.startswith("mol_")),
                       key=lambda s: int(s.split("_")[1]))
        idx, feats, trace = [], [], []
        for nm in names:
            grp = f[nm][KT]
            idx.append(int(nm.split("_")[1]))
            feats.append(grp["coeffs"][:])
            trace.append(float(grp.attrs["trace"]))
    X = np.asarray(feats, float) / np.asarray(trace)[:, None]   # I8: unit trace
    idx = np.asarray(idx)

    rows = {int(r["idx"]): r for r in csv.DictReader(open(SCREEN))}
    keep = np.array([i in rows for i in idx])
    X, idx = X[keep], idx[keep]
    r = [rows[i] for i in idx]
    gap = np.array([float(x["gap_Ha"]) for x in r])
    def col(key, default=0.0):
        # the tier-3 RDKit fields are blank where perception failed
        return np.array([float(x[key]) if x[key] else default for x in r])

    desc = dict(
        DoU=col("DoU"),
        n_heavy=col("n_heavy"),
        pi_atoms=col("largest_pi_atoms"),
        aromatic=col("n_aromatic_atoms"),
        pi_missing=np.array([0.0 if x["largest_pi_atoms"] else 1.0 for x in r]),
    )
    formula = np.array([x["formula"] for x in r])
    for el in ("C", "H", "N", "O", "F"):
        desc[f"n_{el}"] = np.array(
            [float(m.group(1) or 1) if (m := re.search(el + r"(\d*)(?![a-z])", fo))
             else 0.0 for fo in formula])

    # state-level classical scalars (spectrum / occupations), from the run file
    with h5py.File(RUN, "r") as f:
        p0, ent, ci_gap, natocc, csq, sc, evals10 = [], [], [], [], [], [], []
        for i in idx:
            g = f[f"mol_{i}"]
            ev = g["evals"]
            e0, e1 = float(ev[0]), float(ev[1])
            ci_gap.append(e1 - e0)
            evals10.append(ev[:10] - e0)
            k = g[KT]
            pk = k["p"][:]
            pk = pk / pk.sum()
            p0.append(float(pk[0]))
            ent.append(float(k["entropy"][()]))
            natocc.append(k["nat_occs"][:])
            csq.append(float(k["c_max_sq"][()]))
            sc.append(float(k["static_corr"][()]))
    state = dict(p0=np.array(p0), entropy=np.array(ent), ci_gap=np.array(ci_gap),
                 c_max_sq=np.array(csq), static_corr=np.array(sc))
    return dict(idx=idx, X=X, gap=gap, desc=desc, formula=formula, state=state,
                nat_occs=np.array(natocc), evals10=np.array(evals10))


# ---------------------------------------------------------------------- main
def main() -> None:
    D = load()
    X, gap = D["X"], D["gap"]
    n = len(gap)
    y = (gap <= np.median(gap)).astype(int)
    out: dict = dict(n_molecules=int(n), gap_median_Ha=float(np.median(gap)))
    print(f"{n} molecules | small-gap {y.sum()} large-gap {(1 - y).sum()}")

    # ---- feature blocks (exact, by construction of the Pauli basis) --------
    Xd, Xo = X[:, :N_DIAG], X[:, N_DIAG:]
    Xz = X[:, :NQ]
    n_xx = Xo.shape[1] // 2
    out["blocks"] = dict(diag=N_DIAG, offdiag=Xo.shape[1], z_single=NQ)

    # sanity: on a real, S_z-conserving sector <XX> = <YY> pairwise
    xx, yy = Xo[:, :n_xx], Xo[:, n_xx:]
    out["xx_equals_yy_max_abs_dev"] = float(np.max(np.abs(xx - yy)))

    # relative magnitude of the coherence channel
    md, mo = np.linalg.norm(Xd, axis=1), np.linalg.norm(Xo, axis=1)
    out["feature_norm_ratio"] = dict(
        median=float(np.median(mo / md)), mean=float(np.mean(mo / md)),
        max=float(np.max(mo / md)))
    # what matters for learning is spread across molecules, not magnitude
    vd, vo = Xd.var(0).sum(), Xo.var(0).sum()
    out["feature_variance_share_offdiag"] = float(vo / (vd + vo))

    # ---- H3/H1: the accuracy ladder ---------------------------------------
    desc = D["desc"]
    Xdesc = np.column_stack([desc[k] for k in sorted(desc)])
    Xdesc2 = np.column_stack([Xdesc, Xdesc ** 2,
                              *[Xdesc[:, i] * Xdesc[:, j]
                                for i in range(Xdesc.shape[1])
                                for j in range(i + 1, Xdesc.shape[1])]])
    st = D["state"]
    Xspec = np.column_stack([st["p0"], st["entropy"], st["ci_gap"],
                             st["c_max_sq"], st["static_corr"], D["evals10"]])
    Xnat = D["nat_occs"]

    sets = {
        "composition (formula + DoU)": Xdesc,
        "composition, quadratic": Xdesc2,
        "natural occupations (8)": Xnat,
        "rho spectrum + CI gap": Xspec,
        "Z only (16)": Xz,
        "diagonal Pauli, Z+ZZ (136)": Xd,
        "off-diagonal Pauli, XX/YY (112)": Xo,
        "full Pauli pool (248)": X,
    }
    ladder = {}
    for nm, F in sets.items():
        a, u, _ = eval_classifier(F, y)
        ladder[nm] = dict(acc=summ(a * 100), auc=summ(u), n_features=F.shape[1])
        print(f"  {nm:34s} acc {np.mean(a) * 100:5.1f} +- "
              f"{np.std(a, ddof=1) * 100:.1f}   auc {np.mean(u):.3f}")
    out["ladder_classification"] = ladder

    # paired quantum - classical, seed by seed (same splits)
    a_full, _, p_full = eval_classifier(X, y)
    a_diag, _, p_diag = eval_classifier(Xd, y)
    d = (a_full - a_diag) * 100
    t = stats.ttest_rel(a_full, a_diag)
    out["quantum_minus_classical_points"] = dict(
        **summ(d), t=float(t.statistic), p=float(t.pvalue))
    print(f"\n  quantum - classical: {np.mean(d):+.2f} +- {np.std(d, ddof=1):.2f} "
          f"points (paired t p={t.pvalue:.3f})")

    # McNemar over pooled held-out predictions
    b = c = 0
    for (pf, yf, _), (pd_, yd, _) in zip(p_full, p_diag):
        cf, cd = pf == yf, pd_ == yd
        b += int(np.sum(cf & ~cd))
        c += int(np.sum(~cf & cd))
    out["mcnemar"] = dict(quantum_only_correct=b, classical_only_correct=c,
                          p=float(stats.binomtest(b, b + c, 0.5).pvalue))

    # ---- H2: is binarisation the problem? ---------------------------------
    reg = {}
    for nm in ("composition (formula + DoU)", "rho spectrum + CI gap",
               "diagonal Pauli, Z+ZZ (136)", "off-diagonal Pauli, XX/YY (112)",
               "full Pauli pool (248)"):
        r2, rho = eval_regressor(sets[nm], gap)
        reg[nm] = dict(r2=summ(r2), spearman=summ(rho))
        print(f"  R2 {nm:34s} {np.mean(r2):6.3f}   rho {np.mean(rho):.3f}")
    out["ladder_regression"] = reg
    r2f, _ = eval_regressor(X, gap)
    r2d, _ = eval_regressor(Xd, gap)
    dr = r2f - r2d
    out["regression_quantum_minus_classical_r2"] = dict(
        **summ(dr), p=float(stats.ttest_rel(r2f, r2d).pvalue))
    print(f"  regression quantum - classical R2: {np.mean(dr):+.4f}")

    # accuracy vs distance from the threshold: is the loss concentrated at the cut?
    _, _, preds = eval_classifier(X, y)
    dist = np.abs(gap - np.median(gap)) * HA_EV
    qs = np.quantile(dist, np.linspace(0, 1, 6))
    binacc = [[] for _ in range(5)]
    for pf, yf, ii in preds:
        for k in range(5):
            m = (dist[ii] >= qs[k]) & (dist[ii] <= qs[k + 1])
            if m.sum():
                binacc[k].append(float(np.mean(pf[m] == yf[m])))
    out["accuracy_vs_threshold_distance"] = dict(
        bin_edges_eV=[float(v) for v in qs],
        acc=[float(np.mean(v) * 100) for v in binacc])
    print("  acc by |gap-median| quintile:",
          [round(np.mean(v) * 100, 1) for v in binacc])

    # ---- H3 mechanism: where does the gap live? ---------------------------
    corr = {}
    for nm, v in [("CI gap E1-E0", st["ci_gap"]), ("ground-state weight p0", st["p0"]),
                  ("von Neumann entropy", st["entropy"]), ("c_max_sq", st["c_max_sq"]),
                  ("static_corr", st["static_corr"]), ("DoU", desc["DoU"]),
                  ("largest pi system", desc["pi_atoms"]),
                  ("n_heavy", desc["n_heavy"])]:
        corr[nm] = dict(pearson=float(stats.pearsonr(gap, v).statistic),
                        spearman=float(stats.spearmanr(gap, v).statistic))
    cache = np.load(CACHE, allow_pickle=True)
    order = {int(i): k for k, i in enumerate(cache["idx"])}
    sel = np.array([order[i] for i in D["idx"]])
    coh = cache["coh_share"][sel]
    corr["off-diagonal coherence share of rho"] = dict(
        pearson=float(stats.pearsonr(gap, coh).statistic),
        spearman=float(stats.spearmanr(gap, coh).statistic))
    out["gap_correlations"] = corr
    out["coherence_share"] = dict(median=float(np.median(coh)),
                                  p90=float(np.percentile(coh, 90)),
                                  max=float(np.max(coh)),
                                  spearman_vs_DoU=float(
                                      stats.spearmanr(coh, desc["DoU"]).statistic))

    # ---- does coherence explain what the diagonal cannot? -----------------
    pred_d = oof_predictions(Xd, gap)
    resid = gap - pred_d
    r2_resid, rho_resid = eval_regressor(Xo, resid)
    perm = []
    rng = np.random.default_rng(3)
    for _ in range(20):
        r2p, _ = eval_regressor(Xo, rng.permutation(resid), seeds=5)
        perm.append(float(np.mean(r2p)))
    out["residual_test"] = dict(
        r2_diag_oof=float(1 - np.var(resid) / np.var(gap)),
        r2_offdiag_on_residual=summ(r2_resid),
        spearman_offdiag_on_residual=summ(rho_resid),
        permuted_null_r2_mean=float(np.mean(perm)),
        permuted_null_r2_p95=float(np.percentile(perm, 95)))
    print(f"  residual test: diag explains R2={1 - np.var(resid) / np.var(gap):.3f}; "
          f"off-diag on the residual R2={np.mean(r2_resid):+.4f} "
          f"(null {np.mean(perm):+.4f})")

    # the same, as a classification: sign of the residual
    yr = (resid <= np.median(resid)).astype(int)
    for nm, F in (("off-diagonal", Xo), ("diagonal", Xd), ("composition", Xdesc)):
        a, u, _ = eval_classifier(F, yr)
        out.setdefault("residual_label_classification", {})[nm] = dict(
            acc=summ(a * 100), auc=summ(u))
        print(f"  residual-label acc, {nm:12s} {np.mean(a) * 100:5.1f}%")

    # ---- Pauli-space screening ratio, for reference against the rho-space one
    dR = X[y == 1].mean(0) - X[y == 0].mean(0)
    out["pauli_screen_ratio"] = float(
        np.linalg.norm(dR[N_DIAG:]) / np.linalg.norm(dR[:N_DIAG]))

    OUT.write_text(json.dumps(out, indent=2))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
