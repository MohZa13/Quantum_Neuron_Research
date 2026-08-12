"""Does the quantum-classical gap grow with multireference character?

`localized_basis_experiment.py` established the invariant on which everything
turns.  The off-diagonal share of rho is **not** basis-invariant -- full
Edmiston-Ruedenberg localization drives it from 0.058 to 0.929 -- but the
inflation is bookkeeping: a zero-correlation single determinant goes to 0.935
under the same rotation.  Restricted to rotations that preserve the reference
determinant (occupied-occupied and virtual-virtual), the share is invariant to
5e-5.  So the split is well defined exactly as long as a dominant reference
exists, and a molecule offers genuine coherence only insofar as it has none.

The basis-invariant discriminant is therefore the natural-occupation spectrum,
summarised by the Head-Gordon count `N_unpaired = sum_i min(n_i, 2 - n_i)`.
QH9 spans 0.0003 to 0.48 of it; twisted ethylene spans 0.19 to 2.00.

This script asks whether the ablation improves *within* QH9's narrow range, so
that extrapolating to OMol25's strongly correlated systems is supported by a
trend rather than by hope.  Two probes:

  1. **Stratified ablation.** Quartiles of `N_unpaired`; full pool against the
     diagonal pool inside each.
  2. **Learning curves.** If coherence features were a better inductive bias
     they would pay off at small training-set sizes even where the diagonal
     wins asymptotically.

Writes ``results/multireference_stratification.json``.

    PYTHONPATH=scripts .venv/bin/python scripts/multireference_stratification.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

import gap_diagnosis as G
from gap_diagnosis_followup import cached_load

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/multireference_stratification.json"
SEEDS = 25

# labels carried forward: the OMol25 evaluation tasks, plus the reference
LABELS = ["ST_cas", "ST_corr", "IP_cas", "IP_corr", "N_unpaired"]


def ablate(X, Xd, y, seeds=SEEDS):
    a_f, _, _ = G.eval_classifier(X, y, seeds)
    a_d, _, _ = G.eval_classifier(Xd, y, seeds)
    d = (a_f - a_d) * 100
    return dict(quantum=G.summ(a_f * 100), classical=G.summ(a_d * 100),
                delta=G.summ(d), p=float(stats.ttest_rel(a_f, a_d).pvalue),
                n=int(len(y)))


def main() -> None:
    D = cached_load()
    z = np.load(REPO / "results/second_quantization_labels.npz")
    order = {int(i): k for k, i in enumerate(z["idx"])}
    sel = np.array([order[i] for i in D["idx"]])
    X = D["X"]
    Xd = X[:, :G.N_DIAG]
    nu = z["N_unpaired"][sel]
    out = {"N_unpaired_range": [float(nu.min()), float(nu.max())]}

    # ---------------------------------------------- 1. stratified ablation
    q = np.quantile(nu, [0, 0.25, 0.5, 0.75, 1.0])
    print("stratified ablation by N_unpaired quartile")
    strat = {}
    for lab in LABELS:
        y_all = z[lab][sel]
        rows = []
        for k in range(4):
            m = (nu >= q[k]) & (nu <= q[k + 1])
            yk = y_all[m]
            y = (yk <= np.median(yk)).astype(int)
            r = ablate(X[m], Xd[m], y)
            r["quartile"] = k + 1
            r["N_unpaired_median"] = float(np.median(nu[m]))
            rows.append(r)
            print(f"  {lab:11s} Q{k + 1} (N_unp {np.median(nu[m]):.3f})  "
                  f"Q {r['quantum']['mean']:5.1f}  C {r['classical']['mean']:5.1f}"
                  f"  delta {r['delta']['mean']:+5.2f}", flush=True)
        deltas = [r["delta"]["mean"] for r in rows]
        mids = [r["N_unpaired_median"] for r in rows]
        strat[lab] = dict(
            quartiles=rows,
            trend_spearman=float(stats.spearmanr(mids, deltas).statistic),
            delta_first=deltas[0], delta_last=deltas[-1])
    out["stratified"] = strat

    # ------------------------------------------------- 2. learning curves
    print("\nlearning curves (full pool vs diagonal pool)")
    curves = {}
    for lab in ("ST_corr", "IP_corr"):
        y_all = z[lab][sel]
        y = (y_all <= np.median(y_all)).astype(int)
        pts = []
        for n_tr in (60, 120, 250, 500, 700):
            accs_f, accs_d = [], []
            for s in range(15):
                rng = np.random.default_rng(500 + s)
                tr, te = G.stratified_split(y, rng)
                idx_tr = np.where(tr)[0]
                rng.shuffle(idx_tr)
                sub = idx_tr[:n_tr]
                for F, acc in ((X, accs_f), (Xd, accs_d)):
                    lam = G.cv_lambda(F[sub], y[sub], G.LAMS_C,
                                      np.random.default_rng(900 + s))
                    Xtr, Xte = G.standardize(F[sub], F[te])
                    w = G.fit_logistic(Xtr, y[sub], lam)
                    acc.append(float(np.mean(
                        (G.logistic_score(Xte, w) > 0) == y[te])))
            af, ad = np.array(accs_f) * 100, np.array(accs_d) * 100
            pts.append(dict(n_train=n_tr, quantum=G.summ(af),
                            classical=G.summ(ad),
                            delta=G.summ(af - ad)))
            print(f"  {lab:8s} n={n_tr:4d}  Q {af.mean():5.1f}  C {ad.mean():5.1f}"
                  f"  delta {(af - ad).mean():+5.2f}", flush=True)
        curves[lab] = pts
    out["learning_curves"] = curves

    OUT.write_text(json.dumps(out, indent=2))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
