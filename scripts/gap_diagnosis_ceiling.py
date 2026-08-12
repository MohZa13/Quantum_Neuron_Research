"""The honest classical ceiling, and the exact R+/R- screen, for the gap label.

``gap_diagnosis.py`` calls the 136 Z/ZZ Pauli features "the classical model".
That is a *subset* of what a dephased model may read: the true classical object
is the full determinant population vector ``diag(rho)`` -- 4900 numbers, not 136.
This script trains on that (via its leading principal components, since
p >> n) so the null result cannot be blamed on an under-powered baseline.

It also reports the project's own screening statistic
``||offdiag(R+ - R-)||_F / ||diag(R+ - R-)||_F`` computed on the true
class-aggregated density matrices, so the gap label is directly comparable to
the numbers already measured for <S^2> (0.122), c (0.162), and the synthetic
purely off-diagonal control (0.34).

Requires ``results/gap_rho_pass.npz``.  Writes ``results/gap_diagnosis_ceiling.json``.

    .venv/bin/python scripts/gap_diagnosis_ceiling.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

import gap_diagnosis as G
from gap_diagnosis_followup import cached_load

REPO = Path(__file__).resolve().parents[1]
PASS = REPO / "results/gap_rho_pass.npz"
OUT = REPO / "results/gap_diagnosis_ceiling.json"
N_PC = 300


def pca(X: np.ndarray, k: int) -> np.ndarray:
    Xc = X - X.mean(0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T, (s[:k] ** 2).sum() / (s ** 2).sum()


def main() -> None:
    z = np.load(PASS)
    D = cached_load()
    assert np.array_equal(z["idx"], D["idx"]), "molecule order mismatch"

    X, gap = D["X"], D["gap"]
    y = z["y"]
    out: dict = {}

    # ---- exact screening ratio on the aggregated density matrices ---------
    out["screen_ratio_rho"] = float(z["screen_ratio"])
    out["dR_norms"] = dict(diag=float(z["dR_diag_fro"]),
                           offdiag=float(z["dR_offdiag_fro"]))
    print(f"R+/R- screen on rho: ||offdiag||/||diag|| = {z['screen_ratio']:.4f}")

    # per-molecule off-diagonal share of rho, in the project's convention
    # ||offdiag(rho)||_F^2 / ||rho||_F^2  (squared, so it matches the logged 6.7%)
    share = z["nrm_offdiag"] ** 2 / (z["nrm_offdiag"] ** 2 + z["nrm_diag"] ** 2)
    out["rho_offdiag_share"] = dict(median=float(np.median(share)),
                                    p90=float(np.percentile(share, 90)),
                                    max=float(np.max(share)))

    # ---- the classical ceiling: the full determinant populations ----------
    diag_rho = z["diag_rho"].astype(np.float64)
    keep = diag_rho.std(0) > 0
    Xpc, ev = pca(diag_rho[:, keep], N_PC)
    print(f"diag(rho): {int(keep.sum())} varying entries -> {N_PC} PCs "
          f"({ev * 100:.2f}% of variance)")
    out["diag_rho_pca"] = dict(n_entries=int(keep.sum()), n_pcs=N_PC,
                               variance_explained=float(ev))

    sets = {
        "diagonal Pauli, Z+ZZ (136)": X[:, :G.N_DIAG],
        "full diag(rho), 300 PCs": Xpc,
        "full Pauli pool (248)": X,
        "full diag(rho) + off-diagonal Pauli": np.column_stack([Xpc, X[:, G.N_DIAG:]]),
    }
    res = {}
    for nm, F in sets.items():
        a, u, _ = G.eval_classifier(F, y)
        res[nm] = dict(acc=G.summ(a * 100), auc=G.summ(u), n_features=F.shape[1])
        print(f"  {nm:38s} acc {np.mean(a) * 100:5.1f} +- "
              f"{np.std(a, ddof=1) * 100:.1f}   auc {np.mean(u):.3f}")
    out["ceiling_classification"] = res

    a_q, _, _ = G.eval_classifier(sets["full diag(rho) + off-diagonal Pauli"], y)
    a_c, _, _ = G.eval_classifier(sets["full diag(rho), 300 PCs"], y)
    d = (a_q - a_c) * 100
    out["ceiling_quantum_minus_classical"] = dict(
        **G.summ(d), p=float(stats.ttest_rel(a_q, a_c).pvalue))
    print(f"  adding the coherence channel to the true classical ceiling: "
          f"{np.mean(d):+.2f} +- {np.std(d, ddof=1):.2f} points")

    # regression, same comparison
    r2c, _ = G.eval_regressor(sets["full diag(rho), 300 PCs"], gap)
    r2q, _ = G.eval_regressor(sets["full diag(rho) + off-diagonal Pauli"], gap)
    out["ceiling_regression"] = dict(classical=G.summ(r2c), quantum=G.summ(r2q),
                                     delta=G.summ(r2q - r2c))
    print(f"  regression R2: classical {np.mean(r2c):.3f} -> "
          f"quantum {np.mean(r2q):.3f}")

    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
