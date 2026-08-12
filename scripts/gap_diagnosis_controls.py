"""Two controls that decide how to read the gap-label null.

1. **Is the off-diagonal block information, or noise?**  Adding the 112 XX/YY
   features to the 136-term diagonal pool *costs* accuracy.  Adding 112
   Gaussian noise features costs the same amount if and only if the coherence
   channel is, for this label, indistinguishable from noise.

2. **How much of rho's coherence can the pool see at all?**  The
   extended-Heisenberg basis is weight <= 2, so it reads single-hop coherences
   only.  Comparing sum_j Tr(rho P_j)^2 / 2^Q against the exact
   ||offdiag(rho)||_F^2 from ``gap_rho_pass.py`` says what fraction of the
   off-diagonal weight is even in the model's field of view -- which separates
   "the states are classical" from "the pool is blind".

Writes ``results/gap_diagnosis_controls.json``.

    PYTHONPATH=scripts .venv/bin/python scripts/gap_diagnosis_controls.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

import gap_diagnosis as G
from gap_diagnosis_followup import cached_load

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/gap_diagnosis_controls.json"
Q = 16                                    # JW register: 2 * ncas


def main() -> None:
    D = cached_load()
    z = np.load(REPO / "results/gap_rho_pass.npz")
    X, gap = D["X"], D["gap"]
    y = (gap <= np.median(gap)).astype(int)
    Xd, Xo = X[:, :G.N_DIAG], X[:, G.N_DIAG:]
    out: dict = {}

    # ---- 1. coherence block vs. an equal number of noise features ---------
    rng = np.random.default_rng(5)
    a_d, _, _ = G.eval_classifier(Xd, y)
    a_q, _, _ = G.eval_classifier(X, y)
    noise_runs = []
    for k in range(5):
        Xn = np.column_stack([Xd, rng.normal(size=(len(y), Xo.shape[1]))])
        a_n, _, _ = G.eval_classifier(Xn, y, seeds=10)
        noise_runs.append(float(np.mean(a_n) * 100))
    out["noise_control"] = dict(
        diagonal_only=G.summ(a_d * 100),
        plus_coherence=G.summ(a_q * 100),
        plus_noise_mean=float(np.mean(noise_runs)),
        plus_noise_std=float(np.std(noise_runs, ddof=1)),
        delta_coherence=float(np.mean(a_q - a_d) * 100),
        delta_noise=float(np.mean(noise_runs) - np.mean(a_d) * 100))
    print(f"  diagonal only          {np.mean(a_d) * 100:5.2f}%")
    print(f"  + 112 coherence feats  {np.mean(a_q) * 100:5.2f}%  "
          f"({np.mean(a_q - a_d) * 100:+.2f})")
    print(f"  + 112 noise feats      {np.mean(noise_runs):5.2f}%  "
          f"({np.mean(noise_runs) - np.mean(a_d) * 100:+.2f})")

    # ---- 2. what fraction of rho's coherence does the pool see? -----------
    # rho = sum_P Tr(rho P) P / 2^Q  over a complete Pauli basis, and the basis
    # is orthogonal with ||P||_F^2 = 2^Q, so ||rho||_F^2 = sum_P Tr(rho P)^2/2^Q.
    seen_off = (Xo ** 2).sum(1) / 2 ** Q          # XX/YY strings are off-diagonal
    true_off = z["nrm_offdiag"] ** 2
    seen_diag = (Xd ** 2).sum(1) / 2 ** Q + 1.0 / 2 ** Q   # + the identity term
    true_diag = z["nrm_diag"] ** 2
    fo, fd = seen_off / true_off, seen_diag / true_diag
    out["pool_coverage"] = dict(
        offdiag_fraction_seen=dict(median=float(np.median(fo)),
                                   p10=float(np.percentile(fo, 10)),
                                   p90=float(np.percentile(fo, 90))),
        diag_fraction_seen=dict(median=float(np.median(fd)),
                                p10=float(np.percentile(fd, 10)),
                                p90=float(np.percentile(fd, 90))),
        n_independent_offdiag_features=int(Xo.shape[1] // 2))
    print(f"\n  pool sees {np.median(fd) * 100:6.2f}% of ||diag(rho)||^2 "
          f"and {np.median(fo) * 100:6.3f}% of ||offdiag(rho)||^2 (medians)")

    # ---- how much off-diagonal weight is there, and what drives it? -------
    share = true_off / (true_off + true_diag)
    dou = D["desc"].item()["DoU"] if hasattr(D["desc"], "item") else D["desc"]["DoU"]
    out["rho_offdiag_share"] = dict(
        median=float(np.median(share)), p10=float(np.percentile(share, 10)),
        p90=float(np.percentile(share, 90)), max=float(np.max(share)),
        spearman_vs_DoU=float(stats.spearmanr(share, dou).statistic),
        spearman_vs_gap=float(stats.spearmanr(share, gap).statistic))
    print(f"  rho off-diagonal share: median {np.median(share) * 100:.2f}%  "
          f"Spearman vs DoU {stats.spearmanr(share, dou).statistic:+.3f}  "
          f"vs gap {stats.spearmanr(share, gap).statistic:+.3f}")

    # ---- 3. does *global* coherence explain what the diagonal missed? -----
    # The pool is a narrow window, so test the label against a coherence
    # measure that is not limited by it: the exact off-diagonal Frobenius
    # share of rho, plus the two whole-state quantumness scalars.
    resid = gap - G.oof_predictions(Xd, gap)
    st = D["state"].item() if hasattr(D["state"], "item") else D["state"]
    probes = dict(rho_offdiag_share=share, static_corr=st["static_corr"],
                  c_max_sq=st["c_max_sq"], entropy=st["entropy"])
    out["global_coherence_vs_residual"] = {
        k: dict(pearson=float(stats.pearsonr(resid, v).statistic),
                spearman=float(stats.spearmanr(resid, v).statistic),
                pearson_with_gap=float(stats.pearsonr(gap, v).statistic))
        for k, v in probes.items()}
    for k, v in out["global_coherence_vs_residual"].items():
        print(f"  residual vs {k:20s} r = {v['pearson']:+.3f} "
              f"(vs the gap itself {v['pearson_with_gap']:+.3f})")

    a_aug, _, _ = G.eval_classifier(
        np.column_stack([Xd, share, st["static_corr"], st["c_max_sq"]]), y)
    out["diagonal_plus_global_coherence"] = dict(
        acc=G.summ(a_aug * 100),
        delta=float(np.mean(a_aug - a_d) * 100),
        p=float(stats.ttest_rel(a_aug, a_d).pvalue))
    print(f"  diagonal + global coherence scalars: {np.mean(a_aug) * 100:5.2f}% "
          f"({np.mean(a_aug - a_d) * 100:+.2f})")

    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
