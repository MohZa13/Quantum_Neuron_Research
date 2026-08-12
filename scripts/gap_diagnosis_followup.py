"""Follow-ups to ``gap_diagnosis.py``: is the fix a different label, or different
molecules?

Three arms.

A. **The correlated part of the gap.** The label under test so far is
   ``gap_Ha`` = eps_LUMO - eps_HOMO from ``eigh(F, S)`` -- a *one-body,
   mean-field* eigenvalue difference, defined before any correlation enters.
   The CASCI spectrum gives a genuinely correlated frontier gap
   ``E_1 - E_0`` in the same active space.  Their difference is the correlation
   correction: the part of the gap no one-body model can produce.  Run the same
   diagonal/off-diagonal ablation on it.

B. **Does the ablation improve where the states are more coherent?** Split the
   1000 molecules by off-diagonal coherence share and re-run the ablation in
   each half.  This is the in-sample version of "find harder molecules".

C. **How much headroom is there in harder molecules / hotter states?**  The
   28-molecule conjugated subset carries kT = 0.1 and kT = 0.25 blocks; measure
   the off-diagonal Frobenius share of rho against the 1000-molecule reference.

Writes ``results/gap_diagnosis_followup.json``.

    .venv/bin/python scripts/gap_diagnosis_followup.py
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

import gap_diagnosis as G

REPO = Path(__file__).resolve().parents[1]
CONJ = REPO / "results/qh9_conjugated_top45.h5"
CACHE = REPO / "results/gap_diagnosis_data.npz"
OUT = REPO / "results/gap_diagnosis_followup.json"


def cached_load() -> dict:
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        D = {k: z[k] for k in z.files}
        D["desc"] = D["desc"].item()
        D["state"] = D["state"].item()
        return D
    D = G.load()
    np.savez_compressed(CACHE, **{k: (np.array(v, dtype=object) if isinstance(v, dict)
                                      else v) for k, v in D.items()})
    return D


def ablation(X, Xd, y, tag, out, seeds=25):
    a_full, u_full, _ = G.eval_classifier(X, y, seeds)
    a_diag, u_diag, _ = G.eval_classifier(Xd, y, seeds)
    d = (a_full - a_diag) * 100
    out[tag] = dict(
        quantum=G.summ(a_full * 100), classical=G.summ(a_diag * 100),
        delta=G.summ(d), p=float(stats.ttest_rel(a_full, a_diag).pvalue),
        auc_quantum=G.summ(u_full), auc_classical=G.summ(u_diag), n=int(len(y)))
    print(f"  {tag:38s} Q {np.mean(a_full) * 100:5.1f}  C {np.mean(a_diag) * 100:5.1f}"
          f"  delta {np.mean(d):+5.2f}  (n={len(y)})")
    return out[tag]


def main() -> None:
    D = cached_load()
    X, gap = D["X"], D["gap"]
    Xd = X[:, :G.N_DIAG]
    st = D["state"]
    out: dict = {}

    # ---------------------------------------------------- A. correlated gap
    ci = st["ci_gap"]
    out["ci_vs_ks_gap"] = dict(
        pearson=float(stats.pearsonr(gap, ci).statistic),
        spearman=float(stats.spearmanr(gap, ci).statistic),
        ci_gap_median_Ha=float(np.median(ci)),
        ks_gap_median_Ha=float(np.median(gap)))
    print(f"CI gap vs KS gap: r = {stats.pearsonr(gap, ci).statistic:.3f}")

    # the part of the correlated gap the mean-field gap cannot predict
    A = np.column_stack([gap, np.ones_like(gap)])
    coef, *_ = np.linalg.lstsq(A, ci, rcond=None)
    corr_resid = ci - A @ coef
    out["correlation_correction"] = dict(
        r2_of_ks_gap_on_ci_gap=float(1 - np.var(corr_resid) / np.var(ci)),
        resid_std_Ha=float(np.std(corr_resid)),
        resid_std_eV=float(np.std(corr_resid) * G.HA_EV))

    print("\nablation, label = median split of ...")
    ablation(X, Xd, (gap <= np.median(gap)).astype(int),
             "KS HOMO-LUMO gap (the label under test)", out)
    ablation(X, Xd, (ci <= np.median(ci)).astype(int), "CASCI frontier gap E1-E0", out)
    ablation(X, Xd, (corr_resid <= np.median(corr_resid)).astype(int),
             "correlation correction (CI gap | KS gap)", out)

    # a coherence-only reference point: the synthetic construction of §4 --
    # label on a purely off-diagonal functional of rho, exactly representable
    rng = np.random.default_rng(11)
    Xo = X[:, G.N_DIAG:]
    w = rng.normal(size=Xo.shape[1])
    synth = Xo @ w
    ablation(X, Xd, (synth <= np.median(synth)).astype(int),
             "synthetic off-diagonal control", out)

    # ------------------------------------------- B. stratify by coherence
    # the density-matrix off-diagonal share, in the project's convention:
    # (Tr rho^2 - Tr Delta(rho)^2) / Tr rho^2 = ||offdiag(rho)||_F^2 / ||rho||_F^2.
    # NOTE this is *not* presentation_cache's `coh_share`, which is the same
    # ratio taken over the 248 Pauli features (three orders of magnitude
    # smaller -- RESEARCH_LOG 2026-08-06).
    cache = np.load(REPO / "results/presentation_cache.npz", allow_pickle=True)
    order = {int(i): k for k, i in enumerate(cache["idx"])}
    sel = np.array([order[i] for i in D["idx"]])
    coh = ((cache["purity"] - cache["dpurity"]) / cache["purity"])[sel]
    med = np.median(coh)
    print("\nablation on the KS gap label, split by coherence share")
    for tag, m in (("least coherent half", coh <= med), ("most coherent half", coh > med)):
        gm = gap[m]
        ablation(X[m], Xd[m], (gm <= np.median(gm)).astype(int),
                 f"KS gap, {tag}", out, seeds=25)
    out["coherence_split_median"] = float(med)

    # ------------------------------------------- C. headroom in harder states
    print("\ncoherence share, conjugated subset")
    conj = {}
    with h5py.File(CONJ, "r") as f:
        names = [k for k in f if k.startswith("mol_")]
        for kt in ("kT_0p1000", "kT_0p2500"):
            shares, n_states = [], []
            for nm in names:
                g = f[nm][kt]
                C = g["civecs"][:]
                p = g["p"][:]
                p = p / p.sum()
                rho = (C * p[:, None]).T @ C
                d = np.diag(rho)
                fro2 = float(np.sum(rho * rho))                 # = Tr rho^2
                shares.append((fro2 - float(np.sum(d ** 2))) / fro2)
                n_states.append(int(len(p)))
            conj[kt] = dict(n_molecules=len(names),
                            median=float(np.median(shares)),
                            p90=float(np.percentile(shares, 90)),
                            max=float(np.max(shares)),
                            median_kept_states=float(np.median(n_states)))
            print(f"  {kt}: median off-diag share {np.median(shares):.4f}  "
                  f"(n={len(names)}, median kept states {np.median(n_states):.0f})")
    out["conjugated_subset"] = conj
    out["reference_1000_coh_share_median"] = float(np.median(coh))

    OUT.write_text(json.dumps(out, indent=2))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
