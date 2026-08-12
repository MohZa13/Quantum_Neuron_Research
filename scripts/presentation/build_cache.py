"""Cache per-molecule summary quantities used by the presentation figures.

Reads the production run file, the Pauli-feature file, the coherence audit CSV
and the conjugation screen once, and writes a small npz so the plotting script
never has to touch the 45 GB HDF5 again.

    .venv/bin/python scripts/presentation/build_cache.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import h5py
import numpy as np

RUN = "results/qh9_dense_cas8-8_kT0p1.h5"
FEAT = "results/qh9_dense_cas8-8_kT0p1_extheis.h5"
COH = "results/coherence_share_kT0p1.csv"
SCREEN = "results/qh9_conjugation_screen_full.csv"
OUT = Path("results/presentation_cache.npz")

KT = "kT_0p1000"
NCAS, NQ = 8, 16


def mol_key(name: str) -> int:
    return int(name.split("_")[1])


def main() -> None:
    t0 = time.time()

    # --- eigenblock summaries -------------------------------------------------
    idx, rank, p0, entropy, gap, tdist, statcorr, cmax, trunc = ([] for _ in range(9))
    purity, dpurity = [], []        # Tr(rho^2) and Tr(Delta(rho)^2)
    cumw = []                       # cumulative weight curves, padded
    with h5py.File(RUN, "r") as f:
        names = sorted((k for k in f if k.startswith("mol_")), key=mol_key)
        for n, name in enumerate(names):
            b = f[name][KT]
            p = b["p"][:]
            E = b["E"][:]
            # Off-diagonal (coherence) content of rho, without ever forming rho:
            #   Tr(rho^2)          = sum_k p_k^2                  (eigenvalues)
            #   Tr(Delta(rho)^2)   = sum_I d_I^2,  d_I = sum_k p_k V_kI^2
            # so ||offdiag(rho)||_F^2 = Tr(rho^2) - Tr(Delta(rho)^2).
            d = (p[:, None] * b["civecs"][:] ** 2).sum(0)
            purity.append(float((p ** 2).sum()))
            dpurity.append(float((d ** 2).sum()))
            idx.append(mol_key(name))
            rank.append(len(p))
            p0.append(float(p[0]))
            entropy.append(float(b["entropy"][()]))
            gap.append(float(E[1] - E[0]) if len(E) > 1 else np.nan)
            tdist.append(float(b["tracedist_gaussian"][()]))
            statcorr.append(float(b["static_corr"][()]))
            cmax.append(float(b["c_max_sq"][()]))
            trunc.append(float(b["truncation_error"][()]))
            if n % 25 == 0:                       # 40 representative curves
                c = np.cumsum(p) / p.sum()
                cumw.append(c)
            if n % 200 == 0:
                print(f"  {n:4d}/{len(names)}  {time.time()-t0:.0f}s", flush=True)

    idx = np.array(idx)
    L = max(len(c) for c in cumw)
    cum = np.ones((len(cumw), L))
    for i, c in enumerate(cumw):
        cum[i, : len(c)] = c

    # --- Pauli features -------------------------------------------------------
    with h5py.File(FEAT, "r") as f:
        names = sorted((k for k in f if k.startswith("mol_")), key=mol_key)
        fidx = np.array([mol_key(n) for n in names])
        X = np.array([f[n][KT]["coeffs"][:] for n in names])
    assert np.array_equal(fidx, idx), "feature/run molecule order differs"

    # blocked layout: 16 Z, then C(16,2)=120 ZZ (i<j), then 112 XX/YY
    nz, nzz = NQ, NQ * (NQ - 1) // 2
    z, zz, xy = X[:, :nz], X[:, nz:nz + nzz], X[:, nz + nzz:]
    iu, ju = np.triu_indices(NQ, k=1)
    prod = z[:, iu] * z[:, ju]                     # factorized <Z_i><Z_j>
    conn = zz - prod                               # connected covariance
    w_single = (z ** 2).sum(1) + (prod ** 2).sum(1)
    w_conn = (conn ** 2).sum(1)
    w_xy = (xy ** 2).sum(1)

    # --- classical descriptors ------------------------------------------------
    coh = {int(r["idx"]): r for r in csv.DictReader(open(COH))}
    scr = {int(r["idx"]): r for r in csv.DictReader(open(SCREEN))}
    def get(d, i, k):
        v = d.get(i, {}).get(k, "")
        return float(v) if v not in ("", None) else np.nan

    coh_share = np.array([get(coh, i, "coh_share") for i in idx])
    coh_max = np.array([get(coh, i, "coh_max") for i in idx])
    dou = np.array([get(coh, i, "DoU") for i in idx])
    pi_atoms = np.array([get(coh, i, "largest_pi_atoms") for i in idx])
    n_heavy = np.array([get(scr, i, "n_heavy") for i in idx])
    formula = np.array([coh[i]["formula"] for i in idx])

    np.savez_compressed(
        OUT, idx=idx, rank=np.array(rank), p0=np.array(p0),
        entropy=np.array(entropy), gap=np.array(gap), tdist=np.array(tdist),
        static_corr=np.array(statcorr), c_max_sq=np.array(cmax),
        trunc=np.array(trunc), cum=cum,
        purity=np.array(purity), dpurity=np.array(dpurity),
        w_single=w_single, w_conn=w_conn, w_xy=w_xy,
        coh_share=coh_share, coh_max=coh_max, dou=dou,
        pi_atoms=pi_atoms, n_heavy=n_heavy, formula=formula,
    )
    tot = w_single + w_conn + w_xy
    print(f"\nwrote {OUT}  ({time.time()-t0:.0f}s)")
    print(f"  thermal rank m: min {min(rank)} median {int(np.median(rank))} max {max(rank)}")
    print(f"  p0: {p0 and min(p0):.3f}-{max(p0):.3f} median {np.median(p0):.3f}")
    print("  feature weight shares (mean over molecules):")
    print(f"    single-mode occupation   {100*np.mean(w_single/tot):.3f}%")
    print(f"    connected covariance     {100*np.mean(w_conn/tot):.3f}%")
    print(f"    hopping coherence        {100*np.mean(w_xy/tot):.4f}%")
    print(f"  Pauli-feature coherence share median {100*np.median(coh_share):.4f}%")
    off = (np.array(purity) - np.array(dpurity)) / np.array(purity)
    print("  density-matrix off-diagonal Frobenius share  "
          f"median {100*np.median(off):.2f}%  "
          f"p10 {100*np.percentile(off,10):.2f}%  p90 {100*np.percentile(off,90):.2f}%")
    ok = np.isfinite(dou)
    from scipy.stats import spearmanr
    print(f"  Spearman(offdiag share, DoU) = {spearmanr(off[ok], dou[ok]).statistic:.3f}")
    print(f"  Spearman(offdiag share, largest pi) = "
          f"{spearmanr(off[ok & np.isfinite(pi_atoms)], pi_atoms[ok & np.isfinite(pi_atoms)]).statistic:.3f}")


if __name__ == "__main__":
    main()
