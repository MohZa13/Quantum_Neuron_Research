"""One streaming pass over the 1000-molecule run file, for the gap-label audit.

Reconstructs rho_m = V^T diag(p) V per molecule in the CI determinant basis and
accumulates the three things the audit needs that cannot be got from the
248-term Pauli export:

1. ``R_plus`` / ``R_minus`` -- the class-aggregated density matrices for the
   median-split HOMO-LUMO gap label.  Their difference feeds the project's R+/R-
   screening metric ``||offdiag(dR)||_F / ||diag(dR)||_F`` (QUANTUM_NEURON.md
   §7), so the gap label gets the *same* number already measured for <S^2>
   (0.122) and c (0.162).
2. ``diag_rho`` -- the full 4900-entry determinant populations per molecule.
   This is exactly what a dephased (classical) model may read, and it is far
   more than the 136 Z/ZZ Pauli features expose, so a classical model trained on
   it is the honest classical ceiling rather than a strawman.
3. Per-molecule diagonal / off-diagonal Frobenius norms of rho.

Output: ``results/gap_rho_pass.npz``.  Cost: reads ~45 GB, ~10-20 min.

    .venv/bin/python scripts/gap_rho_pass.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "results/qh9_dense_cas8-8_kT0p1.h5"
SCREEN = REPO / "results/qh9_conjugation_screen_full.csv"
OUT = REPO / "results/gap_rho_pass.npz"
KT = "kT_0p1000"


def main() -> None:
    gaps = {int(r["idx"]): float(r["gap_Ha"])
            for r in csv.DictReader(open(SCREEN))}

    with h5py.File(RUN, "r") as f:
        names = sorted((k for k in f if k.startswith("mol_")),
                       key=lambda s: int(s.split("_")[1]))
        idxs = np.array([int(n.split("_")[1]) for n in names])
        g = np.array([gaps[i] for i in idxs])
        y = (g <= np.median(g)).astype(int)          # 1 = small gap
        dim = f[names[0]][KT]["civecs"].shape[1]
        print(f"{len(names)} molecules, dim {dim}, "
              f"small-gap {y.sum()} large-gap {(1 - y).sum()}", flush=True)

        R = [np.zeros((dim, dim)), np.zeros((dim, dim))]   # R[0] = large gap
        diag_rho = np.zeros((len(names), dim), dtype=np.float32)
        nrm_d = np.zeros(len(names))
        nrm_o = np.zeros(len(names))
        t0 = time.time()

        for m, name in enumerate(names):
            grp = f[name][KT]
            C = grp["civecs"][:]                     # (n_kept, dim), real
            p = grp["p"][:]
            p = p / p.sum()                          # INVARIANTS I8: unit trace
            rho = (C * p[:, None]).T @ C
            R[y[m]] += rho
            d = np.diag(rho)
            diag_rho[m] = d.astype(np.float32)
            nrm_d[m] = np.linalg.norm(d)
            nrm_o[m] = np.sqrt(max(np.sum(rho * rho) - nrm_d[m] ** 2, 0.0))
            if (m + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {m + 1}/{len(names)}  {el:6.1f}s  "
                      f"eta {el / (m + 1) * (len(names) - m - 1):6.1f}s",
                      flush=True)

    dR = R[1] - R[0]
    dd = np.diag(dR).copy()
    off = np.sqrt(max(np.sum(dR * dR) - np.sum(dd ** 2), 0.0))
    ratio = off / np.linalg.norm(dd)
    print(f"\nR+/R- screen (gap label): ||offdiag||/||diag|| = {ratio:.4f}")

    np.savez_compressed(
        OUT, idx=idxs, gap=g, y=y, diag_rho=diag_rho,
        nrm_diag=nrm_d, nrm_offdiag=nrm_o,
        dR_diag=dd, dR_offdiag_fro=np.array(off),
        dR_diag_fro=np.array(np.linalg.norm(dd)), screen_ratio=np.array(ratio),
        R_plus_diag=np.diag(R[1]).copy(), R_minus_diag=np.diag(R[0]).copy(),
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
