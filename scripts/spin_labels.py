"""Singlet / triplet-open-shell labels for qthermal thermal states.

For every molecule in a Module-H run file this computes the thermal expectation
of the total-spin operator and, crucially, splits it into the part a classical
model can see and the part only coherence carries.

On the fixed (nelecas, S_z = 0) sector, S_z is identically zero on every
determinant, so S^2 = S_-S_+ there, and that operator splits cleanly in the
determinant basis:

    S^2  =  D  +  S2_od

    D       diagonal:  sum_p n_{p,beta} (1 - n_{p,alpha}), i.e. the number of
            singly-occupied orbitals holding a beta electron -- half the
            open-shell count.  "HOW MANY unpaired electrons are there."
            Readable from diag(rho) alone: a purely classical quantity.

    S2_od   off-diagonal:  spin exchange between two orbitals p != q.
            "HOW are those unpaired spins COUPLED."  The singlet and the M_s=0
            triplet built on the same two determinants have *identical*
            diagonals and opposite-sign off-diagonals, so this distinction is
            100% coherence, by construction rather than by empirical luck.

Hence three labels per molecule, all thermal expectations over
rho = sum_k p_k |psi_k><psi_k|:

    S2 = Tr(rho S^2)      total open-shell / triplet character (mixed)
    D  = Tr(rho D)        the classical, diagonal-readable part
    c  = Tr(rho S2_od)    the coherence-only part;  c = S2 - D

Per open-shell pair c is 0 for a closed shell, -1 for singlet-coupled and +1
for triplet-coupled spins.

The sector S^2 matrix is molecule-independent, so it is built once (from
PySCF's ``contract_ss`` applied to unit vectors) and cached.  Everything after
that is a sparse apply against the stored eigenblock -- no truncation, no
per-root loop, and no dim x dim density matrix.

The script also projects each thermal state onto the ``--n-qubits`` most
populated determinants (aggregate over a sample) and writes the resulting
dense rho stack, which is what the classifier comparison consumes.

Usage:
    python -m scripts.spin_labels --in results/qh9_dense_cas8-8_kT0p1.h5 \
        --out results/spin_labels_kT0p1.npz --rho-out /path/rho.npy --n-qubits 10
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse
from pyscf import fci

logger = logging.getLogger("spin_labels")

_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def hill_formula(Z: np.ndarray) -> str:
    """Hill-order formula (C, H, then alphabetical)."""
    counts: dict[str, int] = {}
    for z in Z:
        s = _SYMBOL.get(int(z), f"Z{int(z)}")
        counts[s] = counts.get(s, 0) + 1

    def piece(sym: str) -> str:
        n = counts.pop(sym, 0)
        return "" if n == 0 else f"{sym}{n if n > 1 else ''}"

    out = piece("C") + piece("H")
    for sym in sorted(counts):
        out += f"{sym}{counts[sym] if counts[sym] > 1 else ''}"
    return out


# --- the sector S^2 operator -------------------------------------------------

def build_s2_sector(ncas: int, na: int, nb: int) -> scipy.sparse.csr_matrix:
    """Sparse (dim, dim) S^2 in the PySCF determinant basis of the sector.

    Column j is ``contract_ss`` applied to determinant j.  S^2 is Hermitian and
    real here, so the result is symmetric; the assert is the check that the
    (na, nb) ravel convention lines up with the rest of the pipeline.
    """
    nstr_a = fci.cistring.num_strings(ncas, na)
    nstr_b = fci.cistring.num_strings(ncas, nb)
    dim = nstr_a * nstr_b
    cols, rows, vals = [], [], []
    unit = np.zeros((nstr_a, nstr_b))
    for j in range(dim):
        ia, ib = divmod(j, nstr_b)
        unit[ia, ib] = 1.0
        col = np.asarray(fci.spin_op.contract_ss(unit, ncas, (na, nb))).ravel()
        unit[ia, ib] = 0.0
        nz = np.nonzero(np.abs(col) > 1e-12)[0]
        rows.append(nz)
        cols.append(np.full(len(nz), j))
        vals.append(col[nz])
    S2 = scipy.sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim))
    asym = abs(S2 - S2.T).max()
    assert asym < 1e-10, f"sector S^2 not symmetric: {asym:.2e}"
    return S2


def diagonal_open_shell_count(ncas: int, na: int, nb: int) -> np.ndarray:
    """D_I = #orbitals with beta occupied and alpha empty, per determinant.

    This is the diagonal of S^2 on an S_z = 0 sector, derived independently of
    ``contract_ss`` so the two can be cross-checked.
    """
    stra = np.asarray(fci.cistring.make_strings(range(ncas), na), dtype=np.int64)
    strb = np.asarray(fci.cistring.make_strings(range(ncas), nb), dtype=np.int64)
    occ_a = (stra[:, None] >> np.arange(ncas)) & 1        # (na_str, ncas)
    occ_b = (strb[:, None] >> np.arange(ncas)) & 1
    # beta occupied AND alpha empty, summed over orbitals, for every (ia, ib)
    D = (occ_b[None, :, :] * (1 - occ_a)[:, None, :]).sum(-1)
    return D.ravel().astype(np.float64)


# --- driver ------------------------------------------------------------------

def select_subspace(f, names, kT_tag, dim, n_keep, n_sample):
    """Indices of the `n_keep` most populated determinants, aggregated over a
    spread-out sample of molecules (they all share one determinant labelling)."""
    step = max(1, len(names) // n_sample)
    W = np.zeros(dim)
    for nm in names[::step][:n_sample]:
        blk = f[nm][kT_tag]
        p = blk["p"][:]
        W += (p[:, None] * blk["civecs"][:] ** 2).sum(0) / p.sum()
    order = np.argsort(W)[::-1]
    return np.sort(order[:n_keep]), W


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="Module-H run file")
    ap.add_argument("--out", required=True, help="output NPZ of labels")
    ap.add_argument("--rho-out", default=None,
                    help="optional .npy for the projected (N, K, K) float32 "
                         "density-matrix stack")
    ap.add_argument("--kT-tag", default=None,
                    help="kT group to read (default: the sole group)")
    ap.add_argument("--n-qubits", type=int, default=10,
                    help="project onto the 2**n most populated determinants")
    ap.add_argument("--sample", type=int, default=80,
                    help="molecules used to choose the subspace")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(message)s")

    t_run = time.perf_counter()
    with h5py.File(args.src, "r") as f:
        meta = dict(f["meta"].attrs)
        ncas = int(meta["ncas"])
        na = nb = int(meta["nelecas"]) // 2
        names = sorted((k for k in f if k.startswith("mol_")),
                       key=lambda s: int(s.split("_")[1]))
        names = [n for n in names if f[n].attrs.get("complete", False)]
        if args.limit:
            names = names[:args.limit]
        tag = args.kT_tag or sorted(k for k in f[names[0]] if k.startswith("kT_"))[0]
        dim = fci.cistring.num_strings(ncas, na) ** 2
        logger.info("%d molecules, ncas=%d, dim=%d, block %s", len(names), ncas, dim, tag)

        # --- the molecule-independent operator, built once ---
        t0 = time.perf_counter()
        S2 = build_s2_sector(ncas, na, nb)
        Ddiag = diagonal_open_shell_count(ncas, na, nb)
        dev = float(np.abs(S2.diagonal() - Ddiag).max())
        assert dev < 1e-10, f"diag(S^2) != open-shell count: {dev:.2e}"
        logger.info("sector S^2 built in %.1fs: %d nonzeros (%.2f%% dense); "
                    "diag matches independent open-shell count to %.1e",
                    time.perf_counter() - t0, S2.nnz, 100 * S2.nnz / dim**2, dev)

        # --- common determinant subspace ---
        K = 1 << args.n_qubits
        keep, W = select_subspace(f, names, tag, dim, K, args.sample)
        logger.info("subspace: top %d determinants (%d qubits), aggregate "
                    "population coverage %.3f%%", K, args.n_qubits,
                    100 * W[keep].sum() / W.sum())

        rho_stack = (np.lib.format.open_memmap(args.rho_out, mode="w+",
                                               dtype=np.float32, shape=(len(names), K, K))
                     if args.rho_out else None)

        rows = []
        for i, nm in enumerate(names):
            blk = f[nm][tag]
            p = blk["p"][:]
            V = blk["civecs"][:]                      # (m, dim)
            trace = float(p.sum())
            # <S^2> = sum_k p_k v_k . (S2 v_k) -- one sparse apply for the block
            S2V = S2.dot(V.T).T                       # (m, dim)
            s2 = float((p * np.einsum("kd,kd->k", V, S2V)).sum() / trace)
            d = float((p * ((V ** 2) @ Ddiag)).sum() / trace)
            if rho_stack is not None:
                A = np.sqrt(p / trace)[:, None] * V[:, keep]      # (m, K)
                rho = A.T @ A
                rho_stack[i] = rho.astype(np.float32)
                kept = float(np.trace(rho))
            else:
                kept = np.nan
            rows.append((int(nm.split("_")[1]), hill_formula(f[nm]["Z"][:]),
                         s2, d, s2 - d, trace, kept, len(p)))
            if (i + 1) % 50 == 0:
                logger.info("  %d/%d  (%.1f s)", i + 1, len(names),
                            time.perf_counter() - t_run)

    idx = np.array([r[0] for r in rows], np.int64)
    out = dict(
        idx=idx,
        formula=np.array([r[1] for r in rows], "S"),
        S2=np.array([r[2] for r in rows]),
        D=np.array([r[3] for r in rows]),
        c=np.array([r[4] for r in rows]),
        trace=np.array([r[5] for r in rows]),
        rho_trace_kept=np.array([r[6] for r in rows]),
        n_states=np.array([r[7] for r in rows], np.int64),
        keep_idx=keep, n_qubits=np.int64(args.n_qubits),
        source=np.str_(args.src), kT_tag=np.str_(tag),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)

    s2, d, c = out["S2"], out["D"], out["c"]
    logger.info("wrote %s", args.out)
    logger.info("<S^2>  min %.4f  median %.4f  max %.4f", s2.min(), np.median(s2), s2.max())
    logger.info("<D>    min %.4f  median %.4f  max %.4f", d.min(), np.median(d), d.max())
    logger.info("c      min %.4f  median %.4f  max %.4f  (frac negative %.1f%%)",
                c.min(), np.median(c), c.max(), 100 * (c < 0).mean())
    logger.info("corr(S2, D) = %.4f   corr(S2, c) = %.4f",
                np.corrcoef(s2, d)[0, 1], np.corrcoef(s2, c)[0, 1])
    if rho_stack is not None:
        kept = out["rho_trace_kept"]
        logger.info("projected rho trace: min %.4f  median %.4f", kept.min(), np.median(kept))
    logger.info("total %.1f s", time.perf_counter() - t_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
