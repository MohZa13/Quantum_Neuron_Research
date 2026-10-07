"""Read the full-chain source-sign MPS states and cut regions out of them.

Needs numpy and h5py, nothing else.  This file is copied next to the data, so a
delivered directory is self-contained; README.md there explains the physics
and the conventions, and this module is the executable half of it.

    import source_sign_mps_reader as smr
    states = smr.manifest("source_sign_mps")          # one row per stored state
    psi = smr.load("source_sign_mps/XXX/XXX_s+1_t06.400.h5")
    psi.hamiltonian, psi.s, psi.t                      # the three labels
    rho = smr.region_rho(psi, k=5, d=2)                # 5 qubits centred at j0 + 2
    rho = smr.reduced_density_matrix(psi, [24, 25, 26])  # any 1-based sites
    ts = smr.training_set("source_sign_mps", "XXX", k=3, ds=range(-5, 6))  # (s, rho) pairs

CONVENTIONS -- the ones that fail silently if you get them wrong:

  * Qubit j of the chain is site j, 1-based, j = 1..N.  The kicked qubit is
    j0 = 25.  A region of k qubits at offset d is the k sites centred on
    j0 + d: first site j0 + d - (k - 1)/2 (k odd, the paper's convention).
  * Each site tensor is A[j] with numpy shape (chi_{j-1}, 2, chi_j); the
    physical index is 0 = spin up (sigma^z = +1), 1 = spin down, and
        psi(b_1, ..., b_N) = A[1][:, b_1, :] @ A[2][:, b_2, :] @ ... @ A[N][:, b_N, :].
  * A region's density matrix is LITTLE-ENDIAN over its qubits in increasing
    site order: qubit q of the region (site sites[q]) is bit q of the row
    index, r = sum_q b_q 2^q, and rho[r, r'] = <r|rho|r'> (ket row, bra
    column).  This is qnn/pools.py's convention and the one used by the
    reduced-state datasets (docs/SOURCE_SIGN_DATASETS.md).  Pass
    ``little_endian=False`` for the textbook kron order instead.
  * The states are complex and must stay complex.  At t = 0 the two labels
    differ ONLY in the imaginary part of every region's rho (rho_- = conj
    rho_+), so a transposed rho -- which is conj(rho) -- silently swaps the
    label.  ``check`` compares this module's contraction against the rho the
    generator computed with ITensor, which is what catches that.

The contraction is gauge-independent (both environments are computed, never
assumed to be identities), so it is correct for any MPS, not only for the
right-canonical form these files are written in.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

__all__ = ["MPSState", "load", "manifest", "region_sites", "region_rho", "training_set",
           "reduced_density_matrix", "all_regions", "expect_sz", "overlap",
           "spin_flip", "norm", "check", "flip_register"]


@dataclass
class MPSState:
    """One stored state: site tensors plus its labels and run metadata."""

    tensors: list                      # A[0..N-1], numpy (chi_l, 2, chi_r) complex128
    attrs: dict = field(default_factory=dict)
    path: str = ""

    @property
    def n(self) -> int:
        return len(self.tensors)

    @property
    def hamiltonian(self) -> str:      # "XX" or "XXX"
        return self.attrs["hamiltonian"]

    @property
    def s(self) -> int:                # +1 or -1, the rotation direction (the label)
        return int(self.attrs["s"])

    @property
    def t(self) -> float:              # cumulative evolution time, units of 1/J
        return float(self.attrs["t"])

    @property
    def j0(self) -> int:               # 1-based kicked qubit
        return int(self.attrs["j0"])

    @property
    def bond_dims(self) -> np.ndarray:
        return np.array([1] + [A.shape[2] for A in self.tensors])


def _attr(v):
    if isinstance(v, bytes):
        return v.decode()
    return v.item() if hasattr(v, "item") else v


def load(path) -> MPSState:
    """Load one state file."""
    with h5py.File(path, "r") as f:
        g = f["mps"]
        n = len(g)
        tensors = [g[f"site_{j:02d}"][()] for j in range(1, n + 1)]
        attrs = {k: _attr(v) for k, v in f.attrs.items()}
    for j in range(n - 1):
        if tensors[j].shape[2] != tensors[j + 1].shape[0]:
            raise ValueError(f"{path}: bond {j + 1} does not match "
                             f"({tensors[j].shape} vs {tensors[j + 1].shape})")
    return MPSState(tensors, attrs, str(path))


def manifest(root) -> list[dict]:
    """One dict per state under ``root``: labels, file, and run diagnostics.

    Reads ``root/manifest.csv`` when present, else scans the files.
    """
    root = Path(root)
    csv_path = root / "manifest.csv"
    if csv_path.exists():
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            r["s"], r["step"] = int(r["s"]), int(r["step"])
            for key in ("t", "energy", "norm_loss_cumulative"):
                r[key] = float(r[key])
            r["maxlinkdim"] = int(r["maxlinkdim"])
        return rows
    return scan(root)


_MANIFEST_KEYS = ("hamiltonian", "s", "t", "step", "file", "maxlinkdim", "energy",
                  "norm_loss_cumulative")


def scan(root) -> list[dict]:
    """Build the manifest rows from the files themselves (attrs only, fast)."""
    root = Path(root)
    rows = []
    for p in sorted(root.glob("*/*_s[+-]1_t*.h5")):
        with h5py.File(p, "r") as f:
            a = {k: _attr(v) for k, v in f.attrs.items()}
        rows.append({"hamiltonian": a["hamiltonian"], "s": int(a["s"]), "t": float(a["t"]),
                     "step": int(a["step"]), "file": str(p.relative_to(root)),
                     "maxlinkdim": int(a["maxlinkdim"]), "energy": float(a["energy"]),
                     "norm_loss_cumulative": float(a["norm_loss_cumulative"])})
    rows.sort(key=lambda r: (r["hamiltonian"], -r["s"], r["step"]))
    return rows


def write_manifest(root) -> Path:
    rows = scan(root)
    path = Path(root) / "manifest.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_MANIFEST_KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "t": f"{r['t']:.3f}", "energy": f"{r['energy']:.12f}",
                        "norm_loss_cumulative": f"{r['norm_loss_cumulative']:.3e}"})
    return path


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------

def region_sites(k: int, d: int, j0: int = 25, n: int = 50) -> list[int]:
    """1-based sites of the k-qubit region centred at j0 + d (k odd)."""
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be odd: the region is centred on j0 + d")
    first = j0 + d - (k - 1) // 2
    sites = list(range(first, first + k))
    if sites[0] < 1 or sites[-1] > n:
        raise ValueError(f"region k={k}, d={d} runs off the {n}-site chain: {sites}")
    return sites


def _left_env(psi: MPSState, upto: int) -> np.ndarray:
    """L over sites 1..upto-1 (1-based), L[a, a'] with a the ket bond."""
    L = np.ones((1, 1), dtype=complex)
    for A in psi.tensors[: upto - 1]:
        L = np.einsum("ab,asc,bsd->cd", L, A, A.conj(), optimize=True)
    return L


def _right_envs(psi: MPSState) -> list:
    """R[j] over sites j+1..N (1-based j, j = 0..N), R[a, a'] with a the ket bond."""
    n = psi.n
    R = [None] * (n + 1)
    R[n] = np.ones((1, 1), dtype=complex)
    for j in range(n, 0, -1):
        A = psi.tensors[j - 1]
        R[j - 1] = np.einsum("asc,bsd,cd->ab", A, A.conj(), R[j], optimize=True)
    return R


def _rho_from_block(L, T, R, k, little_endian) -> np.ndarray:
    """rho from a contracted block T (chi_l, 2^k physical, chi_r), first site first."""
    chil, chir = T.shape[0], T.shape[-1]
    K = 2 ** k
    T = T.reshape(chil, K, chir)             # C order: first site most significant
    # X[l, b, r] = sum_{l', r'} L[l, l'] conj(T)[l', b, r'] R[r, r'], every step a GEMM
    # (np.einsum on two operands without `optimize` runs a C loop that never calls BLAS).
    X = L @ T.conj().reshape(chil, K * chir)
    if not (R.shape == (1, 1) or np.abs(R - np.eye(chir)).max() < 1e-12):
        X = X.reshape(chil * K, chir) @ R.T  # skipped when right-orthonormal (the files are)
    rho = np.tensordot(T, X.reshape(chil, K, chir), axes=([0, 2], [0, 2]))
    rho = rho / np.trace(rho).real
    if little_endian:                        # reverse the bit order: site q -> bit q
        perm = _bit_reverse(k)
        rho = rho[np.ix_(perm, perm)]
    return 0.5 * (rho + rho.conj().T)


def _bit_reverse(k: int) -> np.ndarray:
    idx = np.arange(2 ** k)
    out = np.zeros_like(idx)
    for q in range(k):
        out |= ((idx >> q) & 1) << (k - 1 - q)
    return out


def _contract_block(psi: MPSState, a: int, b: int) -> np.ndarray:
    """T = A[a] A[a+1] ... A[b] with open physical legs, shape (chi, 2, ..., 2, chi)."""
    T = psi.tensors[a - 1]
    for j in range(a + 1, b + 1):
        T = np.tensordot(T, psi.tensors[j - 1], axes=([T.ndim - 1], [0]))
    return T


def reduced_density_matrix(psi: MPSState, sites, little_endian: bool = True) -> np.ndarray:
    """rho on the given 1-based sites (any set), unit trace, 2^k x 2^k.

    Contiguous sites are contracted directly.  For a non-contiguous set the
    contiguous span is contracted and the gaps are traced out afterwards, so
    the cost grows as 4^(span); spans up to ~10 sites are practical.
    """
    sites = sorted(int(j) for j in sites)
    if len(set(sites)) != len(sites) or sites[0] < 1 or sites[-1] > psi.n:
        raise ValueError(f"bad site list {sites} for an {psi.n}-site chain")
    a, b = sites[0], sites[-1]
    span = b - a + 1
    L = _left_env(psi, a)
    R = np.ones((1, 1), dtype=complex)
    for A in reversed(psi.tensors[b:]):
        R = np.einsum("asc,bsd,cd->ab", A, A.conj(), R, optimize=True)
    T = _contract_block(psi, a, b)
    rho = _rho_from_block(L, T, R, span, little_endian=False)   # big-endian over the span
    if span != len(sites):
        keep = [j - a for j in sites]
        rho = rho.reshape([2] * (2 * span))
        drop = [q for q in range(span) if q not in keep]
        for q in sorted(drop, reverse=True):
            rho = np.trace(rho, axis1=q, axis2=q + rho.ndim // 2)
        k = len(sites)
        rho = rho.reshape(2 ** k, 2 ** k)
    else:
        k = span
    if little_endian:
        perm = _bit_reverse(k)
        rho = rho[np.ix_(perm, perm)]
    return rho


def region_rho(psi: MPSState, k: int, d: int, little_endian: bool = True) -> np.ndarray:
    """rho of the k-qubit region centred at j0 + d (the paper's (k, d))."""
    return reduced_density_matrix(psi, region_sites(k, d, psi.j0, psi.n), little_endian)


def all_regions(psi: MPSState, k: int, little_endian: bool = True, ds=None):
    """Yield (d, sites, rho) for every k-qubit region of the chain, in one sweep.

    Shares the environments across regions: O(N) contractions in total rather
    than O(N) per region.  ``ds`` restricts the output to those offsets (each
    must fit on the chain); they come out in increasing d either way.
    """
    off = (k - 1) // 2
    first = range(1, psi.n - k + 2)
    if ds is not None:
        want = {int(d) for d in ds}
        for d in want:
            region_sites(k, d, psi.j0, psi.n)            # raises if off the chain
        first = [a for a in first if a + off - psi.j0 in want]
    if not len(first):
        return
    R = _right_envs(psi)
    L = np.ones((1, 1), dtype=complex)
    a_done = 1                                           # L covers sites 1 .. a_done - 1
    for a in first:
        for A in psi.tensors[a_done - 1:a - 1]:
            L = np.einsum("ab,asc,bsd->cd", L, A, A.conj(), optimize=True)
        a_done = a
        b = a + k - 1
        T = _contract_block(psi, a, b)
        yield a + off - psi.j0, list(range(a, b + 1)), _rho_from_block(L, T, R[b], k, little_endian)


def training_set(root, hamiltonian: str, k: int, ds, times=None,
                 little_endian: bool = True) -> dict:
    """The whiteboard's training set {(s_i, rho_{t_i, R_i}(s_i))} for one
    Hamiltonian and region size: every stored state (both labels) at every
    requested time, cut at every offset in ``ds``.

    ``times=None`` takes every stored step; otherwise only states whose t is
    within 1e-9 of a listed time.  Returns arrays rho (M, 2^k, 2^k), s, t, d,
    and file, with both labels of one (t, d) cell adjacent -- keep them on the
    same side of any train/test split.
    """
    root = Path(root)
    rows = [r for r in manifest(root) if r["hamiltonian"] == hamiltonian]
    if times is not None:
        want = np.asarray(times, dtype=float)
        rows = [r for r in rows if np.abs(want - r["t"]).min() < 1e-9]
    rows.sort(key=lambda r: (r["step"], -r["s"]))
    out = {"rho": [], "s": [], "t": [], "d": [], "file": []}
    for r in rows:
        for d, _, rho in all_regions(load(root / r["file"]), k, little_endian, ds=ds):
            out["rho"].append(rho)
            out["s"].append(r["s"])
            out["t"].append(r["t"])
            out["d"].append(d)
            out["file"].append(r["file"])
    order = np.lexsort((-np.array(out["s"]), np.array(out["d"]), np.array(out["t"])))
    return {"rho": np.stack(out["rho"])[order], "s": np.array(out["s"], dtype=np.int8)[order],
            "t": np.array(out["t"])[order], "d": np.array(out["d"], dtype=np.int16)[order],
            "file": np.array(out["file"])[order]}


# ---------------------------------------------------------------------------
# whole-chain quantities and checks
# ---------------------------------------------------------------------------

def overlap(phi: MPSState, psi: MPSState) -> complex:
    """<phi|psi>."""
    E = np.ones((1, 1), dtype=complex)
    for A, B in zip(phi.tensors, psi.tensors):
        E = np.einsum("ab,asc,bsd->cd", E, A.conj(), B, optimize=True)
    return complex(E[0, 0])


def norm(psi: MPSState) -> float:
    return float(np.sqrt(overlap(psi, psi).real))


def spin_flip(psi: MPSState) -> MPSState:
    """F|psi> with F = prod_j sigma^x_j: swap the physical index on every site."""
    return MPSState([A[:, ::-1, :].copy() for A in psi.tensors], dict(psi.attrs), psi.path)


def expect_sz(psi: MPSState) -> np.ndarray:
    """<sigma^z_j> for j = 1..N."""
    R = _right_envs(psi)
    L = np.ones((1, 1), dtype=complex)
    z = np.array([1.0, -1.0])
    out = np.empty(psi.n)
    nrm = overlap(psi, psi).real
    for j, A in enumerate(psi.tensors, start=1):
        out[j - 1] = np.einsum("ab,asc,s,bsd,cd->", L, A, z, A.conj(), R[j], optimize=True).real
        L = np.einsum("ab,asc,bsd->cd", L, A, A.conj(), optimize=True)
    return out / nrm


def flip_register(rho: np.ndarray) -> np.ndarray:
    """X on every qubit of a region: rho -> X^(x)k rho X^(x)k (an index permutation)."""
    K = rho.shape[-1]
    perm = np.arange(K) ^ (K - 1)
    return rho[..., perm, :][..., :, perm]


def check(psi: MPSState, tol: float = 1e-8) -> dict:
    """Verify one state against the numbers its generator stored beside it.

    norm, <sigma^z_j> for every j, and the rho on sites j0-1..j0+1 -- the last
    two computed by ITensor in the generator, so agreement checks the tensor
    layout, the site order, the physical-index order and the ket/bra order of
    this reader in one go.  Raises on failure; returns the deviations.
    """
    with h5py.File(psi.path, "r") as f:
        sz_ref = f["checks/sz"][()]
        rho_ref = f["checks/rho_k3_d0"][()]
    out = {
        "norm_dev": abs(norm(psi) - 1.0),
        "sz_dev": float(np.abs(expect_sz(psi) - sz_ref).max()),
        "rho3_dev": float(np.abs(reduced_density_matrix(psi, [psi.j0 - 1, psi.j0, psi.j0 + 1])
                                 - rho_ref).max()),
    }
    bad = {k: v for k, v in out.items() if v > tol}
    if bad:
        raise AssertionError(f"{psi.path}: {bad}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="check every state under a directory and "
                                             "(re)write its manifest.csv")
    ap.add_argument("root")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--no-check", action="store_true", help="only write the manifest")
    args = ap.parse_args()
    if not args.no_check:
        worst = {}
        rows = scan(args.root)
        for i, r in enumerate(rows):
            dev = check(load(Path(args.root) / r["file"]), args.tol)
            for key, val in dev.items():
                worst[key] = max(worst.get(key, 0.0), val)
        print(f"checked {len(rows)} states; worst deviations: {worst}")
    print("wrote", write_manifest(args.root))
