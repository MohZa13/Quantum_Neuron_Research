"""Screen QH9 molecules for conjugation / quantum-behavior potential.

Streams records from the raw QH9Stable.db and, for each molecule, computes three
complementary rankings (see docs discussion, thermal_pipeline_report.md):

  Tier 1  degree of unsaturation (DoU) -- formula only, ~free. Counts rings +
          pi-bonds but does NOT distinguish conjugated from isolated unsaturation.
  Tier 2  frontier gap + frontier density of states -- from eigh(F, S), free.
          The *direct* proxy for thermal coherence at kT: small gap / many
          near-degenerate frontier orbitals => quantum behavior. gap tracks
          mixing at -0.78 in the 50-molecule set.
  Tier 3  largest connected conjugated pi-subsystem (atoms) -- the true
          structural conjugation metric. Requires RDKit bond perception from
          geometry (rdDetermineBonds). Gated behind availability; the script
          degrades to Tiers 1-2 if RDKit is absent.

Output: a CSV sorted by ascending frontier gap (most quantum first), plus a
printed top-N table. Screen the whole set (~131k molecules) or a --limit slice
before committing the expensive thermal pipeline.

Usage:
    python -m scripts.screen_conjugation --qh9-path data/QH9Stable.db \
        --limit 1000 --kT 0.1 --out results/qh9_conjugation_screen.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
import scipy.linalg

from qthermal.loader import CachedUnitDetector, iter_records
from qthermal.orbitals import build_mol, overlap

logger = logging.getLogger("screen_conjugation")

_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
_ANGSTROM_PER_BOHR = 0.529177210903

FIELDS = ["idx", "formula", "n_heavy", "DoU", "gap_Ha",
          "n_frontier_within_kT", "largest_pi_atoms", "n_aromatic_atoms"]


def _coerce_row(r: dict) -> dict:
    """Coerce a checkpoint row (all strings, from CSV) back to native types."""
    def maybe_int(v):
        return "" if v == "" else int(v)
    return {
        "idx": int(r["idx"]),
        "formula": r["formula"],
        "n_heavy": int(r["n_heavy"]),
        "DoU": float(r["DoU"]),
        "gap_Ha": float(r["gap_Ha"]),
        "n_frontier_within_kT": int(r["n_frontier_within_kT"]),
        "largest_pi_atoms": maybe_int(r["largest_pi_atoms"]),
        "n_aromatic_atoms": maybe_int(r["n_aromatic_atoms"]),
    }


def _load_checkpoint(path: Path):
    """Return (rows, done_idx_set) from an existing partial file, or ([], set())."""
    if not path.exists():
        return [], set()
    rows = []
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append(_coerce_row(r))
            except (KeyError, ValueError):
                continue  # tolerate a torn final line from an abrupt kill
    done = {r["idx"] for r in rows}
    return rows, done

# --- optional RDKit (Tier 3) -------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_RDKIT = False


def formula(Z: np.ndarray) -> str:
    """Hill-order formula string (C, H, then others alphabetical)."""
    counts: dict[str, int] = {}
    for z in Z:
        counts[_SYMBOL.get(int(z), f"Z{int(z)}")] = counts.get(
            _SYMBOL.get(int(z), f"Z{int(z)}"), 0) + 1

    def piece(sym: str) -> str:
        n = counts.pop(sym, 0)
        return "" if n == 0 else f"{sym}{n if n > 1 else ''}"

    out = piece("C") + piece("H")
    for sym in sorted(counts):
        out += f"{sym}{counts[sym] if counts[sym] > 1 else ''}"
    return out


def degree_of_unsaturation(Z: np.ndarray) -> float:
    """DoU for a neutral CHONF molecule: (2*C + 2 + N - H - F) / 2. O is inert."""
    nC = int(np.sum(Z == 6))
    nN = int(np.sum(Z == 7))
    nH = int(np.sum(Z == 1))
    nF = int(np.sum(Z == 9))
    return (2 * nC + 2 + nN - nH - nF) / 2.0


def frontier_metrics(F: np.ndarray, S: np.ndarray, nelec: int, kT: float):
    """Return (homo_lumo_gap, n_frontier_within_kT) from the eigh(F, S) spectrum.

    n_frontier_within_kT counts single-particle levels lying within kT of the
    mid-gap Fermi level -- the near-degenerate frontier density that thermal
    population at kT can reach.
    """
    eps = scipy.linalg.eigh(F, S, eigvals_only=True)
    nocc = nelec // 2
    homo = float(eps[nocc - 1])
    lumo = float(eps[nocc])
    gap = lumo - homo
    e_fermi = 0.5 * (homo + lumo)
    n_frontier = int(np.sum(np.abs(eps - e_fermi) <= kT))
    return gap, n_frontier


def largest_conjugated_system(Z: np.ndarray, R_angstrom: np.ndarray):
    """(largest_pi_atoms, n_aromatic_atoms) via RDKit bond perception, or None.

    Builds a molecule from geometry, perceives bond orders/aromaticity, then
    returns the atom count of the largest connected subgraph spanned by
    conjugated bonds. Returns None on any perception failure.
    """
    if not _HAVE_RDKIT:
        return None
    lines = [str(len(Z)), ""]
    for z, xyz in zip(Z, R_angstrom):
        sym = _SYMBOL.get(int(z), "X")
        lines.append(f"{sym} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    xyz_block = "\n".join(lines) + "\n"
    try:
        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None:
            return None
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception:
        return None

    n_atoms = mol.GetNumAtoms()
    parent = list(range(n_atoms))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for bond in mol.GetBonds():
        if bond.GetIsConjugated():
            ra, rb = find(bond.GetBeginAtomIdx()), find(bond.GetEndAtomIdx())
            if ra != rb:
                parent[ra] = rb

    conj_atoms = {a for bond in mol.GetBonds() if bond.GetIsConjugated()
                  for a in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())}
    sizes: dict[int, int] = {}
    for a in conj_atoms:
        root = find(a)
        sizes[root] = sizes.get(root, 0) + 1
    largest = max(sizes.values()) if sizes else 0
    n_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    return largest, n_aromatic


def screen(qh9_path: str, limit: int | None, kT: float,
           checkpoint: Path, flush_every: int = 500):
    """Screen records, appending each row to ``checkpoint`` as it is computed.

    Resumable: rows already present in ``checkpoint`` are loaded and their
    molecules skipped, so a killed run continues where it left off. The append
    file is flushed (and fsync'd) every ``flush_every`` new rows. Returns the
    full sorted row list plus (n_new_ok, n_fail) for this invocation.
    """
    rows, done = _load_checkpoint(checkpoint)
    if done:
        logger.info("resuming: %d molecules already in %s", len(done), checkpoint)

    detector = CachedUnitDetector(recheck_every=100)
    n_ok = n_fail = 0
    fh = checkpoint.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if not done:
        writer.writeheader()
    try:
        for ordinal, rec in enumerate(iter_records(qh9_path, limit=limit)):
            if rec.idx in done:
                continue
            try:
                unit = detector.unit_for(rec, ordinal)
                mol = build_mol(rec, unit)
                S = overlap(mol)
                gap, n_frontier = frontier_metrics(rec.F, S, mol.nelectron, kT)
                R_ang = rec.R if unit == "Angstrom" else rec.R * _ANGSTROM_PER_BOHR
                conj = largest_conjugated_system(rec.Z, R_ang)
                row = {
                    "idx": rec.idx,
                    "formula": formula(rec.Z),
                    "n_heavy": int(np.sum(rec.Z > 1)),
                    "DoU": degree_of_unsaturation(rec.Z),
                    "gap_Ha": round(gap, 5),
                    "n_frontier_within_kT": n_frontier,
                    "largest_pi_atoms": "" if conj is None else conj[0],
                    "n_aromatic_atoms": "" if conj is None else conj[1],
                }
                writer.writerow(row)
                rows.append(row)
                n_ok += 1
                if n_ok % flush_every == 0:
                    fh.flush()
                    os.fsync(fh.fileno())
            except Exception as exc:
                n_fail += 1
                logger.debug("skip idx=%d: %s", rec.idx, exc)
            if (ordinal + 1) % 100 == 0:
                logger.info("processed %d (new_ok=%d fail=%d, total=%d)",
                            ordinal + 1, n_ok, n_fail, len(rows))
    finally:
        fh.flush()
        os.fsync(fh.fileno())
        fh.close()

    # Most quantum first: ascending frontier gap.
    rows.sort(key=lambda r: float(r["gap_Ha"]))
    return rows, n_ok, n_fail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qh9-path", default="data/QH9Stable.db",
                    help="QH9Stable.db file or QH9 root directory")
    ap.add_argument("--limit", type=int, default=None,
                    help="screen only the first N records (default: all)")
    ap.add_argument("--kT", type=float, default=0.1,
                    help="temperature (Ha) for the frontier-DOS window")
    ap.add_argument("--out", default="results/qh9_conjugation_screen.csv")
    ap.add_argument("--checkpoint", default=None,
                    help="partial append-file for resume (default: <out>.partial)")
    ap.add_argument("--flush-every", type=int, default=500,
                    help="fsync the checkpoint every N new rows")
    ap.add_argument("--top", type=int, default=20,
                    help="rows to print to stdout")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not _HAVE_RDKIT:
        logger.warning("RDKit not available -- Tier 3 (structural conjugation) "
                       "columns will be blank; ranking uses Tiers 1-2 only.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint \
        else out_path.with_suffix(out_path.suffix + ".partial")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    rows, n_ok, n_fail = screen(args.qh9_path, args.limit, args.kT,
                                checkpoint, args.flush_every)

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("\nscreened %d molecules this run (%d skipped); %d total -> %s",
                n_ok, n_fail, len(rows), out_path)
    logger.info("\nTop %d by ascending frontier gap (most quantum first):", args.top)
    header = f"{'idx':>6} {'formula':<10} {'DoU':>4} {'gap_Ha':>7} " \
             f"{'nFront':>6} {'piAtoms':>7} {'arom':>4}"
    logger.info(header)
    logger.info("-" * len(header))
    for r in rows[:args.top]:
        logger.info(f"{r['idx']:>6} {r['formula']:<10} {r['DoU']:>4.1f} "
                    f"{r['gap_Ha']:>7.4f} {r['n_frontier_within_kT']:>6} "
                    f"{str(r['largest_pi_atoms']):>7} {str(r['n_aromatic_atoms']):>4}")


if __name__ == "__main__":
    main()
