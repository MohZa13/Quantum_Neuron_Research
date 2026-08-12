"""Bridge steps 1-3: qthermal thermal states -> quantum-neuron training data.

Reads a Module-H run file (e.g. results/qh9_dense_cas8-8_kT0p1.h5), turns each
molecule's thermal Gibbs state at one temperature into an eigenblock training
record ``{rho_m, y_m}``, and serialises the batch to an HDF5 file that the
Julia (Yao / ITensor) Algorithm-8/9 trainer can load.

Representation.  rho_m = sum_k p_k |v_k><v_k| is exported in *eigenblock* form,
not as a dense matrix (2**(2*ncas) is 65536 for CAS(8,8) -- a dense rho is 4e9
complex entries and impossible; the eigenblock is tiny):

  - ``weights``  p_k, the Boltzmann weights (sum = 1 - truncation_error);
  - ``amps``     (m, dim) real amplitudes of each |v_k> over the fixed
                 particle-number / Sz sector, with Jordan-Wigner parity signs
                 folded in;
  - ``basis_indices`` (dim,) the computational-basis position of each sector
                 amplitude on Q = 2*ncas qubits -- identical for every molecule,
                 so it is stored once at the top level.

The full state vector is recovered as ``v = zeros(2**Q); v[basis_indices] =
amps[k]`` (Julia: ``v[basis_indices .+ 1] = amps[k, :]``). This is exact and
compact, and feeds both Julia paths: the sampling wrapper (draw k ~ p_k, run the
pure-state circuit on |v_k>) and an explicit density-matrix build.

Labels.  y_m in {-1, +1} is the key modeling choice, and is pluggable via
``--label``.  The DEFAULT (``h5:static_corr``, median threshold) is a
PLACEHOLDER: static correlation is a multireference score, more "collective"
than raw composition, but you should choose the target deliberately -- per the
coherence-confound finding, a label a classical model can read straight off
single-qubit occupations makes the quantum neuron pointless.

Usage:
    python -m scripts.export_thermal_training \
        --in results/qh9_dense_cas88_5mols.h5 \
        --out results/thermal_training_5mol.h5 \
        --ordering interleaved --label h5:static_corr
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

# Reuse the pipeline's own encoders: jw_basis_indices/jw_parity_signs turn the
# CI states into qubit-basis positions, kT_tag builds the "kT_0p1000" group name.
from qthermal.encode import jw_basis_indices, jw_parity_signs
from qthermal.io_hdf5 import kT_tag

logger = logging.getLogger("export_thermal_training")

# Atomic number -> element symbol, for building human-readable formulas (CHONF).
_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def hill_formula(Z: np.ndarray) -> str:
    """Hill-order formula (C, H, then alphabetical) for provenance."""
    # Tally how many atoms of each element the molecule has.
    counts: dict[str, int] = {}
    for z in Z:
        s = _SYMBOL.get(int(z), f"Z{int(z)}")
        counts[s] = counts.get(s, 0) + 1

    # Format one element as e.g. "C2" (or just "C" when there is a single atom).
    def piece(sym: str) -> str:
        n = counts.pop(sym, 0)
        return "" if n == 0 else f"{sym}{n if n > 1 else ''}"

    # Hill convention: carbon and hydrogen first, then everything else A->Z.
    out = piece("C") + piece("H")
    for sym in sorted(counts):
        out += f"{sym}{counts[sym] if counts[sym] > 1 else ''}"
    return out


def resolve_block(mol_grp, kT: float | None):
    """Return (tag, block) for the requested kT, or the sole block if kT is None."""
    # Each molecule group holds one subgroup per temperature, named "kT_<tag>".
    tags = sorted(k for k in mol_grp if k.startswith("kT_"))
    if not tags:
        raise ValueError("molecule group has no kT_ blocks")
    # A temperature was requested: translate it to a tag and look it up.
    if kT is not None:
        tag = kT_tag(kT)
        if tag not in tags:
            raise KeyError(f"{tag} not in {tags}")
        return tag, mol_grp[tag]
    # No temperature given: fine only if the molecule has exactly one block.
    if len(tags) > 1:
        raise ValueError(f"multiple kT blocks {tags}; pass --kT to choose")
    return tags[0], mol_grp[tags[0]]


def truncate_to_weight(p: np.ndarray, keep: float) -> int:
    """Number of leading eigenstates whose cumulative weight first reaches `keep`.

    `p` is already sorted descending (ground state first). keep >= 1 keeps all.
    """
    # Fast path: keep every eigenstate in the thermal mixture.
    if keep >= 1.0:
        return len(p)
    # Otherwise walk down the sorted weights and stop once we have enough of the
    # thermal probability, dropping the negligible long tail of rare states.
    csum = np.cumsum(p) / p.sum()
    return int(np.searchsorted(csum, keep) + 1)


def extract_record(mol_grp, *, ncas, na, nb, ordering, kT, weight_keep,
                   signs: np.ndarray):
    """Build one eigenblock training record from a molecule group.

    Returns a dict with idx, formula, kT, weights (m,), amps (m, dim),
    trace, truncation_error, and the raw per-block diagnostics (for labelling).
    """
    # The molecule's QH9 index is the number in its group name ("mol_<idx>").
    idx = int(str(mol_grp.name).rsplit("_", 1)[1])
    _, blk = resolve_block(mol_grp, kT)

    # p = Boltzmann weights of the thermal state's eigenstates (sorted, high->low).
    # Keep only the leading states that carry the requested share of the weight.
    p = blk["p"][:]
    m = truncate_to_weight(p, weight_keep)
    p = p[:m]
    # civecs = those eigenstates written over the CASCI determinant basis. The
    # parity signs convert them into the qubit (Jordan-Wigner) convention; the
    # basis positions live in `signs`'s partner array and are written out once.
    civecs = blk["civecs"][:m]                # (m, dim), real, determinant basis
    amps = civecs * signs[None, :]            # fold JW parity signs into amplitudes

    # Grab every scalar diagnostic stored on the block (static_corr, entropy,
    # ...) so any of them can later serve as the classification label.
    diagnostics = {k: float(blk[k][()]) for k in blk
                   if isinstance(blk[k], h5py.Dataset) and blk[k].shape == ()}
    # Package one training example: the eigenblock (weights + amps) plus provenance.
    return {
        "idx": idx,
        "formula": hill_formula(mol_grp["Z"][:]),
        "kT": float(blk.attrs["kT"]),
        "weights": np.ascontiguousarray(p, dtype=np.float64),
        "amps": np.ascontiguousarray(amps, dtype=np.float64),
        "trace": float(p.sum()),
        "truncation_error": float(blk["truncation_error"][()]),
        "n_states_full": int(blk["p"].shape[0]),
        "diagnostics": diagnostics,
    }


# --- labelling ---------------------------------------------------------------

def load_label_values(spec: str, records: list[dict]):
    """Resolve one scalar per record from a label spec.

    ``h5:<key>``            per-block diagnostic stored in the run file
    ``csv:<path>:<col>``    external table keyed by molecule idx
    """
    # Option A: read one number per molecule straight from the run file's
    # per-block diagnostics (e.g. h5:static_corr).
    if spec.startswith("h5:"):
        key = spec[3:]
        vals = []
        for r in records:
            if key not in r["diagnostics"]:
                raise KeyError(f"diagnostic {key!r} not in block; have "
                               f"{sorted(r['diagnostics'])}")
            vals.append(r["diagnostics"][key])
        return np.asarray(vals, dtype=np.float64)

    # Option B: join against an external CSV on the molecule index and pull one
    # column (e.g. csv:results/qh9_conjugation_screen_full.csv:gap_Ha).
    if spec.startswith("csv:"):
        import csv as _csv
        _, path, col = spec.split(":", 2)
        table = {}
        with open(path, newline="") as fh:
            for row in _csv.DictReader(fh):
                table[int(row["idx"])] = float(row[col])
        # Every exported molecule must have a row in the table.
        missing = [r["idx"] for r in records if r["idx"] not in table]
        if missing:
            raise KeyError(f"idx {missing[:5]}... absent from {path}")
        return np.asarray([table[r["idx"]] for r in records], dtype=np.float64)

    raise ValueError(f"label spec must start with 'h5:' or 'csv:', got {spec!r}")


def assign_labels(values: np.ndarray, threshold):
    """+1 where value > threshold else -1; threshold 'median' uses the batch median."""
    # Turn the continuous per-molecule values into a binary target by cutting at
    # a threshold ("median" splits the batch 50/50; or pass an explicit number).
    thr = float(np.median(values)) if threshold == "median" else float(threshold)
    y = np.where(values > thr, 1, -1).astype(np.int8)
    # Warn loudly if the cut leaves every molecule in one class -- nothing to learn.
    if len(np.unique(y)) == 1:
        logger.warning("all labels are %d -- threshold %.4g does not split this "
                       "batch; classifier has nothing to learn", y[0], thr)
    return y, thr


# --- driver ------------------------------------------------------------------

def export(src_path: str, out_path: str, *, kT, ordering, weight_keep,
           label_spec, threshold, limit):
    # --- Step 1: read the run file and extract one eigenblock per molecule ---
    with h5py.File(src_path, "r") as src:
        # Active-space size (ncas orbitals, na alpha + nb beta electrons) is the
        # same for every molecule; from it we precompute the JW sector mapping
        # (basis positions + parity signs) once and reuse it for all molecules.
        meta = dict(src["meta"].attrs)
        ncas = int(meta["ncas"])
        na = nb = int(meta["nelecas"]) // 2
        basis_indices = jw_basis_indices(ncas, na, nb, ordering)
        signs = jw_parity_signs(ncas, na, nb, ordering)

        # Walk molecule groups in QH9-index order, skipping any unfinished ones.
        names = sorted((k for k in src if k.startswith("mol_")),
                       key=lambda s: int(s.split("_")[1]))
        records = []
        for name in names:
            grp = src[name]
            if not grp.attrs.get("complete", False):
                logger.warning("%s incomplete; skipping", name)
                continue
            records.append(extract_record(
                grp, ncas=ncas, na=na, nb=nb, ordering=ordering, kT=kT,
                weight_keep=weight_keep, signs=signs))
            # Stop early once we have the requested number of molecules.
            if limit is not None and len(records) >= limit:
                break

    if not records:
        raise SystemExit("no complete molecules found in source file")

    # --- Step 2: attach a binary label y_m to every molecule ---
    values = load_label_values(label_spec, records)
    labels, thr = assign_labels(values, threshold)

    # --- Step 3: serialise everything into one self-contained HDF5 file ---
    dim = len(basis_indices)      # amplitudes per state (fixed particle sector)
    Q = 2 * ncas                  # number of qubits the states live on
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as out:
        # Top-level metadata: how to interpret every example downstream.
        mg = out.create_group("meta")
        mg.attrs.update(
            source_file=str(src_path), n_system_qubits=Q, ncas=ncas,
            nalpha=na, nbeta=nb, ordering=ordering, sector_dim=dim,
            encoding="jw_sector_sparse", kT=records[0]["kT"],
            label_spec=label_spec, label_threshold=thr,
            n_examples=len(records))
        # The sparse-state map, shared by every example (stored just once).
        out.create_dataset("basis_indices",
                           data=basis_indices.astype(np.int64))
        # Parallel per-molecule arrays for easy batch loading on the Julia side.
        out.create_dataset("labels", data=labels)
        out.create_dataset("mol_indices",
                           data=np.array([r["idx"] for r in records], np.int64))
        out.create_dataset("formulas",
                           data=np.array([r["formula"] for r in records], "S"))
        out.create_dataset("label_values", data=values)
        # One group per training example holding that molecule's thermal state.
        for k, r in enumerate(records):
            g = out.create_group(f"example_{k}")
            g.create_dataset("amps", data=r["amps"], compression="gzip")
            g.create_dataset("weights", data=r["weights"])
            g.attrs.update(mol_idx=r["idx"], label=int(labels[k]),
                           formula=r["formula"], trace=r["trace"],
                           truncation_error=r["truncation_error"],
                           n_states=r["amps"].shape[0],
                           n_states_full=r["n_states_full"])

    _report(records, labels, values, thr, Q, dim, out_path)


def _report(records, labels, values, thr, Q, dim, out_path):
    # Print a compact summary: file, sparsity, class balance, and a per-molecule
    # table so you can eyeball the split before handing the file to the trainer.
    logger.info("wrote %d training examples -> %s", len(records), out_path)
    logger.info("system qubits Q=%d, sector dim=%d (%.1f%% sparse)",
                Q, dim, 100 * (1 - dim / (1 << Q)))
    logger.info("labels: %d positive / %d negative (threshold %.4g)",
                int(np.sum(labels == 1)), int(np.sum(labels == -1)), thr)
    logger.info("\n%6s  %-11s %6s %8s %8s %9s",
                "idx", "formula", "label", "value", "n_states", "trace")
    for r, y, v in zip(records, labels, values):
        logger.info("%6d  %-11s %+6d %8.4f %8d %9.5f",
                    r["idx"], r["formula"], y, v, r["amps"].shape[0], r["trace"])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="Module-H run file")
    ap.add_argument("--out", required=True, help="output training HDF5")
    ap.add_argument("--kT", type=float, default=None,
                    help="temperature block to export (default: the sole block)")
    ap.add_argument("--ordering", choices=("blocked", "interleaved"),
                    default="interleaved",
                    help="JW wire layout; interleaved keeps same-orbital "
                         "alpha/beta pairs adjacent (nearest-neighbor ansatz)")
    ap.add_argument("--weight-keep", type=float, default=1.0,
                    help="keep leading eigenstates up to this cumulative "
                         "Boltzmann weight (default 1.0 = all)")
    ap.add_argument("--label", default="h5:static_corr",
                    help="label spec: h5:<diagnostic> or csv:<path>:<col>")
    ap.add_argument("--label-threshold", default="median",
                    help="'median' (default) or an explicit float")
    ap.add_argument("--limit", type=int, default=None,
                    help="export only the first N complete molecules")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(message)s")
    export(args.src, args.out, kT=args.kT, ordering=args.ordering,
           weight_keep=args.weight_keep, label_spec=args.label,
           threshold=args.label_threshold, limit=args.limit)


if __name__ == "__main__":
    main()
