#!/usr/bin/env python3
"""Convert compact QH9 Slater weight datasets into explicit qubit state vectors.

This script reads one or more HDF5 files containing compact Slater determinant
weight matrices W and expands them into explicit dense computational-basis state
vectors and/or sparse N-electron sector amplitudes.
"""

import argparse
import glob
import h5py
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert compact Slater W datasets into explicit qubit states."
    )
    parser.add_argument(
        "--in",
        dest="in_paths",
        action="append",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more input HDF5 paths. Supports glob patterns and repeated values.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output HDF5 path.",
    )
    parser.add_argument(
        "--format",
        choices=["dense", "sparse-sector", "both"],
        default="dense",
        help="Output format: dense computational-basis state vectors, sparse sector amplitudes, or both.",
    )
    parser.add_argument(
        "--max-state-qubits",
        type=int,
        default=24,
        help="Maximum allowed qubit count for dense output.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to convert across all input files.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting row index (inclusive) within each input file.",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Stopping row index (exclusive) within each input file.",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Path to a text/json/npy file containing specific row indices to convert.",
    )
    parser.add_argument(
        "--dtype",
        choices=["complex64", "complex128"],
        default="complex64",
        help="Output complex data type.",
    )
    parser.add_argument(
        "--compression",
        choices=["none", "gzip", "lzf"],
        default="gzip",
        help="Compression for HDF5 datasets.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Gzip compression level.",
    )
    parser.add_argument(
        "--renormalize",
        action="store_true",
        help="Normalize each state vector after construction.",
    )
    parser.add_argument(
        "--check-w-isometry",
        action="store_true",
        help="Check that W_i^H W_i is approximately identity before expanding.",
    )
    parser.add_argument(
        "--isometry-tol",
        type=float,
        default=1e-7,
        help="Tolerance for isometry checking.",
    )
    parser.add_argument(
        "--norm-tol",
        type=float,
        default=1e-6,
        help="Tolerance for state vector norm deviation from 1.",
    )
    parser.add_argument(
        "--bit-ordering",
        choices=["little", "big"],
        default="little",
        help="Basis bit ordering for mapping occupied modes to basis indices.",
    )
    parser.add_argument(
        "--method",
        choices=["determinant", "creation-operator"],
        default="determinant",
        help="Dense expansion method. Determinant is the default and most robust.",
    )
    parser.add_argument(
        "--on-error",
        choices=["skip", "raise"],
        default="skip",
        help="Whether to skip or raise on invalid samples.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Placeholder for future append/resume behavior; currently ignored.",
    )
    return parser.parse_args()


def resolve_input_paths(patterns):
    paths = []
    for group in patterns:
        for pattern in group:
            matches = sorted(glob.glob(pattern, recursive=True))
            if not matches:
                raise FileNotFoundError(f"No input files found for pattern: {pattern}")
            for path in matches:
                if os.path.isdir(path):
                    continue
                if path not in paths:
                    paths.append(path)
    if not paths:
        raise FileNotFoundError("No input files were resolved.")
    return paths


def load_indices_file(indices_path):
    if indices_path is None:
        return None
    ext = Path(indices_path).suffix.lower()
    if ext == ".npy":
        indices = np.load(indices_path)
    else:
        with open(indices_path, "r", encoding="utf-8") as handle:
            if ext == ".json":
                indices = json.load(handle)
            else:
                indices = [int(line.strip()) for line in handle if line.strip()]
    if isinstance(indices, dict):
        result = {}
        for key, value in indices.items():
            result[str(key)] = np.asarray(value, dtype=np.int64)
        return result
    return np.asarray(indices, dtype=np.int64)


def parse_source_indices(source_path, row_indices):
    if row_indices is None:
        return None
    if isinstance(row_indices, np.ndarray):
        return row_indices
    if isinstance(row_indices, dict):
        key = str(source_path)
        if key in row_indices:
            return np.asarray(row_indices[key], dtype=np.int64)
        key = os.path.basename(source_path)
        if key in row_indices:
            return np.asarray(row_indices[key], dtype=np.int64)
        raise ValueError(
            f"Indices file contains a dict but no entry matches '{source_path}' or '{os.path.basename(source_path)}'."
        )
    raise ValueError("Unsupported indices file content type.")


def ensure_dataset_exists(dataset: h5py.Group, name: str) -> h5py.Dataset:
    if name not in dataset:
        raise KeyError(f"Required dataset '{name}' not found in input file.")
    return cast(h5py.Dataset, dataset[name])


def ensure_optional_dataset(dataset: h5py.Group, name: str) -> Optional[h5py.Dataset]:
    if name in dataset:
        return cast(h5py.Dataset, dataset[name])
    return None


def modes_to_basis_index(occ_modes, n_qubits, bit_ordering="little"):
    index = 0
    if bit_ordering == "little":
        for p in occ_modes:
            index |= 1 << int(p)
    else:
        for p in occ_modes:
            index |= 1 << (n_qubits - 1 - int(p))
    return index


def reverse_bit_ordering(index, n_qubits):
    result = 0
    for p in range(n_qubits):
        if index >> p & 1:
            result |= 1 << (n_qubits - 1 - p)
    return result


def reorder_state_vector(psi, n_qubits):
    size = psi.shape[0]
    permuted = np.empty_like(psi)
    for index in range(size):
        permuted[reverse_bit_ordering(index, n_qubits)] = psi[index]
    return permuted


def slater_W_to_dense_state(
    W_i: np.ndarray,
    nelec: int,
    bit_ordering: str = "little",
    dtype: np.dtype = np.dtype(np.complex128),
    method: str = "determinant",
) -> np.ndarray:
    n_qubits = W_i.shape[0]
    if nelec < 0 or nelec > n_qubits:
        raise ValueError(f"Invalid electron count {nelec} for {n_qubits} qubits.")
    if method == "determinant":
        size = 1 << n_qubits
        psi = np.zeros(size, dtype=dtype)
        for occ_modes in combinations(range(n_qubits), nelec):
            submatrix = W_i[np.asarray(occ_modes, dtype=np.int64), :nelec]
            amplitude = np.linalg.det(submatrix)
            basis_index = modes_to_basis_index(occ_modes, n_qubits, bit_ordering)
            psi[basis_index] = amplitude
        return psi

    if method == "creation-operator":
        size = 1 << n_qubits
        psi = np.zeros(size, dtype=dtype)
        psi[0] = 1.0
        for k in range(nelec):
            coeffs = W_i[:, k]
            next_psi = np.zeros_like(psi)
            for bits in range(size):
                amplitude = psi[bits]
                if amplitude == 0:
                    continue
                for p in range(n_qubits):
                    if bits >> p & 1:
                        continue
                    sign = -1.0 if bin(bits & ((1 << p) - 1)).count("1") % 2 else 1.0
                    next_bits = bits | (1 << p)
                    next_psi[next_bits] += coeffs[p] * amplitude * sign
            psi = next_psi
        if bit_ordering == "big":
            psi = reorder_state_vector(psi, n_qubits)
        return psi

    raise ValueError(f"Unknown expansion method: {method}")


def slater_W_to_sparse_sector(
    W_i: np.ndarray,
    nelec: int,
    bit_ordering: str = "little",
    dtype: np.dtype = np.dtype(np.complex128),
) -> Tuple[np.ndarray, np.ndarray]:
    n_qubits = W_i.shape[0]
    basis_indices = []
    amplitudes = []
    for occ_modes in combinations(range(n_qubits), nelec):
        submatrix = W_i[np.asarray(occ_modes, dtype=np.int64), :nelec]
        amplitude = np.linalg.det(submatrix)
        basis_index = modes_to_basis_index(occ_modes, n_qubits, bit_ordering)
        basis_indices.append(basis_index)
        amplitudes.append(amplitude)
    if len(basis_indices) == 0:
        basis_indices = np.array([], dtype=np.uint64 if n_qubits <= 63 else h5py.string_dtype(encoding="utf-8"))
        amplitudes = np.array([], dtype=dtype)
    else:
        basis_indices = np.asarray(basis_indices, dtype=np.uint64 if n_qubits <= 63 else object)
        amplitudes = np.asarray(amplitudes, dtype=dtype)
    return basis_indices, amplitudes


def isometry_error(W_i, nelec):
    if nelec == 0:
        return 0.0
    gram = W_i.conj().T @ W_i
    identity = np.eye(nelec, dtype=gram.dtype)
    return float(np.linalg.norm(gram - identity))


def normalize_state_vector(psi):
    norm = float(np.linalg.norm(psi))
    if norm == 0.0:
        return psi, norm
    return psi / norm, norm


def normalize_amplitudes(amplitudes):
    norm = float(np.linalg.norm(amplitudes))
    if norm == 0.0:
        return amplitudes, norm
    return amplitudes / norm, norm


def format_basis_index(index, n_qubits):
    return str(index)


def selected_rows_for_input(input_h5, start, stop, row_indices):
    if row_indices is None:
        return list(range(start, stop))
    rows = [int(i) for i in row_indices if start <= int(i) < stop]
    rows.sort()
    return rows


def collect_output_counts(
    source_files: List[str],
    args: argparse.Namespace,
    indices: Optional[Any],
    n_qubits: int,
) -> Tuple[List[Tuple[str, List[int]]], int, int]:
    selected_rows: List[Tuple[str, List[int]]] = []
    total_samples = 0
    total_sparse_amplitudes = 0
    for source_path in source_files:
        with h5py.File(source_path, "r") as input_h5:
            w_ds = cast(h5py.Dataset, input_h5["W"])
            n_samples = w_ds.shape[0]
            start = max(0, args.start)
            stop = n_samples if args.stop is None else min(args.stop, n_samples)
            if start < 0 or stop < 0 or start >= n_samples or stop < start:
                raise ValueError(f"Invalid start/stop range for {source_path}: {start}:{stop}")
            row_indices = parse_source_indices(source_path, indices)
            rows = selected_rows_for_input(input_h5, start, stop, row_indices)
            if args.max_samples is not None:
                rows = rows[: max(0, args.max_samples - total_samples)]
            if args.format in ("sparse-sector", "both"):
                nelec_ds = cast(h5py.Dataset, input_h5["nelec"])
                for row in rows:
                    nelec = int(nelec_ds[row])
                    total_sparse_amplitudes += math.comb(n_qubits, nelec)
            selected_rows.append((source_path, rows))
            total_samples += len(rows)
            if args.max_samples is not None and total_samples >= args.max_samples:
                break
    return selected_rows, total_samples, total_sparse_amplitudes


def build_output_file(args, source_files, n_qubits, total_samples, total_sparse_amplitudes, source_attrs):
    compression = None if args.compression == "none" else args.compression
    compression_opts = None
    if args.compression == "gzip":
        compression_opts = args.compression_level

    if args.format in ("dense", "both") and n_qubits > args.max_state_qubits:
        raise ValueError(
            f"Dense output requires n_qubits <= {args.max_state_qubits}. Found n_qubits={n_qubits}"
        )

    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    h5_kwargs = {"compression": compression}
    if compression_opts is not None:
        h5_kwargs["compression_opts"] = compression_opts

    def chunk_1d(length):
        return (min(length, 1024),) if length > 0 else None

    h5_out = h5py.File(args.out, "w")
    h5_out.attrs["created_by"] = "build_qh9_state_vectors.py"
    h5_out.attrs["source_files_json"] = json.dumps(source_files)
    h5_out.attrs["source_file_count"] = len(source_files)
    h5_out.attrs["input_representation"] = "Slater W matrices"
    if args.format == "dense":
        h5_out.attrs["output_representation"] = "dense computational-basis state vectors"
    elif args.format == "sparse-sector":
        h5_out.attrs["output_representation"] = "sparse particle-sector amplitudes"
    else:
        h5_out.attrs["output_representation"] = "dense computational-basis state vectors and sparse particle-sector amplitudes"
    h5_out.attrs["bit_ordering"] = args.bit_ordering
    if source_attrs.get("spin_orbital_ordering") is not None:
        h5_out.attrs["fermionic_mode_ordering"] = source_attrs.get("spin_orbital_ordering")
    h5_out.attrs["amplitude_convention"] = "psi[index(I)] = det(W_i[I, :]) for sorted occupied modes I"
    h5_out.attrs["state_dtype"] = args.dtype
    h5_out.attrs["max_state_qubits"] = args.max_state_qubits
    h5_out.attrs["n_qubits"] = n_qubits
    h5_out.attrs["n_amplitudes"] = 1 << n_qubits
    h5_out.attrs["dense_warning"] = "dense state vectors scale as 2**n_qubits"
    h5_out.attrs["label_source"] = source_attrs.get("label_source", "")
    if source_attrs.get("gap_threshold_ev") is not None:
        h5_out.attrs["gap_threshold_ev"] = source_attrs.get("gap_threshold_ev")
    h5_out.attrs["method"] = args.method
    h5_out.attrs["on_error"] = args.on_error
    h5_out.attrs["renormalize"] = bool(args.renormalize)
    h5_out.attrs["check_w_isometry"] = bool(args.check_w_isometry)
    h5_out.attrs["isometry_tol"] = args.isometry_tol
    h5_out.attrs["norm_tol"] = args.norm_tol

    dtype = np.dtype(args.dtype)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    datasets = {}
    if args.format in ("dense", "both"):
        total_amplitudes = 1 << n_qubits
        chunk_size = min(total_amplitudes, 2 ** 20)
        datasets["psi"] = h5_out.create_dataset(
            "psi",
            shape=(total_samples, total_amplitudes),
            dtype=dtype,
            chunks=(1, chunk_size),
            **h5_kwargs,
        )
    if args.format in ("sparse-sector", "both"):
        if n_qubits <= 63:
            basis_dtype = np.uint64
        else:
            basis_dtype = string_dtype
        datasets["sector_basis_indices"] = h5_out.create_dataset(
            "sector_basis_indices",
            shape=(total_sparse_amplitudes,),
            dtype=basis_dtype,
            chunks=chunk_1d(total_sparse_amplitudes),
            **h5_kwargs,
        )
        datasets["sector_amplitudes"] = h5_out.create_dataset(
            "sector_amplitudes",
            shape=(total_sparse_amplitudes,),
            dtype=dtype,
            chunks=chunk_1d(total_sparse_amplitudes),
            **h5_kwargs,
        )
        datasets["sector_offsets"] = h5_out.create_dataset(
            "sector_offsets",
            shape=(total_samples + 1,),
            dtype=np.int64,
            chunks=chunk_1d(total_samples + 1),
            **h5_kwargs,
        )
    datasets["y_gap_binary"] = h5_out.create_dataset(
        "y_gap_binary",
        shape=(total_samples,),
        dtype=np.int64,
        chunks=chunk_1d(total_samples),
        **h5_kwargs,
    )
    datasets["nelec"] = h5_out.create_dataset(
        "nelec",
        shape=(total_samples,),
        dtype=np.int64,
        chunks=chunk_1d(total_samples),
        **h5_kwargs,
    )
    datasets["n_occ_spatial"] = h5_out.create_dataset(
        "n_occ_spatial",
        shape=(total_samples,),
        dtype=np.int64,
        chunks=chunk_1d(total_samples),
        **h5_kwargs,
    )
    datasets["gap_qh9_ev"] = h5_out.create_dataset(
        "gap_qh9_ev",
        shape=(total_samples,),
        dtype=np.float64,
        chunks=chunk_1d(total_samples),
        **h5_kwargs,
    )
    datasets["gap_qm9_ev"] = h5_out.create_dataset(
        "gap_qm9_ev",
        shape=(total_samples,),
        dtype=np.float64,
        chunks=chunk_1d(total_samples),
        **h5_kwargs,
    )
    datasets["homo_qh9_ev"] = h5_out.create_dataset(
        "homo_qh9_ev",
        shape=(total_samples,),
        dtype=np.float64,
        chunks=chunk_1d(total_samples),
        **h5_kwargs,
    )
    datasets["lumo_qh9_ev"] = h5_out.create_dataset(
        "lumo_qh9_ev",
        shape=(total_samples,),
        dtype=np.float64,
        chunks=chunk_1d(total_samples),
        **h5_kwargs,
    )
    datasets["qh9_index"] = h5_out.create_dataset(
        "qh9_index",
        shape=(total_samples,),
        dtype=np.int64,
        chunks=chunk_1d(total_samples),
        **h5_kwargs,
    )
    datasets["original_index"] = h5_out.create_dataset(
        "original_index",
        shape=(total_samples,),
        dtype=np.int64,
        chunks=chunk_1d(total_samples),
        **h5_kwargs,
    )
    datasets["source_file_id"] = h5_out.create_dataset(
        "source_file_id",
        shape=(total_samples,),
        dtype=np.int64,
        chunks=chunk_1d(total_samples),
        **h5_kwargs,
    )
    datasets["source_row"] = h5_out.create_dataset(
        "source_row",
        shape=(total_samples,),
        dtype=np.int64,
        chunks=chunk_1d(total_samples),
        **h5_kwargs,
    )
    datasets["state_norm"] = h5_out.create_dataset(
        "state_norm",
        shape=(total_samples,),
        dtype=np.float64,
        chunks=chunk_1d(total_samples),
        **h5_kwargs,
    )
    datasets["w_isometry_error"] = h5_out.create_dataset(
        "w_isometry_error",
        shape=(total_samples,),
        dtype=np.float64,
        chunks=chunk_1d(total_samples),
        **h5_kwargs,
    )
    datasets["metadata_json"] = h5_out.create_dataset(
        "metadata_json",
        shape=(total_samples,),
        dtype=string_dtype,
        chunks=chunk_1d(total_samples),
        **h5_kwargs,
    )

    return h5_out, datasets


def read_scalar_dataset(dataset: Optional[h5py.Dataset], row: int, default: Any, dtype: Callable[[Any], Any]) -> Any:
    if dataset is None:
        return default
    try:
        return dtype(cast(h5py.Dataset, dataset)[row])
    except Exception:
        return default


def create_metadata_json(sample_meta: Dict[str, Any]) -> str:
    return json.dumps(sample_meta, sort_keys=True)


def main():
    args = parse_args()
    if args.skip_existing:
        print("Warning: --skip-existing is not implemented; behavior is unchanged.")

    source_files = resolve_input_paths(args.in_paths)
    indices = load_indices_file(args.indices)

    source_attrs: Dict[str, Any] = {}
    n_qubits: Optional[int] = None
    label_source: Optional[str] = None
    gap_threshold_ev: Optional[float] = None
    for source_path in source_files:
        with h5py.File(source_path, "r") as input_h5:
            ensure_dataset_exists(input_h5, "W")
            ensure_dataset_exists(input_h5, "nelec")
            ensure_dataset_exists(input_h5, "y_gap_binary")
            w_ds = cast(h5py.Dataset, input_h5["W"])
            w_shape = w_ds.shape
            if len(w_shape) != 3 or w_shape[1] != w_shape[2]:
                raise ValueError(
                    f"Input file {source_path} has invalid W shape {w_shape}; expected (n_samples,n_qubits,n_qubits)."
                )
            file_n_qubits = int(w_shape[1])
            if n_qubits is None:
                n_qubits = file_n_qubits
            elif file_n_qubits != n_qubits:
                raise ValueError(
                    f"Input files have inconsistent n_qubits: {n_qubits} vs {file_n_qubits}."
                )
            if label_source is None:
                label_source = input_h5.attrs.get("label_source")
            elif input_h5.attrs.get("label_source") != label_source:
                label_source = None
            if gap_threshold_ev is None:
                gap_threshold_ev = input_h5.attrs.get("gap_threshold_ev")
            elif input_h5.attrs.get("gap_threshold_ev") != gap_threshold_ev:
                gap_threshold_ev = None
            if "spin_orbital_ordering" in input_h5.attrs:
                source_attrs["spin_orbital_ordering"] = input_h5.attrs["spin_orbital_ordering"]

    source_attrs["label_source"] = label_source
    source_attrs["gap_threshold_ev"] = gap_threshold_ev

    assert n_qubits is not None
    n_qubits = cast(int, n_qubits)
    if args.format in ("dense", "both") and n_qubits > args.max_state_qubits:
        raise ValueError(
            f"n_qubits={n_qubits} exceeds --max-state-qubits={args.max_state_qubits}. Use sparse-sector or lower the limit."
        )

    if args.format == "dense" and args.method == "creation-operator" and args.bit_ordering == "big":
        print("Using creation-operator method with big bit ordering; the final state will be remapped accordingly.")

    selected_rows_per_file, total_samples, total_sparse_amplitudes = collect_output_counts(
        source_files, args, indices, n_qubits
    )
    h5_out, datasets = build_output_file(
        args, source_files, n_qubits, total_samples, total_sparse_amplitudes, source_attrs
    )
    skipped_reasons = {}
    sample_index = 0
    sparse_index = 0
    sector_offsets = np.zeros(total_samples + 1, dtype=np.int64)

    for source_file_id, (source_path, selected_rows) in enumerate(selected_rows_per_file):
        with h5py.File(source_path, "r") as input_h5:
            w_ds = cast(h5py.Dataset, input_h5["W"])
            nelec_ds = cast(h5py.Dataset, input_h5["nelec"])
            y_gap_binary_ds = cast(h5py.Dataset, input_h5["y_gap_binary"])
            n_occ_spatial_ds = ensure_optional_dataset(input_h5, "n_occ_spatial")
            gap_qh9_ev_ds = ensure_optional_dataset(input_h5, "gap_qh9_ev")
            gap_qm9_ev_ds = ensure_optional_dataset(input_h5, "gap_qm9_ev")
            homo_qh9_ev_ds = ensure_optional_dataset(input_h5, "homo_qh9_ev")
            lumo_qh9_ev_ds = ensure_optional_dataset(input_h5, "lumo_qh9_ev")
            qh9_index_ds = ensure_optional_dataset(input_h5, "qh9_index")
            original_index_ds = ensure_optional_dataset(input_h5, "original_index")
            for row in selected_rows:
                if args.max_samples is not None and sample_index >= args.max_samples:
                    break
                basis_indices: np.ndarray = np.array([], dtype=np.uint64)
                amplitudes: np.ndarray = np.empty(0, dtype=np.dtype(args.dtype))
                try:
                    nelec = int(nelec_ds[row])
                    W_i = w_ds[row, :, :nelec]
                    if W_i.shape != (n_qubits, nelec):
                        raise ValueError(
                            f"Sample {source_path}:{row}: physical W shape {W_i.shape} does not match expected ({n_qubits},{nelec})."
                        )
                    if args.check_w_isometry:
                        w_err = isometry_error(W_i, nelec)
                    else:
                        w_err = np.nan
                    if args.check_w_isometry and w_err > args.isometry_tol:
                        raise ValueError(
                            f"W isometry error {w_err:.3e} exceeds tol {args.isometry_tol}"
                        )
                    if args.format in ("dense", "both"):
                        psi = slater_W_to_dense_state(
                            W_i,
                            nelec,
                            bit_ordering=args.bit_ordering,
                            dtype=np.dtype(args.dtype),
                            method=args.method,
                        )
                        psi_norm = float(np.linalg.norm(psi))
                    else:
                        psi = None
                        psi_norm = None
                    if args.format in ("sparse-sector", "both"):
                        basis_indices, amplitudes = slater_W_to_sparse_sector(
                            W_i,
                            nelec,
                            bit_ordering=args.bit_ordering,
                            dtype=np.dtype(args.dtype),
                        )
                        sector_norm = float(np.linalg.norm(amplitudes))
                        if psi is None:
                            psi_norm = sector_norm
                    if args.renormalize:
                        if psi is not None:
                            psi, pre_norm = normalize_state_vector(psi)
                        if args.format in ("sparse-sector", "both"):
                            amplitudes, _ = normalize_amplitudes(amplitudes)
                    if not args.renormalize and psi_norm is not None:
                        if abs(psi_norm - 1.0) > args.norm_tol:
                            raise ValueError(
                                f"Sample {source_path}:{row} norm deviation {psi_norm:.6e} exceeds tol {args.norm_tol}."
                            )
                    if args.format in ("dense", "both"):
                        datasets["psi"][sample_index, :] = psi
                    if args.format in ("sparse-sector", "both"):
                        if len(basis_indices):
                            if n_qubits <= 63:
                                datasets["sector_basis_indices"][sparse_index : sparse_index + len(basis_indices)] = basis_indices
                            else:
                                datasets["sector_basis_indices"][sparse_index : sparse_index + len(basis_indices)] = [
                                    format_basis_index(int(b), n_qubits) for b in basis_indices
                                ]
                            datasets["sector_amplitudes"][sparse_index : sparse_index + len(amplitudes)] = amplitudes
                            sparse_index += len(amplitudes)
                        sector_offsets[sample_index + 1] = sparse_index
                    datasets["y_gap_binary"][sample_index] = int(y_gap_binary_ds[row])
                    datasets["nelec"][sample_index] = nelec
                    datasets["n_occ_spatial"][sample_index] = read_scalar_dataset(n_occ_spatial_ds, row, -1, int)
                    datasets["gap_qh9_ev"][sample_index] = read_scalar_dataset(gap_qh9_ev_ds, row, np.nan, float)
                    datasets["gap_qm9_ev"][sample_index] = read_scalar_dataset(gap_qm9_ev_ds, row, np.nan, float)
                    datasets["homo_qh9_ev"][sample_index] = read_scalar_dataset(homo_qh9_ev_ds, row, np.nan, float)
                    datasets["lumo_qh9_ev"][sample_index] = read_scalar_dataset(lumo_qh9_ev_ds, row, np.nan, float)
                    datasets["qh9_index"][sample_index] = read_scalar_dataset(qh9_index_ds, row, -1, int)
                    datasets["original_index"][sample_index] = read_scalar_dataset(original_index_ds, row, -1, int)
                    datasets["source_file_id"][sample_index] = source_file_id
                    datasets["source_row"][sample_index] = row
                    datasets["state_norm"][sample_index] = psi_norm if psi_norm is not None else np.nan
                    datasets["w_isometry_error"][sample_index] = w_err
                    datasets["metadata_json"][sample_index] = create_metadata_json(
                        {
                            "source_file": source_path,
                            "source_row": row,
                            "qh9_index": read_scalar_dataset(qh9_index_ds, row, -1, int),
                            "original_index": read_scalar_dataset(original_index_ds, row, -1, int),
                            "nelec": nelec,
                            "n_qubits": n_qubits,
                            "n_occ_spatial": read_scalar_dataset(n_occ_spatial_ds, row, -1, int),
                            "gap_qh9_ev": read_scalar_dataset(gap_qh9_ev_ds, row, np.nan, float),
                            "homo_qh9_ev": read_scalar_dataset(homo_qh9_ev_ds, row, np.nan, float),
                            "lumo_qh9_ev": read_scalar_dataset(lumo_qh9_ev_ds, row, np.nan, float),
                            "state_norm": psi_norm if psi_norm is not None else np.nan,
                            "w_isometry_error": w_err,
                        }
                    )
                    sample_index += 1
                except Exception as exc:
                    reason = str(exc)
                    skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                    if args.on_error == "raise":
                        h5_out.close()
                        raise
                    continue
    if args.format in ("sparse-sector", "both"):
        datasets["sector_offsets"][:] = sector_offsets

    h5_out.attrs["converted_samples"] = sample_index
    h5_out.close()

    print(f"Wrote {sample_index} converted samples to {args.out}")
    if skipped_reasons:
        print("Skipped samples:")
        for reason, count in skipped_reasons.items():
            print(f"  {count}: {reason}")


if __name__ == "__main__":
    main()
