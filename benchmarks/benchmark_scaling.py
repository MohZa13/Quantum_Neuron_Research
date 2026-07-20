"""Reproducible scaling comparison for the original and PennyLane notebooks.

The benchmark times one loss-and-gradient evaluation. One-time dataset reduction
or representation construction is reported separately as ``setup_seconds``.
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import numpy as np

from notebook_test_utils import (
    REPO_ROOT,
    csr_payload_bytes,
    densities,
    fused_payload_bytes,
    load_original_namespace,
    load_pennylane_namespace,
    median_runtime,
    original_loss_and_gradient,
    random_state_vectors,
)


OUTPUT_DIRECTORY = REPO_ROOT / "pennylane_project" / "benchmarks"
RESULTS_PATH = OUTPUT_DIRECTORY / "scaling_results.csv"
QUICK_RESULTS_PATH = OUTPUT_DIRECTORY / "scaling_results_quick.csv"
TEMPERATURE = 2.0


def timer(function):
    import time

    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def target_labels(rng, dense_terms, vectors):
    weights = rng.uniform(-1.0, 1.0, len(dense_terms))
    hamiltonian = sum(weight * term for weight, term in zip(weights, dense_terms))
    labels = np.sign(
        np.real(np.einsum("bi,ij,bj->b", vectors.conj(), hamiltonian, vectors))
    )
    labels[labels == 0] = 1
    return labels


def benchmark_case(original, penny, n, samples, sweep, include_original=True, repeats=3):
    rng = np.random.default_rng(100_000 + 1_000 * n + samples)
    vectors = random_state_vectors(rng, n, samples)
    states = densities(vectors)
    dense_quantum = original["generate_paulis"](n, "quantum")
    dense_classical = original["generate_paulis"](n, "classical")
    labels = target_labels(rng, dense_quantum, vectors)
    quantum_weights = rng.uniform(-0.4, 0.4, len(dense_quantum))
    classical_weights = rng.uniform(-0.4, 0.4, len(dense_classical))
    dim = 2**n
    rows = []

    def record(
        model,
        method,
        function,
        setup_seconds,
        data_bytes,
        operator_bytes,
        degree=None,
        method_repeats=None,
        notes="",
    ):
        result, seconds, timings = median_runtime(
            function,
            repeats=method_repeats or repeats,
            warmups=0 if method.startswith("original") else 1,
        )
        loss, gradient = result
        rows.append(
            {
                "sweep": sweep,
                "n": n,
                "dimension": dim,
                "samples": samples,
                "quantum_terms": 6 * n - 3,
                "classical_terms": n * (n + 1) // 2,
                "model": model,
                "method": method,
                "setup_seconds": setup_seconds,
                "pass_seconds": seconds,
                "minimum_seconds": min(timings),
                "data_bytes": int(data_bytes),
                "operator_bytes": int(operator_bytes),
                "static_bytes": int(data_bytes + operator_bytes),
                "chebyshev_degree": degree,
                "loss": float(loss),
                "gradient_norm": float(np.linalg.norm(gradient)),
                "notes": notes,
            }
        )

    dense_quantum_bytes = sum(term.nbytes for term in dense_quantum)
    dense_classical_bytes = sum(term.nbytes for term in dense_classical)

    if include_original:
        record(
            "quantum",
            "original_naive",
            lambda: original_loss_and_gradient(
                original,
                quantum_weights,
                states,
                labels,
                dense_quantum,
                TEMPERATURE,
            ),
            0.0,
            states.nbytes,
            dense_quantum_bytes,
            method_repeats=1,
            notes="Corrected dense baseline; uses dense matrices and per-sample traces.",
        )
        record(
            "classical",
            "original_naive",
            lambda: original_loss_and_gradient(
                original,
                classical_weights,
                states,
                labels,
                dense_classical,
                TEMPERATURE,
            ),
            0.0,
            states.nbytes,
            dense_classical_bytes,
            method_repeats=1,
            notes="Dense FCIM eigendecomposition despite all terms commuting.",
        )

    record(
        "quantum",
        "penny_dense_vectorized",
        lambda: penny["compute_loss_and_grads_vectorized"](
            quantum_weights, states, labels, dense_quantum, TEMPERATURE
        ),
        0.0,
        states.nbytes,
        dense_quantum_bytes,
        notes="Intermediate migration baseline; retains dense states and terms.",
    )

    symbolic_quantum = penny["generate_paulis_symbolic"](n, "quantum")
    sparse_quantum, sparse_setup = timer(
        lambda: penny["precompute_sparse_paulis_direct"](n, "quantum")
    )
    aggregates, aggregate_setup = timer(
        lambda: penny["aggregate_states_by_label"](vectors, labels)
    )
    record(
        "quantum",
        "penny_exact_aggregate",
        lambda: penny["compute_loss_and_grads_aggregated_symbolic"](
            quantum_weights,
            aggregates,
            symbolic_quantum,
            TEMPERATURE,
            n,
            sparse_quantum,
        ),
        sparse_setup + aggregate_setup,
        sum(aggregate.nbytes for aggregate in aggregates),
        csr_payload_bytes(sparse_quantum),
        notes="Exact; per-pass cost is independent of sample count after aggregation.",
    )

    fused_quantum, fused_setup = timer(
        lambda: penny["build_fused_pauli_kernel"](
            n, "quantum", complex_dtype="complex64"
        )
    )
    bound = 1.05 * np.sum(np.abs(quantum_weights))
    degree = penny["select_adaptive_chebyshev_degree"](
        TEMPERATURE,
        bound,
        tolerance=1e-5,
        min_degree=8,
        max_degree=64,
    )
    record(
        "quantum",
        "penny_chebyshev_matrix_free",
        lambda: penny["compute_loss_and_grads_chebyshev"](
            quantum_weights,
            vectors.astype(np.complex64),
            labels,
            fused_quantum,
            TEMPERATURE,
            degree=degree,
            chunk_size=min(32, samples),
            spectral_bound=bound,
            complex_dtype="complex64",
            term_chunk_size=8,
        ),
        fused_setup,
        vectors.astype(np.complex64).nbytes,
        fused_payload_bytes(fused_quantum),
        degree=degree,
        method_repeats=1,
        notes="Approximate; matrix-free and linear in samples and polynomial degree.",
    )

    features, feature_setup = timer(lambda: penny["build_fcim_feature_matrix"](n))
    probabilities, probability_setup = timer(
        lambda: penny["aggregate_basis_probabilities_by_label"](vectors, labels)
    )
    record(
        "classical",
        "penny_diagonal_fcim",
        lambda: penny["compute_loss_and_grads_fcim_diagonal"](
            classical_weights, probabilities, features, TEMPERATURE
        ),
        feature_setup + probability_setup,
        sum(probability.nbytes for probability in probabilities),
        features.nbytes,
        notes="Exact diagonal calculation; no eigendecomposition.",
    )

    del states, vectors
    gc.collect()
    return rows


def run_benchmarks(quick=False):
    original = load_original_namespace()
    penny = load_pennylane_namespace()
    rows = []

    sample_values = (16, 64, 256) if quick else (16, 64, 256, 1000)
    for samples in sample_values:
        rows.extend(
            benchmark_case(
                original,
                penny,
                n=4,
                samples=samples,
                sweep="samples_at_n4",
                include_original=True,
                repeats=2 if quick else 3,
            )
        )

    qubit_values = (2, 3, 4, 5) if quick else (2, 3, 4, 5, 6)
    for n in qubit_values:
        rows.extend(
            benchmark_case(
                original,
                penny,
                n=n,
                samples=32,
                sweep="qubits_at_m32",
                include_original=True,
                repeats=2 if quick else 3,
            )
        )

    # Larger optimized-only cases demonstrate where the original dense path is
    # intentionally no longer run. Keep M small so this benchmark remains CI-safe.
    if not quick:
        for n in (7, 8, 9, 10):
            rows.extend(
                benchmark_case(
                    original,
                    penny,
                    n=n,
                    samples=16,
                    sweep="optimized_qubits_at_m16",
                    include_original=False,
                    repeats=2,
                )
            )

    output_path = QUICK_RESULTS_PATH if quick else RESULTS_PATH
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows, output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use smaller sweeps")
    arguments = parser.parse_args()
    rows, output_path = run_benchmarks(quick=arguments.quick)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
