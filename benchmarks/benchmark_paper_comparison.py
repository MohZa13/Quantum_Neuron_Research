"""Paper-style comparison data for the original and PennyLane log-loss notebooks.

This script creates three CSV files for visual comparisons at selected qubit
counts, defaulting to n = 2, 4, 7:

* paper_training_curves_2_4_7.csv
* paper_efficiency_summary_2_4_7.csv
* paper_sampling_efficiency_2_4_7.csv

For qubit counts above the default full-original threshold, the script first
times a short original-dense probe. If the extrapolated full original run is
under ``--full-original-threshold-seconds`` (300 seconds by default), it runs the
full original comparison; otherwise it keeps the original as a measured probe.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import time
from pathlib import Path

import numpy as np

from notebook_test_utils import (
    REPO_ROOT,
    csr_payload_bytes,
    densities,
    fused_payload_bytes,
    load_original_namespace,
    load_pennylane_namespace,
    original_loss_and_gradient,
    random_state_vectors,
)


OUTPUT_DIRECTORY = REPO_ROOT / "benchmarks"
TEMPERATURE = 2.0
DEFAULT_QUBITS = (2, 4, 7)
DEFAULT_SAMPLE_GRID = (16, 64, 256, 1000)


def seconds_now() -> float:
    return time.perf_counter()


def elapsed_since(start: float) -> float:
    return time.perf_counter() - start


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def dense_hamiltonian(weights: np.ndarray, terms: list[np.ndarray]) -> np.ndarray:
    return sum(
        (weight * term for weight, term in zip(weights, terms)),
        start=np.zeros_like(terms[0]),
    )


def dense_energy_labels(
    vectors: np.ndarray, hamiltonian: np.ndarray
) -> np.ndarray:
    energies = np.real(
        np.einsum("bi,ij,bj->b", vectors.conj(), hamiltonian, vectors, optimize=True)
    )
    labels = np.sign(energies)
    labels[labels == 0] = 1
    return labels.astype(float)


def dense_accuracy(
    weights: np.ndarray,
    terms: list[np.ndarray],
    vectors: np.ndarray,
    labels: np.ndarray,
) -> float:
    hamiltonian = dense_hamiltonian(weights, terms)
    predictions = dense_energy_labels(vectors, hamiltonian)
    return float(np.mean(predictions == labels) * 100.0)


def make_case_data(
    original: dict,
    n: int,
    samples: int,
    validation_samples: int,
    seed: int,
):
    """Create one controlled dataset shared by all implementations."""
    rng = np.random.default_rng(seed + 10_000 * n + samples)
    train_vectors = random_state_vectors(rng, n, samples)
    validation_vectors = random_state_vectors(rng, n, validation_samples)
    train_states = densities(train_vectors)
    validation_states = densities(validation_vectors)

    dense_quantum = original["generate_paulis"](n, "quantum")
    dense_classical = original["generate_paulis"](n, "classical")

    target_weights = rng.uniform(-2.0, 2.0, len(dense_quantum))
    target_hamiltonian = dense_hamiltonian(target_weights, dense_quantum)
    train_labels = dense_energy_labels(train_vectors, target_hamiltonian)
    validation_labels = dense_energy_labels(validation_vectors, target_hamiltonian)

    init_quantum = rng.uniform(-0.5, 0.5, len(dense_quantum))
    init_classical = rng.uniform(-0.5, 0.5, len(dense_classical))

    return {
        "rng": rng,
        "train_vectors": train_vectors,
        "validation_vectors": validation_vectors,
        "train_states": train_states,
        "validation_states": validation_states,
        "train_labels": train_labels,
        "validation_labels": validation_labels,
        "dense_quantum": dense_quantum,
        "dense_classical": dense_classical,
        "target_weights": target_weights,
        "init_quantum": init_quantum,
        "init_classical": init_classical,
    }


def record_curve(
    rows: list[dict],
    *,
    n: int,
    samples: int,
    validation_samples: int,
    requested_epochs: int,
    epoch: int,
    method: str,
    model: str,
    loss: float,
    accuracy: float | None,
    epoch_seconds: float,
    measured: bool,
    notes: str = "",
):
    rows.append(
        {
            "n": n,
            "samples": samples,
            "validation_samples": validation_samples,
            "requested_epochs": requested_epochs,
            "epoch": epoch,
            "method": method,
            "model": model,
            "loss": float(loss),
            "validation_accuracy_pct": (
                "" if accuracy is None else float(accuracy)
            ),
            "epoch_seconds": float(epoch_seconds),
            "measured": bool(measured),
            "notes": notes,
        }
    )


def record_summary(
    rows: list[dict],
    *,
    n: int,
    samples: int,
    measured_samples: int,
    validation_samples: int,
    requested_epochs: int,
    completed_epochs: int,
    method: str,
    model: str,
    optimizer: str,
    setup_seconds: float,
    measured_train_seconds: float,
    median_epoch_seconds: float,
    estimated_total_seconds: float,
    static_bytes: int,
    final_loss: float,
    final_accuracy: float,
    loss_error_vs_dense: float | str = "",
    gradient_error_vs_dense: float | str = "",
    measured_full_training: bool = True,
    notes: str = "",
):
    rows.append(
        {
            "n": n,
            "samples": samples,
            "measured_samples": measured_samples,
            "validation_samples": validation_samples,
            "requested_epochs": requested_epochs,
            "completed_epochs": completed_epochs,
            "method": method,
            "model": model,
            "optimizer": optimizer,
            "setup_seconds": float(setup_seconds),
            "measured_train_seconds": float(measured_train_seconds),
            "median_epoch_seconds": float(median_epoch_seconds),
            "estimated_total_seconds": float(estimated_total_seconds),
            "static_bytes": int(static_bytes),
            "final_loss": float(final_loss),
            "final_validation_accuracy_pct": float(final_accuracy),
            "loss_error_vs_dense": loss_error_vs_dense,
            "gradient_error_vs_dense": gradient_error_vs_dense,
            "measured_full_training": bool(measured_full_training),
            "notes": notes,
        }
    )


def initial_error_metrics(
    original: dict,
    penny: dict,
    n: int,
    data: dict,
    sparse_quantum,
    symbolic_quantum,
    fused_quantum,
    aggregates,
    chebyshev_degree: int,
) -> dict:
    """Compare the optimized paths to the dense objective at identical weights."""
    init_weights = data["init_quantum"]
    dense_loss, dense_gradient = original_loss_and_gradient(
        original,
        init_weights,
        data["train_states"],
        data["train_labels"],
        data["dense_quantum"],
        TEMPERATURE,
    )
    exact_loss, exact_gradient = penny["compute_loss_and_grads_aggregated_symbolic"](
        init_weights,
        aggregates,
        symbolic_quantum,
        TEMPERATURE,
        n,
        sparse_quantum,
    )
    bound = max(1.05 * float(np.sum(np.abs(init_weights))), 1e-6)
    cheb_loss, cheb_gradient = penny["compute_loss_and_grads_chebyshev"](
        init_weights,
        data["train_vectors"].astype(np.complex64),
        data["train_labels"],
        fused_quantum,
        TEMPERATURE,
        degree=chebyshev_degree,
        chunk_size=min(32, len(data["train_vectors"])),
        spectral_bound=bound,
        complex_dtype=np.complex64,
        term_chunk_size=8,
    )
    return {
        "dense_loss": float(dense_loss),
        "dense_gradient": dense_gradient,
        "exact_loss_error": float(abs(exact_loss - dense_loss)),
        "exact_gradient_error": float(np.max(np.abs(exact_gradient - dense_gradient))),
        "cheb_loss_error": float(abs(float(cheb_loss) - float(exact_loss))),
        "cheb_gradient_error": float(
            np.max(np.abs(np.asarray(cheb_gradient) - exact_gradient))
        ),
    }


def prepare_pennylane_representations(
    penny: dict,
    n: int,
    data: dict,
):
    setup_start = seconds_now()
    symbolic_quantum = penny["generate_paulis_symbolic"](n, "quantum")
    sparse_quantum = penny["precompute_sparse_paulis_direct"](n, "quantum")
    aggregates = penny["aggregate_states_by_label"](
        data["train_vectors"], data["train_labels"]
    )
    exact_setup = elapsed_since(setup_start)

    cheb_start = seconds_now()
    fused_quantum = penny["build_fused_pauli_kernel"](
        n, "quantum", complex_dtype="complex64"
    )
    cheb_setup = elapsed_since(cheb_start)

    fcim_start = seconds_now()
    fcim_features = penny["build_fcim_feature_matrix"](n)
    basis_probabilities = penny["aggregate_basis_probabilities_by_label"](
        data["train_vectors"], data["train_labels"]
    )
    fcim_setup = elapsed_since(fcim_start)

    initial_bound = max(1.05 * float(np.sum(np.abs(data["init_quantum"]))), 1e-6)
    chebyshev_degree = penny["select_adaptive_chebyshev_degree"](
        TEMPERATURE,
        initial_bound,
        tolerance=1e-5,
        min_degree=8,
        max_degree=64,
    )

    return {
        "symbolic_quantum": symbolic_quantum,
        "sparse_quantum": sparse_quantum,
        "aggregates": aggregates,
        "exact_setup_seconds": exact_setup,
        "fused_quantum": fused_quantum,
        "cheb_setup_seconds": cheb_setup,
        "chebyshev_degree": chebyshev_degree,
        "fcim_features": fcim_features,
        "basis_probabilities": basis_probabilities,
        "fcim_setup_seconds": fcim_setup,
    }


def estimate_original_dense_training_seconds(
    original: dict,
    data: dict,
    *,
    requested_epochs: int,
    target_samples: int,
    probe_epochs: int,
    learning_rate: float,
) -> dict:
    """Estimate full original dense runtime from a short measured probe."""
    quantum_weights = data["init_quantum"].copy()
    classical_weights = data["init_classical"].copy()
    quantum_times: list[float] = []
    classical_times: list[float] = []

    for _ in range(max(1, probe_epochs)):
        quantum_start = seconds_now()
        _, quantum_gradient = original_loss_and_gradient(
            original,
            quantum_weights,
            data["train_states"],
            data["train_labels"],
            data["dense_quantum"],
            TEMPERATURE,
        )
        quantum_times.append(elapsed_since(quantum_start))
        quantum_weights -= learning_rate * quantum_gradient

        classical_start = seconds_now()
        _, classical_gradient = original_loss_and_gradient(
            original,
            classical_weights,
            data["train_states"],
            data["train_labels"],
            data["dense_classical"],
            TEMPERATURE,
        )
        classical_times.append(elapsed_since(classical_start))
        classical_weights -= learning_rate * classical_gradient

    measured_samples = len(data["train_vectors"])
    sample_scale = target_samples / measured_samples
    quantum_epoch = median(quantum_times)
    classical_epoch = median(classical_times)
    combined_epoch = quantum_epoch + classical_epoch
    return {
        "measured_samples": measured_samples,
        "target_samples": target_samples,
        "probe_epochs": max(1, probe_epochs),
        "quantum_epoch_seconds": quantum_epoch,
        "classical_epoch_seconds": classical_epoch,
        "combined_epoch_seconds": combined_epoch,
        "sample_scale": sample_scale,
        "estimated_total_seconds": combined_epoch * sample_scale * requested_epochs,
    }


def train_original_dense(
    original: dict,
    n: int,
    data: dict,
    requested_epochs: int,
    epochs_to_run: int,
    learning_rate: float,
    curve_rows: list[dict],
    summary_rows: list[dict],
    target_samples_for_estimate: int | None = None,
):
    method = "original_dense" if epochs_to_run >= requested_epochs else "original_dense_probe"
    notes = (
        "Full dense notebook-style training."
        if method == "original_dense"
        else "Short dense timing probe; total training time is extrapolated."
    )
    quantum_weights = data["init_quantum"].copy()
    classical_weights = data["init_classical"].copy()
    quantum_times: list[float] = []
    classical_times: list[float] = []
    final_quantum_loss = math.nan
    final_classical_loss = math.nan

    for epoch in range(epochs_to_run):
        quantum_start = seconds_now()
        final_quantum_loss, quantum_gradient = original_loss_and_gradient(
            original,
            quantum_weights,
            data["train_states"],
            data["train_labels"],
            data["dense_quantum"],
            TEMPERATURE,
        )
        quantum_seconds = elapsed_since(quantum_start)
        quantum_times.append(quantum_seconds)
        quantum_accuracy = dense_accuracy(
            quantum_weights,
            data["dense_quantum"],
            data["validation_vectors"],
            data["validation_labels"],
        )
        record_curve(
            curve_rows,
            n=n,
            samples=len(data["train_vectors"]),
            validation_samples=len(data["validation_vectors"]),
            requested_epochs=requested_epochs,
            epoch=epoch,
            method=method,
            model="quantum",
            loss=final_quantum_loss,
            accuracy=quantum_accuracy,
            epoch_seconds=quantum_seconds,
            measured=True,
            notes=notes,
        )
        quantum_weights -= learning_rate * quantum_gradient

        classical_start = seconds_now()
        final_classical_loss, classical_gradient = original_loss_and_gradient(
            original,
            classical_weights,
            data["train_states"],
            data["train_labels"],
            data["dense_classical"],
            TEMPERATURE,
        )
        classical_seconds = elapsed_since(classical_start)
        classical_times.append(classical_seconds)
        classical_accuracy = dense_accuracy(
            classical_weights,
            data["dense_classical"],
            data["validation_vectors"],
            data["validation_labels"],
        )
        record_curve(
            curve_rows,
            n=n,
            samples=len(data["train_vectors"]),
            validation_samples=len(data["validation_vectors"]),
            requested_epochs=requested_epochs,
            epoch=epoch,
            method=method,
            model="classical",
            loss=final_classical_loss,
            accuracy=classical_accuracy,
            epoch_seconds=classical_seconds,
            measured=True,
            notes=notes,
        )
        classical_weights -= learning_rate * classical_gradient

    dense_quantum_bytes = sum(term.nbytes for term in data["dense_quantum"])
    dense_classical_bytes = sum(term.nbytes for term in data["dense_classical"])
    actual_samples = len(data["train_vectors"])
    target_samples = target_samples_for_estimate or actual_samples
    sample_scale = target_samples / actual_samples
    bytes_per_dense_state = data["train_states"].nbytes // actual_samples
    train_state_bytes_estimate = bytes_per_dense_state * target_samples
    measured_full = epochs_to_run >= requested_epochs
    estimate_note = ""
    if sample_scale != 1:
        estimate_note = (
            f" Training-time and memory estimates are scaled from "
            f"{actual_samples} measured samples to {target_samples} requested samples."
        )

    quantum_epoch = median(quantum_times)
    classical_epoch = median(classical_times)
    record_summary(
        summary_rows,
        n=n,
        samples=target_samples,
        measured_samples=actual_samples,
        validation_samples=len(data["validation_vectors"]),
        requested_epochs=requested_epochs,
        completed_epochs=epochs_to_run,
        method=method,
        model="quantum",
        optimizer="gd",
        setup_seconds=0.0,
        measured_train_seconds=sum(quantum_times),
        median_epoch_seconds=quantum_epoch,
        estimated_total_seconds=quantum_epoch * sample_scale * requested_epochs,
        static_bytes=train_state_bytes_estimate + dense_quantum_bytes,
        final_loss=final_quantum_loss,
        final_accuracy=dense_accuracy(
            quantum_weights,
            data["dense_quantum"],
            data["validation_vectors"],
            data["validation_labels"],
        ),
        measured_full_training=measured_full,
        notes=notes
        + estimate_note
        + " Gradient uses the corrected dense notebook derivative.",
    )
    record_summary(
        summary_rows,
        n=n,
        samples=target_samples,
        measured_samples=actual_samples,
        validation_samples=len(data["validation_vectors"]),
        requested_epochs=requested_epochs,
        completed_epochs=epochs_to_run,
        method=method,
        model="classical",
        optimizer="gd",
        setup_seconds=0.0,
        measured_train_seconds=sum(classical_times),
        median_epoch_seconds=classical_epoch,
        estimated_total_seconds=classical_epoch * sample_scale * requested_epochs,
        static_bytes=train_state_bytes_estimate + dense_classical_bytes,
        final_loss=final_classical_loss,
        final_accuracy=dense_accuracy(
            classical_weights,
            data["dense_classical"],
            data["validation_vectors"],
            data["validation_labels"],
        ),
        measured_full_training=measured_full,
        notes=notes
        + estimate_note
        + " Dense FCIM eigendecomposition is used even though the model is diagonal.",
    )


def train_pennylane_exact(
    penny: dict,
    n: int,
    data: dict,
    reps: dict,
    requested_epochs: int,
    learning_rate: float,
    curve_rows: list[dict],
    summary_rows: list[dict],
    errors: dict,
):
    quantum_weights = data["init_quantum"].copy()
    classical_weights = data["init_classical"].copy()
    quantum_times: list[float] = []
    classical_times: list[float] = []
    final_quantum_loss = math.nan
    final_classical_loss = math.nan

    for epoch in range(requested_epochs):
        quantum_start = seconds_now()
        final_quantum_loss, quantum_gradient = penny[
            "compute_loss_and_grads_aggregated_symbolic"
        ](
            quantum_weights,
            reps["aggregates"],
            reps["symbolic_quantum"],
            TEMPERATURE,
            n,
            reps["sparse_quantum"],
        )
        quantum_seconds = elapsed_since(quantum_start)
        quantum_times.append(quantum_seconds)
        quantum_accuracy = penny["calculate_accuracy_matrix_free"](
            quantum_weights,
            reps["sparse_quantum"],
            data["validation_vectors"],
            data["validation_labels"],
            "numpy",
            np.complex128,
            32,
            8,
        )
        record_curve(
            curve_rows,
            n=n,
            samples=len(data["train_vectors"]),
            validation_samples=len(data["validation_vectors"]),
            requested_epochs=requested_epochs,
            epoch=epoch,
            method="penny_exact_aggregate",
            model="quantum",
            loss=final_quantum_loss,
            accuracy=quantum_accuracy,
            epoch_seconds=quantum_seconds,
            measured=True,
            notes="Exact PennyLane symbolic/CSR path with label aggregation.",
        )
        quantum_weights -= learning_rate * quantum_gradient

        classical_start = seconds_now()
        final_classical_loss, classical_gradient = penny[
            "compute_loss_and_grads_fcim_diagonal"
        ](
            classical_weights,
            reps["basis_probabilities"],
            reps["fcim_features"],
            TEMPERATURE,
        )
        classical_seconds = elapsed_since(classical_start)
        classical_times.append(classical_seconds)
        classical_accuracy = penny["calculate_fcim_accuracy"](
            classical_weights,
            reps["fcim_features"],
            data["validation_vectors"],
            data["validation_labels"],
            "numpy",
            np.complex128,
        )
        record_curve(
            curve_rows,
            n=n,
            samples=len(data["train_vectors"]),
            validation_samples=len(data["validation_vectors"]),
            requested_epochs=requested_epochs,
            epoch=epoch,
            method="penny_diagonal_fcim",
            model="classical",
            loss=final_classical_loss,
            accuracy=classical_accuracy,
            epoch_seconds=classical_seconds,
            measured=True,
            notes="Exact diagonal FCIM path; no dense eigendecomposition.",
        )
        classical_weights -= learning_rate * classical_gradient

    aggregate_bytes = sum(aggregate.nbytes for aggregate in reps["aggregates"])
    sparse_bytes = csr_payload_bytes(reps["sparse_quantum"])
    probability_bytes = sum(p.nbytes for p in reps["basis_probabilities"])
    feature_bytes = reps["fcim_features"].nbytes
    quantum_epoch = median(quantum_times)
    classical_epoch = median(classical_times)

    record_summary(
        summary_rows,
        n=n,
        samples=len(data["train_vectors"]),
        measured_samples=len(data["train_vectors"]),
        validation_samples=len(data["validation_vectors"]),
        requested_epochs=requested_epochs,
        completed_epochs=requested_epochs,
        method="penny_exact_aggregate",
        model="quantum",
        optimizer="gd",
        setup_seconds=reps["exact_setup_seconds"],
        measured_train_seconds=sum(quantum_times),
        median_epoch_seconds=quantum_epoch,
        estimated_total_seconds=reps["exact_setup_seconds"]
        + quantum_epoch * requested_epochs,
        static_bytes=aggregate_bytes + sparse_bytes,
        final_loss=final_quantum_loss,
        final_accuracy=penny["calculate_accuracy_matrix_free"](
            quantum_weights,
            reps["sparse_quantum"],
            data["validation_vectors"],
            data["validation_labels"],
            "numpy",
            np.complex128,
            32,
            8,
        ),
        loss_error_vs_dense=errors["exact_loss_error"],
        gradient_error_vs_dense=errors["exact_gradient_error"],
        notes="Exact optimized objective and gradient.",
    )
    record_summary(
        summary_rows,
        n=n,
        samples=len(data["train_vectors"]),
        measured_samples=len(data["train_vectors"]),
        validation_samples=len(data["validation_vectors"]),
        requested_epochs=requested_epochs,
        completed_epochs=requested_epochs,
        method="penny_diagonal_fcim",
        model="classical",
        optimizer="gd",
        setup_seconds=reps["fcim_setup_seconds"],
        measured_train_seconds=sum(classical_times),
        median_epoch_seconds=classical_epoch,
        estimated_total_seconds=reps["fcim_setup_seconds"]
        + classical_epoch * requested_epochs,
        static_bytes=probability_bytes + feature_bytes,
        final_loss=final_classical_loss,
        final_accuracy=penny["calculate_fcim_accuracy"](
            classical_weights,
            reps["fcim_features"],
            data["validation_vectors"],
            data["validation_labels"],
            "numpy",
            np.complex128,
        ),
        notes="Exact optimized FCIM objective using the diagonal basis.",
    )


def train_pennylane_chebyshev(
    penny: dict,
    n: int,
    data: dict,
    reps: dict,
    requested_epochs: int,
    learning_rate: float,
    curve_rows: list[dict],
    summary_rows: list[dict],
    errors: dict,
):
    quantum_weights = data["init_quantum"].copy()
    vectors64 = data["train_vectors"].astype(np.complex64)
    quantum_times: list[float] = []
    degrees: list[int] = []
    final_quantum_loss = math.nan

    for epoch in range(requested_epochs):
        bound = max(1.05 * float(np.sum(np.abs(quantum_weights))), 1e-6)
        degree = penny["select_adaptive_chebyshev_degree"](
            TEMPERATURE,
            bound,
            tolerance=1e-5,
            min_degree=8,
            max_degree=64,
        )
        degrees.append(int(degree))
        quantum_start = seconds_now()
        final_quantum_loss, quantum_gradient = penny["compute_loss_and_grads_chebyshev"](
            quantum_weights,
            vectors64,
            data["train_labels"],
            reps["fused_quantum"],
            TEMPERATURE,
            degree=degree,
            chunk_size=min(32, len(vectors64)),
            spectral_bound=bound,
            complex_dtype=np.complex64,
            term_chunk_size=8,
        )
        quantum_seconds = elapsed_since(quantum_start)
        quantum_times.append(quantum_seconds)
        quantum_accuracy = penny["calculate_accuracy_matrix_free"](
            quantum_weights,
            reps["fused_quantum"],
            data["validation_vectors"].astype(np.complex64),
            data["validation_labels"],
            "numpy",
            np.complex64,
            32,
            8,
        )
        record_curve(
            curve_rows,
            n=n,
            samples=len(data["train_vectors"]),
            validation_samples=len(data["validation_vectors"]),
            requested_epochs=requested_epochs,
            epoch=epoch,
            method="penny_chebyshev_matrix_free",
            model="quantum",
            loss=final_quantum_loss,
            accuracy=quantum_accuracy,
            epoch_seconds=quantum_seconds,
            measured=True,
            notes=(
                "Approximate matrix-free Chebyshev path; adaptive degree="
                f"{degree}."
            ),
        )
        quantum_weights -= learning_rate * np.asarray(quantum_gradient)

    quantum_epoch = median(quantum_times)
    record_summary(
        summary_rows,
        n=n,
        samples=len(data["train_vectors"]),
        measured_samples=len(data["train_vectors"]),
        validation_samples=len(data["validation_vectors"]),
        requested_epochs=requested_epochs,
        completed_epochs=requested_epochs,
        method="penny_chebyshev_matrix_free",
        model="quantum",
        optimizer="gd",
        setup_seconds=reps["cheb_setup_seconds"],
        measured_train_seconds=sum(quantum_times),
        median_epoch_seconds=quantum_epoch,
        estimated_total_seconds=reps["cheb_setup_seconds"]
        + quantum_epoch * requested_epochs,
        static_bytes=vectors64.nbytes + fused_payload_bytes(reps["fused_quantum"]),
        final_loss=float(final_quantum_loss),
        final_accuracy=penny["calculate_accuracy_matrix_free"](
            quantum_weights,
            reps["fused_quantum"],
            data["validation_vectors"].astype(np.complex64),
            data["validation_labels"],
            "numpy",
            np.complex64,
            32,
            8,
        ),
        loss_error_vs_dense=errors["cheb_loss_error"],
        gradient_error_vs_dense=errors["cheb_gradient_error"],
        notes=(
            "Approximate matrix-free path. Degree range observed during training: "
            f"{min(degrees)}-{max(degrees)}."
        ),
    )


def benchmark_sample_efficiency(
    original: dict,
    penny: dict,
    qubits: tuple[int, ...],
    sample_grid: tuple[int, ...],
    validation_samples: int,
    seed: int,
    max_full_original_n: int,
    original_probe_samples: int,
) -> list[dict]:
    rows: list[dict] = []
    original_estimates: dict[int, dict[str, float]] = {}

    for n in qubits:
        for samples in sample_grid:
            data = make_case_data(
                original,
                n,
                samples,
                validation_samples=max(8, min(validation_samples, 64)),
                seed=seed + 500_000,
            )
            reps = prepare_pennylane_representations(penny, n, data)

            # Original dense: measure small/allowed cases, extrapolate otherwise.
            can_measure_original = (
                n <= max_full_original_n or samples <= original_probe_samples
            )
            if can_measure_original:
                dense_start = seconds_now()
                q_loss, _ = original_loss_and_gradient(
                    original,
                    data["init_quantum"],
                    data["train_states"],
                    data["train_labels"],
                    data["dense_quantum"],
                    TEMPERATURE,
                )
                dense_seconds = elapsed_since(dense_start)
                original_estimates[n] = {
                    "samples": float(samples),
                    "seconds": float(dense_seconds),
                }
                measured = True
                notes = "Measured dense quantum loss/gradient pass."
            else:
                estimate = original_estimates.get(n)
                if estimate is None:
                    # Fallback: measure a tiny probe for this n and scale linearly
                    # with sample count. The dense method is empirically linear in
                    # samples once H is fixed.
                    probe_data = make_case_data(
                        original,
                        n,
                        original_probe_samples,
                        validation_samples=8,
                        seed=seed + 700_000,
                    )
                    probe_start = seconds_now()
                    original_loss_and_gradient(
                        original,
                        probe_data["init_quantum"],
                        probe_data["train_states"],
                        probe_data["train_labels"],
                        probe_data["dense_quantum"],
                        TEMPERATURE,
                    )
                    probe_seconds = elapsed_since(probe_start)
                    estimate = {
                        "samples": float(original_probe_samples),
                        "seconds": float(probe_seconds),
                    }
                    original_estimates[n] = estimate
                    del probe_data
                dense_seconds = estimate["seconds"] * samples / estimate["samples"]
                q_loss = math.nan
                measured = False
                notes = (
                    "Estimated from a measured dense probe and linear sample scaling."
                )

            rows.append(
                {
                    "n": n,
                    "samples": samples,
                    "method": "original_dense",
                    "model": "quantum",
                    "pass_seconds": float(dense_seconds),
                    "setup_seconds": 0.0,
                    "static_bytes": int(
                        data["train_states"].nbytes
                        + sum(term.nbytes for term in data["dense_quantum"])
                    ),
                    "loss": "" if math.isnan(q_loss) else float(q_loss),
                    "measured": measured,
                    "notes": notes,
                }
            )

            exact_start = seconds_now()
            exact_loss, _ = penny["compute_loss_and_grads_aggregated_symbolic"](
                data["init_quantum"],
                reps["aggregates"],
                reps["symbolic_quantum"],
                TEMPERATURE,
                n,
                reps["sparse_quantum"],
            )
            exact_seconds = elapsed_since(exact_start)
            rows.append(
                {
                    "n": n,
                    "samples": samples,
                    "method": "penny_exact_aggregate",
                    "model": "quantum",
                    "pass_seconds": float(exact_seconds),
                    "setup_seconds": float(reps["exact_setup_seconds"]),
                    "static_bytes": int(
                        sum(aggregate.nbytes for aggregate in reps["aggregates"])
                        + csr_payload_bytes(reps["sparse_quantum"])
                    ),
                    "loss": float(exact_loss),
                    "measured": True,
                    "notes": "Measured exact aggregate quantum pass.",
                }
            )

            bound = max(1.05 * float(np.sum(np.abs(data["init_quantum"]))), 1e-6)
            degree = penny["select_adaptive_chebyshev_degree"](
                TEMPERATURE,
                bound,
                tolerance=1e-5,
                min_degree=8,
                max_degree=64,
            )
            cheb_start = seconds_now()
            cheb_loss, _ = penny["compute_loss_and_grads_chebyshev"](
                data["init_quantum"],
                data["train_vectors"].astype(np.complex64),
                data["train_labels"],
                reps["fused_quantum"],
                TEMPERATURE,
                degree=degree,
                chunk_size=min(32, samples),
                spectral_bound=bound,
                complex_dtype=np.complex64,
                term_chunk_size=8,
            )
            cheb_seconds = elapsed_since(cheb_start)
            rows.append(
                {
                    "n": n,
                    "samples": samples,
                    "method": "penny_chebyshev_matrix_free",
                    "model": "quantum",
                    "pass_seconds": float(cheb_seconds),
                    "setup_seconds": float(reps["cheb_setup_seconds"]),
                    "static_bytes": int(
                        data["train_vectors"].astype(np.complex64).nbytes
                        + fused_payload_bytes(reps["fused_quantum"])
                    ),
                    "loss": float(cheb_loss),
                    "measured": True,
                    "notes": f"Measured matrix-free Chebyshev pass; degree={degree}.",
                }
            )

            del data, reps
            gc.collect()

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect paper-style training, timing, memory, and error data for "
            "logloss_nqubits vs logloss_nqubits_pennylane."
        )
    )
    parser.add_argument("--qubits", nargs="+", type=int, default=list(DEFAULT_QUBITS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-full-original-n",
        type=int,
        default=4,
        help=(
            "Run full original dense training through this qubit count. Larger "
            "n values use timing probes unless this threshold is raised."
        ),
    )
    parser.add_argument("--original-probe-epochs", type=int, default=2)
    parser.add_argument(
        "--original-probe-samples",
        type=int,
        default=16,
        help="Maximum samples used for dense n>max-full-original-n probes.",
    )
    parser.add_argument(
        "--full-original-threshold-seconds",
        type=float,
        default=300.0,
        help=(
            "For n>max-full-original-n, run a short original-dense probe first. "
            "If the extrapolated complete original run is at or below this many "
            "seconds, run the full original instead of keeping it as a probe. "
            "Use 0 to always keep the probe for large n."
        ),
    )
    parser.add_argument(
        "--sample-grid",
        nargs="+",
        type=int,
        default=list(DEFAULT_SAMPLE_GRID),
    )
    parser.add_argument(
        "--skip-sample-sweep",
        action="store_true",
        help="Only generate training curves and summary rows.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help=(
            "Use paper-style scale: 500 epochs and 1000 training states. This "
            "can take much longer, especially for original dense n=4."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.paper:
        args.epochs = 500
        args.samples = 1000
        args.validation_samples = 500

    qubits = tuple(args.qubits)
    suffix = "_".join(str(n) for n in qubits)
    curves_path = OUTPUT_DIRECTORY / f"paper_training_curves_{suffix}.csv"
    summary_path = OUTPUT_DIRECTORY / f"paper_efficiency_summary_{suffix}.csv"
    sampling_path = OUTPUT_DIRECTORY / f"paper_sampling_efficiency_{suffix}.csv"

    print("Loading notebook definitions...")
    original = load_original_namespace()
    penny = load_pennylane_namespace()

    curve_rows: list[dict] = []
    summary_rows: list[dict] = []

    for n in qubits:
        print(f"\n=== n={n}: controlled training comparison ===")
        data = make_case_data(
            original,
            n,
            samples=args.samples,
            validation_samples=args.validation_samples,
            seed=args.seed,
        )
        reps = prepare_pennylane_representations(penny, n, data)

        original_data = data
        original_epochs = args.epochs
        error_data = data
        error_reps = reps
        probe_data = None
        if n > args.max_full_original_n:
            probe_samples = min(args.samples, args.original_probe_samples)
            probe_data = (
                data
                if probe_samples == args.samples
                else make_case_data(
                    original,
                    n,
                    samples=probe_samples,
                    validation_samples=args.validation_samples,
                    seed=args.seed + 123_000,
                )
            )
            estimate = estimate_original_dense_training_seconds(
                original,
                probe_data,
                requested_epochs=args.epochs,
                target_samples=args.samples,
                probe_epochs=args.original_probe_epochs,
                learning_rate=args.learning_rate,
            )
            print(
                "Original dense probe estimate: "
                f"{estimate['estimated_total_seconds']:.2f}s combined "
                f"({estimate['measured_samples']} measured samples -> "
                f"{estimate['target_samples']} requested samples, "
                f"{args.epochs} epochs)."
            )
            should_run_full_original = (
                args.full_original_threshold_seconds > 0
                and estimate["estimated_total_seconds"]
                <= args.full_original_threshold_seconds
            )
            if should_run_full_original:
                print(
                    "Estimate is within "
                    f"{args.full_original_threshold_seconds:.0f}s; running full "
                    "original dense comparison."
                )
            else:
                print(
                    "Estimate exceeds "
                    f"{args.full_original_threshold_seconds:.0f}s; keeping the "
                    "original dense path as a measured probe."
                )
                original_data = probe_data
                original_epochs = args.original_probe_epochs
                error_data = probe_data
                error_reps = (
                    reps
                    if probe_data is data
                    else prepare_pennylane_representations(penny, n, error_data)
                )

        errors = initial_error_metrics(
            original,
            penny,
            n,
            error_data,
            error_reps["sparse_quantum"],
            error_reps["symbolic_quantum"],
            error_reps["fused_quantum"],
            error_reps["aggregates"],
            error_reps["chebyshev_degree"],
        )

        print(
            f"Original dense: running {original_epochs}/{args.epochs} epochs "
            f"with {len(original_data['train_vectors'])} measured samples."
        )
        train_original_dense(
            original,
            n,
            original_data,
            requested_epochs=args.epochs,
            epochs_to_run=original_epochs,
            learning_rate=args.learning_rate,
            curve_rows=curve_rows,
            summary_rows=summary_rows,
            target_samples_for_estimate=args.samples,
        )

        print(
            f"PennyLane exact/diagonal: running {args.epochs} epochs "
            f"with {len(data['train_vectors'])} samples."
        )
        train_pennylane_exact(
            penny,
            n,
            data,
            reps,
            requested_epochs=args.epochs,
            learning_rate=args.learning_rate,
            curve_rows=curve_rows,
            summary_rows=summary_rows,
            errors=errors,
        )

        print(
            f"PennyLane Chebyshev: running {args.epochs} epochs "
            f"with {len(data['train_vectors'])} samples."
        )
        train_pennylane_chebyshev(
            penny,
            n,
            data,
            reps,
            requested_epochs=args.epochs,
            learning_rate=args.learning_rate,
            curve_rows=curve_rows,
            summary_rows=summary_rows,
            errors=errors,
        )

        cleanup_original_data = original_data is not data
        cleanup_error_reps = error_reps is not reps
        cleanup_probe_data = (
            probe_data is not None and probe_data is not data
            and probe_data is not original_data and probe_data is not error_data)
        if cleanup_original_data:
            del original_data
        if cleanup_error_reps:
            del error_reps
        if cleanup_probe_data:
            del probe_data
        del data, reps
        gc.collect()

    write_csv(curves_path, curve_rows)
    write_csv(summary_path, summary_rows)
    print(f"\nWrote {len(curve_rows)} training-curve rows to {curves_path}")
    print(f"Wrote {len(summary_rows)} summary rows to {summary_path}")

    if not args.skip_sample_sweep:
        print("\n=== Sampling-efficiency sweep ===")
        sampling_rows = benchmark_sample_efficiency(
            original,
            penny,
            qubits,
            tuple(args.sample_grid),
            validation_samples=args.validation_samples,
            seed=args.seed,
            max_full_original_n=args.max_full_original_n,
            original_probe_samples=args.original_probe_samples,
        )
        write_csv(sampling_path, sampling_rows)
        print(f"Wrote {len(sampling_rows)} sampling rows to {sampling_path}")


if __name__ == "__main__":
    main()
