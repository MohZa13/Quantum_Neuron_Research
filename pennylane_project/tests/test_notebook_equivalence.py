"""Correctness and integration comparisons for both log-loss notebooks."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from notebook_test_utils import (
    densities,
    load_original_namespace,
    load_pennylane_namespace,
    original_loss_and_gradient,
    random_state_vectors,
)


@contextlib.contextmanager
def temporary_working_directory():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "outputs").mkdir()
        os.chdir(path)
        try:
            yield path
        finally:
            os.chdir(previous)


class NotebookEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = load_original_namespace()
        cls.penny = load_pennylane_namespace()

    def setUp(self):
        self.rng = np.random.default_rng(20260622)

    def test_parameter_counts_match_paper_models(self):
        for n in range(1, 6):
            with self.subTest(n=n, model="quantum"):
                self.assertEqual(len(self.original["generate_paulis"](n, "quantum")), 6 * n - 3)
                self.assertEqual(len(self.penny["generate_paulis_symbolic"](n, "quantum")), 6 * n - 3)
            with self.subTest(n=n, model="classical"):
                expected = n * (n + 1) // 2
                self.assertEqual(len(self.original["generate_paulis"](n, "classical")), expected)
                self.assertEqual(len(self.penny["generate_paulis_symbolic"](n, "classical")), expected)

    def test_dense_symbolic_direct_csr_and_fused_terms_agree(self):
        for n in range(1, 5):
            for model in ("quantum", "classical"):
                with self.subTest(n=n, model=model):
                    dense_terms = self.original["generate_paulis"](n, model)
                    symbolic_terms = self.penny["generate_paulis_symbolic"](n, model)
                    direct_terms = self.penny["precompute_sparse_paulis_direct"](n, model)
                    fused = self.penny["build_fused_pauli_kernel"](n, model)
                    weights = self.rng.normal(size=len(dense_terms))
                    block = random_state_vectors(self.rng, n, 5).T

                    dense_action = sum(
                        (weight * term @ block for weight, term in zip(weights, dense_terms)),
                        start=np.zeros_like(block),
                    )
                    csr_action = self.penny["apply_pauli_hamiltonian"](
                        weights, direct_terms, block
                    )
                    fused_action = self.penny["apply_pauli_hamiltonian"](
                        weights, fused, block
                    )
                    np.testing.assert_allclose(csr_action, dense_action, atol=1e-12)
                    np.testing.assert_allclose(fused_action, dense_action, atol=1e-12)

                    for dense, symbolic, direct in zip(dense_terms, symbolic_terms, direct_terms):
                        symbolic_matrix = self.penny["qml"].matrix(
                            symbolic, wire_order=tuple(range(n))
                        )
                        np.testing.assert_allclose(symbolic_matrix, dense, atol=1e-12)
                        np.testing.assert_allclose(direct.toarray(), dense, atol=1e-12)

    def test_vector_and_density_energy_and_accuracy_agree(self):
        n, samples = 4, 15
        vectors = random_state_vectors(self.rng, n, samples)
        states = densities(vectors)
        terms = self.original["generate_paulis"](n, "quantum")
        weights = self.rng.normal(size=len(terms))
        hamiltonian = sum(weight * term for weight, term in zip(weights, terms))
        labels = np.where(self.rng.random(samples) < 0.5, -1, 1)

        vector_energies = self.penny["state_energies"](hamiltonian, vectors)
        density_energies = self.penny["state_energies"](hamiltonian, states)
        np.testing.assert_allclose(vector_energies, density_energies, atol=1e-12)
        self.assertEqual(
            self.penny["calculate_accuracy"](hamiltonian, vectors, labels),
            self.penny["calculate_accuracy"](hamiltonian, states, labels),
        )

    def test_label_aggregates_are_exact_for_vectors_and_densities(self):
        vectors = random_state_vectors(self.rng, 4, 20)
        states = densities(vectors)
        labels = np.array([1, -1] * 10)
        vector_aggregates = self.penny["aggregate_states_by_label"](vectors, labels)
        density_aggregates = self.penny["aggregate_states_by_label"](states, labels)
        for vector_aggregate, density_aggregate in zip(vector_aggregates, density_aggregates):
            np.testing.assert_allclose(vector_aggregate, density_aggregate, atol=1e-12)
        self.assertAlmostEqual(
            np.trace(vector_aggregates[0] + vector_aggregates[1]).real, 1.0, places=12
        )

    def test_original_loss_and_gradient_match_vectorized_dense_reference(self):
        n, samples, temperature = 3, 12, 2.0
        vectors = random_state_vectors(self.rng, n, samples)
        states = densities(vectors)
        labels = np.where(self.rng.random(samples) < 0.5, -1, 1)
        terms = self.original["generate_paulis"](n, "quantum")
        weights = self.rng.uniform(-0.4, 0.4, len(terms))

        original_loss, original_gradient = original_loss_and_gradient(
            self.original, weights, states, labels, terms, temperature
        )
        penny_loss, penny_gradient = self.penny["compute_loss_and_grads_vectorized"](
            weights, states, labels, terms, temperature
        )
        self.assertAlmostEqual(original_loss, penny_loss, places=12)
        np.testing.assert_allclose(original_gradient, penny_gradient, atol=1e-10)

    def test_optimized_gradient_matches_finite_difference(self):
        n, samples = 3, 10
        vectors = random_state_vectors(self.rng, n, samples)
        labels = np.where(self.rng.random(samples) < 0.5, -1, 1)
        terms = self.penny["generate_paulis_symbolic"](n, "quantum")
        sparse = self.penny["precompute_sparse_paulis_direct"](n, "quantum")
        aggregates = self.penny["aggregate_states_by_label"](vectors, labels)
        weights = self.rng.uniform(-0.3, 0.3, len(terms))
        loss, gradient = self.penny["compute_loss_and_grads_aggregated_symbolic"](
            weights, aggregates, terms, 2.0, n, sparse
        )
        self.assertTrue(np.isfinite(loss))
        epsilon = 1e-6
        for index in (0, len(weights) // 2, len(weights) - 1):
            direction = np.zeros_like(weights)
            direction[index] = epsilon
            plus = self.penny["compute_loss_and_grads_aggregated_symbolic"](
                weights + direction, aggregates, terms, 2.0, n, sparse
            )[0]
            minus = self.penny["compute_loss_and_grads_aggregated_symbolic"](
                weights - direction, aggregates, terms, 2.0, n, sparse
            )[0]
            finite_difference = (plus - minus) / (2 * epsilon)
            self.assertAlmostEqual(gradient[index], finite_difference, places=7)

    def test_dense_generic_aggregate_and_diagonal_fcim_paths_agree(self):
        n, samples = 4, 16
        vectors = random_state_vectors(self.rng, n, samples)
        states = densities(vectors)
        labels = np.where(self.rng.random(samples) < 0.5, -1, 1)
        temperature = 2.0

        quantum_dense = self.original["generate_paulis"](n, "quantum")
        quantum_symbolic = self.penny["generate_paulis_symbolic"](n, "quantum")
        quantum_sparse = self.penny["precompute_sparse_paulis_direct"](n, "quantum")
        quantum_weights = self.rng.uniform(-0.4, 0.4, len(quantum_dense))
        dense_result = self.penny["compute_loss_and_grads_vectorized"](
            quantum_weights, states, labels, quantum_dense, temperature
        )
        aggregate_result = self.penny["compute_loss_and_grads_aggregated_symbolic"](
            quantum_weights,
            self.penny["aggregate_states_by_label"](vectors, labels),
            quantum_symbolic,
            temperature,
            n,
            quantum_sparse,
        )
        self.assertAlmostEqual(dense_result[0], aggregate_result[0], places=11)
        np.testing.assert_allclose(dense_result[1], aggregate_result[1], atol=1e-10)

        classical_symbolic = self.penny["generate_paulis_symbolic"](n, "classical")
        classical_sparse = self.penny["precompute_sparse_paulis_direct"](n, "classical")
        classical_weights = self.rng.uniform(-0.4, 0.4, len(classical_symbolic))
        generic_result = self.penny["compute_loss_and_grads_symbolic"](
            classical_weights,
            states,
            labels,
            classical_symbolic,
            temperature,
            n,
            classical_sparse,
        )
        diagonal_result = self.penny["compute_loss_and_grads_fcim_diagonal"](
            classical_weights,
            self.penny["aggregate_basis_probabilities_by_label"](vectors, labels),
            self.penny["build_fcim_feature_matrix"](n),
            temperature,
        )
        self.assertAlmostEqual(generic_result[0], diagonal_result[0], places=11)
        np.testing.assert_allclose(generic_result[1], diagonal_result[1], atol=1e-10)

    def test_fixed_and_adaptive_chebyshev_match_exact_backend(self):
        n, samples = 4, 18
        vectors = random_state_vectors(self.rng, n, samples)
        labels = np.where(self.rng.random(samples) < 0.5, -1, 1)
        terms = self.penny["generate_paulis_symbolic"](n, "quantum")
        sparse = self.penny["precompute_sparse_paulis_direct"](n, "quantum")
        fused = self.penny["build_fused_pauli_kernel"](n, "quantum")
        weights = self.rng.uniform(-0.3, 0.3, len(terms))
        bound = 1.05 * np.sum(np.abs(weights))
        exact = self.penny["compute_loss_and_grads_aggregated_symbolic"](
            weights,
            self.penny["aggregate_states_by_label"](vectors, labels),
            terms,
            2.0,
            n,
            sparse,
        )
        degree = self.penny["select_adaptive_chebyshev_degree"](
            2.0, bound, tolerance=1e-7, min_degree=8, max_degree=128
        )
        chebyshev = self.penny["compute_loss_and_grads_chebyshev"](
            weights,
            vectors,
            labels,
            fused,
            2.0,
            degree=degree,
            chunk_size=7,
            spectral_bound=bound,
        )
        self.assertLess(abs(exact[0] - chebyshev[0]), 2e-7)
        self.assertLess(np.max(np.abs(exact[1] - chebyshev[1])), 2e-7)

    def test_complex64_tracks_complex128(self):
        n, samples = 5, 20
        vectors = random_state_vectors(self.rng, n, samples)
        labels = np.where(self.rng.random(samples) < 0.5, -1, 1)
        terms = self.penny["generate_paulis_symbolic"](n, "quantum")
        weights = self.rng.uniform(-0.3, 0.3, len(terms))
        aggregates = self.penny["aggregate_states_by_label"](vectors, labels)
        results = {}
        for dtype in ("complex64", "complex128"):
            sparse = self.penny["precompute_sparse_paulis_direct"](n, "quantum", dtype)
            results[dtype] = self.penny["compute_loss_and_grads_aggregated_symbolic"](
                weights,
                aggregates,
                terms,
                2.0,
                n,
                sparse,
                complex_dtype=dtype,
            )
        self.assertLess(abs(results["complex64"][0] - results["complex128"][0]), 1e-5)
        self.assertLess(
            np.max(np.abs(results["complex64"][1] - results["complex128"][1])),
            1e-5,
        )

    def test_minibatch_adam_is_reproducible_and_records_adaptive_degree(self):
        arguments = dict(
            n=4,
            epochs=4,
            optimizer="adam",
            adam_batch_size=6,
            adam_seed=91,
            quantum_method="chebyshev",
            adaptive_chebyshev=True,
            chebyshev_min_degree=8,
            chebyshev_max_degree=48,
            chebyshev_tolerance=1e-5,
            chebyshev_chunk_size=4,
            num_training_states=24,
            num_validation_states=8,
            validation_frequency=0,
            early_stopping_patience=None,
            complex_dtype="complex64",
        )
        histories = []
        with temporary_working_directory() as directory:
            for _ in range(2):
                np.random.seed(404)
                with contextlib.redirect_stdout(io.StringIO()):
                    histories.append(self.penny["optimize_phase2"](**arguments))
            metrics = self.penny["pd"].read_csv(
                directory / "outputs" / "logloss_4qubit_phase2.csv"
            )
        np.testing.assert_allclose(histories[0][0], histories[1][0], atol=0)
        np.testing.assert_allclose(histories[0][1], histories[1][1], atol=0)
        self.assertEqual(set(metrics["Batch_Size"]), {6})
        self.assertTrue(metrics["Chebyshev_Degree"].notna().all())

    def test_auto_n11_route_does_not_call_dense_eigh(self):
        with temporary_working_directory():
            with mock.patch.object(
                self.penny["np"].linalg,
                "eigh",
                side_effect=AssertionError("dense eigh called"),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    history_q, history_c = self.penny["optimize_phase2"](
                        n=11,
                        epochs=1,
                        optimizer="adam",
                        adam_batch_size=2,
                        adam_seed=3,
                        quantum_method="auto",
                        adaptive_chebyshev=True,
                        chebyshev_min_degree=4,
                        chebyshev_max_degree=8,
                        chebyshev_tolerance=1e-4,
                        chebyshev_chunk_size=2,
                        num_training_states=4,
                        num_validation_states=2,
                        validation_frequency=0,
                        early_stopping_patience=None,
                        complex_dtype="complex64",
                    )
        self.assertEqual(len(history_q), 1)
        self.assertTrue(np.isfinite(history_q[0] + history_c[0]))

    def test_backend_auto_and_requested_gpu_behavior(self):
        _, backend_name = self.penny["resolve_array_backend"]("auto")
        self.assertIn(backend_name, {"numpy", "cupy"})
        if backend_name == "numpy":
            with self.assertRaisesRegex(RuntimeError, "GPU backend unavailable"):
                self.penny["resolve_array_backend"]("cupy")


if __name__ == "__main__":
    unittest.main()
