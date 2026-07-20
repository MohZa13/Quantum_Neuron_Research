"""Utilities shared by notebook equivalence tests and scaling benchmarks."""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_NOTEBOOK = REPO_ROOT / "notebooks" / "paper" / "logloss.ipynb"
PENNYLANE_NOTEBOOK = REPO_ROOT / "notebooks" / "pennylane" / "logloss_pennylane.ipynb"


def load_notebook_namespace(path: Path, cell_indices: tuple[int, ...]) -> dict:
    """Execute selected code cells from a notebook into one namespace."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    notebook = json.loads(path.read_text())
    namespace = {"__name__": f"notebook_{path.stem}"}
    for index in cell_indices:
        cell = notebook["cells"][index]
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
    return namespace


def load_original_namespace() -> dict:
    """Load definitions without executing the original 750-epoch run."""
    return load_notebook_namespace(ORIGINAL_NOTEBOOK, (0, 1, 2, 3))


def load_pennylane_namespace() -> dict:
    """Load definitions without executing benchmark/plot cells."""
    return load_notebook_namespace(PENNYLANE_NOTEBOOK, (1, 2, 3, 4, 5, 6, 7))


def random_state_vectors(rng: np.random.Generator, n: int, samples: int, dtype=np.complex128):
    dim = 2**n
    vectors = rng.normal(size=(samples, dim)) + 1j * rng.normal(size=(samples, dim))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors.astype(dtype)


def densities(vectors: np.ndarray) -> np.ndarray:
    return np.einsum("bi,bj->bij", vectors, vectors.conj(), optimize=True)


def original_loss_and_gradient(namespace, weights, states, labels, paulis, temperature=2.0):
    """Notebook-faithful original quantum/classical loss and gradient."""
    hamiltonian = sum(
        (weight * term for weight, term in zip(weights, paulis)),
        start=np.zeros_like(paulis[0]),
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    loss = 0.0
    for label, state in zip(labels, states):
        diagonal = temperature * np.logaddexp(0, -label * eigenvalues / temperature)
        loss_matrix = eigenvectors @ np.diag(diagonal) @ eigenvectors.T.conj()
        loss += np.real(np.trace(loss_matrix @ state))
    loss /= len(states)

    gradient = np.empty(len(paulis))
    for index, term in enumerate(paulis):
        gradient[index] = np.mean(
            [
                namespace["dfj"](
                    label, state, eigenvalues, eigenvectors, term, temperature
                )
                for label, state in zip(labels, states)
            ]
        )
    return float(loss), gradient


def median_runtime(function, repeats=3, warmups=1):
    for _ in range(warmups):
        function()
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
    return result, statistics.median(samples), samples


def csr_payload_bytes(operators) -> int:
    return sum(
        operator.data.nbytes + operator.indices.nbytes + operator.indptr.nbytes
        for operator in operators
    )


def fused_payload_bytes(kernel) -> int:
    return int(kernel["sources"].nbytes + kernel["phases"].nbytes)
