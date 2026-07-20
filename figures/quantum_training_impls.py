"""
Two implementations of the quantum Heisenberg logistic-loss training pass
from the paper's Fig. 8 / Sec. VI.D.2 experiment:

- ``run_original``: the code as written in notebooks/paper/logloss.ipynb --
  dense Pauli matrices, per-state Python loop for the loss, per-parameter
  per-state Python loop (``dfj``) for the gradient.
- ``run_pennylane``: the code as written in notebooks/pennylane/logloss_pennylane.ipynb
  ``optimize_phase2`` quantum pass -- PennyLane symbolic Pauli operators,
  label-aggregated density matrices (R+/R-), vectorized gradient.

Both are reproduced verbatim (only trimmed to the quantum-only computation,
dropping the classical FCIM pass, which this project takes from the digitized
Fig. 8 image instead of rerunning). Given the same seed, they consume the
random number generator in the same order, so they train on bit-identical
states/labels/initial weights -- a true apples-to-apples comparison.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pennylane as qml

T = 2.0
ETA = 0.1
N_TRAIN = 1000
N_VAL = 500

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def krons(ops):
    res = ops[0]
    for op in ops[1:]:
        res = np.kron(res, op)
    return res


def generate_paulis_quantum(n):
    """Dense Heisenberg (quantum) Pauli terms -- NN XX+YY+ZZ + X+Y+Z, 6n-3 terms."""
    paulis = []
    base_ops = [X, Y, Z]
    for op in base_ops:
        for i in range(n - 1):
            j = i + 1
            op_list = [I2] * n
            op_list[i] = op
            op_list[j] = op
            paulis.append(krons(op_list))
    for op in base_ops:
        for i in range(n):
            op_list = [I2] * n
            op_list[i] = op
            paulis.append(krons(op_list))
    return paulis


def make_training_states(n, num_states=N_TRAIN):
    states = []
    dim = 2 ** n
    for _ in range(num_states):
        vec = np.random.randn(dim) + 1j * np.random.randn(dim)
        vec /= np.linalg.norm(vec)
        states.append(np.outer(vec, vec.conj()))
    return np.array(states)


def make_validation_set(n, num_states=N_VAL):
    return make_training_states(n, num_states)


def calculate_accuracy(H_model, states, true_labels):
    energies = np.array([np.real(np.trace(H_model @ rho)) for rho in states])
    predictions = np.sign(energies)
    predictions[predictions == 0] = 1
    return float(np.mean(predictions == true_labels) * 100)


def fdd_logloss_matrix(y, T, eigenvalues):
    l = eigenvalues.reshape(-1, 1)
    k = eigenvalues.reshape(1, -1)
    diff = l - k
    with np.errstate(divide="ignore", invalid="ignore"):
        res = (T * np.log(1 + np.exp(-y * l / T)) - T * np.log(1 + np.exp(-y * k / T))) / diff
    derivative = -y / (1 + np.exp(y * l / T))
    mask = np.abs(diff) < 1e-10
    return np.where(mask, derivative, res)


def dfj(y, rho, eigvals, eigvecs, H_j_basis, T):
    # NOTE: the paper-original dfj divided H_j_basis by T here, an extraneous
    # factor -- the paper's Theorem 5 / Eq. (63) gradient is dH/dw_j = H_j
    # directly (see docs/paper_comparison_guide.md, "Important note about
    # gradients"). notebooks/paper/logloss.ipynb and this reproduction both
    # use the corrected form so the comparison is apples-to-apples.
    H_j_tilde = eigvecs.T.conj() @ H_j_basis @ eigvecs
    rho_tilde = eigvecs.T.conj() @ rho @ eigvecs
    F = fdd_logloss_matrix(y, T, eigvals)
    return np.real(np.sum(F * H_j_tilde * rho_tilde.T))


def _prepare_common_problem(n, seed):
    """Shared data generation, replicated identically in both code paths."""
    np.random.seed(seed)
    training_states = make_training_states(n, N_TRAIN)
    val_states = make_validation_set(n, N_VAL)
    pauli_q = generate_paulis_quantum(n)
    target_p = (np.random.random(len(pauli_q)) - 0.5) * 4
    est_q_init = (np.random.random(len(pauli_q)) - 0.5)
    return training_states, val_states, pauli_q, target_p, est_q_init


def run_original(n: int, epochs: int, seed: int, log_every: int = 20):
    """Verbatim `logloss.ipynb` quantum pass (dense loop `dfj` gradient)."""
    training_states, val_states, pauli_q, target_p, est_q = _prepare_common_problem(n, seed)
    N_states = len(training_states)

    H_target = sum(p * mat for p, mat in zip(target_p, pauli_q))
    ys = np.array([np.sign(np.real(np.trace(H_target @ rho))) for rho in training_states])
    ys[ys == 0] = 1
    ys_val = np.array([np.sign(np.real(np.trace(H_target @ rho))) for rho in val_states])
    ys_val[ys_val == 0] = 1

    history = []
    accuracy = {}
    t_start = time.perf_counter()
    for epoch in range(epochs):
        H_q = sum(p * mat for p, mat in zip(est_q, pauli_q))
        eval_q, evec_q = np.linalg.eigh(H_q)

        l_q = 0.0
        for i in range(N_states):
            yi = ys[i]
            m_loss_q = evec_q @ np.diag(T * np.log(1 + np.exp(-yi * eval_q / T))) @ evec_q.T.conj()
            l_q += np.real(np.trace(m_loss_q @ training_states[i]))
        l_q /= N_states
        history.append(l_q)

        grad_q = np.zeros(len(pauli_q))
        for j in range(len(pauli_q)):
            g_j = sum(dfj(ys[i], training_states[i], eval_q, evec_q, pauli_q[j], T)
                      for i in range(N_states))
            grad_q[j] = g_j / N_states
        est_q -= ETA * grad_q

        if epoch % log_every == 0 or epoch == epochs - 1:
            accuracy[epoch] = calculate_accuracy(H_q, val_states, ys_val)
    elapsed = time.perf_counter() - t_start

    return {"history": history, "accuracy": accuracy, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# PennyLane symbolic implementation (verbatim `optimize_phase2` quantum pass)
# ---------------------------------------------------------------------------

def _single_pauli(op_char, wire):
    return {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}[op_char](wire)


def generate_paulis_symbolic_quantum(n):
    paulis = []
    for op_char in ["X", "Y", "Z"]:
        for i in range(n - 1):
            paulis.append(_single_pauli(op_char, i) @ _single_pauli(op_char, i + 1))
    for op_char in ["X", "Y", "Z"]:
        paulis.extend(_single_pauli(op_char, i) for i in range(n))
    return paulis


def precompute_sparse_paulis(pauli_ops, n):
    wire_order = tuple(range(n))
    return [op.sparse_matrix(wire_order=wire_order, format="csr") for op in pauli_ops]


def build_hamiltonian_matrix_from_symbolic(weights, pauli_ops, n, sparse_paulis):
    H_sparse = sum((w * op for w, op in zip(weights, sparse_paulis)),
                   start=sparse_paulis[0] * 0)
    return H_sparse.toarray()


def make_state_vectors(n, num_states):
    """Same RNG consumption pattern as make_training_states, but stores vectors."""
    dim = 2 ** n
    states = np.empty((num_states, dim), dtype=complex)
    for i in range(num_states):
        vec = np.random.randn(dim) + 1j * np.random.randn(dim)
        states[i] = vec / np.linalg.norm(vec)
    return states


def state_energies(H_model, states):
    return np.real(np.einsum("bi,ij,bj->b", states.conj(), H_model, states, optimize=True))


def aggregate_states_by_label(states, ys):
    dim = states.shape[1]
    aggregates = []
    for label in (+1, -1):
        selected = states[ys == label]
        aggregate = np.einsum("bi,bj->ij", selected, selected.conj(), optimize=True)
        aggregates.append(aggregate / len(states))
    return tuple(aggregates)


def compute_fdd_matrix(eigvals, y, T):
    l = eigvals.reshape(-1, 1)
    k = eigvals.reshape(1, -1)
    diff = l - k
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (T * np.log1p(np.exp(-y * l / T)) - T * np.log1p(np.exp(-y * k / T))) / diff
    derivative = -y / (1 + np.exp(y * l / T))
    mask = np.abs(diff) < 1e-10
    return np.where(mask, derivative, F)


def compute_loss_and_grads_aggregated_symbolic(weights, label_aggregates, pauli_ops, T, n, sparse_paulis):
    R_plus, R_minus = label_aggregates
    H_matrix = build_hamiltonian_matrix_from_symbolic(weights, pauli_ops, n, sparse_paulis)
    eigvals, eigvecs = np.linalg.eigh(H_matrix)

    R_plus_tilde = eigvecs.T.conj() @ R_plus @ eigvecs
    R_minus_tilde = eigvecs.T.conj() @ R_minus @ eigvecs
    diag_loss_plus = T * np.logaddexp(0, -eigvals / T)
    diag_loss_minus = T * np.logaddexp(0, eigvals / T)
    loss = np.real(
        np.dot(diag_loss_plus, np.diag(R_plus_tilde))
        + np.dot(diag_loss_minus, np.diag(R_minus_tilde)))

    F_plus = compute_fdd_matrix(eigvals, y=+1, T=T)
    F_minus = compute_fdd_matrix(eigvals, y=-1, T=T)
    derivative_eigenbasis = F_plus * R_plus_tilde.T + F_minus * R_minus_tilde.T
    derivative_matrix = eigvecs.conj() @ derivative_eigenbasis @ eigvecs.T
    grads = np.array([np.real(op.multiply(derivative_matrix).sum()) for op in sparse_paulis])
    return loss, grads


def run_pennylane(n: int, epochs: int, seed: int, log_every: int = 20):
    """Verbatim `logloss_pennylane.ipynb` `optimize_phase2` quantum pass."""
    np.random.seed(seed)
    training_states = make_state_vectors(n, N_TRAIN)
    val_states = make_state_vectors(n, N_VAL)
    pauli_q_sym = generate_paulis_symbolic_quantum(n)
    sparse_q = precompute_sparse_paulis(pauli_q_sym, n)

    target_p = (np.random.random(len(pauli_q_sym)) - 0.5) * 4
    est_q = (np.random.random(len(pauli_q_sym)) - 0.5)

    H_target = build_hamiltonian_matrix_from_symbolic(target_p, pauli_q_sym, n, sparse_q)
    ys = np.sign(state_energies(H_target, training_states))
    ys[ys == 0] = 1
    ys_val = np.sign(state_energies(H_target, val_states))
    ys_val[ys_val == 0] = 1

    label_aggregates = aggregate_states_by_label(training_states, ys)

    history = []
    accuracy = {}
    t_start = time.perf_counter()
    for epoch in range(epochs):
        l_q, grad_q = compute_loss_and_grads_aggregated_symbolic(
            est_q, label_aggregates, pauli_q_sym, T, n, sparse_q)
        history.append(l_q)
        est_q -= ETA * grad_q

        if epoch % log_every == 0 or epoch == epochs - 1:
            H_q = build_hamiltonian_matrix_from_symbolic(est_q, pauli_q_sym, n, sparse_q)
            predictions = np.sign(state_energies(H_q, val_states))
            predictions[predictions == 0] = 1
            accuracy[epoch] = float(np.mean(predictions == ys_val) * 100)
    elapsed = time.perf_counter() - t_start

    return {"history": history, "accuracy": accuracy, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# PennyLane classical FCIM diagonal solver (verbatim `optimize_phase2`
# classical pass -- no eigendecomposition, since FCIM terms all commute).
# ---------------------------------------------------------------------------

def build_fcim_feature_matrix(n):
    """Computational-basis eigenvalues of all ZZ terms followed by Z terms."""
    basis_indices = np.arange(2 ** n, dtype=np.uint64)[:, None]
    shifts = np.arange(n - 1, -1, -1, dtype=np.uint64)
    z_values = 1 - 2 * ((basis_indices >> shifts) & 1).astype(float)
    columns = [z_values[:, i] * z_values[:, j]
               for i, j in itertools.combinations(range(n), 2)]
    columns.extend(z_values[:, i] for i in range(n))
    return np.column_stack(columns)


def aggregate_basis_probabilities_by_label(states, ys):
    probabilities = np.abs(states) ** 2
    return tuple(probabilities[ys == label].sum(axis=0) / len(states)
                 for label in (+1, -1))


def compute_loss_and_grads_fcim_diagonal(weights, label_basis_probabilities, feature_matrix, T):
    p_plus, p_minus = label_basis_probabilities
    energies = feature_matrix @ weights
    loss_plus = T * np.logaddexp(0, -energies / T)
    loss_minus = T * np.logaddexp(0, energies / T)
    loss = np.dot(p_plus, loss_plus) + np.dot(p_minus, loss_minus)

    derivative_plus = -np.exp(-np.logaddexp(0, energies / T))
    derivative_minus = np.exp(-np.logaddexp(0, -energies / T))
    weighted_derivative = p_plus * derivative_plus + p_minus * derivative_minus
    grads = feature_matrix.T @ weighted_derivative
    return float(np.real(loss)), np.real(grads)


def calculate_fcim_accuracy(weights, feature_matrix, states, true_labels):
    basis_energies = feature_matrix @ weights
    probabilities = np.abs(states) ** 2
    energies = probabilities @ basis_energies
    predictions = np.sign(np.real(energies))
    predictions[predictions == 0] = 1
    return float(np.mean(predictions == true_labels) * 100)


def run_pennylane_matched(n: int, epochs: int, seed: int, log_every: int = 20):
    """Verbatim `logloss_pennylane.ipynb` `optimize_phase2`: quantum AND
    classical FCIM trained together on the same states/target/labels, so the
    two curves form a genuinely matched single-trial comparison (mirrors how
    the original `optimize()` trains both models side by side on shared data).

    RNG consumption order matches `run_pennylane` exactly up through `est_q`,
    so the quantum curve here is bit-identical to `run_pennylane`'s; `est_c`
    is drawn afterward, same as in `optimize_phase2`.
    """
    np.random.seed(seed)
    training_states = make_state_vectors(n, N_TRAIN)
    val_states = make_state_vectors(n, N_VAL)
    pauli_q_sym = generate_paulis_symbolic_quantum(n)
    sparse_q = precompute_sparse_paulis(pauli_q_sym, n)
    fcim_features = build_fcim_feature_matrix(n)

    target_p = (np.random.random(len(pauli_q_sym)) - 0.5) * 4
    est_q = (np.random.random(len(pauli_q_sym)) - 0.5)
    est_c = (np.random.random(fcim_features.shape[1]) - 0.5)

    H_target = build_hamiltonian_matrix_from_symbolic(target_p, pauli_q_sym, n, sparse_q)
    ys = np.sign(state_energies(H_target, training_states))
    ys[ys == 0] = 1
    ys_val = np.sign(state_energies(H_target, val_states))
    ys_val[ys_val == 0] = 1

    label_aggregates = aggregate_states_by_label(training_states, ys)
    label_basis_probabilities = aggregate_basis_probabilities_by_label(training_states, ys)

    history_q = []
    history_c = []
    accuracy_q = {}
    accuracy_c = {}
    t_start = time.perf_counter()
    for epoch in range(epochs):
        l_q, grad_q = compute_loss_and_grads_aggregated_symbolic(
            est_q, label_aggregates, pauli_q_sym, T, n, sparse_q)
        history_q.append(l_q)
        est_q -= ETA * grad_q

        l_c, grad_c = compute_loss_and_grads_fcim_diagonal(
            est_c, label_basis_probabilities, fcim_features, T)
        history_c.append(l_c)
        est_c -= ETA * grad_c

        if epoch % log_every == 0 or epoch == epochs - 1:
            H_q = build_hamiltonian_matrix_from_symbolic(est_q, pauli_q_sym, n, sparse_q)
            predictions = np.sign(state_energies(H_q, val_states))
            predictions[predictions == 0] = 1
            accuracy_q[epoch] = float(np.mean(predictions == ys_val) * 100)
            accuracy_c[epoch] = calculate_fcim_accuracy(est_c, fcim_features, val_states, ys_val)
    elapsed = time.perf_counter() - t_start

    return {
        "history_q": history_q,
        "history_c": history_c,
        "accuracy_q": accuracy_q,
        "accuracy_c": accuracy_c,
        "elapsed": elapsed,
    }
