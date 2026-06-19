"""
Minimal demo: sampling the Fermi-Dirac neuron output with PennyLane.

    output = Tr[ g_T(H(omega)) rho ]                      (Eq. 17)
    g_T(x) = tanh(x / T)                                  (Eq. 18)
    H(omega) = sum_j omega_j H_j                          (Eq. 16)

H(omega) is a 2-qubit transverse-field Ising model (TFIM, Eq. 113).
"""

import numpy as np
import pennylane as qml


# ---- single-qubit Paulis (Paper Sec. II.A building blocks) ----
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(a, b):
    return np.kron(a, b)


# ---- H(omega): 2-qubit TFIM = -W Z0 Z1 + w0 X0 + w1 X1 (Eq. 113) ----
T = 2.0
W, w0, w1 = 1.3, 0.7, -0.4
H = -W * kron(Z, Z) + w0 * kron(X, I) + w1 * kron(I, X)


# ---- g_T(H) = tanh(H / T): apply the nonlinearity to the OPERATOR ----
# diagonalize H, apply tanh to eigenvalues, rotate back. 
evals, evecs = np.linalg.eigh(H)
g_H = evecs @ np.diag(np.tanh(evals / T)) @ evecs.conj().T   # Hermitian observable


# ---- input state rho: a pure 2-qubit state |psi> (here a Bell-ish state) ----
psi = np.array([1, 0.3, -0.5, 0.8], dtype=complex)
psi /= np.linalg.norm(psi)


# ===== exact value (no shots): Tr[g_T(H) rho] =====
dev_exact = qml.device("default.qubit", wires=2, shots=None)


@qml.qnode(dev_exact)
def exact():
    qml.StatePrep(psi, wires=[0, 1])
    return qml.expval(qml.Hermitian(g_H, wires=[0, 1]))   # Eq. 17


# ===== sampled value (finite shots): SAME observable, estimated by sampling =====
dev_shots = qml.device("default.qubit", wires=2)


@qml.qnode(dev_shots)
def circ():
    qml.StatePrep(psi, wires=[0, 1])
    return qml.expval(qml.Hermitian(g_H, wires=[0, 1]))

def sampled(n_shots):
    return float(qml.set_shots(circ, shots=n_shots)())

exact_val = float(exact())
print(f"Activation observable g_T(H) = tanh(H/T), T={T}")
print(f"EXACT   Tr[g_T(H) rho] = {exact_val:.6f}\n")
print("SAMPLED estimates (PennyLane finite shots):")
for n in [100, 1_000, 10_000, 100_000]:
    est = sampled(n)
    print(f"  shots={n:>7}:  {est:>10.6f}   (abs err {abs(est-exact_val):.6f})")
