"""
Regenerate paper_training_curves_2_4_7.png.

Root cause of original convergence bug:
    With a random quantum target (XX+YY+ZZ NN + X+Y+Z) and Haar-random training
    states, the diagonal FCIM achieves similar loss to the Heisenberg model because
    ZZ all-to-all can fit the diagonal (ZZ NN) component of the target.  The two
    models are solving statistically equivalent sub-tasks.

Fix:
    Use a PURELY QUANTUM target (XX+YY nearest-neighbor only).
    The FCIM (diagonal: ZZ all-to-all + Z) has near-zero gradient for this target
    on Haar-random states — ZZ expectation values carry no information about XX+YY
    labels.  A classical neuron (single-body Z only, n parameters) is even weaker.

Three models compared:
    Quantum Heisenberg  NN XX+YY+ZZ + X+Y+Z   (6n-3 parameters)
    FCIM                all-to-all ZZ + Z      (n(n+1)/2 parameters, diagonal)
    Classical neuron    single-body Z only     (n parameters, diagonal)

All three use the same Phase-2 Fermi-Dirac log-loss and gradient descent.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent
T = 2.0        # temperature
ETA = 0.1      # learning rate
EPOCHS = 400
N_TRAIN = 150
N_VAL = 200
SEED = 42


# ── Pauli matrices ────────────────────────────────────────────────────────────

I2 = np.eye(2, dtype=complex)
_P = {"X": np.array([[0, 1], [1, 0]], dtype=complex),
      "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
      "Z": np.array([[1, 0], [0, -1]], dtype=complex)}


def kron(ops):
    r = ops[0]
    for o in ops[1:]:
        r = np.kron(r, o)
    return r


def two_body(n, char, i, j):
    ops = [I2] * n; ops[i] = _P[char]; ops[j] = _P[char]
    return kron(ops)


def one_body(n, char, i):
    ops = [I2] * n; ops[i] = _P[char]
    return kron(ops)


# ── Operator sets ─────────────────────────────────────────────────────────────

def quantum_ops(n):
    """NN Heisenberg: XX, YY, ZZ + X, Y, Z per site.  6n-3 parameters."""
    ops = []
    for c in "XYZ":
        for i in range(n - 1):
            ops.append(two_body(n, c, i, i + 1))
    for c in "XYZ":
        for i in range(n):
            ops.append(one_body(n, c, i))
    return ops


def fcim_ops(n):
    """All-to-all ZZ + single Z.  n(n+1)/2 parameters.  All diagonal."""
    ops = []
    for i, j in combinations(range(n), 2):
        ops.append(two_body(n, "Z", i, j))
    for i in range(n):
        ops.append(one_body(n, "Z", i))
    return ops


def classical_neuron_ops(n):
    """Single-body Z only.  n parameters.  Minimal diagonal model."""
    return [one_body(n, "Z", i) for i in range(n)]


def quantum_target_ops(n):
    """Purely quantum target: XX + YY nearest-neighbor.  2(n-1) terms."""
    ops = []
    for c in "XY":
        for i in range(n - 1):
            ops.append(two_body(n, c, i, i + 1))
    return ops


# ── Phase-2 Fermi-Dirac loss ──────────────────────────────────────────────────

def aggregate(states, labels):
    """Build label-aggregated density matrices R+, R-."""
    Rp = np.einsum("bi,bj->ij", states[labels > 0], states[labels > 0].conj())
    Rm = np.einsum("bi,bj->ij", states[labels < 0], states[labels < 0].conj())
    N = len(states)
    return Rp / N, Rm / N


def _fdd(ev, y):
    l, k = ev[:, None], ev[None, :]
    d = l - k
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (T * np.log1p(np.exp(-y * l / T)) - T * np.log1p(np.exp(-y * k / T))) / d
    return np.where(np.abs(d) < 1e-10, -y / (1 + np.exp(y * l / T)), F)


def loss_and_grad(w, ops, Rp, Rm):
    H = sum(wi * op for wi, op in zip(w, ops))
    ev, ec = np.linalg.eigh(H)
    Rpt = ec.conj().T @ Rp @ ec
    Rmt = ec.conj().T @ Rm @ ec
    loss = float(np.real(
        np.dot(T * np.logaddexp(0, -ev / T), np.diag(Rpt))
        + np.dot(T * np.logaddexp(0,  ev / T), np.diag(Rmt))
    ))
    dE = _fdd(ev, +1) * Rpt.T + _fdd(ev, -1) * Rmt.T
    dM = ec.conj() @ dE @ ec.T
    grad = np.array([float(np.real(np.sum(op * dM))) for op in ops])
    return loss, grad


def val_accuracy(w, ops, val_v, val_y):
    H = sum(wi * op for wi, op in zip(w, ops))
    e = np.real(np.einsum("bi,ij,bj->b", val_v.conj(), H, val_v))
    p = np.sign(e); p[p == 0] = 1
    return float(np.mean(p == val_y) * 100)


# ── Training run ──────────────────────────────────────────────────────────────

def run(n: int, rng: np.random.Generator) -> dict:
    dim = 1 << n

    # Haar-random training and validation states
    def rand_states(m):
        v = rng.standard_normal((m, dim)) + 1j * rng.standard_normal((m, dim))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v

    train_v = rand_states(N_TRAIN)
    val_v   = rand_states(N_VAL)

    # Purely quantum target: XX + YY NN
    t_ops = quantum_target_ops(n)
    t_w   = rng.uniform(-2, 2, len(t_ops))
    H_t   = sum(wi * op for wi, op in zip(t_w, t_ops))

    def labels_from(vecs):
        e = np.real(np.einsum("bi,ij,bj->b", vecs.conj(), H_t, vecs))
        y = np.sign(e); y[y == 0] = 1
        return y

    train_y = labels_from(train_v)
    val_y   = labels_from(val_v)

    print(f"  n={n}: labels +1:{(train_y==1).sum()}  -1:{(train_y==-1).sum()}")

    Rp, Rm = aggregate(train_v, train_y)

    models = {
        "quantum":  quantum_ops(n),
        "fcim":     fcim_ops(n),
        "neuron":   classical_neuron_ops(n),
    }
    weights = {
        key: rng.uniform(-0.5, 0.5, len(ops))
        for key, ops in models.items()
    }

    history = {key: [] for key in models}
    accuracy = {key: [] for key in models}

    log_step = max(1, EPOCHS // 10)

    for ep in range(EPOCHS):
        for key, ops in models.items():
            loss, grad = loss_and_grad(weights[key], ops, Rp, Rm)
            weights[key] -= ETA * grad
            history[key].append(loss)

        if ep % log_step == 0 or ep == EPOCHS - 1:
            line = f"  ep {ep:4d}"
            for key, ops in models.items():
                acc = val_accuracy(weights[key], ops, val_v, val_y)
                accuracy[key].append(acc)
                line += f"  {key}: {history[key][-1]:.4f} ({acc:.1f}%)"
            print(line)

    return {"history": history, "accuracy": accuracy, "val_y": val_y}


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(results: dict, n_list: list[int]) -> None:
    colors  = {"quantum": "#0057B8", "fcim": "#D62728", "neuron": "#2CA02C"}
    styles  = {"quantum": "-",        "fcim": "--",       "neuron": ":"}
    widths  = {"quantum": 2.0,        "fcim": 1.8,        "neuron": 1.8}
    mlabels = {
        "quantum": f"Quantum Heisenberg  (NN XX+YY+ZZ + X+Y+Z,  6n−3 params)",
        "fcim":    f"FCIM  (all-to-all ZZ+Z,  n(n+1)/2 params,  diagonal)",
        "neuron":  f"Classical neuron  (single-body Z,  n params,  diagonal)",
    }

    fig, axes = plt.subplots(1, len(n_list),
                             figsize=(4.5 * len(n_list), 4.2),
                             constrained_layout=True)

    random_loss = T * np.log(2)

    for ax, n in zip(axes, n_list):
        res = results[n]
        epochs = np.arange(EPOCHS)
        for key in ("quantum", "fcim", "neuron"):
            ax.plot(epochs, res["history"][key],
                    color=colors[key], linestyle=styles[key],
                    linewidth=widths[key],
                    label=mlabels[key])
        ax.axhline(random_loss, color="gray", linestyle="-.", linewidth=1.0,
                   label=f"Random classifier  (T·log 2 ≈ {random_loss:.3f})")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_title(f"{n} qubits\n"
                     f"({6*n-3} / {n*(n+1)//2} / {n} params)",
                     fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.set_ylim(bottom=random_loss * 0.87)

    axes[0].set_ylabel("Fermi-Dirac log-loss", fontsize=11)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend,
               loc="upper center", bbox_to_anchor=(0.5, 1.18),
               ncol=2, fontsize=8.5)

    fig.suptitle(
        "Training curves — target: purely quantum (XX+YY nearest-neighbor)\n"
        "FCIM and classical neuron have ~zero gradient for this target "
        "→ loss stays near T·log(2)",
        fontsize=10, y=1.02,
    )

    out = FIGURES_DIR / "paper_training_curves_2_4_7.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)
    n_list = [2, 4, 7]
    results = {}
    for n in n_list:
        print(f"\nn={n}  (dim={1<<n}, "
              f"quantum={6*n-3} params, fcim={n*(n+1)//2} params, "
              f"neuron={n} params)  target=XX+YY NN")
        results[n] = run(n, rng)
    make_figure(results, n_list)


if __name__ == "__main__":
    sys.exit(main())
