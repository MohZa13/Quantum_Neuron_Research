"""Demonstrate the quantum-neuron training ability on real thermal states.

Loads the 248 extended-Heisenberg Pauli features Tr(rho_m P_j) that the pipeline
already computed for the 1000-molecule CAS(8,8), kT=0.1 set, labels each molecule
saturated vs. unsaturated (degree of unsaturation, from the conjugation screen),
holds out 30% of molecules, and trains the Hamiltonian classifier's decision rule
-- a logistic loss on H(theta) = sum_j theta_j P_j, whose score is linear in the
Pauli features -- by gradient descent. Plots train/test loss and accuracy vs.
iteration: the loss falls and held-out accuracy rises, i.e. it learns and
generalizes to molecules it never saw.
"""
from __future__ import annotations

import csv
import h5py
import numpy as np
import matplotlib.pyplot as plt

RNG = np.random.default_rng(7)

# --- load real thermal-state features ---------------------------------------
with h5py.File("results/qh9_dense_cas8-8_kT0p1_extheis.h5", "r") as f:
    idxs, feats = [], []
    for name in sorted((k for k in f if k.startswith("mol_")),
                       key=lambda s: int(s.split("_")[1])):
        idxs.append(int(name.split("_")[1]))
        feats.append(f[name]["kT_0p1000"]["coeffs"][:])
idxs = np.array(idxs)
X = np.asarray(feats, dtype=np.float64)          # (N, 248) = Tr(rho_m P_j)

# --- labels: small vs large HOMO-LUMO gap (median split), joined from screen -
gaps = {int(r["idx"]): float(r["gap_Ha"])
        for r in csv.DictReader(open("results/qh9_conjugation_screen_full.csv"))}
g = np.array([gaps[i] for i in idxs])
y = (g <= np.median(g)).astype(int)                    # 1 = small gap (reactive)
print(f"{len(y)} molecules | small-gap {y.sum()} / large-gap {(1-y).sum()}")

# --- standardize features ----------------------------------------------------
mu, sd = X.mean(0), X.std(0)
sd[sd < 1e-9] = 1.0
X = (X - mu) / sd

# --- stratified 70/30 split --------------------------------------------------
def split(y, frac=0.30):
    te = []
    for c in (0, 1):
        c_idx = np.where(y == c)[0]
        RNG.shuffle(c_idx)
        te.append(c_idx[:int(len(c_idx) * frac)])
    te = np.concatenate(te)
    mask = np.zeros(len(y), bool); mask[te] = True
    return ~mask, mask
tr, te = split(y)

# --- logistic-loss training (the classifier's decision rule) ----------------
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

w = np.zeros(X.shape[1]); b = 0.0
lr, l2, iters = 0.35, 2e-3, 300
hist = {"tr_loss": [], "te_loss": [], "tr_acc": [], "te_acc": []}

def bce(p, yy): return float(-np.mean(yy*np.log(p+1e-12) + (1-yy)*np.log(1-p+1e-12)))

for _ in range(iters):
    p = sigmoid(X[tr] @ w + b)
    g = X[tr].T @ (p - y[tr]) / len(tr) + l2 * w
    gb = float(np.mean(p - y[tr]))
    w -= lr * g; b -= lr * gb
    ptr, pte = sigmoid(X[tr] @ w + b), sigmoid(X[te] @ w + b)
    hist["tr_loss"].append(bce(ptr, y[tr]))
    hist["te_loss"].append(bce(pte, y[te]))
    hist["tr_acc"].append(float(np.mean((ptr > .5) == y[tr]) * 100))
    hist["te_acc"].append(float(np.mean((pte > .5) == y[te]) * 100))

base = max(y[te].mean(), 1 - y[te].mean()) * 100
print(f"final: train acc {hist['tr_acc'][-1]:.1f}%  test acc {hist['te_acc'][-1]:.1f}%"
      f"  (majority baseline {base:.1f}%)")

# --- figure ------------------------------------------------------------------
INK, MUT, HAIR = "#191A24", "#565873", "#E2E2EC"
ACCENT, GRAY, AMBER = "#6C5CE0", "#9A9CB4", "#D9822B"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": HAIR,
                     "axes.linewidth": 1, "text.color": INK,
                     "axes.labelcolor": MUT, "xtick.color": MUT,
                     "ytick.color": MUT, "figure.dpi": 200})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.2))
it = np.arange(1, iters + 1)

a1.plot(it, hist["tr_loss"], color=GRAY, lw=2, label="train")
a1.plot(it, hist["te_loss"], color=ACCENT, lw=2.4, label="held-out molecules")
a1.set_title("Training loss", fontsize=13, color=INK, weight="bold", loc="left", pad=10)
a1.set_xlabel("iteration"); a1.set_ylabel("logistic loss")
a1.legend(frameon=False, fontsize=10.5, loc="upper right")

a2.axhline(base, color=AMBER, lw=1.4, ls=(0, (4, 3)), label=f"guess majority ({base:.0f}%)")
a2.plot(it, hist["tr_acc"], color=GRAY, lw=2, label="train")
a2.plot(it, hist["te_acc"], color=ACCENT, lw=2.4, label="held-out molecules")
a2.set_title("Classification accuracy", fontsize=13, color=INK, weight="bold", loc="left", pad=10)
a2.set_xlabel("iteration"); a2.set_ylabel("accuracy (%)"); a2.set_ylim(45, 101)
a2.annotate(f"{hist['te_acc'][-1]:.0f}%", (it[-1], hist["te_acc"][-1]),
            color=ACCENT, fontsize=12, weight="bold", ha="right", va="bottom",
            xytext=(-4, 4), textcoords="offset points")
a2.legend(frameon=False, fontsize=10.5, loc="center right", bbox_to_anchor=(1.0, 0.42))

for ax in (a1, a2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=HAIR, lw=.8, alpha=.7)
    ax.set_axisbelow(True)

fig.suptitle("Quantum neuron learning to classify molecular thermal states",
             fontsize=15, weight="bold", color=INK, x=0.065, ha="left", y=0.99)
fig.text(0.065, 0.005,
         "1000 QH9 molecules · CAS(8,8) · kT=0.1 Ha · label: below- vs above-median HOMO–LUMO gap "
         "(balanced 50/50) · 248 thermal-state Pauli features · tested on 300 unseen molecules",
         fontsize=8.4, color=MUT, ha="left")
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
out = "figures/qh9_quantum_neuron_training.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
