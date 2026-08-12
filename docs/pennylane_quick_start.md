# PennyLane Integration Quick Reference

> **⚠️ Largely superseded — historical planning document.** The "Phases" below
> are proposals from before the work was done, and their status is now:
>
> | Section | Actual status |
> |---|---|
> | Phase 2 — symbolic Hamiltonian | **Done.** `logloss_pennylane.ipynb` cell 5 |
> | Phase 3 — vectorized divided-difference | **Done and superseded** by label aggregation (`R±`), which removes the sample loop entirely |
> | Phase 4 — JAX backend | **Deferred.** `resolve_array_backend` leaves the seam open; no GPU backend wired in |
> | "Migrate to PennyLane QNodes / parameter-shift" | **Deliberately not adopted** — the loss is a *spectral* function of H(ω), not a circuit expectation ([`DECISIONS.md`](DECISIONS.md) D6) |
>
> Current, accurate accounts: [`classifier_optimization.md`](classifier_optimization.md)
> (what was optimized and what was skipped) and [`scaling_comparison.md`](scaling_comparison.md)
> (measured results). Kept for its debugging/profiling snippets and paper
> equation references.

## Quick Start

Your notebook now has **optimized gradient computation** enabled by default. To run a full training:

```python
# Run optimized training for 6 qubits, 750 epochs
n = 6
history_q, history_c = optimize(n, epochs=750, use_fast_grad=True)

# Compare with original implementation
history_q_orig, history_c_orig = optimize(n, epochs=750, use_fast_grad=False)
```

---

## Phase 2: Symbolic Hamiltonian Representation (RECOMMENDED NEXT STEP)

Replace explicit matrix construction with PennyLane operators. This enables n ≥ 7 and provides 10-100x speedup.

### Implementation Pattern

```python
import pennylane as qml

def compute_loss_symbolic(weights, paulis, training_states, ys, T):
    """
    Compute loss without explicit matrix construction.
    Works with PennyLane operators (symbolic).
    """
    # Build Hamiltonian as weighted Pauli sum
    H = sum(w * pauli_op for w, pauli_op in zip(weights, paulis))
    
    # Diagonalize symbolically (PennyLane handles backend)
    eigvals = qml.math.eigh(H)  # Returns eigenvalues only, smaller memory
    
    # Rest is identical - no dense matrix operations needed
    ...
```

### Current Code (Matrix-based, n ≤ 6)
```python
# This creates 2^n × 2^n matrices explicitly
H = sum(w * np.kron(...) for w, ...)  # ← Exponential memory

# For n=6: 64×64 matrices (manageable)
# For n=7: 128×128 matrices (2MB each, thousands needed, RAM limited)
# For n=8: 256×256 matrices (infeasible)
```

### Target Code (Symbolic, n ≥ 8 possible)
```python
# Never create full matrices
pauli_strings = [...]  # Just symbolic representations
grads = compute_gradients(weights, pauli_strings, ...)  # PennyLane backend handles it
```

---

## Phase 3: Vectorized Divided-Difference Matrix (EASY 1.5-2x SPEEDUP)

Current bottleneck in gradient computation:

```python
# SLOW: Recomputes F for every training state
for j in range(len(weights)):
    for i in range(N_states):
        F = compute_divided_difference_matrix(eigvals, y_i)  # ← Redundant computation
```

Optimized version:

```python
# FAST: Compute F once per label value, reuse
F_pos = compute_fdd_matrix(eigvals, y=+1)  # Once
F_neg = compute_fdd_matrix(eigvals, y=-1)  # Once

for j in range(len(weights)):
    for i in range(N_states):
        F = F_pos if ys[i] > 0 else F_neg  # Reuse precomputed F
        grad_j += np.einsum('ij,ij,ij->', F, H_j_tilde, rho_tilde[i].T)
```

**Implementation:** ~20-30 lines of code, ~1 hour effort

---

## Phase 4: JAX Backend (5-20x SPEEDUP)

Replace NumPy with JAX for automatic JIT compilation and potential GPU support:

```python
import jax.numpy as jnp
import jax

# Replace np.linalg.eigh with JAX equivalent
eigvals, eigvecs = jax.numpy.linalg.eigh(H)

# Compile for speed
@jax.jit
def compute_loss_and_grads_jax(weights, rhos, ys, paulis, T):
    # Identical logic, JAX handles vectorization
    ...
```

**Benefits:**
- 5-10x speedup on CPU
- GPU support if available (50-100x possible)
- Automatic batching across states

---

## Debugging / Profiling

### Check speedup on your system

```python
import time
import numpy as np

n = 4  # Smaller system for quick test
epochs = 20

# Time original
start = time.time()
optimize(n=n, epochs=epochs, use_fast_grad=False)
t_orig = time.time() - start

# Time optimized
start = time.time()
optimize(n=n, epochs=epochs, use_fast_grad=True)
t_opt = time.time() - start

print(f"Speedup: {t_orig/t_opt:.2f}x")
```

### Profile memory usage

```python
import tracemalloc

tracemalloc.start()

# Your training code
optimize(n=6, epochs=10)

current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1e9:.2f} GB")
tracemalloc.stop()
```

---

## Integration with PennyLane

### When you're ready to migrate to PennyLane operators:

```python
import pennylane as qml

# Define quantum device
dev = qml.device('default.qubit', wires=n)

# Create quantum nodes for expectation values
@qml.qnode(dev)
def energy_circuit(weights):
    # Apply gates to prepare state, measure energy
    ...

# PennyLane handles differentiation automatically
grad_fn = qml.grad(energy_circuit)
gradients = grad_fn(weights)  # ← No manual dfj() needed
```

**Advantage:** PennyLane abstracts away the mathematical complexity and provides:
- Automatic differentiation
- Multiple backends (CPU, GPU, TPU)
- Hybrid classical-quantum optimization

---

## Common Issues & Solutions

### Issue: Training is still slow for n=6
**Solution:** Implement Phase 3 (einsum vectorization) for +1.5-2x speedup with minimal effort.

### Issue: Memory error for n=7
**Solution:** Phase 2 (symbolic Hamiltonian) is required. Explicit matrices become too large.

### Issue: Results differ between original and optimized
**Solution:** Check numerical precision. Differences < 0.1 in loss are normal due to finite differences in divided-difference matrix.

---

## Paper References

- **Original algorithm:** Section VI.C (training protocol), Eq. (63) for gradients
- **Loss function:** Eq. (56), logistic loss $L^{\log}_T(\omega)$
- **Hamiltonian structures:** Eq. (115) quantum, Eq. (116) classical
