# Quick Reference: PennyLane Phase 1 Optimization

## TL;DR Bottleneck Summary

| Bottleneck | Operation | Cost | Fix | Speedup |
|---|---|---|---|---|
| **#1: CRITICAL** | Eigendecomposition (eigh) | O(2^(3n)) | Cache structure | 1.5-2x |
| **#2: CRITICAL** | Manual gradient (dfj) | O(8^n × Ns) | PennyLane autodiff | 5-10x |
| **#3: HIGH** | Explicit Hamiltonian | O(2^(2n)) mem | Symbolic (Phase 2) | 10-100x |
| **#4: MEDIUM** | Trace computation | O(8^n × Ns) | Vectorize einsum | 1.5-2x |

**Total Phase 1 speedup: 5-10x** | **Total all phases: 100-500x**

---

## Phase 1: Replace Manual Gradients with Autodiff

### Old Code (Bottleneck)
```python
# Manual eigenbasis rotations - SLOW
def dfj(y, rho, eigvals, eigvecs, H_j_basis, T):
    H_j_tilde = eigvecs.T.conj() @ (H_j_basis / T) @ eigvecs  # O(2^(3n))
    rho_tilde = eigvecs.T.conj() @ rho @ eigvecs
    F = fdd_logloss_matrix(y, T, eigvals)
    return np.real(np.sum(F * H_j_tilde * rho_tilde.T))

# In training loop:
for j in range(num_params):
    for i in range(num_states):
        g_j = dfj(...)  # Called 36 × 1000 = 36K times per epoch
```

### New Code (Phase 1)
```python
import pennylane as qml

@qml.qnode(qml.device("default.qubit", wires=n), 
           diff_method="parameter-shift")
def circuit_loss(params):
    # Forward pass: compute loss
    H = sum(p * mat for p, mat in zip(params, pauli_terms))
    eval_H, evec_H = np.linalg.eigh(H)
    # ... compute logistic loss ...
    return loss_value

# Gradient via automatic differentiation
grad_fn = qml.grad(circuit_loss)
gradient = grad_fn(params)  # 5-10x faster than manual dfj
```

### Benefits
- **5-10x faster** gradient computation (autodiff vs manual eigenbasis rotations)
- **Numerically stable** (parameter-shift rule proven in literature)
- **Minimal code changes** (replace gradient loop only)
- **Scales better** (no explicit matrix constructions in QNode)

---

## Implementation Steps

### Step 1: Create Autodiff Loss Function
```python
def create_loss_fn(training_states, pauli_terms, T):
    def loss(params):
        H = sum(p * mat for p, mat in zip(params, pauli_terms))
        eval_H, evec_H = np.linalg.eigh(H)
        total = 0.0
        for i, (rho, y_label) in enumerate(training_states):
            loss_diag = T * np.log(1 + np.exp(-y_label * eval_H / T))
            m_loss = evec_H @ np.diag(loss_diag) @ evec_H.T.conj()
            total += np.real(np.trace(m_loss @ rho))
        return total / len(training_states)
    return loss
```

### Step 2: Replace Gradient Computation
```python
# OLD (in optimize loop)
grad_q = np.zeros(len(pauli_q))
for j in range(len(pauli_q)):
    g_j = sum(dfj(...) for i in range(N_states))  # SLOW
    grad_q[j] = g_j / N_states

# NEW (with PennyLane)
loss_fn = create_loss_fn(training_states, pauli_q, T)
grad_fn = qml.grad(loss_fn)
grad_q = grad_fn(est_q)  # FAST + automatic
```

### Step 3: Update Training Loop
```python
# Simple swap: replace gradient computation
for epoch in range(epochs):
    H_q = sum(p * mat for p, mat in zip(est_q, pauli_q))
    # ... loss computation (unchanged) ...
    
    # OLD: grad_q = manual_gradient_computation()
    # NEW: 
    grad_q = qml.grad(loss_fn)(est_q)
    
    est_q -= eta * grad_q  # Update (unchanged)
```

---

## Validation Checklist

- [ ] Gradient accuracy: `rel_error < 1e-4` vs manual dfj
- [ ] Loss values match: `abs(loss_new - loss_old) < 1e-6`
- [ ] Convergence behavior identical on small test (n=3)
- [ ] Training accuracies agree across epochs
- [ ] Walltime speedup measured and logged

---

## Expected Timeline

| Phase | Task | Time | Speedup |
|-------|------|------|---------|
| **1** | Replace gradients | 2-3 hours | 5-10x |
| **1b** | Validate & test | 1-2 hours | - |
| **2** | Symbolic Hamiltonian | 4-6 hours | 10-100x |
| **3** | Batch einsum | 1-2 hours | 1.5-2x |
| **4** | GPU/Catalyst | 2-3 hours | 5-20x |

---

## Troubleshooting

**Issue: Gradients don't match original**
- Check: Finite difference epsilon (default 1e-4)
- Solution: Adjust `eps` parameter in numerical gradient

**Issue: Training doesn't converge**
- Check: Learning rate may need adjustment (autodiff is exact)
- Solution: Reduce `eta` or use adaptive optimizer (Adam)

**Issue: Memory usage increased**
- Check: PennyLane circuit not being freed
- Solution: Use `cache_execute=False` or restart kernel between epochs

---

## Performance Profile (Expected)

```
n=3 qubits (8×8):
  Original:  0.5s/epoch
  Phase 1:   0.1s/epoch  (5x faster)

n=4 qubits (16×16):
  Original:  2s/epoch
  Phase 1:   0.4s/epoch  (5x faster)

n=5 qubits (32×32):
  Original:  12s/epoch
  Phase 1:   2s/epoch    (6x faster)

n=6 qubits (64×64):
  Original:  50s/epoch
  Phase 1:   10s/epoch   (5x faster)
```

---

## References

- PennyLane autodiff: https://pennylane.ai/qml/glossary/autodiff.html
- Parameter-shift rule: https://pennylane.ai/qml/glossary/parameter_shift.html
- Gradient computation methods: https://pennylane.ai/qml/glossary/grad_methods.html

---

## Next Steps After Phase 1

1. **Phase 2**: Implement symbolic Hamiltonian with `qml.Hamiltonian`
   - Avoid building 2^n × 2^n matrices
   - Expected: 10-100x speedup
   
2. **Phase 3**: Vectorize batch operations with einsum
   - Replace nested loops
   - Expected: 1.5-2x speedup
   
3. **Phase 4**: GPU acceleration with Catalyst
   - JIT compilation to XLA
   - Expected: 5-20x speedup (hardware dependent)
