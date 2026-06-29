# Quantum Machine Learning Bottleneck Analysis
## Fermi-Dirac Neuron Classifier (logloss_nqubits_pennylane.ipynb)

---

## 1. CRITICAL BOTTLENECKS

### **Bottleneck #1: Explicit Hamiltonian Matrix Construction** ⚠️ HIGH PRIORITY
**Location:** `optimize()` line 239, 248  
**Problem:**
- Every epoch builds full 2^n × 2^n Hamiltonian matrices via Kronecker sums
- For n=6: 64×64 matrices per term; 36 quantum terms + 26 classical terms
- **Cost per epoch:** ~O(2^(2n) × num_parameters) = O(4096 × 62) = ~250K dense matrix operations
- 1000+ states × 750 epochs = **750M+ explicit matrix constructions**

**Impact:** Memory bandwidth + cache misses dominate runtime

**Scaling:** O(4^n) memory, O(4^n) operations per forward pass

---

### **Bottleneck #2: Full Eigendecomposition Every Epoch** ⚠️ CRITICAL
**Location:** `optimize()` lines 239-240, 248-249  
**Problem:**
- `np.linalg.eigh()` on 64×64 matrices **every epoch** for both models
- Complexity: O((2^n)³) = O(2^(3n)) per eigendecomposition
- For n=6: **~262,144 ops per eigh call × 2 models × 750 epochs = 393M eigenops**
- This is the **single largest computational cost**

**Impact:** CPU-bound; numpy/LAPACK overhead is significant

**Scaling:** O(8^n) flops

---

### **Bottleneck #3: Manual Gradient via Eigenbasis Rotations** ⚠️ HIGH PRIORITY
**Location:** `dfj()` function (lines 178–182) + `optimize()` lines 251–255  
**Problem:**
- For each parameter j ∈ {1..num_params} and each state i ∈ {1..1000}:
  - Transform to eigenbasis: `eigvecs.T.conj() @ H_j @ eigvecs` → O((2^n)³)
  - Weighted sum: `F * H_tilde * rho_tilde.T` → O((2^n)²)
- Total: **1000 states × 36 params × O(262K) = 9.4B matrix operations per epoch**
- Called in inner loop: for loop at line 253, then again for classical at line 263

**Impact:** Manual autodiff is orders of magnitude slower than automatic differentiation

**Scaling:** O(8^n × num_parameters × num_states)

---

### **Bottleneck #4: Loss Trace Computation** ⚠️ MEDIUM PRIORITY
**Location:** `optimize()` lines 242–248  
**Problem:**
- For each state: construct matrix loss `evec @ diag(...) @ evec.T.conj()` → O((2^n)³)
- Trace operation: O((2^n)²)
- Repeated for 1000 states every epoch
- Cost: ~1000 × 750 × 262K = **196.9B matrix ops**

**Impact:** Sequential nested loops; poor cache locality

**Scaling:** O(8^n × num_states × epochs)

---

### **Bottleneck #5: Accuracy Computation** ⚠️ LOWER PRIORITY
**Location:** `calculate_accuracy()` lines 166–171  
**Problem:**
- Trace every full Hamiltonian with 500 validation states
- Runs every 20 epochs
- Cost: 500 states × (750 epochs / 20) × O((2^n)²) = ~750M trace ops

**Impact:** Lower relative cost but still significant; easily parallelizable

---

## 2. SCALING ANALYSIS

| Operation | Cost | 4Q | 5Q | 6Q | 7Q |
|-----------|------|-----|--------|--------|----------|
| Eigh O(2^(3n)) | 2³ⁿ | 4K | 32K | 262K | 2.1M |
| Matrix mult O(2^(3n)) | 2³ⁿ | 4K | 32K | 262K | 2.1M |
| Trace O(2^(2n)) | 2²ⁿ | 256 | 1K | 4K | 16K |
| **Total per epoch** | ~2²ⁿ × (Ns + Np × Ns) | 1.3M | 11M | 96M | 760M |

**Current regime (n=6, 1000 states, 750 epochs):** ~72 billion elementary operations

---

## 3. PennyLane OPTIMIZATION STRATEGY

### **Phase 1: Replace Manual Gradients with Autodiff** (Biggest speedup)
```python
import pennylane as qml

# Instead of manual dfj() + eigenbasis rotations:
dev = qml.device("default.qubit", wires=n)

@qml.qnode(dev)
def circuit(params, state_index, y_label):
    # Load state and compute loss
    qml.QubitStateVector(training_states[state_index].flatten(), wires=range(n))
    # Apply parameterized Hamiltonian
    for j, coeff in enumerate(params):
        qml.PauliRot(coeff, paulis_strings[j], wires=...)
    # Measure loss observable
    return qml.expval(loss_obs)

# Automatic differentiation via parameter-shift rule
grad = qml.grad(circuit, argnum=0)
gradient = grad(params, state_idx, y_label)
```

**Benefit:**
- Eliminates manual eigenbasis rotation loops (9.4B ops → automatic)
- Parameter-shift rule: only 2 forward passes per parameter (not eigendecomposition)
- Vectorizes over batch of states
- **Expected speedup: 5-10x on gradients alone**

---

### **Phase 2: Use PennyLane's Statevector Backend with Caching**
```python
# Current: rebuild H every iteration
# PennyLane: cache Hamiltonian structure

@qml.qnode(dev, diff_method="parameter-shift", cache_execute=True)
def batch_loss(params):
    # Vectorized over all training states in one pass
    losses = []
    for state_idx in batch_indices:
        qml.QubitStateVector(..., wires=...)
        loss_val = compute_fermi_dirac_loss(params, T)
        losses.append(loss_val)
    return qml.math.sum(losses)
```

**Benefit:**
- Avoids redundant eigendecompositions for repeated forward passes
- Batch processing reduces Python overhead
- **Expected speedup: 2-3x on forward pass**

---

### **Phase 3: Avoid Explicit Matrix Construction (Critical for 7+ qubits)**
```python
# Current: H = sum(p * full_matrix) → 2^n × 2^n explicit
# PennyLane: symbolic Hamiltonian

H_terms = [
    (coeff, qml.PauliX(0) @ qml.PauliX(1)),  # nearest-neighbor XX
    (coeff, qml.PauliY(0) @ qml.PauliY(1)),  # nearest-neighbor YY
    # ... etc
]
H = qml.Hamiltonian(coeffs, ops)  # Sparse symbolic representation

@qml.qnode(dev)
def circuit(params):
    qml.QubitStateVector(state, wires=range(n))
    return qml.expval(H)
```

**Benefit:**
- Never constructs 2^n × 2^n matrices explicitly
- State operations only: O(2^n) memory/time, not O(2^(2n))
- **Expected speedup: 10-100x for n ≥ 6** (memory-bound improvement)

---

### **Phase 4: GPU Acceleration via Catalyst**
```python
# PennyLane Catalyst (compiled JAX backend)
import pennylane as qml
from pennylane.labs import catalyst

dev = qml.device("lightning.qubit", wires=n)

@catalyst.qjit
@qml.qnode(dev)
def compiled_circuit(params, state):
    qml.QubitStateVector(state, wires=range(n))
    for j, coeff in enumerate(params):
        apply_hamiltonian_term(coeff, j)
    return qml.expval(H)

# Full training loop compiled to XLA
compiled_grad = catalyst.grad(compiled_circuit)
```

**Benefit:**
- JIT compilation + fusion of operations
- Can offload to GPU (if available)
- **Expected speedup: 5-20x depending on hardware**

---

## 4. IMMEDIATE OPTIMIZATION ACTIONS

### **Quick Win #1: Batch Eigendecompositions** (30 min, 2-3x speedup)
```python
# Current: eigh(H_q) inside epoch loop
# Solution: cache eigendecomposition structure

# Pre-compute for initialization
H_q = sum(p * mat for p, mat in zip(est_q, pauli_q))
eval_q, evec_q = np.linalg.eigh(H_q)

# Inside epoch loop, only update when params change significantly
for epoch in range(epochs):
    H_q = sum(p * mat for p, mat in zip(est_q, pauli_q))
    if epoch % update_freq == 0:  # Only recompute every N epochs
        eval_q, evec_q = np.linalg.eigh(H_q)
    # Reuse eigendecomposition...
```
**Trade-off:** Introduces approximation; works best with line-search or trust-region optimizer

---

### **Quick Win #2: Vectorize State Trace Operations** (15 min, 1.5x speedup)
```python
# Current: for i in range(N_states): ... np.trace(...)
# Solution: batch traces

# Vectorized trace: Tr(AB) = sum(A * B.T)
traces = np.einsum('kij,jik->k', evec_q, 
                    evec_q.T.conj() @ diag_matrix @ evec_q)
```
**Benefit:** NumPy's einsum is optimized for tensor contractions

---

### **Quick Win #3: Use scipy.linalg.eigh_tridiagonalize for Partial Eigendecomposition** (20 min, 1.5-2x speedup)
```python
# Only need lowest/highest eigenvalues for loss computation
# Use partial eigendecomposition if only computing loss (not all eigenvectors needed)
from scipy.sparse.linalg import eigsh
eval_q, evec_q = eigsh(H_q_sparse, k=half_dim, which='both')
```
**Caveat:** Requires sparse representation of H

---

## 5. NUMERICAL CORRECTNESS CONSIDERATIONS

### **Concern #1: Autodiff Stability**
- Parameter-shift rule works for all observables (unlike VJP)
- May need to tune shift offset (typically π/2 works)
- Verified to be numerically stable in PennyLane

**Mitigation:** Test against manual gradients on small system (n=2-3)

### **Concern #2: Eigendecomposition Caching Approximation**
- Reusing old eigenvectors after parameter update introduces error
- Error grows with epoch number and parameter changes

**Mitigation:** Use trust-region optimizer with adaptive recomputation

### **Concern #3: Sparse Representation Precision**
- PauliRot uses sparse exponential; numerical errors accumulate
- May affect convergence for 1000+ epochs

**Mitigation:** Run convergence test on small subset; compare to reference

---

## 6. RECOMMENDED IMPLEMENTATION ROADMAP

| Phase | Action | Speedup | Effort | Code Changes |
|-------|--------|---------|--------|--------------|
| 1 | Replace `dfj()` with PennyLane autodiff | 5-10x | Medium | Moderate refactor |
| 2 | Use symbolic Hamiltonian (avoid matrix construction) | 10-100x | High | Major refactor |
| 3 | Implement batched forward pass | 2-3x | Low | Minor |
| 4 | Add Catalyst JIT compilation | 5-20x | Medium | Wrapper only |
| **Combined** | **All phases** | **100-500x** | **High** | **Full PennyLane port** |

---

## 7. EXAMPLE: Phase 1 Implementation (PennyLane Autodiff Only)

See accompanying notebook: `optimize_with_pennylane_autodiff.ipynb`

**Expected improvement:** 5-10x speedup on gradient computation, ~2-3x overall

**Path forward:** Implement Phase 1 first, validate accuracy, then proceed to Phases 2-4
