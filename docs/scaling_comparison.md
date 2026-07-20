# Log-loss notebook correctness and scaling comparison

## Scope and reproducibility

This report compares `notebooks/paper/logloss.ipynb` with
`notebooks/pennylane/logloss_pennylane.ipynb`. Both train a quantum
Heisenberg classifier and a classical fully connected Ising model (FCIM) using
the matrix logistic loss. The comparison covers correctness, one-pass runtime,
one-time setup, static representation memory, training-sample scaling, and
qubit scaling.

Run the correctness suite:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m unittest \
  discover -s tests -v
```

Run the full benchmark:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python \
  benchmarks/benchmark_scaling.py
```

Use `--quick` for a CI-sized subset. Raw results are in
[`benchmarks/scaling_results.csv`](benchmarks/scaling_results.csv).

The measurements below were collected on an Intel Core i5-8350U (4 cores / 8
threads), Python 3.12.3, NumPy 2.4.6, SciPy 1.18.0, and PennyLane 0.45.0. CuPy
was unavailable, so all timings are CPU timings. Each value is one complete
loss-and-gradient pass. Optimized methods use the median of repeated runs;
the expensive original method uses one run. Setup is measured separately.

Static bytes include the persistent training representation and Hamiltonian
term representation, not transient BLAS/eigensolver workspace.

## Correctness results

All 12 automated tests pass. They verify:

- Quantum and classical parameter counts.
- Dense, PennyLane-symbolic, direct-CSR, and fused-bitwise Pauli equivalence.
- Pure-vector and density-matrix energy/accuracy equivalence.
- Exact binary-label aggregation for pure and mixed states.
- Dense, aggregate, diagonal-FCIM, and Chebyshev loss/gradient parity.
- Analytic gradients against central finite differences.
- Adaptive Chebyshev accuracy at its requested tolerance.
- Complex64 agreement with complex128.
- Reproducible mini-batch Adam and recorded batch/degree metadata.
- Automatic n=11 matrix-free routing without any call to dense `eigh`.
- CPU/GPU backend selection and missing-CuPy behavior.

### Original-gradient correction

The original notebook's `dfj` helper has been corrected so the Hamiltonian basis
derivative is `dH/dω_j = H_j`, not `H_j/T`. With this fix, the dense original
gradient agrees directly with the finite-difference-validated PennyLane dense,
aggregate, and symbolic paths. The loss was already the same; the correction
removes the old temperature-scaled training-step discrepancy.

## Algorithmic comparison

Let:

- `n` be the qubit count;
- `d = 2^n` be Hilbert-space dimension;
- `M` be the number of training states;
- `Pq = 6n - 3` be the Heisenberg parameter count;
- `Pc = n(n+1)/2` be the FCIM parameter count;
- `K` be Chebyshev degree;
- `B` be Adam mini-batch size.

| Method | Persistent training/term memory | Dominant quantum pass cost | Sample dependence |
|---|---:|---:|---:|
| Original notebook | `O(Md² + Pq d²)` | up to `O(Pq M d³)` from repeated rotations | Linear or worse in `M` |
| Penny dense/vectorized | `O(Md² + Pq d²)` | `O((M+Pq)d³ + PqMd²)` | Linear in `M` |
| Penny exact aggregate | `O(d² + Pq d)` | `O(d³)` | Independent of `M` after one-time aggregation |
| Penny Chebyshev full batch | `O(Md + Pq d + KBd)` | `O(KMPq d)` | Linear in `M` |
| Penny Chebyshev + Adam | `O(Md + Pq d + KBd)` | `O(KBPq d)` per update | Linear in mini-batch `B` |
| Original FCIM | `O(Md² + Pc d²)` | Dense eigensolver and rotations | Linear or worse in `M` |
| Penny diagonal FCIM | `O(Pc d)` | `O(Pc d)` | Independent of `M` after probability aggregation |

Exact aggregation is the best small/medium-`n` full-batch algorithm. Chebyshev
is the large-`n` route because it removes `O(d²)` matrices and `O(d³)`
diagonalization, but its full-batch cost still grows with `M`. Mini-batch Adam
replaces that `M` factor with `B`.

## Scaling with training samples (`n=4`)

### Quantum pass

| Samples | Original | Dense/vectorized | Exact aggregate | Matrix-free Chebyshev | Original / exact speedup |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.0229 s | 0.00256 s | 0.00403 s | 0.00541 s | 5.7× |
| 64 | 0.1047 s | 0.00987 s | 0.00472 s | 0.0126 s | 22.2× |
| 256 | 0.3672 s | 0.0360 s | 0.00372 s | 0.0559 s | 98.6× |
| 1,000 | 1.3376 s | 0.1667 s | 0.00770 s | 0.2740 s | 173.8× |

From 16 to 1,000 samples (62.5× more data):

- Original runtime increased 58×.
- Dense/vectorized runtime increased 65×.
- Full-batch Chebyshev increased 51×.
- Exact-aggregate runtime remained approximately flat; its 0.00594-second
  one-time aggregation/setup is amortized over training.

Static quantum representation at 1,000 samples fell from 3.99 MiB in the
original notebook to 0.0156 MiB with exact aggregation, a 256× reduction.

### Classical FCIM pass

| Samples | Original dense FCIM | Diagonal FCIM | Speedup |
|---:|---:|---:|---:|
| 16 | 0.0104 s | 0.030 ms | 345× |
| 64 | 0.0447 s | 0.034 ms | 1,303× |
| 256 | 0.1703 s | 0.028 ms | 6,122× |
| 1,000 | 0.6686 s | 0.033 ms | 20,478× |

The optimized FCIM pass is independent of sample count after its one-time
basis-probability aggregation.

## Comparable qubit scaling (`M=32`)

| Qubits | Dimension | Original quantum | Exact aggregate | Chebyshev | Original / exact | Original / Chebyshev |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 0.0206 s | 0.00244 s | 0.00438 s | 8.4× | 4.7× |
| 3 | 8 | 0.0517 s | 0.00524 s | 0.00749 s | 9.9× | 6.9× |
| 4 | 16 | 0.0654 s | 0.00457 s | 0.00655 s | 14.3× | 10.0× |
| 5 | 32 | 0.1739 s | 0.0202 s | 0.0165 s | 8.6× | 10.6× |
| 6 | 64 | 6.5899 s | 0.0123 s | 0.0335 s | 535× | 197× |

The original implementation's repeated parameter/sample eigenbasis rotations
become dominant at n=6. Exact aggregation removes the sample factor, while the
Chebyshev path avoids dense spectral work but has polynomial-recurrence overhead
at these small dimensions.

## Optimized large-qubit scaling (`M=16`)

The original method is intentionally omitted beyond n=6. Running its dense
parameter/sample loops at these sizes is not a useful use of benchmark time.

### Runtime

| Qubits | Dimension | Dense/vectorized | Exact aggregate | Chebyshev | Exact / Chebyshev |
|---:|---:|---:|---:|---:|---:|
| 7 | 128 | 0.154 s | 0.0326 s | 0.0450 s | 0.72× |
| 8 | 256 | 0.748 s | 0.0531 s | 0.0748 s | 0.71× |
| 9 | 512 | 4.801 s | 0.303 s | 0.166 s | 1.82× |
| 10 | 1,024 | 28.886 s | 1.951 s | 0.285 s | 6.86× |

The empirical CPU crossover occurs at n=9 for this workload. At n≤8, exact
aggregation remains slightly faster. At n=10, adaptive Chebyshev selected degree
18 and was 6.9× faster than exact diagonalization.

### Static representation memory

| Qubits | Dense/vectorized | Exact aggregate | Chebyshev | Dense / Chebyshev |
|---:|---:|---:|---:|---:|
| 7 | 13.750 MiB | 0.614 MiB | 0.073 MiB | 189× |
| 8 | 61.000 MiB | 2.264 MiB | 0.163 MiB | 374× |
| 9 | 268.000 MiB | 8.598 MiB | 0.361 MiB | 742× |
| 10 | 1,168.000 MiB | 33.336 MiB | 0.793 MiB | 1,473× |

At n=10, Chebyshev is also 42× smaller than exact aggregation. This memory
scaling—not only pass time—is why automatic routing switches to matrix-free
Chebyshev beyond n=10.

Across all benchmark cases, adaptive Chebyshev with tolerance `1e-5` had a
maximum absolute loss difference of `1.06e-6` and a median difference of
`1.71e-7` relative to exact aggregation.

## Practical recommendations

1. **n≤8, fixed binary dataset:** use exact aggregation. It is exact, fast, and
   independent of sample count after setup.
2. **n=9–10:** benchmark both exact aggregation and Chebyshev; the crossover
   depends on CPU/GPU and sample count.
3. **n>10:** use matrix-free Chebyshev with complex64 and mini-batch Adam. Dense
   eigendecomposition and dense label aggregates should be avoided.
4. **Large M:** exact aggregation remains insensitive to M, whereas Chebyshev
   should use Adam mini-batches so per-step work is `O(B)`, not `O(M)`.
5. **Classical model:** always use the diagonal FCIM backend.
6. **Reproducible comparisons:** set NumPy and Adam seeds, record the active
   Chebyshev degree, and compare losses—not raw original gradients unless the
   original `1/T` factor is corrected.

## Limitations

- Timings are machine-specific and sensitive to BLAS threading and CPU scaling.
- GPU/fused-kernel execution is tested structurally but was not timed because
  CuPy/CUDA is unavailable on this machine.
- Static byte counts omit transient workspace and Python object overhead.
- Chebyshev is approximate; tighter tolerances increase degree and runtime.
- Mini-batch Adam changes optimization noise and iteration count, so per-pass
  speedups do not directly predict time-to-accuracy without a convergence study.
