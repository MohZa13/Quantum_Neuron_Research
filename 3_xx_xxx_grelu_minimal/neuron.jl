# neuron.jl — the quantized GReLU neuron (He, Liu & Wilde, "Fermi–Dirac machines as
# quantizations of neurons", arXiv:2605.24386, Sec. IV.B), trained with
# Algorithm 8, and the Algorithm 8 circuit simulated shot by shot.
#
# A classical neuron computes ReLU(w·x).  The quantum version takes a density
# matrix rho and a Hamiltonian built from trainable weights,
#
#     H(theta) = theta_0 I + sum_j theta_j H_j     (H_j = Pauli strings: Z, ZZ, XX, YY)
#
# and its output is   Tr[ GReLU_T(H(theta)) rho ],   where
#
#     GReLU_T(x) = x Phi(x/T) + T phi(x/T)      (a smoothed ReLU; Phi, phi =
#                                               standard normal cdf and pdf).
#
# Training (Algorithm 8).  Squared loss
#
#     L(theta) = mean_m (Tr[GReLU_T(H) rho_m] - t_m)^2,    t = 0 (XX) or 1 (XXX).
#
# Algorithm 8 is a quantum circuit whose measured outcomes average to dL/dtheta
# (Section 2 below simulates it shot by shot).  Training uses that average
# directly: the EXACT gradient, which is what the circuit gives in the limit
# of infinitely many runs.  The identity term theta_0 is a bias, needed so the
# output can sit near 0 for XX states and near 1 for XXX states.
#
# Classification.  Predict XXX when the output is > 1/2.  Algorithm 5
# ("firing" the neuron) uses ONE copy of rho per run and returns a random
# number whose average is the neuron output.  In the eigenbasis of H, one run
# is: pick an eigenvalue E with probability <E|rho|E>, then return
# ReLU(E + T z) with z ~ N(0, 1).

using LinearAlgebra, Random, SparseArrays, Statistics
using SpecialFunctions: erf

# ------------------------------------------------------------- activation ---

Phi(x) = (1 + erf(x / sqrt(2))) / 2
phi(x) = exp(-x^2 / 2) / sqrt(2pi)
grelu(x, T) = x * Phi(x / T) + T * phi(x / T)

# ---------------------------------------------------- Pauli strings and H ---

const PAULI = Dict('I' => ComplexF64[1 0; 0 1], 'X' => ComplexF64[0 1; 1 0],
                   'Y' => ComplexF64[0 -im; im 0], 'Z' => ComplexF64[1 0; 0 -1])

"Sparse matrix of a Pauli string on n qubits, e.g. ops = [(3, 'X'), (4, 'X')]."
function pauli(n, ops)
  factors = [sparse(PAULI['I']) for _ in 1:n]
  for (site, P) in ops
    factors[site] = sparse(PAULI[P])
  end
  return real(reduce(kron, factors))        # every string we use is real
end

"""
The neuron's terms: the identity (bias), then Z_i, Z_iZ_{i+1}, X_iX_{i+1},
Y_iY_{i+1} (38 terms at n = 10).  Returns (names, matrices).
"""
function neuron_terms(n)
  names, ops = String["I"], Vector{Tuple{Int,Char}}[Tuple{Int,Char}[]]
  for i in 1:n
    push!(names, "Z$i"); push!(ops, [(i, 'Z')])
  end
  for P in "ZXY", i in 1:(n - 1)
    push!(names, "$P$i$P$(i+1)"); push!(ops, [(i, P), (i + 1, P)])
  end
  return names, [pauli(n, o) for o in ops]
end

hamiltonian(terms, theta) = Matrix(sum(t .* Hj for (t, Hj) in zip(theta, terms)))

# ------------------------------------------------- the states, seen by H ---
#
# Every input state is rho_m = W diag(p_m) W' (its chain's eigenvectors W and
# Boltzmann weights p_m; see the data section).  With H = V diag(E) V', the
# populations of rho_m on H's eigenvectors are (A.^2) p_m, with A = V'W.

"For each state m in `idx`: its populations <E|rho_m|E> on the eigenvectors V of H."
function populations(ds, idx, V)
  pops = zeros(size(V, 1), length(idx))
  for c in unique(ds.chain[idx])
    A2 = (V' * ds.chains[c].W) .^ 2
    for (i, m) in enumerate(idx)
      ds.chain[m] == c && (pops[:, i] = A2 * boltzmann(ds.chains[c], ds.beta[m]))
    end
  end
  return pops
end

targets(ds, idx) = (ds.y[idx] .+ 1) ./ 2          # 0 for XX, 1 for XXX

# ------------------------------------------------ exact loss and gradient ---
#
# Derivative of a matrix function (Daleckii–Krein):
#   d/dtheta_j Tr[f(H) R] = Tr[H_j V (G ∘ V'RV) V'],   G_ab = (f(E_a) - f(E_b)) / (E_a - E_b)

function divided_differences(E, T)
  G = similar(E, length(E), length(E))
  for (k, b) in enumerate(E), (i, a) in enumerate(E)
    G[i, k] = abs(a - b) < 1e-9 ? Phi((a + b) / 2T) :          # a ≈ b: f'(a) = Phi(a/T)
                                  (grelu(a, T) - grelu(b, T)) / (a - b)
  end
  return G
end

gradient_from(terms, V, K) = (M = V * K * V'; [dot(Hj, M) for Hj in terms])


"Algorithm 8's loss and its exact gradient, on the states `idx`."
function square_loss_grad(ds, idx, terms, theta, T)
  E, V = eigen(Symmetric(hamiltonian(terms, theta)))
  chains = unique(ds.chain[idx])
  A = Dict(c => V' * ds.chains[c].W for c in chains)     # each chain's eigenvectors, seen by H
  p = [boltzmann(ds.chains[ds.chain[m]], ds.beta[m]) for m in idx]
  out = [dot(grelu.(E, T), (A[ds.chain[m]] .^ 2) * p[i]) for (i, m) in enumerate(idx)]
  r = out .- targets(ds, idx)                            # residuals vs targets {0, 1}
  # gradient of mean(r^2) = gradient of Tr[GReLU(H) Rc] with Rc = (2/M) sum_m r_m rho_m
  Rc = zeros(size(V))
  for c in chains
    q = sum(2r[i] / length(idx) .* p[i] for (i, m) in enumerate(idx) if ds.chain[m] == c)
    Rc .+= A[c] * Diagonal(q) * A[c]'
  end
  return mean(abs2, r), gradient_from(terms, V, divided_differences(E, T) .* Rc)
end

"""
Exact decision scores on the states `idx` (the infinite-copy limit):
Tr[GReLU(H) rho] - 1/2.  Predict XXX (+1) when the score is >= 0.
"""
function scores(ds, idx, terms, theta, T)
  E, V = eigen(Symmetric(hamiltonian(terms, theta)))
  return populations(ds, idx, V)' * grelu.(E, T) .- 0.5
end

# ---------------------------------------------------- firing (Algorithm 5) ---

"One firing of the neuron on one copy of the state: ReLU(E + T z)."
function fire(E, cdf, T, rng)
  i = min(searchsortedfirst(cdf, rand(rng)), length(E))
  return max(E[i] + T * randn(rng), 0.0)
end

"Classify one state from `ncopies` firings: XXX (+1) if the average output is > 1/2."
function classify_by_firing(E, pop, T, ncopies, rng)
  cdf = cumsum(max.(pop, 0) ./ sum(max.(pop, 0)))
  return mean(fire(E, cdf, T, rng) for _ in 1:ncopies) > 0.5 ? 1.0 : -1.0
end

# -------------------------------------------- Algorithm 8, shot by shot ---

"One run of a circuit whose ±1 outcome has mean e."
shot(rng, e) = rand(rng) < (1 + clamp(e, -1, 1)) / 2 ? 1.0 : -1.0

"Draw a term index k with probability |theta_k| / ||theta||_1 (cq = its cumulative sum)."
sampleq(rng, cq) = min(searchsortedfirst(cq, rand(rng)), length(cq))

"Tr[H_j rho_m] for every state m in `idx` and term j (does not depend on theta)."
function pauli_expectations(ds, idx, terms)
  C = zeros(length(idx), length(terms))
  for c in unique(ds.chain[idx])
    W = ds.chains[c].W
    D = [vec(sum(W .* (Hj * W); dims=1)) for Hj in terms]   # <w|H_j|w> per eigenvector
    for (i, m) in enumerate(idx)
      ds.chain[m] == c || continue
      p = boltzmann(ds.chains[c], ds.beta[m])
      C[i, :] = [dot(p, d) for d in D]
    end
  end
  return C
end

"""
Value block, one run on one copy of rho_m: an unbiased estimate of Tr[GReLU_T(H) rho_m].
`E2pop` = E.^2 .* <E|rho_m|E>,  `Cm` = Tr[H_j rho_m].
"""
function value_shot(E, E2pop, Cm, theta, cq, T, rng)
  nt1 = norm(theta, 1)
  k = sampleq(rng, cq)                                     # theta picks the measured term
  a = nt1 / 2 * sign(theta[k]) * shot(rng, Cm[k])           # estimates Tr[H rho] / 2
  lam, v, t = rand(rng), rand(rng), randn(rng) / T
  e = sum(E2pop[i] * cos(lam * v * t * E[i]) for i in eachindex(E)) / nt1^2   # Hadamard test
  return T / sqrt(2pi) + a + nt1^2 / (sqrt(2pi) * T) * lam * shot(rng, e)
end

"""
The value block alone, run many times on one state m: the running estimate of
Tr[GReLU_T(H) rho_m] after each number of runs in `runs`, and the exact value.
"""
function value_block_estimates(ds, m, Cm, terms, theta, T; runs, rng)
  E, V = eigen(Symmetric(hamiltonian(terms, theta)))
  pop = populations(ds, [m], V)[:, 1]
  E2pop, cq = E .^ 2 .* pop, cumsum(abs.(theta) ./ norm(theta, 1))
  total, out = 0.0, Float64[]
  for n in 1:maximum(runs)
    total += value_shot(E, E2pop, Cm, theta, cq, T, rng)
    n in runs && push!(out, total / n)
  end
  return out, dot(grelu.(E, T), pop)
end

"""
Gradient block's Hadamard test (Theorem 17, paper Fig. 11), for every j at once:
Re Tr[H_k H_j e^{iH tau} e^{-iH s tau} rho e^{iH s tau}].  `Rt` = V' rho V.
"""
function hadamard_thm17(terms, E, V, Rt, s, tau, k)
  X = V * (Rt .* cos.(tau .* ((1 - s) .* E .+ s .* E'))) * V'
  Y = X * terms[k]
  return [dot(Hj, Y) for Hj in terms]                      # Tr[H_j X H_k]
end

"""
Simulated Algorithm 8: `nruns` runs of the circuit on the training states `idx`.
Each run draws a state m and outputs 2 b1 b2_j for every j, from two independent
copies of rho_m.  Returns the outputs as a (terms × runs) matrix; their mean
estimates the squared-loss gradient.
"""
function alg8_circuit_runs(ds, idx, C, terms, theta, T; nruns, rng)
  E, V = eigen(Symmetric(hamiltonian(terms, theta)))         # the circuit is built from H(theta)
  A = Dict(c => V' * ds.chains[c].W for c in unique(ds.chain[idx]))
  t = targets(ds, idx)
  nt1 = norm(theta, 1)
  cq = cumsum(abs.(theta) ./ nt1)
  pref = sqrt(2 / pi) * nt1 / T
  out = zeros(length(theta), nruns)
  for r in 1:nruns
    i = rand(rng, eachindex(idx))
    Ac, p = A[ds.chain[idx[i]]], boltzmann(ds.chains[ds.chain[idx[i]]], ds.beta[idx[i]])
    # copy 1: value block
    b1 = value_shot(E, E .^ 2 .* ((Ac .^ 2) * p), C[i, :], theta, cq, T, rng) - t[i]
    # copy 2: gradient block
    s, v, tt = rand(rng), rand(rng), randn(rng) / T
    k = sampleq(rng, cq)
    e = hadamard_thm17(terms, E, V, Ac * Diagonal(p) * Ac', s, v * tt, k)
    for j in eachindex(theta)
      b2 = shot(rng, C[i, j]) / 2 + pref * s * sign(theta[k]) * shot(rng, e[j])
      out[j, r] = 2 * b1 * b2
    end
  end
  return out
end
