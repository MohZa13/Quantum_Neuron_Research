# neuron.jl — the quantized GReLU neuron for a small region of k qubits (here k = 3, so every matrix is 8 × 8).
#
# What the neuron does, in plain words:
#   * It has one weight θ_j for each Pauli term H_j: the identity I (a bias), Z on each qubit, and ZZ, XX, YY on
#     each pair of neighbouring qubits.
#   * The weights build a Hamiltonian  H(θ) = Σ_j θ_j H_j.
#   * Its output on a region state ρ is  Tr[GReLU_T(H(θ)) ρ]:  the activation applied to H(θ), then averaged
#     over the state.
#   * It predicts s = +1 when the output is above 1/2, and s = −1 otherwise.
#
# Training (Algorithm 8 of He, Liu & Wilde, arXiv:2605.24386):
#   loss L(θ) = average over samples of (output − target)²,   target = 1 for s = +1 and 0 for s = −1.
# Adam lowers this loss using its exact gradient, which is what the Algorithm 8 circuit would give after
# infinitely many runs.

using LinearAlgebra, Random, Statistics
using SpecialFunctions: erf

# ---- the activation function ----
Phi(x) = (1 + erf(x / sqrt(2))) / 2            # standard normal cumulative distribution
phi(x) = exp(-x^2 / 2) / sqrt(2pi)             # standard normal bell curve
grelu(x, T) = x * Phi(x / T) + T * phi(x / T)  # smoothed ReLU

# ---- Pauli matrices ----
const PAULI = Dict('I' => ComplexF64[1 0; 0 1], 'X' => ComplexF64[0 1; 1 0],
                   'Y' => ComplexF64[0 -im; im 0], 'Z' => ComplexF64[1 0; 0 -1])
# (Row/column 1 of each matrix is spin up, Z = +1; row/column 2 is spin down, Z = −1.)

"""
A Pauli string on k qubits, e.g. pauli(3, [(1, 'X'), (2, 'X')]) = X ⊗ X ⊗ I.
Qubits not listed get the identity.  kron(A, B, C): A acts on qubit 1, the most significant bit.
"""
function pauli(k, ops)
  factors = [PAULI['I'] for _ in 1:k]
  for (q, P) in ops
    factors[q] = PAULI[P]
  end
  return reduce(kron, factors)
end

"""
The neuron's terms on a region whose qubits are the chain sites `sites` (e.g. [24, 25, 26]):
the identity (bias), Z on every qubit, then ZZ, XX and YY on every neighbouring pair.
Names use chain-site numbers, e.g. "Z25" or "X24X25".  Returns (names, matrices).
"""
function neuron_terms(sites)
  k = length(sites)
  names, ops = ["I"], [Tuple{Int,Char}[]]
  for q in 1:k
    push!(names, "Z$(sites[q])"); push!(ops, [(q, 'Z')])
  end
  for P in "ZXY", q in 1:(k - 1)
    push!(names, "$P$(sites[q])$P$(sites[q+1])"); push!(ops, [(q, P), (q + 1, P)])
  end
  return names, [pauli(k, o) for o in ops]
end

"H(θ) = Σ_j θ_j H_j."
hamiltonian(terms, theta) = Hermitian(sum(t .* Hj for (t, Hj) in zip(theta, terms)))

# ---- the neuron's output on each state ----
"Neuron outputs Tr[GReLU_T(H) ρ] for every state in `rhos`."
function outputs(terms, theta, rhos, T)
  E, V = eigen(hamiltonian(terms, theta))       # energies E and eigenvectors V of H(θ)
  g = grelu.(E, T)                              # activation of each energy
  # diag(V' ρ V) = how much of ρ sits on each eigenvector; the output is the weighted sum of activations
  return [real(dot(g, diag(V' * ρ * V))) for ρ in rhos]
end

# ---- exact loss and gradient for Algorithm 8 ----
"""
Divided differences of GReLU between every pair of energies:
G[a, b] = (f(E_a) − f(E_b)) / (E_a − E_b), and the derivative f'(E_a) = Φ(E_a/T) when E_a ≈ E_b.
"""
function divided_differences(E, T)
  G = zeros(length(E), length(E))
  for b in eachindex(E), a in eachindex(E)
    G[a, b] = abs(E[a] - E[b]) < 1e-9 ? Phi((E[a] + E[b]) / 2T) :
                                        (grelu(E[a], T) - grelu(E[b], T)) / (E[a] - E[b])
  end
  return G
end

"""
Squared loss and its exact gradient with respect to every θ_j.
  loss       = mean over samples of (output − target)²
  gradient_j = Tr[ H_j  V (G ∘ V† R V) V† ],   R = (2/M) Σ_m (output_m − target_m) ρ_m
"""
function loss_grad(terms, theta, rhos, targets, T)
  E, V = eigen(hamiltonian(terms, theta))
  out = [real(dot(grelu.(E, T), diag(V' * ρ * V))) for ρ in rhos]
  r = out .- targets                                       # how far each output is from its target
  R = sum((2 * r[m] / length(rhos)) .* rhos[m] for m in eachindex(rhos))   # residual-weighted mix of states
  K = V * (divided_differences(E, T) .* (V' * R * V)) * V'
  return mean(abs2, r), [real(dot(Hj, K)) for Hj in terms]
end

# ---- firing the neuron (Algorithm 5): one copy of ρ per firing ----
"One firing: pick energy E_a with probability ⟨a|ρ|a⟩ (via its cumulative sum `cdf`), return ReLU(E_a + T z)."
function fire(E, cdf, T, rng)
  a = min(searchsortedfirst(cdf, rand(rng)), length(E))
  return max(E[a] + T * randn(rng), 0.0)
end

"Classify one state from `ncopies` firings: s = +1 if the average firing is above 1/2."
function classify_by_firing(E, pop, T, ncopies, rng)
  cdf = cumsum(max.(pop, 0) ./ sum(max.(pop, 0)))
  return mean(fire(E, cdf, T, rng) for _ in 1:ncopies) > 0.5 ? 1 : -1
end
