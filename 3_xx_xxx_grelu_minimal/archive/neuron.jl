# neuron.jl — the quantized GReLU neuron (He, Liu & Wilde, "Fermi–Dirac machines
# as quantizations of neurons", arXiv:2605.24386, Sec. IV.B).
#
# A classical neuron computes ReLU(w·x).  The quantum version takes a density
# matrix rho and a Hamiltonian built from trainable weights,
#
#     H(theta) = sum_j theta_j H_j        (H_j = Pauli strings: Z, ZZ, XX, YY)
#
# and its output is   Tr[ GReLU_T(H(theta)) rho ],   where
#
#     GReLU_T(x) = x Phi(x/T) + T phi(x/T)      (a smoothed ReLU; Phi, phi =
#                                               standard normal cdf and pdf).
#
# Training.  Algorithms 8 and 9 of the paper are quantum circuits whose
# measured outcomes average to the gradient of a loss.  Here we use that
# average directly: the EXACT gradient, which is what the circuits give in the
# limit of infinitely many runs.  (Folder 2 simulates the circuits shot by shot.)
#
#   Algorithm 9 — margin loss   L = mean_m Tr[GReLU_T(-y_m H) rho_m]
#                 classify by the sign of Tr[H rho]
#   Algorithm 8 — squared loss  L = mean_m (Tr[GReLU_T(H) rho_m] - t_m)^2,  t = 0 (XX) or 1 (XXX)
#                 classify by output > 1/2; needs a constant (identity) term
#
# Classification.  Algorithm 5 ("firing" the neuron) uses ONE copy of rho per
# run and returns a random number whose average is the neuron output.  In the
# eigenbasis of H, one run is: pick an eigenvalue E with probability <E|rho|E>,
# then return ReLU(E + T z) with z ~ N(0, 1).

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
The neuron's terms: Z_i, Z_iZ_{i+1}, X_iX_{i+1}, Y_iY_{i+1} (37 terms at n = 10),
plus the identity first if `bias` (Algorithm 8 needs it).  Returns (names, matrices).
"""
function neuron_terms(n; bias=false)
  names, ops = String[], Vector{Tuple{Int,Char}}[]
  bias && (push!(names, "I"); push!(ops, Tuple{Int,Char}[]))
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
# Boltzmann weights p_m; see data.jl).  With H = V diag(E) V', the populations
# of rho_m on H's eigenvectors are (A.^2) p_m, with A = V'W.

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

"Class averages (1/M) sum rho_m over the XXX states and over the XX states of `idx`."
function class_averages(ds, idx)
  R = Dict(+1.0 => zeros(2^ds.n, 2^ds.n), -1.0 => zeros(2^ds.n, 2^ds.n))
  for m in idx
    ch = ds.chains[ds.chain[m]]
    R[ds.y[m]] .+= ch.W * Diagonal(boltzmann(ch, ds.beta[m]) ./ length(idx)) * ch.W'
  end
  return R[+1.0], R[-1.0]
end

# --------------------------------------------- exact losses and gradients ---
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

"""
Algorithm 9's loss and its exact gradient.  The loss is linear in the states, so it
only needs the class averages Rp (XXX states) and Rm (XX states) from `class_averages`.
"""
function margin_loss_grad(terms, theta, T, Rp, Rm)
  E, V = eigen(Symmetric(hamiltonian(terms, theta)))
  Rp, Rm = V' * Rp * V, V' * Rm * V
  # XXX states (y = +1) see GReLU(-H), whose eigenvalues are -E; XX states see GReLU(H)
  L = dot(grelu.(-E, T), diag(Rp)) + dot(grelu.(E, T), diag(Rm))
  K = divided_differences(E, T) .* Rm .- divided_differences(-E, T) .* Rp
  return L, gradient_from(terms, V, K)
end

"Algorithm 8's loss and its exact gradient, on the states `idx`."
function square_loss_grad(ds, idx, terms, theta, T)
  E, V = eigen(Symmetric(hamiltonian(terms, theta)))
  chains = unique(ds.chain[idx])
  A = Dict(c => V' * ds.chains[c].W for c in chains)     # each chain's eigenvectors, seen by H
  p = [boltzmann(ds.chains[ds.chain[m]], ds.beta[m]) for m in idx]
  out = [dot(grelu.(E, T), (A[ds.chain[m]] .^ 2) * p[i]) for (i, m) in enumerate(idx)]
  r = out .- (ds.y[idx] .+ 1) ./ 2                       # residuals vs targets {0, 1}
  # gradient of mean(r^2) = gradient of Tr[GReLU(H) Rc] with Rc = (2/M) sum_m r_m rho_m
  Rc = zeros(size(V))
  for c in chains
    q = sum(2r[i] / length(idx) .* p[i] for (i, m) in enumerate(idx) if ds.chain[m] == c)
    Rc .+= A[c] * Diagonal(q) * A[c]'
  end
  return mean(abs2, r), gradient_from(terms, V, divided_differences(E, T) .* Rc)
end

"""
Exact predictions on the states `idx` (the infinite-copy limit):
Algorithm 9 model -> score Tr[H rho];  Algorithm 8 model -> score Tr[GReLU(H) rho] - 1/2.
Predict XXX (+1) when the score is >= 0.
"""
function scores(kind, ds, idx, terms, theta, T)
  E, V = eigen(Symmetric(hamiltonian(terms, theta)))
  pops = populations(ds, idx, V)
  return kind == :margin ? pops' * E : pops' * grelu.(E, T) .- 0.5
end

# ---------------------------------------------------- firing (Algorithm 5) ---

"One firing of the neuron on one copy of the state: ReLU(E + T z)."
function fire(E, cdf, T, rng; sign=1)
  i = min(searchsortedfirst(cdf, rand(rng)), length(E))
  return max(sign * E[i] + T * randn(rng), 0.0)
end

"""
Classify one state from `ncopies` firings.
Algorithm 8 model: average output > 1/2.
Algorithm 9 model: needs Tr[H rho] = Tr[GReLU(H) rho] - Tr[GReLU(-H) rho], so it
spends half the copies on the neuron built from H and half on the one from -H.
"""
function classify_by_firing(kind, E, pop, T, ncopies, rng)
  cdf = cumsum(max.(pop, 0) ./ sum(max.(pop, 0)))
  if kind == :square
    return mean(fire(E, cdf, T, rng) for _ in 1:ncopies) > 0.5 ? 1.0 : -1.0
  end
  h = max(ncopies ÷ 2, 1)
  s = mean(fire(E, cdf, T, rng) for _ in 1:h) - mean(fire(E, cdf, T, rng; sign=-1) for _ in 1:h)
  return s >= 0 ? 1.0 : -1.0
end
