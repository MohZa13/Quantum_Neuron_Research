# algorithm9.jl
#
# Algorithm 9 (Appendix C.2, p. 45) with ITensorMPS: the LOGISTIC-loss gradient,
# with the training states rho_m prepared by DMRG and every circuit carried as
# a Matrix Product State.
#
# Theorem 5 / Eq. (C27):
#
#   d/dtheta_j  T Tr[ln(I + e^{-y_m H(theta)/T}) rho_m]
#       = -y_m/2 Tr[H_j rho_m]
#         + ||theta||_1/(2T) E_{s~U[0,1], k~q, t~gamma}
#             [ s Re Tr[ sgn(theta_k) H_k H_j
#                        e^{i y_m H(theta) t/T} U^{y_m H(theta)}_{st/T}(rho_m) ] ]
#
# Circuit of Fig. 11.  H_j is unitary as well as Hermitian, so it rides inside
# the controlled block and H_k is the measured observable.  Expectation values
# are taken on the final MPS rather than sampled.
#
# This backend is for validation and for reaching larger n on a single gradient
# evaluation; it is far too slow to drive an optimiser loop -- use Alg9Yao for
# that.  Both modules expose the same setup / logloss_gradient / exact_logloss
# interface, so train_alg9.jl can take either.
#
# Model: 1D spin-1/2 chain, ancilla on site 1, system on sites 2..N+1.
# J = 4N-3 terms:  Z_i, Z_iZ_{i+1}, X_iX_{i+1}, Y_iY_{i+1}.  y_m in {-1,+1}.

module Alg9ITensor

using ITensors, ITensorMPS
using LinearAlgebra, Random, Statistics

# ---------------------------------------------------------------- helpers ---

"Projector |n><n| on site index s."
function proj(s, n)
  P = ITensor(s', dag(s))
  P[s' => n, s => n] = 1.0
  return P
end

"Hadamard gate on site index s."
function hadamard(s)
  A = ITensor(s', dag(s))
  A[s' => 1, s => 1] = 1 / sqrt(2)
  A[s' => 1, s => 2] = 1 / sqrt(2)
  A[s' => 2, s => 1] = 1 / sqrt(2)
  A[s' => 2, s => 2] = -1 / sqrt(2)
  return A
end

"Control gate G on ancilla index anc (|0> -> identity, |1> -> G)."
function controlled(anc, G)
  ss = inds(G; plev=0)
  Id = reduce(*, [op("Id", s) for s in ss])
  return proj(anc, 1) * Id + proj(anc, 2) * G
end

# ------------------------------------------- high-peak-tent density (C2) ----

gam(t) = (2 / pi) * log(coth(pi * abs(t) / 2))

# Even point count keeps the t = 0 log singularity off the grid.
const GAM_GRID = collect(range(-30.0, 30.0; length=60_000))
const GAM_CDF = let c = cumsum(gam.(GAM_GRID)); c ./ c[end] end

"Sample t ~ gamma by grid inverse-CDF."
sample_gam(rng) = GAM_GRID[searchsortedfirst(GAM_CDF, rand(rng))]

# ------------------------------------------------------- Hamiltonian terms ---

model_terms(N) = vcat([(:z, i) for i in 1:N],
                      [(:zz, i) for i in 1:(N - 1)],
                      [(:xx, i) for i in 1:(N - 1)],
                      [(:yy, i) for i in 1:(N - 1)])

"Single-letter Pauli name of a two-site term kind."
pauli_name(kind) = kind === :zz ? "Z" : kind === :xx ? "X" : "Y"

function term_label((kind, i))
  kind === :z && return "Z_$i"
  kind === :zz && return "Z_$i Z_$(i+1)"
  kind === :xx && return "X_$i X_$(i+1)"
  return "Y_$i Y_$(i+1)"
end

"Pauli string of term (kind, i) as an ITensor on the system sites."
function term_op(sys, (kind, i))
  kind === :z && return op("Z", sys[i])
  o = pauli_name(kind)
  return op(o, sys[i]) * op(o, sys[i + 1])
end

"""
Bond Hamiltonians h_b (b = 1..N-1) summing to H(theta) = sum_j theta_j H_j.
Two-site terms sit on their own bond; the single-site Z terms are split evenly
over the bonds touching that site, so the sum is exact and gates stay two-site.
"""
function bond_hamiltonians(sys, theta, terms)
  N = length(sys)
  nbonds(i) = (i == 1 || i == N) ? 1 : 2
  hb = [0.0 * op("Id", sys[b]) * op("Id", sys[b + 1]) for b in 1:(N - 1)]
  for (j, (kind, i)) in enumerate(terms)
    iszero(theta[j]) && continue
    if kind === :z
      for b in (i - 1, i)
        (1 <= b <= N - 1) || continue
        other = (i == b) ? b + 1 : b
        hb[b] += (theta[j] / nbonds(i)) * op("Z", sys[i]) * op("Id", sys[other])
      end
    else
      o = pauli_name(kind)
      hb[i] += theta[j] * op(o, sys[i]) * op(o, sys[i + 1])
    end
  end
  return hb
end

"""
Apply exp(-i H tau) to psi by second-order Trotter.  If `anc` is an Index every
gate is controlled on it, which is exact because C(AB) = C(A)C(B).  Pass tau < 0
for exp(+i H |tau|).
"""
function evolve(psi, hb, tau; anc=nothing, dtmax=0.2, cutoff=1e-10, maxdim=100)
  iszero(tau) && return psi
  nsteps = max(1, ceil(Int, abs(tau) / dtmax))
  dt = tau / nsteps
  half = [exp(-im * dt / 2 * h) for h in hb]
  gates = vcat(half, reverse(half))
  anc === nothing || (gates = [controlled(anc, g) for g in gates])
  for _ in 1:nsteps
    psi = apply(gates, psi; cutoff, maxdim)
  end
  return normalize!(psi)
end

# ------------------------------------------------------ training states rho_m --

"""
rho_m = |psi_m><psi_m|, with |psi_m> the DMRG ground state of the transverse
field Ising model at field h_m -- one training example per field value.
"""
function training_state(sys, h; nsweeps=8)
  N = length(sys)
  os = OpSum()
  for i in 1:(N - 1)
    os += -1.0, "Z", i, "Z", i + 1
  end
  for i in 1:N
    os += -h, "X", i
  end
  psi0 = random_mps(sys; linkdims=4)
  _, psi = dmrg(MPO(os, sys), psi0; nsweeps, maxdim=[10, 20, 50, 100, 100],
                cutoff=1e-10, outputlevel=0)
  return psi
end

"Embed a system MPS as |0>_anc ox |psi>, giving the full (N+1)-site MPS."
with_ancilla(anc, psi) = MPS(vcat([onehot(anc => 1)], [psi[i] for i in 1:length(psi)]))

# ------------------------------------------------- Hadamard test (Fig. 11) --

"""
Everything in Fig. 11 up to (not including) the H_j gate.  None of it depends
on j, so one prefix is reused for all J gradient components -- which matters
here, because the Trotter evolution is the expensive part.
"""
function hadamard_prefix(anc, psi_m_full, hb, t, s, T; ev...)
  psi = evolve(psi_m_full, hb, s * t / T; ev...)      # U^{y_m H}_{st/T}(rho_m)
  psi = apply(hadamard(anc), psi)
  return evolve(psi, hb, -t / T; anc=anc, ev...)      # controlled e^{+i y_m H t/T}
end

"The j-dependent tail of Fig. 11: controlled-H_j, then the closing Hadamard."
function hadamard_finish(anc, mid, Hj)
  psi = apply(controlled(anc, Hj), mid)
  return apply(hadamard(anc), psi)
end

"<Z_anc * A> on the final MPS."
z_expect(psi, anc, A) = real(inner(psi, apply(op("Z", anc) * A, psi)))

# ------------------------------------------------ ancilla-free path ---------

"""
Ancilla-free evaluation of exactly what the Fig. 11 Hadamard test measures.

The ancilla exists so that HARDWARE can read off Re<phi|H_k H_j V|phi> without
access to amplitudes.  A tensor network has that access, so the ancilla and the
controlled gates are pure overhead -- and expensive overhead here, because the
ancilla sits at site 1 while the Trotter gates act on sites (i, i+1), so every
controlled gate is non-local and `apply` pays O(N) swaps for it.

Writing |phi> = e^{-i y_m H st/T}|psi_m> and V = e^{+i y_m H t/T},

    <Z_anc H_k> on the Fig. 11 output  ==  Re <H_k phi | H_j chi>,   chi = V phi

with both evolutions ordinary two-site TEBD on the system alone: no ancilla, no
controlled gates, no swaps.  This is an identity, not an approximation, so
agreement with `method=:hadamard` also tests the controlled-gate construction.

Returns the j-th value for every j; |phi> and |chi> are shared across j.
"""
function sample_overlap(psi_m, hb, t, s, T, Hops, k; ev...)
  phi = evolve(psi_m, hb, s * t / T; ev...)
  chi = evolve(phi, hb, -t / T; ev...)
  a = apply(Hops[k], phi)                         # H_k |phi>
  return [real(inner(a, apply(Hj, chi))) for Hj in Hops]
end

# --------------------------------------------------------- Algorithm 9 ------

"""
    algorithm9(st, m; nsamples, seed, method, maxdim, cutoff, dtmax)

Steps 1-5 of Algorithm 9 for training example m, returning Theta_j of Eq. (C41)
for every j at once.  The sampled (s, t, k) are shared across j, so the
evolution is simulated once per draw rather than once per (draw, j).

`method=:hadamard` runs the paper's circuit with the ancilla; `method=:overlap`
drops it (see `sample_overlap`), which is the same number in exact arithmetic
but avoids the non-local controlled gates.  Use `:overlap` for large n.

`maxdim`, `cutoff` and `dtmax` are the truncation and Trotter-step controls --
sweep them with convergence_checks.jl before trusting a result.
"""
function algorithm9(st, m::Int; nsamples::Int=64, seed::Int=1,
                    method::Symbol=:hadamard, maxdim::Int=100,
                    cutoff::Float64=1e-10, dtmax::Float64=0.2)
  method in (:hadamard, :overlap) ||
    throw(ArgumentError("method must be :hadamard or :overlap, got $method"))

  (; anc, sys, terms, theta, rho, y, T) = st
  J = length(terms)
  rng = MersenneTwister(seed)
  ev = (; maxdim, cutoff, dtmax)

  nt1 = norm(theta, 1)
  cq = cumsum(abs.(theta) ./ nt1)                  # q(k) = |theta_k| / ||theta||_1
  hb = bond_hamiltonians(sys, y[m] .* theta, terms)   # y_m H(theta)
  Hops = [term_op(sys, t) for t in terms]
  psi_m = method === :hadamard ? with_ancilla(anc, rho[m]) : rho[m]

  acc = zeros(J)
  for _ in 1:nsamples
    # Step 2: s ~ U[0,1], k ~ q, t ~ gamma.
    s, t = rand(rng), sample_gam(rng)
    k = searchsortedfirst(cq, rand(rng))
    w = nt1 / (2 * T) * s * sign(theta[k])          # Eq. (C42) prefactor

    # Steps 3-4.
    if method === :overlap
      acc .+= w .* sample_overlap(psi_m, hb, t, s, T, Hops, k; ev...)
    else
      mid = hadamard_prefix(anc, psi_m, hb, t, s, T; ev...)
      for j in 1:J
        acc[j] += w * z_expect(hadamard_finish(anc, mid, Hops[j]), anc, Hops[k])
      end
    end
  end

  # Step 5.
  return acc ./ nsamples
end

"First term of Eq. (C27): -y_m/2 Tr[H_j rho_m], for every j."
first_terms(st, m::Int) =
  [-st.y[m] / 2 * real(inner(st.rho[m], apply(term_op(st.sys, t), st.rho[m])))
   for t in st.terms]

"""
    logloss_gradient(st; nsamples, seed)

Gradient of L^log(theta) = (1/M) sum_m T Tr[ln(I + e^{-y_m H(theta)/T}) rho_m],
assembled from Eq. (C27): first term + Algorithm 9, averaged over the training
set.  Same interface as Alg9Yao.logloss_gradient.
"""
function logloss_gradient(st; nsamples::Int=64, seed::Int=1, kw...)
  M = length(st.rho)
  g = sum(first_terms(st, m) .+ algorithm9(st, m; nsamples, seed=seed + 1000m, kw...)
          for m in 1:M)
  return g ./ M
end

# ------------------------------------------------- exact classical checks ---
# Dense reference, for small N only.  Site 1 is the LEAST significant factor,
# matching the column-major ordering of `Array(prod(psi), sys...)`.

const PAULI = Dict("I" => ComplexF64[1 0; 0 1], "X" => ComplexF64[0 1; 1 0],
                   "Y" => ComplexF64[0 -im; im 0], "Z" => ComplexF64[1 0; 0 -1])

function dense_term(N, (kind, i))
  names = fill("I", N)
  if kind === :z
    names[i] = "Z"
  else
    names[i] = names[i + 1] = pauli_name(kind)
  end
  return reduce(kron, (PAULI[names[q]] for q in N:-1:1))
end

dense_ham(N, theta, terms) =
  sum(theta[j] * dense_term(N, terms[j]) for j in eachindex(terms))

"""
Dense state vector of an MPS on `sys`.  `sys[1]` is listed first, so it is the
fastest-varying (least significant) index under column-major `vec` -- which is
exactly the convention `dense_term` builds by kron-ing from site N down to 1.
"""
dense_vec(psi, sys) = vec(Array(prod(psi), sys...))

matfun(f, A) = (F = eigen(Hermitian(A)); F.vectors * Diagonal(f.(F.values)) * F.vectors')

"Exact L^log(theta), same convention as `logloss_gradient` integrates."
function exact_logloss(st, theta)
  A = dense_ham(st.n, theta, st.terms)
  return mean(begin
                Lm = st.T * matfun(x -> log1p(exp(-st.y[m] * x / st.T)), A)
                real(dot(v, Lm, v))
              end for (m, v) in enumerate(st.vecs))
end

"Exact gradient of L^log by central differences -- reference only."
function exact_logloss_gradient(st, theta; h=1e-5)
  map(eachindex(theta)) do j
    tp = copy(theta); tp[j] += h
    tm = copy(theta); tm[j] -= h
    (exact_logloss(st, tp) - exact_logloss(st, tm)) / 2h
  end
end

# ------------------------------------------------------------------ setup ---

function setup(; n::Int=4, T::Float64=2.0, seed::Int=1234)
  sites = siteinds("Qubit", n + 1)
  anc, sys = sites[1], sites[2:end]
  terms = model_terms(n)
  theta = randn(MersenneTwister(seed), length(terms))
  rho = [training_state(sys, h) for h in (0.5, 1.0, 1.5)]
  return (; n, T, sites, anc, sys, terms, theta,
          rho, vecs=[dense_vec(p, sys) for p in rho],
          y=[1.0, -1.0, 1.0])                       # y_m in {-1,+1}
end

"Rebuild `st` with new parameters (rho_m, terms, labels unchanged)."
with_theta(st, theta) = merge(st, (; theta))

# --------------------------------------------------------------------- demo --

if get(ENV, "ALG9_DEMO", "1") == "1"
  st = setup(; n=4)
  g = logloss_gradient(st; nsamples=64)
  ge = exact_logloss_gradient(st, st.theta)
  println("Alg9ITensor demo")
  println("  |g_alg9 - g_exact| / |g_exact| = ", round(norm(g - ge) / norm(ge), digits=4))
  println("  cos(g_alg9, g_exact)           = ", round(dot(g, ge) / (norm(g) * norm(ge)), digits=4))
end

end  # module Alg9ITensor
