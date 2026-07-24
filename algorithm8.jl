# algorithm8.jl
#
# Algorithm 8 (Appendix B, p. 40-41): hybrid quantum-classical estimation of
# the j-th partial derivative of the squared loss L^(2)(theta), via Eq. (B4):
#
#   d/dtheta_j L^(2) = 2||theta||_1 / T^2 *
#       E_{k~q, t1,t2~mu, s1,s2,lambda~U[0,1], m~[M]} [
#         sgn(theta_k) * Re Tr[ ((H_k - y_m I) ox H_j)
#                               (e^{iH(k,lambda)t1/T} ox e^{iH(theta)t2/T})
#                               (U^{H(k,lambda)}_{s1t1/T} ox U^{H(theta)}_{s2t2/T})
#                               (rho_m ox rho_m) ] ]
#
# The two tensor factors are independent Hadamard-test circuits (Fig. 10).
# Per block: ancilla in |0>, H, controlled-e^{+iHt/T}, H, then measure
# sigma_Z on the ancilla and the Pauli observable on the system.  For a
# Hermitian A and unitary V, that circuit satisfies E[Z * A] = Re Tr[A V sigma].
#
# Deviation from the paper, as requested: instead of sampling measurement
# outcomes in step 4, each block returns its final Matrix Product State.  The
# training states rho_m are pure states prepared by DMRG.
#
# Model: 1D spin-1/2 chain, ancilla on site 1, system on sites 2..N+1.
# J = 4N-3 terms, in this order:  Z_i (i=1..N), then Z_iZ_{i+1}, X_iX_{i+1},
# Y_iY_{i+1} (i=1..N-1).
#
# Everything lives in a module so this file and algorithm8_yao.jl can be
# included into the same session -- they share several names (sample_mu,
# model_terms, training_state, algorithm8, control) that would otherwise
# collide in Main.

module Alg8ITensor

using ITensors, ITensorMPS
using LinearAlgebra, Random

# ---------------------------------------------------------------- helpers ---

"Projector |n><n| on site index s."
function proj(s, n)
  P = ITensor(s', dag(s))
  P[s' => n, s => n] = 1.0
  return P
end

"Hadamard gate on site index s."
function hadamard(s)
  H = ITensor(s', dag(s))
  H[s' => 1, s => 1] = 1 / sqrt(2)
  H[s' => 1, s => 2] = 1 / sqrt(2)
  H[s' => 2, s => 1] = 1 / sqrt(2)
  H[s' => 2, s => 2] = -1 / sqrt(2)
  return H
end

"Control gate G on ancilla index anc (|0> -> identity, |1> -> G)."
function control(anc, G)
  ss = inds(G; plev=0)
  Id = reduce(*, [op("Id", s) for s in ss])
  return proj(anc, 1) * Id + proj(anc, 2) * G
end

# --------------------------------------------------- density mu(t), Eq (A2) --

mu(t) = iszero(t) ? 1 / pi : t / (2 * sinh(pi * t / 2))

# mu decays like |t|exp(-pi|t|/2), so +/-40 is far beyond machine precision.
const MU_GRID = collect(range(-40.0, 40.0; length=80_001))
const MU_CDF = let c = cumsum(mu.(MU_GRID)); c ./ c[end] end

"Sample t ~ mu by grid inverse-CDF."
sample_mu(rng) = MU_GRID[searchsortedfirst(MU_CDF, rand(rng))]

# ------------------------------------------------------- Hamiltonian terms ---

"The J = 4N-3 parameterised terms H_j, as (kind, site) pairs."
model_terms(N) = vcat([(:z, i) for i in 1:N],
                      [(:zz, i) for i in 1:(N - 1)],
                      [(:xx, i) for i in 1:(N - 1)],
                      [(:yy, i) for i in 1:(N - 1)])

"Single-letter Pauli name of a two-site term kind."
pauli_name(kind) = kind === :zz ? "Z" : kind === :xx ? "X" : "Y"

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
Apply exp(-i H tau) to psi by second-order Trotter (as in the ITensorMPS
TEBD tutorial).  If `anc` is an Index, every gate is controlled on it, which
is exact because C(AB) = C(A)C(B).  Pass tau < 0 for exp(+i H |tau|).
"""
function evolve(psi, hb, tau; anc=nothing, dtmax=0.2, cutoff=1e-10, maxdim=100)
  iszero(tau) && return psi
  nsteps = max(1, ceil(Int, abs(tau) / dtmax))
  dt = tau / nsteps
  half = [exp(-im * dt / 2 * h) for h in hb]
  gates = vcat(half, reverse(half))
  anc === nothing || (gates = [control(anc, g) for g in gates])
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
  H = MPO(os, sys)
  psi0 = random_mps(sys; linkdims=4)
  _, psi = dmrg(H, psi0; nsweeps, maxdim=[10, 20, 50, 100, 100], cutoff=1e-10,
                outputlevel=0)
  return psi
end

"Embed a system MPS as |0>_anc ox |psi>, giving the full (N+1)-site MPS."
with_ancilla(anc, psi) = MPS(vcat([onehot(anc => 1)], [psi[i] for i in 1:length(psi)]))

# ----------------------------------------------------- one Hadamard-test block --

"""
Step 3 + the circuit of Fig. 10 for one tensor factor.  Returns the final MPS
(ancilla on site 1) -- the paper's step-4 measurement is deliberately not done.
"""
function hadamard_test_block(anc, sys, theta, terms, psi_m, t, s, T)
  hb = bond_hamiltonians(sys, theta, terms)
  psi = with_ancilla(anc, psi_m)
  psi = evolve(psi, hb, s * t / T)              # U^H_{st/T}(rho_m), uncontrolled
  psi = apply(hadamard(anc), psi)
  psi = evolve(psi, hb, -t / T; anc=anc)        # controlled e^{+iHt/T}
  psi = apply(hadamard(anc), psi)
  return psi
end

"<Z_anc>  and  <Z_anc * A> on the block's final state."
function z_expect(psi, anc, A=nothing)
  G = A === nothing ? op("Z", anc) : op("Z", anc) * A
  return real(inner(psi, apply(G, psi)))
end

# --------------------------------------------------------------- Algorithm 8 --

"""
    algorithm8(; j, N, T, nsamples, seed)

Steps 1-5 of Algorithm 8.  Each iteration returns the two final MPSs together
with the sample value Y_ell of Eq. (B9) (computed from expectation values of
those MPSs rather than from sampled measurement outcomes).
"""
function algorithm8(; j::Int=1, N::Int=6, T::Float64=2.0, nsamples::Int=8,
                    seed::Int=1234)
  @assert N >= 2
  rng = MersenneTwister(seed)

  sites = siteinds("Qubit", N + 1)
  anc = sites[1]
  sys = sites[2:end]

  terms = model_terms(N)
  J = length(terms)
  @assert 1 <= j <= J
  theta = randn(rng, J)

  # M training examples: rho_m from DMRG, with labels y_m.
  fields = [0.5, 1.0, 1.5]
  M = length(fields)
  rho = [training_state(sys, h) for h in fields]
  y = [0.0, 1.0, -1.0]

  # Step 1: sample budget from Eq. (B8) (reported; we run `nsamples`).
  Hmax = 1.0                       # ||H_j|| = 1 for Pauli strings
  eps, delta = 0.1, 0.05
  Lbound = ceil(Int, (norm(theta, 1) * (Hmax + maximum(abs, y)) * Hmax /
                      (T^2 * eps))^2 * log(1 / delta))
  @info "Eq. (B8) sample budget" L = Lbound running = nsamples

  q = abs.(theta) ./ norm(theta, 1)         # q(k) = |theta_k| / ||theta||_1
  results = []

  for _ in 1:nsamples
    # Step 2: sample k~q, t1,t2~mu, s1,s2,lambda~U[0,1], m~[M].
    k = searchsortedfirst(cumsum(q), rand(rng))
    t1, t2 = sample_mu(rng), sample_mu(rng)
    s1, s2 = rand(rng), rand(rng)
    lambda = rand(rng)
    m = rand(rng, 1:M)

    # theta^(k)(lambda) = (0,...,0, lambda*theta_k, theta_{k+1},...,theta_J).
    theta_k = vcat(zeros(k - 1), [lambda * theta[k]], theta[(k + 1):end])

    # Steps 3-4, both blocks of Fig. 10.
    psi1 = hadamard_test_block(anc, sys, theta_k, terms, rho[m], t1, s1, T)
    psi2 = hadamard_test_block(anc, sys, theta, terms, rho[m], t2, s2, T)

    # Eq. (B9): E[Z*(X - y_m)] = <Z*H_k> - y_m<Z>, and E[Z*X] = <Z*H_j>.
    b1 = z_expect(psi1, anc, term_op(sys, terms[k])) - y[m] * z_expect(psi1, anc)
    b2 = z_expect(psi2, anc, term_op(sys, terms[j]))
    Y = 2 * norm(theta, 1) / T^2 * sign(theta[k]) * b1 * b2

    push!(results, (; psi1, psi2, Y, k, m, t1, t2, s1, s2, lambda))
  end

  # Step 5: average the Y_ell.
  grad_j = sum(r.Y for r in results) / nsamples
  return (; grad_j, samples=results, theta, sites, terms)
end

# --------------------------------------------------------------------- demo --

out = algorithm8(; j=1, N=6, nsamples=8)
@show out.grad_j
@show maxlinkdim(out.samples[end].psi1)

end  # module Alg8ITensor
