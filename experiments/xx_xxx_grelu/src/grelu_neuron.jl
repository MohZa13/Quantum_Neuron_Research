# grelu_neuron.jl
#
# The quantized Gaussian-smoothed ReLU (GReLU) neuron of He, Liu & Wilde,
# "Fermi-Dirac machines as quantizations of neurons" (Sec. IV.B, App. F.3-F.4),
# trained with the GReLU versions of Algorithms 8 and 9 on MIXED input states.
#
#   GReLU_T(x) = x Phi(x/T) + T phi(x/T)              Eq. (92)
#   d/dx GReLU_T(x) = Phi(x/T)                        Eq. (96)
#
# Theorem 17 / Eq. (F84) is the GReLU analogue of Theorem 5 (the one Algorithm 9
# samples).  The logistic loss is softplus of the margin, T ln(1+e^{-yH/T}) =
# r_T(-yH), so "gaussifying" it replaces softplus by GReLU:
#
#   Alg 9 (margin loss)  L_9(theta) = (1/M) sum_m Tr[GReLU_T(-y_m H(theta)) rho_m]
#
#     d/dtheta_j Tr[GReLU_T(-y H) rho] = -y/2 Tr[H_j rho]
#        + sqrt(2/pi) ||theta||_1/T  E_{t~nu_T, v,s~U, k~q}
#            [ s Re Tr[ sgn(theta_k) H_k H_j e^{i y H v t} U^{yH}_{s v t}(rho) ] ]
#
#   i.e. Algorithm 9's circuit (Fig. 11) with (t/T, t~gamma) -> (v t, t~nu_T,
#   v~U) and prefactor 1/(2T) -> sqrt(2/pi)/T.  nu_T is N(0, 1/T^2), Eq. (F59).
#   Decision rule: L_m(+1) - L_m(-1) = -Tr[H rho_m], so predict sign Tr[H rho_m].
#
#   Alg 8 (squared loss) L_8(theta) = (1/M) sum_m (Tr[GReLU_T(H) rho_m] - yt_m)^2
#
#     with targets yt in {0, 1} (GReLU >= 0, so +-1 targets are unreachable).
#     As in Appendix B, one sample is 2 * b1 * b2 from two independent blocks
#     on copies of rho_m: b1 estimates Tr[GReLU(H) rho_m] - yt_m, b2 estimates
#     d/dtheta_j Tr[GReLU(H) rho_m] (Theorem 17 with y = +1).  The value comes
#     from the fundamental theorem of calculus along lambda*H (Lemma 5):
#
#       Tr[GReLU_T(H) rho] = T/sqrt(2pi) + 1/2 Tr[H rho]
#          + ||theta||_1^2/(sqrt(2pi) T) E_{lambda,v~U, t~nu_T, j,k~q}
#              [ lambda sgn(theta_j) sgn(theta_k) Re Tr[H_j H_k e^{-i lambda v t H} rho] ]
#
#     Decision rule: predict XXX when Tr[GReLU(H) rho_m] > 1/2.
#
# SIMULATION.  The inputs here are 10-qubit thermal (mixed) states, so the Yao
# statevector route of algorithm9_yao.jl would need a 21-qubit purification per
# circuit.  Instead every circuit is evaluated exactly in the eigenbasis of
# H(theta), which is computed once per optimiser step: all evolutions are then
# diagonal phases, and the Hadamard-test expectation <Z_anc (x) H_k> =
# Re Tr[H_k U rho] is a trace.  `sampled=true` then draws the +-1 shot from that
# expectation, which is the exact outcome distribution of one run of the circuit.
#
# Two structural facts keep this cheap:
#   * every model term (Z, ZZ, XX, YY, I) is a REAL operator that commutes with
#     the global parity prod_i Z_i, so H(theta), rho_m and every product H_k H_j
#     are real and block diagonal: two 512x512 blocks at n = 10;
#   * the data Hamiltonians (XX, XXX) also conserve total S^z, so each input
#     state is stored as its Hamiltonian's eigenvectors per S^z sector, and all
#     10 temperatures of one chain share them.
#
# `exact` gradients (Daleckii-Krein divided differences) are the expected value
# of the Monte-Carlo estimators -- the infinite-shot limit -- and are what the
# unit tests compare the estimators against.

module GReLUNeuron

using LinearAlgebra, Random, Statistics
using SpecialFunctions: erf

export Basis, Pauli, Family, Problem, model_terms, term_label, grelu,
       margin_loss_grad, square_loss_grad, alg9_grelu_gradient, alg8_grelu_gradient,
       neuron_outputs, margin_scores, value_estimate, FiringSampler, fire

const BOp = Vector{Matrix{Float64}}          # one dense block per parity sector

# ------------------------------------------------------------ activation ---

Phi(x) = (1 + erf(x / sqrt(2))) / 2          # standard normal CDF, Eq. (87)
phi(x) = exp(-x^2 / 2) / sqrt(2pi)           # standard normal pdf, Eq. (88)
grelu(x, T) = x * Phi(x / T) + T * phi(x / T)
dgrelu(x, T) = Phi(x / T)

"First divided difference of GReLU_T; the derivative at the midpoint when a ~ b."
function divdiff(a, b, T)
  d = a - b
  abs(d) > 1e-7 * (1 + abs(a) + abs(b)) ? (grelu(a, T) - grelu(b, T)) / d :
                                           dgrelu((a + b) / 2, T)
end

# ---------------------------------------------------------- parity basis ---

"""
Computational basis of n qubits split by parity.  Site 1 is the MOST significant
bit, i.e. kron(sigma_1, ..., sigma_n) -- the order the MPO contraction produces.
"""
struct Basis
  n::Int
  idx::Vector{Vector{Int}}      # 0-based full index of each local row, per block
  pos::Vector{Int}              # full index + 1 -> local row in its block
  blk::Vector{Int}              # full index + 1 -> block (1 even, 2 odd)
end

function Basis(n::Int)
  par = [count_ones(x) & 1 for x in 0:(2^n - 1)]
  idx = [findall(==(p), par) .- 1 for p in (0, 1)]
  pos, blk = zeros(Int, 2^n), zeros(Int, 2^n)
  for b in 1:2, (r, x) in enumerate(idx[b])
    pos[x + 1], blk[x + 1] = r, b
  end
  return Basis(n, idx, pos, blk)
end

blocksizes(B::Basis) = length.(B.idx)
zero_op(B::Basis) = [zeros(d, d) for d in blocksizes(B)]

"Parity blocks -> dense 2^n x 2^n matrix in the full computational basis."
function to_full(B::Basis, A::BOp)
  F = zeros(2^B.n, 2^B.n)
  for b in 1:2
    ix = B.idx[b] .+ 1
    F[ix, ix] .= A[b]
  end
  return F
end

"Dense full-basis matrix -> parity blocks (the off-block part must vanish)."
to_blocks(B::Basis, F::AbstractMatrix) = [Matrix{Float64}(real(F[B.idx[b] .+ 1, B.idx[b] .+ 1])) for b in 1:2]

# ------------------------------------------------------ Hamiltonian terms ---

"J = 4n-3 terms of Alg9Yao (Z_i, Z_iZ_{i+1}, X_iX_{i+1}, Y_iY_{i+1}), plus I if `bias`."
model_terms(n; bias::Bool=false) =
  vcat(bias ? [(:id, 0)] : Tuple{Symbol,Int}[],
       [(:z, i) for i in 1:n], [(:zz, i) for i in 1:(n - 1)],
       [(:xx, i) for i in 1:(n - 1)], [(:yy, i) for i in 1:(n - 1)])

function term_label((kind, i))
  kind === :id && return "I"
  kind === :z && return "Z$i"
  kind === :zz && return "Z$(i)Z$(i+1)"
  kind === :xx && return "X$(i)X$(i+1)"
  return "Y$(i)Y$(i+1)"
end

function term_ops((kind, i))
  kind === :id && return Tuple{Int,Char}[]
  kind === :z && return [(i, 'Z')]
  kind === :zz && return [(i, 'Z'), (i + 1, 'Z')]
  kind === :xx && return [(i, 'X'), (i + 1, 'X')]
  kind === :yy && return [(i, 'Y'), (i + 1, 'Y')]
  throw(ArgumentError("unknown term $kind"))
end

"""
A Pauli string as a signed permutation inside each parity block:
P e_r = c[r] e_{to[r]}.  Only real, parity-preserving strings are allowed.
"""
struct Pauli
  term::Tuple{Symbol,Int}
  to::Vector{Vector{Int}}
  c::Vector{Vector{Float64}}
end

function Pauli(B::Basis, term)
  n, ops = B.n, term_ops(term)
  flip = 0
  for (s, o) in ops
    o in ('X', 'Y') && (flip |= 1 << (n - s))
  end
  iseven(count_ones(flip)) || throw(ArgumentError("$term does not preserve parity"))
  ny = count(o -> o[2] == 'Y', ops)
  iseven(ny) || throw(ArgumentError("$term is not a real operator"))
  to = [similar(B.idx[b]) for b in 1:2]
  c = [zeros(length(B.idx[b])) for b in 1:2]
  for b in 1:2, (r, x) in enumerate(B.idx[b])
    v = ny == 2 ? -1.0 : 1.0                 # i^2 from Y|b> = i(-1)^b |~b>
    for (s, o) in ops
      o in ('Z', 'Y') && (x >> (n - s)) & 1 == 1 && (v = -v)
    end
    to[b][r] = B.pos[(x ⊻ flip) + 1]
    c[b][r] = v
  end
  return Pauli(term, to, c)
end

"Tr[P A] = sum_r c_r A[r, to_r]."
function ptrace(P::Pauli, A::BOp)
  s = 0.0
  for b in 1:2
    to, c, Ab = P.to[b], P.c[b], A[b]
    @inbounds for r in eachindex(c)
      s += c[r] * Ab[r, to[r]]
    end
  end
  return s
end

"A * P: column r is c_r * A[:, to_r]."
function rmul(A::BOp, P::Pauli)
  return map(1:2) do b
    M = similar(A[b])
    @inbounds for r in eachindex(P.c[b])
      @views M[:, r] .= P.c[b][r] .* A[b][:, P.to[b][r]]
    end
    M
  end
end

"<w|P|w> for every column w of a block-b matrix W (rows = the block's local basis)."
function pdiag(P::Pauli, b::Int, W::AbstractMatrix)
  to, c = P.to[b], P.c[b]
  out = zeros(size(W, 2))
  @inbounds for i in axes(W, 2), r in eachindex(c)
    out[i] += W[to[r], i] * c[r] * W[r, i]
  end
  return out
end

"H(theta) = sum_j theta_j H_j as parity blocks."
function hamiltonian(B::Basis, paulis, theta)
  H = zero_op(B)
  for (P, t) in zip(paulis, theta), b in 1:2
    iszero(t) && continue
    @inbounds for r in eachindex(P.c[b])
      H[b][P.to[b][r], r] += t * P.c[b][r]
    end
  end
  return H
end

"(eigenvalues, eigenvectors) of each parity block."
diagonalise(H::BOp) = [(F = eigen(Symmetric(h)); (F.values, F.vectors)) for h in H]

# --------------------------------------------------------- thermal states ---

"""
A chain Hamiltonian that conserves S^z, stored as its eigenvectors per S^z
sector.  rho(beta) = sum_k W_k diag(e^{-beta eps_k}/Z) W_k' for every beta.
`sec[k] = (b, rows)`: the sector's parity block and local rows in that block.
"""
struct Family
  sec::Vector{Tuple{Int,Vector{Int}}}
  W::Vector{Matrix{Float64}}
  eps::Vector{Vector{Float64}}
end

"Diagonalise an S^z-conserving Hamiltonian given as parity blocks."
function Family(B::Basis, H::BOp)
  n = B.n
  sec, W, eps = Tuple{Int,Vector{Int}}[], Matrix{Float64}[], Vector{Float64}[]
  for k in 0:n
    b = iseven(k) ? 1 : 2
    rows = [r for (r, x) in enumerate(B.idx[b]) if count_ones(x) == k]
    other = setdiff(eachindex(B.idx[b]), rows)
    isempty(other) || maximum(abs, H[b][rows, other]; init=0.0) < 1e-12 ||
      throw(ArgumentError("Hamiltonian does not conserve S^z"))
    F = eigen(Symmetric(H[b][rows, rows]))
    push!(sec, (b, rows)); push!(W, F.vectors); push!(eps, F.values)
  end
  return Family(sec, W, eps)
end

"Boltzmann weights e^{-beta eps}/Z, per sector."
function boltzmann(F::Family, beta)
  emin = minimum(minimum, F.eps)
  w = [exp.(-beta .* (e .- emin)) for e in F.eps]
  Z = sum(sum, w)
  return [x ./ Z for x in w]
end

energy(F::Family, beta) = sum(dot(p, e) for (p, e) in zip(boltzmann(F, beta), F.eps))
logZ(F::Family, beta) = (e0 = minimum(minimum, F.eps);
                         -beta * e0 + log(sum(sum(exp.(-beta .* (e .- e0))) for e in F.eps)))

"sum_k W_k diag(w_k) W_k' assembled into parity blocks."
function assemble(B::Basis, F::Family, w)
  R = zero_op(B)
  for ((b, rows), Wk, wk) in zip(F.sec, F.W, w)
    R[b][rows, rows] .+= Wk * Diagonal(wk) * Wk'
  end
  return R
end

rho(B::Basis, F::Family, beta) = assemble(B, F, boltzmann(F, beta))

"diag(W_k' A W_k) per sector -- everything Tr[A rho(beta)] needs, for all beta."
sector_diag(F::Family, A::BOp) =
  [vec(sum(Wk .* (A[b][rows, rows] * Wk); dims=1)) for ((b, rows), Wk) in zip(F.sec, F.W)]

# ---------------------------------------------------------------- problem ---

"""
One labelled set of states (a training set, a CV fold, or the test set).
`C[m, j] = Tr[H_j rho_m]` is theta-independent and computed once.  `Rp`, `Rm`
are (1/M) sum over the y = +1 / -1 states: the margin loss is linear in rho,
so they are all its exact value and gradient need.
"""
struct Problem
  B::Basis
  paulis::Vector{Pauli}
  T::Float64
  fams::Vector{Family}
  fam::Vector{Int}
  beta::Vector{Float64}
  y::Vector{Float64}                 # +-1
  C::Matrix{Float64}
  Rp::BOp
  Rm::BOp
end

function Problem(B::Basis, paulis, T, fams, fam, beta, y)
  M, J = length(fam), length(paulis)
  C = zeros(M, J)
  Rp, Rm = zero_op(B), zero_op(B)
  for f in unique(fam)
    ms = findall(==(f), fam)
    F = fams[f]
    # <w|H_j|w> for each eigenvector, embedded in its parity block
    D = map(paulis) do P
      map(zip(F.sec, F.W)) do ((b, rows), Wk)
        E = zeros(length(P.c[b]), size(Wk, 2))
        E[rows, :] .= Wk
        pdiag(P, b, E)
      end
    end
    wp = [zeros(length(e)) for e in F.eps]
    wm = [zeros(length(e)) for e in F.eps]
    for m in ms
      p = boltzmann(F, beta[m])
      for j in 1:J
        C[m, j] = sum(dot(pk, dk) for (pk, dk) in zip(p, D[j]))
      end
      for k in eachindex(p)
        (y[m] > 0 ? wp : wm)[k] .+= p[k] ./ M
      end
    end
    Rp .+= assemble(B, F, wp)
    Rm .+= assemble(B, F, wm)
  end
  return Problem(B, collect(paulis), Float64(T), fams, collect(fam), collect(beta),
                 collect(Float64, y), C, Rp, Rm)
end

nstates(p::Problem) = length(p.y)

# --------------------------------------------------- exact loss + gradient ---

"Divided-difference matrix of GReLU_T on one block's spectrum."
ddmatrix(E, T) = [divdiff(a, b, T) for a in E, b in E]

"""
    margin_loss_grad(prob, theta) -> (L_9, gradient)

Exact Alg 9 objective (1/M) sum_m Tr[GReLU_T(-y_m H) rho_m] and its gradient,
by Daleckii-Krein: d Tr[f(H) R] = Tr[H_j V (f^[1] o V'RV) V'].
Uses GReLU(-x) = GReLU(x) - x, so the y = +1 divided differences are F - 1.
"""
function margin_loss_grad(prob::Problem, theta)
  eig = diagonalise(hamiltonian(prob.B, prob.paulis, theta))
  L = 0.0
  K = map(1:2) do b
    E, V = eig[b]
    Rp, Rm = V' * prob.Rp[b] * V, V' * prob.Rm[b] * V
    L += dot(grelu.(E, prob.T), diag(Rp) .+ diag(Rm)) - dot(E, diag(Rp))
    V * (ddmatrix(E, prob.T) .* (Rp .+ Rm) .- Rp) * V'
  end
  return L, [ptrace(P, K) for P in prob.paulis]
end

"Tr[GReLU_T(H) rho_m] for every state -- the neuron's output."
function neuron_outputs(prob::Problem, theta; eig=nothing)
  eig === nothing && (eig = diagonalise(hamiltonian(prob.B, prob.paulis, theta)))
  G = [V * Diagonal(grelu.(E, prob.T)) * V' for (E, V) in eig]
  v = zeros(nstates(prob))
  for f in unique(prob.fam)
    d = sector_diag(prob.fams[f], G)
    for m in findall(==(f), prob.fam)
      v[m] = sum(dot(p, dk) for (p, dk) in zip(boltzmann(prob.fams[f], prob.beta[m]), d))
    end
  end
  return v
end

"Tr[H(theta) rho_m] for every state: the Alg 9 decision score (sign = label)."
margin_scores(prob::Problem, theta) = prob.C * theta

"sum_m c_m rho_m as parity blocks."
function mix(prob::Problem, c)
  R = zero_op(prob.B)
  for f in unique(prob.fam)
    F = prob.fams[f]
    w = [zeros(length(e)) for e in F.eps]
    for m in findall(==(f), prob.fam)
      for (wk, pk) in zip(w, boltzmann(F, prob.beta[m]))
        wk .+= c[m] .* pk
      end
    end
    R .+= assemble(prob.B, F, w)
  end
  return R
end

"""
    square_loss_grad(prob, theta, yt) -> (L_8, gradient, outputs)

Exact Alg 8 objective (1/M) sum_m (Tr[GReLU_T(H) rho_m] - yt_m)^2 and its
gradient (2/M) sum_m r_m d Tr[GReLU(H) rho_m] = d Tr[GReLU(H) R_c] with the
residual-weighted mixture R_c = (2/M) sum_m r_m rho_m.
"""
function square_loss_grad(prob::Problem, theta, yt)
  eig = diagonalise(hamiltonian(prob.B, prob.paulis, theta))
  v = neuron_outputs(prob, theta; eig)
  r = v .- yt
  Rc = mix(prob, 2 .* r ./ nstates(prob))
  K = map(1:2) do b
    E, V = eig[b]
    V * (ddmatrix(E, prob.T) .* (V' * Rc[b] * V)) * V'
  end
  return mean(abs2, r), [ptrace(P, K) for P in prob.paulis], v
end

# ---------------------------------------------- Monte-Carlo (the algorithms) ---

"One run of a circuit whose +-1 outcome has mean `e`, or `e` itself if not sampled."
shot(rng, e, sampled) = sampled ? (rand(rng) < (1 + clamp(e, -1, 1)) / 2 ? 1.0 : -1.0) : e

"k ~ q(k) = |theta_k| / ||theta||_1."
sampleq(rng, cq) = min(searchsortedfirst(cq, rand(rng)), length(cq))

"V' rho_m V per block -- the input state in H(theta)'s eigenbasis."
function rho_eig(prob::Problem, eig, m::Int)
  R = rho(prob.B, prob.fams[prob.fam[m]], prob.beta[m])
  return [V' * R[b] * V for (b, (_, V)) in enumerate(eig)]
end

"Re X for X with eigenbasis entries Rt_ab e^{i omega(E_a, E_b)}, back in the computational basis."
re_evolved(eig, Rt, omega) =
  [V * (Rt[b] .* cos.(omega.(E, E'))) * V' for (b, (E, V)) in enumerate(eig)]

"""
Fig. 11 with the Theorem 17 timing: expectation of Z_anc (x) H_k after the
controlled H_j e^{iHvt}, on U_{svt}(rho), for every j:
Re Tr[H_k H_j e^{iH tau} e^{-iH s tau} rho e^{iH s tau}],  tau = v t.
(The sign y of the evolution drops out: it only enters through cos.)
"""
function hadamard_thm17(prob::Problem, eig, Rt, s, tau, k)
  X = re_evolved(eig, Rt, (a, b) -> tau * ((1 - s) * a + s * b))
  Y = rmul(X, prob.paulis[k])                # Tr[H_k H_j X] = Tr[H_j X H_k]
  return [ptrace(P, Y) for P in prob.paulis]
end

"The value circuit: Re Tr[H_j H_k e^{-i tau H} rho]."
function hadamard_value(prob::Problem, eig, Rt, tau, j, k)
  X = re_evolved(eig, Rt, (a, b) -> tau * a)
  return ptrace(prob.paulis[k], rmul(X, prob.paulis[j]))
end

"""
    alg9_grelu_gradient(prob, theta; nsamples, rng, sampled, aggregate)

GReLU Algorithm 9: an unbiased estimate of the margin-loss gradient from
`nsamples` runs.  Each run draws m uniformly, measures H_j on one copy of rho_m
(first term) and runs the Theorem 17 Hadamard test on another (zeta_j).

`aggregate=true` draws only the CLASS of m and feeds the circuit the class-
averaged state.  The circuit is linear in its input, so a uniformly drawn rho_m
and the class average give identical outcome statistics; this only skips
rotating each rho_m into the eigenbasis.
"""
function alg9_grelu_gradient(prob::Problem, theta; nsamples::Int=128,
                             rng=Random.default_rng(), sampled::Bool=true,
                             aggregate::Bool=true)
  eig = diagonalise(hamiltonian(prob.B, prob.paulis, theta))
  J, M, T = length(theta), nstates(prob), prob.T
  nt1 = norm(theta, 1)
  cq = cumsum(abs.(theta) ./ nt1)
  pref = sqrt(2 / pi) * nt1 / T
  if aggregate
    wp = sum(prob.y .> 0) / M
    Rbar = Dict(+1.0 => [V' * (prob.Rp[b] ./ wp) * V for (b, (_, V)) in enumerate(eig)],
                -1.0 => [V' * (prob.Rm[b] ./ (1 - wp)) * V for (b, (_, V)) in enumerate(eig)])
  end
  acc = zeros(J)
  for _ in 1:nsamples
    m = rand(rng, 1:M)
    Rt = aggregate ? Rbar[prob.y[m]] : rho_eig(prob, eig, m)
    s, v, t = rand(rng), rand(rng), randn(rng) / T        # t ~ nu_T = N(0, 1/T^2)
    k = sampleq(rng, cq)
    e = hadamard_thm17(prob, eig, Rt, s, v * t, k)
    for j in 1:J
      acc[j] += -prob.y[m] / 2 * shot(rng, prob.C[m, j], sampled) +
                pref * s * sign(theta[k]) * shot(rng, e[j], sampled)
    end
  end
  return acc ./ nsamples
end

"""
One run of the value block: an unbiased +-scale estimate of Tr[GReLU_T(H) rho_m]
(b1 before the target is subtracted), from two copies of rho_m -- one Pauli
measurement (k ~ q) and one value-circuit run.

`literal=true` samples (j, k) ~ q x q and runs the Hadamard test for
Re Tr[H_j H_k e^{-i tau H} rho].  The output's scale depends only on lambda, so
averaging the +-1 outcome over (j, k) first is exact in distribution: it is
Bernoulli with mean Re Tr[H^2 e^{-i tau H} rho] / ||theta||_1^2, which is
diagonal in H's eigenbasis and needs only `Rd = diag(V' rho_m V)`.  That is the
default (`literal=false`), O(2^n) per shot instead of two matrix products.
"""
function value_shot(prob::Problem, eig, Rt, Rd, m, theta, cq, rng, sampled; literal::Bool=false)
  nt1, T = norm(theta, 1), prob.T
  k = sampleq(rng, cq)
  a = nt1 / 2 * sign(theta[k]) * shot(rng, prob.C[m, k], sampled)
  lam, v, t = rand(rng), rand(rng), randn(rng) / T
  tau = lam * v * t
  if literal
    j2, k2 = sampleq(rng, cq), sampleq(rng, cq)
    sg, e = sign(theta[j2]) * sign(theta[k2]), hadamard_value(prob, eig, Rt, tau, j2, k2)
  else
    sg = 1.0
    e = sum(sum(E .^ 2 .* cos.(tau .* E) .* d) for ((E, _), d) in zip(eig, Rd)) / nt1^2
  end
  return T / sqrt(2pi) + a + nt1^2 / (sqrt(2pi) * T) * lam * sg * shot(rng, e, sampled)
end

"diag(V' rho_m V) per block: the input's populations in H(theta)'s eigenbasis."
function rho_eig_diag(prob::Problem, eig, m::Int)
  R = rho(prob.B, prob.fams[prob.fam[m]], prob.beta[m])
  return [vec(sum(V .* (R[b] * V); dims=1)) for (b, (_, V)) in enumerate(eig)]
end

"""
    alg8_grelu_gradient(prob, theta, yt; nsamples, rng, sampled)

GReLU Algorithm 8: unbiased squared-loss gradient.  Each run draws m, estimates
b1 ~ Tr[GReLU(H) rho_m] - yt_m and b2_j ~ d_j Tr[GReLU(H) rho_m] on independent
copies of rho_m, and outputs 2 b1 b2_j (Eq. (49) structure).  No aggregation:
the product of two copies is not linear in rho_m.
"""
function alg8_grelu_gradient(prob::Problem, theta, yt; nsamples::Int=128,
                             rng=Random.default_rng(), sampled::Bool=true,
                             literal::Bool=false)
  eig = diagonalise(hamiltonian(prob.B, prob.paulis, theta))
  J, M, T = length(theta), nstates(prob), prob.T
  nt1 = norm(theta, 1)
  cq = cumsum(abs.(theta) ./ nt1)
  pref = sqrt(2 / pi) * nt1 / T
  acc = zeros(J)
  for _ in 1:nsamples
    m = rand(rng, 1:M)
    Rt = rho_eig(prob, eig, m)
    b1 = value_shot(prob, eig, Rt, [diag(r) for r in Rt], m, theta, cq, rng, sampled;
                    literal) - yt[m]
    s, v, t = rand(rng), rand(rng), randn(rng) / T
    k = sampleq(rng, cq)
    e = hadamard_thm17(prob, eig, Rt, s, v * t, k)
    for j in 1:J
      b2 = shot(rng, prob.C[m, j], sampled) / 2 + pref * s * sign(theta[k]) * shot(rng, e[j], sampled)
      acc[j] += 2 * b1 * b2
    end
  end
  return acc ./ nsamples
end

"""
    value_estimate(prob, theta, m; nshots, rng) -> (mean, standard error)

Finite-shot readout of the neuron output Tr[GReLU_T(H) rho_m] -- what inference
costs on hardware.
"""
function value_estimate(prob::Problem, theta, m::Int; nshots::Int=256,
                        rng=Random.default_rng(), eig=nothing)
  eig === nothing && (eig = diagonalise(hamiltonian(prob.B, prob.paulis, theta)))
  cq = cumsum(abs.(theta) ./ norm(theta, 1))
  Rd = rho_eig_diag(prob, eig, m)
  xs = [value_shot(prob, eig, nothing, Rd, m, theta, cq, rng, true) for _ in 1:nshots]
  return mean(xs), std(xs) / sqrt(nshots)
end

# ------------------------------------ Gaussian Algorithm 5: the neuron firing ---

"""
Gaussian Algorithm 5 (paper Sec. IV.B, the GReLU version of Algorithm 5): the
neuron realised on ONE copy of rho.

  1. control qumode in the Gaussian momentum state of width T1 (density phi_T1),
     data register in rho;
  2. apply exp(i x ⊗ H(theta)/T2);
  3. measure the control's momentum p;
  4. output T2 * ReLU(p).

The coupling shifts the control's momentum by an eigenvalue of H/T2, so in the
eigenbasis of H(theta) one run is EXACTLY: draw eigenvalue E with Born
probability <E|rho|E>, then p = E/T2 + T1*Z with Z ~ N(0, 1), and output
T2 ReLU(p) = ReLU(E + T Z), T = T1 T2.  Its mean is E[ReLU(E + TZ)] =
GReLU_T(E) (Eq. 93), averaged over the populations: Tr[GReLU_T(H) rho].
Only the product T = T1 T2 enters the output distribution.

`FiringSampler` precomputes the populations for one (theta, rho_m); `sign=-1`
fires the neuron built from -H(theta) (same eigenvectors, eigenvalues -E),
which the Alg 9 model needs: Tr[H rho] = Tr[GReLU(H) rho] - Tr[GReLU(-H) rho].
"""
struct FiringSampler
  E::Vector{Float64}           # eigenvalues of H(theta), all blocks
  cdf::Vector{Float64}         # cumulative Born probabilities for rho_m
  T::Float64
end

function FiringSampler(prob::Problem, eig, m::Int)
  Rd = rho_eig_diag(prob, eig, m)
  E = reduce(vcat, [e for (e, _) in eig])
  w = max.(reduce(vcat, Rd), 0.0)            # clip ~1e-17 round-off negatives
  return FiringSampler(E, cumsum(w ./ sum(w)), prob.T)
end

"One firing of the neuron with Hamiltonian sign*H(theta): ReLU(sign*E + T Z)."
function fire(fs::FiringSampler, rng; sign::Real=1)
  i = min(searchsortedfirst(fs.cdf, rand(rng)), length(fs.E))
  return max(sign * fs.E[i] + fs.T * randn(rng), 0.0)
end

end  # module GReLUNeuron
