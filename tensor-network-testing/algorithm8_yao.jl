# algorithm8_yao.jl
#
# Algorithm 8 (Appendix B, p. 40-41) in Yao.jl, plus the logistic loss.
#
# (1) Algorithm 8 estimates d/dtheta_j L^(2)(theta) via Eq. (B4).  Two
#     independent Hadamard tests (Fig. 10); Eq. (B9) combines their outcomes.
#
# (2) "From the gradient, the logloss": Theorem 13 / Eq. (C44) is the
#     logistic-loss analogue of Theorem 2 -- integrating the gradient formula
#     over the parameters (fundamental theorem of calculus) gives the loss
#     VALUE as an expectation of the same Hadamard-test quantities:
#
#       T Tr[ln(I + e^{-y_m H(theta)/T}) rho_m]
#           = -y_m/2 Tr[H(theta) rho_m]   +   Theta_m
#
#     The first term is Eq. (C77) (plain Pauli measurement); Theta_m is
#     estimated by Algorithm 10 with the circuit of Fig. 11.
#
# Model (matching algorithm8.jl): 1D spin-1/2 chain, ancilla on qubit 1,
# system on qubits 2..n+1.  J = 4n-3 terms:  Z_i, Z_iZ_{i+1}, X_iX_{i+1},
# Y_iY_{i+1}.  g_T(H) = tanh(H/T), so L^(2) and the logloss both have exact
# classical values at these sizes -- used at the bottom as a check.
#
# Everything lives in a module so this file and algorithm8.jl can be included
# into the same session without their shared names colliding in Main.

module Alg8Yao

using Yao
using LinearAlgebra, Random, Statistics

const P1 = matblock(ComplexF64[0 0; 0 1])   # |1><1|, the ancilla control

# ------------------------------------------------------ Hamiltonian terms ---

model_terms(n) = vcat([(:z, i) for i in 1:n],
                      [(:zz, i) for i in 1:(n - 1)],
                      [(:xx, i) for i in 1:(n - 1)],
                      [(:yy, i) for i in 1:(n - 1)])

"Sites (shifted by `off`) and Pauli factors of one term H_j."
function term_sites((kind, i), off)
  kind === :z && return [(i + off, Z)]
  kind === :zz && return [(i + off, Z), (i + 1 + off, Z)]
  kind === :xx && return [(i + off, X), (i + 1 + off, X)]
  return [(i + off, Y), (i + 1 + off, Y)]
end

"H_j as a Pauli-string block on N qubits."
pauli(N, term, off) = kron(N, (q => g for (q, g) in term_sites(term, off))...)

"H(theta) = sum_j theta_j H_j on N qubits, system starting at qubit 1+off."
ham(N, theta, terms, off) =
  sum(theta[j] * pauli(N, terms[j], off) for j in eachindex(terms))

"|1><1|_anc ox H(theta).  exp(-i * this * tau) IS the controlled evolution."
ham_ctrl(N, theta, terms) =
  sum(theta[j] * kron(N, 1 => P1, (q => g for (q, g) in term_sites(terms[j], 1))...)
      for j in eachindex(terms))

# ---------------------------------------------- probability densities -------

mu(t) = t / (2 * sinh(pi * t / 2))                    # Eq. (A2)
gam(t) = (2 / pi) * log(coth(pi * abs(t) / 2))        # Eq. (C2), two-sided

"Grid inverse-CDF sampler.  Even `npts` keeps t = 0 off the grid."
function sampler(f, lo, hi, npts)
  grid = collect(range(lo, hi; length=npts))
  cdf = cumsum(f.(grid))
  cdf ./= cdf[end]
  return rng -> grid[searchsortedfirst(cdf, rand(rng))]
end

const sample_mu = sampler(mu, -30.0, 30.0, 60_000)
const sample_gam = sampler(gam, -30.0, 30.0, 60_000)

# ------------------------------------------------------ training states -----

"rho_m = |psi_m><psi_m|: ground state of the TFIM at field h, on n qubits."
function training_state(n, h)
  hs = sum(-1.0 * kron(n, i => Z, i + 1 => Z) for i in 1:(n - 1)) +
       sum(-h * put(n, i => X) for i in 1:n)
  return eigen(Hermitian(Matrix(mat(hs)))).vectors[:, 1]
end

"Embed an n-qubit state vector as |0>_anc ox |psi> (ancilla = qubit 1 = LSB)."
with_ancilla(v) = ArrayReg(kron(v, ComplexF64[1, 0]))

# ------------------------------------------------- Hadamard test (Fig. 10) --

"""
Step 3 + the circuit of Fig. 10 for one tensor factor: prepare U^H_{st/T}(rho),
Hadamard, controlled-e^{+iHt/T}, Hadamard.  `post` is an extra controlled gate
inserted before the second Hadamard (the H_j of Fig. 11); `nothing` for Fig. 10.
"""
function hadamard_test(N, reg0, hu, hc, t, s, T; post=nothing)
  reg = copy(reg0)
  apply!(reg, time_evolve(hu, s * t / T))    # exp(-i H st/T)
  apply!(reg, put(N, 1 => Yao.H))
  apply!(reg, time_evolve(hc, -t / T))       # exp(+i |1><1| ox H t/T)
  post === nothing || apply!(reg, post)
  apply!(reg, put(N, 1 => Yao.H))
  return reg
end

"Basis rotation taking a Pauli string to the computational basis."
function rotate_to_z!(reg, N, term)
  for (q, g) in term_sites(term, 1)
    if g isa XGate
      apply!(reg, put(N, q => Yao.H))
    elseif g isa YGate
      apply!(reg, put(N, q => ConstGate.Sdag))
      apply!(reg, put(N, q => Yao.H))
    end
  end
  return reg
end

"""
Step 4's measurement: one shot of (Z_anc, H_term).  The two observables act on
disjoint qubits, so sequential projective measurement is a valid joint sample.
Returns (Z, X) with Z in {-1,1} and X in spec(H_term) = {-1,1}.
"""
function measure_zx!(reg, N, term)
  rotate_to_z!(reg, N, term)
  z = 1 - 2 * Int(measure!(reg, 1))
  x = 1
  for (q, _) in term_sites(term, 1)
    x *= 1 - 2 * Int(measure!(reg, q))
  end
  return z, x
end

"Noiseless counterpart of `measure_zx!`: (<Z>, <Z H_term>)."
function expect_zx(reg, N, term)
  za = put(N, 1 => Z)
  return real(expect(za, reg)), real(expect(chain(N, pauli(N, term, 1), za), reg))
end

# --------------------------------------------------------- Algorithm 8 ------

"""
    algorithm8(setup; j, nsamples, seed, sampled)

Steps 1-5.  `sampled=true` uses single-shot outcomes as in the paper;
`sampled=false` uses exact expectation values (same mean, far less variance).
"""
function algorithm8(st; j::Int, nsamples::Int=2000, seed::Int=1, sampled::Bool=true)
  (; N, terms, theta, rho, y, T) = st
  rng = MersenneTwister(seed)
  J, M = length(terms), length(rho)
  @assert 1 <= j <= J

  # Step 1: sample budget of Eq. (B8) (reported; the loop runs `nsamples`).
  eps, delta = 0.1, 0.05
  L = ceil(Int, (norm(theta, 1) * (1 + maximum(abs, y)) / (T^2 * eps))^2 * log(1 / delta))
  @info "Algorithm 8" j Eq_B8_budget = L running = nsamples

  q = abs.(theta) ./ norm(theta, 1)          # q(k) = |theta_k| / ||theta||_1
  cq = cumsum(q)
  Ys = Float64[]

  for _ in 1:nsamples
    # Step 2.
    k = searchsortedfirst(cq, rand(rng))
    t1, t2 = sample_mu(rng), sample_mu(rng)
    s1, s2, lam = rand(rng), rand(rng), rand(rng)
    m = rand(rng, 1:M)

    # theta^(k)(lambda) = (0,...,0, lambda*theta_k, theta_{k+1},...,theta_J).
    thk = vcat(zeros(k - 1), [lam * theta[k]], theta[(k + 1):end])

    # Steps 3-4: both blocks of Fig. 10.
    r1 = hadamard_test(N, rho[m], ham(N, thk, terms, 1), ham_ctrl(N, thk, terms), t1, s1, T)
    r2 = hadamard_test(N, rho[m], ham(N, theta, terms, 1), ham_ctrl(N, theta, terms), t2, s2, T)

    # Eq. (B9).
    if sampled
      z1, x1 = measure_zx!(r1, N, terms[k])
      z2, x2 = measure_zx!(r2, N, terms[j])
      b1, b2 = (x1 - y[m]) * z1, x2 * z2
    else
      e1z, e1x = expect_zx(r1, N, terms[k])
      _, e2x = expect_zx(r2, N, terms[j])
      b1, b2 = e1x - y[m] * e1z, e2x
    end
    push!(Ys, 2 * norm(theta, 1) / T^2 * sign(theta[k]) * b1 * b2)
  end

  # Step 5.
  return mean(Ys), std(Ys) / sqrt(nsamples)
end

# ------------------------------- logistic loss from the gradient (C44) ------

"Eq. (C77): estimate -y_m/2 Tr[H(theta) rho_m] by sampling j ~ |theta_j|/||theta||_1."
function first_term(st, m, rng, nsamples)
  (; N, terms, theta, rho, y) = st
  cq = cumsum(abs.(theta) ./ norm(theta, 1))
  vals = map(1:nsamples) do _
    j = searchsortedfirst(cq, rand(rng))
    r = copy(rho[m])
    rotate_to_z!(r, N, terms[j])
    x = prod(1 - 2 * Int(measure!(r, q)) for (q, _) in term_sites(terms[j], 1))
    -norm(theta, 1) * y[m] / 2 * sign(theta[j]) * x
  end
  return mean(vals)
end

"""
Algorithm 10 (steps 1-5): estimate Theta_m of Eq. (C79) with the circuit of
Fig. 11 -- same Hadamard test, but the controlled unitary is H_j e^{i y_m H t/T}
and the measured observable is H_k.
"""
function algorithm10(st, m, rng, nsamples)
  (; N, terms, theta, rho, y, T) = st
  J = length(terms)
  Ws = Float64[]

  for _ in 1:nsamples
    # Step 2: j uniform on [J]; s, lambda uniform; t ~ gamma; k ~ q_{j,lambda}.
    j = rand(rng, 1:J)
    s, lam, t = rand(rng), rand(rng), sample_gam(rng)
    thj = vcat(zeros(j - 1), [lam * theta[j]], theta[(j + 1):end])
    n1 = norm(thj, 1)                              # ||theta^(j)(lambda)||_1
    iszero(n1) && (push!(Ws, 0.0); continue)
    k = searchsortedfirst(cumsum(abs.(thj) ./ n1), rand(rng))

    # Steps 3-4: Hamiltonian is y_m * H(theta^(j)(lambda)); H_j rides inside
    # the controlled block (it is both unitary and Hermitian).
    hu = ham(N, y[m] .* thj, terms, 1)
    hc = ham_ctrl(N, y[m] .* thj, terms)
    sj = term_sites(terms[j], 1)
    gj = length(sj) == 1 ? sj[1][2] : kron((g for (_, g) in sj)...)
    cHj = control(N, 1, Tuple(q for (q, _) in sj) => gj)
    r = hadamard_test(N, rho[m], hu, hc, t, s, T; post=cHj)

    z, x = measure_zx!(r, N, terms[k])
    push!(Ws, J * n1 / (2 * T) * s * sign(theta[k]) * z * x)   # Eq. (C83)
  end
  return mean(Ws)
end

"L^log(theta) = (1/M) sum_m T Tr[ln(I + e^{-y_m H(theta)/T}) rho_m], via (C44)."
function logloss(st; nsamples::Int=4000, seed::Int=7)
  rng = MersenneTwister(seed)
  per_m = [first_term(st, m, rng, nsamples) + algorithm10(st, m, rng, nsamples)
           for m in eachindex(st.rho)]
  return mean(per_m)
end

# ------------------------------------------------- exact classical checks ---

"f(A) for Hermitian A by eigendecomposition."
matfun(f, A) = (F = eigen(Hermitian(A)); F.vectors * Diagonal(f.(F.values)) * F.vectors')

"Exact L^(2)(theta) = (1/M) sum_m (Tr[tanh(H/T) rho_m] - y_m)^2."
function exact_sqloss(st, theta)
  A = Matrix(mat(ham(st.n, theta, st.terms, 0)))
  g = matfun(x -> tanh(x / st.T), A)
  return mean((real(dot(v, g, v)) - st.y[m])^2 for (m, v) in enumerate(st.vecs))
end

"Exact logistic loss, same convention as `logloss`."
function exact_logloss(st, theta)
  A = Matrix(mat(ham(st.n, theta, st.terms, 0)))
  return mean(begin
                Lm = st.T * matfun(x -> log1p(exp(-st.y[m] * x / st.T)), A)
                real(dot(v, Lm, v))
              end for (m, v) in enumerate(st.vecs))
end

# ------------------------------------------------------------------ setup ---

function setup(; n::Int=4, T::Float64=2.0, seed::Int=1234)
  terms = model_terms(n)
  theta = randn(MersenneTwister(seed), length(terms))
  vecs = [training_state(n, h) for h in (0.5, 1.0, 1.5)]
  return (; n, N=n + 1, T, terms, theta,
          vecs, rho=[with_ancilla(v) for v in vecs],
          y=[1.0, -1.0, 1.0])                       # y_m in {-1,+1}
end

# --------------------------------------------------------------------- demo --

st = setup(; n=4)
j = 1

g, se = algorithm8(st; j, nsamples=4000, sampled=true)
h = 1e-4
tp = copy(st.theta); tp[j] += h
tm = copy(st.theta); tm[j] -= h
println("d/dtheta_$j L^(2)  estimate = $(round(g, digits=4)) +/- $(round(se, digits=4))")
println("                   exact    = $(round((exact_sqloss(st, tp) - exact_sqloss(st, tm)) / 2h, digits=4))")

println("logloss            estimate = $(round(logloss(st; nsamples=4000), digits=4))")
println("                   exact    = $(round(exact_logloss(st, st.theta), digits=4))")

end  # module Alg8Yao
