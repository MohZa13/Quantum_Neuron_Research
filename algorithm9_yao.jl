# algorithm9_yao.jl
#
# Algorithm 9 (Appendix C.2, p. 45) in Yao.jl: the LOGISTIC-loss gradient.
#
# Theorem 5 / Eq. (C27):
#
#   d/dtheta_j  T Tr[ln(I + e^{-y_m H(theta)/T}) rho_m]
#       = -y_m/2 Tr[H_j rho_m]                                    (first term)
#         + ||theta||_1/(2T) E_{s~U[0,1], k~q, t~gamma}           (= Theta_j)
#             [ s Re Tr[ sgn(theta_k) H_k H_j
#                        e^{i y_m H(theta) t/T} U^{y_m H(theta)}_{st/T}(rho_m) ] ]
#
# Algorithm 9 estimates Theta_j with the circuit of Fig. 11.  H_j is unitary as
# well as Hermitian (a Pauli string), so Re Tr[H_k H_j V rho] = Re Tr[H_k (H_j V) rho]:
# H_j rides INSIDE the controlled block and H_k is the measured observable.
#
# The first term is a single Pauli expectation on rho_m -- computed directly.
#
# Unlike Algorithm 8 this gives the gradient of the loss we actually want to
# minimise, so `logloss_gradient` can be handed straight to an optimiser.
# Algorithm 10 / Eq. (C44) is included to estimate the loss VALUE.
#
# Model: 1D spin-1/2 chain, ancilla on qubit 1, system on qubits 2..n+1.
# J = 4n-3 terms:  Z_i, Z_iZ_{i+1}, X_iX_{i+1}, Y_iY_{i+1}.  y_m in {-1,+1}.

module Alg9Yao

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

function term_label((kind, i))
  kind === :z && return "Z_$i"
  kind === :zz && return "Z_$i Z_$(i+1)"
  kind === :xx && return "X_$i X_$(i+1)"
  return "Y_$i Y_$(i+1)"
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

"Ancilla-controlled Pauli string H_j (the H_j gate of Fig. 11)."
function cpauli(N, term)
  sj = term_sites(term, 1)
  gj = length(sj) == 1 ? sj[1][2] : kron((g for (_, g) in sj)...)
  return control(N, 1, Tuple(q for (q, _) in sj) => gj)
end

# ------------------------------------------- high-peak-tent density (C2) ----

gam(t) = (2 / pi) * log(coth(pi * abs(t) / 2))

"Grid inverse-CDF sampler.  Even `npts` keeps the t = 0 log singularity off-grid."
function sampler(f, lo, hi, npts)
  grid = collect(range(lo, hi; length=npts))
  cdf = cumsum(f.(grid))
  cdf ./= cdf[end]
  return rng -> grid[searchsortedfirst(cdf, rand(rng))]
end

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

# ------------------------------------------------- Hadamard test (Fig. 11) --

"""
Everything in Fig. 11 up to (not including) the H_j gate: state preparation
U^{y_m H}_{st/T}(rho_m), Hadamard, controlled-e^{+i y_m H t/T}.  None of this
depends on j, so one prefix is reused for all J gradient components.
"""
function hadamard_prefix(N, reg0, hu, hc, t, s, T)
  reg = copy(reg0)
  apply!(reg, time_evolve(hu, s * t / T))    # exp(-i y_m H st/T)
  apply!(reg, put(N, 1 => Yao.H))
  apply!(reg, time_evolve(hc, -t / T))       # exp(+i |1><1| ox y_m H t/T)
  return reg
end

"The j-dependent tail of Fig. 11: controlled-H_j, then the closing Hadamard."
function hadamard_finish(N, mid, cHj)
  reg = copy(mid)
  apply!(reg, cHj)
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
Step 4's measurement: one shot of Z_anc * H_term.  The two observables act on
disjoint qubits, so sequential projective measurement is a valid joint sample.
"""
function measure_zx!(reg, N, term)
  rotate_to_z!(reg, N, term)
  zx = 1 - 2 * Int(measure!(reg, 1))
  for (q, _) in term_sites(term, 1)
    zx *= 1 - 2 * Int(measure!(reg, q))
  end
  return zx
end

"Noiseless counterpart of `measure_zx!`: <Z_anc H_term>."
expect_zx(reg, N, term) = real(expect(chain(N, pauli(N, term, 1), put(N, 1 => Z)), reg))

# ------------------------------------------------ ancilla-free path ---------

"""
Ancilla-free evaluation of exactly what the Fig. 11 Hadamard test measures.

The ancilla exists so that HARDWARE can read off Re<phi|H_k H_j V|phi> without
access to amplitudes.  A simulator has that access, so the ancilla, the
controlled evolution, and (in the MPS backend) the swaps it forces are pure
overhead.  Writing |phi> = e^{-i y_m H st/T}|psi_m> and V = e^{+i y_m H t/T},

    <Z_anc H_k> on the Fig. 11 output  ==  Re <H_k phi | H_j chi>,   chi = V phi

with both evolutions ordinary local time evolution on the system alone.  This
is an identity, not an approximation, so agreement with `method=:hadamard` is
also a test of the controlled-gate construction.

Returns the j-th value for every j; |phi> and |chi> are shared across j.
"""
function sample_overlap(n, psi_m, hu, t, s, T, ops, k)
  phi = copy(psi_m)
  apply!(phi, time_evolve(hu, s * t / T))     # exp(-i y_m H st/T)
  chi = copy(phi)
  apply!(chi, time_evolve(hu, -t / T))        # exp(+i y_m H t/T)

  a = copy(phi)
  apply!(a, ops[k])                           # H_k |phi>
  av = statevec(a)
  return map(ops) do Hj
    b = copy(chi)
    apply!(b, Hj)                             # H_j |chi>
    real(dot(av, statevec(b)))
  end
end

# --------------------------------------------------------- Algorithm 9 ------

"""
    algorithm9(st, m; nsamples, seed, sampled, method)

Steps 1-5 of Algorithm 9 for training example m, returning Theta_j of Eq. (C41)
for every j at once.  The sampled (s, t, k) are shared across j -- common random
numbers, and it lets the expensive evolution be simulated once per draw instead
of once per (draw, j).

`method=:hadamard` runs the paper's circuit with the ancilla; `sampled=true`
then uses single-shot outcomes and `sampled=false` exact expectation values.
`method=:overlap` drops the ancilla entirely (see `sample_overlap`) and is
inherently exact; it gives values identical to `:hadamard, sampled=false`.
"""
function algorithm9(st, m::Int; nsamples::Int=500, seed::Int=1, sampled::Bool=true,
                    method::Symbol=:hadamard)
  method in (:hadamard, :overlap) ||
    throw(ArgumentError("method must be :hadamard or :overlap, got $method"))
  (method === :overlap && sampled) &&
    throw(ArgumentError(":overlap has no ancilla to measure; it is exact -- pass sampled=false"))

  (; n, N, terms, theta, rho, rho_sys, y, T) = st
  J = length(terms)
  rng = MersenneTwister(seed)

  nt1 = norm(theta, 1)
  cq = cumsum(abs.(theta) ./ nt1)          # q(k) = |theta_k| / ||theta||_1

  if method === :hadamard
    hu = ham(N, y[m] .* theta, terms, 1)   # y_m H(theta), system on qubits 2..n+1
    hc = ham_ctrl(N, y[m] .* theta, terms)
    cHj = [cpauli(N, terms[j]) for j in 1:J]
  else
    hu = ham(n, y[m] .* theta, terms, 0)   # y_m H(theta) on the system alone
    ops = [pauli(n, terms[j], 0) for j in 1:J]
  end

  acc = zeros(J)
  for _ in 1:nsamples
    # Step 2: s ~ U[0,1], k ~ q, t ~ gamma.
    s, t = rand(rng), sample_gam(rng)
    k = searchsortedfirst(cq, rand(rng))
    w = nt1 / (2 * T) * s * sign(theta[k])          # Eq. (C42) prefactor

    # Steps 3-4.
    if method === :overlap
      acc .+= w .* sample_overlap(n, rho_sys[m], hu, t, s, T, ops, k)
    else
      mid = hadamard_prefix(N, rho[m], hu, hc, t, s, T)
      for j in 1:J
        r = hadamard_finish(N, mid, cHj[j])
        acc[j] += w * (sampled ? measure_zx!(r, N, terms[k]) : expect_zx(r, N, terms[k]))
      end
    end
  end

  # Step 5.
  return acc ./ nsamples
end

"First term of Eq. (C27): -y_m/2 Tr[H_j rho_m], for every j."
first_terms(st, m::Int) =
  [-st.y[m] / 2 * real(expect(pauli(st.N, t, 1), st.rho[m])) for t in st.terms]

"""
    logloss_gradient(st; nsamples, seed, sampled)

Gradient of L^log(theta) = (1/M) sum_m T Tr[ln(I + e^{-y_m H(theta)/T}) rho_m],
assembled from Eq. (C27): first term + Algorithm 9, averaged over the training
set.  This is the objective's true gradient, so it can drive an optimiser.
"""
function logloss_gradient(st; nsamples::Int=500, seed::Int=1, sampled::Bool=true,
                          method::Symbol=:hadamard)
  M = length(st.rho)
  g = sum(first_terms(st, m) .+
          algorithm9(st, m; nsamples, seed=seed + 1000m, sampled, method) for m in 1:M)
  return g ./ M
end

# ------------------------- Algorithm 10: the loss VALUE, Eq. (C44) ----------

"""
Algorithm 10: estimate Theta_m of Eq. (C79).  Same circuit as Algorithm 9, but
j is uniform on [J], the Hamiltonian uses theta^(j)(lambda), and k ~ q_{j,lambda}.
"""
function algorithm10(st, m::Int, rng, nsamples::Int)
  (; N, terms, theta, rho, y, T) = st
  J = length(terms)
  vals = map(1:nsamples) do _
    j = rand(rng, 1:J)
    s, lam, t = rand(rng), rand(rng), sample_gam(rng)
    thj = vcat(zeros(j - 1), [lam * theta[j]], theta[(j + 1):end])
    n1 = norm(thj, 1)                                 # ||theta^(j)(lambda)||_1
    iszero(n1) && return 0.0
    k = searchsortedfirst(cumsum(abs.(thj) ./ n1), rand(rng))

    mid = hadamard_prefix(N, rho[m], ham(N, y[m] .* thj, terms, 1),
                          ham_ctrl(N, y[m] .* thj, terms), t, s, T)
    r = hadamard_finish(N, mid, cpauli(N, terms[j]))
    J * n1 / (2 * T) * s * sign(theta[k]) * measure_zx!(r, N, terms[k])   # (C83)
  end
  return mean(vals)
end

"L^log(theta) via Eq. (C44): -y_m/2 Tr[H(theta) rho_m] + Theta_m, averaged over m."
function logloss(st; nsamples::Int=4000, seed::Int=7)
  rng = MersenneTwister(seed)
  return mean(-st.y[m] / 2 * real(expect(ham(st.N, st.theta, st.terms, 1), st.rho[m])) +
              algorithm10(st, m, rng, nsamples) for m in eachindex(st.rho))
end

# ------------------------------------------------- exact classical checks ---

"f(A) for Hermitian A by eigendecomposition."
matfun(f, A) = (F = eigen(Hermitian(A)); F.vectors * Diagonal(f.(F.values)) * F.vectors')

"Exact L^log(theta), same convention as `logloss`."
function exact_logloss(st, theta)
  A = Matrix(mat(ham(st.n, theta, st.terms, 0)))
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
  terms = model_terms(n)
  theta = randn(MersenneTwister(seed), length(terms))
  vecs = [training_state(n, h) for h in (0.5, 1.0, 1.5)]
  return (; n, N=n + 1, T, terms, theta, vecs,
          rho=[with_ancilla(v) for v in vecs],           # (n+1) qubits, for :hadamard
          rho_sys=[ArrayReg(ComplexF64.(v)) for v in vecs],  # n qubits, for :overlap
          y=[1.0, -1.0, 1.0])                       # y_m in {-1,+1}
end

"Rebuild `st` with new parameters (rho_m, terms, labels unchanged)."
with_theta(st, theta) = merge(st, (; theta))

# --------------------------------------------------------------------- demo --

if get(ENV, "ALG9_DEMO", "1") == "1"
  st = setup(; n=4)
  g = logloss_gradient(st; nsamples=400, sampled=false)
  ge = exact_logloss_gradient(st, st.theta)
  println("Alg9Yao demo")
  println("  |g_alg9 - g_exact| / |g_exact| = ", round(norm(g - ge) / norm(ge), digits=4))
  println("  cos(g_alg9, g_exact)           = ", round(dot(g, ge) / (norm(g) * norm(ge)), digits=4))
  println("  logloss: Eq.(C44) ", round(logloss(st; nsamples=2000), digits=4),
          "   exact ", round(exact_logloss(st, st.theta), digits=4))
end

end  # module Alg9Yao
