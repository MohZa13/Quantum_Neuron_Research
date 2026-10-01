# qumode_itensor.jl
#
# Gaussian Algorithm 5 (the GReLU neuron FIRING) simulated literally in ITensor,
# with the control qumode as an explicit bosonic mode -- a validation of the
# exact sampler `fire` in grelu_neuron.jl, on the n = 4 fixture.
#
# Paper (He, Liu & Wilde, "Fermi-Dirac machines as quantizations of neurons"):
#   Algorithm 5 (Sec. III.A, smooth-ReLU neuron) and its Gaussian version for
#   GReLU (Sec. IV.B): GReLU_T(x) = x Phi(x/T) + T phi(x/T)          Eq. (92)
#   = E[ReLU(x + T Z)]                                               Eq. (93)
#   = T2 (ReLU * phi_T1)(x/T2),  T = T1 T2                           Eq. (94)
#   i.e. Theorem 3 / Algorithm 3 (quantum convolution) with the logistic control
#   state replaced by the Gaussian one.
#
# Procedure simulated here, one run of the neuron on one copy of rho:
#   1. control qumode in the pure Gaussian state whose momentum density is
#      phi_T1(p) = N(0, T1^2); data register (n qubits) in rho;
#   2. apply U = exp(i x ⊗ H(theta) / T2), x = (a + a†)/sqrt(2);
#   3. measure the qumode's momentum p (homodyne);
#   4. output T2 ReLU(p).
#
# ITensor implementation (ITensors.jl v0.9.30, ITensorMPS.jl v0.4.1):
#   * the qumode is one `siteind("Boson"; dim = d)` -- the "Boson" site type, an
#     alias of "Qudit": a d-level truncated Fock space with operators "a",
#     "adag", "N" (ITensorMPS docs, "Included SiteTypes" -> "Boson"/"Qudit",
#     https://docs.itensor.org/ITensorMPS/stable/IncludedSiteTypes.html;
#     source ITensors/src/lib/SiteTypes/src/sitetypes/{boson,qudit}.jl);
#   * the qubits are `siteinds("Qubit", n)`; H(theta) is built with `OpSum` ->
#     `MPO` and contracted to one operator ITensor with `prod`;
#   * x ⊗ H is an outer product of ITensors; U = `exp(im * x⊗H / T2)`, the
#     ITensor matrix exponential over primed/unprimed index pairs
#     (ITensors/src/tensor_operations/matrix_algebra.jl, `exp(A::ITensor, Linds,
#     Rinds; ishermitian)`);
#   * the joint state |c><c| ⊗ rho evolves as U rho U† with
#     `apply(U, rho; apply_dag = true)` (ITensors `product(A, B; apply_dag)`,
#     src/tensor_operations/tensor_algebra.jl);
#   * the qubits are traced out with `delta(s', s)` contractions, leaving the
#     qumode's d x d density matrix;
#   * homodyne statistics: P(p) = <p|rho_q|p>, with <p|n> = (-i)^n psi_n(p) and
#     psi_n the Hermite functions (standard quadrature conventions, e.g.
#     Weedbrook et al., Rev. Mod. Phys. 84, 621 (2012), Sec. II).
#
# The initial Gaussian state's Fock coefficients are computed numerically from
# its momentum wavefunction with the SAME <p|n> used for the measurement, so no
# separate squeezing-sign convention enters.  T1 = 1/sqrt(2) is the vacuum.
#
# Checks (written to ../results/qumode_itensor_validation.csv):
#   * mean output = Tr[GReLU_T(H) rho] (the neuron's exact output);
#   * full output distribution = the exact sampler's: max |CDF difference|;
#   * convergence in the Fock cutoff d, and two different squeezings T1 at
#     fixed T = T1 T2 (only the product T should matter);
#   * the neuron's classification from the simulated outputs.
#
#   julia experiments/xx_xxx_grelu/src/qumode_itensor.jl

isdefined(Main, :CFG) || include(joinpath(@__DIR__, "train_xx_xxx_grelu.jl"))   # GReLUNeuron, XXXData, train, helpers

module QumodeITensor

using ITensors, ITensorMPS, LinearAlgebra, Statistics
using SpecialFunctions: erf

const OPNAME = Dict(:z => "Z", :zz => "Z", :xx => "X", :yy => "Y")

"H(theta) on ITensor qubit sites, as one operator ITensor (indices s', s)."
function hamiltonian(sites, terms, theta)
  os = OpSum()
  for ((kind, i), th) in zip(terms, theta)
    if kind === :id
      os += th, "Id", 1
    elseif kind === :z
      os += th, "Z", i
    else
      os += th, OPNAME[kind], i, OPNAME[kind], i + 1
    end
  end
  return prod(MPO(os, sites))
end

"""
A dense density matrix in the full computational basis (site 1 = most
significant bit) as an ITensor with row indices s' and column indices s.
"""
function rho_itensor(sites, R::AbstractMatrix)
  n = length(sites)
  A = reshape(ComplexF64.(R), ntuple(_ -> 2, 2n))   # column-major: site n varies fastest
  return itensor(A, [sites[k]' for k in n:-1:1]..., [sites[k] for k in n:-1:1]...)
end

"Hermite functions psi_n(p), n = 0..d-1, on the grid (d x length(p))."
function hermite_functions(d, p)
  H = zeros(d, length(p))
  H[1, :] .= pi^(-1 / 4) .* exp.(-p .^ 2 ./ 2)
  d > 1 && (H[2, :] .= sqrt(2) .* p .* H[1, :])
  for n in 2:(d - 1)
    H[n + 1, :] .= sqrt(2 / n) .* p .* H[n, :] .- sqrt((n - 1) / n) .* H[n - 1, :]
  end
  return H
end

"<p|n> = (-i)^n psi_n(p): momentum wavefunctions of the Fock states."
momentum_wavefunctions(d, p) = [(-im)^(n - 1) for n in 1:d] .* hermite_functions(d, p)

"""
Fock coefficients of the pure Gaussian state with momentum density N(0, T1^2):
its momentum wavefunction is (2 pi T1^2)^(-1/4) exp(-p^2 / (4 T1^2)).
"""
function gaussian_fock(d, T1, p, Phi)
  dp = p[2] - p[1]
  f = (2pi * T1^2)^(-1 / 4) .* exp.(-p .^ 2 ./ (4T1^2))
  c = vec(conj(Phi) * f) .* dp                  # c_n = ∫ conj(<p|n>) f(p) dp
  return c ./ norm(c)
end

"""
    homodyne_density(Hm, rho, sites, d, T1, T2; p) -> (P(p), leak)

Run Algorithm 5 once on the ITensor state |c><c| ⊗ rho and return the momentum
density of the qumode on the grid `p`, plus 1 - ∫P (Fock-truncation leakage).
"""
function homodyne_density(Hm, rho, sites, d, T1, T2; p=range(-25, 25; length=6001), cache=Dict())
  key = (objectid(Hm), d, T1, T2)
  U, b, Phi, c = get!(cache, key) do
    b = siteind("Boson"; dim=d)
    x = (op("a", b) + op("adag", b)) / sqrt(2)
    U = exp(im * (x * Hm) / T2)                           # exp(i x ⊗ H / T2)
    Phi = momentum_wavefunctions(d, collect(p))
    c = gaussian_fock(d, T1, collect(p), Phi)
    (U, b, Phi, c)
  end
  rq0 = itensor(c * c', b', b)                            # |c><c| on the qumode
  rj = apply(U, rq0 * rho; apply_dag=true)                # U (|c><c| ⊗ rho) U†
  for s in sites
    rj *= delta(s', s)                                    # trace out the qubits
  end
  M = Matrix(rj, b', b)
  P = real.(vec(sum((transpose(Phi) * M) .* transpose(conj(Phi)); dims=2)))   # <p|rho_q|p>
  return P, 1 - sum(P) * step(p)
end

"Output y = T2 ReLU(p): mean, and CDF on y-grid `ys` (trapezoid + linear interpolation)."
function output_stats(P, p, T2, ys)
  dp = step(p)
  mean_y = sum(T2 .* max.(p, 0) .* P) * dp
  cum = [0.0; cumsum((P[1:end-1] .+ P[2:end]) ./ 2) .* dp]
  function at(q)
    i = clamp(searchsortedlast(p, q), 1, length(p) - 1)
    f = (q - p[i]) / dp
    return (1 - f) * cum[i] + f * cum[i + 1]
  end
  return mean_y, [at(y / T2) for y in ys]
end

siteinds_qubit(n) = siteinds("Qubit", n)
scalar_re(t) = real(scalar(t))

"Exact sampler's output CDF: P(ReLU(E + T Z) <= y) = sum_E w_E Phi((y - E)/T), y >= 0."
exact_cdf(E, w, T, ys) = [sum(w .* (1 .+ erf.((y .- E) ./ (T * sqrt(2)))) ./ 2) for y in ys]

end # module QumodeITensor

# ------------------------------------------------------------------- driver ---

function qumode_validation()
  Q = QumodeITensor
  FIX = joinpath(ROOT, "data", "xx_xxx_thermal_states", "fixture_n4.h5")
  ds = XXXData.load(FIX; verbose=false)
  n = ds.n
  sites = Q.siteinds_qubit(n)
  rows = NamedTuple[]
  curves = NamedTuple[]
  ex_m = findfirst(m -> ds.model[m] == "XXX" && ds.kT[m] == minimum(ds.kT), 1:length(ds))   # example state
  ys = range(0, 30; length=601)
  p = range(-25, 25; length=6001)
  for kind in (:margin, :square)
    hp = chosen(kind)
    paulis = [Pauli(ds.B, t) for t in terms_for(ds, kind)]
    prob = subproblem(ds, 1:length(ds), paulis, hp.T)
    theta = train(kind, prob, nothing; l2=hp.l2, monitor=false, verbose=false).theta
    T = prob.T
    exact_out = GN.neuron_outputs(prob, theta)
    Hfull = GN.to_full(ds.B, GN.hamiltonian(ds.B, paulis, theta))
    F = eigen(Symmetric(Hfull))
    signs = kind === :margin ? (1, -1) : (1,)
    Hm = Dict(sg => Q.hamiltonian(sites, terms_for(ds, kind), sg .* theta) for sg in signs)
    cache = Dict()
    for (d, T1) in ((20, 1 / sqrt(2)), (40, 1 / sqrt(2)), (60, 1 / sqrt(2)), (60, 0.45), (80, 0.45))
      T2 = T / T1
      for m in 1:length(ds)
        R = GN.to_full(ds.B, GN.rho(ds.B, ds.fams[ds.fam[m]], ds.beta[m]))
        rho = Q.rho_itensor(sites, R)
        w = max.(real.(diag(F.vectors' * R * F.vectors)), 0.0)
        means = Dict{Int,Float64}()
        for sg in signs
          P, leak = Q.homodyne_density(Hm[sg], rho, sites, d, T1, T2; p, cache)
          mu, cdf = Q.output_stats(P, p, T2, ys)
          ex = Q.exact_cdf(sg .* F.values, w, T, ys)
          exact_mean = sum(w .* GN.grelu.(sg .* F.values, T))
          means[sg] = mu
          if m == ex_m && sg == 1
            append!(curves, [(; loss_type=kind, d, T1, y_out=yv, cdf_itensor=c1, cdf_exact=c2)
                             for (yv, c1, c2) in zip(ys, cdf, ex)])
          end
          push!(rows, (; loss_type=kind, d, T1, T2, sample=ds.names[m], kT=ds.kT[m], y=Int(ds.y[m]),
                       sign=sg, mean_itensor=mu, mean_exact=exact_mean, cdf_maxdiff=maximum(abs.(cdf .- ex)),
                       leak))
        end
        # classification from the simulated outputs (exact means of the simulated distribution)
        score = kind === :margin ? means[1] - means[-1] : means[1] - 0.5
        rows[end] = merge(rows[end], (; score_itensor=score))
      end
      sel = filter(r -> r.loss_type == kind && r.d == d && r.T1 == T1, rows)
      cls = filter(r -> haskey(r, :score_itensor), sel)
      acc = mean((r.score_itensor >= 0) == (r.y > 0) for r in cls)
      @printf("   %s  d=%2d T1=%.3f T2=%.3f:  max |mean - exact| %.1e   max |ΔCDF| %.1e   max leak %.1e   accuracy %.3f (%d states)\n",
              LOSS_NAME[kind], d, T1, T2, maximum(abs(r.mean_itensor - r.mean_exact) for r in sel),
              maximum(r.cdf_maxdiff for r in sel), maximum(r.leak for r in sel), acc, length(cls))
    end
    # sanity: the ITensor H agrees with the parity-block H, and rho's layout with it
    m = 1
    R = GN.to_full(ds.B, GN.rho(ds.B, ds.fams[ds.fam[m]], ds.beta[m]))
    tH = Q.scalar_re(Hm[1] * Q.rho_itensor(sites, R))
    @printf("   %s  check: Tr[H rho] ITensor %.12f vs exact %.12f\n", LOSS_NAME[kind], tH, prob.C[m, :] ⋅ theta)
  end
  write_rows(joinpath(OUT, "qumode_itensor_validation.csv"),
             [merge((; score_itensor=NaN), r) for r in rows])
  write_rows(joinpath(OUT, "qumode_itensor_example.csv"), curves)
  println("   example state: ", ds.names[ex_m], " (", ds.model[ex_m], ", kT = ", ds.kT[ex_m], ")")
  return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
  qumode_validation()
end
