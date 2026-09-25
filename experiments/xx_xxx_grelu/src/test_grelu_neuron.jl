# test_grelu_neuron.jl
#
# Checks for grelu_neuron.jl + xx_xxx_data.jl on the n = 4 fixture
# (data/xx_xxx_thermal_states/fixture_n4.h5), in seconds:
#
#   1. the reader: ED energy / log Z vs the file (inside `load`), and the
#      contracted MPOs vs ED against the file's own per-level trace distance
#   2. the Pauli signed-permutation algebra vs explicit kron matrices
#   3. exact (Daleckii-Krein) gradients vs central finite differences
#   4. Theorem 17: the Monte-Carlo estimators (Alg 9 and Alg 8, GReLU) are
#      unbiased -- their batch means match the exact gradient within error bars
#   5. the neuron-value estimator vs the exact output
#   6. Gaussian Algorithm 5 (one firing = ReLU(E + T Z)): its mean is the exact
#      output, and fire(H) - fire(-H) recovers the Alg 9 score Tr[H rho]
#
#   julia experiments/xx_xxx_grelu/src/test_grelu_neuron.jl

include(joinpath(@__DIR__, "grelu_neuron.jl"))
include(joinpath(@__DIR__, "xx_xxx_data.jl"))

using .GReLUNeuron, .XXXData
using HDF5, LinearAlgebra, Printf, Random, Statistics
const GN = GReLUNeuron

const FIXTURE = joinpath(@__DIR__, "..", "..", "..", "data", "xx_xxx_thermal_states", "fixture_n4.h5")
nfail = 0
check(ok, msg) = (ok || (global nfail += 1); println(ok ? "  pass  " : "  FAIL  ", msg))

println("\n1. reader")
ds = XXXData.load(FIXTURE)
for level in ("chi4", "chi8")
  res = XXXData.check_mpo(ds, 1:length(ds); level, verbose=false)
  stored = h5open(FIXTURE) do f
    [read(attributes(f["samples"][r.name]["mpo"][level])["trace_distance"]) for r in res]
  end
  d = maximum(abs.([r.td for r in res] .- stored))
  check(d < 1e-6, @sprintf("%s: our MPO-vs-ED trace distance matches the file's (max diff %.1e, td up to %.1e)",
                           level, d, maximum(stored)))
end
split, fold = XXXData.make_split(ds; test_per_class=1, nfolds=3)
check(all(length(unique(split[ds.group .== g])) == 1 for g in unique(ds.group)),
      "split keeps every chain (all its kT rungs) on one side")

println("\n2. Pauli algebra")
const P2 = Dict('I' => [1.0 0; 0 1], 'X' => [0.0 1; 1 0], 'Y' => ComplexF64[0 -im; im 0],
                'Z' => [1.0 0; 0 -1])
function dense_pauli(n, term)
  ops = fill('I', n)
  for (s, o) in GN.term_ops(term); ops[s] = o; end
  return reduce(kron, [P2[o] for o in ops])
end
B = ds.B
terms = model_terms(4; bias=true)
paulis = [Pauli(B, t) for t in terms]
err = maximum(maximum(abs, GN.to_full(B, GN.hamiltonian(B, [P], [1.0])) - dense_pauli(4, P.term))
              for P in paulis)
check(err == 0, "signed permutations reproduce kron(sigma_1..sigma_n) for all $(length(terms)) terms")
A = [randn(d, d) for d in GN.blocksizes(B)]
err = maximum(abs(GN.ptrace(P, A) - tr(dense_pauli(4, P.term) * GN.to_full(B, A))) for P in paulis)
check(err < 1e-12, "ptrace = Tr[P A]")

println("\n3. exact gradients vs finite differences")
T = 0.8
prob = GN.Problem(B, paulis, T, ds.fams, ds.fam, ds.beta, ds.y)
theta = 0.6 .* randn(MersenneTwister(3), length(paulis))
yt = (ds.y .+ 1) ./ 2
fd(f, th; h=1e-5) = [(f(setindex!(copy(th), th[j] + h, j)) - f(setindex!(copy(th), th[j] - h, j))) / 2h
                     for j in eachindex(th)]
L9, g9 = margin_loss_grad(prob, theta)
g9fd = fd(th -> margin_loss_grad(prob, th)[1], theta)
check(norm(g9 - g9fd) / norm(g9fd) < 1e-7, @sprintf("margin (Alg 9) loss: rel err %.1e", norm(g9 - g9fd) / norm(g9fd)))
L8, g8, _ = square_loss_grad(prob, theta, yt)
g8fd = fd(th -> square_loss_grad(prob, th, yt)[1], theta)
check(norm(g8 - g8fd) / norm(g8fd) < 1e-7, @sprintf("square (Alg 8) loss: rel err %.1e", norm(g8 - g8fd) / norm(g8fd)))
# the decision identity L_m(+1) - L_m(-1) = -Tr[H rho_m]
eig = GN.diagonalise(GN.hamiltonian(B, paulis, theta))
Hful = GN.to_full(B, GN.hamiltonian(B, paulis, theta))
Fh = eigen(Symmetric(Hful))
gfun(f) = Fh.vectors * Diagonal(f.(Fh.values)) * Fh.vectors'
r1 = GN.to_full(B, GN.rho(B, ds.fams[1], ds.beta[1]))
lhs = tr(gfun(x -> grelu(-x, T)) * r1) - tr(gfun(x -> grelu(x, T)) * r1)
check(abs(lhs + tr(Hful * r1)) < 1e-12, "GReLU(-H) - GReLU(H) = -H  (decision rule sign Tr[H rho])")
check(maximum(abs, GN.neuron_outputs(prob, theta) .-
              [tr(gfun(x -> grelu(x, T)) * GN.to_full(B, GN.rho(B, ds.fams[ds.fam[m]], ds.beta[m])))
               for m in 1:length(ds)]) < 1e-12, "neuron_outputs = Tr[GReLU(H) rho_m] (dense reference)")

println("\n4. Monte-Carlo estimators are unbiased (batch means +- 4 SE)")
function mc_check(name, est, gex; nb=40)
  G = reduce(hcat, [est(b) for b in 1:nb])
  mu, se = vec(mean(G; dims=2)), vec(std(G; dims=2)) ./ sqrt(nb)
  z = maximum(abs.(mu .- gex) ./ se)
  check(z < 4.5, @sprintf("%-38s max |z| = %.2f   rel err %.3f", name, z, norm(mu - gex) / norm(gex)))
end
mc_check("Alg 9, exact circuit expectations",
         b -> alg9_grelu_gradient(prob, theta; nsamples=400, rng=MersenneTwister(b), sampled=false), g9)
mc_check("Alg 9, single shots, class aggregate",
         b -> alg9_grelu_gradient(prob, theta; nsamples=400, rng=MersenneTwister(b)), g9)
mc_check("Alg 9, single shots, per-state rho_m",
         b -> alg9_grelu_gradient(prob, theta; nsamples=400, rng=MersenneTwister(b), aggregate=false), g9)
mc_check("Alg 8, exact circuit expectations",
         b -> alg8_grelu_gradient(prob, theta, yt; nsamples=400, rng=MersenneTwister(b), sampled=false), g8)
mc_check("Alg 8, single shots",
         b -> alg8_grelu_gradient(prob, theta, yt; nsamples=400, rng=MersenneTwister(b)), g8)
mc_check("Alg 8, single shots, literal value circuit",
         b -> alg8_grelu_gradient(prob, theta, yt; nsamples=400, rng=MersenneTwister(b), literal=true), g8)

println("\n5. neuron value estimator")
v = GN.neuron_outputs(prob, theta)
zs = [begin
        mu, se = value_estimate(prob, theta, m; nshots=4000, rng=MersenneTwister(m))
        (mu - v[m]) / se
      end for m in 1:6]
check(maximum(abs, zs) < 4.5, @sprintf("value_estimate vs exact on 6 states: max |z| = %.2f", maximum(abs, zs)))

println("\n6. Gaussian Algorithm 5 (the neuron firing)")
eig5 = GN.diagonalise(GN.hamiltonian(B, paulis, theta))
zf, zd = Float64[], Float64[]
for m in 1:6
  fs = FiringSampler(prob, eig5, m)
  rng = MersenneTwister(50 + m)
  a = [fire(fs, rng) for _ in 1:40_000]
  b = [fire(fs, rng; sign=-1) for _ in 1:40_000]
  push!(zf, (mean(a) - v[m]) / (std(a) / sqrt(length(a))))
  push!(zd, (mean(a) - mean(b) - prob.C[m, :] ⋅ theta) / sqrt(var(a) / length(a) + var(b) / length(b)))
end
check(maximum(abs, zf) < 4.5, @sprintf("mean firing = Tr[GReLU(H) rho] on 6 states: max |z| = %.2f", maximum(abs, zf)))
check(maximum(abs, zd) < 4.5, @sprintf("fire(H) - fire(-H) = Tr[H rho] (Alg 9 decision): max |z| = %.2f", maximum(abs, zd)))

println(nfail == 0 ? "\nall checks passed" : "\n$nfail check(s) FAILED")
nfail == 0 || exit(1)
