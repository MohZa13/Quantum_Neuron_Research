# data.jl — load the XX vs XXX thermal states and split them into train / test.
#
# The dataset (data/xx_xxx_thermal_states/xx_xxx_n10.h5) holds 840 thermal
# states of random 10-qubit spin chains:
#
#   XX  chain:  H = sum_i J_i (X_i X_{i+1} + Y_i Y_{i+1})                label y = -1
#   XXX chain:  H = sum_i J_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})  label y = +1
#
# 84 chains (42 of each kind), each at 10 temperatures kT.  The file also ships
# each state as an MPO, but we rebuild every state exactly from its couplings
# J instead: rho = exp(-H/kT) / Z.  We diagonalise each chain once,
# H = W diag(eps) W', and then every temperature is just a different set of
# Boltzmann weights on the same eigenvectors.

using HDF5, LinearAlgebra, Random

include(joinpath(@__DIR__, "neuron.jl"))   # pauli(), used to build the chain Hamiltonians

struct Chain
  W::Matrix{Float64}     # eigenvectors of the chain Hamiltonian (columns)
  eps::Vector{Float64}   # its eigenvalues
end

"Boltzmann weights e^{-beta eps} / Z: the state's populations in the chain's eigenbasis."
function boltzmann(ch::Chain, beta)
  w = exp.(-beta .* (ch.eps .- minimum(ch.eps)))
  return w ./ sum(w)
end

struct Dataset
  n::Int
  chains::Vector{Chain}
  chain::Vector{Int}      # which chain each state comes from
  group::Vector{String}   # "XX:3" = model XX, draw 3 (one per chain)
  model::Vector{String}
  kT::Vector{Float64}
  beta::Vector{Float64}
  y::Vector{Float64}      # -1 = XX, +1 = XXX
end

function chain_hamiltonian(n, J, xxx::Bool)
  H = zeros(2^n, 2^n)
  for i in 1:(n - 1)
    H .+= J[i] .* (pauli(n, [(i, 'X'), (i + 1, 'X')]) .+ pauli(n, [(i, 'Y'), (i + 1, 'Y')]))
    xxx && (H .+= J[i] .* pauli(n, [(i, 'Z'), (i + 1, 'Z')]))
  end
  return H
end


"Read the dataset and diagonalise every chain (~1 min)."
function load_dataset(path)
  rows = []
  n = 0
  h5open(path, "r") do f
    n = Int(read(attributes(f["meta"])["n"]))
    for name in sort(keys(f["samples"]))
      g = f["samples"][name]
      a = attributes(g)
      push!(rows, (model=String(read(a["model"])), draw=Int(read(a["draw"])),
                   kT=Float64(read(a["kT"])), beta=Float64(read(a["beta"])),
                   J=Vector{Float64}(read(g["J"])), energy=Float64(read(a["energy_ed"]))))
    end
  end

  group = ["$(r.model):$(r.draw)" for r in rows]
  groups = unique(group)
  chains = map(groups) do g
    r = rows[findfirst(==(g), group)]
    F = eigen(Symmetric(chain_hamiltonian(n, r.J, r.model == "XXX")))
    Chain(F.vectors, F.values)
  end
  chain = [findfirst(==(g), groups) for g in group]

  # sanity check: our rebuilt states have the energies the dataset generator recorded
  dE = maximum(abs(dot(boltzmann(chains[chain[m]], r.beta), chains[chain[m]].eps) - r.energy)
               for (m, r) in enumerate(rows))
  dE < 1e-8 || error("rebuilt states disagree with the file (max |dE| = $dE)")

  model = [r.model for r in rows]
  return Dataset(n, chains, chain, group, model, [r.kT for r in rows], [r.beta for r in rows],
                 [m == "XXX" ? 1.0 : -1.0 for m in model])
end

"""
Random split over all 840 states, stratified by class: `test_per_class` XX states
and `test_per_class` XXX states go to the test set, the rest to training.
Chains and temperatures are not kept together.
"""
function split_by_state(ds::Dataset; test_per_class=80, seed=20260923)
  rng = MersenneTwister(seed)
  test = falses(length(ds.y))
  for cls in ("XX", "XXX")
    test[shuffle(rng, findall(ds.model .== cls))[1:test_per_class]] .= true
  end
  return findall(.!test), findall(test)
end
