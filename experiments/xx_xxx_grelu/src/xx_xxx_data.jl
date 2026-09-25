# xx_xxx_data.jl
#
# Loader, integrity checks and train/test split for the XX vs XXX thermal-state
# dataset in data/xx_xxx_thermal_states/ (see its HANDOFF.md / CLAUDE.md).
#
#   H_XX  = sum_i J_i (X_i X_{i+1} + Y_i Y_{i+1})              label 0 -> y = -1
#   H_XXX = sum_i J_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}) label 1 -> y = +1
#
# The quantum neuron consumes the density matrix itself, so each state is
# rebuilt EXACTLY from its stored couplings J by diagonalisation (what the
# handoff recommends) rather than from the chi-truncated MPO -- the chi64 MPO
# has eigenvalues down to -1.8e-3, i.e. it is not quite a state.  `check_mpo`
# contracts the stored MPOs and reports their trace distance to the rebuilt
# rho, which both validates this reader's conventions (site order, J order,
# Pauli normalisation) and quantifies what the MPO route would have cost.
#
# A note on axis order: HDF5.jl reads the Julia-written tensors back in their
# original (n, C, 2, 2, C) order.  The reversal warned about in the handoff
# only affects h5py.
#
# Requires grelu_neuron.jl to be included first.

module XXXData

using HDF5, LinearAlgebra, Printf, Random, Statistics
using ..GReLUNeuron: Basis, Family, Pauli, hamiltonian, rho, energy, logZ, to_full

export load, make_split, write_split, read_split, check_mpo

struct Dataset
  path::String
  n::Int
  names::Vector{String}          # sample_00000 ...
  model::Vector{String}          # "XX" | "XXX"
  draw::Vector{Int}
  label::Vector{Int}             # 0 = XX, 1 = XXX
  y::Vector{Float64}             # -1 = XX, +1 = XXX
  kT::Vector{Float64}
  beta::Vector{Float64}
  J::Matrix{Float64}             # (M, n-1)
  energy_ed::Vector{Float64}
  logZ_ed::Vector{Float64}
  td_mps::Vector{Float64}
  group::Vector{String}          # "XX:3" -- split on this, never on draw
  fam::Vector{Int}               # index into fams (one per group)
  fams::Vector{Family}
  B::Basis
end

Base.length(ds::Dataset) = length(ds.y)

_text(v) = v isa AbstractString ? String(v) : String(copy(v))

"The data Hamiltonian of one chain, as parity blocks."
function chain_hamiltonian(B::Basis, J, xxx::Bool)
  n = B.n
  terms = vcat([(:xx, i) for i in 1:(n - 1)], [(:yy, i) for i in 1:(n - 1)],
               xxx ? [(:zz, i) for i in 1:(n - 1)] : Tuple{Symbol,Int}[])
  w = vcat(J, J, xxx ? J : Float64[])
  return hamiltonian(B, [Pauli(B, t) for t in terms], w)
end

"""
    load(path) -> Dataset

Reads the labels/metadata and diagonalises each chain once (one `Family` per
(model, draw) group; its 10 temperatures share the eigenvectors).  Checks
every sample's ED energy and log Z against the file's `energy_ed`/`logZ_ed`.
"""
function load(path::AbstractString; verbose::Bool=true)
  t0 = time()
  cols = Dict{Symbol,Vector}(k => [] for k in (:name, :model, :draw, :label, :kT, :beta,
                                               :J, :E, :lZ, :td))
  n = 0
  h5open(path, "r") do f
    meta = attributes(f["meta"])
    haskey(meta, "complete") && read(meta["complete"]) == true ||
      error("$path: meta/complete not set -- torn file")
    n = Int(read(meta["n"]))
    for name in sort(keys(f["samples"]))
      g = f["samples"][name]
      a = attributes(g)
      push!(cols[:name], name)
      push!(cols[:model], _text(read(a["model"])))
      push!(cols[:draw], Int(read(a["draw"])))
      push!(cols[:label], Int(read(a["label"])))
      push!(cols[:kT], Float64(read(a["kT"])))
      push!(cols[:beta], Float64(read(a["beta"])))
      push!(cols[:J], Vector{Float64}(read(g["J"])))
      push!(cols[:E], Float64(read(a["energy_ed"])))
      push!(cols[:lZ], Float64(read(a["logZ_ed"])))
      push!(cols[:td], Float64(read(a["td_mps"])))
    end
  end

  B = Basis(n)
  model = String.(cols[:model])
  draw = Int.(cols[:draw])
  label = Int.(cols[:label])
  all((model .== "XXX") .== (label .== 1)) || error("label and model disagree")
  group = ["$(m):$(d)" for (m, d) in zip(model, draw)]
  groups = unique(group)
  fam = [findfirst(==(g), groups) for g in group]
  J = permutedims(reduce(hcat, cols[:J]))

  fams = map(groups) do g
    m = findfirst(==(g), group)
    for m2 in findall(==(g), group)            # one chain, one J
      J[m2, :] == J[m, :] || error("group $g has inconsistent couplings")
    end
    Family(B, chain_hamiltonian(B, J[m, :], model[m] == "XXX"))
  end

  ds = Dataset(String(path), n, String.(cols[:name]), model, draw, label,
               2.0 .* label .- 1.0, Float64.(cols[:kT]), Float64.(cols[:beta]), J,
               Float64.(cols[:E]), Float64.(cols[:lZ]), Float64.(cols[:td]),
               group, fam, fams, B)

  # ED agreement with the generator's own dense reference
  dE = maximum(abs(energy(ds.fams[ds.fam[m]], ds.beta[m]) - ds.energy_ed[m]) for m in 1:length(ds))
  dZ = maximum(abs(logZ(ds.fams[ds.fam[m]], ds.beta[m]) - ds.logZ_ed[m]) for m in 1:length(ds))
  (dE < 1e-8 && dZ < 1e-8) ||
    error(@sprintf("ED disagrees with file: max|dE| = %.2e, max|dlogZ| = %.2e", dE, dZ))
  if verbose
    @printf("%s\n  n = %d   %d samples   %d chains (%d XX / %d XXX)   kT = %s\n",
            path, n, length(ds), length(groups), count(startswith("XX:"), groups),
            count(startswith("XXX:"), groups), string(sort(unique(ds.kT); rev=true)))
    @printf("  ED vs file: max|dE| = %.1e   max|dlogZ| = %.1e   (%.1fs)\n", dE, dZ, time() - t0)
  end
  return ds
end

# --------------------------------------------------------------- MPO check ---

"""
Contract one stored MPO level to a dense 2^n x 2^n matrix (site 1 = most
significant bit, matching `Basis`), slicing away the zero padding with `chi`.
"""
function mpo_dense(path, name, level)
  h5open(path, "r") do f
    lg = f["samples"][name]["mpo"][level]
    t = read(lg["tensors"])                    # (n, C, 2, 2, C) in Julia
    chi = Int.(read(lg["chi"]))
    n = size(t, 1)
    L = ones(1, 1, 1)                                         # (R, C, bond)
    for s in 1:n
      A = t[s, 1:chi[s], :, :, 1:chi[s + 1]]                 # (a, r, c, b)
      R, Cc, a = size(L)
      P = reshape(reshape(L, R * Cc, a) * reshape(A, a, :), R, Cc, 2, 2, chi[s + 1])
      L = reshape(permutedims(P, (3, 1, 4, 2, 5)), 2R, 2Cc, chi[s + 1])  # new site = LSB
    end
    return L[:, :, 1]
  end
end

"""
    check_mpo(ds, which; level) -> Vector of (sample, trace distance, min eigenvalue)

Trace distance between the contracted MPO (renormalised to unit trace) and the
ED state for the samples in `which`.  Small values confirm site order, J order
and conventions; they are expected to be ~1e-3 at chi64 (truncation), not 1e-7.
"""
function check_mpo(ds::Dataset, which; level=nothing, verbose=true)
  level === nothing && (level = h5open(ds.path, "r") do f
    "chi$(maximum(read(attributes(f["meta"])["chi_mpo_levels"])))"
  end)
  out = map(which) do m
    Rm = mpo_dense(ds.path, ds.names[m], level)
    Rm = (Rm + Rm') ./ (2 * tr(Rm))
    Re = to_full(ds.B, rho(ds.B, ds.fams[ds.fam[m]], ds.beta[m]))
    ev = eigvals(Symmetric(Rm - Re))
    (; m, name=ds.names[m], group=ds.group[m], kT=ds.kT[m],
       td=sum(abs, ev) / 2, mineig=minimum(eigvals(Symmetric(Rm))))
  end
  if verbose
    @printf("MPO %s vs ED (%d samples): trace distance  median %.2e  max %.2e   min eig(rho_mpo) %.2e\n",
            level, length(out), median(o.td for o in out), maximum(o.td for o in out),
            minimum(o.mineig for o in out))
  end
  return out
end

# ------------------------------------------------------------------- split ---

"""
    make_split(ds; test_per_class, nfolds, seed) -> (split, fold)

Stratified GROUP hold-out: whole chains (model:draw -- all 10 temperatures
together) go to test, `test_per_class` per class.  The remaining chains get a
stratified group k-fold id 1..nfolds for hyperparameter selection (each fold
holds the same number of chains per class, +-1).  Test samples get fold 0.
"""
function make_split(ds::Dataset; test_per_class::Int=8, nfolds::Int=5, seed::Int=20260923)
  rng = MersenneTwister(seed)
  split = fill("train", length(ds))
  fold = zeros(Int, length(ds))
  for cls in ("XX", "XXX")
    gs = shuffle(rng, sort(unique(ds.group[ds.model .== cls]); by=g -> parse(Int, split_group(g)[2])))
    for (i, g) in enumerate(gs)
      ms = findall(==(g), ds.group)
      if i <= test_per_class
        split[ms] .= "test"
      else
        fold[ms] .= mod1(i - test_per_class, nfolds)
      end
    end
  end
  return split, fold
end

split_group(g) = Tuple(Base.split(g, ':'))

function write_split(path, ds::Dataset, split, fold)
  mkpath(dirname(path))
  open(path, "w") do io
    println(io, "index,sample,model,draw,group,label,kT,split,fold")
    for m in 1:length(ds)
      @printf(io, "%d,%s,%s,%d,%s,%d,%.4f,%s,%d\n", m, ds.names[m], ds.model[m],
              ds.draw[m], ds.group[m], ds.label[m], ds.kT[m], split[m], fold[m])
    end
  end
  return path
end

"Read a split written by `write_split` (so every run uses the same one)."
function read_split(path, ds::Dataset)
  lines = readlines(path)[2:end]
  length(lines) == length(ds) || error("split has $(length(lines)) rows, dataset $(length(ds))")
  split, fold = String[], Int[]
  for (m, l) in enumerate(lines)
    c = Base.split(l, ',')
    c[2] == ds.names[m] || error("split row $m is $(c[2]), dataset has $(ds.names[m])")
    push!(split, c[8]); push!(fold, parse(Int, c[9]))
  end
  return split, fold
end

"Human-readable split summary."
function describe_split(ds::Dataset, split, fold)
  for s in ("train", "test")
    ms = findall(==(s), split)
    @printf("  %-5s  %3d samples  %2d chains  (%d XX / %d XXX chains)\n", s, length(ms),
            length(unique(ds.group[ms])), length(unique(ds.group[ms][ds.model[ms] .== "XX"])),
            length(unique(ds.group[ms][ds.model[ms] .== "XXX"])))
  end
  for k in sort(unique(fold[fold .> 0]))
    ms = findall(==(k), fold)
    @printf("  fold %d %3d samples  %2d chains  (%d XX / %d XXX)\n", k, length(ms),
            length(unique(ds.group[ms])), length(unique(ds.group[ms][ds.model[ms] .== "XX"])),
            length(unique(ds.group[ms][ds.model[ms] .== "XXX"])))
  end
end

end  # module XXXData
