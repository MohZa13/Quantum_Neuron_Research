# data.jl — load the region states, put their indices in the right order, check them, and split train/test.

using HDF5, LinearAlgebra, Printf, Random

include(joinpath(@__DIR__, "neuron.jl"))   # pauli(), used by the checks below

# ---- conversion 1: axis order ----
# Python wrote rho[sample, ket, bra]. Julia reads the axes backwards, as r[bra, ket, sample].
# Swapping the first two axes gives ρ[ket, bra, sample]: row = ket, the usual way to write a density matrix.
read_rho_stack(dset) = permutedims(read(dset), (2, 1, 3))

# ---- conversion 2: bit order ----
# File order ("little-endian"):  index = b_0 + 2 b_1 + 4 b_2      (b_q = 0 for spin up on region qubit q)
# kron order ("big-endian"):     index = 4 b_0 + 2 b_1 + b_2
# Going from one to the other reverses the k bits of the index (0-based).
reverse_bits(i, k) = sum(((i >> q) & 1) << (k - 1 - q) for q in 0:(k - 1))

"Re-index a 2^k × 2^k matrix from file (little-endian) order to kron (big-endian) order."
function to_kron_order(ρ, k)
  p = [reverse_bits(i, k) for i in 0:(2^k - 1)] .+ 1   # new index -> old index (+1: Julia counts from 1)
  return ρ[p, p]
end

struct Regions
  sites::Vector{Int}                  # chain sites of the region, e.g. [24, 25, 26]
  rho::Vector{Matrix{ComplexF64}}     # the region states, in kron order
  s::Vector{Int}                      # the label: +1 or −1 (direction of the turn on qubit 25)
  t::Vector{Float64}                  # evolution time
  cell::Vector{Int}                   # which time cell; the two labels of one time share it
  trace_distance::Vector{Float64}     # how different the two labels' states are at this time (0 to 1)
end

"Load a region file written by make_regions.py, keeping the times in [tmin, tmax]."
function load_regions(path; tmin=-Inf, tmax=Inf)
  h5open(path, "r") do f
    k = Int(read(attributes(f)["k"]))
    first_site = read(f["first_site"])
    all(==(first_site[1]), first_site) || error("this loader expects one region (a single value of d)")
    t = read(f["t"])
    keep = findall(x -> tmin - 1e-6 <= x <= tmax + 1e-6, t)    # times are compared with a small tolerance
    rho = read_rho_stack(f["rho"])                             # conversion 1 (axis order)
    return Regions(collect(Int(first_site[1]) .+ (0:(k - 1))),
                   [to_kron_order(rho[:, :, m], k) for m in keep],   # conversion 2 (bit order)
                   Int.(read(f["s"])[keep]), t[keep], Int.(read(f["pair_id"])[keep]),
                   read(f["trace_distance"])[keep])
  end
end

"Path of the full-chain state file for one (Hamiltonian, label, time), e.g. XXX/XXX_s+1_t06.400.h5."
mps_file(dir, ham, s, t) = joinpath(dir, ham, @sprintf("%s_s%+d_t%06.3f.h5", ham, s, t))

"Run checks (a)-(d) on the loaded states.  `mps_dir` is the dataset folder holding XX/ and XXX/."
function check_regions(R::Regions, mps_dir, ham)
  k = length(R.sites)
  report(name, dev) = (@printf("  %-58s largest deviation %.1e\n", name, dev);
                       dev < 1e-8 || error("check failed: $name"))

  # (a) proper density matrices: Hermitian, trace 1, no negative eigenvalues
  report("(a) ρ is Hermitian", maximum(norm(ρ - ρ') for ρ in R.rho))
  report("(a) trace of ρ is 1", maximum(abs(tr(ρ) - 1) for ρ in R.rho))
  report("(a) no negative eigenvalues", max(0.0, -minimum(minimum(eigvals(Hermitian(ρ))) for ρ in R.rho)))

  # (b) bit order: ⟨Z_q⟩ from our ρ and our kron-order Z  vs  ⟨Z⟩ the simulation stored for that chain site
  Zs = [pauli(k, [(q, 'Z')]) for q in 1:k]
  dev_b, gap = 0.0, 0.0
  for m in eachindex(R.rho)
    sz = h5open(f -> read(f["checks/sz"]), mps_file(mps_dir, ham, R.s[m], R.t[m]))   # ⟨Z⟩ on sites 1..50
    ours = [real(tr(Z * R.rho[m])) for Z in Zs]
    dev_b = max(dev_b, maximum(abs.(ours .- sz[R.sites])))
    gap = max(gap, abs(sz[R.sites[1]] - sz[R.sites[end]]))
  end
  report("(b) ⟨Z⟩ per qubit matches the stored chain values", dev_b)
  @printf("      (for scale: ⟨Z⟩ on site %d and site %d differ by up to %.2f, so a reversed order would fail)\n",
          R.sites[1], R.sites[end], gap)

  # (c) axis order: compare with the 3-qubit ρ on sites 24-26 that the simulation stored (checks/rho_k3_d0).
  #     That copy is also little-endian and also arrives transposed in Julia, so it gets the same two conversions.
  if R.sites == [24, 25, 26]
    dev_c = 0.0
    for m in eachindex(R.rho)
      stored = h5open(f -> read(f["checks/rho_k3_d0"]), mps_file(mps_dir, ham, R.s[m], R.t[m]))
      dev_c = max(dev_c, maximum(abs.(to_kron_order(transpose(stored), 3) .- R.rho[m])))
    end
    report("(c) ρ matches the simulation's own copy (sites 24-26)", dev_c)
  end

  # (d) spin flip: within each time cell, ρ(s = −1) = X⊗X⊗X ρ(s = +1) X⊗X⊗X
  F = pauli(k, [(q, 'X') for q in 1:k])
  dev_d = 0.0
  for c in unique(R.cell)
    p = findfirst(m -> R.cell[m] == c && R.s[m] == 1, eachindex(R.s))
    n = findfirst(m -> R.cell[m] == c && R.s[m] == -1, eachindex(R.s))
    dev_d = max(dev_d, maximum(abs.(F * R.rho[p] * F .- R.rho[n])))
  end
  report("(d) the two labels are each other's spin flip", dev_d)
end

"Hold out whole time cells (both labels together). Returns the indices of the training and test samples."
function split_by_cell(R::Regions; test_fraction=0.2, seed=20261007)
  cells = sort(unique(R.cell))
  test_cells = Set(shuffle(MersenneTwister(seed), cells)[1:round(Int, test_fraction * length(cells))])
  is_test = [c in test_cells for c in R.cell]
  return findall(.!is_test), findall(is_test)
end
