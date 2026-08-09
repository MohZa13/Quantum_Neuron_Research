# Shared test fixtures and the exact references everything is checked against.
#
# Two independent references are used, deliberately:
#
#   1. `exact_boltzmann` -- Boltzmann sums over the `evals` dataset that the
#      Python pipeline wrote.  Cross-language, and the only check that touches
#      real molecular data.
#   2. `dense_thermal` -- `exp(-beta H)` from the MPO densified and
#      diagonalised in Julia, with no tensor network in the answer.  Available
#      for any small case, including random Hamiltonians with no physical
#      structure for a method to accidentally exploit.
#
# A bug would have to fool both to survive.

using QThermalMPS
using ITensors, ITensorMPS
using LinearAlgebra, Random

ITensors.disable_warn_order()

const REPO_RESULTS = normpath(joinpath(@__DIR__, "..", "..", "results"))

"Run file from the Python pipeline, or `nothing` when it was never generated."
function optional_run_file(name)
    p = joinpath(REPO_RESULTS, name)
    return isfile(p) ? p : nothing
end

"""
    random_case(ncas; seed) -> (h1, g)

Random one- and two-electron integrals carrying the physical permutation
symmetries of chemist-notation `(pq|rs)`.  Random rather than molecular on
purpose: real integrals are strongly structured, and a representation bug can
hide inside that structure.
"""
function random_case(ncas::Int; seed::Int = 0)
    rng = MersenneTwister(seed)
    h = randn(rng, ncas, ncas)
    h = (h + transpose(h)) / 2
    V0 = randn(rng, ncas, ncas, ncas, ncas)
    g = similar(V0)
    for p in 1:ncas, q in 1:ncas, r in 1:ncas, s in 1:ncas
        g[p, q, r, s] = (
            V0[p, q, r, s] + V0[q, p, r, s] + V0[p, q, s, r] + V0[q, p, s, r] +
                V0[r, s, p, q] + V0[s, r, p, q] + V0[r, s, q, p] + V0[s, r, q, p]
        ) / 8
    end
    return h, g
end

"""
    sector_indices(L) -> Vector{Int}

1-based rows of the `2^(2 ncas)` qubit register that lie in the
`(nalpha, nbeta)` sector, under the register convention wire 0 = most
significant bit.
"""
function sector_indices(L::PurificationLayout)
    Q = L.nwires
    keep = Int[]
    for r in 0:((1 << Q) - 1)
        na = nb = 0
        for w in 0:(Q - 1)
            if ((r >> (Q - 1 - w)) & 1) == 1
                L.spin_of_wire[w + 1] == 0 ? (na += 1) : (nb += 1)
            end
        end
        (na == L.nalpha && nb == L.nbeta) && push!(keep, r + 1)
    end
    return keep
end

"Everything a small case needs, built once."
struct Fixture
    h1::Matrix{Float64}
    g::Array{Float64, 4}
    L::PurificationLayout
    sites::Vector{<:Index}
    psites::Vector{<:Index}
    H::MPO                       # on the purification chain
    psi0::MPS
    dim::Int
    keep::Vector{Int}
    evals::Vector{Float64}       # exact sector spectrum
    evecs::Matrix{Float64}
end

function fixture(ncas, nalpha, nbeta; ordering = :blocked, seed = 0, h1 = nothing, g = nothing)
    if h1 === nothing
        h1, g = random_case(ncas; seed = seed)
    end
    L = PurificationLayout(ncas, nalpha, nbeta; ordering = ordering)
    sites = build_sites(L)
    psites = physical_indices(L, sites)
    H = purification_mpo(h1, g, L, sites)
    psi0 = infinite_temperature_mps(L, sites)
    keep = sector_indices(L)
    Hd = real(dense_mpo(physical_mpo(h1, g, L, psites), psites))
    e = eigen(Hermitian(Hd[keep, keep]))
    return Fixture(
        h1, g, L, sites, psites, H, psi0, sector_dimension(L),
        keep, e.values, e.vectors
    )
end

"Exact `(logZ, E, S, rho)` in the sector, from a dense spectrum."
function dense_thermal(evals::Vector{Float64}, evecs::Matrix{Float64}, beta::Real)
    lw = -beta .* evals
    m = maximum(lw)
    w = exp.(lw .- m)
    Zs = sum(w)
    logZ = m + log(Zs)
    p = w ./ Zs
    E = sum(p .* evals)
    return (logZ = logZ, E = E, S = beta * E + logZ,
            rho = evecs * Diagonal(p) * transpose(evecs))
end

"Exact `(logZ, E, S)` from a stored eigenvalue list."
function exact_boltzmann(evals::AbstractVector{<:Real}, beta::Real)
    lw = -beta .* evals
    m = maximum(lw)
    logZ = m + log(sum(exp.(lw .- m)))
    p = exp.(lw .- logZ)
    E = sum(p .* evals)
    return (logZ = logZ, E = E, S = beta * E + logZ)
end

"""
    dense_hamiltonian_reference(h1, g, L) -> Matrix{Float64}

`H` over the whole `2^(2 ncas)` register, built BY HAND from `(h1, g)` with
explicit Jordan-Wigner parity signs and no tensor network anywhere.

This is the independent implementation.  Comparing eigenvalues only would
check the spectrum, which is basis-free and would pass even if the register
convention were permuted or sign-flipped; comparing the full matrix pins the
basis, and the basis is exactly what `rho` has to agree with Module I about.

Convention: wire `w` is bit `Q-1-w` of the row index (wire 0 most
significant), and the string sign for an operator on wire `w` counts the
occupied wires with index `< w` -- ITensor's convention for sites in chain
order.
"""
function dense_hamiltonian_reference(
        h1::AbstractMatrix, g::AbstractArray{<:Any, 4}, L::PurificationLayout
    )
    ncas, Q = L.ncas, L.nwires
    K = 1 << Q
    H = zeros(Float64, K, K)

    bitof(r, w) = (r >> (Q - 1 - w)) & 1
    jwsign(r, w) = iseven(count_ones(r >> (Q - w))) ? 1 : -1   # wires 0..w-1

    function ann(state, w)                       # (sign, r) -> (sign, r) or nothing
        state === nothing && return nothing
        s, r = state
        bitof(r, w) == 1 || return nothing
        return (s * jwsign(r, w), r ⊻ (1 << (Q - 1 - w)))
    end
    function cre(state, w)
        state === nothing && return nothing
        s, r = state
        bitof(r, w) == 0 || return nothing
        return (s * jwsign(r, w), r | (1 << (Q - 1 - w)))
    end

    wire = [[jw_wire(p, sp, ncas, L.ordering) for p in 0:(ncas - 1)] for sp in 0:1]

    for r in 0:(K - 1)
        for p in 1:ncas, q in 1:ncas
            c = h1[p, q]
            iszero(c) && continue
            for sp in 1:2
                st = cre(ann((1, r), wire[sp][q]), wire[sp][p])
                st === nothing && continue
                H[st[2] + 1, r + 1] += c * st[1]
            end
        end
        for p in 1:ncas, q in 1:ncas, rr in 1:ncas, s in 1:ncas
            c = 0.5 * g[p, q, rr, s]
            iszero(c) && continue
            for s1 in 1:2, s2 in 1:2
                # a+_p a+_r a_s a_q, applied right to left
                st = ann((1, r), wire[s1][q])
                st = ann(st, wire[s2][s])
                st = cre(st, wire[s2][rr])
                st = cre(st, wire[s1][p])
                st === nothing && continue
                H[st[2] + 1, r + 1] += c * st[1]
            end
        end
    end
    return H
end

"""
    partial_trace_register(rho, Q, wires) -> Matrix

Trace a `2^Q` register density matrix down to `wires` (0-based, wire 0 = most
significant bit), by brute force.  Independent of `physical_rdm`, which is the
point.
"""
function partial_trace_register(rho::AbstractMatrix, Q::Int, wires::AbstractVector{Int})
    ws = sort(collect(wires))
    rest = [w for w in 0:(Q - 1) if !(w in ws)]
    k = length(ws)
    out = zeros(eltype(rho), 1 << k, 1 << k)
    # bit position of wire w in the row index
    shift(w) = Q - 1 - w
    for a in 0:((1 << k) - 1), b in 0:((1 << k) - 1)
        acc = zero(eltype(rho))
        for e in 0:((1 << length(rest)) - 1)
            ra = 0
            rb = 0
            for (i, w) in enumerate(ws)
                bit = (a >> (k - i)) & 1
                ra |= bit << shift(w)
                bit2 = (b >> (k - i)) & 1
                rb |= bit2 << shift(w)
            end
            for (i, w) in enumerate(rest)
                bit = (e >> (length(rest) - i)) & 1
                ra |= bit << shift(w)
                rb |= bit << shift(w)
            end
            acc += rho[ra + 1, rb + 1]
        end
        out[a + 1, b + 1] = acc
    end
    return out
end

"Dense Kronecker product of a Pauli string, first character = wire 0 = MSB."
function pauli_dense(pstring::AbstractString)
    P = Dict(
        'I' => ComplexF64[1 0; 0 1], 'X' => ComplexF64[0 1; 1 0],
        'Y' => ComplexF64[0 -im; im 0], 'Z' => ComplexF64[1 0; 0 -1]
    )
    M = ComplexF64[1;;]
    for c in pstring
        M = kron(M, P[c])
    end
    return M
end
