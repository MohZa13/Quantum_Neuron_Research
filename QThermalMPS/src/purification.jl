# The infinite-temperature purification, and lifting physical operators onto
# the purification chain.
#
# THE BETA = 0 STATE.  Within the CASCI sector rho(0) is the *sector*
# identity, P/dim, not the identity on the whole Fock space, so its
# purification is the sector-maximally-entangled state
#
#     |Psi(0)> = (1/sqrt(dim)) sum_{n in sector} |n>_phys |n>_anc.
#
# The textbook construction -- a product of local Bell pairs, bond dimension
# 1 -- purifies the GRAND-CANONICAL identity instead, and is a superposition
# of every particle number.  Projecting it back onto the sector is the whole
# difficulty, and it is worth the trouble twice over: the sector is the
# physics the Python pipeline defines, and pinning the flux is what keeps the
# U(1)xU(1) block sparsity that makes the evolution cheap.
#
# What makes it cheap to write down is that across any cut the Schmidt
# vectors are labelled by the running (N_left, Sz_left) alone -- one vector
# per label -- so the exact beta = 0 state has bond dimension O(ncas^2), not
# O(dim).  Rather than emit those tensors by hand (and hand-derive ITensor's
# QN direction conventions with them), we reach the same state by applying a
# pair-raising operator, which routes every convention through ITensor:
#
#     |Psi(0)>  ∝  (sum_{w in alpha} R_pw R_aw)^nalpha
#                  (sum_{w in beta}  R_pw R_aw)^nbeta  |vac>.
#
# `R` is the PARITY-FREE "Raise" of `sites.jl`, not "Cdag".  With bosonic
# ancillas the honest pair creator c+_{p,w} b+_{a,w} is odd, distinct wires
# anticommute, and the square of the sum vanishes identically -- the
# construction would silently return zero.  "Raise" operators commute, so the
# power expands to the flat sum over sector configurations with all
# coefficients +1.
#
# Signs would not have mattered anyway: for ANY |Psi(0)> = sum_n lam_n
# |n>|n> with |lam_n| = 1/sqrt(dim),
#
#     Tr_anc e^{-bH/2}|Psi(0)><Psi(0)|e^{-bH/2} = e^{-bH/2} P e^{-bH/2},
#
# because the ancilla overlap <n'|n> kills every cross term and lam_n^2 = 1/dim
# on the survivors.  The flat choice is simply the one with no bookkeeping.

using ITensors, ITensorMPS

"""
    pair_raise_opsum(L, spin) -> OpSum

`sum_w R_{phys(w)} R_{anc(w)}` over the wires of one spin.  See the file
header for why these are "Raise" and not "Cdag".
"""
function pair_raise_opsum(L::PurificationLayout, spin::Int)
    os = OpSum()
    for w in 0:(L.nwires - 1)
        L.spin_of_wire[w + 1] == spin || continue
        os += 1.0, "Raise", L.physpos[w + 1], "Raise", L.ancpos[w + 1]
    end
    return os
end

"""
    infinite_temperature_mps(L, sites; cutoff, maxdim) -> MPS

The beta = 0 purification of the CASCI sector: `Tr_anc |Psi><Psi| = P/dim`.

Exact up to `cutoff`; the intermediate bond dimensions are bounded by the
number of reachable `(N_left, Sz_left)` labels, so the defaults never bind.
"""
function infinite_temperature_mps(
        L::PurificationLayout, sites::Vector{<:Index};
        cutoff::Float64 = 1.0e-16, maxdim::Int = 10_000
    )
    psi = MPS(sites, fill("Emp", L.nsites))
    for (spin, n) in ((0, L.nalpha), (1, L.nbeta))
        n == 0 && continue
        K = MPO(pair_raise_opsum(L, spin), sites)
        for _ in 1:n
            psi = apply(K, psi; cutoff = cutoff, maxdim = maxdim)
            # The raw power carries a factorial; renormalising every step
            # keeps the tensors in a sane range and costs nothing.
            normalize!(psi)
        end
    end
    normalize!(psi)
    return psi
end

"""
    sector_dimension(L) -> Int

`binomial(ncas, nalpha) * binomial(ncas, nbeta)` -- the CI sector dimension,
which is also `1/Tr[rho(0)^2]` and the norm^2 of the unnormalised beta = 0
state.  Needed to turn the evolved norm into a partition function.
"""
sector_dimension(L::PurificationLayout) =
    binomial(L.ncas, L.nalpha) * binomial(L.ncas, L.nbeta)

"""
    inflate_mpo(Hphys, L, sites) -> MPO

Lift an MPO defined on the bare `2*ncas`-wire physical chain onto the
`4*ncas`-site purification chain by splicing an identity tensor in at each
ancilla position.

Two things are bought at once.  The MPO compiler only ever sees the physical
chain, which is half as long; and `H (x) I_anc` becomes structural rather
than conventional -- there is no path by which a Jordan-Wigner string could
acquire a factor on an ancilla, because the ancilla tensors are identities
that were never part of the compilation.  `purification_mpo` cross-checks
this against a direct build on the full chain.

`Hphys` must have been built on the same `Index` objects that `sites` carries
at the physical positions.
"""
function inflate_mpo(Hphys::MPO, L::PurificationLayout, sites::Vector{<:Index})
    Q = L.nwires
    length(Hphys) == Q ||
        throw(DimensionMismatch("Hphys has $(length(Hphys)) sites, expected $Q"))

    Hc = copy(Hphys)
    H = MPO(L.nsites)
    for w in 1:Q
        sa = sites[L.ancpos[w]]
        if w < Q
            # Splice the ancilla into the existing link: physical site w keeps
            # a fresh copy `m`, the ancilla carries `delta(dag(m), l)`, and
            # `l` still joins it to physical site w+1.
            l = commonind(Hc[w], Hc[w + 1])
            m = sim(l)
            H[L.physpos[w]] = replaceind(Hc[w], l => m)
            H[L.ancpos[w]] = ITensors.denseblocks(delta(dag(m), l)) * op("Id", sa)
        else
            # Last physical site has no right link; manufacture a trivial one
            # so every consecutive pair of MPO tensors shares an index.
            m = Index(QN() => 1; tags = "Link,anc=$w")
            H[L.physpos[w]] = Hc[w] * onehot(m => 1)
            H[L.ancpos[w]] = onehot(dag(m) => 1) * op("Id", sa)
        end
    end
    return H
end
