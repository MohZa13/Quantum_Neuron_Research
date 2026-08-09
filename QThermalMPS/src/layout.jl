# Wire conventions and the purification chain layout.
#
# The wire map is a direct port of `qthermal/encode.py::jw_wire`, so a state
# produced here is comparable term-by-term with the Python encoder's
# `extended_heisenberg_labels` ordering.  Do not "improve" the conventions
# here in isolation -- they are a contract with the Python side.

"""
    jw_wire(p, spin, ncas, ordering) -> Int

0-based Jordan-Wigner wire of spin-orbital `(p, spin)`; `spin` 0 = alpha,
1 = beta.  Mirrors `qthermal/encode.py::jw_wire`.

  * `:blocked`     -> `p + spin*ncas`  (alpha block first, then beta)
  * `:interleaved` -> `2p + spin`      (same-orbital pairs adjacent)
"""
function jw_wire(p::Int, spin::Int, ncas::Int, ordering::Symbol)
    ordering === :blocked && return p + spin * ncas
    ordering === :interleaved && return 2p + spin
    throw(ArgumentError("unknown ordering $ordering (:blocked or :interleaved)"))
end

"""
    PurificationLayout

Chain geometry for a purified active space.

The chain interleaves each physical wire with its own ancilla,
`[phys_0, anc_0, phys_1, anc_1, ...]`, so the beta = 0 purification is a
product of nearest-neighbour maximally entangled pairs (bond dimension 1
between pairs).  Grouping all physicals first would make that state
maximally entangled across the midpoint cut, with bond dimension equal to
the full sector dimension -- the whole point of the interleaving.

Fields:
  * `nwires`      `2*ncas` spin-orbitals
  * `nsites`      `4*ncas` chain sites (physical + ancilla)
  * `physpos[w+1]` 1-based chain position of JW wire `w`
  * `ancpos[w+1]`  1-based chain position of wire `w`'s ancilla
  * `spin_of_wire[w+1]`, `orb_of_wire[w+1]` inverse of `jw_wire`
"""
struct PurificationLayout
    ncas::Int
    nalpha::Int
    nbeta::Int
    ordering::Symbol
    nwires::Int
    nsites::Int
    physpos::Vector{Int}
    ancpos::Vector{Int}
    spin_of_wire::Vector{Int}
    orb_of_wire::Vector{Int}
end

function PurificationLayout(ncas::Int, nalpha::Int, nbeta::Int;
                            ordering::Symbol=:interleaved)
    ncas >= 1 || throw(ArgumentError("ncas must be >= 1"))
    0 <= nalpha <= ncas || throw(ArgumentError("nalpha=$nalpha outside [0,$ncas]"))
    0 <= nbeta <= ncas || throw(ArgumentError("nbeta=$nbeta outside [0,$ncas]"))

    nwires = 2ncas
    spin_of_wire = Vector{Int}(undef, nwires)
    orb_of_wire = Vector{Int}(undef, nwires)
    for p in 0:(ncas - 1), s in 0:1
        w = jw_wire(p, s, ncas, ordering)
        spin_of_wire[w + 1] = s
        orb_of_wire[w + 1] = p
    end

    # chain position 2w+1 is physical wire w, 2w+2 is its ancilla (1-based)
    physpos = [2w + 1 for w in 0:(nwires - 1)]
    ancpos = [2w + 2 for w in 0:(nwires - 1)]

    return PurificationLayout(ncas, nalpha, nbeta, ordering, nwires,
                              2nwires, physpos, ancpos,
                              spin_of_wire, orb_of_wire)
end

"Chain position (1-based) of the physical site carrying spin-orbital (p, spin)."
function physsite(L::PurificationLayout, p::Int, spin::Int)
    return L.physpos[jw_wire(p, spin, L.ncas, L.ordering) + 1]
end

nelecas(L::PurificationLayout) = L.nalpha + L.nbeta

"True if chain position `j` holds a physical (as opposed to ancilla) site."
isphysical(::PurificationLayout, j::Int) = isodd(j)
