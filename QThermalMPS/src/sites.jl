# Site indices and local operators for the purification chain.
#
# Physical sites are ITensor "Fermion" sites (local dim 2), one per JW wire,
# so the physical half of the chain IS the qubit register the Python encoder
# targets -- Pauli strings read off directly, no unfolding.
#
# ANCILLAS ARE BOSONIC.  They carry their own U(1) quantum numbers ("Na",
# "Sza") with NO fermion-parity flag, and `has_fermion_string` is false for
# every operator on them.  The consequence that matters is that a
# Jordan-Wigner string running between two physical wires passes through the
# intervening ancillas as the IDENTITY, so the fermionic algebra realised on
# the physical sites is exactly the *physical-only* JW of `qthermal/encode.py`
# -- and therefore
#
#     rho = Tr_anc |Psi><Psi|            (plain qubit partial trace)
#
# is the Module I encoded state, with no dressing.  That equality is the whole
# reason the purification is usable as a neuron input.
#
# THIS IS NOT A FREE CHOICE.  Make the ancillas fermionic instead and the JW
# strings of H pick up a Z on every intervening ancilla; the qubit partial
# trace then computes the wrong object.  Concretely, for two wires at
# half filling with H = -t(c+_0 c_1 + h.c.), first order in beta:
#
#     fermionic ancillas, naive Tr_anc  ->  <10|rho|01> = 0
#     bosonic   ancillas, naive Tr_anc  ->  <10|rho|01> = beta*t/2   (correct)
#
# `test/test_ancilla_statistics.jl` runs exactly this case.  With fermionic
# ancillas the correct value is still recoverable -- as <Psi|P (x) Z_anc|Psi>,
# strings and all -- but every downstream consumer would have to know the rule.
# Bosonic ancillas push that complexity into one place: the beta = 0 state.
#
# The price is that the pair creator c+_{p,w} b+_{a,w} is ODD, so distinct
# wires anticommute, (sum_w X_w)^2 = 0, and the infinite-temperature state
# CANNOT be built by exponentiating a pair creator.  `purification.jl` builds
# it from parity-free "Raise" operators instead, which is what the two
# definitions below exist for.
#
# Quantum numbers earn their keep: U(1)_Nf x U(1)_Sz is exactly the
# (nelecas, S_z = 0) sector the CASCI pipeline already lives in, and pinning
# the MPS flux is what confines the evolution to it.

using ITensors

# --------------------------------------------------------------- ancillas

# No dynamics ever acts on an ancilla, so the operator set is minimal:
# identity (MPO padding), number (diagnostics), and the raising/lowering pair
# used to build the beta = 0 state.
ITensors.op(::OpName"Id", ::SiteType"Anc", s::Index) =
    ITensors.denseblocks(delta(s', dag(s)))
ITensors.op(::OpName"I", st::SiteType"Anc", s::Index) = ITensors.op(OpName("Id"), st, s)

# "F" is the Jordan-Wigner string operator.  Defining it as the IDENTITY is
# the operational content of "ancillas are bosonic": when ITensor threads a
# fermion string across the chain it multiplies each intervening site by "F",
# and on an ancilla that must do nothing.  Without this definition a directly
# built purification-chain MPO would simply fail to construct.
ITensors.op(::OpName"F", st::SiteType"Anc", s::Index) = ITensors.op(OpName("Id"), st, s)

ITensors.op!(Op::ITensor, ::OpName"N", ::SiteType"Anc", s::Index) = (Op[s' => 2, s => 2] = 1.0)
ITensors.op!(Op::ITensor, ::OpName"Raise", ::SiteType"Anc", s::Index) = (Op[s' => 2, s => 1] = 1.0)
ITensors.op!(Op::ITensor, ::OpName"Lower", ::SiteType"Anc", s::Index) = (Op[s' => 1, s => 2] = 1.0)

ITensors.has_fermion_string(::OpName, ::SiteType"Anc") = false

ITensors.val(::ValName"Emp", ::SiteType"Anc") = 1
ITensors.val(::ValName"Occ", ::SiteType"Anc") = 2
ITensors.val(::ValName"0", st::SiteType"Anc") = ITensors.val(ValName("Emp"), st)
ITensors.val(::ValName"1", st::SiteType"Anc") = ITensors.val(ValName("Occ"), st)

ITensors.state(::StateName"Emp", ::SiteType"Anc") = [1.0, 0.0]
ITensors.state(::StateName"Occ", ::SiteType"Anc") = [0.0, 1.0]
ITensors.state(::StateName"0", st::SiteType"Anc") = ITensors.state(StateName("Emp"), st)
ITensors.state(::StateName"1", st::SiteType"Anc") = ITensors.state(StateName("Occ"), st)

# ------------------------------------------------- parity-free raising ops
#
# DELIBERATE PIRACY: "Raise"/"Lower" are new operator names on ITensor's own
# "Fermion" site type.  They are the matrices of "Cdag"/"C" with the fermion
# string flag switched OFF, and they exist for exactly one purpose -- writing
# down the infinite-temperature purification, which is a statement about
# occupation-number basis states and carries no fermionic phase.  Using them
# anywhere a *physical* fermion operator is meant is a bug: the Hamiltonian
# must always be written with "C"/"Cdag".
#
# If a future ITensors release defines these names, Julia will emit a
# method-overwrite warning rather than silently disagreeing.
ITensors.op!(Op::ITensor, ::OpName"Raise", ::SiteType"Fermion", s::Index) = (Op[s' => 2, s => 1] = 1.0)
ITensors.op!(Op::ITensor, ::OpName"Lower", ::SiteType"Fermion", s::Index) = (Op[s' => 1, s => 2] = 1.0)
ITensors.has_fermion_string(::OpName"Raise", ::SiteType"Fermion") = false
ITensors.has_fermion_string(::OpName"Lower", ::SiteType"Fermion") = false

# ----------------------------------------------------------------- indices

"Spin quantum number of a wire: +1 for alpha (spin 0), -1 for beta (spin 1)."
szval(spin::Int) = spin == 0 ? 1 : -1

"""
    physical_index(j, spin; conserve_sz) -> Index

Fermionic site index for chain position `j` carrying a spin-`spin`
spin-orbital.  Occupation carries `("Nf", n, -1)` (the -1 modulus is
ITensor's fermion-parity flag) and, when requested, `("Sz", n*szval(spin))`.
"""
function physical_index(j::Int, spin::Int; conserve_sz::Bool = true)
    sz = szval(spin)
    space = if conserve_sz
        [QN(("Nf", 0, -1), ("Sz", 0)) => 1, QN(("Nf", 1, -1), ("Sz", sz)) => 1]
    else
        [QN(("Nf", 0, -1)) => 1, QN(("Nf", 1, -1)) => 1]
    end
    return Index(space; tags = "Fermion,Site,n=$j")
end

"""
    ancilla_index(j, spin; conserve_sz) -> Index

Bosonic partner index.  Mirrors the physical occupation through independent
quantum numbers ("Na", "Sza") so that pinning the MPS flux pins the PHYSICAL
sector; see the file header for why the absence of a `-1` modulus here is
load-bearing rather than incidental.
"""
function ancilla_index(j::Int, spin::Int; conserve_sz::Bool = true)
    sz = szval(spin)
    space = if conserve_sz
        [QN(("Na", 0), ("Sza", 0)) => 1, QN(("Na", 1), ("Sza", sz)) => 1]
    else
        [QN(("Na", 0)) => 1, QN(("Na", 1)) => 1]
    end
    return Index(space; tags = "Anc,Site,n=$j")
end

"""
    build_sites(L::PurificationLayout; conserve_sz=true) -> Vector{Index}

Site indices for the whole `4*ncas` chain, in chain order.
"""
function build_sites(L::PurificationLayout; conserve_sz::Bool = true)
    sites = Vector{Index}(undef, L.nsites)
    for w in 0:(L.nwires - 1)
        spin = L.spin_of_wire[w + 1]
        sites[L.physpos[w + 1]] = physical_index(
            L.physpos[w + 1], spin; conserve_sz = conserve_sz
        )
        sites[L.ancpos[w + 1]] = ancilla_index(
            L.ancpos[w + 1], spin; conserve_sz = conserve_sz
        )
    end
    return sites
end

"""
    sector_flux(L; conserve_sz) -> QN

Total quantum number of the beta = 0 purification: `nelecas` fermions with
`S_z = 0`, mirrored on the ancillas.  Pinning this flux is what restricts the
evolution to the CASCI `(nelecas, S_z = 0)` sector -- the Hamiltonian
conserves it, and ancilla occupations are individually frozen, so it holds
for every beta.
"""
function sector_flux(L::PurificationLayout; conserve_sz::Bool = true)
    ne = nelecas(L)
    szt = szval(0) * L.nalpha + szval(1) * L.nbeta      # = nalpha - nbeta
    return if conserve_sz
        QN(("Nf", ne, -1), ("Sz", szt), ("Na", ne), ("Sza", szt))
    else
        QN(("Nf", ne, -1), ("Na", ne))
    end
end
