# The fused backend: each Jordan-Wigner wire and its ancilla as ONE dim-4 site.
#
# Half the chain of the split backend at the SAME bond dimension -- that
# equality is measured, not hoped for (scratch geometry study, h2o CAS(8,6),
# kT = 0.25: qubit-blocked chi = 221 on 24 sites, fused-blocked chi = 222 on
# 12, orbital-fused chi = 664).  The phys|anc bond of the split chain carries
# only the local pair entanglement, so fusing it into the site is free; fusing
# the two SPIN channels instead (ITensor "Electron" sites) forces the orbital
# geometry these thermal states pay 3x chi for, which at chi^3 per update is
# a ~30x loss.  That is why this file exists and an Electron-site backend
# deliberately does not.
#
# Two structural simplifications fall out of the fusion:
#
#   * Jordan-Wigner strings cannot dress the ancillas, STRUCTURALLY: the
#     ancilla is inside the site, and the site's string operator is
#     F = (-1)^{n_phys} regardless of ancilla occupation.  The split
#     backend's bosonic-ancilla convention (sites.jl's long header) becomes a
#     two-line operator definition.
#   * The MPO is compiled directly on the chain -- no inflate step -- because
#     the fused site spends only THREE QNVals (Nf, Sz, Na; Sza is dropped,
#     see below), leaving the fourth for the compiler's internal charge.
#     The split purification chain spends all four, which is exactly why it
#     cannot be compiled on directly (hamiltonian.jl).
#
# DROPPING Sza IS DELIBERATE.  The split backend conserves it; here it would
# be the fifth QNVal and ITensor stops at four.  Nothing physical is lost --
# ancilla occupations are individually conserved by H (x) I, so the total Na
# plus the beta = 0 construction already pin the ancilla sector -- only some
# block fineness, and with it a little sparsity.
#
# Site basis, in index order 1..4, as (n_phys, n_anc):
#
#   1: (0,0)   2: (0,1)   3: (1,0)   4: (1,1)
#
# The wire ordering is BLOCKED and only blocked: alpha wires then beta wires,
# site w+1 = wire w.  Interleaved lost by 4-5x in the split backend and the
# fusion does not change that arithmetic.

using ITensors, ITensorMPS

# ------------------------------------------------------------- site type

function ITensors.space(
        ::SiteType"PurFermion";
        spin::Int = 0, conserve_sz::Bool = true
    )
    sz = szval(spin)
    return if conserve_sz
        [
            QN(("Nf", 0, -1), ("Sz", 0), ("Na", 0)) => 1,
            QN(("Nf", 0, -1), ("Sz", 0), ("Na", 1)) => 1,
            QN(("Nf", 1, -1), ("Sz", sz), ("Na", 0)) => 1,
            QN(("Nf", 1, -1), ("Sz", sz), ("Na", 1)) => 1,
        ]
    else
        [
            QN(("Nf", 0, -1), ("Na", 0)) => 1,
            QN(("Nf", 0, -1), ("Na", 1)) => 1,
            QN(("Nf", 1, -1), ("Na", 0)) => 1,
            QN(("Nf", 1, -1), ("Na", 1)) => 1,
        ]
    end
end

# Physical-fermion operators, ancilla untouched.  `C`/`Cdag` carry the string
# flag; `F` is the string operator itself, blind to the ancilla by
# construction rather than by convention.
ITensors.op!(Op::ITensor, ::OpName"N", ::SiteType"PurFermion", s::Index) =
    (Op[s' => 3, s => 3] = 1.0; Op[s' => 4, s => 4] = 1.0)
ITensors.op!(Op::ITensor, ::OpName"Na", ::SiteType"PurFermion", s::Index) =
    (Op[s' => 2, s => 2] = 1.0; Op[s' => 4, s => 4] = 1.0)
function ITensors.op!(Op::ITensor, ::OpName"F", ::SiteType"PurFermion", s::Index)
    Op[s' => 1, s => 1] = 1.0
    Op[s' => 2, s => 2] = 1.0
    Op[s' => 3, s => 3] = -1.0
    return Op[s' => 4, s => 4] = -1.0
end
ITensors.op!(Op::ITensor, ::OpName"C", ::SiteType"PurFermion", s::Index) =
    (Op[s' => 1, s => 3] = 1.0; Op[s' => 2, s => 4] = 1.0)
ITensors.op!(Op::ITensor, ::OpName"Cdag", ::SiteType"PurFermion", s::Index) =
    (Op[s' => 3, s => 1] = 1.0; Op[s' => 4, s => 2] = 1.0)
ITensors.op(::OpName"Id", ::SiteType"PurFermion", s::Index) =
    ITensors.denseblocks(delta(s', dag(s)))

# The pair-raiser |11><00| and its adjoint: the beta = 0 builders.  Declared
# parity-free exactly as in the split backend -- the proof in purification.jl
# that only |lambda_n| = 1/sqrt(dim) matters covers whatever signs an honest
# fermionic pair creator would have carried.
ITensors.op!(Op::ITensor, ::OpName"Raise", ::SiteType"PurFermion", s::Index) =
    (Op[s' => 4, s => 1] = 1.0)
ITensors.op!(Op::ITensor, ::OpName"Lower", ::SiteType"PurFermion", s::Index) =
    (Op[s' => 1, s => 4] = 1.0)

ITensors.has_fermion_string(::OpName"C", ::SiteType"PurFermion") = true
ITensors.has_fermion_string(::OpName"Cdag", ::SiteType"PurFermion") = true
ITensors.has_fermion_string(::OpName, ::SiteType"PurFermion") = false

ITensors.val(::ValName"00", ::SiteType"PurFermion") = 1
ITensors.val(::ValName"01", ::SiteType"PurFermion") = 2
ITensors.val(::ValName"10", ::SiteType"PurFermion") = 3
ITensors.val(::ValName"11", ::SiteType"PurFermion") = 4
ITensors.state(::StateName"00", ::SiteType"PurFermion") = [1.0, 0.0, 0.0, 0.0]
ITensors.state(::StateName"01", ::SiteType"PurFermion") = [0.0, 1.0, 0.0, 0.0]
ITensors.state(::StateName"10", ::SiteType"PurFermion") = [0.0, 0.0, 1.0, 0.0]
ITensors.state(::StateName"11", ::SiteType"PurFermion") = [0.0, 0.0, 0.0, 1.0]

# --------------------------------------------------------------- layout

"""
    FusedLayout(ncas, nalpha, nbeta)

Blocked-order fused chain: `2*ncas` sites, site `w+1` carrying wire `w`
(alpha wires `0:ncas-1`, then beta).  Mirrors the fields consumers of
`PurificationLayout` read, so shared helpers stay duck-typed.
"""
struct FusedLayout
    ncas::Int
    nalpha::Int
    nbeta::Int
    nwires::Int
    nsites::Int
    spin_of_wire::Vector{Int}
    orb_of_wire::Vector{Int}
end

function FusedLayout(ncas::Int, nalpha::Int, nbeta::Int)
    ncas >= 1 || throw(ArgumentError("ncas must be >= 1"))
    0 <= nalpha <= ncas || throw(ArgumentError("nalpha=$nalpha outside [0,$ncas]"))
    0 <= nbeta <= ncas || throw(ArgumentError("nbeta=$nbeta outside [0,$ncas]"))
    nwires = 2ncas
    spin_of_wire = [w < ncas ? 0 : 1 for w in 0:(nwires - 1)]
    orb_of_wire = [w < ncas ? w : w - ncas for w in 0:(nwires - 1)]
    return FusedLayout(ncas, nalpha, nbeta, nwires, nwires,
                       spin_of_wire, orb_of_wire)
end

nelecas(L::FusedLayout) = L.nalpha + L.nbeta
sector_dimension(L::FusedLayout) =
    binomial(L.ncas, L.nalpha) * binomial(L.ncas, L.nbeta)

"""
    build_sites(L::FusedLayout; conserve_sz=true) -> Vector{Index}

One "PurFermion" index per wire, in blocked order.
"""
function build_sites(L::FusedLayout; conserve_sz::Bool = true)
    return [
        siteind("PurFermion"; spin = L.spin_of_wire[w + 1],
                conserve_sz = conserve_sz,
                addtags = "n=$(w + 1)")
            for w in 0:(L.nwires - 1)
    ]
end

function sector_flux(L::FusedLayout; conserve_sz::Bool = true)
    ne = nelecas(L)
    szt = L.nalpha - L.nbeta
    return conserve_sz ? QN(("Nf", ne, -1), ("Sz", szt), ("Na", ne)) :
        QN(("Nf", ne, -1), ("Na", ne))
end

# --------------------------------------------------------- construction

"""
    purification_mpo(h1, g, L::FusedLayout, sites; tol, alg=:itensor, ...) -> MPO

`H (x) I_anc`, compiled DIRECTLY on the fused chain: `qc_opsum` is reused
verbatim (its `Cdag`/`C`/wire-position interface does not care about the local
dimension).

THE DEFAULT COMPILER HERE IS `:itensor`, NOT `:fast`, AND THIS IS A
CORRECTNESS CHOICE.  `ITensorMPOConstruction.MPO_new` mis-signs fermionic
operators on this site type: the compiled hopping terms acquire an extra
`(-1)^{n_anc}` on the operator's own site — measured element-by-element
against the split `H (x) I` at ncas = 2 (160 flipped elements, every one on
an ancilla-occupied destination), where ITensor's own compiler reproduces the
reference exactly.  The failure is invisible to `<H>`/`<H^2>` at beta = 0 and
to every thermodynamic self-consistency check (the mis-signed operator is a
sign-gauge conjugate S H S of the right one), which is precisely the failure
family this package keeps meeting; only an evolved state exported to the fixed
register shows it.  `test_fused.jl` pins the dense equality so a compiler
change cannot regress silently.  Cost of `:itensor` is once per molecule and
irrelevant next to the ladder.
"""
function purification_mpo(
        h1::AbstractMatrix{Float64}, g::AbstractArray{Float64, 4},
        L::FusedLayout, sites::Vector{<:Index};
        tol::Float64 = 1.0e-14, alg::Symbol = :itensor, kwargs...
    )
    os = qc_opsum(h1, g, L.ncas, :blocked, w -> w + 1; tol = tol)
    return build_mpo(os, sites; alg = alg, kwargs...)
end

"""
    infinite_temperature_mps(L::FusedLayout, sites; cutoff, maxdim) -> MPS

The beta = 0 sector purification.  Same pair-raising construction as the split
backend, but the raiser is now a SINGLE-SITE operator, so each application is
an MPO of bond dimension 2.
"""
function infinite_temperature_mps(
        L::FusedLayout, sites::Vector{<:Index};
        cutoff::Float64 = 1.0e-16, maxdim::Int = 10_000
    )
    @timeit TIMER "setup: psi0 (beta=0 MPS)" begin
        psi = MPS(sites, fill("00", L.nsites))
        for (spin, n) in ((0, L.nalpha), (1, L.nbeta))
            n == 0 && continue
            os = OpSum()
            for w in 0:(L.nwires - 1)
                L.spin_of_wire[w + 1] == spin || continue
                os += 1.0, "Raise", w + 1
            end
            K = MPO(os, sites)
            for _ in 1:n
                psi = apply(K, psi; cutoff = cutoff, maxdim = maxdim)
                normalize!(psi)
            end
        end
        normalize!(psi)
    end
    return psi
end

# ------------------------------------------------------------- read-out

"""
    occupations(psi, L::FusedLayout) -> (phys, anc)

Same contract as the split backend's method: `<n_w>` per wire, both halves.
"""
function occupations(psi::MPS, L::FusedLayout)
    ph = zeros(Float64, L.nwires)
    an = zeros(Float64, L.nwires)
    for w in 1:L.nwires
        ph[w] = _site_expect(psi, w, "N")
        an[w] = _site_expect(psi, w, "Na")
    end
    return ph, an
end

"""
    physical_rdm(psi, L::FusedLayout, wires; maxwires=12) -> Matrix{Float64}

Reduced density matrix over the listed wires, in the SAME register convention
as the split backend (listed wires ascending, first = most significant bit) --
the two backends' outputs are element-for-element comparable, and
`test_fused.jl` asserts it against the dense reference.

The ancilla trace is LOCAL here.  Each kept site's dim-4 index is split by an
exact unfuse isometry `U[s, p, a]` (built from the site's own four QNs, so it
is block-sparse and works whether or not Sz is conserved); the ancilla legs of
ket and bra then contract directly, and the physical legs stay open.  Dropped
sites contract ket against bra outright.  Bra bonds are primed by index
identity, not by tag, for the reason documented at the split `physical_rdm`.
"""
function physical_rdm(
        psi::MPS, L::FusedLayout, wires::AbstractVector{Int};
        maxwires::Int = 12, combine::Bool = true, meet::Bool = true
    )
    isempty(wires) && throw(ArgumentError("no wires requested"))
    all(0 .<= wires .< L.nwires) ||
        throw(ArgumentError("wires must lie in 0:$(L.nwires - 1)"))
    length(wires) <= maxwires ||
        throw(ArgumentError(
            "requested $(length(wires)) wires (2^$(2 * length(wires)) matrix); " *
                "raise maxwires only if you mean it"
        ))

    ws = sort(collect(wires))
    keep = Set(w + 1 for w in ws)
    s = siteinds(psi)

    ket_open = Dict{Int, Index}()
    bra_open = Dict{Int, Index}()
    function pieces(j)
        bra = prime(dag(psi[j]), uniqueinds(psi[j], s[j]))
        j in keep || return psi[j], bra
        # Site QNs in index order 1..4 = (0,0),(0,1),(1,0),(1,1): qs[1], qs[3]
        # span the physical bit at n_anc = 0; qs[1], qs[2] span the ancilla
        # bit at n_phys = 0.  Their sums reproduce every site QN exactly, so
        # the unfuse isometry U is flux-zero by construction.
        qs = [qn(space(s[j])[k]) for k in 1:4]
        p = Index([qs[1] => 1, qs[3] => 1]; tags = "phys,w=$(j - 1)",
                  dir = dir(s[j]))
        a = Index([qs[1] => 1, qs[2] => 1]; tags = "anc,w=$(j - 1)",
                  dir = dir(s[j]))
        U = ITensor(dag(s[j]), p, a)
        for (k, pi, ai) in ((1, 1, 1), (2, 1, 2), (3, 2, 1), (4, 2, 2))
            U[dag(s[j]) => k, p => pi, a => ai] = 1.0
        end
        ket_open[j] = p
        bra_open[j] = prime(dag(p))
        Ub = prime(dag(U), s[j])               # contracts the bra site leg
        Ub = prime(Ub, p)                      # bra physical leg -> p'
        # ancilla legs stay unprimed on both, so they contract: the trace.
        return psi[j] * U, prime(bra, s[j]) * Ub
    end
    open_of(j) = (ket_open[j], bra_open[j])

    # Same as the split backend: the order-2k final environment is intentional;
    # `maxwires` is the real guard.
    E = ITensors.@disable_warn_order _sweep_env(
        pieces, open_of, keep, length(psi), combine, meet
    )

    kets = [ket_open[w + 1] for w in ws]
    bras = [bra_open[w + 1] for w in ws]
    T = array(E, reverse(kets)..., reverse(bras)...)
    K = 1 << length(ws)
    return Matrix{Float64}(real(reshape(T, K, K)))
end

physical_rho(psi::MPS, L::FusedLayout; maxwires::Int = 12) =
    physical_rdm(psi, L, 0:(L.nwires - 1); maxwires = maxwires)

"""
    thermal_ladder(case, kTs, ::Val{:fused}; ...) -> (L, sites, H, snaps)

Fused-backend convenience entry point from a Module G record.  The evolution
driver is the SAME `thermal_ladder(H, psi0, dim, betas)` as the split backend
-- it never looks at the layout.
"""
function thermal_ladder(
        case::MoleculeCase, kTs::AbstractVector{<:Real}, ::Val{:fused};
        conserve_sz::Bool = true, tol::Float64 = 1.0e-14,
        mpo_kwargs = (;), kwargs...
    )
    all(kTs .> 0) || throw(ArgumentError("temperatures must be positive"))
    L, sites = @timeit TIMER "setup: layout+sites" begin
        Lb = FusedLayout(case.ncas, case.nalpha, case.nbeta)
        (Lb, build_sites(Lb; conserve_sz = conserve_sz))
    end
    H = purification_mpo(case.h1, case.g, L, sites; tol = tol, mpo_kwargs...)
    psi0 = infinite_temperature_mps(L, sites)
    betas = sort([1 / kT for kT in kTs])
    snaps = thermal_ladder(H, psi0, sector_dimension(L), betas; kwargs...)
    return L, sites, H, snaps
end
