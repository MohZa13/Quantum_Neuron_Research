# The active-space Hamiltonian as an ITensor OpSum / MPO.
#
# Built from exactly the two tensors the Python pipeline already stores:
#
#   H = ecore
#     + sum_{pq,s}      h1eff[p,q]   a+_{ps} a_{qs}
#     + 1/2 sum_{pqrs}  g[p,q,r,s]   a+_{ps1} a+_{rs2} a_{ss2} a_{qs1}
#
# with `g` in CHEMIST notation (pq|rs), the full ncas^4 tensor as stored by
# `qthermal/hamiltonian.py` (NOT s8-packed).  This mirrors
# `qthermal/encode.py::jw_hamiltonian` term for term, so the two agree by
# construction.
#
# Jordan-Wigner is canonical and implicit: ITensor's "Fermion" site type
# carries the fermion-parity flag, and OpSum -> MPO inserts the JW strings
# itself under the same convention the Python encoder applies by hand.  We
# never write a Pauli string out explicitly -- doing so would throw away the
# U(1) block sparsity that makes the whole thing tractable.
#
# `ecore` is deliberately NOT folded into the MPO: the pipeline's stored
# energies exclude it (`TruncatedEnsemble.E`), and keeping it out means the
# MPO's spectrum is directly comparable with the `evals` dataset.

using ITensors, ITensorMPS
using ITensorMPOConstruction: MPO_new

"""
    qc_opsum(h1, g, ncas, ordering, wirepos; tol) -> OpSum

Second-quantized active-space Hamiltonian as an `OpSum`.

`wirepos(w)` maps a 0-based JW wire to its 1-based chain position, which is
what lets the same builder serve both the bare physical chain (validation,
DMRG) and the interleaved purification chain.

Coefficients below `tol` are dropped -- on real CASCI tensors this removes a
large fraction of the `ncas^4` two-body terms at no accuracy cost.
"""
function qc_opsum(
        h1::AbstractMatrix{Float64}, g::AbstractArray{Float64, 4},
        ncas::Int, ordering::Symbol, wirepos::Function;
        tol::Float64 = 1.0e-14
    )
    size(h1) == (ncas, ncas) || throw(DimensionMismatch("h1 must be ($ncas,$ncas)"))
    size(g) == (ncas, ncas, ncas, ncas) ||
        throw(DimensionMismatch("g must be ($ncas,$ncas,$ncas,$ncas)"))

    # 0-based wire -> 1-based chain position, precomputed
    wire = [[wirepos(jw_wire(p, s, ncas, ordering)) for p in 0:(ncas - 1)] for s in 0:1]

    return @timeit TIMER "setup: H opsum" _qc_opsum(h1, g, ncas, wire, tol)
end

function _qc_opsum(h1, g, ncas, wire, tol)
    os = OpSum()

    # `ITensorMPS.add!`, NOT `os += ...`: += copies the whole OpSum per term,
    # which is quadratic in the O(ncas^4) term count -- measured 38.5 s and
    # 131 GiB of churn at ncas = 12, and ~40 min extrapolated at ncas = 20.
    # add! appends in place (83x at 20k terms) and builds the identical OpSum.
    @inbounds for p in 1:ncas, q in 1:ncas
        c = h1[p, q]
        abs(c) < tol && continue
        for s in 1:2
            ITensorMPS.add!(os, c, "Cdag", wire[s][p], "C", wire[s][q])
        end
    end

    @inbounds for p in 1:ncas, q in 1:ncas, r in 1:ncas, s in 1:ncas
        c = 0.5 * g[p, q, r, s]
        abs(c) < tol && continue
        for s1 in 1:2, s2 in 1:2
            wp, wq = wire[s1][p], wire[s1][q]
            wr, ws = wire[s2][r], wire[s2][s]
            # a+_p a+_r a_s a_q vanishes identically when the two creators
            # (or the two annihilators) are the same spin-orbital.
            (wp == wr || wq == ws) && continue
            ITensorMPS.add!(os, c, "Cdag", wp, "Cdag", wr, "C", ws, "C", wq)
        end
    end

    return os
end

"""
    op_basis(sites) -> Vector{Vector{String}}

Per-site operator basis for `ITensorMPOConstruction`, identity first.

This is not decoration.  `MPO_new` requires every term to carry at most one
operator per site, and the two-body terms violate that whenever a creator and
an annihilator share a spin-orbital (`p == q`, `p == s`, ...).  Given a basis
it rewrites such products into it -- `Cdag*C -> N`, `C*Cdag -> I - N` -- and
`{I, C, Cdag, N}` spans the whole 2x2 algebra of a "Fermion" site, so the
rewrite is always possible.  At most two operators can ever land on one site
(the creators are distinct by the `wp == wr` skip, and so are the
annihilators), which is why no longer products need to be representable.
"""
op_basis(sites::Vector{<:Index}) =
    [hastags(s, "Anc") ? ["I"] : ["I", "C", "Cdag", "N"] for s in sites]

"""
    build_mpo(os, sites; alg, graph_alg, cutoff, check) -> MPO

Compile an `OpSum`.  `alg`:

  * `:fast` (default) -- `ITensorMPOConstruction.MPO_new`, a bipartite-graph
    construction that is exact by design.  It is the only option that stays
    usable as `ncas` grows: the term count is `O(ncas^4)`.
  * `:itensor` -- ITensor's built-in `MPO(os, sites; cutoff)`.  Kept as the
    independent reference the fast path is tested against, not as a fallback.

`graph_alg` selects `MPO_new`'s internal algorithm: `"VC"` (minimum vertex
cover, the default) or `"QR"`.  The package's own electronic-structure
benchmark has them tie on bond dimension while `"VC"` is several times faster
and produces a markedly sparser MPO, which is what later DMRG/TDVP contraction
cost actually tracks.

`cutoff` applies to `:itensor` only: `MPO_new` builds exactly, and we do not
want it compressed -- `energy_variance` uses `H^2` as a convergence probe and
an approximate H would put a floor under it.

`check` turns on `MPO_new`'s input validation and per-tensor flux checks.  They
are off by default because they are expensive and because the real check is
`test/test_hamiltonian.jl`, which reproduces the stored CASCI spectrum.
"""
function build_mpo(
        os::OpSum, sites::Vector{<:Index};
        alg::Symbol = :fast, graph_alg::String = "VC",
        cutoff::Float64 = 1.0e-15, check::Bool = false
    )
    if alg === :fast
        return @timeit TIMER "setup: H compile (MPO_new)" MPO_new(
            os, sites;
            alg = graph_alg,
            basis_op_cache_vec = op_basis(sites),
            check_for_errors = check, checkflux = check
        )
    end
    alg === :itensor &&
        return @timeit TIMER "setup: H compile (itensor)" MPO(os, sites; cutoff = cutoff)
    throw(ArgumentError("unknown MPO alg $alg (:fast or :itensor)"))
end

"""
    physical_wirepos(L) / purification_wirepos(L)

The two `wirepos` maps: the bare `2*ncas`-site physical chain used for
validation, and the interleaved `4*ncas`-site purification chain.
"""
physical_wirepos(::PurificationLayout) = (w -> w + 1)
purification_wirepos(L::PurificationLayout) = (w -> L.physpos[w + 1])

"""
    physical_sites(L; conserve_sz) -> Vector{Index}

Bare `2*ncas`-site fermionic chain (no ancillas) -- the validation register.
"""
function physical_sites(L::PurificationLayout; conserve_sz::Bool = true)
    return [
        physical_index(w + 1, L.spin_of_wire[w + 1]; conserve_sz = conserve_sz)
            for w in 0:(L.nwires - 1)
    ]
end

"""
    physical_indices(L, sites) -> Vector{Index}

The physical sub-chain of a purification chain, in wire order.  These are the
same `Index` objects `sites` holds, which is what makes `inflate_mpo` legal.
"""
physical_indices(L::PurificationLayout, sites::Vector{<:Index}) =
    [sites[L.physpos[w + 1]] for w in 0:(L.nwires - 1)]

"""
    physical_flux(L; conserve_sz) -> QN

Target quantum number of the bare physical chain: `nelecas` electrons at
`S_z = nalpha - nbeta`.
"""
function physical_flux(L::PurificationLayout; conserve_sz::Bool = true)
    ne = nelecas(L)
    szt = L.nalpha - L.nbeta
    return conserve_sz ? QN(("Nf", ne, -1), ("Sz", szt)) : QN(("Nf", ne, -1))
end

"""
    physical_mpo(h1, g, L, psites; tol, alg, cutoff) -> MPO

H on a bare `2*ncas`-wire chain.
"""
function physical_mpo(
        h1::AbstractMatrix{Float64}, g::AbstractArray{Float64, 4},
        L::PurificationLayout, psites::Vector{<:Index};
        tol::Float64 = 1.0e-14, kwargs...
    )
    os = qc_opsum(h1, g, L.ncas, L.ordering, physical_wirepos(L); tol = tol)
    return build_mpo(os, psites; kwargs...)
end

"""
    purification_mpo(h1, g, L, sites; tol, alg, ...) -> MPO

`H (x) I_anc` on the `4*ncas`-site purification chain: compile on the physical
chain, then splice identity tensors in at the ancilla positions.

COMPILING ON THE FULL CHAIN IS NOT AN OPTION, which is worth stating because
it looks like the obvious way to do this.  ITensor's `QN` holds at most FOUR
`QNVal`s, and the purification chain already spends all four on
`(Nf, Sz, Na, Sza)`; `MPO(os, sites)` needs one more for its own bookkeeping
and dies with "QN already contains maximum number of QNVals".  The
`ITensorMPOConstruction` path does not die -- it silently produces an operator
that differs from this one in the SIGN of some off-diagonal elements, which
`<Psi(0)|H|Psi(0)>` and `<Psi(0)|H^2|Psi(0)>` both fail to notice (they are
sums of squares) and only an evolved state exposes.

Building on the physical chain sidesteps both: it spends only two quantum
numbers, it is half the chain to compile, and `H (x) I_anc` becomes structural
rather than conventional -- there is no path by which a Jordan-Wigner string
could acquire a factor on an ancilla, because the ancilla tensors are
identities that were never part of the compilation.
"""
function purification_mpo(
        h1::AbstractMatrix{Float64}, g::AbstractArray{Float64, 4},
        L::PurificationLayout, sites::Vector{<:Index};
        tol::Float64 = 1.0e-14, kwargs...
    )
    Hp = physical_mpo(h1, g, L, physical_indices(L, sites); tol = tol, kwargs...)
    return @timeit TIMER "setup: H inflate" inflate_mpo(Hp, L, sites)
end
