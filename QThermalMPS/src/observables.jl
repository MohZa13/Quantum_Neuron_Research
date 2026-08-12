# Reading physical quantities off a purification.
#
# The one fact this whole file rests on is the ancilla convention of
# `sites.jl`: ancillas are bosonic, so a physical operator O lifts to the
# purification chain as O (x) I_anc with no Jordan-Wigner dressing, and
#
#     <Psi| O (x) I |Psi>  =  Tr[rho O]
#
# for the Module I encoded rho.  Everything below is a way of evaluating that
# contraction without ever forming rho -- except `physical_rho`, which forms
# it on purpose and is guarded accordingly.
#
# WHAT THE NEURON LAYER CONSUMES.  `qnn/states.py` wants dense (M, K, K)
# stacks.  Three routes out, in increasing order of how large a register they
# survive:
#
#   physical_rho  exact, whole register, dies at 2^(2 ncas) -- ncas <= 6
#   physical_rdm  exact, any chosen subset of wires, cost 2^(2|wires|)
#   pauli_expect  one number per operator, never forms a matrix at all
#
# `physical_rdm` is the one that scales: the quantum layer's cost is O(K^3) in
# the register size regardless of where the state came from, so a register
# small enough to train on is always small enough to materialise.

using ITensors, ITensorMPS
using LinearAlgebra

"""
    sector_mean_energy(h1, g, L) -> Float64

`Tr[H P] / dim` over the CASCI sector, in closed form -- an EXACT reference
for `<Psi(0)|H|Psi(0)>` that costs `O(ncas^2)` and never touches a spectrum.

This is the only external check that survives past dense reach.  At
`ncas <= 8` the MPO can be validated against the stored `evals`; at `ncas = 10`
the sector holds 63,504 states and there is no spectrum to compare to, but the
beta = 0 purification is exactly `P/dim`, so its energy is this number and a
disagreement means the MPO is wrong.

Under a uniform average over sector determinants only two kinds of two-body
term survive the trace: the direct one (`p=q`, `r=s`, giving `+n_p n_r`) and
the exchange one (`p=s`, `r=q` at equal spin, giving `-n_p n_r` after
reordering).  Everything else moves a determinant somewhere else and has no
diagonal.  Verified to 1e-14 against `mean(evals)` on CAS(8,6) and CAS(8,8).
"""
function sector_mean_energy(
        h1::AbstractMatrix{Float64}, g::AbstractArray{Float64, 4},
        L::PurificationLayout
    )
    ncas, na, nb = L.ncas, L.nalpha, L.nbeta
    ncas >= 2 || throw(ArgumentError("closed form needs ncas >= 2"))
    aa, ab = na / ncas, nb / ncas
    ba = na * (na - 1) / (ncas * (ncas - 1))
    bb = nb * (nb - 1) / (ncas * (ncas - 1))

    e1 = 0.0
    for p in 1:ncas
        e1 += h1[p, p]
    end
    e1 *= (aa + ab)

    dsum = 0.0                    # sum over all p,r of g[p,p,r,r]
    offsum = 0.0                  # sum over p != r of g[p,p,r,r] - g[p,r,r,p]
    @inbounds for p in 1:ncas, r in 1:ncas
        dsum += g[p, p, r, r]
        p == r && continue
        offsum += g[p, p, r, r] - g[p, r, r, p]
    end
    return e1 + 0.5 * ((ba + bb) * offsum + 2 * aa * ab * dsum)
end

"""
    energy(psi, H) -> Float64

`<H>` on a normalised purification.  Equal to `Tr[rho H]`, the thermal energy
excluding `ecore`.
"""
energy(psi::MPS, H::MPO) = real(inner(psi', H, psi))

"""
    energy_variance(psi, H) -> Float64

`<H^2> - <H>^2`.  NOTE this is the PHYSICAL energy variance of a thermal
state, not a convergence diagnostic -- it is large and nonzero for a converged
answer, unlike the ground-state case.  Use it as `C_v * kT^2`, or as one more
quantity to check against an exact Boltzmann sum; do not read it as an error.
"""
function energy_variance(psi::MPS, H::MPO)
    e = energy(psi, H)
    return real(inner(H, psi, H, psi)) - e^2
end

"""
    occupations(psi, L) -> (phys, anc)

`<n_w>` on every physical wire and its ancilla, in wire order.

THE ANCILLA NUMBERS ARE NOT FROZEN, which is worth saying because `H` acts as
the identity on the ancillas and it is natural to expect they would be.  That
argument works for real time and fails for imaginary time: `e^{-bH/2}` is not
unitary, so it reweights the Schmidt spectrum and the ancilla marginal moves
with it.  What is true is stronger and more useful --

    rho_anc = A^T A = A A^T = rho_phys ,   A = e^{-bH/2} P ,

because `A` is real symmetric and the beta = 0 state carries no relative signs.
So `anc` must equal `phys` wire for wire at EVERY temperature.  That identity
is sensitive to ordering and sign conventions on both halves of the chain, so
it makes a good running check; `<n_a> == nalpha/ncas` does not, and is only
true at beta = 0.
"""
function occupations(psi::MPS, L::PurificationLayout)
    ph = zeros(Float64, L.nwires)
    an = zeros(Float64, L.nwires)
    for w in 1:L.nwires
        ph[w] = _site_expect(psi, L.physpos[w], "N")
        an[w] = _site_expect(psi, L.ancpos[w], "N")
    end
    return ph, an
end

# Applied-operator overlap rather than an orthogonality-centre contraction:
# same reason `physical_rdm` sweeps the chain -- the states coming out of the
# subspace expansion are not canonical, and a gauge assumption here would be
# silently wrong rather than loudly.
function _site_expect(psi::MPS, j::Int, opname::String)
    phi = copy(psi)
    sj = siteind(phi, j)
    phi[j] = noprime(op(opname, sj) * phi[j])
    return real(inner(psi, phi))
end

# --------------------------------------------------------- dense extraction

"""
    _sweep_env(pieces, open_of, keep, N, combine, meet) -> ITensor

Shared environment builder behind both backends' `physical_rdm`.

`pieces(j)` returns the (ket, bra) tensors of site `j`, with the bra's bonds
primed and, on kept sites, its open leg primed; `open_of(j)` names the (ket,
bra) open indices a kept site leaves behind.  The function contracts the whole
chain -- one sweep, or two half-sweeps joined at a bond that splits the kept
set evenly (`meet`) -- folding each side's opens into combined QN indices as
they accumulate (`combine`) and undoing every combiner before returning, so
callers see the individual open indices regardless of path.
"""
function _sweep_env(
        pieces::Function, open_of::Function, keep, N::Int,
        combine::Bool, meet::Bool
    )
    kept_sorted = sort(collect(keep))
    # Split the chain after the site holding the ceil(k/2)-th kept wire, so
    # each half-environment carries at most half the open register.
    mid = meet && length(kept_sorted) > 1 ?
        kept_sorted[cld(length(kept_sorted), 2)] : N

    cs = ITensor[]
    function half(range)
        E = ITensor(1.0)
        ket_comb = nothing
        bra_comb = nothing
        for j in range
            ket, bra = pieces(j)
            E = E * ket * bra
            if combine && j in keep
                ko, bo = open_of(j)
                kC = ket_comb === nothing ? combiner(ko) : combiner(ket_comb, ko)
                bC = bra_comb === nothing ? combiner(bo) : combiner(bra_comb, bo)
                E = E * kC * bC
                push!(cs, kC)
                push!(cs, bC)
                ket_comb = combinedind(kC)
                bra_comb = combinedind(bC)
            end
        end
        return E
    end

    E = mid >= N ? half(1:N) : half(1:mid) * half(N:-1:(mid + 1))
    for C in reverse(cs)
        E = E * dag(C)
    end
    return E
end

"""
    dense_purification(psi, sites) -> Array

Contract the whole MPS into one dense array, one axis per chain site.  Tests
and tiny cases only: this is `2^(4 ncas)` numbers.
"""
function dense_purification(psi::MPS, sites::Vector{<:Index})
    length(sites) <= 24 ||
        throw(ArgumentError("refusing to densify $(length(sites)) sites"))
    T = psi[1]
    for j in 2:length(psi)
        T *= psi[j]
    end
    return array(T, sites...)
end

"""
    dense_mpo(H, sites) -> Matrix

Contract an MPO into a dense `2^N x 2^N` matrix, rows indexed by the primed
(bra) site indices.  Tests only -- this is what lets a small case be checked
against `exp(-beta*H)` computed with no tensor network anywhere in sight.
"""
function dense_mpo(H::MPO, sites::Vector{<:Index})
    N = length(sites)
    N <= 12 || throw(ArgumentError("refusing to densify a $N-site MPO"))
    T = H[1]
    for j in 2:N
        T *= H[j]
    end
    # Reversed: `reshape` is column-major, so the FIRST axis varies fastest and
    # would become the least significant bit.  Feeding the sites in reverse
    # puts site 1 in the most significant position, which is the register
    # convention Module I uses (wire 0 = MSB).
    A = array(T, reverse(prime.(sites))..., reverse(sites)...)
    K = 1 << N
    return reshape(A, K, K)
end

"""
    physical_rdm(psi, L, wires; maxwires=12) -> Matrix{Float64}

`rho` reduced onto a subset of PHYSICAL wires: ancillas and the unlisted
physical wires are traced out.

`wires` are 0-based Jordan-Wigner wires.  The returned matrix is indexed the
way Module I indexes the register -- the listed wires in ascending order, the
first of them most significant -- so for `wires = 0:(2 ncas - 1)` this is
exactly `qthermal.mps`'s encoded state.

THE BRA IS PRIMED BY INDEX, NOT BY TAG.  The idiomatic spelling of this
contraction is `prime(dag(psi), "Link")`, and it is a trap: `expand` returns an
MPS whose new bonds are tagged `"sum"` rather than `"Link"`, so those bonds
never get primed, ket and bra contract straight through them, and the result
is silently multiplied by 2 for every bond missed (`tr(rho) = 16` on an
eight-site chain).  Nothing about the state is wrong when this happens -- norm
1, flux right, overlap 1 with the state it came from -- so it reads as a
physics bug and is not one.  Priming whatever is not the site index is
tag-independent and cannot miss.

Sweeping the whole chain rather than just the window the wires span is the
same kind of insurance: it costs `O(N chi^3)` and removes any dependence on
the MPS being in canonical form, which the expanded states are not.

Cost is `O(4^|wires|)` in memory; the `maxwires` guard is there because the
failure mode is an allocation the machine cannot satisfy.

THE CONTRACTION MEETS IN THE MIDDLE (`meet = true`).  Sweeping one
environment across the whole chain carries all `k` open wires through every
bond -- `sum_k 4^k chi^3` work, ~3e13 FLOPs for the ncas = 10, chi = 256
export, which is why production spent ~15 minutes per state.  Building a left
and a right environment that each carry only their own half of the open wires
and joining them once across the middle bond costs `O(4^{k/2} chi^3)` plus one
`4^k chi^2` join -- hundreds of times less.  Within each half, ket-side and
bra-side opens are folded into one combined QN index each (`combine = true`),
which is what keeps the block COUNT from exploding even where the FLOPs are
small; the combiners are undone before extraction, so the register layout of
the result is untouched.  Both knobs exist so the paths can be benchmarked
and cross-checked against each other; the tests pin all of them to the same
answer.
"""
function physical_rdm(
        psi::MPS, L::PurificationLayout, wires::AbstractVector{Int};
        maxwires::Int = 12, combine::Bool = true, meet::Bool = true
    )
    isempty(wires) && throw(ArgumentError("no wires requested"))
    all(0 .<= wires .< L.nwires) ||
        throw(ArgumentError("wires must lie in 0:$(L.nwires - 1)"))
    length(wires) <= maxwires ||
        throw(ArgumentError(
            "requested $(length(wires)) wires (2^$(2*length(wires)) matrix); " *
                "raise maxwires only if you mean it"
        ))

    ws = sort(collect(wires))
    pos = [L.physpos[w + 1] for w in ws]          # ascending, since physpos is
    keep = Set(pos)

    p = psi
    s = siteinds(p)

    # ket-opener for one site: returns (ket tensor, bra tensor); the bra has
    # bonds primed by index identity and, on kept sites, the site index primed.
    function pieces(j)
        Adj = prime(dag(p[j]), uniqueinds(p[j], s[j]))
        j in keep && (Adj = prime(Adj, s[j]))
        return p[j], Adj
    end
    open_of(j) = (s[j], prime(dag(s[j])))

    # The final environment intentionally carries all 2k open legs (k = 10 is
    # order 20), so ITensors' order-14 warning would fire -- with a full stack
    # trace -- on every join of every export.  The `maxwires` guard above is
    # the real protection; silence the advisory one for this contraction.
    E = ITensors.@disable_warn_order _sweep_env(
        pieces, open_of, keep, length(p), combine, meet
    )

    # Reversed for the same reason as `dense_mpo`: column-major `reshape` makes
    # the leading axis least significant, and the register convention is that
    # the lowest-numbered wire is MOST significant.
    kept = [s[j] for j in pos]
    T = array(E, reverse(kept)..., reverse(prime.(kept))...)
    K = 1 << length(ws)
    rho = reshape(T, K, K)
    return Matrix{Float64}(real(rho))
end

"""
    physical_rho(psi, L; maxwires=12) -> Matrix{Float64}

The full physical-register density matrix, `2^(2 ncas)` square.  A convenience
wrapper on [`physical_rdm`](@ref) over every wire; the guard bites at
`ncas > 6`, which is the point past which the dense route was never the plan.
"""
physical_rho(psi::MPS, L::PurificationLayout; maxwires::Int = 12) =
    physical_rdm(psi, L, 0:(L.nwires - 1); maxwires = maxwires)

# ------------------------------------------------------ Pauli expectations

# Qubit Paulis on a JW wire, with |Emp> = |0> and |Occ> = |1>.  These are
# *qubit* operators, not fermionic ones -- they carry no Jordan-Wigner string,
# which is exactly right: a Pauli string on the register is the JW image of
# some fermionic operator, and re-inserting strings would double-count.
#
# X and Y are NOT representable as single QN-block-sparse ITensors: each mixes
# flux +1 and -1.  So they are expanded into raising/lowering products, and
# only the products with net zero (Nf, Sz) can contribute in a flux-definite
# state -- the rest are not merely small, they connect different sectors and
# `inner` would refuse them.  For the pool operators that matter (X_pX_q,
# Y_pY_q) the filter leaves two terms out of four.
#
# Z needs no expansion: on a two-level fermion site the Jordan-Wigner string
# operator IS Pauli Z, `op("F") = diag(+1, -1)`, already flux-zero and already
# a single block-sparse tensor.  Expanding it as `Id - 2N` instead would make
# a length-k Z string cost 2^k terms for no reason.
#
# Entries are `(coefficient, operator name, delta Nf)`; delta Sz follows from
# delta Nf and the wire's spin.
const _PAULI_TERMS = Dict(
    'Z' => [(1.0 + 0im, "F", 0)],
    'X' => [(1.0 + 0im, "Raise", +1), (1.0 + 0im, "Lower", -1)],
    'Y' => [(0.0 + 1im, "Raise", +1), (0.0 - 1im, "Lower", -1)],
)

"""
    pauli_expect(psi, L, pstring) -> ComplexF64

`Tr[rho P]` for a Pauli string over the physical register.  `pstring` is a
string of `I`/`X`/`Y`/`Z` of length `2*ncas`, character `w+1` acting on JW
wire `w`.

Never forms `rho`, and never grows a bond: each surviving term is a product of
single-site operators applied straight to the MPS tensors.  The work is
`2^(#X + #Y)` terms before the flux filter, so this is a tool for local pool
operators, not for dense strings.
"""
function pauli_expect(psi::MPS, L::PurificationLayout, pstring::AbstractString)
    length(pstring) == L.nwires ||
        throw(ArgumentError("pstring has length $(length(pstring)), expected $(L.nwires)"))

    active = [(w, c) for (w, c) in enumerate(pstring) if c != 'I']
    for (_, c) in active
        haskey(_PAULI_TERMS, c) || throw(ArgumentError("bad Pauli character '$c'"))
    end
    isempty(active) && return ComplexF64(norm(psi)^2)

    total = ComplexF64(0)
    # Cartesian product over the raising/lowering expansion of each factor.
    choices = [_PAULI_TERMS[c] for (_, c) in active]
    for combo in Iterators.product(choices...)
        # Both conserved charges must come back to zero.  Sz as well as Nf:
        # a raise on an alpha wire and a lower on a beta wire cancel in
        # particle number but shift Sz by 2, and `inner` rejects that outright
        # rather than returning the zero it mathematically is.
        dn = 0
        dsz = 0
        for (k, (w, _)) in enumerate(active)
            dn += combo[k][3]
            dsz += combo[k][3] * szval(L.spin_of_wire[w])
        end
        (dn == 0 && dsz == 0) || continue

        coef = prod(t[1] for t in combo)
        iszero(coef) && continue
        phi = copy(psi)
        for (k, (w, _)) in enumerate(active)
            j = L.physpos[w]
            sj = siteind(phi, j)
            phi[j] = noprime(op(combo[k][2], sj) * phi[j])
        end
        total += coef * inner(psi, phi)
    end
    return total
end
