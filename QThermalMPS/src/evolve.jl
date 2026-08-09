# Imaginary-time evolution of the purification: the whole temperature ladder
# in one pass.
#
#     |Psi(beta)> = e^{-beta H / 2} |Psi(0)> ,   Tr_anc |Psi><Psi| = rho(beta)
#
# applied by TDVP against the Hamiltonian MPO.  TDVP rather than Trotter gates
# because H here is all-to-all in the orbitals: a TEBD circuit would need a
# swap network whose depth eats the advantage, while TDVP works directly with
# the long-range MPO.  Two-site TDVP specifically -- the one-site variant
# cannot grow a bond, and the beta = 0 state starts at bond dimension O(ncas^2)
# while the interesting temperatures need far more.
#
# ONE PASS, EVERY TEMPERATURE.  Evolution sweeps *through* every intermediate
# beta, so a snapshot at each requested kT is free.  The Python pipeline
# re-truncates a spectrum per temperature; here the ladder costs the same as
# its coldest rung.
#
# THE NORM IS THE PARTITION FUNCTION.  <Psi(0)|Psi(0)> = 1 and rho(0) = P/dim,
# so
#
#     Z(beta) = Tr e^{-beta H} = dim * || e^{-beta H/2} |Psi(0)> ||^2,
#
# which the evolution hands over for free provided we do not throw the norm
# away.  That is why the state is renormalised in bounded chunks with the
# logarithm accumulated, rather than by passing `normalize=true` to TDVP: the
# norm carries log Z, and log Z carries the free energy and the entropy.
# Chunking exists because the raw norm grows like e^{-beta E_min / 2}, which
# overflows a Float64 somewhere around beta = 60 for these molecules.

using ITensors, ITensorMPS
using Printf

"""
    thermal_ladder(H, psi0, dim, betas; kwargs...) -> Vector{ThermalSnapshot}

Evolve `psi0` (the beta = 0 purification, normalised) through the ascending
`betas`, snapshotting each.  `dim` is the CI sector dimension
([`sector_dimension`](@ref)), needed to turn the norm into `log Z`.

THE BOND SPACE HAS TO BE OPENED FIRST.  TDVP evolves *within* the manifold of
its current bond dimension, and the beta = 0 purification starts at
`O(ncas^2)` while the interesting temperatures need orders of magnitude more.
Two-site TDVP can in principle grow a bond, but only along directions the
current tensors already have weight in -- and under U(1)xU(1) symmetry whole
quantum-number blocks are simply absent, so it frequently cannot grow at all.
The failure is silent and vicious: the run completes, the norm still
integrates `-<H>` exactly (so `logZ` and `energy` stay mutually consistent),
and the answer is converged to the wrong state.  It is also invisible to a
step-size study -- halving `dbeta` reproduces the same wrong number to every
digit, because the manifold is exactly invariant.

So a global Krylov subspace expansion is run BEFORE each chunk while the bond
dimension is still growing, adding directions from `H|psi>` at zero amplitude:
the state is unchanged, the manifold it may move in is not.  Expansion pauses
once `chi` has failed to grow `stall_limit` times in a row, and RE-ARMS at
geometrically spaced betas -- a stall at high temperature says nothing about
whether more bonds will be needed later, because chi does not grow
monotonically: measured on CAS(8,6) it peaks around beta = 4 and falls again
by beta = 40.  Switching the expansion off for good at the first stall would
freeze the manifold exactly before the hard region.  Re-arming at doublings
keeps the extra cost logarithmic in beta.  Pass `expand_cutoff = nothing` to
disable it, but read `test/test_evolve.jl` first.

THE STEP SIZE IS GRADED.  Local TDVP error scales with how fast the state is
moving, and `d|Psi>/d beta = -H|Psi>/2` slows down as beta grows: the thermal
weight concentrates on a shrinking part of the spectrum, so the energy spread
driving the evolution shrinks with it.  A uniform grid therefore spends most
of its steps where nothing is happening -- reaching kT = 0.025 means beta = 40,
which is 800 steps at `dbeta = 0.05` and the last 700 of them are wasted.  The
step used at `beta` is

    dbeta_eff = min(dbeta_max, dbeta * (1 + beta / ramp))

which is still a single-knob convergence study: `dbeta` scales the whole
schedule, so halving it halves every step.

Keywords:
  * `dbeta = 0.05`  -- TDVP step at beta = 0, and the scale of the whole
    schedule.  The accuracy knob; halve it and re-run to certify a result
    (`converge_dbeta`).
  * `dbeta_max = 10*dbeta`, `ramp = 1.0` -- the grading above.  Set
    `dbeta_max = dbeta` to recover a uniform grid.
  * `maxdim = 256`, `cutoff = 1e-10` -- bond truncation.
  * `nsite = 2` -- two-site TDVP.  Setting this to 1 freezes the bond
    dimension outright.
  * `chunk = 1.0` -- longest stretch of beta between renormalisations.
    Clamped internally against the running energy so the norm cannot overflow.
  * `expand_cutoff = 1e-9`, `expand_every = 10*dbeta`, `expand_krylovdim = 2`,
    `stall_limit = 3` -- the subspace expansion schedule above.
"""
function thermal_ladder(
        H::MPO, psi0::MPS, dim::Int, betas::AbstractVector{<:Real};
        dbeta::Float64 = 0.05, dbeta_max::Union{Nothing, Float64} = nothing,
        ramp::Float64 = 1.0,
        maxdim::Int = 256, cutoff::Float64 = 1.0e-10,
        nsite::Int = 2, chunk::Float64 = 1.0,
        expand_cutoff::Union{Nothing, Float64} = 1.0e-9,
        expand_every::Union{Nothing, Float64} = nothing,
        expand_krylovdim::Int = 2, stall_limit::Int = 3,
        outputlevel::Int = 0, verbose::Bool = false
    )
    isempty(betas) && return ThermalSnapshot[]
    issorted(betas) || throw(ArgumentError("betas must be ascending"))
    first(betas) >= 0 || throw(ArgumentError("betas must be non-negative"))
    dbeta > 0 || throw(ArgumentError("dbeta must be positive"))
    ramp > 0 || throw(ArgumentError("ramp must be positive"))
    dmax = dbeta_max === nothing ? 10 * dbeta : dbeta_max
    dmax >= dbeta || throw(ArgumentError("dbeta_max must be >= dbeta"))
    every = expand_every === nothing ? 10 * dbeta : expand_every
    step_at(b) = min(dmax, dbeta * (1 + b / ramp))

    psi = copy(psi0)
    normalize!(psi)
    loghalf = 0.0                 # log || e^{-beta H/2} |Psi0> ||
    beta = 0.0
    out = ThermalSnapshot[]

    growing = expand_cutoff !== nothing
    chi_prev = maxlinkdim(psi)
    stalls = 0
    next_rearm = 2 * every

    # Overflow guard, priced once.  The un-normalised norm over a stretch
    # `span` grows no faster than exp(span * |E_min| / 2), and `E_min` cannot
    # sit more than a few standard deviations below the beta = 0 mean.  Ten is
    # generous, and this costs one MPO application instead of one per chunk --
    # which at chi = 256 is not a rounding error.
    e0 = energy(psi, H)
    sigma0 = sqrt(max(energy_variance(psi, H), 0.0))
    safe = 60.0 / max(abs(e0) + 10 * sigma0, 1.0)

    for target in betas
        t0 = time()
        nsteps_total = 0
        while beta < target - 1.0e-14
            # Bound the stretch so the un-normalised norm stays in range:
            # it grows no faster than exp(dbeta_chunk * |E| / 2) per chunk.
            span = min(target - beta, growing ? every : chunk, safe)
            # Grade the step by where we are, holding it fixed across the
            # chunk so `tdvp` can amortise its environment build over `n`.
            n = max(1, ceil(Int, span / step_at(beta)))
            step = span / n

            if growing
                # Adds directions, not amplitude -- the state is unchanged, so
                # the norm ratio is 1 and this line is bookkeeping insurance
                # rather than physics.
                psi = expand(
                    psi, H; alg = "global_krylov",
                    krylovdim = expand_krylovdim, cutoff = expand_cutoff
                )
                loghalf += log(norm(psi))
                normalize!(psi)
            end

            psi = tdvp(
                H, -step * n / 2, psi;
                nsteps = n, nsite = nsite, maxdim = maxdim, cutoff = cutoff,
                normalize = false, outputlevel = outputlevel
            )
            nrm = norm(psi)
            isfinite(nrm) && nrm > 0 ||
                error("evolution norm went to $nrm at beta=$(beta + span); " *
                      "reduce `chunk` or `dbeta`")
            loghalf += log(nrm)
            normalize!(psi)

            if growing
                chi = maxlinkdim(psi)
                stalls = chi > chi_prev ? 0 : stalls + 1
                chi_prev = chi
                (stalls >= stall_limit || chi >= maxdim) && (growing = false)
            end

            beta += span
            nsteps_total += n

            # Re-arm the expansion at doublings of beta: chi is not monotonic,
            # so a stall here does not mean the manifold is big enough for the
            # colder rungs still to come.
            if beta >= next_rearm
                next_rearm *= 2
                if expand_cutoff !== nothing && maxlinkdim(psi) < maxdim
                    growing = true
                    stalls = 0
                end
            end
        end
        beta = max(beta, float(target))

        logZ = log(dim) + 2 * loghalf
        e = energy(psi, H)
        # S = beta*E + log Z, which is ln(dim) at beta = 0 and ln(degeneracy)
        # as beta -> infinity.  F follows from S, not the other way round.
        s = beta * e + logZ
        f = beta > 0 ? -logZ / beta : -Inf
        push!(
            out, ThermalSnapshot(
                beta > 0 ? 1 / beta : Inf, beta, copy(psi), logZ, e, f, s,
                maxlinkdim(psi), nsteps_total, time() - t0
            )
        )
        if verbose
            @printf(
                "    kT=%8.4f beta=%7.3f  chi=%5d  E=%.8f  S=%.6f  logZ=%.6f  %d steps  %.1fs\n",
                out[end].kT, beta, out[end].maxlinkdim, e, s, logZ,
                nsteps_total, out[end].seconds
            )
            flush(stdout)
        end
    end
    return out
end

"""
    thermal_ladder(case, kTs; ordering, ...) -> (layout, sites, H, snapshots)

Convenience entry point straight from a Module G run file record: build the
layout, sites, MPO and beta = 0 state, then evolve.  `kTs` are temperatures,
converted to betas and sorted descending in kT (ascending in beta) internally.

Returns everything needed to keep querying the result, because the snapshots
alone do not carry the site indices their MPSs are defined on.
"""
function thermal_ladder(
        case::MoleculeCase, kTs::AbstractVector{<:Real};
        ordering::Symbol = :blocked, conserve_sz::Bool = true,
        tol::Float64 = 1.0e-14, mpo_kwargs = (;), kwargs...
    )
    all(kTs .> 0) || throw(ArgumentError("temperatures must be positive"))
    L = PurificationLayout(case.ncas, case.nalpha, case.nbeta; ordering = ordering)
    sites = build_sites(L; conserve_sz = conserve_sz)
    H = purification_mpo(case.h1, case.g, L, sites; tol = tol, mpo_kwargs...)
    psi0 = infinite_temperature_mps(L, sites)

    betas = sort([1 / kT for kT in kTs])
    snaps = thermal_ladder(H, psi0, sector_dimension(L), betas; kwargs...)
    return L, sites, H, snaps
end

"""
    converge_dbeta(H, psi0, dim, betas; dbetas, kwargs...) -> Vector

Run the ladder at several TDVP step sizes and report the drift in `energy`
and `logZ` between successive refinements.

This is the honest convergence statement for a production run, where no exact
spectrum exists to check against.  It is deliberately not folded into
`thermal_ladder`: it multiplies the cost, and the caller should decide when to
pay it.
"""
function converge_dbeta(
        H::MPO, psi0::MPS, dim::Int, betas::AbstractVector{<:Real};
        dbetas::Vector{Float64} = [0.2, 0.1, 0.05], kwargs...
    )
    runs = [thermal_ladder(H, psi0, dim, betas; dbeta = d, kwargs...) for d in dbetas]
    rows = NamedTuple[]
    for (k, d) in enumerate(dbetas)
        for (i, b) in enumerate(betas)
            prev = k == 1 ? nothing : runs[k - 1][i]
            push!(
                rows, (
                    dbeta = d, beta = float(b),
                    energy = runs[k][i].energy, logZ = runs[k][i].logZ,
                    d_energy = prev === nothing ? NaN : runs[k][i].energy - prev.energy,
                    d_logZ = prev === nothing ? NaN : runs[k][i].logZ - prev.logZ,
                    maxlinkdim = runs[k][i].maxlinkdim,
                )
            )
        end
    end
    return rows
end
