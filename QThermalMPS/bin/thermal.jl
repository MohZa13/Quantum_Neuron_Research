#!/usr/bin/env julia
#
# Module K CLI: QH9 active-space integrals -> thermal purification MPS ladder.
#
# Reads a Module G run file (the `(ecore, h1eff, g)` a `qthermal.run` produced),
# builds the Jordan-Wigner MPO, purifies the sector at beta = 0, and evolves in
# imaginary time, writing one snapshot per requested temperature.
#
#   julia -t 8 --project=QThermalMPS QThermalMPS/bin/thermal.jl \
#       --in results/qh9_dense_cas88_5mols.h5 \
#       --out results/qh9_cas88_mps.h5 \
#       --kT 1.0,0.5,0.25,0.1 --maxdim 300 --rho-wires 0:9
#
# `-t 8` matters: the block-sparse contractions are threaded and this is where
# nearly all the time goes.  Nothing else here is parallel, so there is no
# point going past the core count.
#
# Exit status is nonzero if any molecule failed; the others are still written,
# because a 5-molecule run should not lose four results to one bad record.
#
# `--profile 1` prints a per-molecule wall-time breakdown of every pipeline
# stage (setup, per-rung expand/tdvp, RDM export, writes) from
# `QThermalMPS.TIMER`, plus per-chunk chi/time trace lines.  `--warmup`
# (default: follows --profile) first runs a tiny synthetic ladder so Julia's
# JIT compilation is not billed to the first molecule's timings.
#
# Performance flags (defaults are the measured-safe choices, RESEARCH_LOG
# 2026-08-10):
#   --backend fused        half-length dim-4 chain.  EXPERIMENTAL past
#                          ncas = 6: at ncas = 10 it is no faster and its
#                          warm rungs misconverge (trap-2 signature) -- see
#                          RESEARCH_LOG 2026-08-10 before using it there
#   --solver-tol X|none    local exponentiate tolerance; empty = cutoff/10
#                          with Lanczos + eager exit; "none" = KrylovKit stock
#   --nsite-capped 1       one-site TDVP for chunks starting at the chi cap;
#                          opt-in, certify against a two-site run first

t0_load = time()
using QThermalMPS
using ITensors, ITensorMPS
using Printf
using TimerOutputs
const T_LOAD = time() - t0_load

function parse_args(argv)
    opts = Dict{String, String}(
        "in" => "", "out" => "", "kT" => "1.0,0.5,0.25,0.1",
        "ordering" => "blocked", "dbeta" => "0.05", "dbeta-max" => "",
        "ramp" => "1.0", "maxdim" => "256", "cutoff" => "1e-9",
        "limit" => "0", "mols" => "", "rho-wires" => "",
        "expand-cutoff" => "1e-9", "tol" => "1e-14", "threads" => "1",
        "profile" => "0", "warmup" => "",
        "backend" => "split", "solver-tol" => "", "nsite-capped" => "",
        "expand-apply-alg" => "",
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        startswith(a, "--") || error("unexpected argument $a")
        key = a[3:end]
        haskey(opts, key) || error("unknown option --$key")
        i + 1 <= length(argv) || error("--$key needs a value")
        opts[key] = argv[i + 1]
        i += 2
    end
    isempty(opts["in"]) && error("--in is required")
    isempty(opts["out"]) && error("--out is required")
    return opts
end

"""
    run_warmup() -> Float64

Absorb JIT compilation before anything is timed: a tiny synthetic CAS(2,2)
case pushed through the same code path as a real molecule -- layout, MPO
compile, beta = 0 state, expand + tdvp ladder, RDM export, HDF5 write.  Costs
seconds; without it the first molecule's profile is dominated by compilation
and cannot be compared with the second's.
"""
function run_warmup(backend::String)
    t0 = time()
    ncas = 2
    h1 = [-1.0 0.15; 0.15 -0.6]
    g0 = [0.05 * (p + 2q + 3r + 4s) / 10
          for p in 1:ncas, q in 1:ncas, r in 1:ncas, s in 1:ncas]
    # 8-fold chemist symmetrisation keeps H Hermitian; the values are arbitrary.
    g = [(g0[p, q, r, s] + g0[q, p, r, s] + g0[p, q, s, r] + g0[q, p, s, r] +
          g0[r, s, p, q] + g0[s, r, p, q] + g0[r, s, q, p] + g0[s, r, q, p]) / 8
         for p in 1:ncas, q in 1:ncas, r in 1:ncas, s in 1:ncas]
    c = MoleculeCase("warmup", Int[], 0.0, h1, g, ncas, 1, 1, nothing)
    L, _, _, snaps = backend == "fused" ?
        thermal_ladder(c, [4.0, 1.0], Val(:fused); dbeta = 0.25, maxdim = 32) :
        thermal_ladder(c, [4.0, 1.0]; dbeta = 0.25, maxdim = 32)
    tmp = tempname() * ".h5"
    try
        write_ladder(tmp, "warmup", L, snaps; rho_wires = 0:3)
    finally
        isfile(tmp) && rm(tmp; force = true)
    end
    return time() - t0
end

"Parse `0:9` or `0,2,4` into wire numbers; empty means no dense rho."
function parse_wires(s)
    isempty(s) && return nothing
    s == "all" && return :all
    if occursin(':', s)
        a, b = split(s, ':')
        return collect(parse(Int, a):parse(Int, b))
    end
    return [parse(Int, x) for x in split(s, ',')]
end

function main(argv)
    opts = parse_args(argv)
    kTs = [parse(Float64, x) for x in split(opts["kT"], ',')]
    all(kTs .> 0) || error("temperatures must be positive")
    ordering = Symbol(opts["ordering"])
    dbeta = parse(Float64, opts["dbeta"])
    dbeta_max = isempty(opts["dbeta-max"]) ? nothing : parse(Float64, opts["dbeta-max"])
    ramp = parse(Float64, opts["ramp"])
    maxdim = parse(Int, opts["maxdim"])
    cutoff = parse(Float64, opts["cutoff"])
    tol = parse(Float64, opts["tol"])
    expcut = opts["expand-cutoff"] == "none" ? nothing :
        parse(Float64, opts["expand-cutoff"])
    wires = parse_wires(opts["rho-wires"])
    profile = opts["profile"] in ("1", "true")
    warmup = isempty(opts["warmup"]) ? profile : opts["warmup"] in ("1", "true")
    backend = opts["backend"]
    backend in ("split", "fused") || error("--backend must be split or fused")
    # --solver-tol: empty = package default (Lanczos + eager at cutoff/10);
    # "none" = KrylovKit stock behavior; a number = that tolerance.
    upk = isempty(opts["solver-tol"]) ? nothing :
        opts["solver-tol"] == "none" ? (;) :
        (; issymmetric = true, eager = true,
           tol = parse(Float64, opts["solver-tol"]))
    nscap = isempty(opts["nsite-capped"]) ? nothing : parse(Int, opts["nsite-capped"])

    println(enable_threading!())
    profile && @printf("load    %.1fs (package import)\n", T_LOAD)
    if warmup
        tw = run_warmup(backend)
        @printf("warmup  %.1fs (JIT compilation)\n", tw)
        flush(stdout)
    end
    meta = read_meta(opts["in"])
    mols = isempty(opts["mols"]) ? list_molecules(opts["in"]) :
        String.(split(opts["mols"], ','))
    lim = parse(Int, opts["limit"])
    lim > 0 && (mols = mols[1:min(lim, end)])

    @printf("input   %s\n", opts["in"])
    @printf("meta    ncas=%s nelecas=%s  molecules=%d\n",
            meta["ncas"], meta["nelecas"], length(mols))
    @printf("ladder  kT=%s  backend=%s ordering=%s dbeta=%.4g maxdim=%d cutoff=%.1e\n",
            opts["kT"], backend, ordering, dbeta, maxdim, cutoff)

    failures = String[]
    for (k, mol) in enumerate(mols)
        try
            profile && reset_timer!(QThermalMPS.TIMER)
            c = read_case(opts["in"], mol)
            t0 = time()
            common = (; tol = tol, dbeta = dbeta, dbeta_max = dbeta_max,
                      ramp = ramp, maxdim = maxdim, cutoff = cutoff,
                      expand_cutoff = expcut, verbose = profile)
            isempty(opts["expand-apply-alg"]) ||
                (common = (; common..., expand_apply_alg = opts["expand-apply-alg"]))
            upk === nothing || (common = (; common..., updater_kwargs = upk))
            nscap === nothing || (common = (; common..., nsite_capped = nscap))
            L, sites, H, snaps = backend == "fused" ?
                thermal_ladder(c, kTs, Val(:fused); common...) :
                thermal_ladder(c, kTs; ordering = ordering, common...)
            write_ladder(
                opts["out"], mol, L, snaps;
                rho_wires = wires, ecore = c.ecore, Z = c.Z,
                attrs_extra = Dict{String, Any}(
                    "dbeta" => dbeta, "maxdim" => maxdim, "cutoff" => cutoff,
                    "source_file" => basename(opts["in"])
                )
            )
            chis = join([string(s.maxlinkdim) for s in snaps], ",")
            @printf("[%3d/%3d] %-10s ncas=%d  chi=%s  %.1fs\n",
                    k, length(mols), mol, c.ncas, chis, time() - t0)
            # In execution order, so the table reads as the pipeline runs.
            profile && (print_timer(QThermalMPS.TIMER; sortby = :firstexec); println())
            # The frozen-manifold signature: chi never left its beta = 0 value.
            # AFTER the timer print: this check rebuilds the beta = 0 state,
            # which would otherwise double-count "setup: psi0" in the profile.
            if maximum(s.maxlinkdim for s in snaps) <= maxlinkdim(
                    infinite_temperature_mps(L, sites))
                @warn "bond dimension never grew for $mol -- suspect a frozen " *
                    "TDVP manifold; see thermal_ladder's docstring" mol
            end
        catch err
            push!(failures, mol)
            @error "failed" mol exception = (err, catch_backtrace())
        end
        flush(stdout)
    end

    if !isempty(failures)
        @printf("\n%d/%d molecules FAILED: %s\n",
                length(failures), length(mols), join(failures, ","))
        exit(1)
    end
    @printf("\nwrote %s\n", opts["out"])
    return 0
end

(abspath(PROGRAM_FILE) == @__FILE__) && main(ARGS)
