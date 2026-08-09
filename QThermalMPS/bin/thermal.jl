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

using QThermalMPS
using ITensors, ITensorMPS
using Printf

function parse_args(argv)
    opts = Dict{String, String}(
        "in" => "", "out" => "", "kT" => "1.0,0.5,0.25,0.1",
        "ordering" => "blocked", "dbeta" => "0.05", "dbeta-max" => "",
        "ramp" => "1.0", "maxdim" => "256", "cutoff" => "1e-9",
        "limit" => "0", "mols" => "", "rho-wires" => "",
        "expand-cutoff" => "1e-9", "tol" => "1e-14", "threads" => "1",
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

    println(enable_threading!())
    meta = read_meta(opts["in"])
    mols = isempty(opts["mols"]) ? list_molecules(opts["in"]) :
        String.(split(opts["mols"], ','))
    lim = parse(Int, opts["limit"])
    lim > 0 && (mols = mols[1:min(lim, end)])

    @printf("input   %s\n", opts["in"])
    @printf("meta    ncas=%s nelecas=%s  molecules=%d\n",
            meta["ncas"], meta["nelecas"], length(mols))
    @printf("ladder  kT=%s  ordering=%s dbeta=%.4g maxdim=%d cutoff=%.1e\n",
            opts["kT"], ordering, dbeta, maxdim, cutoff)

    failures = String[]
    for (k, mol) in enumerate(mols)
        try
            c = read_case(opts["in"], mol)
            t0 = time()
            L, sites, H, snaps = thermal_ladder(
                c, kTs; ordering = ordering, tol = tol,
                dbeta = dbeta, dbeta_max = dbeta_max, ramp = ramp,
                maxdim = maxdim, cutoff = cutoff, expand_cutoff = expcut
            )
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
            # The frozen-manifold signature: chi never left its beta = 0 value.
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

abspath(PROGRAM_FILE) == @__FILE__ && main(ARGS)
