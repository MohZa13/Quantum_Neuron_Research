# Reading Module-G run files (`qthermal/io_hdf5.py` layout).
#
# STORAGE ORDER.  h5py writes C order; HDF5.jl hands back Julia (column-major)
# arrays with the axes REVERSED.  A Python `g[p,q,r,s]` of shape
# (ncas,ncas,ncas,ncas) arrives as `g_raw[s,r,q,p]`.  Every read below
# permutes back, so downstream code indexes exactly like the Python side.
# `h1eff` is symmetric so the transpose is invisible there -- it is permuted
# anyway, because relying on that would be a trap for the next reader.

using HDF5

"""
    MoleculeCase

One molecule's active-space Hamiltonian plus whatever exact reference the run
file happens to carry.  `evals` is present only for dense-solver runs
(`qthermal/io_hdf5.py` contract: readers must not assume it).
"""
struct MoleculeCase
    name::String
    Z::Vector{Int}
    ecore::Float64
    h1::Matrix{Float64}
    g::Array{Float64,4}
    ncas::Int
    nalpha::Int
    nbeta::Int
    evals::Union{Nothing,Vector{Float64}}
end

nelec(c::MoleculeCase) = c.nalpha + c.nbeta

"""
    read_case(path, molname) -> MoleculeCase

Read one `mol_*` group.  `nalpha`/`nbeta` come from `/meta`'s `nelecas`
(Phase 1 is S_z = 0 by construction, so they are equal).
"""
function read_case(path::AbstractString, molname::AbstractString)
    return h5open(path, "r") do f
        meta = attrs(f["meta"])
        ncas = Int(meta["ncas"])
        ne = Int(meta["nelecas"])
        iseven(ne) || error("nelecas=$ne is odd; Phase 1 assumes S_z = 0")

        grp = f[molname]
        haskey(grp, "h1eff") || error("$molname has no h1eff")

        h1 = permutedims(read(grp["h1eff"]), (2, 1))
        g = permutedims(read(grp["g"]), (4, 3, 2, 1))
        ecore = read(grp["ecore"])
        Z = Int.(read(grp["Z"]))
        evals = haskey(grp, "evals") ? Vector{Float64}(read(grp["evals"])) : nothing

        size(h1) == (ncas, ncas) || error("h1eff shape $(size(h1)) != ($ncas,$ncas)")
        size(g) == (ncas, ncas, ncas, ncas) || error("g shape $(size(g)) wrong")

        MoleculeCase(molname, Z, Float64(ecore), h1, g, ncas, ne ÷ 2, ne ÷ 2, evals)
    end
end

"List the `mol_*` groups in a run file, in file order."
function list_molecules(path::AbstractString)
    return h5open(path, "r") do f
        sort([k for k in keys(f) if startswith(k, "mol_")],
             by = k -> parse(Int, split(k, "_")[2]))
    end
end

"Run-level `/meta` attributes as a Dict."
function read_meta(path::AbstractString)
    return h5open(path, "r") do f
        Dict{String,Any}(String(k) => v for (k, v) in pairs(attrs(f["meta"])))
    end
end

"""
    read_thermal_block(path, molname, kTtag) -> NamedTuple

The stored truncated ensemble for one temperature: `E`, `p`, and `civecs`
returned as `(dim, m)` (Python stores `(m, dim)`; the axes reverse on read).
"""
function read_thermal_block(path::AbstractString, molname::AbstractString,
                            kTtag::AbstractString)
    return h5open(path, "r") do f
        b = f[molname][kTtag]
        (kT = Float64(attrs(b)["kT"]),
         E = Vector{Float64}(read(b["E"])),
         p = Vector{Float64}(read(b["p"])),
         civecs = read(b["civecs"]),                     # already (dim, m)
         truncation_error = Float64(read(b["truncation_error"])),
         entropy = Float64(read(b["entropy"])))
    end
end

"Temperature group tags of one molecule, e.g. \"kT_0p1000\"."
function list_kT_tags(path::AbstractString, molname::AbstractString)
    return h5open(path, "r") do f
        sort([k for k in keys(f[molname]) if startswith(k, "kT_")])
    end
end

# ------------------------------------------------------------------ writing

"""
    kT_tag(kT) -> String

`"kT_0p1000"` -- the Python side's group naming (`qthermal/io_hdf5.py`), so a
ladder written here lands in groups a Module G reader already understands.
"""
kT_tag(kT::Real) = "kT_" * replace(@sprintf("%.4f", kT), "." => "p")

"""
    write_ladder(path, molname, L, snaps; rho_wires, ecore, Z, attrs, mode)

Append one molecule's ladder to an HDF5 file.

Each temperature group carries the thermodynamic scalars, the purification MPS
itself, and -- when `rho_wires` is given -- a dense reduced density matrix over
those wires, which is the form `qnn/states.py` consumes.  Pass
`rho_wires = :all` for the whole register (guarded; see `physical_rdm`).

RESUME SAFETY follows `io_hdf5.py`: `complete` is written last, and a group
that lacks it is deleted and rewritten rather than trusted.  That is the same
rule the Python pipeline uses, so a run interrupted here is recoverable the
same way.

STORAGE ORDER: h5py hands Julia matrices back transposed.  The density
matrices written here are symmetric, so this is invisible for them -- but
`wires` and other vectors are not, and are documented as written.
"""
function write_ladder(
        path::AbstractString, molname::AbstractString,
        L, snaps::Vector{ThermalSnapshot};
        rho_wires = nothing, ecore::Float64 = 0.0,
        Z::Union{Nothing, Vector{Int}} = nothing,
        attrs_extra::Dict{String, <:Any} = Dict{String, Any}(),
        mol_attrs::Dict{String, <:Any} = Dict{String, Any}(),
        maxwires::Int = 12
    )
    wires = rho_wires === :all ? collect(0:(L.nwires - 1)) :
        (rho_wires === nothing ? nothing : collect(rho_wires))

    h5open(path, isfile(path) ? "r+" : "w") do f
        if !haskey(f, "meta")
            m = create_group(f, "meta")
            attributes(m)["ncas"] = L.ncas
            attributes(m)["nelecas"] = nelecas(L)
            attributes(m)["nalpha"] = L.nalpha
            attributes(m)["nbeta"] = L.nbeta
            attributes(m)["ordering"] = L isa FusedLayout ? "blocked" :
                String(L.ordering)
            attributes(m)["backend"] = L isa FusedLayout ? "fused" : "split"
            attributes(m)["nwires"] = L.nwires
            attributes(m)["nsites"] = L.nsites
            attributes(m)["sector_dim"] = sector_dimension(L)
            attributes(m)["source"] = "QThermalMPS purification + TDVP"
            for (k, v) in attrs_extra
                attributes(m)[k] = v
            end
        end

        haskey(f, molname) && delete_object(f, molname)
        grp = create_group(f, molname)
        attributes(grp)["ecore"] = ecore
        # Per-molecule facts (validation energies, errors) go HERE, not in
        # `attrs_extra`: meta is written once per FILE, so molecule-specific
        # values passed there are silently dropped for every molecule but the
        # first -- which is how mol_3's DMRG numbers briefly ended up as
        # file-level attributes.
        for (k, v) in mol_attrs
            attributes(grp)[k] = v
        end
        Z === nothing || (grp["Z"] = Z)

        for s in snaps
            tag = kT_tag(s.kT)
            haskey(grp, tag) && delete_object(grp, tag)
            b = create_group(grp, tag)
            attributes(b)["kT"] = s.kT
            attributes(b)["beta"] = s.beta
            b["logZ"] = s.logZ
            b["energy"] = s.energy
            b["free_energy"] = s.free_energy
            b["entropy"] = s.entropy
            b["maxlinkdim"] = s.maxlinkdim
            b["steps"] = s.steps
            b["seconds"] = s.seconds
            write(b, "psi", s.psi)
            if wires !== nothing
                b["rho"] = physical_rdm(s.psi, L, wires; maxwires = maxwires)
                b["rho_wires"] = wires
            end
            attributes(b)["complete"] = true
        end
        attributes(grp)["complete"] = true
    end
    return path
end

"""
    read_ladder(path, molname) -> Vector{NamedTuple}

The scalars back out, in ascending beta.  The MPSs are left on disk: reading
them needs the site indices, which `read_purification` returns alongside.
"""
function read_ladder(path::AbstractString, molname::AbstractString)
    return h5open(path, "r") do f
        grp = f[molname]
        tags = sort([k for k in keys(grp) if startswith(k, "kT_")])
        rows = [
            (kT = Float64(attrs(grp[t])["kT"]), beta = Float64(attrs(grp[t])["beta"]),
             logZ = read(grp[t]["logZ"]), energy = read(grp[t]["energy"]),
             free_energy = read(grp[t]["free_energy"]),
             entropy = read(grp[t]["entropy"]),
             maxlinkdim = Int(read(grp[t]["maxlinkdim"])), tag = t)
                for t in tags
        ]
        sort(rows; by = r -> r.beta)
    end
end

"Read one stored purification MPS back."
function read_purification(path::AbstractString, molname::AbstractString, tag::AbstractString)
    return h5open(path, "r") do f
        read(f[molname][tag], "psi", MPS)
    end
end
