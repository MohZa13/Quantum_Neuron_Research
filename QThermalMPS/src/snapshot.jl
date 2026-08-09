# One rung of the temperature ladder.
#
# Split into its own file because `io.jl` writes snapshots and `evolve.jl`
# reads run files: leaving the struct in `evolve.jl` makes the two includes
# circular.

using ITensorMPS
using Printf

"""
    ThermalSnapshot

`psi` is the purification at `beta`; `Tr_anc` of it is `rho`, and every scalar
here is derived from it and cached at snapshot time.

`logZ`, `free_energy` and `entropy` exclude `ecore`, matching the pipeline's
stored `TruncatedEnsemble.E`.  Add `ecore` back at the end if you want total
energies: it shifts `energy` and `free_energy` by `ecore` and leaves `entropy`
alone.

`maxlinkdim` is not decoration.  A run whose bond dimension never rises above
its beta = 0 value is the signature of the frozen-manifold failure described
in `evolve.jl`, and it is the only field that shows it.
"""
struct ThermalSnapshot
    kT::Float64
    beta::Float64
    psi::MPS
    logZ::Float64
    energy::Float64
    free_energy::Float64
    entropy::Float64
    maxlinkdim::Int
    steps::Int
    seconds::Float64
end

function Base.show(io::IO, s::ThermalSnapshot)
    return @printf(
        io,
        "ThermalSnapshot(kT=%.4g beta=%.4g E=%.8f S=%.5f chi=%d %.1fs)",
        s.kT, s.beta, s.energy, s.entropy, s.maxlinkdim, s.seconds
    )
end
