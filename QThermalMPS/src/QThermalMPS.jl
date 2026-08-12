"""
    QThermalMPS

Thermal states of CASCI active spaces as purification matrix product states,
produced by imaginary-time evolution rather than diagonalization.

The pipeline this replaces (`qthermal/diagonalize.py` -> `thermal.py` ->
`mps.py`) forms the thermal state from an explicit eigenbasis, which caps out
where the sector dimension does: CAS(8,8) is dim 4900, CAS(8,12) is 245,025 --
already past the dense guardrail -- and Module J's `2^Q` scatter dies at
ncas = 8.  Here nothing is ever diagonalized and no `2^Q` object is formed:

    |Psi(0)>  = sector-projected maximally entangled physical-ancilla state
    |Psi(b)>  = exp(-b*H/2) |Psi(0)> / norm,   Tr_anc |Psi><Psi| = rho(b)

with `exp(-b*H/2)` applied by TDVP against the Hamiltonian MPO built straight
from the stored `(h1eff, g)` tensors under a canonical Jordan-Wigner mapping.

Because imaginary-time evolution sweeps *through* every intermediate beta, one
run yields the entire kT ladder -- the Python pipeline re-truncates a spectrum
per temperature; here each temperature is a snapshot.

Entry points: [`thermal_state`](@ref), [`thermal_ladder`](@ref).
"""
module QThermalMPS

using ITensors
using ITensorMPS
using HDF5
using LinearAlgebra
using Printf
using Random
using Statistics
using TimerOutputs

"""
    TIMER

Package-wide `TimerOutputs.TimerOutput` behind every `@timeit` section in the
pipeline: setup (opsum, MPO compile, inflate, beta = 0 state), the ladder
(per-rung `expand` / `tdvp` / renormalisation / snapshot energies), and I/O
(`read_case`, MPS writes, RDM export).  Always on -- the per-section overhead
is ~1 microsecond against stages that run for seconds -- so any run can be
asked where its time went after the fact:

    using TimerOutputs
    print_timer(QThermalMPS.TIMER; sortby = :firstexec)
    reset_timer!(QThermalMPS.TIMER)          # per-molecule accounting

`bin/thermal.jl --profile 1` prints exactly this per molecule.
"""
const TIMER = TimerOutput()

include("layout.jl")
include("sites.jl")
include("purification.jl")
include("hamiltonian.jl")
include("snapshot.jl")
include("io.jl")
include("observables.jl")
include("evolve.jl")
include("fused.jl")

# layout
export PurificationLayout, jw_wire, physsite, nelecas, isphysical
# sites
export build_sites, physical_index, ancilla_index, sector_flux, szval
# purification
export infinite_temperature_mps, pair_raise_opsum, sector_dimension, inflate_mpo
# hamiltonian
export qc_opsum, build_mpo, physical_mpo, purification_mpo,
       physical_sites, physical_indices, physical_flux,
       physical_wirepos, purification_wirepos
# io
export MoleculeCase, read_case, list_molecules, read_meta,
       read_thermal_block, list_kT_tags, nelec,
       kT_tag, write_ladder, read_ladder, read_purification
# observables
export energy, energy_variance, occupations, dense_purification, dense_mpo,
       physical_rdm, physical_rho, pauli_expect, sector_mean_energy
# evolve
export ThermalSnapshot, thermal_ladder, converge_dbeta
# fused backend
export FusedLayout
export enable_threading!

"""
    enable_threading!(; blocksparse=true, blas=nothing)

Turn on ITensor's threaded block-sparse contractions, which is where the time
goes once the bond dimension is large and the tensors are U(1)xU(1) blocked.
Must be called at runtime (it is a global switch, so the package does not flip
it on import), and only helps if Julia itself was started with threads:

    julia -t 8 --project=. ...

BLAS and Strided are serialised alongside it, which is ITensor's own
recommendation and not a detail: the QN blocks here are small and numerous, so
threading at the block level and again inside each block's BLAS call
oversubscribes the cores and loses to either one alone.  Pass `blas` to
override.
"""
function enable_threading!(; blocksparse::Bool = true, blas::Union{Nothing, Int} = nothing)
    if blocksparse
        ITensors.enable_threaded_blocksparse()
        ITensors.NDTensors.Strided.disable_threads()
        LinearAlgebra.BLAS.set_num_threads(blas === nothing ? 1 : blas)
    else
        ITensors.disable_threaded_blocksparse()
        blas === nothing || LinearAlgebra.BLAS.set_num_threads(blas)
    end
    return (threads = Threads.nthreads(),
            blocksparse = blocksparse,
            blas = LinearAlgebra.BLAS.get_num_threads())
end

end # module
