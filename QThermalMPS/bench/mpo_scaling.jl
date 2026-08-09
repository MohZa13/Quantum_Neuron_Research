# Where does MPO construction time go, and how does it scale?
# The generic OpSum -> MPO compiler is the suspected bottleneck; the two
# levers are the integral screening tolerance and the term count itself.
using QThermalMPS
using ITensors, ITensorMPS
using Printf

const RESULTS = joinpath(@__DIR__, "..", "..", "results")

function bench(case_path, mol, tols)
    c = read_case(case_path, mol)
    L = PurificationLayout(c.ncas, c.nalpha, c.nbeta; ordering=:interleaved)
    sites = physical_sites(L)
    @printf("\n=== %s %s  ncas=%d  (%d physical sites) ===\n",
            basename(case_path), mol, c.ncas, length(sites))
    @printf("%-10s %10s %10s %10s %10s\n", "tol", "terms", "opsum_s", "mpo_s", "maxdim")
    for tol in tols
        t0 = time()
        os = qc_opsum(c.h1, c.g, c.ncas, L.ordering, physical_wirepos(L); tol=tol)
        t1 = time()
        H = MPO(os, sites; cutoff=1e-16)
        t2 = time()
        @printf("%-10.0e %10d %10.2f %10.2f %10d\n",
                tol, length(os), t1 - t0, t2 - t1, maxlinkdim(H))
        flush(stdout)
    end
end

bench(joinpath(RESULTS, "h2o_cas8-6_kT0p025.h5"), "mol_0", [1e-14, 1e-10, 1e-8, 1e-6])
bench(joinpath(RESULTS, "qh9_dense_cas88_5mols.h5"), "mol_0", [1e-14, 1e-10, 1e-8, 1e-6])
