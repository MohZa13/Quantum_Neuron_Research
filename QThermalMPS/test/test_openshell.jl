# Open-shell (nalpha != nbeta, S_z != 0) sectors, end to end.
#
# The MPS layer was audited spin-general on 2026-08-11 (layouts, QN flux,
# psi0, closed form, both backends); the one blocker was `read_case`'s
# S_z = 0 file contract, now extended via `nalpha`/`nbeta` meta attributes.
# These tests pin that generality against the same two independent references
# the closed-shell suite uses: the by-hand dense Hamiltonian and dense
# exp(-beta H) in the sector.

using HDF5

const OS_SECTORS = [(3, 2, 1), (4, 3, 1), (2, 1, 0)]

@testset "layout and flux carry S_z = nalpha - nbeta" begin
    for (ncas, na, nb) in OS_SECTORS
        L = PurificationLayout(ncas, na, nb; ordering = :blocked)
        @test sector_dimension(L) == binomial(ncas, na) * binomial(ncas, nb)
        f = sector_flux(L)
        @test val(f, "Nf") == na + nb
        @test val(f, "Sz") == na - nb
        @test val(f, "Sza") == na - nb
        Lf = FusedLayout(ncas, na, nb)
        @test sector_dimension(Lf) == sector_dimension(L)
        ff = sector_flux(Lf)
        @test val(ff, "Nf") == na + nb
        @test val(ff, "Sz") == na - nb
    end
end

@testset "beta = 0 purification: flux, norm, closed-form energy" begin
    for (ncas, na, nb) in OS_SECTORS
        F = fixture(ncas, na, nb; ordering = :blocked, seed = 40 + ncas)
        @test flux(F.psi0) == sector_flux(F.L)
        @test norm(F.psi0) ≈ 1.0 atol = 1.0e-12
        e0 = energy(F.psi0, F.H)
        # sector_mean_energy's derivation never assumed na == nb; this is the
        # first place that claim is tested rather than trusted.
        @test e0 ≈ sector_mean_energy(F.h1, F.g, F.L) atol = 1.0e-10
        # ... and against the dense sector spectrum, independently.
        @test e0 ≈ sum(F.evals) / length(F.evals) atol = 1.0e-10
    end
end

@testset "open-shell ladder matches dense exp(-beta H)" begin
    F = fixture(3, 2, 1; ordering = :blocked, seed = 77)
    betas = [0.5, 2.0]
    snaps = thermal_ladder(
        F.H, F.psi0, F.dim, betas;
        dbeta = 0.0125, maxdim = 256, cutoff = 1.0e-12
    )
    for (i, b) in enumerate(betas)
        ref = dense_thermal(F.evals, F.evecs, b)
        @test snaps[i].logZ ≈ ref.logZ atol = 5.0e-3
        @test snaps[i].energy ≈ ref.E atol = 5.0e-3
        @test snaps[i].entropy ≈ ref.S atol = 5.0e-3
    end

    # rho over the whole register against the dense sector state, embedded.
    rho = physical_rho(snaps[end].psi, F.L)
    ref = dense_thermal(F.evals, F.evecs, betas[end])
    R = zeros(1 << F.L.nwires, 1 << F.L.nwires)
    R[F.keep, F.keep] = ref.rho
    @test maximum(abs.(rho - R)) < 5.0e-3
    # subsystem RDM against a brute-force partial trace of the same reference
    sub = physical_rdm(snaps[end].psi, F.L, [0, 1, 3])
    @test maximum(abs.(sub - real(partial_trace_register(R, F.L.nwires, [0, 1, 3])))) < 5.0e-3
end

@testset "zipup expansion works in an open-shell sector" begin
    F = fixture(3, 2, 1; ordering = :blocked, seed = 78)
    snaps = thermal_ladder(
        F.H, F.psi0, F.dim, [1.0];
        dbeta = 0.025, maxdim = 128, cutoff = 1.0e-12,
        expand_apply_alg = "zipup"
    )
    ref = dense_thermal(F.evals, F.evecs, 1.0)
    @test snaps[1].energy ≈ ref.E atol = 5.0e-3
end

@testset "fused backend, open-shell sector" begin
    ncas, na, nb = 3, 2, 1
    h1, g = random_case(ncas; seed = 79)
    Lref = PurificationLayout(ncas, na, nb; ordering = :blocked)
    F = fixture(ncas, na, nb; ordering = :blocked, seed = 79, h1 = h1, g = g)
    L = FusedLayout(ncas, na, nb)
    s = build_sites(L)
    H = purification_mpo(h1, g, L, s)
    p0 = infinite_temperature_mps(L, s)
    @test energy(p0, H) ≈ sector_mean_energy(h1, g, Lref) atol = 1.0e-10
    snaps = thermal_ladder(
        H, p0, sector_dimension(L), [1.0];
        dbeta = 0.0125, maxdim = 256, cutoff = 1.0e-12
    )
    ref = dense_thermal(F.evals, F.evecs, 1.0)
    @test snaps[1].energy ≈ ref.E atol = 5.0e-3
    @test snaps[1].logZ ≈ ref.logZ atol = 5.0e-3
end

@testset "read_case: nalpha/nbeta schema, legacy fallback, validation" begin
    mktempdir() do dir
        ncas = 3
        h1, g = random_case(ncas; seed = 80)
        write_stub = function (path; extra = Dict{String, Any}(), ne = 3)
            h5open(path, "w") do f
                m = create_group(f, "meta")
                attrs(m)["ncas"] = ncas
                attrs(m)["nelecas"] = ne
                for (k, v) in extra
                    attrs(m)[k] = v
                end
                grp = create_group(f, "mol_0")
                grp["h1eff"] = h1
                grp["g"] = g
                grp["ecore"] = 0.0
                grp["Z"] = [1]
            end
        end

        p1 = joinpath(dir, "open.h5")
        write_stub(p1; extra = Dict("nalpha" => 2, "nbeta" => 1), ne = 3)
        c = read_case(p1, "mol_0")
        @test (c.nalpha, c.nbeta) == (2, 1)
        # h5 round trip transposes axes; read_case permutes back.
        @test c.h1 ≈ h1
        @test c.g ≈ g

        p2 = joinpath(dir, "legacy_odd.h5")
        write_stub(p2; ne = 3)
        @test_throws ErrorException read_case(p2, "mol_0")

        p3 = joinpath(dir, "legacy_even.h5")
        write_stub(p3; ne = 4)
        c3 = read_case(p3, "mol_0")
        @test (c3.nalpha, c3.nbeta) == (2, 2)

        p4 = joinpath(dir, "inconsistent.h5")
        write_stub(p4; extra = Dict("nalpha" => 2, "nbeta" => 2), ne = 3)
        @test_throws ErrorException read_case(p4, "mol_0")
    end
end
