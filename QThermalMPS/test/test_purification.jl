# The beta = 0 state is the one object in the package with a closed form:
# rho(0) = P / dim exactly.  Anything short of machine precision here is a bug,
# not an approximation, so the tolerances are tight on purpose.

@testset "infinite-temperature purification" begin
    cases = [
        (2, 1, 1, :interleaved), (3, 2, 2, :blocked), (3, 2, 1, :interleaved),
        (4, 2, 2, :blocked), (4, 2, 2, :interleaved), (4, 3, 1, :blocked),
        (2, 2, 2, :blocked),                      # full shell: dim 1, a pure state
        (3, 0, 2, :blocked),                      # empty alpha shell
    ]
    for (ncas, na, nb, ord) in cases
        L = PurificationLayout(ncas, na, nb; ordering = ord)
        sites = build_sites(L)
        psi = infinite_temperature_mps(L, sites)
        dim = sector_dimension(L)
        tag = "ncas=$ncas ($na,$nb) $ord"

        @test norm(psi) ≈ 1.0 atol = 1e-12
        @test flux(psi) == sector_flux(L)

        rho = physical_rho(psi, L)
        keep = sector_indices(L)
        @test length(keep) == dim

        # rho = P/dim: identity on the sector, zero off it.
        @test tr(rho) ≈ 1.0 atol = 1e-12
        @test isapprox(rho[keep, keep], Matrix(I, dim, dim) / dim; atol = 1e-12)
        @test norm(rho)^2 - norm(rho[keep, keep])^2 < 1e-24
        # Purity is 1/dim exactly, which is the sharpest single scalar.
        @test tr(rho * rho) ≈ 1 / dim atol = 1e-12

        # Ancilla occupations mirror the physical ones and are individually
        # conserved, so they are a running sector check later.
        ph, an = occupations(psi, L)
        for w in 1:L.nwires
            want = (L.spin_of_wire[w] == 0 ? na : nb) / ncas
            @test ph[w] ≈ want atol = 1e-12
            @test an[w] ≈ want atol = 1e-12
        end
    end
end

@testset "beta = 0 bond dimension stays polynomial" begin
    # The Schmidt vectors across any cut are labelled by the running
    # (N_left, Sz_left) alone, so chi is O(ncas^2) and nowhere near `dim`.
    for ncas in 3:6
        L = PurificationLayout(ncas, ncas ÷ 2, ncas ÷ 2; ordering = :blocked)
        psi = infinite_temperature_mps(L, build_sites(L))
        @test maxlinkdim(psi) <= (ncas + 1)^2
        @test maxlinkdim(psi) < sector_dimension(L) || ncas <= 2
    end
end
