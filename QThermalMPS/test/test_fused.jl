# The fused wire+ancilla backend, held to the same gates as the split one --
# and additionally to AGREEMENT with it, state by state, in the register basis.

@testset "fused MPO equals H (x) I element-by-element" begin
    # The compiler gate.  MPO_new mis-signs hopping terms on this site type
    # (an extra (-1)^{n_anc} on the operator site); ITensor's compiler is
    # exact, and this test pins that choice.  Deliberately elementwise over
    # the full 4^N x 4^N matrix: the mis-sign is a sign-gauge conjugation and
    # every trace-like or self-consistency quantity is blind to it.
    ncas, na, nb = 2, 1, 1
    h1, g = random_case(ncas; seed = 5)
    Lref = PurificationLayout(ncas, na, nb; ordering = :blocked)
    psites = physical_sites(Lref)
    Hd = real(dense_mpo(physical_mpo(h1, g, Lref, psites), psites))

    L = FusedLayout(ncas, na, nb)
    sites = build_sites(L)
    H = purification_mpo(h1, g, L, sites)            # default compiler
    N = length(sites)
    T = H[1]
    for j in 2:N
        T *= H[j]
    end
    A = array(T, reverse(prime.(sites))..., reverse(sites)...)
    Hf = reshape(A, 4^N, 4^N)

    function split_idx(k)
        d = digits(k - 1; base = 4, pad = N)
        rp = 0
        ra = 0
        for site in 1:N
            v = d[N - site + 1]
            rp |= (v >= 2 ? 1 : 0) << (N - site)
            ra |= (v == 1 || v == 3 ? 1 : 0) << (N - site)
        end
        return rp, ra
    end
    worst = 0.0
    for kb in 1:4^N, ka in 1:4^N
        rpa, raa = split_idx(ka)
        rpb, rab = split_idx(kb)
        want = raa == rab ? Hd[rpa + 1, rpb + 1] : 0.0
        worst = max(worst, abs(Hf[ka, kb] - want))
    end
    @test worst < 1e-10
end

@testset "fused beta = 0 purification is exact" begin
    for (ncas, na, nb) in [(2, 1, 1), (3, 2, 1), (4, 2, 2)]
        L = FusedLayout(ncas, na, nb)
        sites = build_sites(L)
        psi = infinite_temperature_mps(L, sites)
        dim = sector_dimension(L)

        @test norm(psi) ≈ 1.0 atol = 1e-12
        @test flux(psi) == sector_flux(L)

        rho = physical_rho(psi, L)
        Lref = PurificationLayout(ncas, na, nb; ordering = :blocked)
        keep = sector_indices(Lref)
        @test length(keep) == dim
        @test tr(rho) ≈ 1.0 atol = 1e-12
        @test isapprox(rho[keep, keep], Matrix(I, dim, dim) / dim; atol = 1e-12)
        @test norm(rho)^2 - norm(rho[keep, keep])^2 < 1e-24
    end
end

@testset "fused MPO: beta = 0 moments equal spectral moments" begin
    for (ncas, na, nb) in [(3, 2, 1), (4, 2, 2)]
        h1, g = random_case(ncas; seed = 101 + ncas)
        L = FusedLayout(ncas, na, nb)
        sites = build_sites(L)
        H = purification_mpo(h1, g, L, sites)
        psi0 = infinite_temperature_mps(L, sites)
        dim = sector_dimension(L)

        Lref = PurificationLayout(ncas, na, nb; ordering = :blocked)
        F = fixture(ncas, na, nb; ordering = :blocked, seed = 101 + ncas,
                    h1 = h1, g = g)
        @test energy(psi0, H) ≈ sum(F.evals) / dim atol = 1e-9
        @test real(inner(H, psi0, H, psi0)) ≈ sum(F.evals .^ 2) / dim atol = 1e-8
    end
end

@testset "fused ladder matches dense exp(-beta H) in the register basis" begin
    # The decisive test: evolve on the fused chain, export the register RDM,
    # compare against dense thermodynamics AND against the split backend's
    # export.  Sign or basis errors in the fused unfuse/trace cannot pass.
    betas = [0.0, 0.4, 1.5]
    for (ncas, na, nb) in [(3, 2, 1), (3, 2, 2)]
        h1, g = random_case(ncas; seed = 111 + ncas)
        F = fixture(ncas, na, nb; ordering = :blocked, seed = 111 + ncas,
                    h1 = h1, g = g)
        L = FusedLayout(ncas, na, nb)
        sites = build_sites(L)
        H = purification_mpo(h1, g, L, sites)
        psi0 = infinite_temperature_mps(L, sites)
        dim = sector_dimension(L)

        snaps = thermal_ladder(H, psi0, dim, betas;
                               dbeta = 0.02, maxdim = 128, cutoff = 1e-14)
        snaps_split = thermal_ladder(F.H, F.psi0, F.dim, betas;
                                     dbeta = 0.02, maxdim = 128, cutoff = 1e-14)
        for (i, b) in enumerate(betas)
            ref = dense_thermal(F.evals, F.evecs, b)
            s = snaps[i]
            @test s.energy ≈ ref.E atol = 2e-2
            @test s.logZ ≈ ref.logZ atol = 2e-2
            rho = physical_rho(s.psi, L)
            @test tr(rho) ≈ 1.0 atol = 1e-9
            @test maximum(abs.(rho[F.keep, F.keep] - ref.rho)) < 1e-3
            # backend agreement: the two chains take DIFFERENT two-site
            # TDVP projections of the same trajectory, so they agree at the
            # integrator-error scale (~1e-4 at this dbeta), not machine
            # precision -- the machine-precision statement is the dense-H
            # equality testset above.
            rho_split = physical_rho(snaps_split[i].psi, F.L)
            @test maximum(abs.(rho - rho_split)) < 2e-3
        end

        # subsets of wires agree between backends too (the qnn export path)
        s = snaps[end]; ss = snaps_split[end]
        for wires in ([0], [0, 1], collect(0:(ncas - 1)))
            @test maximum(abs.(physical_rdm(s.psi, L, wires) -
                               physical_rdm(ss.psi, F.L, wires))) < 2e-3
        end
    end
end

@testset "fused occupations and ancilla mirror" begin
    F = fixture(3, 2, 1; ordering = :blocked, seed = 121)
    L = FusedLayout(3, 2, 1)
    sites = build_sites(L)
    H = purification_mpo(F.h1, F.g, L, sites)
    psi0 = infinite_temperature_mps(L, sites)
    snaps = thermal_ladder(H, psi0, sector_dimension(L), [0.0, 1.0]; dbeta = 0.02)
    for s in snaps
        ph, an = occupations(s.psi, L)
        @test sum(ph) ≈ nelecas(L) atol = 1e-8
        @test sum(an) ≈ nelecas(L) atol = 1e-8
        @test isapprox(an, ph; atol = s.beta == 0 ? 1e-10 : 5e-3)
    end
end

@testset "fused chain is half the split chain" begin
    L = FusedLayout(5, 3, 2)
    @test L.nsites == 10
    @test PurificationLayout(5, 3, 2).nsites == 20
    @test sector_dimension(L) == binomial(5, 3) * binomial(5, 2)
end
