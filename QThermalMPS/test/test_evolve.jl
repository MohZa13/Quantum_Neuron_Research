# The imaginary-time ladder, against exact thermodynamics.
#
# Note what is NOT a valid check here: `logZ` and `energy` are guaranteed
# mutually consistent (`d logZ / d beta = -<H>` holds exactly even for a TDVP
# evolution constrained to the wrong manifold, because the tangent-space
# projection preserves `<Psi|H|Psi>`).  So thermodynamic self-consistency
# proves nothing about correctness, and every assertion below is against an
# EXTERNAL reference.

@testset "beta = 0 snapshot is exact" begin
    F = fixture(3, 2, 1; ordering = :blocked, seed = 41)
    s = thermal_ladder(F.H, F.psi0, F.dim, [0.0])[1]
    @test s.beta == 0.0
    @test s.logZ ≈ log(F.dim) atol = 1e-12
    @test s.entropy ≈ log(F.dim) atol = 1e-12          # ln(dim), maximally mixed
    @test s.energy ≈ sum(F.evals) / F.dim atol = 1e-9
end

@testset "ladder matches dense exp(-beta H)" begin
    betas = [0.0, 0.4, 1.5, 5.0]
    for (ncas, na, nb, ord) in [(3, 2, 1, :blocked), (3, 2, 2, :blocked),
            (4, 2, 2, :blocked)]
        F = fixture(ncas, na, nb; ordering = ord, seed = 51 + ncas)
        snaps = thermal_ladder(
            F.H, F.psi0, F.dim, betas;
            dbeta = 0.02, maxdim = 128, cutoff = 1e-14
        )
        for (i, b) in enumerate(betas)
            ref = dense_thermal(F.evals, F.evecs, b)
            s = snaps[i]
            # Tolerances sized for the TDVP step actually used (dbeta = 0.02,
            # second order).  They are three orders of magnitude tighter than
            # the O(1) errors a frozen manifold produces, which is the failure
            # this is guarding against.
            @test s.energy ≈ ref.E atol = 2e-2
            @test s.logZ ≈ ref.logZ atol = 2e-2
            @test s.entropy ≈ ref.S atol = 2e-2
            rho = physical_rho(s.psi, F.L)
            # Elementwise, not `isapprox`: the matrix form of `isapprox` tests
            # the Frobenius norm, which grows with the number of entries and so
            # silently tightens as `ncas` rises.
            @test maximum(abs.(rho[F.keep, F.keep] - ref.rho)) < 1e-3
            @test norm(rho)^2 - norm(rho[F.keep, F.keep])^2 < 1e-18
            @test tr(rho) ≈ 1.0 atol = 1e-9
        end
    end
end

@testset "subspace expansion is what makes it converge" begin
    # The regression test for the silent failure described in `evolve.jl`.
    #
    # The fixture is `:interleaved` on purpose.  Whether two-site TDVP can open
    # the bond space at all depends on the layout: `:blocked` at this size
    # happens to grow to the full sector dimension unaided, so it cannot
    # exhibit the bug, while `:interleaved` freezes at (almost) its beta = 0
    # bond dimension every time -- and the exact state there needs chi = 27
    # against a beta = 0 value of 6.
    F = fixture(3, 2, 1; ordering = :interleaved, seed = 61)
    betas = [3.0]
    ref = dense_thermal(F.evals, F.evecs, betas[1])
    chi0 = maxlinkdim(F.psi0)

    off = thermal_ladder(F.H, F.psi0, F.dim, betas;
                         dbeta = 0.02, maxdim = 64, expand_cutoff = nothing)[1]
    on = thermal_ladder(F.H, F.psi0, F.dim, betas;
                        dbeta = 0.02, maxdim = 64)[1]

    # The visible symptom: chi barely leaves where it started.
    @test off.maxlinkdim <= 2 * chi0
    @test on.maxlinkdim > off.maxlinkdim
    @test abs(on.energy - ref.E) < abs(off.energy - ref.E)

    # And the two diagnostics that DO NOT catch it, asserted as such so that a
    # future reader does not reach for them:
    #   (a) the step size is not what is wrong -- halving it changes nothing;
    off2 = thermal_ladder(F.H, F.psi0, F.dim, betas;
                          dbeta = 0.01, maxdim = 64, expand_cutoff = nothing)[1]
    @test isapprox(off.energy, off2.energy; rtol = 1e-3)
    @test abs(off.energy - ref.E) > 0.1          # still badly wrong
    #   (b) logZ and energy stay mutually consistent while both are wrong:
    #       d logZ / d beta = -<H> is preserved by the tangent-space
    #       projection, so it holds on the wrong manifold too.
    eps = 0.05
    near = thermal_ladder(F.H, F.psi0, F.dim, [betas[1] - eps, betas[1] + eps];
                          dbeta = 0.01, maxdim = 64, expand_cutoff = nothing)
    dlogZ = (near[2].logZ - near[1].logZ) / (2 * eps)
    @test isapprox(dlogZ, -off.energy; rtol = 5e-3)
end

@testset "converges in dbeta" begin
    F = fixture(3, 2, 1; ordering = :blocked, seed = 71)
    betas = [2.0]
    ref = dense_thermal(F.evals, F.evecs, betas[1])
    errs = Float64[]
    for d in (0.08, 0.04, 0.02)
        s = thermal_ladder(F.H, F.psi0, F.dim, betas; dbeta = d, maxdim = 128,
                           cutoff = 1e-14)[1]
        push!(errs, abs(s.energy - ref.E))
    end
    @test errs[2] < errs[1]
    @test errs[3] < errs[2]
    @test errs[3] < 1e-3
end

@testset "one pass gives the whole ladder" begin
    # Snapshotting mid-flight must equal stopping at that beta.
    F = fixture(3, 2, 2; ordering = :blocked, seed = 81)
    together = thermal_ladder(F.H, F.psi0, F.dim, [0.5, 1.5, 3.0]; dbeta = 0.02)
    alone = thermal_ladder(F.H, F.psi0, F.dim, [1.5]; dbeta = 0.02)[1]
    @test together[2].energy ≈ alone.energy atol = 1e-8
    @test together[2].logZ ≈ alone.logZ atol = 1e-8
end

@testset "argument validation" begin
    F = fixture(2, 1, 1; ordering = :blocked, seed = 91)
    @test_throws ArgumentError thermal_ladder(F.H, F.psi0, F.dim, [1.0, 0.5])
    @test_throws ArgumentError thermal_ladder(F.H, F.psi0, F.dim, [-1.0])
    @test_throws ArgumentError thermal_ladder(F.H, F.psi0, F.dim, [1.0]; dbeta = 0.0)
    @test isempty(thermal_ladder(F.H, F.psi0, F.dim, Float64[]))
end

@testset "real CASCI ladder vs stored evals" begin
    path = optional_run_file("h2o_cas8-6_kT0p025.h5")
    if path === nothing
        @info "skipping: results/h2o_cas8-6_kT0p025.h5 not present (gitignored)"
    else
        c = read_case(path, "mol_0")
        kTs = [1.0, 0.25]
        L, sites, H, snaps = thermal_ladder(
            c, kTs; ordering = :blocked, dbeta = 0.05, maxdim = 200, cutoff = 1e-10
        )
        @test length(snaps) == length(kTs)
        for s in snaps
            ref = exact_boltzmann(c.evals, s.beta)
            # These tolerances are set by the BUDGET the test spends, not by
            # the method's ceiling.  `maxdim = 200` sits just under the chi the
            # exact purification needs here (221 at cutoff 1e-10, measured --
            # see README), and `dbeta = 0.05` is coarse.  Both are deliberate:
            # this test exists to show the pipeline lands on the right answer
            # on real integrals, and the accuracy knobs are certified
            # separately by "converges in dbeta" against a dense reference.
            @test s.energy ≈ ref.E atol = 5e-2
            @test s.logZ ≈ ref.logZ atol = 1e-1
            @test s.entropy ≈ ref.S atol = 1e-1
            # rho_anc = rho_phys at every temperature (see `occupations`); the
            # residual is the TDVP discretisation error, here at dbeta = 0.05.
            ph, an = occupations(s.psi, L)
            @test isapprox(an, ph; atol = 2e-2)
            @test sum(ph) ≈ nelecas(L) atol = 1e-6
        end
    end
end
