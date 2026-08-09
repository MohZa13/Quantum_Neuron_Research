# The read-out layer.  These are the functions the neuron side actually calls,
# so they are checked against dense linear algebra on the same states rather
# than against each other.

@testset "physical_rdm is a partial trace of physical_rho" begin
    F = fixture(3, 2, 1; ordering = :blocked, seed = 5)
    snaps = thermal_ladder(F.H, F.psi0, F.dim, [0.0, 0.8]; dbeta = 0.02, cutoff = 1e-14)
    Q = F.L.nwires

    for s in snaps
        full = physical_rho(s.psi, F.L)
        @test tr(full) ≈ 1.0 atol = 1e-9
        @test isapprox(full, transpose(full); atol = 1e-9)

        for wires in ([0], [1], [0, 1], [0, 2], [1, 2, 3], collect(0:(Q - 2)))
            sub = physical_rdm(s.psi, F.L, wires)
            @test size(sub) == (1 << length(wires), 1 << length(wires))
            @test tr(sub) ≈ 1.0 atol = 1e-9
            # Trace `full` down to the same wires and compare.
            want = partial_trace_register(full, Q, wires)
            @test isapprox(sub, want; atol = 1e-9)
        end
    end
end

@testset "pauli_expect equals Tr[rho P]" begin
    for ord in (:blocked, :interleaved)
        F = fixture(3, 2, 2; ordering = ord, seed = 9)
        snaps = thermal_ladder(F.H, F.psi0, F.dim, [0.0, 0.6]; dbeta = 0.02, cutoff = 1e-14)
        Q = F.L.nwires
        for s in snaps
            rho = physical_rho(s.psi, F.L)
            rng = MersenneTwister(3)
            for _ in 1:24
                ps = String(rand(rng, ['I', 'I', 'I', 'Z', 'X', 'Y'], Q))
                got = pauli_expect(s.psi, F.L, ps)
                want = tr(pauli_dense(ps) * rho)
                @test isapprox(got, want; atol = 1e-8)
            end
            # A string chosen to be nonzero rather than random, so the test
            # cannot pass by comparing zero to zero.  It has to sit on two
            # wires of the SAME spin: X_p X_q between an alpha and a beta wire
            # shifts S_z by two and is exactly zero by symmetry, as is any
            # off-diagonal string at beta = 0 where rho is proportional to the
            # sector projector.
            @test isapprox(pauli_expect(s.psi, F.L, "Z" * "I"^(Q - 1)),
                           tr(pauli_dense("Z" * "I"^(Q - 1)) * rho); atol = 1e-8)
            same = findall(==(0), F.L.spin_of_wire)[1:2]
            chars = fill('I', Q)
            chars[same] .= 'X'
            xx = String(chars)
            s.beta > 0 && @test abs(tr(pauli_dense(xx) * rho)) > 1e-6
            @test isapprox(pauli_expect(s.psi, F.L, xx),
                           tr(pauli_dense(xx) * rho); atol = 1e-8)
        end
    end
end

@testset "ancilla marginal mirrors the physical one" begin
    # rho_anc = rho_phys at every beta (see `occupations`).  Both halves of the
    # chain and the sign convention of the beta = 0 state have to be right for
    # this to hold, so it is a sharper running check than any single number.
    # The identity is exact for the exact state, so at beta = 0 it holds to
    # machine precision.  At beta > 0 the residual is a genuine measure of the
    # TDVP discretisation error -- the integrator does not treat the two halves
    # of the chain identically -- and tracks dbeta, not a tolerance we chose.
    F = fixture(3, 2, 1; ordering = :blocked, seed = 13)
    snaps = thermal_ladder(F.H, F.psi0, F.dim, [0.0, 0.5, 2.0]; dbeta = 0.02)
    for s in snaps
        ph, an = occupations(s.psi, F.L)
        @test sum(ph) ≈ nelecas(F.L) atol = 1e-8
        @test sum(an) ≈ nelecas(F.L) atol = 1e-8
        @test isapprox(an, ph; atol = s.beta == 0 ? 1e-10 : 5e-3)
    end
    # At beta = 0 -- and only there -- both sit at the shell filling.
    ph0, an0 = occupations(snaps[1].psi, F.L)
    for w in 1:F.L.nwires
        want = (F.L.spin_of_wire[w] == 0 ? F.L.nalpha : F.L.nbeta) / F.L.ncas
        @test ph0[w] ≈ want atol = 1e-8
        @test an0[w] ≈ want atol = 1e-8
    end
end

@testset "register index convention (wire 0 = MSB)" begin
    # A state with wire 0 certainly occupied and the rest empty must put its
    # weight at the row with the top bit set.
    L = PurificationLayout(2, 1, 0; ordering = :blocked)
    sites = build_sites(L)
    st = fill("Emp", L.nsites)
    st[L.physpos[1]] = "Occ"                 # wire 0
    st[L.ancpos[1]] = "Occ"
    psi = MPS(sites, st)
    rho = physical_rho(psi, L)
    @test rho[1 + (1 << (L.nwires - 1)), 1 + (1 << (L.nwires - 1))] ≈ 1.0 atol = 1e-12
    @test tr(rho) ≈ 1.0 atol = 1e-12
end
