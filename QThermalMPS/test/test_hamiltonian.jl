# The Hamiltonian gate.  Everything downstream is worthless if the MPO is not
# H, so it is checked three independent ways:
#
#   * against a by-hand second-quantised matrix (pins the BASIS, not just the
#     spectrum -- see `dense_hamiltonian_reference`),
#   * `:fast` against ITensor's own compiler,
#   * the inflated purification MPO against `H (x) I` on an EVOLVED state,
#   * and, when the Python run files exist, against their stored `evals`.

@testset "MPO equals the by-hand Hamiltonian" begin
    for (ncas, na, nb, ord) in [
            (2, 1, 1, :blocked), (3, 2, 1, :interleaved),
            (3, 2, 2, :blocked), (4, 2, 2, :interleaved),
        ]
        h1, g = random_case(ncas; seed = ncas)
        L = PurificationLayout(ncas, na, nb; ordering = ord)
        sites = build_sites(L)
        psites = physical_indices(L, sites)

        Href = dense_hamiltonian_reference(h1, g, L)
        Hmpo = real(dense_mpo(physical_mpo(h1, g, L, psites), psites))
        @test isapprox(Hmpo, Href; atol = 1e-10)
        @test isapprox(Hmpo, transpose(Hmpo); atol = 1e-10)
    end
end

@testset ":fast agrees with ITensor's compiler" begin
    for (ncas, na, nb, ord) in [(3, 2, 1, :blocked), (4, 2, 2, :interleaved)]
        h1, g = random_case(ncas; seed = 11 + ncas)
        L = PurificationLayout(ncas, na, nb; ordering = ord)
        sites = build_sites(L)
        psites = physical_indices(L, sites)
        A = real(dense_mpo(physical_mpo(h1, g, L, psites; alg = :fast), psites))
        B = real(dense_mpo(physical_mpo(h1, g, L, psites; alg = :itensor), psites))
        @test isapprox(A, B; atol = 1e-10)
        # "QR" and "VC" are exact constructions of the same operator.
        C = real(dense_mpo(
            physical_mpo(h1, g, L, psites; alg = :fast, graph_alg = "QR"), psites))
        @test isapprox(A, C; atol = 1e-10)
    end
end

@testset "the purification MPO really is H (x) I" begin
    # The property `inflate_mpo` is supposed to deliver, checked where it can
    # actually fail: on an EVOLVED state, not the symmetric beta = 0 one.
    # `<Psi(0)|H|Psi(0)>` and `<Psi(0)|H^2|Psi(0)>` are both sums of squares of
    # matrix elements and so are blind to sign errors; a state at beta > 0 is
    # not.  (This is how the discarded `mode = :direct` build was caught.)
    for (ncas, na, nb, ord) in [(3, 2, 1, :blocked), (3, 1, 1, :interleaved),
            (2, 1, 1, :blocked)]
        h1, g = random_case(ncas; seed = 21 + ncas)
        L = PurificationLayout(ncas, na, nb; ordering = ord)
        sites = build_sites(L)
        psites = physical_indices(L, sites)
        H = purification_mpo(h1, g, L, sites)
        Hdense = real(dense_mpo(physical_mpo(h1, g, L, psites), psites))
        psi0 = infinite_temperature_mps(L, sites)

        for b in (0.0, 0.7, 2.0)
            psi = thermal_ladder(H, psi0, sector_dimension(L), [b];
                                 dbeta = 0.05, maxdim = 128)[1].psi
            # <Psi|H_purification|Psi> must equal Tr[rho H_physical].
            @test energy(psi, H) ≈ tr(physical_rho(psi, L) * Hdense) atol = 1e-8
        end
    end
end

@testset "H commutes with every ancilla number" begin
    # The other half of `H (x) I`, as a commutator: [H, n_{a_w}] = 0 for every
    # wire.  Stated this way rather than as "applying H leaves <n_a> alone",
    # which is FALSE -- H commuting with n_a makes n_a conserved under unitary
    # evolution, and e^{-bH/2} is not unitary.
    L = PurificationLayout(3, 2, 1; ordering = :blocked)
    sites = build_sites(L)
    h1, g = random_case(3; seed = 4)
    H = purification_mpo(h1, g, L, sites)
    psi = infinite_temperature_mps(L, sites)
    hpsi = apply(H, psi; cutoff = 1e-16, maxdim = 2000)
    @test flux(hpsi) == flux(psi)

    napply(p, j) = (q = copy(p); q[j] = noprime(op("N", siteind(q, j)) * q[j]); q)
    scale = norm(hpsi)
    for w in 1:L.nwires
        j = L.ancpos[w]
        a = apply(H, napply(psi, j); cutoff = 1e-16, maxdim = 2000)   # H n_a |psi>
        b = napply(hpsi, j)                                            # n_a H |psi>
        @test norm(a - b) < 1e-8 * max(scale, 1.0)
    end
end

@testset "beta = 0 moments equal spectral moments" begin
    # <Psi0|H^k|Psi0> = mean(evals^k) exactly, because rho(0) = P/dim.  This
    # tests the MPO and the beta = 0 state simultaneously, at any size, with
    # no dense object anywhere.
    for (ncas, na, nb, ord) in [(4, 2, 2, :blocked), (4, 2, 2, :interleaved),
            (5, 3, 2, :blocked)]
        F = fixture(ncas, na, nb; ordering = ord, seed = 31 + ncas)
        @test energy(F.psi0, F.H) ≈ sum(F.evals) / F.dim atol = 1e-9
        m2 = real(inner(F.H, F.psi0, F.H, F.psi0))
        @test m2 ≈ sum(F.evals .^ 2) / F.dim atol = 1e-8
    end
end

@testset "stored CASCI spectrum (Python run files)" begin
    path = optional_run_file("h2o_cas8-6_kT0p025.h5")
    if path === nothing
        @info "skipping: results/h2o_cas8-6_kT0p025.h5 not present (gitignored)"
    else
        c = read_case(path, "mol_0")
        for ord in (:blocked, :interleaved)
            L = PurificationLayout(c.ncas, c.nalpha, c.nbeta; ordering = ord)
            sites = build_sites(L)
            H = purification_mpo(c.h1, c.g, L, sites)
            psi0 = infinite_temperature_mps(L, sites)
            @test energy(psi0, H) ≈ sum(c.evals) / length(c.evals) atol = 1e-9
            @test real(inner(H, psi0, H, psi0)) ≈
                sum(c.evals .^ 2) / length(c.evals) atol = 1e-7
            @test length(c.evals) == sector_dimension(L)
        end
    end
end
