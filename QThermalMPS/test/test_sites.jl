# The ancilla statistics decision, tested as the concrete numerical claim it is.
#
# `sites.jl` argues that ancillas must be BOSONIC, because otherwise the
# Jordan-Wigner strings of H acquire a factor on every intervening ancilla and
# the plain qubit partial trace stops computing rho.  The cleanest witness is
# two same-spin wires at one electron with a bare hopping,
#
#     H = -t (c+_0 c_1 + c+_1 c_0),
#
# whose exact thermal state has off-diagonal element beta*t/2 + O(beta^2)
# between |10> and |01>.  With fermionic ancillas the naive trace returns
# EXACTLY ZERO there -- the two contributions cancel against the ancilla Z.
# So a nonzero, correct off-diagonal is a direct test of the convention.

@testset "ancilla statistics" begin
    t = 0.75
    ncas, nalpha, nbeta = 2, 1, 0
    h1 = [0.0 -t; -t 0.0]
    g = zeros(Float64, ncas, ncas, ncas, ncas)

    F = fixture(ncas, nalpha, nbeta; ordering = :blocked, h1 = h1, g = g)
    @test F.dim == 2

    # The two sector states: wire 0 occupied, or wire 1 occupied (wire 0 = MSB
    # of a 4-wire register, beta wires empty).
    r10 = 1 + (1 << (F.L.nwires - 1))          # 1000
    r01 = 1 + (1 << (F.L.nwires - 2))          # 0100
    @test sort(F.keep) == sort([r10, r01])

    for beta in (0.05, 0.2, 1.0)
        snaps = thermal_ladder(F.H, F.psi0, F.dim, [beta]; dbeta = 0.01, maxdim = 64, cutoff = 1e-14)
        rho = physical_rho(snaps[1].psi, F.L)
        ref = dense_thermal(F.evals, F.evecs, beta)

        off = rho[r10, r01]
        # The value fermionic ancillas would have produced.
        @test abs(off) > 1e-3
        # First-order prediction, and the exact value.
        @test isapprox(off, beta * t / 2; atol = 0.06 * beta)
        @test isapprox(rho[F.keep, F.keep], ref.rho; atol = 1e-6)
        # Nothing outside the sector.
        @test norm(rho)^2 - norm(rho[F.keep, F.keep])^2 < 1e-20
    end
end

@testset "site indices and flux" begin
    L = PurificationLayout(3, 2, 1; ordering = :blocked)
    sites = build_sites(L)
    @test length(sites) == L.nsites == 4 * 3
    # physical sites carry Nf, ancillas carry Na, and they are distinct spaces
    @test all(hastags(sites[L.physpos[w + 1]], "Fermion") for w in 0:(L.nwires - 1))
    @test all(hastags(sites[L.ancpos[w + 1]], "Anc") for w in 0:(L.nwires - 1))

    psi0 = infinite_temperature_mps(L, sites)
    @test flux(psi0) == sector_flux(L)

    # "Raise" must NOT carry a Jordan-Wigner string -- the beta = 0
    # construction is only correct because these commute across wires.
    @test ITensors.has_fermion_string("Raise", sites[L.physpos[1]]) == false
    @test ITensors.has_fermion_string("Cdag", sites[L.physpos[1]]) == true
    @test ITensors.has_fermion_string("Raise", sites[L.ancpos[1]]) == false

    # "F" on an ancilla is the identity: that IS "strings skip ancillas".
    sa = sites[L.ancpos[1]]
    @test op("F", sa) ≈ op("Id", sa)
end
