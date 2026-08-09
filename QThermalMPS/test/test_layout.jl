# Wire conventions.  These are a CONTRACT with `qthermal/encode.py`, not a
# local design choice, so they are pinned by literal expected values rather
# than by re-deriving them from the same formula the code uses.

@testset "jw_wire matches the Python encoder" begin
    # blocked: p + spin*ncas ; interleaved: 2p + spin
    @test [jw_wire(p, s, 4, :blocked) for s in 0:1, p in 0:3] ==
        [0 1 2 3; 4 5 6 7]
    @test [jw_wire(p, s, 4, :interleaved) for s in 0:1, p in 0:3] ==
        [0 2 4 6; 1 3 5 7]
    @test_throws ArgumentError jw_wire(0, 0, 4, :spiral)
end

@testset "layout geometry" begin
    for ord in (:blocked, :interleaved), ncas in 2:5
        L = PurificationLayout(ncas, ncas - 1, 1; ordering = ord)
        @test L.nwires == 2ncas
        @test L.nsites == 4ncas
        @test nelecas(L) == ncas
        # every wire appears exactly once, physical and ancilla positions
        # partition the chain, and each ancilla sits immediately right of its
        # own physical site (that adjacency is what makes beta = 0 cheap).
        @test sort(vcat(L.physpos, L.ancpos)) == collect(1:L.nsites)
        @test all(L.ancpos[w] == L.physpos[w] + 1 for w in 1:L.nwires)
        @test all(isphysical(L, j) == (j in L.physpos) for j in 1:L.nsites)
        # spin_of_wire / orb_of_wire invert jw_wire
        for p in 0:(ncas - 1), s in 0:1
            w = jw_wire(p, s, ncas, ord)
            @test L.spin_of_wire[w + 1] == s
            @test L.orb_of_wire[w + 1] == p
            @test physsite(L, p, s) == L.physpos[w + 1]
        end
        @test count(==(0), L.spin_of_wire) == ncas
    end
end

@testset "layout validation" begin
    @test_throws ArgumentError PurificationLayout(0, 0, 0)
    @test_throws ArgumentError PurificationLayout(3, 4, 1)
    @test_throws ArgumentError PurificationLayout(3, 1, -1)
end

@testset "sector_dimension" begin
    @test sector_dimension(PurificationLayout(6, 4, 4)) == binomial(6, 4)^2
    @test sector_dimension(PurificationLayout(8, 4, 4)) == 4900
    @test sector_dimension(PurificationLayout(6, 3, 3)) == 400
end
