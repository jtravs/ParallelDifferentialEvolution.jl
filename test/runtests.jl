using ParallelDifferentialEvolution
using Test

@testset "ParallelDifferentialEvolution.jl" begin
    function bounds(x)
        minx = -10.0
        maxx = 10.0
        x .* (maxx .- minx) .+ minx
    end
    f(x) = map(xi -> sum(bounds(xi).^2) / length(xi), x)
    m, fitness, track = diffevo(f, 1)
    @test isapprox(bounds(m)[1], 0.0, atol=1e-14)
    @test isapprox(fitness, 0.0, atol=1e-14)
    m, fitness, track = diffevo(f, 3)
    @test all(isapprox.(bounds(m), 0.0, atol=1e-14))
    @test isapprox(fitness, 0.0, atol=1e-14)
end
