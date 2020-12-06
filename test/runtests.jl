using ParallelDifferentialEvolution
using Test
import Distributed: pmap

@testset "ParallelDifferentialEvolution.jl" begin
    function bounds(x)
        minx = -10.0
        maxx = 10.0
        x .* (maxx .- minx) .+ minx
    end
    f(x) = sum(x.^2) / length(x)
    m, fitness = diffevo(f, bounds, 1)
    @test isapprox(bounds(m)[1], 0.0, atol=1e-6)
    @test isapprox(fitness, 0.0, atol=1e-13)
    m, fitness = diffevo(f, bounds, 1, cb=logcb)
    @test isapprox(bounds(m)[1], 0.0, atol=1e-6)
    @test isapprox(fitness, 0.0, atol=1e-13)
    m, fitness = diffevo(f, bounds, 3)
    @test all(isapprox.(bounds(m), 0.0, atol=1e-6))
    @test isapprox(fitness, 0.0, atol=1e-13)
    m, fitness = diffevo(f, bounds, 10, maxiter=2000)
    @test all(isapprox.(bounds(m), 0.0, atol=1e-6))
    @test isapprox(fitness, 0.0, atol=1e-13)
    m, fitness = diffevo(f, bounds, 10, maxiter=2000, fmap=pmap)
    @test all(isapprox.(bounds(m), 0.0, atol=1e-6))
    @test isapprox(fitness, 0.0, atol=1e-13)
end
