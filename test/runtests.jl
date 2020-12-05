using ParallelDifferentialEvolution
using Test

@testset "ParallelDifferentialEvolution.jl" begin
    function bounds(x)
        minx = -10.0
        maxx = 10.0
        x .* (maxx .- minx) .+ minx
    end
    function f(x)
        x = bounds(x)
        [sum(x[:,j].^2) / length(x) for j = 1:size(x,2)]
    end
    m, fitness, track = diffevo(f, 1, np=10)
    @test isapprox(bounds(m)[1], 0.0, atol=1e-30)
    @test isapprox(fitness, 0.0, atol=1e-30)
    m, fitness, track = diffevo(f, 3)
    @test all(isapprox.(bounds(m), 0.0, atol=1e-14))
    @test isapprox(fitness, 0.0, atol=1e-30)
end
