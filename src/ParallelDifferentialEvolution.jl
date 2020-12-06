module ParallelDifferentialEvolution
export diffevo

import StatsBase
import Logging
import Statistics: mean, std

# fo must iterate over individuals and return an array of fitnesses
# we minimise this fitness
# d is dimensionality of problem
# F should be 0.4 < F < 1.0; small F speeds convergence but can cause premature convergence
# CR should be 0.1 < CR < 1.0; higher CR speeds convergence
# np should be between 5d and 10d, and must be at least 4
# see https://doi.org/10.1023/A:1008202821328
function diffevo(fo, d; F=0.8, CR=0.7, np=d*7, maxiter=1000)
    x = [rand(d) for i in 1:np]
    f = fo(x)
    im = argmin(f)
    m = x[im]
    trials = similar(x)
    track = [(m, f[im], mean(f), std(f))]
    for i in 1:maxiter
        for j in 1:np
            ii = [k for k in 1:np if k != j]
            a, b, c = x[StatsBase.sample(ii, 3, replace=false)]
            mut = clamp.(a .+ F .* (b .- c), 0.0, 1.0)
            cp = rand(d) .< CR
            if !any(cp)
                cp[rand(1:d)] = true
            end
            trials[j] = ifelse.(cp, mut, x[j])
        end
        tf = fo(trials)
        for j in 1:np
            if tf[j] < f[j]
                f[j] = tf[j]
                x[j] = trials[j]
                if f[j] < f[im]
                    im = j
                    m = trials[j]
                end
            end
        end
        push!(track, (m, f[im], mean(f), std(f)))
        @info "generation: $i:" track[end]
        flush(stdout)
        flush(stderr)
    end
    m, f[im], track
end

end
