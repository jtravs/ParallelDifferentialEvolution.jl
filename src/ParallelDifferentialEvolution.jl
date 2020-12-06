module ParallelDifferentialEvolution
export diffevo, nullcb, logcb

import StatsBase
import Logging
import Statistics: mean, std
import Dates

function nullcb(gen, x, im, f, con, etime)
end

function logcb(gen, x, im, f, con, etime)
    @info "generation: $gen; convergence: $con; elapsed time: $etime" x[im], f[im], mean(f), std(f)
    flush(stdout)
    flush(stderr)
end

# fo is the objective function (must be @everywhere for distributed use) 
# we minimise the fitness
# bounds translates each parameter from [0, 1] to [min, max]
# d is dimensionality of problem
# F should be 0.4 < F < 1.0; small F speeds convergence but can cause premature convergence
# CR should be 0.1 < CR < 1.0; higher CR speeds convergence
# np should be between 5d and 10d, and must be at least 4
# see https://doi.org/10.1023/A:1008202821328
# rtol and atol set convergence tolerance: std(f) < (atol + rtol*abs(mean(f)))
# cb is a callback to use after eacg generation: defaulst to null, logcb prints stats
# fmap is the map to use to evaluate fitness: either `map` or `pmap`
function diffevo(fo, d; F=0.8, CR=0.7, np=d*10,
                 maxiter=1000, rtol=1e-3, atol=1e-14, cb=nullcb, fmap=map)
    x = [rand(d) for i in 1:np]
    f = fmap(fo, x)
    im = argmin(f)
    m = x[im]
    trials = similar(x)
    stime = Dates.now()
    for gen in 1:maxiter
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
        tf = fmap(fo, trials)
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
        con = std(f) / (atol + rtol*abs(mean(f)))
        etime = floor(Dates.now() - stime, Dates.Second)
        cb(gen, x, im, f, con, etime)
        if con < 1.0
           @info "converged on generation $gen"
           return m, f[im], etime
        end
    end
    @info "maximum number of $gen iterations reached"
    m, f[im], etime
end

end
