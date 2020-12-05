module ParallelDifferentialEvolution
export diffevo

import StatsBase

# fo must iterate over individuals and return an array of fitnesses
# d is dimensionality of problem
# F should be 0.4 < F < 1.0; small F speeds convergence, can cause premature convergence
# CR should be 0.1 < CR < 1.0; higher speeds convergence
# np should be between 5 and 10 * dimensionality; must be at least 4
# see https://doi.org/10.1023/A:1008202821328
function diffevo(fo, d; F=0.8, CR=0.7, np=d*7, maxiter=1000)
    x = rand(Float64, (d, np))
    f = fo(x)
    im = argmin(f)
    m = x[:,im]
    trials = similar(x)
    track = [(m, f[im])]
    for i in 1:maxiter
        for j in 1:np
            ii = [i for i in 1:np if i != j]
            sel = x[:, StatsBase.sample(ii, 3, replace=false)]
            mut = clamp.(sel[:,1] .+ F .* (sel[:,2] .- sel[:,3]), 0.0, 1.0)
            cp = rand(d) .< CR
            if !any(cp)
                cp[rand(1:d)] = true
            end
            trials[:,j] .= ifelse.(cp, mut, x[:,j])
        end
        tf = fo(trials)
        for j in 1:np
            if tf[j] < f[j]
                f[j] = tf[j]
                x[:,j] .= trials[:,j]
                if f[j] < f[im]
                    im = j
                    m = trials[:,j]
                end
            end
        end
        push!(track, (m, f[im]))
    end
    m, f[im], track
end

end
