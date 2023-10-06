function my_lyapunov(sys, T, Ttr, dt; d0 = 1e-6, num_avg = 100, max_period = 10000)
    # integrate trajectory
    λs = Array{Float64,1}(undef, ceil(Int, T/dt))
    for i in 1:(T+Ttr)
        d = λdist(psys)/d0
        λrescale!(psys, d)
        step!(psys)
        if i>Ttr
            λs[i-Ttr] = log(d)
        end
    end
    # build average of cumsums starting from different points over max_period
    max_i = ceil(Int, max_period/dt)
    λtot = zeros(length(λs)-max_i)
    idxs = floor.(Int, LinRange(1, max_i-1, num_avg))
    for i in idxs
        #return length(collect(1:(length(λs)-max_i)))
        λtot = λtot .+ cumsum(λs[i:end-max_i+i-1])./collect(1:(length(λs)-max_i))
    end
    λtot = λtot ./ length(idxs)
    return sum(λs[end-max_i+1:end])/max_i
end
# Do final rescale, in case no other happened

function λrescale!(pds::ParallelDynamicalSystem, a)
    u1 = current_state(pds, 1)
    u2 = current_state(pds, 2)
    if ismutable(u2) # if mutable we assume `Array`
        @. u2 = u1 + (u2 - u1)/a
    else # if not mutable we assume `SVector`
        u2 = @. u1 + (u2 - u1)/a
    end
    set_state!(pds, u2, 2)
end
function λdist(ds::ParallelDynamicalSystem)
    u1 = current_state(ds, 1)
    u2 = current_state(ds, 2)
    # Compute euclidean dinstace in a loop (don't care about static or not)
    d = zero(eltype(u1))
    @inbounds for i in 1:6
        d += (u1[i] - u2[i])^2
    end
    return sqrt(d)
end
