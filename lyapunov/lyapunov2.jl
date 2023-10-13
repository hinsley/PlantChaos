function lyapunov2(pds::ParallelDynamicalSystem, T;
        Ttr = 0, Δt = 1, d0 = λdist(pds), d0_upper = d0*1e+3, d0_lower = d0*1e-3,
        show_progress = false, num_avgs = 1000, time_between = 100
    )
    #progress = ProgressMeter.Progress(round(Int, T);
    #    desc = "Lyapunov exponent: ", dt = 1.0, enabled = show_progress
    #)
    # transient
    while current_time(pds) - initial_time(pds) < Ttr
        step!(pds, Δt)
        d = λdist(pds)
        # We do the rescaling to orient the difference vector
        d0_lower ≤ d ≤ d0_upper || λrescale!(pds, d/d0)
    end
    # Set up algorithm
    t0 = current_time(pds)
    d = λdist(pds)
    d == 0 && error("Initial distance between states is zero!!!")
    d != d0 && λrescale!(pds, d/d0)
    λs = zeros(num_avgs)
    # Perform algorithm
    t = t0
    while current_time(pds) < t0 + T
        d = λdist(pds)
        if !(d0_lower ≤ d ≤ d0_upper)
            error(
                "After rescaling, the distance of reference and test states "*
                "was not `d0_lower ≤ d ≤ d0_upper` as expected. "*
                "Perhaps you are using a dynamical system where the algorithm doesn't work."
            )
        end
        # evolve until rescaling
        while d0_lower ≤ d ≤ d0_upper
            step!(pds, Δt)
            d = λdist(pds)
            current_time(pds) ≥ t0 + T && break
        end
        # local lyapunov exponent is the relative distance of the trajectories
        a = d/d0
        t = current_time(pds)
        for (i,λ) in enumerate(λs)
            #if t > t0+time_between*(i-1) && t < (T-(num_avgs-i)*time_between)
             λs[i] += log(a)
            #end
        end
        λrescale!(pds, a)
        #ProgressMeter.update!(progress, round(Int, current_time(pds)))
    end
    # Do final rescale, in case no other happened
    d = λdist(pds)
    a = d/d0
    λs[end] += log(a)
    return sum(λs)/num_avgs/(current_time(pds) - t0 - time_between*(num_avgs-1))
end

#@time lyapunov2(psys, 1000000, num_avgs = 1000, time_between = 10)

function λdist(ds::ParallelDynamicalSystem)
u1 = current_state(ds, 1)
u2 = current_state(ds, 2)
# Compute euclidean dinstace in a loop (don't care about static or not)
d = zero(eltype(u1))
@inbounds for i in eachindex(u1)
    d += (u1[i] - u2[i])^2
end
return sqrt(d)
end

# TODO: Would be nice to generalize this so that it can accept a user-defined function
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