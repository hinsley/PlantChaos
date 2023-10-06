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
