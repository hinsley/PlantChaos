function generate_ics!(ics_probs, p, eq, start, stop, res)
    x = eq[1]
    cas = range(eq[5] + start, eq[5] + (1 - stop)/10, length = res)

    ics = Vector{SVector{6,Float64}}(undef, res)
    Threads.@threads for i=1:length(cas)
        fastu = solve(
            remake(ics_probs[Threads.threadid()], 
                p = vcat(p[1:15], x, cas[i]),
                u0 = eq[[3,4,6]]
            ),
            NewtonRaphson()
        ).u
        ics[i] = @SVector [x, 0.0, fastu[1], fastu[2], cas[i], fastu[3]]
    end
    return ics
end

function condition(u, t, integrator)
    p = integrator.p
    dx = get_du(integrator)[1]
    if (u[5] < p.eq[5]) || (dx > 0)
        return 1.0
    end
    (t < 5000) ? 1f0 : -1*(u[1]-p.eq[1])
end
