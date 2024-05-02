function generate_ics!(ics_probs, p, eq, start, stop, res)
    x = eq[1]
    cas = range(eq[5] + start, eq[5] + stop, length = res)

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
    fn = -1*(u[1]-p.eq[1])
    if (dx > 0) || (u[5] < p.eq[5])
        return sign(fn)*1.0
    end

    (t < 50) ? sign(fn)*1f0 : fn
end

function affect!(integrator)
    p = integrator.p
    if p.count < 1
        p.count += 1
    else
        terminate!(integrator)
    end
end

