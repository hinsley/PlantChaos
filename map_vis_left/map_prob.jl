mutable struct Params
    p
    eq
    sd
    last_cross
    last_ca
    ready
end

function vcondition(out, u, t, integrator)
    p = integrator.p
    if t < 100
        out[1] = 1
        return
    end
    """if (u[5] < p.eq[5] + .01)
        out[1] = -1
        return
    end"""
    ref_opp = (p.sd[1] - p.eq[1])/(p.sd[5] - p.eq[5])*(p.sd[5])
    ref_adj = (p.sd[5])
    opp = u[1] - (p.sd[1] - (p.sd[1] - p.eq[1])/(p.sd[5] - p.eq[5])*(p.sd[5]))
    adj = u[5]
    dtheta = -atan(opp/adj) + atan(ref_opp/ref_adj)

    out[1] = dtheta
    out[2] = -mapf(u, integrator.p, 0.0)[6]
    nothing
end

function vaffect!(integrator, event_idx)
    p = integrator.p
    u = integrator.u
    if event_idx == 1 # section crossing
        p.last_cross = integrator.u[5]
        if p.ready
            terminate!(integrator)
        end
    end
    if event_idx == 2 # voltage max
        if (u[6] < p.sd[6]) && (u[1] > p.eq[1]) 
            p.ready = true
        end
    end
end

vcb = VectorContinuousCallback(vcondition, vaffect!, nothing, 2)

ics_probs = [NonlinearProblem(fastplant, zeros(3), zeros(17)) for i=1:Threads.nthreads()]
map_prob = ODEProblem{false}(mapf, @SVector(zeros(6)), (0e0, 1e6), zeros(17))
monteprob = EnsembleProblem(map_prob, safetycopy=false)

function calculate_return_map(monteprob, ics_probs, p, slider1, slider2; resolution = 100)
    v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
    if length(v_eqs) < 3
        return fill(NaN, 6)
    end
    v_saddle = v_eqs[3]
    Ca_saddle = Equilibria.Ca_null_Ca(p, v_saddle)
    x_saddle = Plant.xinf(p, v_saddle)
    saddle = [x_saddle, 0.0, Plant.ninf(v_saddle), Plant.hinf(v_saddle), Ca_saddle, v_saddle]
    v_eq = v_eqs[2]
    Ca_eq = Equilibria.Ca_null_Ca(p, v_eq)
    x_eq = Plant.xinf(p, v_eq)
    eq = [x_eq, 0.0, Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]

    mapics = generate_ics!(ics_probs, p, eq, saddle, slider2, slider1, resolution)
    preimage = [e[5] for e in mapics]

    # calculate the trajectory for every value along the ca = ca_eq line
    prob_func(prob, i, repeat) = remake(prob, u0=mapics[i])
    monteprob = remake(monteprob, prob_func = prob_func, p = Params(p, eq, saddle, 0.0, 0.0, false))
    sol = solve(monteprob, RK4(), EnsembleThreads(), trajectories=resolution,
        callback=vcb, verbose=true, abstol = 1e-8, reltol = 1e-8)
    
    mapsol = map(sol) do e
        ts = range(0, e.t[end], length = 500)
        return e(ts, idxs = [5,1,6])
    end
    # generate data to plot the map

    vmap = [e.prob.p.last_cross for e in sol.u]
    cass, xss, vss = glue_trajs(mapsol)
    points = [e[end] for e in sol.u]

    # get the horizontal lines going to the saddle focus
    ln1 = [
        (preimage[end], eq[5]),
        (preimage[1], eq[5]),
    ]
    # get the horizontal lines going to the saddle periodic orbit
    saddle_po = calculate_hom_box(reverse(vmap), reverse(preimage))
    if isnan(saddle_po)
        ln2 = [
            (preimage[1], eq[5]),
            (preimage[1], eq[5]),
        ]
    else
        ln2 = [
            (preimage[1], saddle_po),
            (saddle_po, saddle_po),
        ]
    end
    # get the horizontal lines coming from the local mins and maxes

    lerp = linear_interpolation(preimage, mapics)

    return (preimage, vmap, cass, xss, vss, ln1, ln2, lerp, eq, points)
end