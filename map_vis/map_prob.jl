function condition(u, t, integrator)
    p = integrator.p
    ((t < 50) || (u[1] > p.eq[1])) ? 1.0 : -u[5] + p.eq[5]
end
function condition(u, t, integrator)
    p = integrator.p
    if u[6] > p.eq[6]
        return 1.0
    end
    # Return the distance between u and the Ca nullcline in x if to the right of the equilibrium.
    (t < 50) ? 1f0 : atan(u[1]/u[5]) - atan(p.eq[1]/p.eq[5])
end
affect!(integrator) = terminate!(integrator) # Stop the solver
cb = ContinuousCallback(condition, affect!, affect_neg! = nothing)
ics_probs = [NonlinearProblem(fastplant, zeros(3), zeros(17)) for i=1:Threads.nthreads()]
map_prob = ODEProblem{false}(mapf, @SVector(zeros(6)), (0e0, 1e5), zeros(17))
monteprob = EnsembleProblem(map_prob, safetycopy=false)

function calculate_return_map(monteprob, ics_probs, p, slider1, slider2; resolution = 100)
    eq = SVector{6}(Equilibria.eq(p))    

    mapics = generate_ics!(ics_probs, p, eq, slider1, slider2, resolution)
    preimage = [e[6] for e in mapics]

    # calculate the trajectory for every value along the ca = ca_eq line
    prob_func(prob, i, repeat) = remake(prob, u0=mapics[i])
    monteprob = remake(monteprob, prob_func = prob_func, p = (p = p, eq = eq))
    sol = solve(monteprob, RK4(), EnsembleThreads(), trajectories=resolution,
        callback=cb, merge_callbacks = true, verbose=false, abstol = 1e-8, reltol = 1e-8)
    
    
    mapsol = map(sol) do e
        ts = range(0, e.t[end], length = 500)
        return e(ts, idxs = [5,1,6])
    end
    # generate data to plot the map

    vmap = [e[3,end] for e in mapsol]
    cass, xss, vss = glue_trajs(mapsol)

    # get the horizontal lines going to the saddle focus
    ln1 = [
        (preimage[end], eq[6]),
        (preimage[1], eq[6]),
    ]
    # get the horizontal lines going to the saddle periodic orbit
    saddle_po = calculate_hom_box(reverse(vmap), reverse(preimage))
    if isnan(saddle_po)
        ln2 = [
            (preimage[1], eq[6]),
            (preimage[1], eq[6]),
        ]
    else
        ln2 = [
            (preimage[1], saddle_po),
            (saddle_po, saddle_po),
        ]
    end
    # get the horizontal lines coming from the local mins and maxes
    lerp = linear_interpolation(preimage, mapics)

    # run map to fast subsystem equilibria for consistency
    
    for i = 2:resolution
        x::Float64 = sol.u[i][1,end]
        Ca::Float64 = sol.u[i][5,end]
        n::Float64 = sol.u[i][3,end]
        h::Float64 = sol.u[i][4,end]
        v::Float64 = sol.u[i][6,end]
        fastu = solve(
            remake(ics_probs[1],
                p = vcat(p[1:15], x, Ca),
                u0 = [n,h,v]
            ),
            NewtonRaphson()
        ).u
        vmap[i] = fastu[3]
        sample_len = size(mapsol[i], 2)
        vss[i*sample_len + i - 1] = fastu[3]
    end

    return (preimage, vmap, cass, xss, vss, ln1, ln2, lerp, eq)
end 

_ans = calculate_return_map(monteprob, ics_probs, p[], mapslider.sliders[1].value[],
    mapslider.sliders[2].value[], resolution = map_resolution)