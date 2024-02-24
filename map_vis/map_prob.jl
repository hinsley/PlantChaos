function output_func(sol,i)
    ts = range(0, sol.t[end], length = 500)
    (sol(ts, idxs = [5,1,6]), false)
end
function condition(u, t, integrator)
    p = integrator.p
    ((t < 50) || (u[1] > p.eq[1])) ? 1.0 : -u[5] + p.eq[5]
end
function condition(u, t, integrator)
    p = integrator.p.p
    if u[6] > integrator.p.eq[6]
        return 1.0
    end
    # Return the distance between u and the Ca nullcline in x if to the right of the equilibrium.
    (t < 50) ? 1f0 : (p[15] * (p[13] * u[1] * (p[12] - u[6] + p[17]) - u[5]))
end
affect!(integrator) = terminate!(integrator) # Stop the solver
cb = ContinuousCallback(condition, affect!, affect_neg! = nothing)
ics_probs = [NonlinearProblem(fastplant, zeros(3), zeros(17)) for i=1:Threads.nthreads()]
map_prob = ODEProblem{false}(mapf, @SVector(zeros(6)), (0e0, 1e5), zeros(17))
monteprob = EnsembleProblem(map_prob, output_func= output_func, safetycopy=false)

function calculate_return_map(monteprob, ics_probs, p, slider1, slider2; resolution = 100)
    eq = SVector{6}(Equilibria.eq(p))
    V_eq = eq[6]
    preimage = range(V_eq - slider1, V_eq - slider2, length=resolution)

    #preimage = collect(range(eq[1] - slider2, eq[1] - slider1, length = resolution))
    mapics = generate_ics_Ca(p, eq, preimage, resolution)

    # calculate the trajectory for every value along the ca = ca_eq line
    prob_func(prob, i, repeat) = remake(prob, u0=mapics[i])
    monteprob = remake(monteprob, prob_func = prob_func,
        output_func = output_func, p = (p = p, eq = eq))
    mapsol = solve(monteprob, RK4(), EnsembleThreads(), trajectories=resolution,
        callback=cb, merge_callbacks = true, verbose=false, abstol = 1e-8, reltol = 1e-8)
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
    return (preimage, vmap, cass, xss, vss, ln1, ln2, lerp, eq)
end 