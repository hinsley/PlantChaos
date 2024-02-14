mutable struct Pars
    p
    eq
    radius
    v_saddle
    spiking
end

function spike_condition(u, t, integrator)
    -f(u, integrator.p, 0.0)[6]
end

function spike_affect!(integrator)
    v = integrator.u[6]
    if v < integrator.p.v_eq
         terminate!(integrator)
    else
        nothing
    end
end
spikecb = ContinuousCallback(spike_condition, spike_affect!, affect_neg! = nothing)

function c_condition(u,t,integrator)
    if (t < 50)
        1
    else
        sqrt(abs2(u[1]-p.eq[1]) + abs2(u[5]-p.eq[5])) - p.radius
    end
end
affect!(integrator) = terminate!(integrator) # Stop the solver
c_cb = ContinuousCallback(c_condition, affect!, affect_neg! = nothing)

cbset = CallbackSet(spikecb, c_cb)


function c_output_func(sol,i)
    ts = range(0, sol.t[end], length = 500)
    p = sol.prob.p
    dist = sqrt(sum(abs2.(sol.u[end] - p.eq)))
    if dist < p.radius
        (sol(ts, idxs = [5,1,6]), true)
    else
        (fill(NaN, 3, 500), false)
    end
end

c_ics_probs = [NonlinearProblem(fastplant, zeros(3), zeros(17)) for i=1:Threads.nthreads()]
c_map_prob = ODEProblem{false}(mapf, @SVector(zeros(6)), (0e0, 1e5), zeros(17))
c_monteprob = EnsembleProblem(map_prob, output_func= output_func, safetycopy=false)

function calculate_circle_map(monteprob, ics_probs, p, slider1, slider2, radius; resolution = 100)
    eq = SVector{6}(Equilibria.eq(p))

    mapics = generate_ics_circle!(ics_probs, p, eq, radius, resolution)
    preimage = range(slider1, slider2, length = resolution)

    # calculate the trajectory for every value along the ca = ca_eq line
    prob_func(prob, i, repeat) = remake(prob, u0=mapics[i])
    monteprob = remake(monteprob, prob_func = prob_func,
        output_func = output_func, p = Pars(p, eq, radius, menu_i = menu_i))
    mapsol = solve(monteprob, RK4(), EnsembleThreads(), trajectories=resolution,
        callback=cb, merge_callbacks = true, verbose=false, abstol = 1e-8, reltol = 1e-8)
    # generate data to plot the map
    if menu_i == 1
        xmap = [e[2,end] for e in mapsol]
    elseif menu_i == 2
        xmap = map(mapsol) do e
            Ca = e[1,end]
            x = e[2,end]
            arg = (Ca-eq[5])/radius
            a = abs(arg) > 1 ? NaN : asin(arg)
            if x > eq[1]
                a
            else
                pi-a
            end
        end
    end

    glue_trajs(mapsol)
    cass, xss, vss = glue_trajs(mapsol)

    # get the horizontal lines going to the saddle focus
    ln1 = [
        (preimage[end], eq[1]),
        (preimage[1], eq[1]),
    ]
    # get the horizontal lines going to the saddle periodic orbit
    saddle_po = calculate_hom_box(xmap, preimage)
    if isnan(saddle_po)
        ln2 = [
            (preimage[1], eq[1]),
            (preimage[1], eq[1]),
        ]
    else
        ln2 = [
            (preimage[end], saddle_po),
            (saddle_po, saddle_po),
        ]
    end
    # get the horizontal lines coming from the local mins and maxes
    if menu_i == 1
        lerp = linear_interpolation(reverse(preimage), reverse(mapics))
    elseif menu_i == 2
        lerp = linear_interpolation(collect(preimage) , mapics)
    end
    return (preimage, xmap, cass, xss, vss, ln1, ln2, lerp, eq)
end 