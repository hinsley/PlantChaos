include("../tools/equilibria.jl")

#function for solving the fast subsystem to generate ics. 
#Can be replaced with simply solving V'=0 with Roots
#and solving for n and h analytically.
function fastplant(u,p)
    Ca = p[17]
    x = p[16]
    n,h,V = u
    P = Plant
    return @SVector [
        P.dn(u[1], u[3]),
        P.dh(u[2], u[3]),
        -(P.II(p, h, V) + P.IK(p, n, V) + P.IT(p, x, V) + 
        P.IKCa(p, Ca, V) + P.Ileak(p, V)) / p[1]
    ]
end
function generate_ics!(ics_probs, p, eq, xs, Ca, res)
    ics = Vector{SVector{6,Float64}}(undef, res)
    Threads.@threads for i=1:length(xs)
        fastu = solve(
            remake(ics_probs[Threads.threadid()], 
                p = vcat(p[1:15], xs[i], Ca),
                u0 = eq[[3,4,6]]
            ),
            NewtonRaphson()
        ).u
        ics[i] = @SVector [xs[i], 0.0, fastu[1], fastu[2], Ca, fastu[3]]
    end
    return ics
end
# this wraps melibenew 
# so that p can also contain the equilibrium for event handling
function mapf(u,p,t) # unpacks the parameter tuple
    p = p.p
    return melibeNew(u,p,t)
end

# for reformatting to plot trajectories in phase space as one.
function glue_trajs(mapsol)
    cass = Float64[]
    xss = Float64[]
    vss = Float64[]
    for i in eachindex(mapsol)
        append!(cass, mapsol[i][1,:])
        push!(cass, NaN)
        append!(xss, mapsol[i][2,:])
        push!(xss, NaN)
        append!(vss, mapsol[i][3,:])
        push!(vss, NaN)
    end
    return cass, xss, vss
end
# for calculating the trajectory leaving from the unstable manifold of the saddle point
function get_saddle_traj(prob,p)
    v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
    if length(v_eqs) < 5
        return fill(NaN, 6)
    end
    v_eq = v_eqs[3]
    Ca_eq = Equilibria.Ca_null_Ca(p, v_eq)
    x_eq = Plant.xinf(p, v_eq)
    saddle = [x_eq, 0.0, Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]
    jac = ForwardDiff.jacobian(u -> melibeNew(u,p,0), saddle)
    vals,vecs = eigen(jac)
    _,i = findmax(real.(vals))
    eps = .001
    upper_u0 = SVector{6}(saddle .+ eps .* real.(vecs)[:,i])
    lower_u0 = SVector{6}(saddle .- eps .* real.(vecs)[:,i])
    upper_prob = remake(prob, u0 = upper_u0, p = (p = p, eq = prob.p.eq))
    upper_sol = solve(upper_prob, RK4(), abstol = 1e-8, reltol = 1e-8, callback = cb)
    lower_prob = remake(prob, u0 = lower_u0, p = (p = p, eq = prob.p.eq))
    lower_sol = solve(lower_prob, RK4(), abstol = 1e-8, reltol = 1e-8, callback = cb)
    return (upper_sol[[5,1,6],:], lower_sol[[5,1,6],:])
end

# for calculating the horizontal lines coming from local mins and maxes
# TODO: RENAME THIS
function calculate_hom_box(xmap, preimage)
    # Get the last place where xmap - preimage changes sign
    for i in eachindex(xmap)
        # Detect a sign change in i+1 versus i (from negative to positive)
        try
            if sign(xmap[i+1] - preimage[i+1]) < sign(xmap[i] - preimage[i])
                # Return the mean of this point and the next.
                residue_i = xmap[i] - preimage[i]
                residue_i1 = xmap[i+1] - preimage[i+1]
                return (residue_i1*preimage[i]-residue_i*preimage[i+1])/(residue_i1-residue_i)
            end
        catch BoundsError
            return NaN
        end
    end
end

# for calculating single points on the map for refining near minima
function xreturn(lerp,prob,x)
    # get initial conditions by linear interpolation
    ics = lerp(x)
    # solve the map
    prob = remake(prob, u0 = ics)
    sol = solve(prob, RK4(), abstol = 1e-8, reltol = 1e-8, callback = cb)
    # return the final value
    return sol[1,end]
end

# for sharpening the maxima and minima
function refine_map!(prob, lerp, xmap, preimage, mx = false)
    min_prom = .02

    xpks = mx ? argmaxima(xmap[]) : argminima(xmap[])
    xpks, _ = peakproms(xpks, xmap[]; minprom = min_prom)
    s = mx ? -1 : 1
    println(xpks)
    for i in xpks
        opt = optimize( x -> s * xreturn(lerp, prob, x), preimage[][i+1], preimage[][i-1])
        xmap.val[i] = s * Optim.minimum(opt)
    end
    xmap[] = xmap[]
    nothing
end
# for higher order maps
function iterate_map(preimage, xmap)
    mx = maximum(xmap)
    mn = minimum(xmap)
    lerp = linear_interpolation(preimage, xmap)
    map = []
    for (i,e) in enumerate(map)
        if e > mn && e < mx
            push!(map, lerp(e))
        else
            push!(map, xreturn())
        end
    end

end

#functions for the ensemble problem

function output_func(sol,i)
    ts = range(0, sol.t[end], length = 500)
    (sol(ts, idxs = [5,1,6]), false)
end
function condition(u, t, integrator)
    p = integrator.p
    # Return the distance between u and the Ca nullcline in x if to the right of the equilibrium.
    ((t < 50) || (u[1] > p.eq[1])) ? 1.0 : -u[5] + p.eq[5]
end
affect!(integrator) = terminate!(integrator) # Stop the solver

function calculate_return_map(monteprob,ics_probs, p, slider1, slider2; resolution = 100)
    eq = SVector{6}(Equilibria.eq(p))

    println(p[17], ",", p[16])

    # find equilibria of fast subsystem along the ca = ca_eq line
    preimage = collect(range(eq[1] - slider2, eq[1] - slider1, length = resolution))
    mapics = generate_ics!(ics_probs, p, eq, preimage, eq[5], resolution)

    # calculate the trajectory for every value along the ca = ca_eq line
    prob_func(prob, i, repeat) = remake(prob, u0=mapics[i])
    monteprob = remake(monteprob, prob_func = prob_func,
        output_func = output_func, p = (p = p, eq = eq))
    mapsol = solve(monteprob, RK4(), EnsembleThreads(), trajectories=resolution,
        callback=cb, merge_callbacks = true, verbose=false, abstol = 1e-8, reltol = 1e-8)
    # generate data to plot the map
    xmap = [e[2,end] for e in mapsol]
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
    lerp = linear_interpolation(reverse(preimage), reverse(mapics))
    return (preimage, xmap, cass, xss, vss, ln1, ln2, lerp, eq)
end 