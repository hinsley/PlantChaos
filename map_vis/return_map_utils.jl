# define ODE
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
function generate_ics!(ics_probs, p, eq, start, stop, res)
    xs = range(eq[1]*start, eq[1]*stop, length = res)
    cas = range(eq[5]*start, eq[5]*stop, length = res)

    ics = Vector{SVector{6,Float64}}(undef, res)
    Threads.@threads for i=1:length(xs)
        fastu = solve(
            remake(ics_probs[Threads.threadid()], 
                p = vcat(p[1:15], xs[i], cas[i]),
                u0 = eq[[3,4,6]]
            ),
            NewtonRaphson()
        ).u
        ics[i] = @SVector [xs[i], 0.0, fastu[1], fastu[2], cas[i], fastu[3]]
    end
    return ics
end

function generate_ics_circle!(ics_probs, p, eq, θs, radius, res)
    ics = Vector{SVector{6,Float64}}(undef, res)
    Threads.@threads for i=1:length(θs)
        theta = θs[i]
        Ca = radius*sin(theta) + eq[5]
        x = radius*cos(theta) + eq[1]
        fastu = solve(
            remake(ics_probs[Threads.threadid()], 
                p = vcat(p[1:15], x, Ca),
                u0 = eq[[3,4,6]]
            ),
            NewtonRaphson()
        ).u
        ics[i] = @SVector [x, 0.0, fastu[1], fastu[2], Ca, fastu[3]]
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
    # converge to fast ss eq
    # upper_prob
    x::Float64 = upper_sol[1,end]
    Ca::Float64  = upper_sol[5,end]
    n::Float64 = upper_sol[3,end]
    h::Float64 = upper_sol[4,end]
    V::Float64 = upper_sol[6,end]
    fastu = solve(
        remake(ics_probs[1],
            p = vcat(p[1:15], x, Ca),
            u0 = [n,h,V]
        ),
        NewtonRaphson()
    ).u
    upper = upper_sol[[5,1,6],:]
    upper[3,end] = fastu[3]
    # lower_prob
    x = lower_sol[1,end]
    Ca  = lower_sol[5,end]
    n = lower_sol[3,end]
    h = lower_sol[4,end]
    V = lower_sol[6,end]
    fastu = solve(
        remake(ics_probs[1],
            p = vcat(p[1:15], x, Ca),
            u0 = [n,h,V]
        ),
        NewtonRaphson()
    ).u
    lower = lower_sol[[5,1,6],:]
    lower[3,end] = fastu[3]

    return upper, lower
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
    # converge to fast subsystem eq
    x::Float64 = sol[1,end]
    Ca::Float64  = sol[5,end]
    n::Float64 = sol[3,end]
    h::Float64 = sol[4,end]
    V::Float64 = sol[6,end]
    fastu = solve(
        remake(ics_probs[1],
            p = vcat(prob.p.p[1:15], x, Ca),
            u0 = [n,h,V]
        ),
        NewtonRaphson()
    ).u
    # return the final value
    return fastu[3]
end

# for sharpening the maxima and minima
function refine_map!(prob, lerp, xmap, preimage, mx = false)
    min_prom = .1

    xpks = mx ? argmaxima(xmap[]) : argminima(xmap[])
    xpks, _ = peakproms(xpks, xmap[]; minprom = min_prom)
    s = mx ? -1 : 1
    println(xpks)
    for i in xpks
        opt = optimize( x -> s * xreturn(lerp, prob, x), preimage[][i-1], preimage[][i+1])
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

