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
const x_offset = 1e-8 # Offset from xinf to avoid numerical issues.

xinfinv(p, xinf) = p[16] - 50.0f0 - log(1.0f0/xinf - 1.0f0)/0.15f0 # Produces voltage.
IKCa(p, V) = p[2]*hinf(V)*minf(V)^3.0f0*(p[8]-V) + p[3]*ninf(V)^4.0f0*(p[9]-V) + p[6]*xinf(p, V)*(p[8]-V) + p[4]*(p[10]-V)/((1.0f0+exp(10.0f0*(V+50.0f0)))*(1.0f0+exp(-(63.0f0+V)/7.8f0))^3.0f0) + p[5]*(p[11]-V)

function x_null_Ca(p, v)
    return 0.5f0*IKCa(p, v)/(p[7]*(v-p[9]) - IKCa(p, v))
end
function Ca_x_eq(p)
    v_eq = find_zeros(v -> Ca_difference(p, v), xinfinv(p, 0.99e0), xinfinv(p, 0.01e0))[2]
    Ca_eq = Ca_null_Ca(p, v_eq)
    x_eq = xinf(p, v_eq)
    return v_eq, Ca_eq, x_eq
end

function Ca_null_Ca(p, v)
    return p[13]*xinf(p, v)*(p[12]-v+p[17])
end

function generate_ics_Ca(p, eq, Vs, res)
    ics = [SVector{6,Float64}([
        xinf(p, V)-x_offset,
        0.0,
        Plant.ninf(V),
        Plant.hinf(V),
        Ca_null_Ca(p, V),
        V]) for V in Vs]
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
    return sol[6,end]
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

