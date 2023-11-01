
map_resolution = 100
max_minima = 10

# generate initial conditions

## find equilibrium in full subsystem
include("../tools/equilibria.jl")

eq_guess = @lift SVector{6}(Equilibria.eq($p) .+ .01)

_eq_prob = NonlinearProblem((x,p) -> melibeNew(x,p,0.0), @SVector(zeros(6)), p[])
eq = @lift solve(
    remake(_eq_prob, p = $p, u0 = $eq_guess),
    NewtonRaphson()
).u

# find equilibria of fast subsystem along the ca = ca_eq line
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

ics_probs = [NonlinearProblem(fastplant, zeros(3), zeros(17)) for i=1:Threads.nthreads()]
ics_xs = @lift range($eq[1] - $(mapslider.sliders[2].value), 
    $eq[1] - $(mapslider.sliders[1].value),
    length = map_resolution)

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

mapics = @lift generate_ics!(ics_probs, $p, $eq, $ics_xs, $eq[5], map_resolution)

# calculate the map trajectory for every value along the ca = ca_eq line

function mapf(u,p,t) # unpacks the parameter tuple
    p = p.p
    return melibeNew(u,p,t)
end

_map_prob = ODEProblem{false}(mapf, @SVector(zeros(6)), (0e0, 1e5), zeros(17))
map_prob = @lift remake(_map_prob, p = (p = $p, eq = $eq))

prob_func(prob, i, repeat) = remake(prob, u0=mapics[][i])

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
cb = ContinuousCallback(condition, affect!, affect_neg! = nothing)

monteprob = @lift EnsembleProblem($map_prob, prob_func=prob_func, output_func= output_func, safetycopy=false)
mapsol = @lift solve($monteprob, RK4(),EnsembleThreads(), trajectories=map_resolution,
    callback=cb, merge_callbacks = true, verbose=false, abstol = 1e-8, reltol = 1e-8)

# calculate the trajectories leaving from the unstable manifold of the saddle point if it exists
function get_saddle(prob, p)
    v_eqs = find_zeros(v -> Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
    if length(v_eqs) < 5
        return fill(NaN, 6)
    end
    v_eq = v_eqs[3]
    Ca_eq = Equilibria.Ca_null_Ca(p, v_eq)
    x_eq = Plant.xinf(p, v_eq)
    return eq = [x_eq, 0.0, Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]
end

saddle = @lift get_saddle(_eq_prob, $p)

# ensure that the map is recalculated whent the map limits are changed
onany(mapslider.sliders[1].value, mapslider.sliders[2].value) do val, _
    mapsol[] = solve(monteprob[], RK4(),EnsembleThreads(), trajectories=map_resolution[],
        callback=cb, merge_callbacks = true, verbose=false, abstol = 1e-8, reltol = 1e-8)
    reset_limits!(mapax)
    reset_limits!(trajax)
end

preimage = @lift getindex.($mapics, 1)

cass = @lift([$mapsol[i][1,:] for i in 1:map_resolution])
xss = @lift([$mapsol[i][2,:] for i in 1:map_resolution])
vss = @lift([$mapsol[i][3,:] for i in 1:map_resolution])

xmap = @lift [$mapsol[i][2,end] for i in 1:map_resolution]

function calculate_hom_box(xmap, preimage)
    d = diff(xmap.- preimage)
    i = findfirst(i -> d[i]*d[i+1] < 0, 1:(length(d)-1))
    return preimage[i], preimage[i+1], xmap[i], xmap[i+1]
end


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

_ans = @lift glue_trajs($mapsol)
cass = @lift $_ans[1]
xss = @lift $_ans[2]
vss = @lift $_ans[3]

#plot
lines!(mapax, preimage, xmap, color = range(0.,1., length=map_resolution), colormap = :thermal)
lines!(mapax, preimage, preimage, color = :white, linestyle = :dash, linewidth = 2,)

colorrng = @lift range(0,1, length = length($cass))

lines!(trajax, cass, xss, vss, color = colorrng, colormap = :thermal,
 linewidth = 1.0, fxaa = false, alpha = 0.5)

function refine_map!(mapsol, mapics, p, eq, xs, Ca, res)
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