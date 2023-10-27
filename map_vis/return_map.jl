
map_resolution = 300

# generate initial conditions
# find equilibrium in full subsystem
include("../tools/equilibria.jl")

eq_guess = @lift SVector{6}(Equilibria.eq($p) .+ .01)

_eq_prob = NonlinearProblem((x,p) -> melibeNew(x,p,0.0), @SVector(zeros(6)), p[])
eq = @lift solve(
    remake(_eq_prob, p = $p, u0 = $eq_guess),
    NewtonRaphson()
).u
# find equilibria of fast subsystem along x= x_eq

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
ics_xs = @lift range($eq[1], $eq[1] - $(mapslider.sliders[1].value), length = map_resolution)

function generate_ics!(ics_probs, p, eq, xs, Ca)
    ics = Vector{SVector{6,Float64}}(undef, map_resolution)
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

mapics = @lift generate_ics!(ics_probs, $p, $eq, $ics_xs, $eq[5])

# integrate to find map

function mapf(u,p,t) # unpacks the parameter tuple
    p = p.p
    return melibeNew(u,p,t)
end

_map_prob = ODEProblem{false}(mapf, @SVector(zeros(6)), (0e0, 1e6), zeros(17))
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

on(mapslider.sliders[1].value) do val
    mapsol[] = solve(monteprob[], RK4(),EnsembleThreads(), trajectories=map_resolution,
        callback=cb, merge_callbacks = true, verbose=false, abstol = 1e-8, reltol = 1e-8)
    reset_limits!(mapax)
    reset_limits!(trajax)
end

preimage = @lift getindex.($mapics, 1)

cass = @lift([$mapsol[i][1,:] for i in 1:map_resolution])
xss = @lift([$mapsol[i][2,:] for i in 1:map_resolution])
vss = @lift([$mapsol[i][3,:] for i in 1:map_resolution])

xmap = @lift [$mapsol[i][2,end] for i in 1:map_resolution]

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
lines!(mapax, preimage, xmap)
lines!(mapax, preimage, preimage, color = :white, linestyle = :dash, linewidth = 2,)

colorrng = @lift range(0,1, length = length($cass))

lines!(trajax, cass, xss, vss, color = colorrng, colormap = :thermal, linewidth = .5, fxaa = false)
 
"""map_point = select_point(mapax.scene, marker = :circle)

on(map_point) do pars
    # do not trigger when reset limit
    if !ispressed(mapax, Keyboard.left_control)
        ca_n, _ = pars
        V_n = find_zero((V) -> Ca_null_Ca(p[], V) - ca_n, -40)
        x_n = xinf(p[], V_n)

        u0.val = (x_n, u0[][2:4]..., ca_n, V_n, u0[][end])
        auto_dt_reset!(dynsys[].integ)
        u0[] = u0[]
    end
end"""
nothing
fig