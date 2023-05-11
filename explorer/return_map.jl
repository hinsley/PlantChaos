using .Plant

map_resolution = 500

x_offset = 1f-4 # Offset from xinf to avoid numerical issues.

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

function Ca_difference(p, v)
    return x_null_Ca(p, v) - Ca_null_Ca(p, v)
end

function generate_ics(p, state; res = map_resolution)
    V_eq, Ca_eq, x_eq = Ca_x_eq(p)
    # Generate initial conditions along the Ca nullcline.
    Vs = range(V_eq, -45f0, length=res)
    u0s = [SVector{7,Float32}([
        xinf(p, V)-x_offset,
        state[2],
        state[3],
        state[4],
        Ca_null_Ca(p, V),
        V, 
        state[7]]) for V in Vs]
end

#generate initial conditions along the Ca nullcline
#mapics = @lift generate_ics($p, $u0)

#set up ensemble problem
#map_prob = @lift ODEProblem{false}(melibeNew, $mapics[1], (0f0, 1f6), $p, save_everystep = true)
#prob_func(prob, i, repeat) = remake(prob, u0=mapics[][i])
#output_func(sol,i) = (sol.u, false)
#monteprob = @lift EnsembleProblem($map_prob, prob_func=prob_func,output_func= output_func, safetycopy=false)

mapics = @lift generate_ics($p, $u0)

map_prob = ODEProblem{false}(melibeNew, u0[], (0f0, 1f6), p[], save_everystep = false)
prob_func(prob, i, repeat) = remake(prob, u0=mapics[][i])
output_func(sol,i) = (sol.u[end][[5,1,6]], false)

function condition(u, t, integrator)
    p = integrator.p
    # Return the distance between u and the Ca nullcline in x if to the right of the equilibrium.
    if u[6] > -20 return -1f0 end
    (t < 50) ? 1f0 : -(p[15] * (p[13] * u[1] * (p[12] - u[6] + p[17]) - u[5]))
end

affect!(integrator) = terminate!(integrator) # Stop the solver
cb = ContinuousCallback(condition, affect!, affect_neg! = nothing, save_positions = (false,false))

monteprob = EnsembleProblem(map_prob, prob_func=prob_func, output_func= output_func, safetycopy=false)
mapsol = Observable(solve(monteprob, BS3(),EnsembleThreads(), trajectories=map_resolution, adaptive = false, dt = 1f0,
    callback=cb, merge_callbacks = true, verbose=false))

function build_map!(prob, mapics; map_resolution = map_resolution)
    #generate initial conditions along the Ca nullcline
    @async begin
        map_prob = remake(prob, u0 = mapics, p = p[])
        monteprob = EnsembleProblem(map_prob, prob_func=prob_func, output_func= output_func, safetycopy=false)
        mapsol[] = solve(monteprob, BS3(),EnsembleThreads(), trajectories=map_resolution, adaptive = false, dt = 1f0,
            callback=cb, merge_callbacks = true, verbose=false)
    end
end

build_map!(map_prob, mapics[])

preimage = @lift getindex.($mapics, 1)
xmap = @lift getindex.($mapsol.u,2)
#plot
lines!(mapax, preimage, xmap)
lines!(mapax, preimage, preimage, color = :white, linestyle = :dash, linewidth = 2,)


#lines(preimage, map)
preimage_phase_space = @lift [Point3f(x[5], x[1], x[6]) for x in $mapics]
scatter!(trajax, preimage_phase_space, visible = cutpoints_tog.active)
xmap_phase_space = @lift [Point3f(x[1], x[2], x[3]) for x in $mapsol]
scatter!(trajax, xmap_phase_space, visible = cutpoints_tog.active)

map_point = select_point(mapax.scene, marker = :circle)

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
end