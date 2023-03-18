IKCa(p, V) = p[2]*Plant.hinf(V)*Plant.minf(V)^3.0f0*(p[8]-V) + p[3]*Plant.ninf(V)^4.0f0*(p[9]-V) + p[6]*Plant.xinf(p, V)*(p[8]-V) + p[4]*(p[10]-V)/((1.0f0+exp(10.0f0*(V+50.0f0)))*(1.0f0+exp(-(63.0f0+V)/7.8f0))^3.0f0) + p[5]*(p[11]-V)
xinfinv(p, xinf) = p[16] - 50.0f0 - log(1.0f0/xinf - 1.0f0)/0.15f0 # Produces voltage.

function x_null_Ca(p, v)
    return 0.5f0*IKCa(p, v)/(p[7]*(v-p[9]) - IKCa(p, v))
end

function Ca_null_Ca(p, v)
    return p[13]*Plant.xinf(p, v)*(p[12]-v+p[17])
end

# The function which must be minimized to find the equilibrium voltage.
function Ca_difference(p, v)
    return x_null_Ca(p, v) - Ca_null_Ca(p, v)
end

# Finds the equilibrium in the slow subsystem.
function Ca_x_eq(p)
    v_eq = find_zeros(v -> Ca_difference(p, v), xinfinv(p, 0.99e0), xinfinv(p, 0.01e0))[2]
    Ca_eq = Ca_null_Ca(p, v_eq)
    x_eq = Plant.xinf(p, v_eq)
    return v_eq, Ca_eq, x_eq
end

function Ca_null_V(p, Ca)
    return find_zero(v -> Ca_null_Ca(p, v) - Ca, (xinfinv(p, 0.99e0), xinfinv(p, 0.01e0)))
end

prob = ODEProblem{false}(Plant.melibeNew, initial_conditions(p), tspan, p)
monteprob = EnsembleProblem(prob, safetycopy=false)
@time sol = solve(monteprob, Tsit5(), EnsembleThreads(), adaptive=false, trajectories=1, dt=1.0f0, abstol=1f-6, reltol=1f-6, verbose=false)


@lift Δx = $p[16]
@lift ΔCa = $p[17]