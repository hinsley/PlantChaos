using Pkg
Pkg.activate("./homoclinic_scan")
using OrdinaryDiffEq, StaticArrays, Roots, GLMakie, NonlinearSolve, LinearAlgebra, ForwardDiff
include("../model/Plant.jl")
include("../tools/equilibria.jl")
include("./utils.jl")


# generate the parameter space
resolution = 100
xspace = range(-2.29, -2.27, length = resolution)
caspace = range(-38.622, -38.616, length = resolution)
space, u0s = makespace(collect(Iterators.product(xspace, caspace)));

spike_cb = ContinuousCallback(spike_condition, spike_affect!, affect_neg! = nothing)

# define the function for ensemble problem
prob = ODEProblem{false}(f, SVector{6}(zeros(6)), (0e0, 100000e0), space[1,1])

function prob_func(prob,i,repeat)
    j = ((i-1) % resolution) + 1
    k = ((i-1) ÷ resolution) + 1
    u0 = u0s[j,k]
    if isnan(u0[1])
        _p = Params(space[j,k].p, 0, 100.0, 0.0, 1.0)
        _u0 = SVector{6}(zeros(6))
        prob = remake(prob, p = _p , u0 = _u0, tspan = (0.0, 1.0))
    else
        p = deepcopy(space[j,k])
        prob = remake(prob, p = p, u0 = u0)
    end
    return prob
end

# define the ensemble problem
scanprob = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func, safetycopy = false)
sol = solve(scanprob, RK4(), EnsembleThreads(), trajectories = resolution^2, callback = spike_cb, save_everystep = false);

# combine the two solutions
results = map(transpose(reshape(sol.u, resolution, resolution))) do x;
    x > 30 ? 30 : x
end;

#saved_results = deepcopy(results)
# plot the results
#results = saved_results

cmap = GLMakie.to_colormap([RGBf(rand(3)...) for _ in 1:50])

begin
    fig = Figure(size = (2000, 2000))
    ax = Axis(fig[1,1], xlabel = "ΔCa", ylabel = "Δx")
    pl = heatmap!(ax, caspace, xspace, results, colormap = cmap)
    Colorbar(fig[1,2], limits = (minimum(results), maximum(results)), label = "spike count", colormap = cmap)
    fig
end


#using JLD2
#JLD2.save("homscan.jld2", "results", results)
#save("homscan.png", fig)

#test plot

let

    f = Figure()
    ax1 = Axis(f[1,1], ylabel = "x")
    ax2 = Axis(f[2,1], ylabel = "x")
    ax3 = Axis(f[3,1], xlabel = "Ca", ylabel = "x")

    ΔCa = -.92190969905
    Δx = -1.11518075
    p = vcat(Plant.default_params[1:15], Δx, ΔCa)
    # calculate initial conditions
    v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
    v_eq = v_eqs[3]
    Ca_eq = Equilibria.Ca_null_Ca(p, v_eq)
    x_eq = Plant.xinf(p, v_eq)
    eq = [x_eq, Plant.default_state[2], Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]
    jac = ForwardDiff.jacobian(u -> Plant.melibeNew(u,p,0), eq)
    vals,vecs = eigen(jac)
    _,ix = findmax(real.(vals))
    eps = .001
    vec = real.(vecs)[:,ix]
    u0 = SVector{6}(eq - eps * vec * sign(vec[1]))

    prob2 = remake(prob, p = Params(p, 0,0.0), u0 = u0)
    sol = solve(prob2, RK4(), abstol = 1e-8, reltol = 1e-8, callback = spike_cb)

    lines!(ax1, sol[5,:], sol[1,:])

    ΔCa = -.9219096991
    Δx = -1.11518075
    p = vcat(Plant.default_params[1:15], Δx, ΔCa)
    # calculate initial conditions
    v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
    v_eq = v_eqs[3]
    Ca_eq = Equilibria.Ca_null_Ca(p, v_eq)
    x_eq = Plant.xinf(p, v_eq)
    eq = [x_eq, Plant.default_state[2], Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]
    jac = ForwardDiff.jacobian(u -> Plant.melibeNew(u,p,0), eq)
    vals,vecs = eigen(jac)
    _,ix = findmax(real.(vals))
    eps = .001
    vec = real.(vecs)[:,ix]
    u0 = SVector{6}(eq - eps * vec * sign(vec[1]))

    prob2 = remake(prob, p = Params(p, 0,0.0), u0 = u0)
    sol = solve(prob2, RK4(), abstol = 1e-8, reltol = 1e-8, callback = spike_cb)

    lines!(ax1, sol[5,:], sol[1,:], linestyle = :dot, color = :black, linewidth = 2)

    ΔCa = 10.394
    Δx = -1.11518075
    p = vcat(Plant.default_params[1:15], Δx, ΔCa)
    # calculate initial conditions
    v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
    v_eq = v_eqs[3]
    Ca_eq = Equilibria.Ca_null_Ca(p, v_eq)
    x_eq = Plant.xinf(p, v_eq)
    eq = [x_eq, Plant.default_state[2], Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]
    jac = ForwardDiff.jacobian(u -> Plant.melibeNew(u,p,0), eq)
    vals,vecs = eigen(jac)
    _,ix = findmax(real.(vals))
    eps = .001
    vec = real.(vecs)[:,ix]
    u0 = SVector{6}(eq - eps * vec * sign(vec[1]))

    prob2 = remake(prob, p = Params(p, 0,0.0), u0 = u0)
    sol = solve(prob2, RK4(), abstol = 1e-8, reltol = 1e-8, callback = spike_cb)

    lines!(ax2, sol[5,:], sol[1,:])

    ΔCa = 10.39
    Δx = -1.11518075
    p = vcat(Plant.default_params[1:15], Δx, ΔCa)
    # calculate initial conditions
    v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
    v_eq = v_eqs[3]
    Ca_eq = Equilibria.Ca_null_Ca(p, v_eq)
    x_eq = Plant.xinf(p, v_eq)
    eq = [x_eq, Plant.default_state[2], Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]
    jac = ForwardDiff.jacobian(u -> Plant.melibeNew(u,p,0), eq)
    vals,vecs = eigen(jac)
    _,ix = findmax(real.(vals))
    eps = .001
    vec = real.(vecs)[:,ix]
    u0 = SVector{6}(eq - eps * vec * sign(vec[1]))

    prob2 = remake(prob, p = Params(p, 0,0.0), u0 = u0)
    sol = solve(prob2, RK4(), abstol = 1e-8, reltol = 1e-8, callback = spike_cb)

    lines!(ax2, sol[5,:], sol[1,:], linestyle = :dot, color = :black, linewidth = 2)



    ΔCa = 25.085
    Δx = -1.11518075
    p = vcat(Plant.default_params[1:15], Δx, ΔCa)
    # calculate initial conditions
    v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
    v_eq = v_eqs[3]
    Ca_eq = Equilibria.Ca_null_Ca(p, v_eq)
    x_eq = Plant.xinf(p, v_eq)
    eq = [x_eq, Plant.default_state[2], Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]
    jac = ForwardDiff.jacobian(u -> Plant.melibeNew(u,p,0), eq)
    vals,vecs = eigen(jac)
    _,ix = findmax(real.(vals))
    eps = .001
    vec = real.(vecs)[:,ix]
    u0 = SVector{6}(eq - eps * vec * sign(vec[1]))

    prob2 = remake(prob, p = Params(p, 0,0.0), u0 = u0)
    sol = solve(prob2, RK4(), abstol = 1e-8, reltol = 1e-8, callback = spike_cb)

    lines!(ax3, sol[5,:], sol[1,:])

    ΔCa = 25.08
    Δx = -1.11518075
    p = vcat(Plant.default_params[1:15], Δx, ΔCa)
    # calculate initial conditions
    v_eqs = find_zeros(v -> Equilibria.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
    v_eq = v_eqs[3]
    Ca_eq = Equilibria.Ca_null_Ca(p, v_eq)
    x_eq = Plant.xinf(p, v_eq)
    eq = [x_eq, Plant.default_state[2], Plant.ninf(v_eq), Plant.hinf(v_eq), Ca_eq, v_eq]
    jac = ForwardDiff.jacobian(u -> Plant.melibeNew(u,p,0), eq)
    vals,vecs = eigen(jac)
    _,ix = findmax(real.(vals))
    eps = .001
    vec = real.(vecs)[:,ix]
    u0 = SVector{6}(eq - eps * vec * sign(vec[1]))

    prob2 = remake(prob, p = Params(p, 0,0.0), u0 = u0)
    sol = solve(prob2, RK4(), abstol = 1e-8, reltol = 1e-8, callback = spike_cb)

    lines!(ax3, sol[5,:], sol[1,:], linestyle = :dot, color = :black, linewidth = 2)



    println(sol.prob.p.count)
    f
end
