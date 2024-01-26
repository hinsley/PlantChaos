using Pkg
Pkg.activate("./homoclinic_scan")
using OrdinaryDiffEq, StaticArrays, Roots, GLMakie, NonlinearSolve, LinearAlgebra, ForwardDiff
include("../model/Plant.jl")
include("../tools/equilibria.jl")
include("./utils.jl")


# generate the parameter space
resolution = 400
xspace = range(-4, 1, length = resolution)
caspace = range(-60, 100, length = resolution)
#xspace = range(-2.2865, -2.2858, length = resolution)
#caspace = range(-38.6282, -38.6275, length = resolution)

(space, u0s) = makespace(collect(Iterators.product(xspace, caspace)));

spike_cb = ContinuousCallback(spike_condition, spike_affect!, affect_neg! = nothing)

# define the function for ensemble problem
prob = ODEProblem{false}(f, SVector{6}(zeros(6)), (0e0, 100000e0), space[1,1])

function prob_func(prob,i,repeat)
    j = ((i-1) % resolution) + 1
    k = ((i-1) ÷ resolution) + 1
    u0 = u0s[j,k]
    if isnan(u0[1])
        _p = Params(space[j,k].p, 0, 100.0)
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

cmap = :darktest #GLMakie.to_colormap([RGBf(rand(3)...) for _ in 1:50])

begin
    fig = Figure(size = (2000, 2000))
    ax = Axis(fig[1,1], xlabel = "ΔCa", ylabel = "Δx")
    pl = heatmap!(ax, caspace, xspace, results, colormap = cmap)
    Colorbar(fig[1,2], limits = (minimum(results), maximum(results)), label = "spike count", colormap = cmap)
    fig
end

# second panel
xspace = range(-2.29, -2.27, length = resolution)
caspace = range(-38.628, -38.616, length = resolution)
(space, u0s) = makespace(collect(Iterators.product(xspace, caspace)));

sol = solve(scanprob, RK4(), EnsembleThreads(), trajectories = resolution^2, callback = spike_cb, save_everystep = false);

# combine the two solutions
results2 = map(transpose(reshape(sol.u, resolution, resolution))) do x;
    x > 8 ? 8 : x
end;

begin
    ax2 = Axis(fig[1,1],
        width=Relative(0.25),
        height=Relative(0.25),
        halign=0.3,
        valign=0.45)
    hidedecorations!(ax2)
    pl = heatmap!(ax2, caspace, xspace, results2, colormap = cmap)
    fig
end

# third panel
xspace = range(-2.2861, -2.2855, length = resolution)
caspace = range(-38.6278, -38.6272, length = resolution)
(space, u0s) = makespace(collect(Iterators.product(xspace, caspace)));

sol = solve(scanprob, RK4(), EnsembleThreads(), trajectories = resolution^2, callback = spike_cb, save_everystep = false);

# combine the two solutions
results3 = map(transpose(reshape(sol.u, resolution, resolution))) do x;
    x > 8 ? 8 : x
end;

begin
    ax3 = Axis(fig[1,1],
        width=Relative(0.25),
        height=Relative(0.25),
        halign=0.75,
        valign=0.55)
    hidedecorations!(ax3)
    pl = heatmap!(ax3, caspace, xspace, results3, colormap = cmap)
    fig
end