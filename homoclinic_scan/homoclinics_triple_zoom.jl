using Pkg
Pkg.activate("./homoclinic_scan")
using OrdinaryDiffEq, StaticArrays, Roots, GLMakie, NonlinearSolve, LinearAlgebra, ForwardDiff
include("../model/Plant.jl")
include("../tools/equilibria.jl")
include("./utils.jl")


# generate the parameter space
resolution = 1000
xspace = range(-4, 1, length = resolution)
caspace = range(-60, 100, length = resolution)
#xspace = range(-2.2865, -2.2858, length = resolution)
#caspace = range(-38.6282, -38.6275, length = resolution)

(space, u0s) = makespace(collect(Iterators.product(xspace, caspace)));

spike_cb = ContinuousCallback(spike_condition, spike_affect!, affect_neg! = nothing)

# define the function for ensemble problem
prob = ODEProblem{false}(f, SVector{6}(zeros(6)), (0e0, 220000e0), space[1,1])

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
scanprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false, output_func=output_func);
sol = solve(scanprob, RK4(), EnsembleThreads(), trajectories = resolution^2, callback = spike_cb, save_everystep = false);


results = map(transpose(reshape(sol.u, resolution, resolution))) do x;
    x > 30 ? 30 : x
end;

# second panel
xspace2 = range(-2.29, -2.27, length = resolution)
caspace2 = range(-38.628, -38.616, length = resolution)
(space, u0s) = makespace(collect(Iterators.product(xspace2, caspace2)));

sol = solve(scanprob, RK4(), EnsembleThreads(), trajectories = resolution^2, callback = spike_cb, save_everystep = false);

# combine the two solutions
results2 = map(transpose(reshape(sol.u, resolution, resolution))) do x;
    x > 30 ? 30 : x
end;

# third panel
xspace3 = range(-2.2861, -2.2855, length = resolution)
caspace3 = range(-38.6278, -38.6272, length = resolution)
(space, u0s) = makespace(collect(Iterators.product(xspace3, caspace3)));

sol = solve(scanprob, RK4(), EnsembleThreads(), trajectories = resolution^2, callback = spike_cb, save_everystep = false);

# combine the two solutions
results3 = map(transpose(reshape(sol.u, resolution, resolution))) do x;
    x > 30 ? 30 : x
end;

cmap = :darktest #GLMakie.to_colormap([RGBf(rand(3)...) for _ in 1:50])

begin
    fig = Figure(size = (2000,2000))
    ax = Axis(fig[1,1], xlabel = "ΔCa", ylabel = "Δx")
    pl = heatmap!(ax, caspace, xspace, results, colormap = cmap)
    Colorbar(fig[1,2], limits = (minimum(results), maximum(results)), label = "spike count", colormap = cmap)
    scale = 1.2
    ax2 = Axis(fig[1,1],
        width=Relative(0.15*scale),
        height=Relative(0.25*scale),
        halign=0.15,
        valign=0.75)
    hidedecorations!(ax2)
    pl = heatmap!(ax2, caspace2, xspace2, results2, colormap = cmap)
    ax3 = Axis(fig[1,1],
        width=Relative(0.15*scale),
        height=Relative(0.25*scale),
        halign=0.4,
        valign=0.85)
    hidedecorations!(ax3)
    pl = heatmap!(ax3, caspace, xspace, results3, colormap = cmap)
    fig
end
save("./hom_scan_double_zoom.png", fig)

"""begin
    a = GLMakie.Screen()
    fig2 = Figure()
    ax = Axis(fig2[1,1])
    heatmap!(ax, caspace, xspace, results3, colormap = cmap)
    Colorbar(fig2[1,2], colormap = cmap, limits = (minimum(results3), maximum(results3)))
    display(a,fig2)
end
"""