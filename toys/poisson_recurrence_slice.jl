# Perform a 1D slice through parameter space, checking
# Poisson recurrence counts & max linger times.

using Plots
using Plots.PlotMeasures
using StaticArrays

include("../model/Plant.jl")
include("../tools/solve.jl")
include("../tools/equilibria.jl")
include("../toys/poisson_stability.jl")

delta = 2e-1 # Radius of delta-ball about the equilibrium.

start_p = [Plant.default_params...]
start_p[17] = -45.0 # Cashift
start_p[16] = -1.1 # xshift
end_p = [Plant.default_params...]
end_p[17] = -20.0 # Cashift
end_p[16] = -1.1 # xshift

resolution = 100 # How many points to sample.
Ca_shifts = LinRange(start_p[17], end_p[17], resolution)
x_shifts = LinRange(start_p[16], end_p[16], resolution)
ps = [[Plant.default_params...] for i in 1:resolution]
for i in 1:resolution
    ps[i][17] = Ca_shifts[i]
    ps[i][16] = x_shifts[i]
end

u0 = @SVector Float32[
    0.2f0, # x
    Plant.default_state[2],
    Plant.default_state[3],
    Plant.default_state[4],
    1.0f0, # Ca
    Plant.default_state[6],
    Plant.default_state[7]
]

tspan = (0.0f0, 1.0f6)
sols = Solve.solve(
    [u0 for _ in 1:resolution],
    ps,
    [tspan for _ in 1:resolution]
)

eqs = [Equilibria.eq(p) for p in ps]

recurrence_counts = [PoissonStability.recurrence_count(sols[i], eqs[i], delta=delta) for i in 1:resolution]
max_linger_times = [PoissonStability.max_linger_time(sols[i], eqs[i], delta=delta) for i in 1:resolution]
total_linger_times = [PoissonStability.total_linger_time(sols[i], eqs[i], delta=delta) for i in 1:resolution]
min_distances = [PoissonStability.min_distance(sols[i], eqs[i]) for i in 1:resolution]

xtick_step = Int(round(resolution/10))
plt1 = plot(
    recurrence_counts,
    xticks=(1:xtick_step:resolution, ["($(round(Ca_shifts[i], digits=3)), $(round(x_shifts[i], digits=3)))" for i in 1:xtick_step:resolution]),
    xrotation=60,
    xlabel="\$(\\Delta_{Ca}, \\Delta_x)\$",
    ylabel="# Poisson recurrences",
    top_margin=5mm,
    bottom_margin=10mm,
    left_margin=5mm,
    right_margin=5mm,
    legend=false
)
display(plt1)
plt2 = plot(
    max_linger_times,
    xticks=(1:xtick_step:resolution, ["($(round(Ca_shifts[i], digits=3)), $(round(x_shifts[i], digits=3)))" for i in 1:xtick_step:resolution]),
    xrotation=60,
    xlabel="\$(\\Delta_{Ca}, \\Delta_x)\$",
    ylabel="Max linger time",
    top_margin=5mm,
    bottom_margin=10mm,
    left_margin=5mm,
    right_margin=5mm,
    legend=false
)
display(plt2)
plt3 = plot(
    total_linger_times,
    xticks=(1:xtick_step:resolution, ["($(round(Ca_shifts[i], digits=3)), $(round(x_shifts[i], digits=3)))" for i in 1:xtick_step:resolution]),
    xrotation=60,
    xlabel="\$(\\Delta_{Ca}, \\Delta_x)\$",
    ylabel="Total linger time",
    top_margin=5mm,
    bottom_margin=10mm,
    left_margin=5mm,
    right_margin=5mm,
    legend=false
)
display(plt3)
plt4 = plot(
    min_distances,
    xticks=(1:xtick_step:resolution, ["($(round(Ca_shifts[i], digits=3)), $(round(x_shifts[i], digits=3)))" for i in 1:xtick_step:resolution]),
    xrotation=60,
    xlabel="\$(\\Delta_{Ca}, \\Delta_x)\$",
    ylabel="Min distance from equilibrium",
    top_margin=5mm,
    bottom_margin=10mm,
    left_margin=5mm,
    right_margin=5mm,
    legend=false
)
display(plt4)