# Perform a 1D slice through parameter space, checking
# Poisson recurrence counts & max linger times.

using Plots
using Plots.PlotMeasures
using StaticArrays

include("../model/Plant.jl")
include("../tools/equilibria.jl")

start_p = [Plant.default_params...]
start_p[17] = -45.0 # Cashift
start_p[16] = -1.1 # xshift
end_p = [Plant.default_params...]
end_p[17] = -20.0 # Cashift
end_p[16] = -1.1 # xshift

resolution = 50 # How many points to sample.
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

using DynamicalSystems
sys = CoupledODEs(Plant.melibeNew, u0, ps[1])

begin
    lyaparray = Float64[]
    for i in 1:resolution
        sys = CoupledODEs(Plant.melibeNew, u0, ps[i])
        lyaparray = [lyaparray; lyapunov(sys, 1000000; Ttr = 10000,d0 = 1e-6, Δt = .1)]
    end
    plot(Ca_shifts, lyaparray, xlabel="Ca_shift", ylabel="Lyapunov exponent", label = "")
end