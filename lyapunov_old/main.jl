using Pkg
Pkg.activate("../lyapunov_old/")
using Plots
using Plots.PlotMeasures
using StaticArrays

include("../model/Plant.jl")
include("../tools/equilibria.jl")

start_p = [Plant.default_params...]
start_p[17] = -60.0 # Cashift
start_p[16] = -.5 # xshift
end_p = [Plant.default_params...]
end_p[17] = -20.0 # Cashift
end_p[16] = -1.5 # xshift

resolution = 1000 # How many points to sample.
Ca_shifts = LinRange(start_p[17], end_p[17], resolution)
x_shifts = LinRange(start_p[16], end_p[16], resolution)
ps = [[Plant.default_params[1:15]; [x_shifts[i], Ca_shifts[j]]] for i in 1:resolution, j in 1:resolution]

u0 = @SVector Float32[
    0.2f0, # x
    Plant.default_state[2],
    Plant.default_state[3],
    Plant.default_state[4],
    1.0f0, # Ca
    Plant.default_state[6],
    Plant.default_state[7]
]

using DynamicalSystems
sys = CoupledODEs(Plant.melibeNew, u0, ps[1])

begin
    lyaparray = Array{Float64}(undef,resolution,resolution)
    for i in 1:resolution, j in 1:resolution
        sys = CoupledODEs(Plant.melibeNew, u0, ps[i,j])
        lyaparray[i,j] = lyapunov(sys, 1000000; Ttr = 10000,d0 = 1e-6, Δt = .1)
    end
    heatmap(x_shifts, Ca_shifts, lyaparray, xlabel="Ca_shift", ylabel="Lyapunov exponent")
end

heatmap(reverse(lyaparray, dims = 1))