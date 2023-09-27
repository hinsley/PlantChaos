# Perform a 1D slice through parameter space, checking
# Poisson recurrence counts & max linger times.

using GLMakie
using StaticArrays

include("../model/Plant.jl")
include("../tools/equilibria.jl")

start_p = [Plant.default_params...]
start_p[17] = -70.0 # Cashift
start_p[16] = -4. # xshift
end_p = [Plant.default_params...]
end_p[17] = 100.0 # Cashift
end_p[16] = 1. # xshift

resolution = 2 # How many points to sample.
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

using OrdinaryDiffEq

function lyapscan(f,u0, ps, T, Ttr = 10000, Δt = .1)
    prob = ODEProblem(f, u0, ps, t)
    # for each u0, run for Ttr and set new u0
    prob_func = (prob, i, repeat) -> remake(prob, ps = ps[i])

    monteprob = EnsembleProblem(prob, prob_func = prob_func)

    # run each trajectory until t

    # calculate parallel trajectories and record magnitudes

    # calculate lyapunov
    return 1/ sum(log, )
end


@time begin
    lyaparray = Array{Float64}(undef,resolution,resolution)
    for I in 1:resolution^2
        i = (I-1)%resolution + 1
        j= convert.(Int64,ceil(I/resolution))
        sys = CoupledODEs(Plant.melibeNew, u0, ps[i,j])
        lyaparray[i,j] = lyapunovspectrum(sys, 1000000, 1; Ttr = 200000, Δt = .1)
    end
end



let fig = Figure(resolution = (2000,2000))
    ax = Axis(fig[1,1], xlabel = L"\Delta Ca", ylabel = L"\Delta x")
    hm = GLMakie.heatmap!(ax, Ca_shifts, x_shifts, lyaparray'; colorrange = (.0000,.00025), 
    colormap = [:midnightblue, :lightskyblue1, :white, :yellow, :orange, :red], lowclip= :black)
    Colorbar(fig[1,2], hm)
    fig
end