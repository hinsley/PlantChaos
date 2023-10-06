# Perform a 1D slice through parameter space, checking
# Poisson recurrence counts & max linger times.
using Pkg
Pkg.activate("./lyapunov")
using GLMakie, StaticArrays, ProgressMeter, DynamicalSystems
using LinearAlgebra: norm
using OrdinaryDiffEq: RK4
include("../model/Plant.jl")
include("./my_lyapunov.jl")

# set up parameter space for scan
start_p = Float64[Plant.default_params...]
start_p[17] = -70.0 # Cashift
start_p[16] = -4. # xshift
end_p = Float64[Plant.default_params...]
end_p[17] = 100.0 # Cashift
end_p[16] = 1. # xshift

resolution = 30 # How many points to sample.
Ca_shifts = LinRange(start_p[17], end_p[17], resolution)
x_shifts = LinRange(start_p[16], end_p[16], resolution)
ps = [[Plant.default_params[1:15]; [x_shifts[i], Ca_shifts[j]]] for i in 1:resolution, j in 1:resolution]

# set up systems
u0 = @SVector Float64[
    0.2, # x
    Plant.default_state[2],
    Plant.default_state[3],
    Plant.default_state[4],
    1.0, # Ca
    Plant.default_state[6],
    #Plant.default_state[7]
]
sys = CoupledODEs(Plant.melibeNew, u0, ps[1]; diffeq = (alg = RK4(), adaptive = false, dt = 1.0))
d0 = 1e-6
_d1 = [randn(), 0., randn(4)...]
_d2 = _d1/norm(_d1)*d0
psys = ParallelDynamicalSystem(sys, [u0, u0 + _d2])
systems = [deepcopy(psys) for _ in 1:Threads.nthreads() - 1]
pushfirst!(systems, psys)

@time my_lyapunov(systems[1], 150000, 50000, 1.0, d0 = d0, num_avg = 100, max_period = 20000)
@time lyapunov(systems[1], 150000; Ttr = 50000, d0 = d0)
traj = trajectory(sys, 150000)
begin
    fig = Figure(resolution = (2000,2000))
    ax = Axis(fig[1,1], xlabel = "t", ylabel = "x")
    lines!(ax, collect(traj[2]), traj[1][:,6])
    fig
end

# perform scan
lyaparray = zeros(Float64, resolution, resolution)
begin
    p = Progress(resolution^2)
    Threads.@threads for I in 1:resolution^2
        i = div(I - 1, resolution) + 1
        j = mod(I - 1, resolution) + 1
        system = systems[Threads.threadid()]
        set_parameter!(system, 16, x_shifts[i])
        set_parameter!(system, 17, Ca_shifts[j])
        break
        lyaparray[i,j] = my_lyapunov(system, 150000, 50000, 1.0, d0 = d0, num_avg = 100, max_period = 10000)
        set_state!(system, u0)
        next!(p)
    end
    finish!(p)
end

# plot results
let fig = Figure(resolution = (2000,2000))
    ax = Axis(fig[1,1], xlabel = L"\Delta Ca", ylabel = L"\Delta x")
    hm = GLMakie.heatmap!(ax, Ca_shifts, x_shifts, lyaparray'; colorrange = (.0000,.00025), 
    colormap = [:midnightblue, :lightskyblue1, :white, :yellow, :orange, :red], lowclip= :black)
    Colorbar(fig[1,2], hm)
    fig
end

