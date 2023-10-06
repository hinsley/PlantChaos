# Perform a 1D slice through parameter space, checking
# Poisson recurrence counts & max linger times.
using Pkg
Pkg.activate("./lyapunov")
using GLMakie
using StaticArrays

include("../model/Plant.jl")

start_p = [Plant.default_params...]
start_p[17] = -70.0 # Cashift
start_p[16] = -4. # xshift
end_p = [Plant.default_params...]
end_p[17] = 100.0 # Cashift
end_p[16] = 1. # xshift

resolution = 100 # How many points to sample.
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

using DynamicalSystems, OrdinaryDiffEq
using LinearAlgebra: norm
sys = CoupledODEs(Plant.melibeNew, u0, ps[1]; diffeq = (alg = RK4(), adaptive = false, dt = 1f0))
d0 = 1e-5
_d1 = [randn(), 0., randn(4)..., 0.]
_d2 = _d1/norm(_d1)*d0
psys = ParallelDynamicalSystem(sys, [u0, u0 + _d2])

function test_lyap_plot(sys, T, Ttr)
    d0 = 1e-2
    _d1 = [randn(), 0., randn(4)..., 0.]
    _d2 = _d1/norm(_d1)*d0
    psys = ParallelDynamicalSystem(sys, [u0, u0 + _d2])
    λs = Float64[]
    for i in 1:T
        d = λdist(psys)/d0
        λrescale!(psys, d)
        step!(psys)
        if i>Ttr
            push!(λs, log(d))
        end
    end
    # build average of normalized cumsums
    num_avg = 500
    max_period = 10000
    dt = 1.0
    max_i = ceil(Int, max_period/dt)
    λtot = zeros(length(λs)-max_i)
    idxs = floor.(Int, LinRange(1, max_i-1, num_avg))
    for i in idxs
        #return length(collect(1:(length(λs)-max_i)))
        λtot = λtot .+ cumsum(λs[i:end-max_i+i-1])./collect(1:(length(λs)-max_i))
    end
    λtot = λtot ./ length(idxs)
    λs = λtot


    F = Figure()
    traj = trajectory(sys, T, Δt = 1.0)
    println(length(traj[1][:,6]))
    ax = Axis(F[1,1], xlabel = "Time", ylabel = "v")
    #lines!(ax, traj[1][:,6]./maximum(traj[1][:,1]).*maximum(λs), color = :blue)
    ais2 = vcat(zeros(Ttr), λs)
    #lines!(ax, ais2, color = :red)
    lines!(ax, λs, color = :green)
    λs = cumsum(λs)./collect(1:length(λs))
    lines!(ax, λs, color = :blue)
    F
end
test_lyap_plot(sys, 500000, 10000)


systems = [deepcopy(psys) for _ in 1:Threads.nthreads() - 1]
pushfirst!(systems, psys)

lyaparray = zeros(Float32, resolution, resolution)
Threads.@threads for I in 1:resolution^2
    i = div(I - 1, resolution) + 1
    j = mod(I - 1, resolution) + 1
    system = systems[Threads.threadid()]
    set_parameter!(system, 16, x_shifts[i])
    set_parameter!(system, 17, Ca_shifts[j])
    lyaparray[i,j] = lyapunov(system, 1000000, Ttr = 100000)
end

let fig = Figure(resolution = (2000,2000))
    ax = Axis(fig[1,1], xlabel = L"\Delta Ca", ylabel = L"\Delta x")
    hm = GLMakie.heatmap!(ax, Ca_shifts, x_shifts, lyaparray'; colorrange = (.0000,.00025), 
    colormap = [:midnightblue, :lightskyblue1, :white, :yellow, :orange, :red], lowclip= :black)
    Colorbar(fig[1,2], hm)
    fig
end